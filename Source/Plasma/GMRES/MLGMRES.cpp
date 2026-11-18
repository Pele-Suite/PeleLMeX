#include <MLGMRES.H>
#include <AMReX_ParmParse.H>

MLGMRESSolver::MLGMRESSolver()
{
  m_verbose = 0;
  m_restart = 1;
}

MLGMRESSolver::~MLGMRESSolver() {}

void
MLGMRESSolver::define(PeleLM* a_pelelm, const int a_nComp, const int a_nGrow)
{
  BL_PROFILE("MLGMRESSolver::define()");

  readParameters();

  m_pelelm = a_pelelm;

  const int finest_level = m_pelelm->finestLevel();

  // Resize level vector
  m_grids.resize(finest_level + 1);
  m_dmap.resize(finest_level + 1);
  Ax.reserve(finest_level + 1);
  res.reserve(finest_level + 1);

  // Grab stuff from AmrCore
  m_geom = m_pelelm->Geom(0, finest_level);
  m_grids = m_pelelm->boxArray(0, finest_level);
  m_dmap = m_pelelm->DistributionMap(0, finest_level);

  // Internal GMRES size
  m_nComp = a_nComp;
  m_nGrow = a_nGrow;

  // Build krylov base memory and work MFs
  KspBase.resize(m_krylovSize + 1);
  for (int n = 0; n <= m_krylovSize; ++n) {
    KspBase[n].reserve(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      KspBase[n].emplace_back(m_grids[lev], m_dmap[lev], m_nComp, m_nGrow);
    }
  }
  for (int lev = 0; lev <= finest_level; ++lev) {
    Ax.emplace_back(m_grids[lev], m_dmap[lev], m_nComp, m_nGrow);
    res.emplace_back(m_grids[lev], m_dmap[lev], m_nComp, m_nGrow);
  }

  // Work Reals
  H.resize(m_krylovSize + 1);
  givens.resize(m_krylovSize + 1);
  for (int n = 0; n <= m_krylovSize; ++n) {
    H[n].resize(m_krylovSize, 0.0);
    givens[n].resize(2, 0.0);
  }
  y.resize(m_krylovSize + 1, 0.0);
  g.resize(m_krylovSize + 1, 0.0);
}

MLJtimesVFunc
MLGMRESSolver::jtimesv() const noexcept
{
  return m_jtv;
}

MLPrecondFunc
MLGMRESSolver::precond() const noexcept
{
  return m_prec;
}

MLNormFunc
MLGMRESSolver::norm() const noexcept
{
  return m_norm;
}

void
MLGMRESSolver::setJtimesV(MLJtimesVFunc a_jtv)
{
  m_jtv = a_jtv;
}

void
MLGMRESSolver::setPrecond(MLPrecondFunc a_prec)
{
  m_prec = a_prec;
}

void
MLGMRESSolver::setNorm(MLNormFunc a_norm)
{
  m_norm = a_norm;
}

int
MLGMRESSolver::solve(
  const amrex::Vector<amrex::MultiFab*>& a_sol,
  const amrex::Vector<amrex::MultiFab*>& a_rhs,
  amrex::Real a_abs_tol,
  amrex::Real a_rel_tol)
{
  BL_PROFILE("MLGMRESSolver::solve()");

  // Checks
  AMREX_ALWAYS_ASSERT(m_jtv != nullptr);
  AMREX_ALWAYS_ASSERT(
    a_sol[0]->nComp() == m_nComp && a_rhs[0]->nComp() == m_nComp);
  if (m_prec == nullptr)
    amrex::Print() << "  Using unpreconditioned GMRES ! Might take a while to "
                      "converge ...\n";

  amrex::Real rhsNorm = computeMLNorm(a_rhs);        // RHS norm
  initResNorm = computeMLResidualNorm(a_sol, a_rhs); // Initial resisual norm
  target_relResNorm = initResNorm * a_rel_tol; // Target relative tolerance
  target_absResNorm = a_abs_tol;               // Target absolute tolerance

  if (m_verbose > 0) {
    amrex::Print() << "  GMRES: Initial rhs = " << rhsNorm << "\n";
    amrex::Print() << "  GMRES: Initial residual = " << initResNorm << "\n";
  }
  if (initResNorm < a_abs_tol) {
    amrex::Print() << "  GMRES: no need for iterations \n";
    return 0;
  }

  iter_count = 0;
  restart_count = 0;
  m_converged = false;
  do {
    // Prepare for solve
    prepareForSolve();
    one_restart(a_sol, a_rhs);
    restart_count++;
  } while (!m_converged && restart_count < m_restart);

  amrex::Real finalResNorm =
    computeMLResidualNorm(a_sol, a_rhs); // Final resisual norm
  if (m_verbose > 0)
    amrex::Print() << "  GMRES: [" << iter_count
                   << "] Final residual, resid/resid0 = " << finalResNorm
                   << ", " << finalResNorm / initResNorm << "\n";
  return iter_count;
}

void
MLGMRESSolver::prepareForSolve()
{
  // Initialize or cleanup GMRES data
  for (int k = 0; k <= m_krylovSize; ++k) {
    g[k] = 0.0;
    y[k] = 0.0;
    for (int n = 0; n < m_krylovSize; ++n) {
      H[k][n] = 0.0;
    }
    givens[k][0] = 0.0;
    givens[k][1] = 0.0;
  }
}

void
MLGMRESSolver::one_restart(
  const amrex::Vector<amrex::MultiFab*>& a_x,
  const amrex::Vector<amrex::MultiFab*>& a_rhs)
{
  int finest_level = m_pelelm->finestLevel();

  computeMLResidual(a_x, a_rhs, GetVecOfPtrs(res));
  amrex::Real resNorm_0 = computeMLNorm(GetVecOfPtrs(res));
  if (m_verbose > 1)
    amrex::Print() << "     [Restart:" << restart_count
                   << "] initial relative res: " << resNorm_0 / initResNorm
                   << "\n";

  // Initialize KspBase with normalized residual
  g[0] = resNorm_0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    res[lev].mult(1.0 / resNorm_0);
    amrex::MultiFab::Copy(KspBase[0][lev], res[lev], 0, 0, m_nComp, 0);
  }

  int k_end = m_krylovSize - 1;
  amrex::Real resNorm = resNorm_0;

  for (int k = 0; k < m_krylovSize; ++k) {
    // Do one GMRES iteration, update the residual norm
    one_iter(k, resNorm);
    iter_count++;

    // Test exit condition
    if (
      (resNorm / initResNorm) < target_relResNorm ||
      resNorm < target_absResNorm) {
      m_converged = true;
      k_end = k;
      break;
    }
  }

  // Solve the minimization problem H.y = g
  y[k_end] = g[k_end] / H[k_end][k_end];
  for (int k = k_end - 1; k >= 0; --k) {
    amrex::Real sum_tmp = 0.0;
    for (int j = k + 1; j <= k_end; ++j) {
      sum_tmp += H[k][j] * y[j];
    }
    y[k] = (g[k] - sum_tmp) / H[k][k];
  }

  // Compute solution update
  updateSolution(k_end, a_x);
}

void
MLGMRESSolver::one_iter(const int iter, amrex::Real& resNorm)
{
  BL_PROFILE("MLGMRESSolver::one_iter()");
  if (m_verbose > 1)
    amrex::Print() << "     [Iter:" << iter_count
                   << "] residual norm: " << resNorm / initResNorm << "\n";
  appendBasisVector(iter, KspBase);
  gramSchmidtOrtho(iter, KspBase);
  resNorm = givensRotation(iter);
}

void
MLGMRESSolver::updateSolution(
  int k_end, const amrex::Vector<amrex::MultiFab*>& a_x)
{
  int finest_level = m_pelelm->finestLevel();
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab update(m_grids[lev], m_dmap[lev], m_nComp, 0);
    update.setVal(0.0);
    for (int i = k_end; i >= 0; --i) {
      amrex::MultiFab::Saxpy(update, y[i], KspBase[i][lev], 0, 0, m_nComp, 0);
    }
    amrex::MultiFab::Add(*a_x[lev], update, 0, 0, m_nComp, 0);
  }

  // TODO Do an average down ?
}

void
MLGMRESSolver::appendBasisVector(
  const int iter, amrex::Vector<amrex::Vector<amrex::MultiFab>>& Base)
{
  if (m_prec == nullptr) {
    MEMBER_FUNC_PTR(*m_pelelm, m_jtv)
    (GetVecOfPtrs(Base[iter]), GetVecOfPtrs(Base[iter + 1]));
  } else {
    MEMBER_FUNC_PTR(*m_pelelm, m_jtv)
    (GetVecOfPtrs(Base[iter]), GetVecOfPtrs(Ax));
    MEMBER_FUNC_PTR(*m_pelelm, m_prec)
    (GetVecOfPtrs(Ax), GetVecOfPtrs(Base[iter + 1]));
  }
}

void
MLGMRESSolver::gramSchmidtOrtho(
  const int iter, amrex::Vector<amrex::Vector<amrex::MultiFab>>& Base)
{
  int finest_level = m_pelelm->finestLevel();
  for (int row = 0; row <= iter; ++row) {
    H[row][iter] = MFVecDot(
      GetVecOfConstPtrs(Base[iter + 1]), 0, GetVecOfConstPtrs(Base[row]), 0,
      m_nComp, 0);
    amrex::Real GS_corr = -H[row][iter];
    MFVecSaxpy(
      GetVecOfPtrs(Base[iter + 1]), GS_corr, GetVecOfConstPtrs(Base[row]), 0, 0,
      m_nComp, 0);
    if (check_GramSchmidtOrtho) {
      amrex::Real Hcorr = MFVecDot(
        GetVecOfConstPtrs(Base[iter + 1]), 0, GetVecOfConstPtrs(Base[row]), 0,
        m_nComp, 0);
      if (std::fabs(Hcorr) > 1.0e-14) {
        H[row][iter] += Hcorr;
        GS_corr = -Hcorr;
        MFVecSaxpy(
          GetVecOfPtrs(Base[iter + 1]), GS_corr, GetVecOfConstPtrs(Base[row]),
          0, 0, m_nComp, 0);
      }
    }
  }
  amrex::Real normNewVec = computeMLNorm(GetVecOfPtrs(Base[iter + 1]));
  H[iter + 1][iter] = normNewVec;
  if (normNewVec > 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      Base[iter + 1][lev].mult(1.0 / normNewVec);
    }
  }
  // m_pelelm->WriteDebugPlotFile(GetVecOfConstPtrs(Base[iter+1]),"KspVec_"+std::to_string(iter_count));
}

amrex::Real
MLGMRESSolver::givensRotation(const int iter)
{
  for (int row = 0; row < iter; ++row) {
    amrex::Real v1 =
      givens[row][0] * H[row][iter] - givens[row][1] * H[row + 1][iter];
    amrex::Real v2 =
      givens[row][1] * H[row][iter] + givens[row][0] * H[row + 1][iter];
    H[row][iter] = v1;
    H[row + 1][iter] = v2;
  }
  amrex::Real norm_Hlast = std::sqrt(
    H[iter][iter] * H[iter][iter] + H[iter + 1][iter] * H[iter + 1][iter]);
  if (norm_Hlast > 0.0) {
    givens[iter][0] = H[iter][iter] / norm_Hlast;
    givens[iter][1] = -H[iter + 1][iter] / norm_Hlast;
    H[iter][iter] =
      givens[iter][0] * H[iter][iter] - givens[iter][1] * H[iter + 1][iter];
    H[iter + 1][iter] = 0.0;
    amrex::Real v1 = givens[iter][0] * g[iter] - givens[iter][1] * g[iter + 1];
    amrex::Real v2 = givens[iter][1] * g[iter] + givens[iter][0] * g[iter + 1];
    g[iter] = v1;
    g[iter + 1] = v2;
  }
  return std::fabs(g[iter + 1]);
}

amrex::Real
MLGMRESSolver::computeMLResidualNorm(
  const amrex::Vector<amrex::MultiFab*>& a_x,
  const amrex::Vector<amrex::MultiFab*>& a_rhs)
{
  computeMLResidual(a_x, a_rhs, GetVecOfPtrs(res));
  return computeMLNorm(GetVecOfPtrs(res));
}

amrex::Real
MLGMRESSolver::computeMLNorm(const amrex::Vector<amrex::MultiFab*>& a_vec)
{
  amrex::Real r = 0.0;
  if (m_norm != nullptr) {
    MEMBER_FUNC_PTR(*m_pelelm, m_norm)(a_vec, r);
  } else {
    for (int comp = 0; comp < m_nComp; ++comp) {
      amrex::Real norm = 0.0;
      for (int lev = 0; lev <= a_vec.size(); ++lev) {
        norm +=
          amrex::MultiFab::Dot(*a_vec[lev], comp, *a_vec[lev], comp, 1, 0);
      }
      r += norm;
    }
    r = std::sqrt(r);
  }
  return r;
}

void
MLGMRESSolver::computeMLResidual(
  const amrex::Vector<amrex::MultiFab*>& a_x,
  const amrex::Vector<amrex::MultiFab*>& a_rhs,
  const amrex::Vector<amrex::MultiFab*>& a_res)
{
  BL_PROFILE("MLGMRESSolver::computeMLResidual()");
  int finest_level = m_pelelm->finestLevel();
  MEMBER_FUNC_PTR(*m_pelelm, m_jtv)(a_x, GetVecOfPtrs(Ax));
  if (m_prec != nullptr) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab::Xpay(Ax[lev], -1.0, *a_rhs[lev], 0, 0, m_nComp, 0);
    }
    MEMBER_FUNC_PTR(*m_pelelm, m_prec)(GetVecOfPtrs(Ax), a_res);
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab::LinComb(
        *a_res[lev], 1.0, Ax[lev], 0, -1.0, *a_rhs[lev], 0, 0, m_nComp, 0);
    }
  }
}

amrex::Real
MLGMRESSolver::MFVecDot(
  const amrex::Vector<const amrex::MultiFab*>& a_mf1,
  int mf1comp,
  const amrex::Vector<const amrex::MultiFab*>& a_mf2,
  int mf2comp,
  int nComp,
  int nGrow)
{
  int finest_level = m_pelelm->finestLevel();
  amrex::Real r = 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    if (lev != finest_level) {
      r += amrex::MultiFab::Dot(
        *(m_pelelm->m_coveredMask[lev]), *a_mf1[lev], mf1comp, *a_mf2[lev],
        mf2comp, nComp, nGrow);
    } else {
      r += amrex::MultiFab::Dot(
        *a_mf1[lev], mf1comp, *a_mf2[lev], mf2comp, nComp, nGrow);
    }
  }
  return r;
}

void
MLGMRESSolver::MFVecSaxpy(
  const amrex::Vector<amrex::MultiFab*>& a_mfdest,
  amrex::Real a_a,
  const amrex::Vector<const amrex::MultiFab*>& a_mfsrc,
  int destComp,
  int srcComp,
  int nComp,
  int nGrow)
{
  int finest_level = m_pelelm->finestLevel();
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Saxpy(
      *a_mfdest[lev], a_a, *a_mfsrc[lev], destComp, srcComp, nComp, nGrow);
  }
}

void
MLGMRESSolver::readParameters()
{
  amrex::ParmParse pp("gmres");

  pp.query("krylovBasis_size", m_krylovSize);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    m_krylovSize > 0, "gmres.krylovBasis_size must be positive");
  pp.query("max_restart", m_restart);
  pp.query("verbose", m_verbose);
  pp.query("checkGSortho", check_GramSchmidtOrtho);
}
