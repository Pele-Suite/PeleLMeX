#include <PeleLMeX.H>
#include <PrecondOp.H>
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>

#ifdef AMREX_USE_EB
#include <AMReX_EB_utils.H>
#endif

//---------------------------------------------------------------------------------------
// Preconditioner Operator
PrecondOp::PrecondOp(PeleLM* a_pelelm) : m_pelelm(a_pelelm)
{
  readParameters();

  // Solve LPInfo
  amrex::LPInfo info_diff;
  info_diff.setMaxCoarseningLevel(m_mg_max_coarsening_level_diff);
  amrex::LPInfo info_Stilda;
  info_Stilda.setMaxCoarseningLevel(m_mg_max_coarsening_level_Stilda);

  // Apply LPInfo (no coarsening)
  amrex::LPInfo info_apply;
  info_apply.setMaxCoarseningLevel(0);

  // Diff/Drift Op
  m_diff.reset(new amrex::MLABecCecLaplacian(
    m_pelelm->Geom(0, m_pelelm->finestLevel()),
    m_pelelm->boxArray(0, m_pelelm->finestLevel()),
    m_pelelm->DistributionMap(0, m_pelelm->finestLevel()), info_diff));
  m_diff->setMaxOrder(m_mg_maxorder);

  // Drift Op
  m_drift.reset(new amrex::MLABecLaplacian(
    m_pelelm->Geom(0, m_pelelm->finestLevel()),
    m_pelelm->boxArray(0, m_pelelm->finestLevel()),
    m_pelelm->DistributionMap(0, m_pelelm->finestLevel()), info_apply));
  m_drift->setMaxOrder(m_mg_maxorder);

  // Stilda Op
  m_Stilda.reset(new amrex::MLABecLaplacian(
    m_pelelm->Geom(0, m_pelelm->finestLevel()),
    m_pelelm->boxArray(0, m_pelelm->finestLevel()),
    m_pelelm->DistributionMap(0, m_pelelm->finestLevel()), info_Stilda));
  m_Stilda->setMaxOrder(m_mg_maxorder);

  // MLMGs
  m_mlmg_diff = std::make_unique<amrex::MLMG>(*m_diff);
  m_mlmg_drift = std::make_unique<amrex::MLMG>(*m_drift);
  m_mlmg_Stilda = std::make_unique<amrex::MLMG>(*m_Stilda);
}

void
PrecondOp::setDiffOpScalars(
  const amrex::Real& aScal, const amrex::Real& bScal, const amrex::Real& cScal)
{
  m_diff->setScalars(aScal, bScal, cScal);
}

void
PrecondOp::setDiffOpRelaxation(const amrex::Real& a_omega)
{
  m_diff->setRelaxation(a_omega);
}

void
PrecondOp::setDriftOpScalars(const amrex::Real& aScal, const amrex::Real& bScal)
{
  m_drift->setScalars(aScal, bScal);
}

void
PrecondOp::setStildaOpScalars(
  const amrex::Real& aScal, const amrex::Real& bScal)
{
  m_Stilda->setScalars(aScal, bScal);
}

void
PrecondOp::setDiffOpBCs(const amrex::BCRec& a_bcrec)
{
  m_diff->setDomainBC(
    getOpBC(amrex::Orientation::low, a_bcrec),
    getOpBC(amrex::Orientation::high, a_bcrec));
}

void
PrecondOp::setDriftOpBCs(const amrex::BCRec& a_bcrec)
{
  m_drift->setDomainBC(
    getOpBC(amrex::Orientation::low, a_bcrec),
    getOpBC(amrex::Orientation::high, a_bcrec));
}

void
PrecondOp::setStildaOpBCs(const amrex::BCRec& a_bcrec)
{
  m_Stilda->setDomainBC(
    getOpBC(amrex::Orientation::low, a_bcrec),
    getOpBC(amrex::Orientation::high, a_bcrec));
}

void
PrecondOp::setDiffOpLevelBC(int lev, amrex::MultiFab* a_MF)
{
  m_diff->setLevelBC(lev, a_MF);
}

void
PrecondOp::setDriftOpLevelBC(int lev, amrex::MultiFab* a_MF)
{
  m_drift->setLevelBC(lev, a_MF);
}

void
PrecondOp::setStildaOpLevelBC(int lev, amrex::MultiFab* a_MF)
{
  m_Stilda->setLevelBC(lev, a_MF);
}

void
PrecondOp::setDiffOpACoeff(int lev, const amrex::Real& acoeff)
{
  m_diff->setACoeffs(lev, acoeff);
}

void
PrecondOp::setDiffOpBCoeff(
  int lev, const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bCoeff)
{
  m_diff->setBCoeffs(lev, a_bCoeff);
}

void
PrecondOp::setDiffOpCCoeff(
  int lev, const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_cCoeff)
{
  m_diff->setCCoeffs(lev, a_cCoeff);
}

void
PrecondOp::getDiffOpDiagonal(int lev, amrex::MultiFab& a_diffOpDiag)
{
  m_diff->getDiagonal(lev, a_diffOpDiag);
}

void
PrecondOp::setDriftOpBCoeff(
  int lev, const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bCoeff)
{
  m_drift->setBCoeffs(lev, a_bCoeff);
}

void
PrecondOp::setStildaOpBCoeff(
  int lev, const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bCoeff)
{
  m_Stilda->setBCoeffs(lev, a_bCoeff);
}

void
PrecondOp::diffOpSolve(
  const amrex::Vector<amrex::MultiFab*>& a_sol,
  const amrex::Vector<const amrex::MultiFab*>& a_rhs,
  const amrex::Real& /*rtol*/,
  const amrex::Real& /*atol*/)
{
  // TODO set all the MLMG options
  m_mlmg_diff->setVerbose(m_diff_verbose);
  m_mlmg_diff->setPreSmooth(m_num_pre_smooth);
  m_mlmg_diff->setPostSmooth(m_num_post_smooth);
  if (m_fixed_mg_it > 0)
    m_mlmg_diff->setFixedIter(m_fixed_mg_it);

  m_mlmg_diff->solve(a_sol, a_rhs, m_mg_rtol, m_mg_atol);
}

void
PrecondOp::StildaOpSolve(
  const amrex::Vector<amrex::MultiFab*>& a_sol,
  const amrex::Vector<const amrex::MultiFab*>& a_rhs,
  const amrex::Real& /*rtol*/,
  const amrex::Real& /*atol*/)
{
  // TODO set all the MLMG options
  m_mlmg_Stilda->setVerbose(m_Stilda_verbose);
  if (m_fixed_mg_it > 0)
    m_mlmg_Stilda->setFixedIter(m_fixed_mg_it);

  m_mlmg_Stilda->solve(a_sol, a_rhs, m_mg_rtol, m_mg_atol);
}

void
PrecondOp::driftOpApply(
  const amrex::Vector<amrex::MultiFab*>& a_Ax,
  const amrex::Vector<amrex::MultiFab*>& a_x)
{
  // TODO set all the MLMG options
  m_mlmg_drift->setVerbose(m_drift_verbose);
  m_mlmg_drift->apply(a_Ax, a_x);
}

amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>
PrecondOp::getOpBC(amrex::Orientation::Side a_side, const amrex::BCRec& a_bc)
{
  amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> r;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (m_pelelm->Geom(0).isPeriodic(idim)) {
      r[idim] = amrex::LinOpBCType::Periodic;
    } else {
      auto amrexbc =
        (a_side == amrex::Orientation::low) ? a_bc.lo(idim) : a_bc.hi(idim);
      if (amrexbc == amrex::BCType::ext_dir) {
        r[idim] = amrex::LinOpBCType::Dirichlet;
      } else if (
        amrexbc == amrex::BCType::foextrap ||
        amrexbc == amrex::BCType::hoextrap ||
        amrexbc == amrex::BCType::reflect_even) {
        r[idim] = amrex::LinOpBCType::Neumann;
      } else if (amrexbc == amrex::BCType::reflect_odd) {
        r[idim] = amrex::LinOpBCType::reflect_odd;
      } else {
        r[idim] = amrex::LinOpBCType::bogus;
      }
    }
  }
  return r;
}

void
PrecondOp::readParameters()
{
  amrex::ParmParse pp("ef.precond");

  pp.query("diff_verbose", m_diff_verbose);
  pp.query("Stilda_verbose", m_Stilda_verbose);
  pp.query("fixedIter", m_fixed_mg_it);
  pp.query("max_coarsening_level_diff", m_mg_max_coarsening_level_diff);
  pp.query("max_coarsening_level_Stilda", m_mg_max_coarsening_level_Stilda);
  pp.query("num_pre_smooth", m_num_pre_smooth);
  pp.query("num_post_smooth", m_num_post_smooth);
  // TODO: add all the user-defined options
}
