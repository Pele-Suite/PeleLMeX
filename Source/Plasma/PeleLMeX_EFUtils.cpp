#include <PeleLMeX.H>
#include <PeleLMeX_K.H>
#include <PeleLMeX_EF_K.H>
#include <PeleLMeX_BCfill.H>
#include <AMReX_FillPatchUtil.H>

amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>
PeleLM::getNLgradPhiVVect()
{
  amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> r;
  r.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    r.push_back(GetArrOfPtrs(m_leveldatanlsolve[lev]->gPhiVOld));
  }
  return r;
}

amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>
PeleLM::getUeffVect()
{
  amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> r;
  r.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    r.push_back(GetArrOfPtrs(m_leveldatanlsolve[lev]->uEffnE));
  }
  return r;
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getNLresidVect()
{
  amrex::Vector<amrex::MultiFab*> r;
  r.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    r.push_back(&(m_leveldatanlsolve[lev]->nlResid));
  }
  return r;
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getNLstateVect()
{
  amrex::Vector<amrex::MultiFab*> r;
  r.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    r.push_back(&(m_leveldatanlsolve[lev]->nlState));
  }
  return r;
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getNLBGChargeVect()
{
  amrex::Vector<amrex::MultiFab*> r;
  r.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    r.push_back(&(m_leveldatanlsolve[lev]->backgroundCharge));
  }
  return r;
}

void
PeleLM::getNLStateScaling(amrex::Real& nEScale, amrex::Real& phiVScale)
{
  amrex::Array<amrex::Real, 2> r = {0.0, 0.0};
  for (int comp = 0; comp < 2; ++comp) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      if (lev != finest_level) {
        r[comp] = amrex::max<amrex::Real>(
          r[comp], m_leveldatanlsolve[lev]->nlState.norm0(
                     *m_coveredMask[lev], comp, 0, true));
      } else {
        r[comp] = amrex::max<amrex::Real>(
          r[comp], m_leveldatanlsolve[lev]->nlState.norm0(comp, 0, true, true));
      }
    }
    amrex::ParallelDescriptor::ReduceRealMax(r[comp]);
  }
  nEScale = r[0];
  phiVScale = r[1];
}

void
PeleLM::getNLResidScaling(amrex::Real& nEScale, amrex::Real& phiVScale)
{
  amrex::Array<amrex::Real, 2> r = {0.0, 0.0};
  for (int comp = 0; comp < 2; ++comp) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      if (lev != finest_level) {
        r[comp] = amrex::max<amrex::Real>(
          r[comp], m_leveldatanlsolve[lev]->nlResid.norm0(
                     *m_coveredMask[lev], comp, 0, true));
      } else {
        r[comp] = amrex::max<amrex::Real>(
          r[comp], m_leveldatanlsolve[lev]->nlResid.norm0(comp, 0, true));
      }
    }
    amrex::ParallelDescriptor::ReduceRealMax(r[comp]);
  }
  nEScale = r[0];
  phiVScale = r[1];
}

void
PeleLM::scaleNLState(
  const amrex::Real /*nEScale*/, const amrex::Real /*phiVScale*/)
{
  for (int lev = 0; lev <= finest_level; ++lev) {
    m_leveldatanlsolve[lev]->nlState.mult(1.0 / nE_scale, 0, 1, m_nGrowState);
    m_leveldatanlsolve[lev]->nlState.mult(1.0 / phiV_scale, 1, 1, m_nGrowState);
  }
}

void
PeleLM::scaleNLResid(
  const amrex::Vector<amrex::MultiFab*>& a_resid,
  const amrex::Real /*nEScale*/,
  const amrex::Real /*phiVScale*/)
{
  for (int lev = 0; lev <= finest_level; ++lev) {
    a_resid[lev]->mult(1.0 / FnE_scale, 0, 1, 1);
    a_resid[lev]->mult(1.0 / FphiV_scale, 1, 1, 1);
  }
}

amrex::BCRec
PeleLM::hackBCChargedParticle(
  const amrex::Real charge, const amrex::BCRec& bc_in)
{
  amrex::BCRec bc_hacked;

  const int* lo_bc = bc_in.lo();
  const int* hi_bc = bc_in.hi();

  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    int lo = lo_bc[idim];
    int hi = hi_bc[idim];
    // Spec is In/Out and it's cathode (neg electrode)
    if (
      (lo_bc[idim] == amrex::BCType::ext_dir ||
       lo_bc[idim] == amrex::BCType::foextrap) &&
      (m_phiV_bcpol.lo(idim) == 2)) {
      if (charge > 0.0) { // Outflow for cation
        lo = amrex::BCType::foextrap;
      } else { // Dirich = 0 for anion
        lo = amrex::BCType::ext_dir;
      }
    } else if (
      (lo_bc[idim] == amrex::BCType::ext_dir ||
       lo_bc[idim] == amrex::BCType::foextrap) &&
      (m_phiV_bcpol.lo(idim) == 1)) {
      if (charge > 0.0) { // Dirich = 0 for cation
        lo = amrex::BCType::ext_dir;
      } else { // Outflow for anion
        lo = amrex::BCType::foextrap;
      }
    }
    if (
      (hi_bc[idim] == amrex::BCType::ext_dir ||
       hi_bc[idim] == amrex::BCType::foextrap) &&
      (m_phiV_bcpol.hi(idim) == 2)) {
      if (charge > 0.0) { // Outflow for cation
        hi = amrex::BCType::foextrap;
      } else { // Dirich = 0 for anion
        hi = amrex::BCType::ext_dir;
      }
    } else if (
      (hi_bc[idim] == amrex::BCType::ext_dir ||
       hi_bc[idim] == amrex::BCType::foextrap) &&
      (m_phiV_bcpol.hi(idim) == 1)) {
      if (charge > 0.0) { // Dirich = 0 for cation
        hi = amrex::BCType::ext_dir;
      } else { // Outflow for anion
        hi = amrex::BCType::foextrap;
      }
    }
    bc_hacked.setLo(idim, lo);
    bc_hacked.setHi(idim, hi);
  }
  return bc_hacked;
}

void
PeleLM::addLorentzVelForces(
  const int lev,
  const amrex::Box& bx,
  const amrex::Real a_time,
  amrex::Array4<amrex::Real> const& force,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& phiV,
  amrex::Array4<const amrex::Real> const& nE)
{
  const auto dx = geom[lev].CellSizeArray();
  amrex::GpuArray<int, 3> blo = bx.loVect3d();
  amrex::GpuArray<int, 3> bhi = bx.hiVect3d();

  amrex::ParallelFor(
    bx, [blo, bhi, a_time, dx, zk = zk, rhoY, nE, phiV,
         force] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      addLorentzForce(i, j, k, blo, bhi, a_time, dx, zk, rhoY, nE, phiV, force);
    });
}

void
PeleLM::initializeElectronNeutral()
{
  // Prob/PMF data
  ProbParm const* lprobparm = prob_parm_d;

  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get level data new time pointer
    auto ldata_p = getLevelDataPtr(lev, AmrNewTime);

    auto const& state_ma = ldata_p->state.arrays();

    amrex::ParallelFor(
      ldata_p->state, [state_ma, lprobparm] AMREX_GPU_DEVICE(
                        int box_no, int i, int j, int k) noexcept {
        amrex::Array4<amrex::Real> rho(state_ma[box_no], DENSITY);
        amrex::Array4<amrex::Real> rhoY(state_ma[box_no], FIRSTSPEC);
        amrex::Array4<amrex::Real> rhoH(state_ma[box_no], RHOH);
        amrex::Array4<amrex::Real> temp(state_ma[box_no], TEMP);
        amrex::Array4<amrex::Real> nE(state_ma[box_no], NE);
        initElecNeutral(i, j, k, rho, rhoY, rhoH, temp, nE, *lprobparm);
      });
    amrex::Gpu::streamSynchronize();
    // Convert I_R(Y_nE) into I_R(nE) and set I_R(Y_nE) to zero
    auto ldataR_p = getLevelDataReactPtr(lev);
    auto const& I_R_ma = ldataR_p->I_R.arrays();
    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real invmwt[NUM_SPECIES] = {0.0};
    eos.inv_molecular_weight(invmwt);
    amrex::ParallelFor(
      ldataR_p->I_R, [I_R_ma, invmwt] AMREX_GPU_DEVICE(
                       int box_no, int i, int j, int k) noexcept {
        amrex::Array4<amrex::Real> YnEdot(I_R_ma[box_no], E_ID);
        amrex::Array4<amrex::Real> nEdot(I_R_ma[box_no], NUM_SPECIES);
        nEdot(i, j, k) = YnEdot(i, j, k) * Na * invmwt[E_ID] * 1.0e3;
        YnEdot(i, j, k) = 0.0;
      });
    amrex::Gpu::streamSynchronize();
  }
}

void
PeleLM::initializeElectronFromMassFraction()
{
}

void
PeleLM::fillPatchExtrap(
  const amrex::Real a_time,
  amrex::Vector<amrex::MultiFab*> const& a_MF,
  const int a_nGrow)
{
  AMREX_ASSERT(a_MF[0]->nComp() <= m_bcrec_force.size());
  const int nComp = a_MF[0]->nComp();

  int lev = 0;
  {
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      bndry_func(geom[lev], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    FillPatchSingleLevel(
      *a_MF[lev], amrex::IntVect(a_nGrow), a_time, {a_MF[lev]}, {a_time}, 0, 0,
      nComp, geom[lev], bndry_func, 0);
  }
  for (lev = 1; lev <= finest_level; ++lev) {
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      crse_bndry_func(
        geom[lev - 1], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      fine_bndry_func(
        geom[lev], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    auto* mapper = getInterpolator();
    FillPatchTwoLevels(
      *a_MF[lev], amrex::IntVect(a_nGrow), a_time, {a_MF[lev - 1]}, {a_time},
      {a_MF[lev]}, {a_time}, 0, 0, nComp, geom[lev - 1], geom[lev],
      crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1), mapper,
      {m_bcrec_force}, 0);
  }
}

void
PeleLM::fillPatchNLnE(
  amrex::Real a_time, amrex::Vector<amrex::MultiFab*> const& a_nE, int a_nGrow)
{
  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();

  int lev = 0;
  {
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirnE<ProblemSpecificFunctions>>>
      bndry_func(
        geom[lev], fetchBCRecArray(NE, 1),
        PeleLMCCFillExtDirnE<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux});
    FillPatchSingleLevel(
      *a_nE[lev], amrex::IntVect(a_nGrow), a_time, {a_nE[lev]}, {a_time}, 0, 0,
      1, geom[lev], bndry_func, 0);
  }
  for (lev = 1; lev <= finest_level; ++lev) {
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirnE<ProblemSpecificFunctions>>>
      crse_bndry_func(
        geom[lev - 1], fetchBCRecArray(NE, 1),
        PeleLMCCFillExtDirnE<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux});
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirnE<ProblemSpecificFunctions>>>
      fine_bndry_func(
        geom[lev], fetchBCRecArray(NE, 1),
        PeleLMCCFillExtDirnE<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux});

    auto* mapper = getInterpolator();
    FillPatchTwoLevels(
      *a_nE[lev], amrex::IntVect(a_nGrow), a_time, {a_nE[lev - 1]}, {a_time},
      {a_nE[lev]}, {a_time}, 0, 0, 1, geom[lev - 1], geom[lev], crse_bndry_func,
      0, fine_bndry_func, 0, refRatio(lev - 1), mapper, fetchBCRecArray(NE, 1),
      0);
  }
}

void
PeleLM::fillPatchNLphiV(
  amrex::Real a_time,
  amrex::Vector<amrex::MultiFab*> const& a_phiV,
  int a_nGrow)
{
  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();

  int lev = 0;
  {
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirPhiV<ProblemSpecificFunctions>>>
      bndry_func(
        geom[lev], fetchBCRecArray(PHIV, 1),
        PeleLMCCFillExtDirPhiV<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux});
    FillPatchSingleLevel(
      *a_phiV[lev], amrex::IntVect(a_nGrow), a_time, {a_phiV[lev]}, {a_time}, 0,
      0, 1, geom[lev], bndry_func, 0);
  }
  for (lev = 1; lev <= finest_level; ++lev) {
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirPhiV<ProblemSpecificFunctions>>>
      crse_bndry_func(
        geom[lev - 1], fetchBCRecArray(PHIV, 1),
        PeleLMCCFillExtDirPhiV<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux});
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirPhiV<ProblemSpecificFunctions>>>
      fine_bndry_func(
        geom[lev], fetchBCRecArray(PHIV, 1),
        PeleLMCCFillExtDirPhiV<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux});

    auto* mapper = getInterpolator();
    FillPatchTwoLevels(
      *a_phiV[lev], amrex::IntVect(a_nGrow), a_time, {a_phiV[lev - 1]},
      {a_time}, {a_phiV[lev]}, {a_time}, 0, 0, 1, geom[lev - 1], geom[lev],
      crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1), mapper,
      fetchBCRecArray(PHIV, 1), 0);
  }
}

void
PeleLM::ionsBalance()
{
  // Compute the sum of ions on the domain boundaries
  amrex::Array<amrex::Real, 2 * AMREX_SPACEDIM> ionsCurrent{0.0};
  for (int n = NUM_SPECIES - NUM_IONS; n < NUM_SPECIES; ++n) {
    for (int i = 0; i < 2 * AMREX_SPACEDIM; ++i) {
      ionsCurrent[i] += m_domainRhoYFlux[2 * n * AMREX_SPACEDIM + i] * zk[n];
    }
  }

  tmpIonsFile << m_nstep << "," << m_cur_time; // Time info
  for (int i = 0; i < 2 * AMREX_SPACEDIM; ++i) {
    tmpIonsFile << "," << ionsCurrent[i]; // ions current as xlo, xhi, ylo, ...
  }
  tmpIonsFile << "\n";
  tmpIonsFile.flush();
}
