#include <PeleLMeX.H>
#include <PeleLMeX_BCfill.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <memory>
#include <queue>
#ifdef AMREX_USE_EB
#include <AMReX_EBInterpolater.H>
#include <AMReX_EB2.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBFabFactory.H>
#endif

// Conversion from physBC to fieldBC maps
// Components are  Interior, Inflow, Outflow, Symmetry, &
// SlipWallAdiab, NoSlipWallAdiab, SlipWallIsoTherm, NoSlipWallIsoTherm.

constexpr int norm_vel_bc[] = {
  amrex::BCType::int_dir,     amrex::BCType::ext_dir, amrex::BCType::foextrap,
  amrex::BCType::reflect_odd, amrex::BCType::ext_dir, amrex::BCType::ext_dir,
  amrex::BCType::ext_dir,     amrex::BCType::ext_dir};

constexpr int tang_vel_bc[] = {
  amrex::BCType::int_dir,      amrex::BCType::ext_dir,  amrex::BCType::foextrap,
  amrex::BCType::reflect_even, amrex::BCType::hoextrap, amrex::BCType::ext_dir,
  amrex::BCType::hoextrap,     amrex::BCType::ext_dir};

constexpr int density_bc[] = {
  amrex::BCType::int_dir,      amrex::BCType::ext_dir,  amrex::BCType::foextrap,
  amrex::BCType::reflect_even, amrex::BCType::foextrap, amrex::BCType::foextrap,
  amrex::BCType::foextrap,     amrex::BCType::foextrap};

constexpr int species_bc[] = {
  amrex::BCType::int_dir,      amrex::BCType::ext_dir,  amrex::BCType::foextrap,
  amrex::BCType::reflect_even, amrex::BCType::foextrap, amrex::BCType::foextrap,
  amrex::BCType::foextrap,     amrex::BCType::foextrap};

constexpr int ode_bc[] = {amrex::BCType::int_dir,  amrex::BCType::ext_dir,
                          amrex::BCType::foextrap, amrex::BCType::reflect_even,
                          amrex::BCType::foextrap, amrex::BCType::foextrap,
                          amrex::BCType::foextrap, amrex::BCType::foextrap};

constexpr int rhoh_bc[] = {amrex::BCType::int_dir,  amrex::BCType::ext_dir,
                           amrex::BCType::foextrap, amrex::BCType::reflect_even,
                           amrex::BCType::foextrap, amrex::BCType::foextrap,
                           amrex::BCType::ext_dir,  amrex::BCType::ext_dir};

constexpr int temp_bc[] = {amrex::BCType::int_dir,  amrex::BCType::ext_dir,
                           amrex::BCType::foextrap, amrex::BCType::reflect_even,
                           amrex::BCType::foextrap, amrex::BCType::foextrap,
                           amrex::BCType::ext_dir,  amrex::BCType::ext_dir};

constexpr int aux_bc[] = {amrex::BCType::int_dir,  amrex::BCType::ext_dir,
                          amrex::BCType::foextrap, amrex::BCType::reflect_even,
                          amrex::BCType::foextrap, amrex::BCType::foextrap,
                          amrex::BCType::ext_dir,  amrex::BCType::ext_dir};

constexpr int divu_bc[] = {
  amrex::BCType::int_dir,      amrex::BCType::reflect_even,
  amrex::BCType::reflect_even, amrex::BCType::reflect_even,
  amrex::BCType::reflect_even, amrex::BCType::reflect_even,
  amrex::BCType::reflect_even, amrex::BCType::reflect_even};

// Following incflo rather than IAMR here
constexpr int force_bc[] = {amrex::BCType::int_dir,  amrex::BCType::foextrap,
                            amrex::BCType::foextrap, amrex::BCType::foextrap,
                            amrex::BCType::foextrap, amrex::BCType::foextrap,
                            amrex::BCType::foextrap, amrex::BCType::foextrap};

#ifdef PELE_USE_PLASMA
constexpr int nE_bc[] = {
  amrex::BCType::int_dir,      amrex::BCType::ext_dir,
  amrex::BCType::foextrap,     amrex::BCType::reflect_even,
  amrex::BCType::reflect_even, amrex::BCType::reflect_even,
  amrex::BCType::ext_dir,      amrex::BCType::ext_dir};

constexpr int phiV_bc[] = {
  amrex::BCType::int_dir, amrex::BCType::ext_dir, amrex::BCType::reflect_even};
#endif

#ifdef PELE_USE_SOOT
constexpr int soot_bc[] = {
  amrex::BCType::int_dir,      amrex::BCType::ext_dir,
  amrex::BCType::foextrap,     amrex::BCType::reflect_even,
  amrex::BCType::reflect_even, amrex::BCType::reflect_even,
  amrex::BCType::ext_dir,      amrex::BCType::ext_dir};
#endif

amrex::InterpBase*
PeleLM::
  getInterpolator( // NOLINT(readability-convert-member-functions-to-static)
    const int a_method,
    const int a_crse_level) const
{
  // Mesh-mapping-aware path: whole-state fillpatch helpers pass the
  // coarse level of the pair (a_crse_level) to opt in; other callers
  // leave it at the default -1 and get the legacy interpolator.
  if (
    a_method == 1 && m_mesh_mapping && a_crse_level >= 0 &&
    a_crse_level < static_cast<int>(m_mapped_interps.size()) &&
    m_mapped_interps[a_crse_level]) {
    return m_mapped_interps[a_crse_level].get();
  }

  amrex::InterpBase* mapper = nullptr;

  switch (a_method) {
  case 0:
    mapper = &amrex::mf_pc_interp;
    break;

  case 1:
#ifdef AMREX_USE_EB
    // Get EB-aware interpolator when needed
    mapper = (EBFactory(0).isAllRegular()) ? &amrex::mf_cell_cons_interp
                                           : &amrex::eb_mf_cell_cons_interp;
#else
    mapper = &amrex::mf_cell_cons_interp;
#endif
    break;

  case 2:
#ifdef AMREX_USE_EB
    amrex::Abort("Regrid interpolation method = 2 not available with EB !");
#else
    mapper = &amrex::mf_linear_slope_minmax_interp;
#endif
    break;

  default:
    amrex::Abort("Unknown interpolation method");
  }

  return mapper;
}

void
PeleLM::setBoundaryConditions()
{

  // Initialize the BCRecs
  m_bcrec_state.resize(NVAR);
  constexpr int sizeForceBC = amrex::max<int>(AMREX_SPACEDIM, NUM_SPECIES + 2);
  m_bcrec_force.resize(sizeForceBC);
  m_bcrec_aux.resize(m_nAux);

  // Convert m_phys_bc into field BCs
  // Get m_phys_bc
  const int* lo_bc = m_phys_bc.lo();
  const int* hi_bc = m_phys_bc.hi();

  // Velocity
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    for (int idim2 = 0; idim2 < AMREX_SPACEDIM; ++idim2) {
      if (idim == idim2) {
        m_bcrec_state[VELX + idim].setLo(idim2, norm_vel_bc[lo_bc[idim2]]);
        m_bcrec_state[VELX + idim].setHi(idim2, norm_vel_bc[hi_bc[idim2]]);
      } else {
        m_bcrec_state[VELX + idim].setLo(idim2, tang_vel_bc[lo_bc[idim2]]);
        m_bcrec_state[VELX + idim].setHi(idim2, tang_vel_bc[hi_bc[idim2]]);
      }
    }
  }

  // General forces: use int_dir in interior and foextrap otherwise
  for (int i = 0; i < sizeForceBC; ++i) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      m_bcrec_force[i].setLo(idim, force_bc[lo_bc[idim]]);
      m_bcrec_force[i].setHi(idim, force_bc[hi_bc[idim]]);
    }
  }

  if (m_incompressible == 0) {
    // Density
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      m_bcrec_state[DENSITY].setLo(idim, density_bc[lo_bc[idim]]);
      m_bcrec_state[DENSITY].setHi(idim, density_bc[hi_bc[idim]]);
    }

    // Species
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      for (int n = 0; n < NUM_SPECIES; ++n) {
        m_bcrec_state[FIRSTSPEC + n].setLo(idim, species_bc[lo_bc[idim]]);
        m_bcrec_state[FIRSTSPEC + n].setHi(idim, species_bc[hi_bc[idim]]);
      }
    }

    // Enthalpy
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      m_bcrec_state[RHOH].setLo(idim, rhoh_bc[lo_bc[idim]]);
      m_bcrec_state[RHOH].setHi(idim, rhoh_bc[hi_bc[idim]]);
    }

    // Temperature
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      m_bcrec_state[TEMP].setLo(idim, temp_bc[lo_bc[idim]]);
      m_bcrec_state[TEMP].setHi(idim, temp_bc[hi_bc[idim]]);
    }

    // rhoRT: reflect even on all but interior bndy
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      m_bcrec_state[RHORT].setLo(idim, divu_bc[lo_bc[idim]]);
      m_bcrec_state[RHORT].setHi(idim, divu_bc[hi_bc[idim]]);
    }

    // ODEs
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      for (int n = 0; n < NUM_ODE; ++n) {
        m_bcrec_state[FIRSTODE + n].setLo(idim, ode_bc[lo_bc[idim]]);
        m_bcrec_state[FIRSTODE + n].setHi(idim, ode_bc[hi_bc[idim]]);
      }
    }

    // divU
    if (m_has_divu != 0) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        m_bcrec_divu.setLo(idim, divu_bc[lo_bc[idim]]);
        m_bcrec_divu.setHi(idim, divu_bc[hi_bc[idim]]);
      }
    }
    // auxiliaries - assumed to be the same as species
    for (int n = 0; n < m_nAux; ++n) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        m_bcrec_aux[n].setLo(idim, aux_bc[lo_bc[idim]]);
        m_bcrec_aux[n].setHi(idim, aux_bc[hi_bc[idim]]);
      }
    }

#ifdef PELE_USE_PLASMA
    // nE
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      m_bcrec_state[NE].setLo(idim, nE_bc[lo_bc[idim]]);
      m_bcrec_state[NE].setHi(idim, nE_bc[hi_bc[idim]]);
    }

    // Get m_phiV_bc
    const int* lo_phibc = m_phiV_bc.lo();
    const int* hi_phibc = m_phiV_bc.hi();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      m_bcrec_state[PHIV].setLo(idim, phiV_bc[lo_phibc[idim]]);
      m_bcrec_state[PHIV].setHi(idim, phiV_bc[hi_phibc[idim]]);
    }

    // Hack charged species BCs
    int FIRSTIONinVar = FIRSTSPEC + NUM_SPECIES - NUM_IONS;
    int FIRSTIONinSpec = NUM_SPECIES - NUM_IONS;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      for (int n = 0; n < NUM_IONS; ++n) {
        auto const bcIonSave = m_bcrec_state[FIRSTIONinVar + n];
        m_bcrec_state[FIRSTIONinVar + n] =
          hackBCChargedParticle(zk[FIRSTIONinSpec + n], bcIonSave);
      }
    }
    // Need to hack nE too actually ...
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      auto const bcnESave = m_bcrec_state[NE];
      m_bcrec_state[NE] = hackBCChargedParticle(-1.0, bcnESave);
    }
#endif
#ifdef PELE_USE_SOOT
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      for (int mom = 0; mom < NUMSOOTVAR; ++mom) {
        m_bcrec_state[FIRSTSOOT + mom].setLo(idim, soot_bc[lo_bc[idim]]);
        m_bcrec_state[FIRSTSOOT + mom].setHi(idim, soot_bc[hi_bc[idim]]);
      }
    }
#endif
    // Used by fetchBCRecDummyArray for slab interpolation in interior-only
    // contexts where domain BCs must not be applied
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      m_bcrec_dummy.setLo(idim, amrex::BCType::int_dir);
      m_bcrec_dummy.setHi(idim, amrex::BCType::int_dir);
    }
  }
}

amrex::Vector<amrex::BCRec>
PeleLM::fetchBCRecArray(const int scomp, const int ncomp)
{
  amrex::Vector<amrex::BCRec> bc(ncomp);
  for (int comp = 0; comp < ncomp; ++comp) {
    bc[comp] = m_bcrec_state[scomp + comp];
  }
  return bc;
}

amrex::Vector<amrex::BCRec>
PeleLM::fetchBCRecAuxArray(const int scomp, const int ncomp)
{
  amrex::Vector<amrex::BCRec> bc(ncomp);
  for (int comp = 0; comp < ncomp; ++comp) {
    bc[comp] = m_bcrec_aux[scomp + comp];
  }
  return bc;
}

amrex::Vector<amrex::BCRec>
PeleLM::fetchBCRecDummyArray(const int ncomp)
{
  amrex::Vector<amrex::BCRec> bc(ncomp);
  for (int comp = 0; comp < ncomp; ++comp) {
    bc[comp] = m_bcrec_dummy;
  }
  return bc;
}

//-----------------------------------------------------------------------------
// The following work directly on the leveldata
// Fill the entire class state at once
void
PeleLM::fillPatchState(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::fillPatchState()");
  for (int lev = 0; lev <= finest_level; ++lev) {
    fillPatchState(lev, a_time);
  }
}

// Fill the a given level class state
void
PeleLM::fillPatchState(const int lev, const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::fillPatchStateLev()");

  auto* ldata_p = getLevelDataPtr(lev, a_time);
  const amrex::Real time = getTime(lev, a_time);

  fillpatch_state(lev, time, ldata_p->state, m_nGrowState);
  if (m_incompressible == 0) {
    if (m_has_divu != 0) {
      fillpatch_divu(lev, time, ldata_p->divu, ldata_p->divu.nGrow());
    }
  }
}

// Fill a state components
void
PeleLM::fillPatchDensity(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::fillPatchDensity()");
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_time);
    const amrex::Real time = getTime(lev, a_time);
    fillpatch_density(lev, time, ldata_p->state, DENSITY, m_nGrowState);
  }
}

void
PeleLM::fillPatchSpecies(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::fillPatchSpecies()");
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_time);
    const amrex::Real time = getTime(lev, a_time);
    fillpatch_species(lev, time, ldata_p->state, FIRSTSPEC, m_nGrowState);
  }
}

void
PeleLM::fillPatchTemp(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::fillPatchTemp()");
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_time);
    const amrex::Real time = getTime(lev, a_time);
    fillpatch_temp(lev, time, ldata_p->state, TEMP, m_nGrowState);
  }
}

void
PeleLM::fillPatchAux(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::fillPatchAux()");
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_time);
    const amrex::Real time = getTime(lev, a_time);
    fillpatch_aux(lev, time, ldata_p->auxiliaries, m_nGrowState);
  }
}

#ifdef PELE_USE_PLASMA
void
PeleLM::fillPatchPhiV(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::fillPatchPhiV()");
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto ldata_p = getLevelDataPtr(lev, a_time);
    const amrex::Real time = getTime(lev, a_time);
    fillpatch_phiV(lev, time, ldata_p->state, PHIV, m_nGrowState);
  }
}
#endif
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// The following return a fillpatched MF ptr at a given level
// Fill the entire state at once
std::unique_ptr<amrex::MultiFab>
PeleLM::fillPatchState(const int lev, const amrex::Real a_time, const int nGrow)
{
  BL_PROFILE("PeleLMeX::fillPatchState()");

  std::unique_ptr<amrex::MultiFab> mf;
  if (m_incompressible != 0) {
    mf = std::make_unique<amrex::MultiFab>(
      grids[lev], dmap[lev], AMREX_SPACEDIM, nGrow, amrex::MFInfo(),
      Factory(lev));
  } else {
    mf = std::make_unique<amrex::MultiFab>(
      grids[lev], dmap[lev], NVAR, nGrow, amrex::MFInfo(), Factory(lev));
  }
  fillpatch_state(lev, a_time, *mf, nGrow);

  return mf;
}

std::unique_ptr<amrex::MultiFab>
PeleLM::fillPatchReact(const int lev, const amrex::Real a_time, const int nGrow)
{
  BL_PROFILE("PeleLMeX::fillPatchReact()");

#ifdef PELE_USE_PLASMA
  constexpr int IRsize = NUM_SPECIES + 1;
#else
  constexpr int IRsize = NUM_SPECIES;
#endif
  std::unique_ptr<amrex::MultiFab> mf;
  mf = std::make_unique<amrex::MultiFab>(
    grids[lev], dmap[lev], IRsize, nGrow, amrex::MFInfo(), Factory(lev));
  fillpatch_reaction(lev, a_time, *mf, nGrow);

  return mf;
}
//-----------------------------------------------------------------------------

// Fill the state
void
PeleLM::fillpatch_state(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_state,
  const int nGhost)
{
  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();
  const int recycFull =
    static_cast<int>(m_recycling_mode == RecyclingMode::Full);

  const int nCompState = (m_incompressible) != 0 ? AMREX_SPACEDIM : NVAR;

  if (m_use_inlet_from_plane != 0) {
    const auto bcrec = fetchBCRecArray(XVEL, XVEL + AMREX_SPACEDIM);
    const auto domain = geom[lev].Domain();
    const int idir = m_inlet_plane_dir;
    auto face_is_recycling_inflow = [=](int is_hi) -> bool {
      for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        const auto& vbc = bcrec[n];
        const int bctype = (is_hi != 0) ? vbc.hi(idir) : vbc.lo(idir);
        if (bctype != amrex::BCType::ext_dir) {
          return false;
        }
      }
      return true;
    };
    const bool zero_lo = face_is_recycling_inflow(0);
    const bool zero_hi = face_is_recycling_inflow(1);
    if (zero_lo || zero_hi) {

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(a_state); mfi.isValid(); ++mfi) {
        const amrex::Box& vbx = mfi.validbox();
        const amrex::Box& gbx = mfi.fabbox();
        if (zero_lo && vbx.smallEnd(idir) == domain.smallEnd(idir)) {
          amrex::Box lobx = gbx;
          lobx.setBig(idir, vbx.smallEnd(idir) - 1);
          if (lobx.ok()) {
            a_state[mfi].setVal<amrex::RunOn::Device>(
              0.0, lobx, VELX, AMREX_SPACEDIM);
          }
        }
        if (zero_hi && vbx.bigEnd(idir) == domain.bigEnd(idir)) {
          amrex::Box hibx = gbx;
          hibx.setSmall(idir, vbx.bigEnd(idir) + 1);
          if (hibx.ok()) {
            a_state[mfi].setVal<amrex::RunOn::Device>(
              0.0, hibx, VELX, AMREX_SPACEDIM);
          }
        }
      }
    }
  }

  fillTurbInflow(a_state, VELX, lev, a_time);

  if (m_use_inlet_from_plane != 0) {
    fillFromRecyclingPlane(a_state, 0, lev);
  }

  if (lev == 0) {
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirState<ProblemSpecificFunctions>>>
      bndry_func(
        geom[lev], fetchBCRecArray(0, nCompState),
        PeleLMCCFillExtDirState<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux,
          static_cast<int>(turb_inflow.is_initialized()),
          m_use_inlet_from_plane, m_inlet_plane_dir, recycFull, m_map_eval});
    FillPatchSingleLevel(
      a_state, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev]->state), &(m_leveldata_new[lev]->state)},
      {m_t_old[lev], m_t_new[lev]}, 0, 0, nCompState, geom[lev], bndry_func, 0);
  } else {

    // Whole-state fill: request the mapping-aware interpolator for the
    // lev-1 -> lev pair (its per-component weights serve both
    // incompressible and compressible state).  getInterpolator falls back
    // to the legacy interpolator when mesh mapping is off or unavailable.
    auto* mapper = getInterpolator(1, lev - 1);

    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirState<ProblemSpecificFunctions>>>
      crse_bndry_func(
        geom[lev - 1], fetchBCRecArray(0, nCompState),
        PeleLMCCFillExtDirState<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux,
          static_cast<int>(turb_inflow.is_initialized()),
          m_use_inlet_from_plane, m_inlet_plane_dir, recycFull, m_map_eval});
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirState<ProblemSpecificFunctions>>>
      fine_bndry_func(
        geom[lev], fetchBCRecArray(0, nCompState),
        PeleLMCCFillExtDirState<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux,
          static_cast<int>(turb_inflow.is_initialized()),
          m_use_inlet_from_plane, m_inlet_plane_dir, recycFull, m_map_eval});
    FillPatchTwoLevels(
      a_state, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev - 1]->state), &(m_leveldata_new[lev - 1]->state)},
      {m_t_old[lev - 1], m_t_new[lev - 1]},
      {&(m_leveldata_old[lev]->state), &(m_leveldata_new[lev]->state)},
      {m_t_old[lev], m_t_new[lev]}, 0, 0, nCompState, geom[lev - 1], geom[lev],
      crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1), mapper,
      fetchBCRecArray(0, nCompState), 0);
  }

  a_state.EnforcePeriodicity(geom[lev].periodicity());
}

// Fill the density
void
PeleLM::fillpatch_density(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_density,
  const int rho_comp,
  const int nGhost)
{
  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();
  if (lev == 0) {

    // Density
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDens<ProblemSpecificFunctions>>>
      bndry_func_rho(
        geom[lev], fetchBCRecArray(DENSITY, 1),
        PeleLMCCFillExtDirDens<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    FillPatchSingleLevel(
      a_density, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev]->state), &(m_leveldata_new[lev]->state)},
      {m_t_old[lev], m_t_new[lev]}, DENSITY, rho_comp, 1, geom[lev],
      bndry_func_rho, 0);

  } else {

    // Interpolator
    auto* mapper = getInterpolator();

    // Density
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDens<ProblemSpecificFunctions>>>
      crse_bndry_func_rho(
        geom[lev - 1], fetchBCRecArray(DENSITY, 1),
        PeleLMCCFillExtDirDens<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDens<ProblemSpecificFunctions>>>
      fine_bndry_func_rho(
        geom[lev], fetchBCRecArray(DENSITY, 1),
        PeleLMCCFillExtDirDens<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    FillPatchTwoLevels(
      a_density, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev - 1]->state), &(m_leveldata_new[lev - 1]->state)},
      {m_t_old[lev - 1], m_t_new[lev - 1]},
      {&(m_leveldata_old[lev]->state), &(m_leveldata_new[lev]->state)},
      {m_t_old[lev], m_t_new[lev]}, DENSITY, rho_comp, 1, geom[lev - 1],
      geom[lev], crse_bndry_func_rho, 0, fine_bndry_func_rho, 0,
      refRatio(lev - 1), mapper, fetchBCRecArray(DENSITY, 1), 0);
  }
}

// Fill the mass fractions
void
PeleLM::fillpatch_species(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_species,
  const int rhoY_comp,
  const int nGhost)
{
  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();
  if (lev == 0) {

    // Species
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirSpec<ProblemSpecificFunctions>>>
      bndry_func(
        geom[lev], fetchBCRecArray(FIRSTSPEC, NUM_SPECIES),
        PeleLMCCFillExtDirSpec<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    FillPatchSingleLevel(
      a_species, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev]->state), &(m_leveldata_new[lev]->state)},
      {m_t_old[lev], m_t_new[lev]}, FIRSTSPEC, rhoY_comp, NUM_SPECIES,
      geom[lev], bndry_func, 0);
  } else {

    // Interpolator
    auto* mapper = getInterpolator();

    // Species
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirSpec<ProblemSpecificFunctions>>>
      crse_bndry_func(
        geom[lev - 1], fetchBCRecArray(FIRSTSPEC, NUM_SPECIES),
        PeleLMCCFillExtDirSpec<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirSpec<ProblemSpecificFunctions>>>
      fine_bndry_func(
        geom[lev], fetchBCRecArray(FIRSTSPEC, NUM_SPECIES),
        PeleLMCCFillExtDirSpec<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    FillPatchTwoLevels(
      a_species, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev - 1]->state), &(m_leveldata_new[lev - 1]->state)},
      {m_t_old[lev - 1], m_t_new[lev - 1]},
      {&(m_leveldata_old[lev]->state), &(m_leveldata_new[lev]->state)},
      {m_t_old[lev], m_t_new[lev]}, FIRSTSPEC, rhoY_comp, NUM_SPECIES,
      geom[lev - 1], geom[lev], crse_bndry_func, 0, fine_bndry_func, 0,
      refRatio(lev - 1), mapper, fetchBCRecArray(FIRSTSPEC, NUM_SPECIES), 0);
  }
}

// Fill temperature
void
PeleLM::fillpatch_temp(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_temp,
  const int temp_comp,
  const int nGhost)
{
  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();
  if (lev == 0) {
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirTemp<ProblemSpecificFunctions>>>
      bndry_func(
        geom[lev], fetchBCRecArray(TEMP, 1),
        PeleLMCCFillExtDirTemp<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    FillPatchSingleLevel(
      a_temp, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev]->state), &(m_leveldata_new[lev]->state)},
      {m_t_old[lev], m_t_new[lev]}, TEMP, temp_comp, 1, geom[lev], bndry_func,
      0);
  } else {

    // Interpolator
    auto* mapper = getInterpolator();

    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirTemp<ProblemSpecificFunctions>>>
      crse_bndry_func(
        geom[lev - 1], fetchBCRecArray(TEMP, 1),
        PeleLMCCFillExtDirTemp<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirTemp<ProblemSpecificFunctions>>>
      fine_bndry_func(
        geom[lev], fetchBCRecArray(TEMP, 1),
        PeleLMCCFillExtDirTemp<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    FillPatchTwoLevels(
      a_temp, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev - 1]->state), &(m_leveldata_new[lev - 1]->state)},
      {m_t_old[lev - 1], m_t_new[lev - 1]},
      {&(m_leveldata_old[lev]->state), &(m_leveldata_new[lev]->state)},
      {m_t_old[lev], m_t_new[lev]}, TEMP, temp_comp, 1, geom[lev - 1],
      geom[lev], crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1),
      mapper, fetchBCRecArray(TEMP, 1), 0);
  }
}

// Fill the auxiliaries
void
PeleLM::fillpatch_aux(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_aux,
  const int nGhost)
{

  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();

  if (lev == 0) {
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirAux<ProblemSpecificFunctions>>>
      bndry_func(
        geom[lev], fetchBCRecAuxArray(0, m_nAux),
        PeleLMCCFillExtDirAux<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    FillPatchSingleLevel(
      a_aux, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev]->auxiliaries),
       &(m_leveldata_new[lev]->auxiliaries)},
      {m_t_old[lev], m_t_new[lev]}, 0, 0, m_nAux, geom[lev], bndry_func, 0);
  } else {

    // Interpolator
    auto* mapper = getInterpolator();

    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirAux<ProblemSpecificFunctions>>>
      crse_bndry_func(
        geom[lev - 1], fetchBCRecAuxArray(0, m_nAux),
        PeleLMCCFillExtDirAux<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirAux<ProblemSpecificFunctions>>>
      fine_bndry_func(
        geom[lev], fetchBCRecAuxArray(0, m_nAux),
        PeleLMCCFillExtDirAux<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    FillPatchTwoLevels(
      a_aux, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev - 1]->auxiliaries),
       &(m_leveldata_new[lev - 1]->auxiliaries)},
      {m_t_old[lev - 1], m_t_new[lev - 1]},
      {&(m_leveldata_old[lev]->auxiliaries),
       &(m_leveldata_new[lev]->auxiliaries)},
      {m_t_old[lev], m_t_new[lev]}, 0, 0, m_nAux, geom[lev - 1], geom[lev],
      crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1), mapper,
      fetchBCRecAuxArray(0, m_nAux), 0);
  }

  a_aux.EnforcePeriodicity(geom[lev].periodicity());
}

#ifdef PELE_USE_PLASMA
// Fill electro-static potential
void
PeleLM::fillpatch_phiV(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_temp,
  const int phiV_comp,
  const int nGhost)
{
  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();
  if (lev == 0) {
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirPhiV<ProblemSpecificFunctions>>>
      bndry_func(
        geom[lev], fetchBCRecArray(PHIV, 1),
        PeleLMCCFillExtDirPhiV<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    FillPatchSingleLevel(
      a_temp, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev]->state), &(m_leveldata_new[lev]->state)},
      {m_t_old[lev], m_t_new[lev]}, PHIV, phiV_comp, 1, geom[lev], bndry_func,
      0);
  } else {

    // Interpolator
    auto* mapper = getInterpolator();

    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirPhiV<ProblemSpecificFunctions>>>
      crse_bndry_func(
        geom[lev - 1], fetchBCRecArray(PHIV, 1),
        PeleLMCCFillExtDirPhiV<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    amrex::PhysBCFunct<
      amrex::GpuBndryFuncFab<PeleLMCCFillExtDirPhiV<ProblemSpecificFunctions>>>
      fine_bndry_func(
        geom[lev], fetchBCRecArray(PHIV, 1),
        PeleLMCCFillExtDirPhiV<ProblemSpecificFunctions>{
          lprobparm, lpmfdata, m_nAux, m_map_eval});
    FillPatchTwoLevels(
      a_temp, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev - 1]->state), &(m_leveldata_new[lev - 1]->state)},
      {m_t_old[lev - 1], m_t_new[lev - 1]},
      {&(m_leveldata_old[lev]->state), &(m_leveldata_new[lev]->state)},
      {m_t_old[lev], m_t_new[lev]}, PHIV, phiV_comp, 1, geom[lev - 1],
      geom[lev], crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1),
      mapper, fetchBCRecArray(PHIV, 1), 0);
  }
}
#endif

// Fill the divU
void
PeleLM::fillpatch_divu(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_divu,
  const int nGhost)
{
  if (lev == 0) {
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      bndry_func(geom[lev], {m_bcrec_divu}, PeleLMCCFillExtDirDummy{m_nAux});
    FillPatchSingleLevel(
      a_divu, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev]->divu), &(m_leveldata_new[lev]->divu)},
      {m_t_old[lev], m_t_new[lev]}, 0, 0, 1, geom[lev], bndry_func, 0);
  } else {

    // Interpolator
    auto* mapper = getInterpolator();

    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      crse_bndry_func(
        geom[lev - 1], {m_bcrec_divu}, PeleLMCCFillExtDirDummy{m_nAux});
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      fine_bndry_func(
        geom[lev], {m_bcrec_divu}, PeleLMCCFillExtDirDummy{m_nAux});
    FillPatchTwoLevels(
      a_divu, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev - 1]->divu), &(m_leveldata_new[lev - 1]->divu)},
      {m_t_old[lev - 1], m_t_new[lev - 1]},
      {&(m_leveldata_old[lev]->divu), &(m_leveldata_new[lev]->divu)},
      {m_t_old[lev], m_t_new[lev]}, 0, 0, 1, geom[lev - 1], geom[lev],
      crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1), mapper,
      {m_bcrec_divu}, 0);
  }
}

// Fillpatch a vector of forces:
// -> actually only modifies the ghost cells : fillBoundary, C/F interp,
// foextrap on domain BCs
void
PeleLM::fillpatch_forces(
  const amrex::Real a_time,
  amrex::Vector<amrex::MultiFab*> const& a_force,
  const int nGrowForce)
{
  AMREX_ASSERT(a_force[0]->nComp() <= m_bcrec_force.size());
  const int nComp = a_force[0]->nComp();

  int lev = 0;
  {
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      bndry_func(geom[lev], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    FillPatchSingleLevel(
      *a_force[lev], amrex::IntVect(nGrowForce), a_time, {a_force[lev]},
      {a_time}, 0, 0, nComp, geom[lev], bndry_func, 0);
  }
  for (lev = 1; lev <= finest_level; ++lev) {
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      crse_bndry_func(
        geom[lev - 1], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      fine_bndry_func(
        geom[lev], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    amrex::Interpolater* mapper = &amrex::pc_interp;
    FillPatchTwoLevels(
      *a_force[lev], amrex::IntVect(nGrowForce), a_time, {a_force[lev - 1]},
      {a_time}, {a_force[lev]}, {a_time}, 0, 0, nComp, geom[lev - 1], geom[lev],
      crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1), mapper,
      {m_bcrec_force}, 0);
  }
}

// Fill the gradp
void
PeleLM::fillpatch_gradp(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_gp,
  const int nGhost)
{
  if (lev == 0) {
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      bndry_func(geom[lev], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    FillPatchSingleLevel(
      a_gp, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev]->gp), &(m_leveldata_new[lev]->gp)},
      {m_t_old[lev], m_t_new[lev]}, 0, 0, AMREX_SPACEDIM, geom[lev], bndry_func,
      0);
  } else {

    // Interpolator
    auto* mapper = getInterpolator();

    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      crse_bndry_func(
        geom[lev - 1], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      fine_bndry_func(
        geom[lev], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    FillPatchTwoLevels(
      a_gp, amrex::IntVect(nGhost), a_time,
      {&(m_leveldata_old[lev - 1]->gp), &(m_leveldata_new[lev - 1]->gp)},
      {m_t_old[lev - 1], m_t_new[lev - 1]},
      {&(m_leveldata_old[lev]->gp), &(m_leveldata_new[lev]->gp)},
      {m_t_old[lev], m_t_new[lev]}, 0, 0, AMREX_SPACEDIM, geom[lev - 1],
      geom[lev], crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1),
      mapper, {m_bcrec_force}, 0);
  }
}

// Fill the reaction data
void
PeleLM::fillpatch_reaction(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_I_R,
  const int nGhost)
{
  if (lev == 0) {
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      bndry_func(geom[lev], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    FillPatchSingleLevel(
      a_I_R, amrex::IntVect(nGhost), a_time, {&(m_leveldatareact[lev]->I_R)},
      {a_time}, 0, 0, nCompIR(), geom[lev], bndry_func, 0);
  } else {

    // Interpolator
    auto* mapper = getInterpolator();

    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      crse_bndry_func(
        geom[lev - 1], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      fine_bndry_func(
        geom[lev], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    FillPatchTwoLevels(
      a_I_R, amrex::IntVect(nGhost), a_time,
      {&(m_leveldatareact[lev - 1]->I_R)}, {a_time},
      {&(m_leveldatareact[lev]->I_R)}, {a_time}, 0, 0, nCompIR(), geom[lev - 1],
      geom[lev], crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1),
      mapper, {m_bcrec_force}, 0);
  }
}

// Fill functC
void
PeleLM::fillpatch_chemFunctCall(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_fctC,
  const int nGhost)
{
  if (lev == 0) {
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      bndry_func(geom[lev], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    FillPatchSingleLevel(
      a_fctC, amrex::IntVect(nGhost), a_time,
      {&(m_leveldatareact[lev]->functC)}, {a_time}, 0, 0, 1, geom[lev],
      bndry_func, 0);
  } else {

    // Interpolator
    auto* mapper = getInterpolator();

    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      crse_bndry_func(
        geom[lev - 1], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
      fine_bndry_func(
        geom[lev], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
    FillPatchTwoLevels(
      a_fctC, amrex::IntVect(nGhost), a_time,
      {&(m_leveldatareact[lev - 1]->functC)}, {a_time},
      {&(m_leveldatareact[lev]->functC)}, {a_time}, 0, 0, 1, geom[lev - 1],
      geom[lev], crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1),
      mapper, {m_bcrec_force}, 0);
  }
}

// Fill the state
void
PeleLM::fillcoarsepatch_state(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_state,
  const int nGhost)
{
  AMREX_ASSERT(lev > 0);
  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();
  const int recycFull =
    static_cast<int>(m_recycling_mode == RecyclingMode::Full);

  const int nCompState = (m_incompressible) != 0 ? AMREX_SPACEDIM : NVAR;

  fillTurbInflow(a_state, VELX, lev, a_time);

  if (m_use_inlet_from_plane != 0) {
    fillFromRecyclingPlane(a_state, 0, lev);
  }

  // Whole-state coarse->fine fill: request the mapping-aware interpolator
  // for the lev-1 -> lev pair (engages only with cell-conservative regrid
  // interp, which it wraps; else getInterpolator returns the legacy one).
  // Derived TEMP/RHORT are restored by the EOS recompute in
  // MakeNewLevelFromCoarse / RemakeLevel.
  auto* mapper = getInterpolator(m_regrid_interp_method, lev - 1);

  amrex::PhysBCFunct<
    amrex::GpuBndryFuncFab<PeleLMCCFillExtDirState<ProblemSpecificFunctions>>>
    crse_bndry_func(
      geom[lev - 1], fetchBCRecArray(0, nCompState),
      PeleLMCCFillExtDirState<ProblemSpecificFunctions>{
        lprobparm, lpmfdata, m_nAux,
        static_cast<int>(turb_inflow.is_initialized()), m_use_inlet_from_plane,
        m_inlet_plane_dir, recycFull, m_map_eval});
  amrex::PhysBCFunct<
    amrex::GpuBndryFuncFab<PeleLMCCFillExtDirState<ProblemSpecificFunctions>>>
    fine_bndry_func(
      geom[lev], fetchBCRecArray(0, nCompState),
      PeleLMCCFillExtDirState<ProblemSpecificFunctions>{
        lprobparm, lpmfdata, m_nAux,
        static_cast<int>(turb_inflow.is_initialized()), m_use_inlet_from_plane,
        m_inlet_plane_dir, recycFull, m_map_eval});
  InterpFromCoarseLevel(
    a_state, amrex::IntVect(nGhost), a_time, m_leveldata_new[lev - 1]->state, 0,
    0, nCompState, geom[lev - 1], geom[lev], crse_bndry_func, 0,
    fine_bndry_func, 0, refRatio(lev - 1), mapper,
    fetchBCRecArray(0, nCompState), 0);
}

// Fill the auxiliaries
void
PeleLM::fillcoarsepatch_aux(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_aux,
  const int nGhost)
{
  AMREX_ASSERT(lev > 0);
  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();

  // Interpolator
  auto* mapper = getInterpolator(m_regrid_interp_method);

  amrex::PhysBCFunct<
    amrex::GpuBndryFuncFab<PeleLMCCFillExtDirAux<ProblemSpecificFunctions>>>
    crse_bndry_func(
      geom[lev - 1], fetchBCRecAuxArray(0, m_nAux),
      PeleLMCCFillExtDirAux<ProblemSpecificFunctions>{
        lprobparm, lpmfdata, m_nAux, m_map_eval});
  amrex::PhysBCFunct<
    amrex::GpuBndryFuncFab<PeleLMCCFillExtDirAux<ProblemSpecificFunctions>>>
    fine_bndry_func(
      geom[lev], fetchBCRecAuxArray(0, m_nAux),
      PeleLMCCFillExtDirAux<ProblemSpecificFunctions>{
        lprobparm, lpmfdata, m_nAux, m_map_eval});
  InterpFromCoarseLevel(
    a_aux, amrex::IntVect(nGhost), a_time,
    m_leveldata_new[lev - 1]->auxiliaries, 0, 0, m_nAux, geom[lev - 1],
    geom[lev], crse_bndry_func, 0, fine_bndry_func, 0, refRatio(lev - 1),
    mapper, fetchBCRecAuxArray(0, m_nAux), 0);
}

// Fill the grad P
void
PeleLM::fillcoarsepatch_gradp(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_gp,
  const int nGhost)
{
  // Interpolator
  auto* mapper = getInterpolator(m_regrid_interp_method);

  amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
    crse_bndry_func(
      geom[lev - 1], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
  amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
    fine_bndry_func(
      geom[lev], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
  InterpFromCoarseLevel(
    a_gp, amrex::IntVect(nGhost), a_time, m_leveldata_new[lev - 1]->gp, 0, 0,
    AMREX_SPACEDIM, geom[lev - 1], geom[lev], crse_bndry_func, 0,
    fine_bndry_func, 0, refRatio(lev - 1), mapper, {m_bcrec_force}, 0);
}

// Fill the divu
void
PeleLM::fillcoarsepatch_divu(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_divu,
  const int nGhost)
{
  // Interpolator
  auto* mapper = getInterpolator(m_regrid_interp_method);

  amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
    crse_bndry_func(
      geom[lev - 1], {m_bcrec_divu}, PeleLMCCFillExtDirDummy{m_nAux});
  amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
    fine_bndry_func(geom[lev], {m_bcrec_divu}, PeleLMCCFillExtDirDummy{m_nAux});
  InterpFromCoarseLevel(
    a_divu, amrex::IntVect(nGhost), a_time, m_leveldata_new[lev - 1]->divu, 0,
    0, 1, geom[lev - 1], geom[lev], crse_bndry_func, 0, fine_bndry_func, 0,
    refRatio(lev - 1), mapper, {m_bcrec_divu}, 0);
}

// Fill coarse patch of reaction
void
PeleLM::fillcoarsepatch_reaction(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_I_R,
  const int nGhost)
{
  // Interpolator
  auto* mapper = getInterpolator(m_regrid_interp_method);

  amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
    crse_bndry_func(
      geom[lev - 1], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
  amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
    fine_bndry_func(
      geom[lev], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
  InterpFromCoarseLevel(
    a_I_R, amrex::IntVect(nGhost), a_time, m_leveldatareact[lev - 1]->I_R, 0, 0,
    nCompIR(), geom[lev - 1], geom[lev], crse_bndry_func, 0, fine_bndry_func, 0,
    refRatio(lev - 1), mapper, {m_bcrec_force}, 0);
}

// Fill coarse patch of chem function call
void
PeleLM::fillcoarsepatch_chemFunctCall(
  const int lev,
  const amrex::Real a_time,
  amrex::MultiFab& a_fctC,
  const int nGhost)
{
  // Interpolator
  auto* mapper = getInterpolator(m_regrid_interp_method);

  amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
    crse_bndry_func(
      geom[lev - 1], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
  amrex::PhysBCFunct<amrex::GpuBndryFuncFab<PeleLMCCFillExtDirDummy>>
    fine_bndry_func(
      geom[lev], {m_bcrec_force}, PeleLMCCFillExtDirDummy{m_nAux});
  InterpFromCoarseLevel(
    a_fctC, amrex::IntVect(nGhost), a_time, m_leveldatareact[lev - 1]->functC,
    0, 0, 1, geom[lev - 1], geom[lev], crse_bndry_func, 0, fine_bndry_func, 0,
    refRatio(lev - 1), mapper, {m_bcrec_force}, 0);
}

// Fill the inflow boundary of a velocity MF
// used for velocity projection
void
PeleLM::setInflowBoundaryVel(
  amrex::MultiFab& a_vel, const int lev, const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::setInflowBoundaryVel()");

  const amrex::Real time = getTime(lev, a_time);

  fillTurbInflow(a_vel, 0, lev, time);

  if (m_use_inlet_from_plane != 0) {
    fillFromRecyclingPlane(a_vel, 0, lev);
  }

  // Create a dummy BCRec from Velocity BCRec keeping only Inflow and set the
  // other to bogus
  auto realVelBCRec = fetchBCRecArray(VELX, AMREX_SPACEDIM);
  amrex::Vector<amrex::BCRec> dummyVelBCRec(AMREX_SPACEDIM);
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    for (int idim2 = 0; idim2 < AMREX_SPACEDIM; ++idim2) {
      if (realVelBCRec[idim].lo(idim2) == amrex::BCType::ext_dir) {
        dummyVelBCRec[idim].setLo(idim2, amrex::BCType::ext_dir);
      } else {
        dummyVelBCRec[idim].setLo(idim2, amrex::BCType::bogus);
      }
      if (realVelBCRec[idim].hi(idim2) == amrex::BCType::ext_dir) {
        dummyVelBCRec[idim].setHi(idim2, amrex::BCType::ext_dir);
      } else {
        dummyVelBCRec[idim].setHi(idim2, amrex::BCType::bogus);
      }
    }
  }

  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();
  const int recycFull =
    static_cast<int>(m_recycling_mode == RecyclingMode::Full);
  amrex::PhysBCFunct<
    amrex::GpuBndryFuncFab<PeleLMCCFillExtDirState<ProblemSpecificFunctions>>>
    bndry_func(
      geom[lev], dummyVelBCRec,
      PeleLMCCFillExtDirState<ProblemSpecificFunctions>{
        lprobparm, lpmfdata, m_nAux,
        static_cast<int>(turb_inflow.is_initialized()), m_use_inlet_from_plane,
        m_inlet_plane_dir, recycFull, m_map_eval});

  bndry_func(a_vel, 0, AMREX_SPACEDIM, a_vel.nGrowVect(), time, 0);

  a_vel.EnforcePeriodicity(geom[lev].periodicity());
}

void
PeleLM::addTurbInflowToBox(
  const amrex::Box& a_bx,
  amrex::FArrayBox& a_data,
  const int lev,
  const int dir,
  const amrex::Orientation::Side& a_side,
  const amrex::Real a_time)
{
#ifdef PELEPHYSICS_TURBINFLOW_HAS_COORD_ADDTURB
  if (m_mesh_mapping) {
    // The turbulence file is indexed by PHYSICAL position, whereas
    // geom[lev] describes the uniform Xi-space grid.  Hand the injection
    // face's physical cell-centre coordinates to TurbInflow explicitly.
    //
    // a_bx comes from adjCell{Lo,Hi}(Domain, dir, .) intersected with a
    // grown tilebox, so its transverse extents never leave the domain and
    // the mapping is only ever evaluated inside [ProbLo, ProbHi].
    int tdir1 = 0;
    int tdir2 = 0;
    pele::physics::turbinflow::TurbInflow::transverseDirs(dir, tdir1, tdir2);
    const auto gd = geom[lev].data();

    amrex::Vector<amrex::Real> x_phys(a_bx.length(tdir1));
    amrex::Vector<amrex::Real> y_phys(a_bx.length(tdir2));
    for (int i = 0; i < static_cast<int>(x_phys.size()); ++i) {
      x_phys[i] =
        x_phys_cc_periodic(m_map_eval, tdir1, a_bx.smallEnd(tdir1) + i, gd);
    }
    for (int j = 0; j < static_cast<int>(y_phys.size()); ++j) {
      y_phys[j] =
        x_phys_cc_periodic(m_map_eval, tdir2, a_bx.smallEnd(tdir2) + j, gd);
    }

    turb_inflow.add_turb(
      a_bx, a_data, 0, geom[lev].Domain(), x_phys, y_phys, a_time, dir, a_side);
    return;
  }
#endif
  turb_inflow.add_turb(a_bx, a_data, 0, geom[lev], a_time, dir, a_side);
}

void
PeleLM::fillTurbInflow(
  amrex::MultiFab& a_vel,
  const int vel_comp,
  const int lev,
  const amrex::Real a_time)
{
  if (turb_inflow.is_initialized()) {

    ProbParm* probparmDD = PeleLM::prob_parm_d;
    ProbParm* probparmDH = PeleLM::prob_parm;

    // Velocity BCs
    auto velBCRec = fetchBCRecArray(VELX, AMREX_SPACEDIM);

    // Copy problem parameter structs to host
    amrex::Gpu::copy(
      amrex::Gpu::deviceToHost, probparmDD, probparmDD + 1, probparmDH);

    // When active control drives the inlet velocity, march through the turb
    // data using the time-integrated controlled velocity (convected distance)
    // instead of turb_conv_vel * time. This keeps the marcher consistent with
    // the time-varying mean inflow and avoids the position jumps that would
    // arise from rescaling turb_conv_vel against absolute time.
    if (m_ctrl_active != 0) {
      const amrex::Real dtl = a_time - m_ctrl_tBase;
      const amrex::Real conv_dist =
        m_turb_conv_dist + m_ctrl_V_in * dtl + 0.5 * m_ctrl_dV * dtl * dtl;
      turb_inflow.set_convected_distance(conv_dist);
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(a_vel, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      amrex::Box const& bx = mfi.growntilebox();
      amrex::FArrayBox& data = a_vel[mfi];

      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {

        auto bndryBoxLO =
          amrex::Box(amrex::adjCellLo(geom[lev].Domain(), dir, 4) & bx);
        if (
          velBCRec[0].lo()[dir] == amrex::BCType::ext_dir && bndryBoxLO.ok()) {
          // Create box with ghost cells and set them to zero
          amrex::IntVect growVect(amrex::IntVect::TheUnitVector());
          constexpr int Grow = 4; // Being conservative
          for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            growVect[n] = Grow;
          }
          growVect[dir] = 0;
          amrex::Box modDom = geom[lev].Domain();
          modDom.grow(growVect);
          auto bndryBoxLO_ghost =
            amrex::Box(amrex::adjCellLo(modDom, dir, Grow) & bx);
          data.setVal<amrex::RunOn::Device>(
            0.0, bndryBoxLO_ghost, vel_comp, AMREX_SPACEDIM);

          addTurbInflowToBox(
            bndryBoxLO, data, lev, dir, amrex::Orientation::low, a_time);
        }

        auto bndryBoxHI =
          amrex::Box(amrex::adjCellHi(geom[lev].Domain(), dir, 4) & bx);
        if (
          velBCRec[0].hi()[dir] == amrex::BCType::ext_dir && bndryBoxHI.ok()) {
          // Create box with ghost cells and set them to zero
          amrex::IntVect growVect(amrex::IntVect::TheUnitVector());
          constexpr int Grow = 4;
          for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            growVect[n] = Grow;
          }
          growVect[dir] = 0;
          amrex::Box modDom = geom[lev].Domain();
          modDom.grow(growVect);
          auto bndryBoxHI_ghost =
            amrex::Box(amrex::adjCellHi(modDom, dir, Grow) & bx);
          data.setVal<amrex::RunOn::Device>(
            0.0, bndryBoxHI_ghost, vel_comp, AMREX_SPACEDIM);

          addTurbInflowToBox(
            bndryBoxHI, data, lev, dir, amrex::Orientation::high, a_time);
        }
      }
    }

    // Copy problem parameter structs back to device
    amrex::Gpu::copy(
      amrex::Gpu::hostToDevice, probparmDH, probparmDH + 1, probparmDD);
  }
}

int
PeleLM::computeRecyclingSrcIndex(int lev) const
{
  const int dir = m_inlet_plane_dir;
  // peleLM.inlet_plane_position is a PHYSICAL coordinate.  Under mesh
  // mapping the AMReX grid is the uniform Xi grid, so invert the mapping
  // before converting to an index; with no mapping active the inverse is
  // the identity and this reduces to the original expression.
  const amrex::Real xi_position =
    m_map_eval.xi_from_x_phys(dir, m_inlet_plane_position, geom[lev].data());
  auto srcIndex = static_cast<int>(std::lround(
    (xi_position - geom[lev].ProbLo()[dir]) / geom[lev].CellSize()[dir] - 0.5));
  const auto& dom = geom[lev].Domain();
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    srcIndex >= dom.smallEnd(dir) && srcIndex <= dom.bigEnd(dir),
    "computeRecyclingSrcIndex: inlet_plane_position maps outside the level "
    "domain");
  return srcIndex;
}

static amrex::DistributionMapping
extendDM(
  const amrex::DistributionMapping& olddm,    // length N
  const amrex::Vector<amrex::Long>& new_wgts, // length M (use 1s if unweighted)
  const amrex::Vector<amrex::Long>& old_wgts) // length N (use 1s if unweighted)
{
  const int nprocs = amrex::ParallelDescriptor::NProcs();
  const auto& pmap = olddm.ProcessorMap();
  const int N = static_cast<int>(pmap.size());
  const int M = static_cast<int>(new_wgts.size());

  // 1) current load per rank from the frozen N entries
  amrex::Vector<amrex::Long> load(nprocs, 0);
  for (int i = 0; i < N; ++i) {
    load[pmap[i]] += old_wgts[i];
  }

  // 2) min-heap of (load, rank)
  using PII = std::pair<amrex::Long, int>;
  std::priority_queue<PII, std::vector<PII>, std::greater<>> pq;
  for (int r = 0; r < nprocs; ++r) {
    pq.emplace(load[r], r);
  }

  // 3) assign M new boxes heaviest-first to lightest rank
  amrex::Vector<int> order(M);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return new_wgts[a] > new_wgts[b];
  });

  amrex::Vector<int> new_pmap(N + M);
  std::copy(pmap.begin(), pmap.end(), new_pmap.begin());

  for (int k : order) {
    auto [ld, r] = pq.top();
    pq.pop();
    new_pmap[N + k] = r;
    pq.emplace(ld + new_wgts[k], r);
  }

  // 4) wrap into a DistributionMapping
  return amrex::DistributionMapping(std::move(new_pmap));
}

void
PeleLM::interpRecyclingSlabFromCoarse(
  amrex::MultiFab& a_fine, const amrex::MultiFab& a_crse, int lev)
{
  AMREX_ASSERT(lev > 0);

  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();
  const int recycFull =
    static_cast<int>(m_recycling_mode == RecyclingMode::Full);
  // All-int_dir dummy BCRecs disable boundary handling: the slab lies in the
  // domain interior, so PhysBCFunct calls on the temporaries are no-ops in
  // practice. The time argument is likewise informational only.
  auto velBCRec = fetchBCRecDummyArray(AMREX_SPACEDIM);

  // Use piecewise-constant interpolation: the source slab is one cell
  // thick along planeDir, so any stencil-based interpolator (e.g.,
  // cell-conservative linear) would read garbage from the slab's
  // planeDir ghost cells. PCInterp has no transverse stencil and is
  // adequate for injecting a fluctuation field across a refinement
  // boundary.
  // NOTE: Declared as InterpBase* (not auto* / MFPCInterp*) so AMReX's
  // FillPatchInterp dispatches through its runtime dynamic_cast path
  // and picks the MultiFab-based interpolator entry point.
  amrex::InterpBase* mapper = &amrex::mf_pc_interp;
  amrex::PhysBCFunct<
    amrex::GpuBndryFuncFab<PeleLMCCFillExtDirState<ProblemSpecificFunctions>>>
    crse_bndry_func(
      geom[lev - 1], velBCRec,
      PeleLMCCFillExtDirState<ProblemSpecificFunctions>{
        lprobparm, lpmfdata, m_nAux,
        static_cast<int>(turb_inflow.is_initialized()), m_use_inlet_from_plane,
        m_inlet_plane_dir, recycFull});
  amrex::PhysBCFunct<
    amrex::GpuBndryFuncFab<PeleLMCCFillExtDirState<ProblemSpecificFunctions>>>
    fine_bndry_func(
      geom[lev], velBCRec,
      PeleLMCCFillExtDirState<ProblemSpecificFunctions>{
        lprobparm, lpmfdata, m_nAux,
        static_cast<int>(turb_inflow.is_initialized()), m_use_inlet_from_plane,
        m_inlet_plane_dir, recycFull});
  amrex::InterpFromCoarseLevel(
    a_fine, amrex::IntVect(0), m_cur_time, a_crse, 0, 0, AMREX_SPACEDIM,
    geom[lev - 1], geom[lev], crse_bndry_func, 0, fine_bndry_func, 0,
    refRatio(lev - 1), mapper, velBCRec, 0);
}

void
PeleLM::buildRecyclingPlaneStorage()
{
  if (m_use_inlet_from_plane == 0) {
    return;
  }

  const int planeDir = m_inlet_plane_dir;
  const int nlevels = finest_level + 1;

  // The slab's union at each level always spans the full transverse
  // cross-section at srcIndex, but the individual boxes and the
  // DistributionMapping follow the level's grids and therefore change across
  // regrids. The running mean is a physical-space quantity and can be carried
  // through with a ParallelCopy. Stash the existing means before reallocating
  // so we can re-deposit them into the new MultiFabs.
  auto saved_mean = std::move(m_inlet_recycling.mean_src);
  const bool saved_initialized = m_inlet_recycling.initialized;
  const int saved_n_samples = m_inlet_recycling.n_samples;

  // The flux-control inlet mask lives on the level-0 slab layout; rebuild
  // it lazily against the new layout.
  m_recycling_inlet_mask.reset();

  m_inlet_recycling.u_src.clear();
  m_inlet_recycling.fluct_src.clear();
  m_inlet_recycling.mean_src.clear();
  m_inlet_recycling.u_src.resize(nlevels);
  m_inlet_recycling.mean_src.resize(nlevels);
  m_inlet_recycling.fluct_src.resize(nlevels);
#ifdef AMREX_USE_EB
  m_inlet_recycling.mask.clear();
  m_inlet_recycling.mask.resize(nlevels);
#endif

  for (int lev = 0; lev < nlevels; ++lev) {
    const int srcIndex = computeRecyclingSrcIndex(lev);
    const amrex::Box& domain = geom[lev].Domain();
    AMREX_ALWAYS_ASSERT(
      srcIndex >= domain.smallEnd(planeDir) &&
      srcIndex <= domain.bigEnd(planeDir));

    // Thin slab spanning the entire transverse cross-section at srcIndex.
    // Build it by intersecting the slab with this level's existing grids (vs an
    // intersections call) so the slab MultiFabs inherit the same processor
    // ownership as the source state data and subsequent ParallelCopy operations
    // preserve locality.
    amrex::Box slab = domain;
    slab.setSmall(planeDir, srcIndex);
    slab.setBig(planeDir, srcIndex);

    const amrex::BoxArray& level_ba = boxArray(lev);
    const amrex::DistributionMapping& level_dm = DistributionMap(lev);
    const auto& level_pmap = level_dm.ProcessorMap();
    amrex::BoxList slab_bl;
    amrex::Vector<int> slab_pmap;
    amrex::Vector<amrex::Long> slab_wgts;
    const auto isects = level_ba.intersections(slab);
    for (const auto& isect : isects) {
      slab_bl.push_back(isect.second);
      slab_wgts.push_back(isect.second.numPts());
      slab_pmap.push_back(level_pmap[isect.first]);
    }

    if (slab_pmap.empty() && lev == 0) {
      // Level 0 grids cover the domain, so this can only trip on a
      // degenerate setup; keep the guard for safety.
      amrex::Print()
        << "WARNING: inlet recycling source slab does not intersect any grids "
        << "on level " << lev << ". Skipping slab storage allocation for slab "
        << slab << " at srcIndex=" << srcIndex << "\n";
      continue;
    }
    amrex::BoxArray slab_ba(slab_bl);
    amrex::DistributionMapping slab_dm;
    if (!slab_pmap.empty()) {
      slab_dm = amrex::DistributionMapping(slab_pmap);
    }

    // Add boxes fillable from next coarser level - check if even lev-1 is
    // not big enough. When this level's grids do not touch the plane at all,
    // the entire slab is carried as coarse-interpolated data: the level may
    // still own part of the inlet face (e.g. refinement around the inlet with
    // the sampling plane downstream at coarser resolution), and
    // fillFromRecyclingPlane needs a populated fluct_src to inject there.
    if (lev > 0) {
      amrex::BoxList unfilled_bl = amrex::complementIn(slab, slab_bl);
      if (unfilled_bl.isNotEmpty()) {
        unfilled_bl.maxSize(max_grid_size[lev]);
        amrex::Vector<amrex::Long> unfilled_wgts;
        for (const auto& it : unfilled_bl) {
          unfilled_wgts.push_back(it.numPts());
        }
        slab_dm = extendDM(slab_dm, unfilled_wgts, slab_wgts);
        slab_bl.join(unfilled_bl);
        slab_ba = amrex::BoxArray(slab_bl);
      }
    }

    m_inlet_recycling.u_src[lev] =
      std::make_unique<amrex::MultiFab>(slab_ba, slab_dm, AMREX_SPACEDIM, 0);
    m_inlet_recycling.mean_src[lev] =
      std::make_unique<amrex::MultiFab>(slab_ba, slab_dm, AMREX_SPACEDIM, 0);
    m_inlet_recycling.fluct_src[lev] =
      std::make_unique<amrex::MultiFab>(slab_ba, slab_dm, AMREX_SPACEDIM, 0);

    // No fluctuation until the next snapshot has populated u_src and
    // recomputed it.
    m_inlet_recycling.fluct_src[lev]->setVal(0.0);

#ifdef AMREX_USE_EB
    // Build a fresh EB factory at the slab's BoxArray to obtain per-cell
    // flags at this level's resolution, then translate to a 0/1 mask:
    // 0 = EB-covered (excluded from the running mean; zero fluctuation),
    // 1 = regular or cut (included).
    auto slab_eb_factory = amrex::makeEBFabFactory(
      geom[lev], slab_ba, slab_dm, {AMREX_D_DECL(0, 0, 0)},
      amrex::EBSupport::basic);
    const auto& flags = slab_eb_factory->getMultiEBCellFlagFab();

    m_inlet_recycling.mask[lev] =
      std::make_unique<amrex::iMultiFab>(slab_ba, slab_dm, 1, 0);
    auto& mask_lev = *m_inlet_recycling.mask[lev];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mask_lev, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto& flagarr = flags.const_array(mfi);
      auto const& mask_arr = mask_lev.array(mfi);
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          mask_arr(i, j, k) = flagarr(i, j, k).isCovered() ? 0 : 1;
        });
    }
#endif

    if (
      saved_initialized && lev < static_cast<int>(saved_mean.size()) &&
      saved_mean[lev]) {
      m_inlet_recycling.mean_src[lev]->ParallelCopy(
        *saved_mean[lev], 0, 0, AMREX_SPACEDIM);
    } else if (
      saved_initialized && lev > 0 &&
      m_inlet_recycling.mean_src[lev - 1] != nullptr) {
      // This level is new since the last (re)build: seed its running mean by
      // interpolating the coarser level's mean (already deposited by this
      // ascending loop). Seeding with zero while `initialized` stays true
      // would make the next fluctuation on this level fluct = u - 0, i.e.
      // the full velocity, and the inlet would receive roughly twice the
      // mean flow until the running mean recovers.
      interpRecyclingSlabFromCoarse(
        *m_inlet_recycling.mean_src[lev], *m_inlet_recycling.mean_src[lev - 1],
        lev);
    } else {
      // Either we never had a mean, or there is no coarser level to seed
      // this level from.
      m_inlet_recycling.mean_src[lev]->setVal(0.0);
    }
  }

  if (saved_initialized) {
    // Carry the running statistics forward for any levels that already had
    // accumulated means. Newly created levels were initialized above without
    // discarding the existing running statistics.
    m_inlet_recycling.initialized = true;
    m_inlet_recycling.n_samples = saved_n_samples;
  } else {
    // No prior running statistics were available; let the next snapshot seed.
    m_inlet_recycling.initialized = false;
    m_inlet_recycling.n_samples = 0;
  }
}

void
PeleLM::updateRecyclingPlaneSnapshot()
{
  if (m_use_inlet_from_plane == 0) {
    return;
  }

  if (m_recycling_needs_rebuild) {
    buildRecyclingPlaneStorage();
    m_recycling_needs_rebuild = false;
  }

  // The new-time velocity is what we sample; ensure storage exists.
  AMREX_ASSERT(
    static_cast<int>(m_inlet_recycling.u_src.size()) == finest_level + 1);

  for (int lev = 0; lev <= finest_level; ++lev) {
    if (m_inlet_recycling.u_src[lev] == nullptr) {
      continue;
    }
    auto& u_src = *m_inlet_recycling.u_src[lev];
    const auto& state_lev = m_leveldata_new[lev]->state;

    // Every level's state covers some, but in general not all, of the
    // transverse cross-section at srcIndex. Strategy:
    //   1. Lev > 0: interpolate from the (already-filled) coarser slab to
    //      get a complete coverage at this level's resolution.
    //   2. Overwrite from this level's own state where it covers, using the
    //      higher-resolution data.
    //   NOTE: Relies on data filled at next coarse level, filled by this loop.
    if (lev > 0) {

      const auto* coarse_u_src = m_inlet_recycling.u_src[lev - 1].get();

      if (coarse_u_src != nullptr) {
        interpRecyclingSlabFromCoarse(u_src, *coarse_u_src, lev);
      }
    }
    // Same-level data overrides the coarse-interpolated baseline anywhere
    // this level's grids cover the slab.
    u_src.ParallelCopy(
      state_lev, VELX, 0, AMREX_SPACEDIM, 0, 0, geom[lev].periodicity());

#ifdef AMREX_USE_EB
    // EB-covered cells contain undefined storage; zero them in u_src so they
    // contribute nothing to the running mean (since 0 is the steady value
    // there) and produce a zero fluctuation downstream.
    if (
      lev < static_cast<int>(m_inlet_recycling.mask.size()) &&
      m_inlet_recycling.mask[lev]) {
      const auto& mask_lev = *m_inlet_recycling.mask[lev];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(u_src, amrex::TilingIfNotGPU()); mfi.isValid();
           ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const auto& mask_arr = mask_lev.const_array(mfi);
        auto const& u_arr = u_src.array(mfi);
        amrex::ParallelFor(
          bx, AMREX_SPACEDIM,
          [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            if (mask_arr(i, j, k) == 0) {
              u_arr(i, j, k, n) = 0.0;
            }
          });
      }
    }
#endif
  }

  if (m_recycling_mode == RecyclingMode::Full) {
    // Full-velocity injection: the sampled velocity itself is the boundary
    // data; no running mean is maintained. fluct_src doubles as the
    // injection buffer consumed by fillFromRecyclingPlane.
    for (int lev = 0; lev <= finest_level; ++lev) {
      if (m_inlet_recycling.u_src[lev] == nullptr) {
        continue;
      }
      amrex::MultiFab::Copy(
        *m_inlet_recycling.fluct_src[lev], *m_inlet_recycling.u_src[lev], 0, 0,
        AMREX_SPACEDIM, 0);
    }
    if (m_inlet_plane_flux_control != 0) {
      applyRecyclingFluxControl();
    }
    if (!m_inlet_recycling.initialized) {
      m_inlet_recycling.initialized = true;
      m_inlet_recycling.n_samples = 1;
    } else {
      ++m_inlet_recycling.n_samples;
    }
    return;
  }

  // Update the running mean and store the current fluctuation.
  // NOTE: m_inlet_recycling.n_samples counts the snapshots accumulated into
  // the running mean; the warmup gate in fillFromRecyclingPlane compares
  // against it so that fluctuations computed from too few samples are not
  // injected.

  if (!m_inlet_recycling.initialized) {
    // Seed: <u> = u_0; fluctuation defined as zero on the seeding sample.
    for (int lev = 0; lev <= finest_level; ++lev) {
      if (m_inlet_recycling.u_src[lev] == nullptr) {
        continue;
      }
      amrex::MultiFab::Copy(
        *m_inlet_recycling.mean_src[lev], *m_inlet_recycling.u_src[lev], 0, 0,
        AMREX_SPACEDIM, 0);
      m_inlet_recycling.fluct_src[lev]->setVal(0.0);
    }
    m_inlet_recycling.initialized = true;
    m_inlet_recycling.n_samples = 1;
    return;
  }

  // alpha for exponential moving average; if no window is set, fall back to a
  // cumulative average via 1/n.
  amrex::Real alpha;
  if (m_inlet_plane_avg_window > 0.0) {
    alpha = amrex::min<amrex::Real>(1.0, m_dt / m_inlet_plane_avg_window);
    if (alpha == 1.0_rt && !m_warned_clipped_recycle_avg_window_this_step) {
      m_warned_clipped_recycle_avg_window_this_step = true;
      amrex::Print()
        << "WARNING: inlet_plane_avg_window <= dt, so recycle averaging "
           "alpha is clipped to 1; the running mean equals the current "
           "sample and the fluctuation will be approximately zero.\n";
    }
  } else {
    // Cumulative average: the mean currently holds n_samples snapshots, so
    // the incoming sample enters with weight 1/(n_samples + 1).
    alpha = 1.0 / static_cast<amrex::Real>(m_inlet_recycling.n_samples + 1);
  }
  const amrex::Real one_minus_alpha = 1.0 - alpha;

  for (int lev = 0; lev <= finest_level; ++lev) {
    if (m_inlet_recycling.u_src[lev] == nullptr) {
      continue;
    }
    auto& mean = *m_inlet_recycling.mean_src[lev];
    auto& u_src = *m_inlet_recycling.u_src[lev];
    auto& fluct = *m_inlet_recycling.fluct_src[lev];

    // fluct = u_src - mean
    amrex::MultiFab::LinComb(
      fluct, 1.0, u_src, 0, -1.0, mean, 0, 0, AMREX_SPACEDIM, 0);
    // mean = (1 - alpha) * mean + alpha * u_src
    amrex::MultiFab::LinComb(
      mean, one_minus_alpha, mean, 0, alpha, u_src, 0, 0, AMREX_SPACEDIM, 0);
  }
  ++m_inlet_recycling.n_samples;
}

amrex::Real
PeleLM::slabVolumeFlux(
  const amrex::MultiFab& a_slab,
  int a_comp,
  const amrex::iMultiFab* a_mask,
  const amrex::IntVect& a_mask_shift)
{
  // Cell face area transverse to the plane direction at level 0. Under mesh
  // mapping this is the unmapped area, but the controller only ever uses
  // RATIOS of integrals computed with identical weights, so a metric factor
  // common to target and sample cancels.
  const auto dx = geom[0].CellSizeArray();
  amrex::Real dA = 1.0;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (idim != m_inlet_plane_dir) {
      dA *= dx[idim];
    }
  }

  amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
  amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

  // a_slab's BoxArray is the (possibly shifted) level-0 slab BoxArray with
  // the slab's DistributionMapping, and so is the mask's, so MFIter indices
  // correspond 1:1 between the two.
  for (amrex::MFIter mfi(a_slab); mfi.isValid(); ++mfi) {
    const amrex::Box& bx = mfi.validbox();
    auto const& u_arr = a_slab.const_array(mfi, a_comp);
    const bool have_mask = (a_mask != nullptr);
    auto const& mask_arr =
      have_mask ? a_mask->const_array(mfi) : amrex::Array4<const int>{};
    const int s0 = a_mask_shift[0];
    const int s1 = (AMREX_SPACEDIM > 1) ? a_mask_shift[1] : 0;
    const int s2 = (AMREX_SPACEDIM > 2) ? a_mask_shift[2] : 0;
    reduce_op.eval(
      bx, reduce_data,
      [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
        amrex::Real m = 1.0;
        if (have_mask) {
          m = static_cast<amrex::Real>(mask_arr(i - s0, j - s1, k - s2));
        }
        return {m * u_arr(i, j, k)};
      });
  }

  amrex::Real sum = amrex::get<0>(reduce_data.value(reduce_op));
  amrex::ParallelDescriptor::ReduceRealSum(sum);
  return sum * dA;
}

amrex::Real
PeleLM::computeRecyclingTargetFlow(amrex::Orientation::Side a_side)
{
  const int planeDir = m_inlet_plane_dir;
  const amrex::Box& domain = geom[0].Domain();

  // Fill one ghost layer of a level-0 velocity MultiFab from bcnormal alone:
  // the recycling preload and synthetic-turbulence injection are disabled in
  // the fill functor, so the ghost cells receive the problem's target mean
  // inflow profile (a bcnormal that increments a preloaded value increments
  // zero here, which is the same thing).
  amrex::MultiFab vel(grids[0], dmap[0], AMREX_SPACEDIM, 1);
  vel.setVal(0.0);

  // Keep only ext_dir faces; everything else is bogus so the fill functor
  // leaves it alone (same construction as setInflowBoundaryVel).
  auto realVelBCRec = fetchBCRecArray(VELX, AMREX_SPACEDIM);
  amrex::Vector<amrex::BCRec> dummyVelBCRec(AMREX_SPACEDIM);
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    for (int idim2 = 0; idim2 < AMREX_SPACEDIM; ++idim2) {
      dummyVelBCRec[idim].setLo(
        idim2, (realVelBCRec[idim].lo(idim2) == amrex::BCType::ext_dir)
                 ? amrex::BCType::ext_dir
                 : amrex::BCType::bogus);
      dummyVelBCRec[idim].setHi(
        idim2, (realVelBCRec[idim].hi(idim2) == amrex::BCType::ext_dir)
                 ? amrex::BCType::ext_dir
                 : amrex::BCType::bogus);
    }
  }

  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();
  amrex::PhysBCFunct<
    amrex::GpuBndryFuncFab<PeleLMCCFillExtDirState<ProblemSpecificFunctions>>>
    bndry_func(
      geom[0], dummyVelBCRec,
      PeleLMCCFillExtDirState<ProblemSpecificFunctions>{
        lprobparm, lpmfdata, m_nAux, /*do_turbInflow=*/0,
        /*do_inletFromPlane=*/0, m_inlet_plane_dir, /*recycling_full=*/0,
        m_map_eval});
  bndry_func(vel, 0, AMREX_SPACEDIM, vel.nGrowVect(), m_cur_time, 0);

  // Copy the innermost ghost layer onto the slab layout and integrate over
  // the inlet-face fluid footprint, i.e. exactly the cross-section (and the
  // same quadrature) that the recycled sample is injected into.
  const int ghostIdx = (a_side == amrex::Orientation::low)
                         ? domain.smallEnd(planeDir) - 1
                         : domain.bigEnd(planeDir) + 1;
  const amrex::MultiFab& slab0 = *m_inlet_recycling.fluct_src[0];
  const int srcIndex = computeRecyclingSrcIndex(0);
  const amrex::IntVect shift_iv =
    amrex::BASISV(planeDir) * (ghostIdx - srcIndex);
  amrex::BoxArray shifted_ba(slab0.boxArray());
  shifted_ba.shift(shift_iv);
  amrex::MultiFab bc_slab(
    shifted_ba, slab0.DistributionMap(), AMREX_SPACEDIM, 0);
  bc_slab.ParallelCopy(vel, 0, 0, AMREX_SPACEDIM, 1, 0);

  const amrex::IntVect mask_shift =
    amrex::BASISV(planeDir) * (ghostIdx - m_recycling_inlet_mask_index);
  return slabVolumeFlux(
    bc_slab, planeDir, m_recycling_inlet_mask.get(), mask_shift);
}

void
PeleLM::applyRecyclingFluxControl()
{
  // Bulk mass-flux controller for full-mode recycling. The recycled inflow
  // is a feedback loop that carries no memory of the intended flow rate;
  // any net div(u) between the inlet and the recycle plane (wall heat
  // transfer, heat release) biases the loop and the bulk flow drifts until
  // the inlet reverses. Rescale the injection buffer so the injected
  // volumetric flow matches the bcnormal inflow target.
  //
  // The correction is multiplicative and applied to ALL velocity
  // components: the sampled fluctuation field is (to low-Mach accuracy)
  // solenoidal, with continuity coupling the normal and tangential
  // components mode by mode, so a uniform scaling preserves that structure
  // (div(gamma*u) = gamma*div(u)) and the injected turbulence passes
  // through the inlet-adjacent projection intact. Scaling the normal
  // component alone would leave an irrotational residual that the
  // projection removes by distorting both components near the inlet.
  // No-slip at walls is preserved (gamma*0 = 0), and the relative
  // turbulence intensity u'/U of the sample is carried through unchanged.
  const int planeDir = m_inlet_plane_dir;

  if (m_inlet_recycling.fluct_src.empty() || !m_inlet_recycling.fluct_src[0]) {
    return;
  }

  // Which side(s) of planeDir carry an ext_dir (inflow) velocity BC. Same
  // convention as fillFromRecyclingPlane: all velocity components on a face
  // must agree on ext_dir.
  auto velBCRec = fetchBCRecArray(VELX, AMREX_SPACEDIM);
  auto faceIsExtDir = [&](amrex::Orientation::Side side) {
    int n_extdir = 0;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      const int bc = (side == amrex::Orientation::low)
                       ? velBCRec[idim].lo()[planeDir]
                       : velBCRec[idim].hi()[planeDir];
      if (bc == amrex::BCType::ext_dir) {
        ++n_extdir;
      }
    }
    return n_extdir == AMREX_SPACEDIM;
  };
  const bool is_lo = faceIsExtDir(amrex::Orientation::low);
  const bool is_hi = faceIsExtDir(amrex::Orientation::high);
  if (is_lo && is_hi) {
    if (!m_warned_recycling_flux_both_sides) {
      m_warned_recycling_flux_both_sides = true;
      amrex::Print()
        << "WARNING: recycling flux control is disabled because both faces "
           "in inlet_plane_dir are inflow; a single bulk correction cannot "
           "serve two opposed inlets fed from one sampling plane.\n";
    }
    return;
  }
  if (!is_lo && !is_hi) {
    return;
  }

  // Lazily build the inlet-face fluid footprint on the slab layout: the
  // 0/1 mask of the first interior cell layer at the inflow face. Both
  // flux integrals are restricted to it, so the controller anchors the
  // flow that is actually injected even when the sampling plane sits in a
  // wider cross-section (e.g. past an expansion) and so that bcnormal
  // values behind an embedded boundary never contaminate the target.
  const amrex::Box& domain0 = geom[0].Domain();
  const int inletIdx =
    is_lo ? domain0.smallEnd(planeDir) : domain0.bigEnd(planeDir);
  if (!m_recycling_inlet_mask) {
    m_recycling_inlet_mask_index = inletIdx;
    const amrex::MultiFab& slab0 = *m_inlet_recycling.fluct_src[0];
    const int srcIndex0 = computeRecyclingSrcIndex(0);
    amrex::BoxArray inlet_ba(slab0.boxArray());
    inlet_ba.shift(amrex::BASISV(planeDir) * (inletIdx - srcIndex0));
    m_recycling_inlet_mask = std::make_unique<amrex::iMultiFab>(
      inlet_ba, slab0.DistributionMap(), 1, 0);
#ifdef AMREX_USE_EB
    auto inlet_eb_factory = amrex::makeEBFabFactory(
      geom[0], inlet_ba, slab0.DistributionMap(), {AMREX_D_DECL(0, 0, 0)},
      amrex::EBSupport::basic);
    const auto& inlet_flags = inlet_eb_factory->getMultiEBCellFlagFab();
    auto& inlet_mask = *m_recycling_inlet_mask;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(inlet_mask, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto& flagarr = inlet_flags.const_array(mfi);
      auto const& mask_arr = inlet_mask.array(mfi);
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          mask_arr(i, j, k) = flagarr(i, j, k).isCovered() ? 0 : 1;
        });
    }
#else
    m_recycling_inlet_mask->setVal(1);
#endif
    const amrex::Long n_fluid = m_recycling_inlet_mask->sum(0);
    amrex::Print() << " Recycling flux control: inlet fluid footprint = "
                   << n_fluid << " of "
                   << m_recycling_inlet_mask->boxArray().numPts()
                   << " face cells\n";
  }

  // Lazily capture the target flow. Signed along +planeDir, so target and
  // sample share a sign for either inflow side and gamma is positive for a
  // healthy through-flow. Assumes a statistically steady bcnormal inflow.
  if (std::isnan(m_inlet_plane_target_flow)) {
    m_inlet_plane_target_flow = computeRecyclingTargetFlow(
      is_lo ? amrex::Orientation::low : amrex::Orientation::high);
    amrex::Print() << " Recycling flux control: target inlet flow rate = "
                   << m_inlet_plane_target_flow << " (measured from bcnormal)"
                   << "\n";
  }
  const amrex::Real q_target = m_inlet_plane_target_flow;
  if (std::abs(q_target) < std::numeric_limits<amrex::Real>::min()) {
    if (!m_warned_recycling_flux_disabled) {
      m_warned_recycling_flux_disabled = true;
      amrex::Print()
        << "WARNING: recycling flux control is disabled because the target "
           "inlet flow rate is zero; set "
           "peleLM.inlet_plane_target_flow_rate or provide a bcnormal with "
           "a net through-flow.\n";
    }
    return;
  }

  // The level-0 slab always spans the full transverse cross-section;
  // restricted to the inlet footprint this is the through-flow of the part
  // of the sample about to be injected. (EB-covered cells at the sampling
  // plane were zeroed when the snapshot was taken.)
  const int srcIndex0 = computeRecyclingSrcIndex(0);
  const amrex::Real q_sample = slabVolumeFlux(
    *m_inlet_recycling.u_src[0], planeDir, m_recycling_inlet_mask.get(),
    amrex::BASISV(planeDir) * (srcIndex0 - m_recycling_inlet_mask_index));

  // A sample flux that collapsed or reversed cannot be rescaled into a
  // meaningful inflow; by the time this trips the run is unsalvageable, so
  // fail loudly rather than inject garbage. (A fresh start needs an initial
  // condition that carries the target through-flow past the sampling plane;
  // full mode has always required that to bootstrap.)
  if (
    q_sample * q_target <= 0.0 ||
    std::abs(q_sample) < 0.05 * std::abs(q_target)) {
    amrex::Print() << "Recycling flux control: sampled plane flow rate = "
                   << q_sample << ", target = " << q_target << "\n";
    amrex::Abort(
      "applyRecyclingFluxControl: the through-flow at the recycling plane "
      "collapsed or reversed (less than 5% of the target inlet flow, or "
      "opposite sign). The recycled inflow can no longer be rescaled to "
      "the target flow rate.");
  }

  amrex::Real gamma = q_target / q_sample;
  constexpr amrex::Real gamma_min = 0.5;
  constexpr amrex::Real gamma_max = 2.0;
  if (gamma < gamma_min || gamma > gamma_max) {
    const amrex::Real gamma_clipped =
      amrex::min(gamma_max, amrex::max(gamma_min, gamma));
    amrex::Print() << "WARNING: recycling flux control factor " << gamma
                   << " clipped to " << gamma_clipped
                   << "; the sampled plane flow is far from the target "
                      "inlet flow.\n";
    gamma = gamma_clipped;
  }

  for (int lev = 0; lev <= finest_level; ++lev) {
    if (m_inlet_recycling.fluct_src[lev] == nullptr) {
      continue;
    }
    m_inlet_recycling.fluct_src[lev]->mult(gamma, 0, AMREX_SPACEDIM, 0);
  }
  m_recycling_last_gamma = gamma;

  if (m_verbose > 2) {
    amrex::Print() << "   Recycling flux control: plane flow = " << q_sample
                   << ", target = " << q_target << ", gamma = " << gamma
                   << "\n";
  }
}

void
PeleLM::fillFromRecyclingPlane(amrex::MultiFab& a_vel, int vel_comp, int lev)
{
  // NOTE: Fluctuation data injected is refreshed only once per time step
  // (rather than per SDC iteration)

  if (m_use_inlet_from_plane == 0) {
    return;
  }

  // Recycling currently only handles cell-centered velocity data; staggered
  // (e.g., MAC) layouts would need a different shift convention.
  AMREX_ASSERT(a_vel.boxArray().ixType().cellCentered());

  const int planeDir = m_inlet_plane_dir;
  const int srcIndex = computeRecyclingSrcIndex(lev);
  const amrex::Box& domain = geom[lev].Domain();

  auto velBCRec = fetchBCRecArray(VELX, AMREX_SPACEDIM);
  const amrex::BoxArray& ba = a_vel.boxArray();
  const int nGrowDest = a_vel.nGrow();

  // Recycling injects all AMREX_SPACEDIM velocity components as a single
  // operation, so it only makes sense when every velocity component on the
  // affected face is ext_dir. Disagreement is almost certainly a malformed
  // input, so abort rather than silently picking a side.
  auto faceIsExtDir = [&](amrex::Orientation::Side side) {
    int n_extdir = 0;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      const int bc = (side == amrex::Orientation::low)
                       ? velBCRec[idim].lo()[planeDir]
                       : velBCRec[idim].hi()[planeDir];
      if (bc == amrex::BCType::ext_dir) {
        ++n_extdir;
      }
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      n_extdir == 0 || n_extdir == AMREX_SPACEDIM,
      "fillFromRecyclingPlane: velocity components disagree on ext_dir for "
      "the recycling face; all components must share the same BC.");
    return n_extdir == AMREX_SPACEDIM;
  };

  // Decide which sides of planeDir need recycling. Uses only global
  // quantities (BCRec + domain + ba), so the result is identical on
  // every rank and the subsequent ParallelAdd calls remain collective.
  bool need_lo = false;
  bool need_hi = false;
  amrex::BoxArray grown_ba(ba);
  grown_ba.grow(nGrowDest);
  if (faceIsExtDir(amrex::Orientation::low)) {
    need_lo =
      grown_ba.intersects(amrex::adjCellLo(domain, planeDir, nGrowDest));
  }
  if (faceIsExtDir(amrex::Orientation::high)) {
    need_hi =
      grown_ba.intersects(amrex::adjCellHi(domain, planeDir, nGrowDest));
  }

  if (!need_lo && !need_hi) {
    return;
  }

  bool set_zero_in_bndry = false;

  // Storage may not yet exist (first call before updateRecyclingPlaneSnapshot,
  // or this level didn't exist when the snapshot last ran). Fall back to the
  // standard ext_dir fill silently.
  if (
    lev >= static_cast<int>(m_inlet_recycling.fluct_src.size()) ||
    !m_inlet_recycling.fluct_src[lev]) {
    set_zero_in_bndry = true;
  }

  // Don't inject anything until the running mean has had a chance to settle.
  // The warmup gate applies only to fluctuations mode: a full-velocity
  // sample is valid from the first snapshot, so full mode injects as soon
  // as the buffer is seeded.
  if (
    !m_inlet_recycling.initialized ||
    (m_recycling_mode == RecyclingMode::Fluctuations &&
     m_inlet_recycling.n_samples <= m_inlet_plane_warmup_steps)) {
    set_zero_in_bndry = true;
  }

  if (set_zero_in_bndry) {

    // Land here if we do not have trustworthy data for inflow
    if (need_lo) {
      auto bndryBox = amrex::Box(domain).grow(planeDir, nGrowDest);
      bndryBox.setBig(planeDir, domain.smallEnd(planeDir) - 1);
      a_vel.setVal(0.0, bndryBox, 0, AMREX_SPACEDIM, nGrowDest);
    }
    if (need_hi) {
      auto bndryBox = amrex::Box(domain).grow(planeDir, nGrowDest);
      bndryBox.setSmall(planeDir, domain.bigEnd(planeDir) + 1);
      a_vel.setVal(0.0, bndryBox, 0, AMREX_SPACEDIM, nGrowDest);
    }

  } else {

    amrex::MultiFab& fluct = *m_inlet_recycling.fluct_src[lev];

    // Fill nGrowDest grow-cell layers on one side of the domain in planeDir.
    // Strategy: one ParallelCopy populates only the innermost grow layer;
    // the remaining (nGrowDest - 1) layers are filled by purely local
    // FArrayBox self-copies inside each a_vel FAB that abuts the boundary.
    auto fill_side = [&](amrex::Orientation::Side side) {
      const bool is_lo = (side == amrex::Orientation::low);

      // planeDir-index of the innermost grow cell (just outside the domain
      // on this side) and the step direction toward deeper layers.
      const int innermost_target =
        is_lo ? (domain.smallEnd(planeDir) - 1) : (domain.bigEnd(planeDir) + 1);
      const int step = is_lo ? -1 : +1;

      // (1) Build a single-cell-thick slab of fluct shifted to the innermost
      //     grow cell. Default factory: no EB ops on the shifted BoxArray.
      const int nshift = innermost_target - srcIndex;
      const amrex::IntVect shift_iv = amrex::BASISV(planeDir) * nshift;

      amrex::BoxArray shifted_ba(fluct.boxArray());
      shifted_ba.shift(shift_iv);
      amrex::MultiFab shifted_fluct(
        shifted_ba, fluct.DistributionMap(), fluct.nComp(), 0, amrex::MFInfo());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(fluct); mfi.isValid(); ++mfi) {
        amrex::FArrayBox const& src_fab = fluct[mfi];
        amrex::FArrayBox& dst_fab = shifted_fluct[mfi];
        const amrex::Box& src_bx = src_fab.box();
        const amrex::Box dst_bx = amrex::shift(src_bx, shift_iv);
        if (amrex::Gpu::inLaunchRegion()) {
          dst_fab.copy<amrex::RunOn::Gpu>(
            src_fab, src_bx, 0, dst_bx, 0, fluct.nComp());
        } else {
          dst_fab.copy<amrex::RunOn::Host>(
            src_fab, src_bx, 0, dst_bx, 0, fluct.nComp());
        }
      }

      // (2) Single collective: populates a_vel's innermost grow layer on
      //     this side wherever shifted_fluct intersects a_vel + nGrowDest.
      a_vel.ParallelCopy(
        shifted_fluct, 0, vel_comp, AMREX_SPACEDIM, 0, nGrowDest);

      // (3) Locally replicate the now-filled innermost layer into the deeper
      //     (nGrowDest - 1) layers, only for a_vel FABs whose valid region
      //     touches this domain face. No MPI.
      if (nGrowDest > 1) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(a_vel); mfi.isValid(); ++mfi) {
          const amrex::Box& valid = mfi.validbox();
          const bool abuts =
            is_lo ? (valid.smallEnd(planeDir) == domain.smallEnd(planeDir))
                  : (valid.bigEnd(planeDir) == domain.bigEnd(planeDir));
          if (!abuts) {
            continue;
          }

          amrex::FArrayBox& fab = a_vel[mfi];

          // Source slice: the freshly-filled innermost grow layer, with the
          // FAB's valid transverse extent (which is the extent ParallelCopy
          // populated when fluct covers the full transverse face).
          amrex::Box src_slice = valid;
          src_slice.setSmall(planeDir, innermost_target);
          src_slice.setBig(planeDir, innermost_target);

          for (int g = 2; g <= nGrowDest; ++g) {
            const int target = innermost_target + step * (g - 1);
            amrex::Box dst_slice = src_slice;
            dst_slice.setSmall(planeDir, target);
            dst_slice.setBig(planeDir, target);
            if (amrex::Gpu::inLaunchRegion()) {
              fab.copy<amrex::RunOn::Gpu>(
                fab, src_slice, vel_comp, dst_slice, vel_comp, AMREX_SPACEDIM);
            } else {
              fab.copy<amrex::RunOn::Host>(
                fab, src_slice, vel_comp, dst_slice, vel_comp, AMREX_SPACEDIM);
            }
          }
        }
      }
    };

    if (need_lo) {
      fill_side(amrex::Orientation::low);
    }
    if (need_hi) {
      fill_side(amrex::Orientation::high);
    }
  }
}
