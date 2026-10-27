#include <PeleLMeX.H>
#include <PeleLMeX_K.H>
#include <PeleLMeX_Utils.H>
#include <pelelmex_prob.H>

#ifdef AMREX_USE_EB
#include <PeleLMeX_BCfillEB.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EB_Redistribution.H>
#include <AMReX_EBMFInterpolater.H>

void
PeleLM::makeEBGeometry()
{
  BL_PROFILE("PeleLMeX::makeEBGeometry()");
  int max_coarsening_level = 100;
  int req_coarsening_level = static_cast<int>(geom.size()) - 1;

  // Read the geometry type and act accordingly
  amrex::ParmParse ppeb2("eb2");
  std::string geom_type;
  ppeb2.get("geom_type", geom_type);

  // At what level should the EB be generated ?
  // Default : max_level
  int prev_max_lvl_eb = max_level;

  // If restarting: use the level at which it was generated earlier
  prev_max_lvl_eb =
    (getRestartEBMaxLevel() > 0) ? getRestartEBMaxLevel() : prev_max_lvl_eb;

  // Manual override if needed -> might incur issues with previously
  // covered/uncovered cell flipping
  ppeb2.query("max_level_generation", prev_max_lvl_eb);
  AMREX_ALWAYS_ASSERT(prev_max_lvl_eb <= max_level);

  // Stash away the level at which we ended up
  m_EB_generate_max_level = prev_max_lvl_eb;

  // Generate the EB data at prev_max_lvl_eb and create consistent coarse
  // version from there.
  if (geom_type == "UserDefined") {
    ProblemSpecificFunctions::EBUserDefined(
      geom[prev_max_lvl_eb], req_coarsening_level, max_coarsening_level);
  } else {
    // If geom_type is not an AMReX recognized type, it'll crash.
    amrex::EB2::Build(
      geom[prev_max_lvl_eb], req_coarsening_level, max_coarsening_level);
  }

  // Add finer level, might be inconsistent with the coarser level created
  // above.
  amrex::EB2::addFineLevels(max_level - prev_max_lvl_eb);
}

void
PeleLM::redistributeAofS(
  const int a_lev,
  const amrex::Real a_dt,
  amrex::MultiFab& a_tmpDiv,
  const int div_comp,
  amrex::MultiFab& a_AofS,
  const int aofs_comp,
  amrex::MultiFab& a_state,
  const int state_comp,
  const int ncomp,
  const amrex::BCRec* d_bc,
  const amrex::Geometry& a_geom) const
{
  BL_PROFILE("PeleLMeX::redistributeAofS()");
  AMREX_ASSERT(a_tmpDiv.nComp() >= div_comp + ncomp);
  AMREX_ASSERT(a_AofS.nComp() >= aofs_comp + ncomp);
  AMREX_ASSERT(a_state.nComp() >= state_comp + ncomp);

  //----------------------------------------------------------------
  const auto& ebfact = EBFactory(a_lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(a_tmpDiv, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {

    amrex::Box const& bx = mfi.tilebox();
    auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
    auto const& flag = flagfab.const_array();
    auto const& divT_ar = a_tmpDiv.array(mfi, div_comp);
    auto const& aofs_ar = a_AofS.array(mfi, aofs_comp);

    if (flagfab.getType(bx) != amrex::FabType::covered) {
      if (flagfab.getType(grow(bx, 4)) != amrex::FabType::regular) {
        AMREX_D_TERM(
          auto apx = ebfact.getAreaFrac()[0]->const_array(mfi);
          , auto apy = ebfact.getAreaFrac()[1]->const_array(mfi);
          , auto apz = ebfact.getAreaFrac()[2]->const_array(mfi););
        AMREX_D_TERM(
          amrex::Array4<amrex::Real const> fcx =
            ebfact.getFaceCent()[0]->const_array(mfi);
          , amrex::Array4<amrex::Real const> fcy =
              ebfact.getFaceCent()[1]->const_array(mfi);
          , amrex::Array4<amrex::Real const> fcz =
              ebfact.getFaceCent()[2]->const_array(mfi););
        auto const& ccc = ebfact.getCentroid().const_array(mfi);
        auto const& vfrac_arr = ebfact.getVolFrac().const_array(mfi);

        // This is scratch space if calling StateRedistribute,
        // but is used as the weights (here set to 1) if calling
        // FluxRedistribute
        amrex::Box gbx = bx;

        if (m_adv_redist_type == "StateRedist") {
          gbx.grow(3);
        } else if (m_adv_redist_type == "FluxRedist") {
          gbx.grow(2);
        }

        amrex::FArrayBox tmpfab(gbx, ncomp, amrex::The_Async_Arena());
        amrex::Array4<amrex::Real> scratch = tmpfab.array(0);
        if (m_adv_redist_type == "FluxRedist") {
          amrex::ParallelFor(
            amrex::Box(scratch),
            [scratch] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              scratch(i, j, k) = 1.;
            });
        }
        ApplyRedistribution(
          bx, ncomp, aofs_ar, divT_ar, a_state.const_array(mfi, state_comp),
          scratch, flag, AMREX_D_DECL(apx, apy, apz), vfrac_arr,
          AMREX_D_DECL(fcx, fcy, fcz), ccc, d_bc, a_geom, a_dt,
          m_adv_redist_type);
      } else {
        // Move data to AofS for regular bx
        amrex::ParallelFor(
          bx, ncomp,
          [aofs_ar,
           divT_ar] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            aofs_ar(i, j, k, n) = divT_ar(i, j, k, n);
          });
      }
    }
  }
}

void
PeleLM::getCoveredIMask(const int a_lev, amrex::iMultiFab& a_imask) const
{
  const auto& ebfact = EBFactory(a_lev);
  const auto& flags = ebfact.getMultiEBCellFlagFab();

  if (
    (a_imask.boxArray() == flags.boxArray()) &&
    (a_imask.DistributionMap() == flags.DistributionMap())) {

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(a_imask, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto& flagarr = flags.const_array(mfi);
      amrex::Array4<int> const& arr = a_imask.array(mfi);
      AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k, {
        if (flagarr(i, j, k).isCovered()) {
          arr(i, j, k) = -1;
        } else {
          arr(i, j, k) = 1;
        }
      });
    }
  } else {

    amrex::iMultiFab mask_tmp(flags.boxArray(), flags.DistributionMap(), 1, 0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mask_tmp, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto& flagarr = flags.const_array(mfi);
      amrex::Array4<int> const& arr = mask_tmp.array(mfi);
      AMREX_HOST_DEVICE_PARALLEL_FOR_3D(bx, i, j, k, {
        if (flagarr(i, j, k).isCovered()) {
          arr(i, j, k) = -1;
        } else {
          arr(i, j, k) = 1;
        }
      });
    }
    a_imask.ParallelCopy(mask_tmp, 0, 0, 1);
  }
}

void
PeleLM::redistributeDiff(
  const int a_lev,
  const amrex::Real a_dt,
  amrex::MultiFab& a_tmpDiv,
  const int div_comp,
  amrex::MultiFab& a_diff,
  const int diff_comp,
  const amrex::MultiFab& a_state,
  const int state_comp,
  const int ncomp,
  const amrex::BCRec* d_bc,
  const amrex::Geometry& a_geom) const
{
  BL_PROFILE("PeleLMeX::redistributeDiff()");
  AMREX_ASSERT(a_tmpDiv.nComp() >= div_comp + ncomp);
  AMREX_ASSERT(a_diff.nComp() >= diff_comp + ncomp);
  AMREX_ASSERT(a_state.nComp() >= state_comp + ncomp);

  //----------------------------------------------------------------
  const auto& ebfact = EBFactory(a_lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(a_tmpDiv, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {

    amrex::Box const& bx = mfi.tilebox();
    auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
    auto const& flag = flagfab.const_array();
    auto const& divT_ar = a_tmpDiv.array(mfi, div_comp);
    auto const& diff_ar = a_diff.array(mfi, diff_comp);

    if (flagfab.getType(bx) != amrex::FabType::covered) {
      if (flagfab.getType(grow(bx, 4)) != amrex::FabType::regular) {
        AMREX_D_TERM(
          auto apx = ebfact.getAreaFrac()[0]->const_array(mfi);
          , auto apy = ebfact.getAreaFrac()[1]->const_array(mfi);
          , auto apz = ebfact.getAreaFrac()[2]->const_array(mfi););
        AMREX_D_TERM(
          amrex::Array4<amrex::Real const> fcx =
            ebfact.getFaceCent()[0]->const_array(mfi);
          , amrex::Array4<amrex::Real const> fcy =
              ebfact.getFaceCent()[1]->const_array(mfi);
          , amrex::Array4<amrex::Real const> fcz =
              ebfact.getFaceCent()[2]->const_array(mfi););
        auto const& ccc = ebfact.getCentroid().const_array(mfi);
        auto const& vfrac_arr = ebfact.getVolFrac().const_array(mfi);

        // This is scratch space if calling StateRedistribute,
        // but is used as the weights (here set to 1) if calling
        // FluxRedistribute
        amrex::Box gbx = bx;

        if (m_diff_redist_type == "StateRedist") {
          gbx.grow(3);
        } else if (m_diff_redist_type == "FluxRedist") {
          gbx.grow(2);
        }

        amrex::FArrayBox tmpfab(gbx, ncomp, amrex::The_Async_Arena());
        amrex::Array4<amrex::Real> scratch = tmpfab.array(0);
        if (m_diff_redist_type == "FluxRedist") {
          amrex::ParallelFor(
            amrex::Box(scratch),
            [scratch] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              scratch(i, j, k) = 1.;
            });
        }
        ApplyRedistribution(
          bx, ncomp, diff_ar, divT_ar, a_state.const_array(mfi, state_comp),
          scratch, flag, AMREX_D_DECL(apx, apy, apz), vfrac_arr,
          AMREX_D_DECL(fcx, fcy, fcz), ccc, d_bc, a_geom, a_dt,
          m_diff_redist_type);
      } else {
        // Move data to AofS for regular bx
        amrex::ParallelFor(
          bx, ncomp,
          [diff_ar,
           divT_ar] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            diff_ar(i, j, k, n) = divT_ar(i, j, k, n);
          });
      }
    }
  }
}

void
PeleLM::initCoveredState()
{
  // Zero velocities, typical values on species, 'cold' temperature
  if (m_incompressible != 0) {
    coveredState_h.resize(AMREX_SPACEDIM);
    AMREX_D_TERM(
      coveredState_h[0] = 0.0;, coveredState_h[1] = 0.0;
      , coveredState_h[2] = 0.0;)
    coveredState_d.resize(AMREX_SPACEDIM);
    amrex::Gpu::copy(
      amrex::Gpu::hostToDevice, coveredState_h.begin(), coveredState_h.end(),
      coveredState_d.begin());
  } else {
    coveredState_h.resize(NVAR);
    AMREX_D_TERM(
      coveredState_h[0] = 0.0;, coveredState_h[1] = 0.0;
      , coveredState_h[2] = 0.0;)
    coveredState_h[DENSITY] = typical_values[DENSITY];
    for (int n = 0; n < NUM_SPECIES; ++n) {
      coveredState_h[FIRSTSPEC + n] = typical_values[FIRSTSPEC + n];
    }
    coveredState_h[RHOH] = typical_values[RHOH];
    coveredState_h[TEMP] = 300.0;
    coveredState_h[RHORT] = typical_values[RHORT];

    coveredState_d.resize(NVAR);
    amrex::Gpu::copy(
      amrex::Gpu::hostToDevice, coveredState_h.begin(), coveredState_h.end(),
      coveredState_d.begin());
  }
}

void
PeleLM::setCoveredState(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::setCoveredState()");
  for (int lev = 0; lev <= finest_level; ++lev) {
    setCoveredState(lev, a_time);
  }
}

void
PeleLM::setCoveredState(const int lev, const TimeStamp a_time)
{
  AMREX_ASSERT(a_time == AmrOldTime || a_time == AmrNewTime);

  auto* ldata_p = getLevelDataPtr(lev, a_time);

  if (m_incompressible != 0) {
    EB_set_covered(ldata_p->state, 0, AMREX_SPACEDIM, coveredState_h);
  } else {
    EB_set_covered(ldata_p->state, 0, NVAR, coveredState_h);
  }
}

void
PeleLM::initialRedistribution()
{
  // Redistribute the initial solution if adv/diff scheme uses State or NewState
  if (
    m_adv_redist_type == "StateRedist" || m_diff_redist_type == "StateRedist") {

    for (int lev = 0; lev <= finest_level; ++lev) {

      // New time
      amrex::Real timeNew = getTime(lev, AmrNewTime);

      // Jungle with Old/New states: fillPatch old and redistribute
      // from Old->New to end up with proper New state
      auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
      auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);

      auto const& fact = EBFactory(lev);

      // State
      if (m_incompressible != 0) {
        amrex::Vector<amrex::Real> stateCovered(AMREX_SPACEDIM, 0.0);
        EB_set_covered(ldataNew_p->state, 0, AMREX_SPACEDIM, stateCovered);
        ldataNew_p->state.FillBoundary(geom[lev].periodicity());
        amrex::MultiFab::Copy(
          ldataOld_p->state, ldataNew_p->state, 0, 0, AMREX_SPACEDIM,
          m_nGrowState);
      } else {
        amrex::Vector<amrex::Real> stateCovered(NVAR, 0.0);
        EB_set_covered(ldataNew_p->state, 0, NVAR, stateCovered);
        ldataNew_p->state.FillBoundary(geom[lev].periodicity());
        amrex::MultiFab::Copy(
          ldataOld_p->state, ldataNew_p->state, 0, 0, NVAR, m_nGrowState);
      }
      fillpatch_state(lev, timeNew, ldataOld_p->state, m_nGrowState);

      for (amrex::MFIter mfi(ldataNew_p->state, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {

        const amrex::Box& bx = mfi.validbox();

        amrex::EBCellFlagFab const& flagfab = fact.getMultiEBCellFlagFab()[mfi];
        amrex::Array4<amrex::EBCellFlag const> const& flag =
          flagfab.const_array();

        if (
          (flagfab.getType(bx) != amrex::FabType::covered) &&
          (flagfab.getType(amrex::grow(bx, 4)) != amrex::FabType::regular)) {
          amrex::Array4<amrex::Real const> AMREX_D_DECL(fcx, fcy, fcz), ccc,
            vfrac, AMREX_D_DECL(apx, apy, apz);
          AMREX_D_TERM(
            fcx = fact.getFaceCent()[0]->const_array(mfi);
            , fcy = fact.getFaceCent()[1]->const_array(mfi);
            , fcz = fact.getFaceCent()[2]->const_array(mfi););
          ccc = fact.getCentroid().const_array(mfi);
          AMREX_D_TERM(
            apx = fact.getAreaFrac()[0]->const_array(mfi);
            , apy = fact.getAreaFrac()[1]->const_array(mfi);
            , apz = fact.getAreaFrac()[2]->const_array(mfi););
          vfrac = fact.getVolFrac().const_array(mfi);

          if (m_incompressible != 0) {
            auto bcRec = fetchBCRecArray(0, AMREX_SPACEDIM);
            auto bcRec_d = convertToDeviceVector(bcRec);
            ApplyInitialRedistribution(
              bx, AMREX_SPACEDIM, ldataNew_p->state.array(mfi, 0),
              ldataOld_p->state.array(mfi, 0), flag,
              AMREX_D_DECL(apx, apy, apz), vfrac, AMREX_D_DECL(fcx, fcy, fcz),
              ccc, bcRec_d.dataPtr(), geom[lev], m_adv_redist_type);
          } else {
            auto bcRec = fetchBCRecArray(0, NVAR);
            auto bcRec_d = convertToDeviceVector(bcRec);
            ApplyInitialRedistribution(
              bx, NVAR, ldataNew_p->state.array(mfi, 0),
              ldataOld_p->state.array(mfi, 0), flag,
              AMREX_D_DECL(apx, apy, apz), vfrac, AMREX_D_DECL(fcx, fcy, fcz),
              ccc, bcRec_d.dataPtr(), geom[lev], m_adv_redist_type);
          }
        }
      }
    }
  }

  // Initialize covered state
  setCoveredState(AmrNewTime);
}

void
PeleLM::getEBDistance(const int a_lev, amrex::MultiFab& a_signDistLev)
{

  BL_PROFILE("PeleLMeX::getEBDistance()");

  if (a_lev == 0) {
    amrex::MultiFab::Copy(a_signDistLev, *m_signedDist0, 0, 0, 1, 0);
    return;
  }

  // A pair of MF to hold crse & fine dist data
  amrex::Array<std::unique_ptr<amrex::MultiFab>, 2> MFpair;

  // Interpolate on successive levels up to a_lev
  for (int lev = 1; lev <= a_lev; ++lev) {

    // Use MF EB interp
    auto& interpolater = amrex::eb_mf_lincc_interp;

    // Get signDist on coarsen fineBA
    amrex::BoxArray coarsenBA(grids[lev].size());
    for (int j = 0, N = static_cast<int>(coarsenBA.size()); j < N; ++j) {
      coarsenBA.set(
        j, interpolater.CoarseBox(grids[lev][j], refRatio(lev - 1)));
    }
    amrex::MultiFab coarsenSignDist(coarsenBA, dmap[lev], 1, 0);
    coarsenSignDist.setVal(0.0);
    amrex::MultiFab* crseSignDist =
      (lev == 1) ? m_signedDist0.get() : MFpair[0].get();
    coarsenSignDist.ParallelCopy(*crseSignDist, 0, 0, 1);

    // Interpolate on current lev
    amrex::MultiFab* currentSignDist;
    if (lev < a_lev) {
      MFpair[1] = std::make_unique<amrex::MultiFab>(
        grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), EBFactory(lev));
    }
    currentSignDist = (lev == a_lev) ? &a_signDistLev : MFpair[1].get();

    interpolater.interp(
      coarsenSignDist, 0, *currentSignDist, 0, 1, amrex::IntVect(0),
      Geom(lev - 1), Geom(lev), Geom(lev).Domain(), refRatio(lev - 1),
      {m_bcrec_force}, 0);

    // Swap MFpair
    if (lev < a_lev) {
      swap(MFpair[0], MFpair[1]);
    }
  }
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getEBState(
  const int first_comp, const int ncomp, const TimeStamp a_time)
{
  AMREX_ASSERT(first_comp >= VELX);
  AMREX_ASSERT(first_comp + ncomp <= NVAR);
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> r;
  r.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    r.push_back(
      std::make_unique<amrex::MultiFab>(
        grids[lev], dmap[lev], ncomp, m_nGrowState, amrex::MFInfo(),
        Factory(lev)));
    getEBState(lev, a_time, *r[lev], first_comp, ncomp);
  }
  return r;
}

std::unique_ptr<amrex::MultiFab>
PeleLM::getEBState(
  const int a_lev,
  const int first_comp,
  const int ncomp,
  const TimeStamp a_time)
{
  AMREX_ASSERT(first_comp >= VELX);
  AMREX_ASSERT(first_comp + ncomp <= NVAR);
  std::unique_ptr<amrex::MultiFab> r = std::make_unique<amrex::MultiFab>(
    grids[a_lev], dmap[a_lev], ncomp, m_nGrowState, amrex::MFInfo(),
    Factory(a_lev));
  getEBState(a_lev, a_time, *r, first_comp, ncomp);

  return r;
}

amrex::FArrayBox
PeleLM::getEBState(
  amrex::MFIter const& a_mfi,
  const int a_lev,
  const int first_comp,
  const int ncomp,
  const TimeStamp a_time)
{
  AMREX_ASSERT(first_comp >= VELX);
  AMREX_ASSERT(first_comp + ncomp <= NVAR);

  ProbParm const* lprobparm = prob_parm_d;
  const auto geomdata = geom[a_lev].data();
  auto time = getTime(a_lev, a_time);

  auto* ldata_p = getLevelDataPtr(a_lev, a_time);

  const auto bx = a_mfi.growntilebox(m_nGrowState);
  amrex::FArrayBox r(bx, ncomp, amrex::The_Async_Arena());
  auto ebscal_arr = r.array();
  const auto& ebfact = EBFactory(a_lev);
  auto const& flagfab = ebfact.getMultiEBCellFlagFab()[a_mfi];
  if (flagfab.getType(bx) == amrex::FabType::singlevalued) {
    const PeleLMFillBCStateEB<
      ProblemSpecificFunctions,
      hasBCNormalEB<const ProblemSpecificFunctions>::value>
      EBfiller{lprobparm, ProblemSpecificFunctions{}};
    auto const& flag = flagfab.const_array();
    const auto& state = ldata_p->state.const_array(a_mfi);
    const auto& ebbc = ebfact.getBndryCent().const_array(a_mfi);
    const auto& ebnorm = ebfact.getBndryNormal().const_array(a_mfi);
    amrex::ParallelFor(
      bx,
      [flag, ebscal_arr, state, ebnorm, ncomp, EBfiller, first_comp, geomdata,
       time, ebbc] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        // Regular/covered cells -> 0.0
        if (flag(i, j, k).isCovered() || flag(i, j, k).isRegular()) {
          for (int n = 0; n < ncomp; ++n) {
            ebscal_arr(i, j, k, n) = 0.0;
          }
        } else { // cut-cells
          EBfiller(
            i, j, k, state, ebscal_arr, first_comp, ncomp, ebbc, ebnorm,
            geomdata, time);
        }
      });
  } else {
    amrex::ParallelFor(
      bx, ncomp,
      [ebscal_arr] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
        ebscal_arr(i, j, k, n) = 0.0;
      });
  }

  return r;
}

void
PeleLM::getEBState(
  const int a_lev,
  const TimeStamp a_time,
  amrex::MultiFab& a_EBstate,
  const int stateComp,
  const int nComp)
{
  AMREX_ASSERT(a_EBstate.nComp() >= nComp);

  // Get Geom / EB data
  ProbParm const* lprobparm = prob_parm_d;
  const auto geomdata = geom[a_lev].data();
  const auto& ebfact = EBFactory(a_lev);
  const amrex::MultiCutFab& bndryCentroid = ebfact.getBndryCent();
  auto time = getTime(a_lev, a_time);
  auto* ldata_p = getLevelDataPtr(a_lev, a_time);

  amrex::MFItInfo mfi_info;
  if (amrex::Gpu::notInLaunchRegion()) {
    mfi_info.EnableTiling().SetDynamic(true);
  }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(a_EBstate, mfi_info); mfi.isValid(); ++mfi) {
    const amrex::Box& bx = mfi.growntilebox();
    auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
    auto const& flag = flagfab.const_array();
    const auto& ebState = a_EBstate.array(mfi);

    if (flagfab.getType(bx) == amrex::FabType::covered) { // Set to zero
      AMREX_PARALLEL_FOR_4D(
        bx, nComp, i, j, k, n, { ebState(i, j, k, n) = 0.0; });
    } else if (flagfab.getType(bx) == amrex::FabType::regular) { // Set to zero
      AMREX_PARALLEL_FOR_4D(
        bx, nComp, i, j, k, n, { ebState(i, j, k, n) = 0.0; });
    } else {
      const PeleLMFillBCStateEB<
        ProblemSpecificFunctions,
        hasBCNormalEB<const ProblemSpecificFunctions>::value>
        EBfiller{lprobparm, ProblemSpecificFunctions{}};
      const auto& state = ldata_p->state.const_array(mfi);
      const auto& ebbc = bndryCentroid.const_array(mfi);
      const auto& ebnorm = ebfact.getBndryNormal().const_array(mfi);
      amrex::ParallelFor(
        bx, [flag, ebState, state, EBfiller, stateComp, nComp, ebnorm, geomdata,
             time, ebbc] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          // Regular/covered cells -> 0.0
          if (flag(i, j, k).isCovered() || flag(i, j, k).isRegular()) {
            for (int n = 0; n < nComp; ++n) {
              ebState(i, j, k, n) = 0.0;
            }
          } else { // cut-cells
            EBfiller(
              i, j, k, state, ebState, stateComp, nComp, ebbc, ebnorm, geomdata,
              time);
          }
        });
    }
  }
}

void
PeleLM::getEBDiff(
  const int a_lev,
  const TimeStamp a_time,
  amrex::MultiFab& a_EBDiff,
  const int diffComp)
{
  // Get Geom / EB data
  ProbParm const* lprobparm = prob_parm_d;
  const auto geomdata = geom[a_lev].data();
  const auto& ebfact = EBFactory(a_lev);
  const amrex::MultiCutFab& bndryCentroid = ebfact.getBndryCent();

  // Get diffusivity cell-centered
  auto* ldata_p = getLevelDataPtr(a_lev, a_time);

  amrex::MFItInfo mfi_info;
  if (amrex::Gpu::notInLaunchRegion()) {
    mfi_info.EnableTiling().SetDynamic(true);
  }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(a_EBDiff, mfi_info); mfi.isValid(); ++mfi) {
    const amrex::Box& bx = mfi.tilebox();
    auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
    auto const& flag = flagfab.const_array();
    const auto& ebdiff = a_EBDiff.array(mfi);
    const auto& diff_cc = ldata_p->diff_cc.const_array(mfi, diffComp);

    if (flagfab.getType(bx) == amrex::FabType::covered) { // Set to zero
      AMREX_PARALLEL_FOR_3D(bx, i, j, k, { ebdiff(i, j, k) = 0.0; });
    } else if (flagfab.getType(bx) == amrex::FabType::regular) { // Set to zero
      AMREX_PARALLEL_FOR_3D(bx, i, j, k, { ebdiff(i, j, k) = 0.0; });
    } else {
      const PeleLMFillBCTypeEB<
        ProblemSpecificFunctions,
        hasBCTypeEB<const ProblemSpecificFunctions>::value>
        EBTypfiller{lprobparm, ProblemSpecificFunctions{}};
      const auto& ebbc = bndryCentroid.const_array(mfi);
      amrex::ParallelFor(
        bx, [flag, ebdiff, diff_cc, EBTypfiller, geomdata, lprobparm,
             ebbc] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          // Regular/covered cells -> 0.0
          if (flag(i, j, k).isCovered() || flag(i, j, k).isRegular()) {
            ebdiff(i, j, k) = 0.0;
          } else { // cut-cells
            int ebflagtype = pelelmex::BCTypeEB::wall_adiab;
            amrex::Real ebfacefrac = 0.0;
            EBTypfiller(
              i, j, k, ebflagtype, ebfacefrac, ebbc, geomdata, *lprobparm);
            // TODO: this only works for temperature at this point
            if (ebflagtype == pelelmex::BCTypeEB::wall_adiab) {
              ebdiff(i, j, k) = 0.0;
            } else {
              ebdiff(i, j, k) = ebfacefrac * diff_cc(i, j, k);
            }
          }
        });
    }
  }
}

void
PeleLM::correct_vel_small_cells(
  amrex::Vector<amrex::MultiFab*> const& a_vel,
  amrex::Vector<amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM>> const&
    a_umac)
{
  BL_PROFILE("PeleLMeX::correct_vel_small_cells");

  for (int lev = 0; lev <= finest_level; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*a_vel[lev], amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      // Tilebox
      const amrex::Box bx = mfi.tilebox();

      amrex::EBCellFlagFab const& flags =
        EBFactory(lev).getMultiEBCellFlagFab()[mfi];

      // Face-centered velocity components
      AMREX_D_TERM(
        const auto& umac_fab = (a_umac[lev][0])->const_array(mfi);
        , const auto& vmac_fab = (a_umac[lev][1])->const_array(mfi);
        , const auto& wmac_fab = (a_umac[lev][2])->const_array(mfi););

      // No cut cells in this FAB
      if (
        (flags.getType(amrex::grow(bx, 0)) == amrex::FabType::covered) ||
        (flags.getType(amrex::grow(bx, 1)) == amrex::FabType::regular)) {
        // do nothing
      } else { // Cut cells in this FAB
        // Face-centered areas
        AMREX_D_TERM(
          const auto& apx_fab =
            EBFactory(lev).getAreaFrac()[0]->const_array(mfi);
          , const auto& apy_fab =
              EBFactory(lev).getAreaFrac()[1]->const_array(mfi);
          , const auto& apz_fab =
              EBFactory(lev).getAreaFrac()[2]->const_array(mfi););

        const auto& vfrac_fab = EBFactory(lev).getVolFrac().const_array(mfi);

        const auto& ccvel_fab = a_vel[lev]->array(mfi);

        // This FAB has cut cells -- we define the centroid value in terms of
        // the MAC velocities onfaces
        amrex::ParallelFor(
          bx, [vfrac_fab, ccvel_fab, umac_fab, apx_fab, vmac_fab, apy_fab
#if (AMREX_SPACEDIM == 3)
               ,
               wmac_fab, apz_fab
#endif
        ] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (vfrac_fab(i, j, k) > 0.0 && vfrac_fab(i, j, k) < 5.e-3) {
              AMREX_D_TERM(
                const amrex::Real u_avg =
                  (apx_fab(i, j, k) * umac_fab(i, j, k) +
                   apx_fab(i + 1, j, k) * umac_fab(i + 1, j, k)) /
                  (apx_fab(i, j, k) + apx_fab(i + 1, j, k));
                , const amrex::Real v_avg =
                    (apy_fab(i, j, k) * vmac_fab(i, j, k) +
                     apy_fab(i, j + 1, k) * vmac_fab(i, j + 1, k)) /
                    (apy_fab(i, j, k) + apy_fab(i, j + 1, k));
                , const amrex::Real w_avg =
                    (apz_fab(i, j, k) * wmac_fab(i, j, k) +
                     apz_fab(i, j, k + 1) * wmac_fab(i, j, k + 1)) /
                    (apz_fab(i, j, k) + apz_fab(i, j, k + 1)););

              AMREX_D_TERM(ccvel_fab(i, j, k, 0) = u_avg;
                           , ccvel_fab(i, j, k, 1) = v_avg;
                           , ccvel_fab(i, j, k, 2) = w_avg;);
            }
          });
      }
    }
  }
}

int
PeleLM::getRestartEBMaxLevel() const
{
  if (m_restart_chkfile.empty()) {
    return -1;
  }
  // Go and parse the line we need
  std::string File(m_restart_chkfile + "/Header");

  amrex::VisMF::IO_Buffer io_buffer(amrex::VisMF::GetIOBufferSize());

  amrex::Vector<char> fileCharPtr;
  amrex::ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
  std::string fileCharPtrString(fileCharPtr.dataPtr());
  std::istringstream is(fileCharPtrString, std::istringstream::in);

  std::string line;

  // Title line
  std::getline(is, line);

  // Finest level
  std::getline(is, line);

  // Step count
  std::getline(is, line);

  // Either what we're looking for or m_cur_time
  std::getline(is, line);

  if (line.find('.') != std::string::npos) {
    return -1;
  }
  int max_eb_rst = -1;
  max_eb_rst = std::stoi(line);
  return max_eb_rst;
}

void
PeleLM::checkEBInflowFunctions()
{
  if (!hasBCNormalEB<const ProblemSpecificFunctions>::value) {
    amrex::Abort(
      "Provided ProblemSpecificFunctions doesn't have a viable bcnormal_eb "
      "function");
  }
  if (!hasBCTypeEB<const ProblemSpecificFunctions>::value) {
    amrex::Abort(
      "Provided ProblemSpecificFunctions doesn't have a viable bctype_eb "
      "function");
  }
  if (m_verbose != 0 && m_useEBinflow != 0) {
    amrex::Print()
      << "WARNING: EB-inflow capability is experimental. Scalar "
         "diffusion is not supported at these boundaries and future "
         "interface changes are possible!\n";
  }
}
#endif
