#include <PeleLMeX.H>
#include <memory>
#include <pelelmex_prob.H>
#ifdef AMREX_USE_EB
#include <AMReX_EB_utils.H>
#endif

void
PeleLM::Init()
{
  BL_PROFILE("PeleLMeX::Init()");

  // Open temporals file
  openTempFile();

  // Check run parameters
  checkRunParams();

  // Initialize data
  initData();
}

void
PeleLM::MakeNewLevelFromScratch(
  const int lev,
  const amrex::Real time,
  const amrex::BoxArray& ba,
  const amrex::DistributionMapping& dm)
{
  BL_PROFILE("PeleLMeX::MakeNewLevelFromScratch()");

  if (m_verbose > 0) {
    amrex::Print() << " Making new level " << lev << " from scratch \n";
    if (m_verbose > 1 && lev > 0) {
      auto const dx = geom[lev].CellSizeArray();
      const amrex::Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
      amrex::Print() << " with " << ba.numPts() << " cells," << ba.size()
                     << " boxes,"
                     << " over "
                     << static_cast<amrex::Real>(ba.numPts()) * vol /
                          geom[0].ProbSize() * 100
                     << "% of the domain \n";
    }
    if (m_verbose > 3 && lev > 0) {
      amrex::Print() << " with BoxArray " << ba << "\n";
    }
  }

  // Pass Box and Dmap to AmrCore
  SetBoxArray(lev, ba);
  SetDistributionMap(lev, dm);

  // Define the FAB Factory
#ifdef AMREX_USE_EB
  m_factory[lev] = makeEBFabFactory(
    geom[lev], grids[lev], dmap[lev], {6, 6, 6}, amrex::EBSupport::full);
#else
  m_factory[lev] = std::make_unique<amrex::FArrayBoxFactory>();
#endif

  // Initialize the LevelData
  m_leveldata_old[lev] = std::make_unique<LevelData>(
    grids[lev], dmap[lev], *m_factory[lev], m_incompressible, m_has_divu,
    m_nAux, m_nGrowState, m_use_soret, static_cast<int>(m_do_les));
  m_leveldata_new[lev] = std::make_unique<LevelData>(
    grids[lev], dmap[lev], *m_factory[lev], m_incompressible, m_has_divu,
    m_nAux, m_nGrowState, m_use_soret, static_cast<int>(m_do_les));

  if (max_level > 0 && lev != max_level) {
    m_coveredMask[lev] =
      std::make_unique<amrex::iMultiFab>(grids[lev], dmap[lev], 1, 0);
  }
  m_resetCoveredMask = 1;

  if (m_do_react != 0) {
    m_leveldatareact[lev] =
      std::make_unique<LevelDataReact>(grids[lev], dmap[lev], *m_factory[lev]);
    m_leveldatareact[lev]->functC.setVal(0.0);
    // Make sure this is initialized - may get used for ErrorEst when setting
    // up levels but before reactions are actually computed
    m_leveldatareact[lev]->I_R.setVal(0.0);
  }

#ifdef PELE_USE_PLASMA
  m_leveldatanlsolve[lev].reset(
    new LevelDataNLSolve(grids[lev], dmap[lev], *m_factory[lev], m_nGrowState));
  if (m_do_extraEFdiags) {
    m_ionsFluxes[lev].reset(
      new amrex::MultiFab(grids[lev], dmap[lev], NUM_IONS * AMREX_SPACEDIM, 0));
  }
#endif

  // Fill the initial solution (if not restarting)
  if (m_restart_chkfile.empty()) {
    if (m_restart_pltfile.empty()) {
      initLevelData(lev);
    } else {
      initLevelDataFromPlt(lev, m_restart_pltfile);
    }
    if (!m_velocity_plotfile.empty()) {
      // Add turbulent velocity from an existing plotfile
      addLevelVelocityDataFromPlt(lev, m_velocity_plotfile);
    }
  }

  // Times
  m_t_new[lev] = time;
  m_t_old[lev] = time - 1.0e200;

  // Load balance
  m_costs[lev] = std::make_unique<amrex::LayoutData<amrex::Real>>(ba, dm);

  // Mac projector
#ifdef AMREX_USE_EB
  macproj = std::make_unique<Hydro::MacProjector>(
    Geom(0, finest_level),
    amrex::MLMG::Location::FaceCentroid, // Location of mac velocity
    amrex::MLMG::Location::FaceCentroid, // Location of beta
    amrex::MLMG::Location::CellCenter);  // Location of solution variable phi
#else
  macproj = std::make_unique<Hydro::MacProjector>(Geom(0, finest_level));
#endif
  m_macProjOldSize = finest_level + 1;
  m_extSource[lev] = std::make_unique<amrex::MultiFab>(
    grids[lev], dmap[lev], NVAR, amrex::max<int>(m_nGrowAdv, m_nGrowMAC),
    amrex::MFInfo(), *m_factory[lev]);
  m_extSource[lev]->setVal(0.);

  // Mesh mapping metric fields (if enabled; otherwise no-op).  Allocate
  // storage here so downstream consumers can index the per-level MFs even
  // though create_map() is not yet called from the numerics paths.
  if (m_mesh_map) {
    m_mesh_map->define(
      lev, grids[lev], dmap[lev], *m_factory[lev], m_nGrowState);
    m_mesh_map->create_map(lev, geom[lev]);
    // Refresh the cached device-callable evaluator so user
    // initdata_mapped / bcnormal code sees the active mapping.
    // (Idempotent across levels: the evaluator parameters do not
    // depend on the level.)
    m_map_eval = m_mesh_map->make_evaluator();
    rebuildMappedInterps();
  }
  defineCellVolume(lev, grids[lev], dmap[lev], *m_factory[lev]);

#ifdef AMREX_USE_EB
  if (lev == 0 && (m_signDistNeeded != 0)) {
    // Set up CC signed distance container to control EB refinement
    m_signedDist0 = std::make_unique<amrex::MultiFab>(
      grids[lev], dmap[lev], 1, 1, amrex::MFInfo(), *m_factory[lev]);

    // Estimate the maximum distance we need in terms of level 0 dx:
    auto extentFactor = static_cast<amrex::Real>(nErrorBuf(0));
    for (int ilev = 1; ilev <= max_level; ++ilev) {
      extentFactor += static_cast<amrex::Real>(nErrorBuf(ilev)) /
                      std::pow(
                        static_cast<amrex::Real>(refRatio(ilev - 1)[0]),
                        static_cast<amrex::Real>(ilev));
    }
    // Account for diagonals
    constexpr amrex::Real sqrt2 = 1.4142135623730951;
    extentFactor *= sqrt2 * m_derefineEBBuffer;

    amrex::MultiFab signDist(
      convert(grids[0], amrex::IntVect::TheUnitVector()), dmap[0], 1, 1,
      amrex::MFInfo(), EBFactory(0));
    FillSignedDistance(signDist, true);

    auto const& sd_cc_ma = m_signedDist0->arrays();
    auto const& sd_nd_ma = signDist.const_arrays();
    amrex::ParallelFor(
      *m_signedDist0, m_signedDist0->nGrowVect(),
      [sd_cc_ma,
       sd_nd_ma] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        sd_cc_ma[box_no](i, j, k) = AMREX_D_TERM(
          sd_nd_ma[box_no](i, j, k) + sd_nd_ma[box_no](i + 1, j, k),
          +sd_nd_ma[box_no](i, j + 1, k) + sd_nd_ma[box_no](i + 1, j + 1, k),
          +sd_nd_ma[box_no](i, j, k + 1) + sd_nd_ma[box_no](i + 1, j, k + 1) +
            sd_nd_ma[box_no](i, j + 1, k + 1) +
            sd_nd_ma[box_no](i + 1, j + 1, k + 1));
        sd_cc_ma[box_no](i, j, k) *= AMREX_D_PICK(0.5, 0.25, 0.125);
      });
    amrex::Gpu::streamSynchronize();
    m_signedDist0->FillBoundary(geom[0].periodicity());
    extendSignedDistance(m_signedDist0.get(), extentFactor);
  }
#endif
}

void
PeleLM::initData()
{
  BL_PROFILE("PeleLMeX::initData()");

  if (m_restart_chkfile.empty()) {
    // Pre-create PltFileManagers so all AMR levels share a single instance
    // rather than each level instantiating its own (which causes repeated large
    // memory allocations for big plotfiles).
    if (!m_restart_pltfile.empty()) {
      m_pltDataManager =
        std::make_unique<pele::physics::pltfilemanager::PltFileManager>(
          m_restart_pltfile);
    }
    if (!m_velocity_plotfile.empty()) {
      m_velPltDataManager =
        std::make_unique<pele::physics::pltfilemanager::PltFileManager>(
          m_velocity_plotfile);
    }

    //----------------------------------------------------------------
    if (!m_initial_grid_file.empty()) {
      InitFromGridFile(m_cur_time);
    } else {
      // This is an AmrCore member function which recursively makes new levels
      // with MakeNewLevelFromScratch.
      InitFromScratch(m_cur_time);
    }

    // Release PltFileManagers now that all levels have been initialized.
    m_pltDataManager.reset();
    m_velPltDataManager.reset();

    resetCoveredMask();
    updateDiagnostics();

#ifdef PELE_USE_SPRAY
    SprayInit();
#endif
#ifdef PELE_USE_RADIATION
    if (do_rad_solve) {
      RadInit();
    }
#endif

    //----------------------------------------------------------------
    // Set typical values
    constexpr int is_init = 1;
    setTypicalValues(AmrNewTime, is_init);

    // initialize temporals
    initTemporals(AmrNewTime);

#ifdef AMREX_USE_EB
    //----------------------------------------------------------------
    // Initial redistribution
    initCoveredState();
    initialRedistribution();
#endif

    //----------------------------------------------------------------
    // AverageDown and FillPatch the NewState
    averageDownState(AmrNewTime);
    fillPatchState(AmrNewTime);

    if (m_nAux > 0) {
      averageDownAux(AmrNewTime);
      fillPatchAux(AmrNewTime);
    }

    //----------------------------------------------------------------
    // Initialize turbulent viscosity for LES
    if (m_do_les) {
      calcTurbViscosity(AmrNewTime);
    }

    //----------------------------------------------------------------
    // Plot state before initial iterations
    if (m_plot_init_state) {
      WritePlotFile();
    }

    //----------------------------------------------------------------
    // If performing UnitTest, let's stop here
    if (runMode() != "normal") {
      return;
    }

    //----------------------------------------------------------------
    // Full-mode recycling injects the sampled velocity itself, so the slab
    // can (and should) be seeded from the initial solution before the init
    // projections and iterations consume inflow BCs. Fluctuations mode
    // seeds at the end of the first step once a mean can accumulate.
    if (
      m_use_inlet_from_plane != 0 && m_recycling_mode == RecyclingMode::Full &&
      !m_inlet_recycling.initialized) {
      updateRecyclingPlaneSnapshot();
    }

    //----------------------------------------------------------------
    // Project initial solution
    projectInitSolution();

    // Active control
    constexpr int is_restart = 0;
    activeControl(is_restart);

    //----------------------------------------------------------------
    // Do initial pressure iterations
    initialIterations();

    if (m_do_reset_time != 0) {
      m_nstep = 0;
    }

    if (m_do_temporals != 0) {
      writeTemporals();
    }

    if (m_plot_int > 0 || m_plot_per_approx > 0. || m_plot_per_exact > 0.) {
      WritePlotFile();
    }
    if (m_check_int > 0 || m_check_per > 0.) {
      WriteCheckPointFile();
    }

    amrex::Print() << PrettyLine;

    // Diagnostics
    doDiagnostics();

  } else {
    //----------------------------------------------------------------
    // Read starting configuration from chk file.
    ReadCheckPointFile();

#ifdef PELE_USE_SPRAY
    SprayInit();
#endif
#ifdef PELE_USE_RADIATION
    if (do_rad_solve) {
      RadInit();
    }
#endif
#ifdef PELE_USE_PLASMA
    // If restarting from a non plasma simulation
    if (m_restart_nonEF) {
      // either pass Y_ne -> nE or initialize nE for electro-neutral
      if (m_restart_electroneutral) {
        initializeElectronNeutral();
      } else {
        initializeElectronFromMassFraction();
      }

      // do an initial Poisson solve
      fillPatchPhiV(AmrNewTime);
      poissonSolveEF(AmrNewTime);

      // Reset time data
      if (m_restart_resetTime) {
        m_nstep = 0;
        m_cur_time = 0.0;
        for (int lev = 0; lev <= finest_level; ++lev) {
          m_t_new[lev] = 0.0;
          m_t_old[lev] = -1.0e200;
        }
        m_dt = -1.0;
        constexpr int is_init = 1;
        const amrex::Real dtInit = computeDt(is_init, AmrNewTime);
        amrex::Print() << " Initial dt: " << dtInit << "\n";
      }

      // Let's write the initial condition
      if (m_plot_int > 0) {
        WritePlotFile();
      }
    }
#endif

    // Regrid after restart if requested
    if (m_regrid_on_restart != 0) {
      amrex::Print() << " Regriding on restart \n";
      for (int lev{finest_level}; lev < max_level; ++lev) {
        regrid(0, m_cur_time);
        // Need to fill the old state to enable regrid on higher levels
        copyStateNewToOld(1);
        copyPressNewToOld();
        if (m_do_react != 0) {
          auto* ldataR_p = getLevelDataReactPtr(lev);
          ldataR_p->I_R.setVal(0.0);
        }
      }
    }

    // Generate the covered cell mask
    m_resetCoveredMask = 1;
    resetCoveredMask();
    updateDiagnostics();

    // Full-mode recycling carries no checkpointed state; seed the slab
    // from the restored solution so the first step's BC fills see the
    // recycled velocity (see the matching call in the fresh-start path).
    if (
      m_use_inlet_from_plane != 0 && m_recycling_mode == RecyclingMode::Full &&
      !m_inlet_recycling.initialized) {
      updateRecyclingPlaneSnapshot();
    }

    // Active control
    constexpr int is_restart = 1;
    activeControl(is_restart);
  }
}

void
PeleLM::initLevelData(const int lev)
{
  BL_PROFILE("PeleLMeX::initLevelData()");

  // Get level data
  auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
  const auto geomdata = geom[lev].data();

  // Pressure and pressure gradients to zero
  ldata_p->press.setVal(0.0);
  ldata_p->gp.setVal(0.0);

  // Prob/PMF data
  ProbParm const* lprobparm = prob_parm_d;
  auto const* lpmfdata = pmf_data.device_parm();

  // don't want to use state for dummy in case user overwrites state in aux
  amrex::MultiFab dummy_mf(grids[lev], dmap[lev], 0, 0);

  auto const& state_ma = ldata_p->state.arrays();
  auto const& aux_ma =
    (m_nAux > 0) ? ldata_p->auxiliaries.arrays() : dummy_mf.arrays();

  // Capture the cached mesh-mapping evaluator by value so user code in
  // ProblemSpecificFunctions::initdata_mapped (if provided) can query
  // physical-space coordinates via e.g. `mmap.x_phys_cc(0, i, geomdata)`.
  // For unmapped runs `mmap` is Kind::Identity and the helpers return
  // the unmapped xi-space coordinate.  Existing problem setups that
  // only define the legacy `initdata` get dispatched there inside
  // call_initdata (a templated wrapper in
  // PeleLMeX_ProblemSpecificFunctions.H); their lambda capture of
  // `mmap` is free (a few bytes, no MultiFab references).
  const MeshMapEvaluator mmap = m_map_eval;
  amrex::ParallelFor(
    ldata_p->state,
    [state_ma, aux_ma, geomdata, lprobparm, lpmfdata, mmap,
     is_incomp =
       m_incompressible] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      call_initdata<ProblemSpecificFunctions>(
        i, j, k, is_incomp, state_ma[box_no], aux_ma[box_no], geomdata, mmap,
        *lprobparm, lpmfdata);
    });
  amrex::Gpu::streamSynchronize();

  if (m_incompressible == 0) {
    // Initialize thermodynamic pressure
    setThermoPress(lev, AmrNewTime);
    if (m_has_divu != 0) {
      ldata_p->divu.setVal(0.0);
    }
  }
}

void
PeleLM::projectInitSolution()
{
  constexpr int is_init = 1;

#ifdef PELE_USE_PLASMA
  poissonSolveEF(AmrNewTime);
  fillPatchPhiV(AmrNewTime);
#endif

  // Post data Init time step estimate
  const amrex::Real dtInit = computeDt(is_init, AmrNewTime);
  amrex::Print() << " Initial dt: " << dtInit << "\n";

  if (m_do_init_proj != 0) {

    amrex::Print() << "\n Doing initial projection(s) \n\n";
    // Subcycling IAMR/PeleLM first does a projection with no reaction divU
    // which can make the dt for evaluating I_R better
    if (m_has_divu != 0) {
      constexpr int is_initialization = 1;    // Yes we are
      constexpr int computeDiffusionTerm = 1; // Needed here
      constexpr int do_avgDown = 1;           // Always

      // Light version of the diffusion data container
      std::unique_ptr<AdvanceDiffData> diffData;
      diffData = std::make_unique<AdvanceDiffData>(
        finest_level, grids, dmap, m_factory, m_nGrowAdv, m_use_wbar,
        m_use_soret, m_nAux, is_initialization);
      calcDivU(
        is_initialization, computeDiffusionTerm, do_avgDown, AmrNewTime,
        diffData);
    }
    initialProjection();

    // If gravity is used, do initial pressure projection to get the hydrostatic
    // pressure
    if (std::abs(m_gravity.sum()) > 0.0) {
      initialPressProjection();
    }

    // Post data init time step estimate
    m_dt = computeDt(is_init, AmrNewTime);
    amrex::Print() << " Initial dt: " << m_dt << "\n";

    //----------------------------------------------------------------
    // Initial velocity projection iterations
    for (int iter = 0; iter < m_numDivuIter; ++iter) {
      if (m_do_react != 0) {
        // The new level data has been filled above
        // Copy new -> old since old used in advanceChemistry
        copyStateNewToOld();

        for (int lev = finest_level; lev >= 0; --lev) {
          // Setup empty forcing
          amrex::MultiFab Forcing(grids[lev], dmap[lev], nCompForcing(), 0);
          Forcing.setVal(0.0);

          if (lev != finest_level) {
            advanceChemistryBAChem(lev, dtInit / 2.0, Forcing);
          } else {
            if (m_max_grid_size_chem.min() > 0) {
              advanceChemistryBAChem(lev, dtInit / 2.0, Forcing);
            } else {
              advanceChemistry(lev, dtInit / 2.0, Forcing);
            }
          }
          if (m_doLoadBalance != 0) {
            loadBalanceChemLev(lev);
          }
        }
        // Copy back old -> new
        copyStateOldToNew();
      }
      if (m_has_divu != 0) {
        constexpr int is_initialization = 1;    // Yes we are
        constexpr int computeDiffusionTerm = 1; // Needed here
        constexpr int do_avgDown = 1;           // Always

        // Light version of the diffusion data container
        std::unique_ptr<AdvanceDiffData> diffData;
        diffData = std::make_unique<AdvanceDiffData>(
          finest_level, grids, dmap, m_factory, m_nGrowAdv, m_use_wbar,
          m_use_soret, m_nAux, is_initialization);
        calcDivU(
          is_initialization, computeDiffusionTerm, do_avgDown, AmrNewTime,
          diffData);
      }
      initialProjection();
    }

    if (m_numDivuIter == 0 && (m_do_react != 0)) {
      for (int lev = 0; lev <= finest_level; ++lev) {
        auto* ldataR_p = getLevelDataReactPtr(lev);
        ldataR_p->I_R.setVal(0.0);
      }
    }
    amrex::Print() << PrettyLine;
  } else {
    // If we didn't do the projection, initialize press/gp(/I_R)
    for (int lev = 0; lev <= finest_level; ++lev) {
      if (m_do_react != 0) {
        auto* ldataR_p = getLevelDataReactPtr(lev);
        ldataR_p->I_R.setVal(0.0);
      }
      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
      ldata_p->press.setVal(0.0);
      ldata_p->gp.setVal(0.0);
    }
  }
}

void
PeleLM::initialIterations()
{
  BL_PROFILE("PeleLMeX::initialIterations()");

  if (m_verbose > 0 && m_init_iter > 0) {
    amrex::Print() << "\n Doing initial pressure iteration(s) \n";
  }

  for (int lev = 0; lev <= finest_level; ++lev) {
    m_t_old[lev] = m_t_new[lev];
  }

  //----------------------------------------------------------------
  // Initial pressure iterations
  for (int iter = 0; iter < m_init_iter; ++iter) {

    if (m_verbose > 0) {
      amrex::Print() << "\n ================   INITIAL ITERATION [" << iter
                     << "]   ================ \n";
    }
    constexpr int is_init = 1;
    Advance(is_init);

    // Pass new pressure and gp from New to Old
    copyPressNewToOld();

    // Copy back old state
    copyStateOldToNew();
  }
}

void
PeleLM::InitFromGridFile(const amrex::Real time)
{
  {
    const amrex::BoxArray& ba = MakeBaseGrids();
    const amrex::DistributionMapping dm(ba);
    MakeNewLevelFromScratch(0, time, ba, dm);
  }
  finest_level = static_cast<int>(m_initial_ba.size());
  for (int lev = 1; lev <= finest_level; ++lev) {
    const amrex::BoxArray ba = m_initial_ba[lev - 1];
    amrex::DistributionMapping dm(ba);
    MakeNewLevelFromScratch(lev, time, ba, dm);
  }
}

void
PeleLM::checkRunParams()
{
#ifdef AMREX_USE_EB
  if (geom[0].IsRZ()) {
    amrex::Abort("RZ geometry is not available with EB");
  }
#endif

#if (AMREX_SPACEDIM == 2)
  if (geom[0].IsRZ() && m_phys_bc.lo(0) != 3) {
    amrex::Abort("x-low must be 'Symmetry' when using RZ coordinate system");
  }
#endif
}
