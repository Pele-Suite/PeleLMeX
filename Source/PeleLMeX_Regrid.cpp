#include <PeleLMeX.H>
#include <memory>

void
PeleLM::regrid(const int lbase, const amrex::Real time, const bool initial)
{
  BL_PROFILE("PeleLMeX::regrid()");

  if ((m_doLoadBalance == 0) && lbase >= max_level) {
    return;
  }

  if (!m_regrid_file.empty()) {
    regridFromGridFile(lbase, time, initial);

  } else {

    // Load balance base grid
    if (lbase == 0 && (m_doLoadBalance != 0) && !initial) {

      if (m_verbose > 0) {
        amrex::Print() << " Load balancing level 0 \n";
      }

      int remakeLevel = 0;

      computeCosts(0);

      // Use efficiency: average MPI rank cost / max cost
      amrex::Real currentEfficiency = 0.0;
      amrex::Real testEfficiency = 0.0;

      amrex::DistributionMapping test_dmap;
      // Build the test dmap, w/o braodcasting
      if (m_loadBalanceMethod == LoadBalanceMethod::SFC) {

        test_dmap = amrex::DistributionMapping::makeSFC(
          *m_costs[0], currentEfficiency, testEfficiency, false,
          amrex::ParallelDescriptor::IOProcessorNumber());

      } else if (m_loadBalanceMethod == LoadBalanceMethod::Knapsack) {

        const amrex::Real navg =
          static_cast<amrex::Real>(grids[0].size()) /
          static_cast<amrex::Real>(amrex::ParallelDescriptor::NProcs());
        const int nmax = static_cast<int>(amrex::max<amrex::Real>(
          std::round(m_loadBalanceKSfactor * navg), std::ceil(navg)));
        test_dmap = amrex::DistributionMapping::makeKnapSack(
          *m_costs[0], currentEfficiency, testEfficiency, nmax, false,
          amrex::ParallelDescriptor::IOProcessorNumber());
      }

      // IO proc determine if the test dmap offers significant improvements
      if (
        (m_loadBalanceEffRatioThreshold > 0.0) &&
        (amrex::ParallelDescriptor::MyProc() ==
         amrex::ParallelDescriptor::IOProcessorNumber())) {
        remakeLevel = static_cast<int>(
          (remakeLevel != 0) ||
          (testEfficiency >
           m_loadBalanceEffRatioThreshold * currentEfficiency));
      }
      amrex::ParallelDescriptor::Bcast(
        &remakeLevel, 1, amrex::ParallelDescriptor::IOProcessorNumber());

      if (m_verbose > 1 && (remakeLevel != 0)) {
        amrex::Print() << " Current LoadBalancing efficiency: "
                       << currentEfficiency << "\n"
                       << " Test LoadBalancing efficiency: " << testEfficiency
                       << " \n";
      }

      // Bcast the test dmap and remake level
      if (remakeLevel != 0) {
        amrex::Vector<int> pmap;
        if (
          amrex::ParallelDescriptor::MyProc() ==
          amrex::ParallelDescriptor::IOProcessorNumber()) {
          pmap = test_dmap.ProcessorMap();
        } else {
#pragma GCC diagnostic ignored "-Wnull-dereference"
          pmap.resize(static_cast<std::size_t>(grids[0].size()));
        }
        amrex::ParallelDescriptor::Bcast(
          pmap.data(), pmap.size(),
          amrex::ParallelDescriptor::IOProcessorNumber());

        if (
          amrex::ParallelDescriptor::MyProc() !=
          amrex::ParallelDescriptor::IOProcessorNumber()) {
          test_dmap = amrex::DistributionMapping(pmap);
        }

        RemakeLevel(0, time, grids[0], test_dmap);
        SetDistributionMap(0, test_dmap);
      }
    }

    if (lbase >= max_level) {
      return;
    }

    int new_finest;
    amrex::Vector<amrex::BoxArray> new_grids(finest_level + 2);
    MakeNewGrids(lbase, time, new_finest, new_grids);

    BL_ASSERT(new_finest <= finest_level + 1);

    bool coarse_ba_changed = false;
    for (int lev = lbase + 1; lev <= new_finest; ++lev) {
      if (lev <= finest_level) // an old level
      {
        bool ba_changed = (new_grids[lev] != grids[lev]);

        // If initial grid generation or not doing load balancing,
        // remake level only if the grid changed using the default DMap strategy
        if (initial || (m_doLoadBalance == 0)) {
          if (ba_changed || coarse_ba_changed) {
            amrex::BoxArray level_grids = grids[lev];
            amrex::DistributionMapping level_dmap = dmap[lev];
            if (ba_changed) {
              level_grids = new_grids[lev];
              level_dmap = amrex::DistributionMapping(level_grids);
            }
            const auto old_num_setdm = num_setdm;
            RemakeLevel(lev, time, level_grids, level_dmap);
            SetBoxArray(lev, level_grids);
            if (old_num_setdm == num_setdm) {
              SetDistributionMap(lev, level_dmap);
            }
          }

          // When doing load balancing, remake level if grid changed or a better
          // dmap can be obtained
        } else {

          int remakeLevel = 0;

          amrex::BoxArray new_ba;
          amrex::DistributionMapping new_dmap;

          // If the grid changed, let's build a new dmap
          if (ba_changed) {

            remakeLevel = 1;

            new_ba = new_grids[lev];
            new_dmap = amrex::DistributionMapping(new_ba);

            // Get the cost on a LayoutData associated with the new grid
            amrex::LayoutData<amrex::Real> new_cost(new_ba, new_dmap);
            computeCosts(lev, new_cost, m_loadBalanceCost);

            if (m_loadBalanceMethod == LoadBalanceMethod::SFC) {
              amrex::Vector<amrex::Real> costsVec(new_ba.size());
              amrex::ParallelDescriptor::GatherLayoutDataToVector(
                new_cost, costsVec,
                amrex::ParallelContext::IOProcessorNumberSub());
              amrex::ParallelDescriptor::Bcast(
                costsVec.data(), costsVec.size(),
                amrex::ParallelContext::IOProcessorNumberSub());
              amrex::Real efficiency;
              new_dmap = amrex::DistributionMapping::makeSFC(
                costsVec, new_ba, efficiency);

            } else if (m_loadBalanceMethod == LoadBalanceMethod::Knapsack) {

              const amrex::Real navg =
                static_cast<amrex::Real>(new_ba.size()) /
                static_cast<amrex::Real>(amrex::ParallelDescriptor::NProcs());
              const int nmax = static_cast<int>(amrex::max<amrex::Real>(
                std::round(m_loadBalanceKSfactor * navg), std::ceil(navg)));
              amrex::Vector<amrex::Real> costsVec(new_ba.size());
              amrex::ParallelDescriptor::GatherLayoutDataToVector(
                new_cost, costsVec,
                amrex::ParallelContext::IOProcessorNumberSub());
              amrex::ParallelDescriptor::Bcast(
                costsVec.data(), costsVec.size(),
                amrex::ParallelContext::IOProcessorNumberSub());
              amrex::Real efficiency;
              new_dmap = amrex::DistributionMapping::makeKnapSack(
                costsVec, efficiency, nmax);
            }

            // Let's see if we can get a better dmap
          } else {

            if (m_verbose > 1) {
              amrex::Print() << " Load balancing level " << lev << "\n";
            }

            // Try to build a new dmap with the same old grid
            new_ba = grids[lev];
            amrex::DistributionMapping test_dmap;

            computeCosts(lev);

            // Use efficiency: average MPI rank cost / max cost
            amrex::Real currentEfficiency = 0.0;
            amrex::Real testEfficiency = 0.0;

            // Build the test dmap, w/o braodcasting
            if (m_loadBalanceMethod == LoadBalanceMethod::SFC) {

              test_dmap = amrex::DistributionMapping::makeSFC(
                *m_costs[lev], currentEfficiency, testEfficiency, false,
                amrex::ParallelDescriptor::IOProcessorNumber());

            } else if (m_loadBalanceMethod == LoadBalanceMethod::Knapsack) {

              const amrex::Real navg =
                static_cast<amrex::Real>(new_ba.size()) /
                static_cast<amrex::Real>(amrex::ParallelDescriptor::NProcs());
              const int nmax = static_cast<int>(amrex::max<amrex::Real>(
                std::round(m_loadBalanceKSfactor * navg), std::ceil(navg)));
              test_dmap = amrex::DistributionMapping::makeKnapSack(
                *m_costs[lev], currentEfficiency, testEfficiency, nmax, false,
                amrex::ParallelDescriptor::IOProcessorNumber());
            }

            // IO proc determine if the test dmap offers significant
            // improvements
            if (
              (m_loadBalanceEffRatioThreshold > 0.0) &&
              (amrex::ParallelDescriptor::MyProc() ==
               amrex::ParallelDescriptor::IOProcessorNumber())) {
              remakeLevel = static_cast<int>(
                (remakeLevel != 0) ||
                (testEfficiency >
                 m_loadBalanceEffRatioThreshold * currentEfficiency));
            }
            amrex::ParallelDescriptor::Bcast(
              &remakeLevel, 1, amrex::ParallelDescriptor::IOProcessorNumber());

            if (m_verbose > 1 && (remakeLevel != 0)) {
              amrex::Print()
                << " Current LoadBalancing efficiency: " << currentEfficiency
                << "\n"
                << " Test LoadBalancing efficiency: " << testEfficiency
                << " \n";
            }

            // Bcast the test dmap if we plan on remaking the level
            if (remakeLevel != 0) {
              amrex::Vector<int> pmap;
              if (
                amrex::ParallelDescriptor::MyProc() ==
                amrex::ParallelDescriptor::IOProcessorNumber()) {
                pmap = test_dmap.ProcessorMap();
              } else {
#pragma GCC diagnostic ignored "-Wnull-dereference"
                pmap.resize(static_cast<std::size_t>(new_ba.size()));
              }
              amrex::ParallelDescriptor::Bcast(
                pmap.data(), pmap.size(),
                amrex::ParallelDescriptor::IOProcessorNumber());

              if (
                amrex::ParallelDescriptor::MyProc() !=
                amrex::ParallelDescriptor::IOProcessorNumber()) {
                test_dmap = amrex::DistributionMapping(pmap);
              }
              new_dmap = test_dmap;
            }
          }

          if (remakeLevel != 0) {
            const auto old_num_setdm = num_setdm;
            RemakeLevel(lev, time, new_ba, new_dmap);
            SetBoxArray(lev, new_ba);
            if (old_num_setdm == num_setdm) {
              SetDistributionMap(lev, new_dmap);
            }
          }
        }
        coarse_ba_changed = ba_changed;
      } else { // a new level, use default DMap strategy
        amrex::DistributionMapping new_dmap(new_grids[lev]);
        const auto old_num_setdm = num_setdm;
        MakeNewLevelFromCoarse(lev, time, new_grids[lev], new_dmap);
        SetBoxArray(lev, new_grids[lev]);
        if (old_num_setdm == num_setdm) {
          SetDistributionMap(lev, new_dmap);
        }
      }
    }

    for (int lev = new_finest + 1; lev <= finest_level; ++lev) {
      ClearLevel(lev);
      ClearBoxArray(lev);
      ClearDistributionMap(lev);
    }

    finest_level = new_finest;

#ifdef PELE_USE_RADIATION
    if (do_rad_solve) {
#ifdef AMREX_USE_EB
      rad_model->initVars(grids, dmap, m_factory);
#else
      rad_model->initVars(grids, dmap);
#endif
    }
#endif
  }
}

void
PeleLM::MakeNewLevelFromCoarse(
  const int lev,
  const amrex::Real time,
  const amrex::BoxArray& ba,
  const amrex::DistributionMapping& dm)
{
  BL_PROFILE("PeleLMeX::MakeNewLevelFromCoarse()");

  if (m_verbose > 0) {
    amrex::Print() << " Making new level " << lev << " from coarse\n";
    if (m_verbose > 1) {
      auto const dx = geom[lev].CellSizeArray();
      amrex::Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
      amrex::Print() << " with " << ba.numPts() << " cells, " << ba.size()
                     << " boxes,"
                     << " over "
                     << static_cast<amrex::Real>(ba.numPts()) * vol /
                          geom[0].ProbSize() * 100
                     << "% of the domain \n";
    }
    if (m_verbose > 3) {
      amrex::Print() << " with BoxArray " << ba << "\n";
    }
  }

  // New level factory
#ifdef AMREX_USE_EB
  std::unique_ptr<amrex::FabFactory<amrex::FArrayBox>> new_fact =
    makeEBFabFactory(geom[lev], ba, dm, {6, 6, 6}, amrex::EBSupport::full);
#else
  std::unique_ptr<amrex::FabFactory<amrex::FArrayBox>> new_fact(
    new amrex::FArrayBoxFactory());
#endif

  // New leveldatas
  std::unique_ptr<LevelData> n_leveldata_old(new LevelData(
    ba, dm, *new_fact, m_incompressible, m_has_divu, m_nAux, m_nGrowState,
    m_use_soret, static_cast<int>(m_do_les)));

  std::unique_ptr<LevelData> n_leveldata_new(new LevelData(
    ba, dm, *new_fact, m_incompressible, m_has_divu, m_nAux, m_nGrowState,
    m_use_soret, static_cast<int>(m_do_les)));

  // Mesh-mapping metrics (and the mapping-aware interpolators) must exist
  // BEFORE the coarse->fine fill so the new level is prolonged
  // conservatively.  Use *new_fact since m_factory[lev] is assigned below.
  if (m_mesh_map) {
    m_mesh_map->define(lev, ba, dm, *new_fact, m_nGrowState);
    m_mesh_map->create_map(lev, geom[lev]);
    m_map_eval = m_mesh_map->make_evaluator();
    rebuildMappedInterps();
  }
  defineCellVolume(lev, ba, dm, *new_fact);

  // Fill the leveldata_new
  fillcoarsepatch_state(lev, time, n_leveldata_new->state, m_nGrowState);
  if (m_nAux > 0) {
    fillcoarsepatch_aux(lev, time, n_leveldata_new->auxiliaries, m_nGrowState);
  }
  fillcoarsepatch_gradp(lev, time, n_leveldata_new->gp, 0);
  n_leveldata_new->press.setVal(0.0);

  if (m_incompressible == 0) {
    if (m_has_divu != 0) {
      fillcoarsepatch_divu(lev, time, n_leveldata_new->divu, 0);
    }
  }

  // Move std::unique_ptr into the PeleLM vector
  m_leveldata_old[lev] = std::move(n_leveldata_old);
  m_leveldata_new[lev] = std::move(n_leveldata_new);
  m_factory[lev] = std::move(new_fact);

  if (m_do_react != 0) {
    std::unique_ptr<LevelDataReact> n_leveldatareact(
      new LevelDataReact(ba, dm, *m_factory[lev]));
    fillcoarsepatch_reaction(lev, time, n_leveldatareact->I_R, 0);
    n_leveldatareact->functC.setVal(0.0);
    m_leveldatareact[lev] = std::move(n_leveldatareact);
  }

  if (m_incompressible == 0) {
    // Enforce density / species density consistency
    // only useful when using cell cons interp
    if (m_regrid_interp_method == 1) {
      setRhoToSumRhoY(lev, AmrNewTime);
    }

    // Recompute temperature from interpolated rhoYs and rhoH
    setTemperature(lev, AmrNewTime);

    // Initialize thermodynamic pressure
    setThermoPress(lev, AmrNewTime);
  }

  if (max_level > 0 && lev != max_level) {
    m_coveredMask[lev] = std::make_unique<amrex::iMultiFab>(ba, dm, 1, 0);
  }
  m_resetCoveredMask = 1;

#ifdef PELE_USE_PLASMA
  m_leveldatanlsolve[lev].reset(
    new LevelDataNLSolve(ba, dm, *m_factory[lev], m_nGrowState));
  if (m_do_extraEFdiags) {
    m_ionsFluxes[lev].reset(
      new amrex::MultiFab(ba, dm, NUM_IONS * AMREX_SPACEDIM, 0));
  }
  m_precond_op.reset();
#endif

  // Load balance
  m_costs[lev] = std::make_unique<amrex::LayoutData<amrex::Real>>(ba, dm);

  // DiffusionOp will be recreated
  m_diffusion_op.reset();
  m_mcdiffusion_op.reset();
  m_diffusionTensor_op.reset();

  // Trigger MacProj reset
  m_macProjNeedReset = 1;
  m_extSource[lev] = std::make_unique<amrex::MultiFab>(
    ba, dm, NVAR, amrex::max<int>(m_nGrowAdv, m_nGrowMAC), amrex::MFInfo(),
    *m_factory[lev]);
  m_extSource[lev]->setVal(0.);

  // Recycling-plane storage spans whatever set of AMR levels currently exists
  // and must be rebuilt whenever that changes.
  m_recycling_needs_rebuild = true;
}

void
PeleLM::RemakeLevel(
  const int lev,
  const amrex::Real time,
  const amrex::BoxArray& ba,
  const amrex::DistributionMapping& dm)
{
  BL_PROFILE("PeleLMeX::RemakeLevel()");

  if (m_verbose > 0) {
    amrex::Print() << " Remaking level " << lev << "\n";
    if (m_verbose > 1) {
      auto const dx = geom[lev].CellSizeArray();
      amrex::Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
      amrex::Print() << " with " << ba.numPts() << " cells," << ba.size()
                     << " boxes,"
                     << " over "
                     << static_cast<amrex::Real>(ba.numPts()) * vol /
                          geom[0].ProbSize() * 100
                     << "% of the domain \n";
    }
    if (m_verbose > 3) {
      amrex::Print() << " with BoxArray " << ba << "\n";
    }
  }

  // New level factory
#ifdef AMREX_USE_EB
  std::unique_ptr<amrex::FabFactory<amrex::FArrayBox>> new_fact =
    makeEBFabFactory(geom[lev], ba, dm, {6, 6, 6}, amrex::EBSupport::full);
#else
  std::unique_ptr<amrex::FabFactory<amrex::FArrayBox>> new_fact(
    new amrex::FArrayBoxFactory());
#endif

  // New leveldatas
  std::unique_ptr<LevelData> n_leveldata_old(new LevelData(
    ba, dm, *new_fact, m_incompressible, m_has_divu, m_nAux, m_nGrowState,
    m_use_soret, static_cast<int>(m_do_les)));

  std::unique_ptr<LevelData> n_leveldata_new(new LevelData(
    ba, dm, *new_fact, m_incompressible, m_has_divu, m_nAux, m_nGrowState,
    m_use_soret, static_cast<int>(m_do_les)));

  // Re-define the mesh-mapping metrics on the new ba/dm and rebuild the
  // interpolators BEFORE the fill, so the remade level uses the current,
  // matching metrics.  Use *new_fact since m_factory[lev] is assigned below.
  if (m_mesh_map) {
    m_mesh_map->define(lev, ba, dm, *new_fact, m_nGrowState);
    m_mesh_map->create_map(lev, geom[lev]);
    m_map_eval = m_mesh_map->make_evaluator();
    rebuildMappedInterps();
  }
  defineCellVolume(lev, ba, dm, *new_fact);

  // Fill the leveldata_new
  fillpatch_state(lev, time, n_leveldata_new->state, m_nGrowState);
  if (m_nAux > 0) {
    fillpatch_aux(lev, time, n_leveldata_new->auxiliaries, m_nGrowState);
  }
  fillpatch_gradp(lev, time, n_leveldata_new->gp, 0);
  n_leveldata_new->press.setVal(0.0);

  if (m_incompressible == 0) {
    if (m_has_divu != 0) {
      fillpatch_divu(lev, time, n_leveldata_new->divu, 1);
    }
  }

  // Move std::unique_ptr into the PeleLM vector
  m_leveldata_old[lev] = std::move(n_leveldata_old);
  m_leveldata_new[lev] = std::move(n_leveldata_new);
  m_factory[lev] = std::move(new_fact);

  if (m_do_react != 0) {
    std::unique_ptr<LevelDataReact> n_leveldatareact(
      new LevelDataReact(ba, dm, *m_factory[lev]));
    fillpatch_reaction(lev, time, n_leveldatareact->I_R, 0);
    n_leveldatareact->functC.setVal(0.0);
    m_leveldatareact[lev] = std::move(n_leveldatareact);
  }

  if (max_level > 0 && lev != max_level) {
    m_coveredMask[lev] = std::make_unique<amrex::iMultiFab>(ba, dm, 1, 0);
  }
  m_resetCoveredMask = 1;

  if (m_incompressible == 0) {
    // Enforce density / species density consistency
    // only useful when using cell cons interp
    if (m_regrid_interp_method == 1) {
      setRhoToSumRhoY(lev, AmrNewTime);
    }

    // Recompute temperature from interpolated rhoYs and rhoH
    setTemperature(lev, AmrNewTime);

    // Initialize thermodynamic pressure
    setThermoPress(lev, AmrNewTime);
  }

#ifdef PELE_USE_PLASMA
  m_leveldatanlsolve[lev].reset(
    new LevelDataNLSolve(ba, dm, *m_factory[lev], m_nGrowState));
  if (m_do_extraEFdiags) {
    m_ionsFluxes[lev].reset(
      new amrex::MultiFab(ba, dm, NUM_IONS * AMREX_SPACEDIM, 0));
  }
  m_precond_op.reset();
#endif

  // Load balance
  m_costs[lev] = std::make_unique<amrex::LayoutData<amrex::Real>>(ba, dm);

  // DiffusionOp will be recreated
  m_diffusion_op.reset();
  m_mcdiffusion_op.reset();
  m_diffusionTensor_op.reset();

  // Trigger MacProj reset
  m_macProjNeedReset = 1;
  m_extSource[lev] = std::make_unique<amrex::MultiFab>(
    ba, dm, NVAR, amrex::max<int>(m_nGrowAdv, m_nGrowMAC), amrex::MFInfo(),
    *m_factory[lev]);
  m_extSource[lev]->setVal(0.);

  // Recycling-plane storage spans whatever set of AMR levels currently exists
  // and must be rebuilt whenever that changes.
  m_recycling_needs_rebuild = true;
}

void
PeleLM::ClearLevel(const int lev)
{
  BL_PROFILE("PeleLMeX::ClearLevel()");

  m_leveldata_old[lev].reset();
  m_leveldata_new[lev].reset();
  if (m_do_react != 0) {
    m_leveldatareact[lev].reset();
  }
  if (max_level > 0 && lev != max_level) {
    m_coveredMask[lev].reset();
  }
  m_baChem[lev].reset();
  m_dmapChem[lev].reset();
  m_factory[lev].reset();
  m_diffusion_op.reset();
  m_mcdiffusion_op.reset();
  m_diffusionTensor_op.reset();
  macproj.reset();
#ifdef PELE_USE_PLASMA
  m_leveldatanlsolve[lev].reset();
  if (m_do_extraEFdiags) {
    m_ionsFluxes[lev].reset();
  }
#endif
  m_extSource[lev]->clear();

  m_costs[lev].reset();
  m_loadBalanceEff[lev] = -1.0;

  // Mesh mapping metric fields (if enabled; otherwise no-op)
  if (m_mesh_map) {
    m_mesh_map->clear_level(lev);
    m_mapped_interps.clear();
  }
  m_cellVol[lev].clear();
  // Recycling-plane storage covers all current levels; flag for rebuild.
  m_recycling_needs_rebuild = true;
}

void
PeleLM::computeCosts(
  const int a_lev,
  amrex::LayoutData<amrex::Real>& a_costs,
  const int a_costMethod)
{
  switch (a_costMethod) {
  case LoadBalanceCost::Ncell: {
    for (amrex::MFIter mfi(a_costs, false); mfi.isValid(); ++mfi) {
      a_costs[mfi] = static_cast<amrex::Real>(mfi.validbox().numPts());
    }
    return;
  }
  case LoadBalanceCost::ChemFunctCallAvg: {
    amrex::MultiFab costMF(a_costs.boxArray(), a_costs.DistributionMap(), 1, 0);
    fillpatch_chemFunctCall(a_lev, m_cur_time, costMF, 0);
    for (amrex::MFIter mfi(costMF, false); mfi.isValid(); ++mfi) {
      a_costs[mfi] = costMF[mfi].sum<amrex::RunOn::Device>(mfi.validbox(), 0) /
                     static_cast<amrex::Real>(mfi.validbox().numPts());
    }
    return;
  }
  case LoadBalanceCost::ChemFunctCallMax: {
    amrex::MultiFab costMF(a_costs.boxArray(), a_costs.DistributionMap(), 1, 0);
    fillpatch_chemFunctCall(a_lev, m_cur_time, costMF, 0);
    for (amrex::MFIter mfi(costMF, false); mfi.isValid(); ++mfi) {
      a_costs[mfi] = costMF[mfi].max<amrex::RunOn::Device>(mfi.validbox(), 0);
    }
    return;
  }
  case LoadBalanceCost::ChemFunctCallSum: {
    amrex::MultiFab costMF(a_costs.boxArray(), a_costs.DistributionMap(), 1, 0);
    fillpatch_chemFunctCall(a_lev, m_cur_time, costMF, 0);
    for (amrex::MFIter mfi(costMF, false); mfi.isValid(); ++mfi) {
      a_costs[mfi] = costMF[mfi].sum<amrex::RunOn::Device>(mfi.validbox(), 0);
    }
    return;
  }
  case LoadBalanceCost::UserDefinedDerivedAvg: {
    amrex::MultiFab costMF(a_costs.boxArray(), a_costs.DistributionMap(), 1, 0);
    costMF.setVal(0.0);
    std::unique_ptr<amrex::MultiFab> mf;
    mf = derive("derUserDefined", m_cur_time, a_lev, 0);
    costMF.ParallelCopy(*mf, 0, 0, 1);
    for (amrex::MFIter mfi(costMF, false); mfi.isValid(); ++mfi) {
      a_costs[mfi] = costMF[mfi].sum<amrex::RunOn::Device>(mfi.validbox(), 0) /
                     static_cast<amrex::Real>(mfi.validbox().numPts());
    }
    return;
  }
  case LoadBalanceCost::UserDefinedDerivedSum: {
    amrex::MultiFab costMF(a_costs.boxArray(), a_costs.DistributionMap(), 1, 0);
    costMF.setVal(0.0);
    std::unique_ptr<amrex::MultiFab> mf;
    mf = derive("derUserDefined", m_cur_time, a_lev, 0);
    costMF.ParallelCopy(*mf, 0, 0, 1);
    for (amrex::MFIter mfi(costMF, false); mfi.isValid(); ++mfi) {
      a_costs[mfi] = costMF[mfi].sum<amrex::RunOn::Device>(mfi.validbox(), 0);
    }
    return;
  }
  default:
    amrex::Abort(" Unknown cost estimate method !");
  }
}

void
PeleLM::computeCosts(const int a_lev)
{
  computeCosts(a_lev, *m_costs[a_lev], m_loadBalanceCost);
}

void
PeleLM::resetMacProjector()
{
  // If nothing has changed, just go back
  if ((m_macProjNeedReset == 0) && (m_macProjOldSize == finest_level + 1)) {
    return;
  }

  // MacProj
#ifdef AMREX_USE_EB
  macproj = std::make_unique<Hydro::MacProjector>(
    Geom(0, finest_level),
    amrex::MLMG::Location::FaceCentroid, // Location of mac velocity
    amrex::MLMG::Location::FaceCentroid, // Location of beta
    amrex::MLMG::Location::CellCenter);  // Location of solution variable phi
#else
  macproj = std::make_unique<Hydro::MacProjector>(Geom(0, finest_level));
#endif

  // Store the old MacProj size and switch off reset flag
  m_macProjOldSize = finest_level + 1;
  m_macProjNeedReset = 0;
}

void
PeleLM::regridFromGridFile(
  const int lbase, const amrex::Real time, const bool /*initial*/)
{
  const int new_finest = static_cast<int>(m_regrid_ba.size());
  amrex::Vector<amrex::BoxArray> new_grids(finest_level + 2);
  BL_ASSERT(new_finest <= finest_level + 1);

  bool coarse_ba_changed = false;
  for (int lev = lbase + 1; lev <= new_finest; ++lev) {
    new_grids[lev] = m_regrid_ba[lev - 1];
    if (lev <= finest_level) { // an old level
      bool ba_changed = (new_grids[lev] != grids[lev]);
      if (ba_changed || coarse_ba_changed) {
        amrex::BoxArray level_grids = grids[lev];
        amrex::DistributionMapping level_dmap = dmap[lev];
        if (ba_changed) {
          level_grids = new_grids[lev];
          level_dmap = amrex::DistributionMapping(level_grids);
        }
        const auto old_num_setdm = num_setdm;
        RemakeLevel(lev, time, level_grids, level_dmap);
        SetBoxArray(lev, level_grids);
        if (old_num_setdm == num_setdm) {
          SetDistributionMap(lev, level_dmap);
        }
      }
      coarse_ba_changed = ba_changed;
      ;
    } else { // a new level
      amrex::DistributionMapping new_dmap(new_grids[lev]);
      const auto old_num_setdm = num_setdm;
      MakeNewLevelFromCoarse(lev, time, new_grids[lev], new_dmap);
      SetBoxArray(lev, new_grids[lev]);
      if (old_num_setdm == num_setdm) {
        SetDistributionMap(lev, new_dmap);
      }
    }
  }
  for (int lev = new_finest + 1; lev <= finest_level; ++lev) {
    ClearLevel(lev);
    ClearBoxArray(lev);
    ClearDistributionMap(lev);
  }

  finest_level = new_finest;
}
