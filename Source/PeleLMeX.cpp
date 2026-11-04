#include <PeleLMeX.H>
#include <memory>

#ifdef PELE_USE_SPRAY
#include "SprayParticles.H"
#endif

pele::physics::PeleParams<pele::physics::transport::TransParm<
  pele::physics::PhysicsType::eos_type,
  pele::physics::PhysicsType::transport_type>>
  PeleLM::trans_parms;

pele::physics::PeleParams<
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>>
  PeleLM::eos_parms;

PeleLM::PeleLM() = default;

PeleLM::~PeleLM()
{
  if (m_incompressible == 0) {
    trans_parms.deallocate();
    eos_parms.deallocate();
    if (m_reactor) {
      m_reactor->close();
    }
  }

  closeTempFile();
  typical_values.clear();
  freeProbParm();
  delete prob_parm;
  amrex::The_Arena()->free(prob_parm_d);
  m_initial_ba.clear();
  m_regrid_ba.clear();
#ifdef PELE_USE_SPRAY
  SprayParticleContainer::SprayCleanUp();
  GhostPC.reset();
  VirtPC.reset();
  SprayPC.reset();
#endif
#ifdef PELE_USE_SOOT
  cleanupSootModel();
#endif
}

PeleLM::LevelData*
PeleLM::getLevelDataPtr(
  const int lev, const TimeStamp a_time, const int /*useUMac*/)
{
  AMREX_ASSERT(
    a_time == AmrOldTime || a_time == AmrNewTime || a_time == AmrHalfTime);
  if (a_time == AmrOldTime) {
    return m_leveldata_old[lev].get();
  }
  if (a_time == AmrNewTime) {
    return m_leveldata_new[lev].get();
  }
  m_leveldata_floating = std::make_unique<LevelData>(
    grids[lev], dmap[lev], *m_factory[lev], m_incompressible, m_has_divu,
    m_nAux, m_nGrowState, m_use_soret, static_cast<int>(m_do_les));
  const amrex::Real time = getTime(lev, a_time);
  fillpatch_state(lev, time, m_leveldata_floating->state, m_nGrowState);
  if (m_nAux > 0) {
    fillpatch_aux(lev, time, m_leveldata_floating->auxiliaries, m_nGrowState);
  }
  return m_leveldata_floating.get();
}

PeleLM::LevelDataReact*
PeleLM::getLevelDataReactPtr(const int lev)
{
  if (m_do_react != 0) {
    return m_leveldatareact[lev].get();
  }
  return nullptr;
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getStateVect(const TimeStamp a_time)
{
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    if (m_incompressible != 0) {
      for (int lev = 0; lev <= finest_level; ++lev) {
        r.push_back(
          std::make_unique<amrex::MultiFab>(
            m_leveldata_old[lev]->state, amrex::make_alias, 0, AMREX_SPACEDIM));
      }
    } else {
      for (int lev = 0; lev <= finest_level; ++lev) {
        r.push_back(
          std::make_unique<amrex::MultiFab>(
            m_leveldata_old[lev]->state, amrex::make_alias, 0, NVAR));
      }
    }
  } else {
    if (m_incompressible != 0) {
      for (int lev = 0; lev <= finest_level; ++lev) {
        r.push_back(
          std::make_unique<amrex::MultiFab>(
            m_leveldata_new[lev]->state, amrex::make_alias, 0, AMREX_SPACEDIM));
      }
    } else {
      for (int lev = 0; lev <= finest_level; ++lev) {
        r.push_back(
          std::make_unique<amrex::MultiFab>(
            m_leveldata_new[lev]->state, amrex::make_alias, 0, NVAR));
      }
    }
  }
  return r;
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getVelocityVect(const TimeStamp a_time)
{
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_old[lev]->state, amrex::make_alias, VELX,
          AMREX_SPACEDIM));
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_new[lev]->state, amrex::make_alias, VELX,
          AMREX_SPACEDIM));
    }
  }
  return r;
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getSpeciesVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_old[lev]->state, amrex::make_alias, FIRSTSPEC,
          NUM_SPECIES));
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_new[lev]->state, amrex::make_alias, FIRSTSPEC,
          NUM_SPECIES));
    }
  }
  return r;
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getDensityVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_old[lev]->state, amrex::make_alias, DENSITY, 1));
    }
  } else if (a_time == AmrNewTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_new[lev]->state, amrex::make_alias, DENSITY, 1));
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      const amrex::Real time = getTime(lev, a_time);
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          grids[lev], dmap[lev], 1, m_nGrowState));
      fillpatch_density(lev, time, *(r[lev]), 0, m_nGrowState);
    }
  }
  return r;
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getTempVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_old[lev]->state, amrex::make_alias, TEMP, 1));
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_new[lev]->state, amrex::make_alias, TEMP, 1));
    }
  }
  return r;
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getRhoHVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_old[lev]->state, amrex::make_alias, RHOH, 1));
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_new[lev]->state, amrex::make_alias, RHOH, 1));
    }
  }
  return r;
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getDivUVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  amrex::Vector<amrex::MultiFab*> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(&(m_leveldata_old[lev]->divu));
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(&(m_leveldata_new[lev]->divu));
    }
  }
  return r;
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getDiffusivityVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  amrex::Vector<amrex::MultiFab*> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(&(m_leveldata_old[lev]->diff_cc));
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(&(m_leveldata_new[lev]->diff_cc));
    }
  }
  return r;
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getViscosityVect(const TimeStamp a_time)
{
  amrex::Vector<amrex::MultiFab*> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(&(m_leveldata_old[lev]->visc_cc));
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(&(m_leveldata_new[lev]->visc_cc));
    }
  }
  return r;
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getIRVect()
{
  amrex::Vector<amrex::MultiFab*> r;
  r.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    r.push_back(&(m_leveldatareact[lev]->I_R));
  }
  return r;
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getAuxVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_nAux > 0);
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_old[lev]->auxiliaries, amrex::make_alias, 0, m_nAux));
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_new[lev]->auxiliaries, amrex::make_alias, 0, m_nAux));
    }
  }
  return r;
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getAuxDiffusivityVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_nAux > 0);
  amrex::Vector<amrex::MultiFab*> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(&(m_leveldata_old[lev]->diff_aux_cc));
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(&(m_leveldata_new[lev]->diff_aux_cc));
    }
  }
  return r;
}

void
PeleLM::averageDownState(const TimeStamp a_time)
{
  const int nCompState = (m_incompressible != 0) ? AMREX_SPACEDIM : NVAR;

  // Velocity components need the Xi-space-conservative restriction;
  // delegate to averageDownVelocity (which is mapping-aware).
  averageDownVelocity(a_time);

  // Remaining components AFTER velocity, if any (compressible runs).
  const int nc_rest = nCompState - AMREX_SPACEDIM;
  if (nc_rest > 0) {
    if (m_mesh_mapping) {
      // detJ-weighted restriction of [DENSITY..RHOH]; components after
      // RHOH (TEMP, RHORT, aux) use arithmetic and the derived ones are
      // overwritten by the EOS recompute below.
      averageDownConservedDensities(a_time);

      const int nc_after = NVAR - (RHOH + 1);
      if (nc_after > 0) {
        for (int lev = finest_level; lev > 0; --lev) {
          auto* ldataFine_p = getLevelDataPtr(lev, a_time);
          auto* ldataCrse_p = getLevelDataPtr(lev - 1, a_time);
#ifdef AMREX_USE_EB
          EB_average_down(
            ldataFine_p->state, ldataCrse_p->state, RHOH + 1, nc_after,
            refRatio(lev - 1));
#else
          average_down(
            ldataFine_p->state, ldataCrse_p->state, RHOH + 1, nc_after,
            refRatio(lev - 1));
#endif
        }
      }

      // Recompute TEMP/RHORT from the restricted densities via the EOS.
      // Done only on the mesh-mapping path here; applying this
      // (more thermodynamically consistent) treatment to the non-mapped
      // path too is a deferred enhancement -- see the tracking GitHub
      // issue -- since it would shift existing results and add cost.
      setTemperature(a_time);
      setThermoPress(a_time);
    } else {
      for (int lev = finest_level; lev > 0; --lev) {
        auto* ldataFine_p = getLevelDataPtr(lev, a_time);
        auto* ldataCrse_p = getLevelDataPtr(lev - 1, a_time);
#ifdef AMREX_USE_EB
        EB_average_down(
          ldataFine_p->state, ldataCrse_p->state, AMREX_SPACEDIM, nc_rest,
          refRatio(lev - 1));
#else
        average_down(
          ldataFine_p->state, ldataCrse_p->state, AMREX_SPACEDIM, nc_rest,
          refRatio(lev - 1));
#endif
      }
    }
  }
}

void
PeleLM::averageDownScalars(const TimeStamp a_time)
{
#ifdef PELE_USE_PLASMA
  constexpr int nScal = NUM_SPECIES + 5;
#else
  constexpr int nScal = NUM_SPECIES + 3;
#endif

  if (m_mesh_mapping) {
    // detJ-weighted restriction of [DENSITY..RHOH]; the rest of the
    // nScal block (TEMP, RHORT/plasma extras) uses arithmetic, with the
    // derived state restored by the EOS recompute below.
    averageDownConservedDensities(a_time);

    const int nc_dens = RHOH - DENSITY + 1; // rho + NUM_SPECIES rhoY + rhoH
    const int nc_after = nScal - nc_dens;
    if (nc_after > 0) {
      for (int lev = finest_level; lev > 0; --lev) {
        auto* ldataFine_p = getLevelDataPtr(lev, a_time);
        auto* ldataCrse_p = getLevelDataPtr(lev - 1, a_time);
#ifdef AMREX_USE_EB
        EB_average_down(
          ldataFine_p->state, ldataCrse_p->state, DENSITY + nc_dens, nc_after,
          refRatio(lev - 1));
#else
        average_down(
          ldataFine_p->state, ldataCrse_p->state, DENSITY + nc_dens, nc_after,
          refRatio(lev - 1));
#endif
      }
    }

    // Recompute TEMP/RHORT from the restricted densities via the EOS
    // (mesh-mapping path only; see averageDownState for the note on the
    // deferred generalization to the non-mapped path).
    setTemperature(a_time);
    setThermoPress(a_time);
    return;
  }

  for (int lev = finest_level; lev > 0; --lev) {
    auto* ldataFine_p = getLevelDataPtr(lev, a_time);
    auto* ldataCrse_p = getLevelDataPtr(lev - 1, a_time);
#ifdef AMREX_USE_EB
    EB_average_down(
      ldataFine_p->state, ldataCrse_p->state, DENSITY, nScal,
      refRatio(lev - 1));
#else
    average_down(
      ldataFine_p->state, ldataCrse_p->state, DENSITY, nScal,
      refRatio(lev - 1));
#endif
  }
}

void
PeleLM::averageDownAux(const TimeStamp a_time)
{
  for (int lev = finest_level; lev > 0; --lev) {
    auto* ldataFine_p = getLevelDataPtr(lev, a_time);
    auto* ldataCrse_p = getLevelDataPtr(lev - 1, a_time);
#ifdef AMREX_USE_EB
    EB_average_down(
      ldataFine_p->auxiliaries, ldataCrse_p->auxiliaries, 0, m_nAux,
      refRatio(lev - 1));
#else
    average_down(
      ldataFine_p->auxiliaries, ldataCrse_p->auxiliaries, 0, m_nAux,
      refRatio(lev - 1));
#endif
  }
}

void
PeleLM::averageDown(
  const TimeStamp a_time, const int state_comp, const int ncomp)
{
  for (int lev = finest_level; lev > 0; --lev) {
    auto* ldataFine_p = getLevelDataPtr(lev, a_time);
    auto* ldataCrse_p = getLevelDataPtr(lev - 1, a_time);
#ifdef AMREX_USE_EB
    EB_average_down(
      ldataFine_p->state, ldataCrse_p->state, state_comp, ncomp,
      refRatio(lev - 1));
#else
    average_down(
      ldataFine_p->state, ldataCrse_p->state, state_comp, ncomp,
      refRatio(lev - 1));
#endif
  }
}

void
PeleLM::rebuildMappedInterps()
{
  // Long-lived interp objects (one per C/F pair) so the address stays
  // stable: AMReX's TheFPinfo cache holds a raw pointer to the interp
  // across all fillpatch calls between regrids.
  if (!m_mesh_mapping || (m_mesh_map == nullptr)) {
    m_mapped_interps.clear();
    return;
  }
  // Size from the MeshMap's defined metric levels, NOT finest_level:
  // during regrid AmrCore updates finest_level only after the
  // MakeNewLevel*/RemakeLevel calls, so finest_level is stale here and a
  // pair would be missed for a newly added top level.
  const int n_pairs = amrex::max<int>(0, m_mesh_map->num_levels() - 1);
  m_mapped_interps.clear();
  m_mapped_interps.resize(n_pairs);
  // Component layout for the interpolator's weights (see its class doc).
  // Incompressible state has only velocity, so leave the density range empty.
  const int dens_start = (m_incompressible != 0) ? -1 : DENSITY;
  const int dens_end = (m_incompressible != 0) ? -2 : RHOH;
  for (int c = 0; c < n_pairs; ++c) {
    // Only build a pair whose two levels both have defined metrics;
    // clear_level() empties a level's MultiFabs without shrinking the
    // vector, so num_levels() can include cleared (coarsened-away) levels.
    if (m_mesh_map->detJ_cc(c).empty() || m_mesh_map->detJ_cc(c + 1).empty()) {
      continue; // leave entry null -> getInterpolator falls back to legacy
    }
    m_mapped_interps[c] = std::make_unique<MeshMappedCellConsInterp>(
      &m_mesh_map->detJ_cc(c), &m_mesh_map->fac_cc(c),
      &m_mesh_map->detJ_cc(c + 1), &m_mesh_map->fac_cc(c + 1), VELX,
      AMREX_SPACEDIM, dens_start, dens_end);
  }
}

void
PeleLM::averageDownVelocity(const TimeStamp a_time)
{
  for (int lev = finest_level; lev > 0; --lev) {
    auto* ldataFine_p = getLevelDataPtr(lev, a_time);
    auto* ldataCrse_p = getLevelDataPtr(lev - 1, a_time);

    if (m_mesh_mapping) {
      // Mass-conservative restriction under mesh mapping: convert v
      // to Xi-space (u_xi_i = v_i * detJ / fac_i), arithmetic-average
      // (which IS volume-conservative in the uniform Xi mesh),
      // convert back to physical space on the coarse level.  Required
      // to preserve discrete div-free across the C/F boundary.
      // No divide guard needed (unlike MeshMappedCellConsInterp, which
      // operates on seeded temporaries): detJ/fac are the MeshMap metrics,
      // strictly positive by construction and defined on every valid cell.
      const int nc = AMREX_SPACEDIM;
      amrex::MultiFab uxi_fine(
        ldataFine_p->state.boxArray(), ldataFine_p->state.DistributionMap(), nc,
        0, amrex::MFInfo(), ldataFine_p->state.Factory());
      amrex::MultiFab::Copy(uxi_fine, ldataFine_p->state, VELX, 0, nc, 0);
      {
        auto const& fac_ma = m_mesh_map->fac_cc(lev).const_arrays();
        auto const& detJ_ma = m_mesh_map->detJ_cc(lev).const_arrays();
        auto const& uxi_ma = uxi_fine.arrays();
        amrex::ParallelFor(
          uxi_fine, amrex::IntVect(0), nc,
          [=] AMREX_GPU_DEVICE(
            int box_no, int i, int j, int k, int n) noexcept {
            uxi_ma[box_no](i, j, k, n) *=
              detJ_ma[box_no](i, j, k) / fac_ma[box_no](i, j, k, n);
          });
        amrex::Gpu::streamSynchronize();
      }

      amrex::MultiFab uxi_crse(
        ldataCrse_p->state.boxArray(), ldataCrse_p->state.DistributionMap(), nc,
        0, amrex::MFInfo(), ldataCrse_p->state.Factory());
      // Stage coarse v in Xi-space so average_down overwrites only the
      // covered region; non-covered cells keep their pre-existing value.
      amrex::MultiFab::Copy(uxi_crse, ldataCrse_p->state, VELX, 0, nc, 0);
      {
        auto const& fac_ma = m_mesh_map->fac_cc(lev - 1).const_arrays();
        auto const& detJ_ma = m_mesh_map->detJ_cc(lev - 1).const_arrays();
        auto const& uxi_ma = uxi_crse.arrays();
        amrex::ParallelFor(
          uxi_crse, amrex::IntVect(0), nc,
          [=] AMREX_GPU_DEVICE(
            int box_no, int i, int j, int k, int n) noexcept {
            uxi_ma[box_no](i, j, k, n) *=
              detJ_ma[box_no](i, j, k) / fac_ma[box_no](i, j, k, n);
          });
        amrex::Gpu::streamSynchronize();
      }

#ifdef AMREX_USE_EB
      amrex::EB_average_down(uxi_fine, uxi_crse, 0, nc, refRatio(lev - 1));
#else
      amrex::average_down(uxi_fine, uxi_crse, 0, nc, refRatio(lev - 1));
#endif

      // Convert coarse u_xi back to physical-space v.
      {
        auto const& fac_ma = m_mesh_map->fac_cc(lev - 1).const_arrays();
        auto const& detJ_ma = m_mesh_map->detJ_cc(lev - 1).const_arrays();
        auto const& uxi_ma = uxi_crse.const_arrays();
        auto const& state_ma = ldataCrse_p->state.arrays();
        amrex::ParallelFor(
          uxi_crse, amrex::IntVect(0), nc,
          [=] AMREX_GPU_DEVICE(
            int box_no, int i, int j, int k, int n) noexcept {
            state_ma[box_no](i, j, k, VELX + n) = uxi_ma[box_no](i, j, k, n) *
                                                  fac_ma[box_no](i, j, k, n) /
                                                  detJ_ma[box_no](i, j, k);
          });
        amrex::Gpu::streamSynchronize();
      }
    } else {
#ifdef AMREX_USE_EB
      EB_average_down(
        ldataFine_p->state, ldataCrse_p->state, VELX, AMREX_SPACEDIM,
        refRatio(lev - 1));
#else
      average_down(
        ldataFine_p->state, ldataCrse_p->state, VELX, AMREX_SPACEDIM,
        refRatio(lev - 1));
#endif
    }
  }
}

void
PeleLM::averageDownConservedDensities(const TimeStamp a_time)
{
  // Volume-conservative restriction of [DENSITY..RHOH] = {rho, rhoY, rhoH}:
  // stage in Xi-space (d_xi = d*detJ), average_down (volume-conservative on
  // the uniform Xi mesh), convert back on the coarse level.  Like
  // averageDownVelocity but with the cell-volume weight detJ only (these are
  // scalar densities, not a flux).  Linear and identical per component, so
  // rho = sum(rhoY) is preserved; TEMP/RHORT are recomputed by the caller.
  // The detJ division needs no guard: detJ is the MeshMap metric, strictly
  // positive by construction and defined on every valid cell visited.
  AMREX_ASSERT(m_mesh_mapping && (m_mesh_map != nullptr));
  const int nc = RHOH - DENSITY + 1; // rho + NUM_SPECIES rhoY + rhoH
  for (int lev = finest_level; lev > 0; --lev) {
    auto* ldataFine_p = getLevelDataPtr(lev, a_time);
    auto* ldataCrse_p = getLevelDataPtr(lev - 1, a_time);

    amrex::MultiFab dxi_fine(
      ldataFine_p->state.boxArray(), ldataFine_p->state.DistributionMap(), nc,
      0, amrex::MFInfo(), ldataFine_p->state.Factory());
    amrex::MultiFab::Copy(dxi_fine, ldataFine_p->state, DENSITY, 0, nc, 0);
    {
      auto const& detJ_ma = m_mesh_map->detJ_cc(lev).const_arrays();
      auto const& dxi_ma = dxi_fine.arrays();
      amrex::ParallelFor(
        dxi_fine, amrex::IntVect(0), nc,
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          dxi_ma[box_no](i, j, k, n) *= detJ_ma[box_no](i, j, k);
        });
      amrex::Gpu::streamSynchronize();
    }

    amrex::MultiFab dxi_crse(
      ldataCrse_p->state.boxArray(), ldataCrse_p->state.DistributionMap(), nc,
      0, amrex::MFInfo(), ldataCrse_p->state.Factory());
    // Stage coarse in Xi-space so average_down overwrites only the covered
    // region; non-covered cells round-trip on the un-weight below.
    amrex::MultiFab::Copy(dxi_crse, ldataCrse_p->state, DENSITY, 0, nc, 0);
    {
      auto const& detJ_ma = m_mesh_map->detJ_cc(lev - 1).const_arrays();
      auto const& dxi_ma = dxi_crse.arrays();
      amrex::ParallelFor(
        dxi_crse, amrex::IntVect(0), nc,
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          dxi_ma[box_no](i, j, k, n) *= detJ_ma[box_no](i, j, k);
        });
      amrex::Gpu::streamSynchronize();
    }

#ifdef AMREX_USE_EB
    amrex::EB_average_down(dxi_fine, dxi_crse, 0, nc, refRatio(lev - 1));
#else
    amrex::average_down(dxi_fine, dxi_crse, 0, nc, refRatio(lev - 1));
#endif

    // Convert coarse d_xi back to physical-space density.
    {
      auto const& detJ_ma = m_mesh_map->detJ_cc(lev - 1).const_arrays();
      auto const& dxi_ma = dxi_crse.const_arrays();
      auto const& state_ma = ldataCrse_p->state.arrays();
      amrex::ParallelFor(
        dxi_crse, amrex::IntVect(0), nc,
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          state_ma[box_no](i, j, k, DENSITY + n) =
            dxi_ma[box_no](i, j, k, n) / detJ_ma[box_no](i, j, k);
        });
      amrex::Gpu::streamSynchronize();
    }
  }
}

void
PeleLM::averageDownReaction()
{
  for (int lev = finest_level; lev > 0; --lev) {
    auto* ldataRFine_p = getLevelDataReactPtr(lev);
    auto* ldataRCrse_p = getLevelDataReactPtr(lev - 1);
#ifdef AMREX_USE_EB
    EB_average_down(
      ldataRFine_p->I_R, ldataRCrse_p->I_R, 0, nCompIR(), refRatio(lev - 1));
#else
    average_down(
      ldataRFine_p->I_R, ldataRCrse_p->I_R, 0, nCompIR(), refRatio(lev - 1));
#endif
  }
}

#ifdef PELE_USE_PLASMA
amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getPhiVVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_old[lev]->state, amrex::make_alias, PHIV, 1));
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_new[lev]->state, amrex::make_alias, PHIV, 1));
    }
  }
  return r;
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>>
PeleLM::getnEVect(const TimeStamp a_time)
{
  AMREX_ASSERT(m_incompressible == 0);
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_old[lev]->state, amrex::make_alias, NE, 1));
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(
        std::make_unique<amrex::MultiFab>(
          m_leveldata_new[lev]->state, amrex::make_alias, NE, 1));
    }
  }
  return r;
}

amrex::Vector<amrex::MultiFab*>
PeleLM::getnEDiffusivityVect(const TimeStamp a_time)
{
  amrex::Vector<amrex::MultiFab*> r;
  r.reserve(finest_level + 1);
  if (a_time == AmrOldTime) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(&(m_leveldata_old[lev]->diffE_cc));
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      r.push_back(&(m_leveldata_new[lev]->diffE_cc));
    }
  }
  return r;
}
#endif
