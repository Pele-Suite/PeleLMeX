#include <PeleLMeX.H>
#include <PeleLMeX_K.H>

// Return velocity forces scaled by rhoInv
// including grapP term if add_gradP = 1
// including divTau if input amrex::Vector not empty
// a_extRhoExplicit: fraction of the -u extrho external source term included
void
PeleLM::getVelForces(
  const TimeStamp a_time,
  const amrex::Vector<amrex::MultiFab*>& a_divTau,
  const amrex::Vector<amrex::MultiFab*>& a_velForce,
  const int nGrowForce,
  const int add_gradP,
  const amrex::Real a_extRhoExplicit)
{
  BL_PROFILE("PeleLMeX::getVelForces()");
  const int has_divTau = static_cast<int>(!a_divTau.empty());

  for (int lev = 0; lev <= finest_level; ++lev) {
    getVelForces(
      a_time, lev, (has_divTau != 0) ? a_divTau[lev] : nullptr, a_velForce[lev],
      add_gradP, a_extRhoExplicit);
  }

  // FillPatch forces
  if (nGrowForce > 0) {
    fillpatch_forces(m_cur_time, a_velForce, nGrowForce);
  }
}

void
PeleLM::getVelForces(
  const TimeStamp a_time,
  const int lev,
  amrex::MultiFab* a_divTau,
  amrex::MultiFab* a_velForce,
  const int add_gradP,
  const amrex::Real a_extRhoExplicit)
{

  // Get level data
  // TODO: the 1 here bypass getting halftime vel and return oldtime vel
  auto* ldata_p = getLevelDataPtr(lev, a_time, 1);

  // Get gradp: if m_t_old < 0.0, we are during initialization -> only NewTime
  // data initialized at this point
  auto* ldataGP_p = (m_t_old[lev] < 0.0) ? getLevelDataPtr(lev, AmrNewTime)
                                         : getLevelDataPtr(lev, AmrOldTime);

  const amrex::Real time = getTime(lev, a_time);

  const int is_incomp = m_incompressible;
  const amrex::Real rho_incomp = m_rho;

  const int pseudo_gravity = m_ctrl_pseudoGravity;
  const amrex::Real dV_control = m_ctrl_dV;
  const auto grav = m_gravity;
  const auto gp0 = m_background_gp;
  const int ps_dir = m_ctrl_flameDir;

  auto const& state_ma = ldata_p->state.const_arrays();
  auto const& ext_ma = m_extSource[lev]->const_arrays();
  auto const& force_ma = a_velForce->arrays();

  amrex::ParallelFor(
    *a_velForce,
    [state_ma, ext_ma, force_ma, grav, gp0, ps_dir, is_incomp, rho_incomp,
     pseudo_gravity, dV_control,
     a_extRhoExplicit] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) {
      amrex::Array4<amrex::Real const> vel(state_ma[box_no], VELX);
      amrex::Array4<amrex::Real const> extmom(ext_ma[box_no], VELX);
      amrex::Array4<amrex::Real const> rho(state_ma[box_no], DENSITY);
      amrex::Array4<amrex::Real const> extrho(ext_ma[box_no], DENSITY);
      // Incompressible: no density in the state and no mass source, so
      // point rho/extrho at something (makeVelForce does not read them)
      if (is_incomp != 0) {
        rho = vel;
        extrho = extmom;
      }
      // background gp, pseudo grav, ext sources
      makeVelForce(
        i, j, k, is_incomp, rho_incomp, pseudo_gravity, ps_dir, grav, gp0,
        dV_control, vel, rho, extmom, extrho, a_extRhoExplicit,
        force_ma[box_no]);
    });
  amrex::Gpu::streamSynchronize();

  if (add_gradP != 0) {
    if (m_mesh_mapping) {
      // gp is stored in Xi-space (component i = fac_i . grad_phys p_i).
      // When subtracting from the physical-space velocity force we need
      // the physical gradient, hence the 1/fac_i per-component factor.
      auto const& gp_ma = ldataGP_p->gp.const_arrays();
      auto const& fac_ma = m_mesh_map->fac_cc(lev).const_arrays();
      auto const& fm_ma = a_velForce->arrays();
      amrex::ParallelFor(
        *a_velForce, amrex::IntVect(0), AMREX_SPACEDIM,
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          fm_ma[box_no](i, j, k, n) -=
            gp_ma[box_no](i, j, k, n) / fac_ma[box_no](i, j, k, n);
        });
      amrex::Gpu::streamSynchronize();
    } else {
      amrex::MultiFab::Subtract(
        *a_velForce, ldataGP_p->gp, 0, 0, AMREX_SPACEDIM, 0);
    }
  }
  const int has_divTau = static_cast<int>(a_divTau != nullptr);
  if (has_divTau != 0) {
    amrex::MultiFab::Add(*a_velForce, *a_divTau, 0, 0, AMREX_SPACEDIM, 0);
  }

#ifndef PELE_USE_PLASMA
  if (m_do_turbulent_forcing)
#endif
  {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*a_velForce, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const auto& bx = mfi.tilebox();
      amrex::FArrayBox DummyFab(bx, 1);
      const auto& rho_arr = (m_incompressible != 0)
                              ? DummyFab.array()
                              : ldata_p->state.const_array(mfi, DENSITY);
      const auto& force_arr = a_velForce->array(mfi);
#ifdef PELE_USE_PLASMA
      const auto& rhoY_arr = (m_incompressible != 0)
                               ? DummyFab.array()
                               : ldata_p->state.const_array(mfi, FIRSTSPEC);
      const auto& phiV_arr = ldata_p->state.const_array(mfi, PHIV);
      const auto& ne_arr = ldata_p->state.const_array(mfi, NE);
      addLorentzVelForces(lev, bx, time, force_arr, rhoY_arr, phiV_arr, ne_arr);
      if (m_do_turbulent_forcing)
#endif
      {
        // Add forcing terms to maintain turbulence
        // note: if m_incompressible == 0 then m_rho is unused by
        // addTurbVelForce
        turb_forcing.addTurbVelForces(
          geom[lev].data(), bx, time, force_arr, rho_arr, is_incomp,
          rho_incomp);
      }
    }
  }

  if (is_incomp != 0) {
    a_velForce->mult(1.0 / rho_incomp, 0, AMREX_SPACEDIM, 0);
  } else {
    amrex::ParallelFor(
      *a_velForce, amrex::IntVect(0), AMREX_SPACEDIM,
      [force_ma, state_ma] AMREX_GPU_DEVICE(
        int box_no, int i, int j, int k, int n) noexcept {
        amrex::Array4<amrex::Real const> rho(state_ma[box_no], DENSITY);
        force_ma[box_no](i, j, k, n) /= rho(i, j, k);
      });
    amrex::Gpu::streamSynchronize();
  }
}

void
PeleLM::addSpark(const TimeStamp a_timestamp)
{
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int n = 0; n < m_n_sparks; ++n) {
      // Do the checks first
      const amrex::Real time = getTime(lev, a_timestamp);
      const bool verb = m_spark_verbose > 1 && lev == 0;
      if (
        time < m_spark_time[n] ||
        time > m_spark_time[n] + m_spark_duration[n]) {
        if (verb) {
          amrex::Print() << m_spark[n] << " not active \n";
        }
        continue;
      }
      const amrex::Real* probLo = geom[lev].ProbLo();
      auto const dx = geom[lev].CellSizeArray();
      amrex::IntVect spark_idx;
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        // Spark location is physical: index it through Xi.
        const amrex::Real xi = m_map_eval.xi_from_x_phys(
          d, m_spark_location[n][d], geom[lev].data());
        spark_idx[d] = static_cast<int>(std::floor((xi - probLo[d]) / dx[d]));
      }
      const amrex::Box domainBox = geom[lev].Domain();
      // just a check
      if (!domainBox.contains(spark_idx)) {
        amrex::Warning(m_spark[n] + " not in domain!");
        continue;
      }
      if (verb) {
        amrex::Print() << m_spark[n] << " active\n";
      }
      auto const* leosparm = eos_parms.device_parm();

      auto const& statema =
        getLevelDataPtr(lev, a_timestamp)->state.const_arrays();
      auto const& extma = m_extSource[lev]->arrays();
      const MeshMapEvaluator mmap = m_map_eval;
      const amrex::GeometryData gd = geom[lev].data();
      amrex::ParallelFor(
        *m_extSource[lev],
        [statema, extma, leosparm, dx, spark_idx, mmap, gd,
         spark_duration = m_spark_duration[n], spark_temp = m_spark_temp[n],
         spark_radius = m_spark_radius
           [n]] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
          auto eos = pele::physics::PhysicsType::eos(leosparm);
          // Physical distance from the centre of the spark cell; the
          // index-space form is kept bit-identical without mapping.
          amrex::Real dist_to_center = 0.0;
          if (!is_identity_map(mmap)) {
            const int ic[3] = {i, j, k};
            amrex::Real d2 = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
              const amrex::Real dxc = mmap.x_phys_cc(d, ic[d], gd) -
                                      mmap.x_phys_cc(d, spark_idx[d], gd);
              d2 += dxc * dxc;
            }
            dist_to_center = std::sqrt(d2);
          } else {
            dist_to_center = std::sqrt(AMREX_D_TERM(
              (i - spark_idx[0]) * (i - spark_idx[0]) * dx[0] * dx[0],
              +(j - spark_idx[1]) * (j - spark_idx[1]) * dx[1] * dx[1],
              +(k - spark_idx[2]) * (k - spark_idx[2]) * dx[2] * dx[2]));
          }
          if (dist_to_center < spark_radius) {
            amrex::Real rhoh_src_loc = 0;
            const amrex::Real rho = statema[box_no](i, j, k, DENSITY);
            amrex::Real Y[NUM_SPECIES];
            for (int ns = 0; ns < NUM_SPECIES; ++ns) {
              Y[ns] = statema[box_no](i, j, k, FIRSTSPEC + ns) / rho;
            }
            eos.TY2H(spark_temp, Y, rhoh_src_loc);
            rhoh_src_loc *= rho * 1e-4 / spark_duration;
            extma[box_no](i, j, k, RHOH) = rhoh_src_loc;
          }
        });
    }
  }
  amrex::Gpu::streamSynchronize();
}

// Manifold model - dissipation rate sources for variances
void
PeleLM::addScalarVarianceSources(const TimeStamp a_timestamp)
{
  BL_PROFILE("PeleLM::addScalarVarianceSources");
  // no scalar dissipation sources if not using a manifold model
#ifndef USE_MANIFOLD_EOS
  amrex::ignore_unused(a_timestamp);
#else

  if (m_add_variance_sources) {
    auto const& leosparm = eos_parms.host_parm();

    // Determine if we have any scalar variances that need sources added
    // Could be moved elsewhere to not do every timestep
    int nvariances = 0;
    int var_of_scalar = -1;
    for (int n = 0; n < MANIFOLD_DIM; ++n) {
      if (leosparm.is_variance_of[n] >= 0) {
        if (!m_do_les) {
          amrex::Abort(
            "PeleLM::addScalarVarianceSources(): cannot add a "
            "scalar dissipation without an active LES model");
        }
        nvariances += 1;
        var_of_scalar = FIRSTSPEC + leosparm.is_variance_of[n];
      }
    }

    if (nvariances > 1) {
      amrex::Abort(
        "PeleLM::addScalarVarianceSources(): currently we only support "
        "manifold models with 0 or 1 variances");
    } else if (nvariances > 0) {

      // Compute scalar gradients (no need to average down here)
      constexpr int do_avgDown = 0;
      auto bcRecScalar = fetchBCRecArray(var_of_scalar, 1);
      constexpr int nGrow = 0; // No need for ghost face on fluxes
      amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> grad_fc(
        finest_level + 1);
      for (int lev = 0; lev <= finest_level; ++lev) {
        const auto& ba = grids[lev];
        const auto& factory = Factory(lev);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          grad_fc[lev][idim].define(
            amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)),
            dmap[lev], 1, nGrow, amrex::MFInfo(), factory);
          grad_fc[lev][idim].setVal(0.0); // Required?
        }
      }
      getDiffusionOp()->computeGradient(
        GetVecOfArrOfPtrs(grad_fc), {},
        GetVecOfConstPtrs(getStateVect(a_timestamp)), {}, bcRecScalar[0],
        do_avgDown, var_of_scalar);

      constexpr amrex::Real fact =
        0.5 / static_cast<amrex::Real>(AMREX_SPACEDIM);
      const amrex::Real C_chi = m_les_c_chi;
      const amrex::Real ScInv = m_Schmidt_inv;

      // Add in Production and Dissipation source terms for subfilter variances
      for (int lev = 0; lev <= finest_level; ++lev) {

        auto* ldata_p = getLevelDataPtr(lev, a_timestamp);

        // Require the turbulent viscosity to be pre-computed
        // it always is stored at AmrOldTime, so we just use that
        // We need cell-centered mu_t but have it at faces
        // The simple interpolation below probably isn't valid for EB
#ifdef AMREX_USE_EB
        amrex::Abort(
          "PeleLM::addScalarVarianceSources(): this is not supported with "
          "EB");
#endif
        // l_scale will also need modification for EB
        const amrex::Real vol = AMREX_D_TERM(
          geom[lev].CellSize(0), *geom[lev].CellSize(1),
          *geom[lev].CellSize(2));
        const amrex::Real l_scale =
          AMREX_D_PICK(vol, std::sqrt(vol), std::cbrt(vol));
        const amrex::Real inv_l_scale2 = 1.0 / (l_scale * l_scale);

        for (int n = 0; n < MANIFOLD_DIM; ++n) {
          if (leosparm.is_variance_of[n] >= 0) {

            AMREX_D_TERM(
              auto const& mut_arr_x =
                m_leveldata_old[lev]->visc_turb_fc[0].const_arrays();
              , auto const& mut_arr_y =
                  m_leveldata_old[lev]->visc_turb_fc[1].const_arrays();
              , auto const& mut_arr_z =
                  m_leveldata_old[lev]->visc_turb_fc[2].const_arrays();)
            AMREX_D_TERM(
              auto const& gx = grad_fc[lev][0].const_arrays();
              , auto const& gy = grad_fc[lev][1].const_arrays();
              , auto const& gz = grad_fc[lev][2].const_arrays();)
            auto const& extma = m_extSource[lev]->arrays();
            auto const& statema = ldata_p->state.const_arrays();

            amrex::ParallelFor(
              *m_extSource[lev], [extma, statema, n, C_chi, ScInv, inv_l_scale2,
                                  mut_arr_x, gx, mut_arr_y, gy
#if (AMREX_SPACEDIM == 3)
                                  ,
                                  mut_arr_z, gz
#endif
            ] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
                // Subfilter Scalar Dissipation: Linear Relaxation model
                // rho chi_sgs = C_chi * mu_t / Delta^2 * Variance
                const amrex::Real mu_t =
                  fact *
                  (AMREX_D_TERM(
                    mut_arr_x[bx](i, j, k) + mut_arr_x[bx](i + 1, j, k),
                    +mut_arr_y[bx](i, j, k) + mut_arr_y[bx](i, j + 1, k),
                    +mut_arr_z[bx](i, j, k) + mut_arr_z[bx](i, j, k + 1)));

                extma[bx](i, j, k, FIRSTSPEC + n) -=
                  C_chi * mu_t * inv_l_scale2 *
                  statema[bx](i, j, k, FIRSTSPEC + n);

                // Production term (w/ Smagorinsky closure for turbulent flux)
                // -2 (rho <u_j C> - rho <u_j><C>) d<C>/dx_j
                // = 2 *mu_t/Sc_t * d<C>/dx_j * d<C>/dx_j
                const amrex::Real mu_grad2 =
                  fact *
                  (AMREX_D_TERM(
                    mut_arr_x[bx](i, j, k) * gx[bx](i, j, k) * gx[bx](i, j, k) +
                      mut_arr_x[bx](i + 1, j, k) * gx[bx](i + 1, j, k) *
                        gx[bx](i + 1, j, k),
                    +mut_arr_y[bx](i, j, k) * gy[bx](i, j, k) *
                        gy[bx](i, j, k) +
                      mut_arr_y[bx](i, j + 1, k) * gy[bx](i, j + 1, k) *
                        gy[bx](i, j + 1, k),
                    +mut_arr_z[bx](i, j, k) * gz[bx](i, j, k) *
                        gz[bx](i, j, k) +
                      mut_arr_z[bx](i, j, k + 1) * gz[bx](i, j, k + 1) *
                        gz[bx](i, j, k + 1)));

                extma[bx](i, j, k, FIRSTSPEC + n) += 2.0 * ScInv * mu_grad2;
              });
          }
        }
      }
      amrex::Gpu::streamSynchronize();
    }
  }
#endif
}

// Calculate additional external sources (soot, radiation, user defined, etc.)
// at t^n. With ext_sources_SDC, also keep S^n and its frozen part in diffData
// for the SDC re-evaluations.
void
PeleLM::getExternalSources(
  const int is_initIter,
  const PeleLM::TimeStamp a_timestamp_old,
  const PeleLM::TimeStamp a_timestamp_new,
  const std::unique_ptr<AdvanceDiffData>& diffData)
{
  if (!m_ext_sources_SDC) {
    assembleExternalSources(
      is_initIter, a_timestamp_old, a_timestamp_new, nullptr);
    return;
  }

  // extSrc_n/np1 are allocated with diffData; the frozen snapshot only
  // when a frozen source is active
  auto& frozen = diffData->extSrc_frozen;
  AMREX_ASSERT(frozen.empty() && !diffData->extSrc_n.empty());
  // m_extSource is zero on entry: snapshot the frozen part on its own
  if (addFrozenExternalSources(is_initIter, a_timestamp_old)) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto& ext = *m_extSource[lev];
      frozen.emplace_back(
        grids[lev], dmap[lev], NVAR, ext.nGrowVect(), amrex::MFInfo(),
        *m_factory[lev]);
      std::swap(frozen[lev], ext);
      ext.setVal(0.0);
    }
  }
  assembleExternalSources(
    is_initIter, a_timestamp_old, a_timestamp_new, &frozen);
  // extSrc_np1 is filled by the first updateExternalSourcesSDC
  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto& ext = *m_extSource[lev];
    amrex::MultiFab::Copy(
      diffData->extSrc_n[lev], ext, 0, 0, NVAR, ext.nGrowVect());
  }
}

// Spray advances its own particles (MKD), soot integrates its source over dt
// internally and radiation needs a global solve, so these are only evaluated
// once per step, from t^n. Returns whether any of them is active.
bool
PeleLM::addFrozenExternalSources(
  const int is_initIter, const PeleLM::TimeStamp a_timestamp)
{
  amrex::ignore_unused(is_initIter, a_timestamp);
  bool added = false;
#ifdef PELE_USE_SPRAY
  if (is_initIter == 0) {
    SprayMKD(m_cur_time, m_dt);
    added = added || do_spray_particles;
  }
#endif
#ifdef PELE_USE_SOOT
  if (do_soot_solve) {
    computeSootSource(a_timestamp, m_dt);
    added = true;
  }
#endif
#ifdef PELE_USE_RADIATION
  if (do_rad_solve) {
    BL_PROFILE_VAR("PeleLM::advance::rad", PLM_RAD);
    computeRadSource(a_timestamp);
    BL_PROFILE_VAR_STOP(PLM_RAD);
    added = true;
  }
#endif
  return added;
}

// divU for the initial projection(s), with the external sources already
// active at t^0. Otherwise u^0 misses the flow they drive, and the velocity
// advection of the first step imprints an O(dt) error that the projections
// do not remove. Only the sources that can be evaluated at an instant are
// included; the spray, soot and radiation sources span a step. The variance
// sources need the turbulent viscosity at the old time, which is not set up
// yet.
void
PeleLM::calcInitDivU(const amrex::Real a_dt)
{
  constexpr int is_initialization = 1;
  constexpr int computeDiffusionTerm = 1;
  constexpr int do_avgDown = 1;
  // Light version of the diffusion data container
  auto diffData = std::make_unique<AdvanceDiffData>(
    finest_level, grids, dmap, m_factory, m_nGrowAdv, m_use_wbar, m_use_soret,
    m_nAux, is_initialization);

  for (int lev = 0; lev <= finest_level; ++lev) {
    m_extSource[lev]->setVal(0.0);
  }
  if (m_n_sparks > 0) {
    addSpark(AmrNewTime);
  }
  addUserExternalSources(AmrNewTime, AmrNewTime, a_dt);

  calcDivU(
    is_initialization, computeDiffusionTerm, do_avgDown, AmrNewTime, diffData,
    GetVecOfConstPtrs(m_extSource));
  for (int lev = 0; lev <= finest_level; ++lev) {
    m_extSource[lev]->setVal(0.0);
  }
}

// ProblemSpecificFunctions::modify_ext_sources on every level
void
PeleLM::addUserExternalSources(
  const PeleLM::TimeStamp a_timestamp_eval,
  const PeleLM::TimeStamp a_timestamp_new,
  const amrex::Real a_dt)
{
  if (!m_user_defined_ext_sources) {
    return;
  }
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p_eval = getLevelDataPtr(lev, a_timestamp_eval);
    auto* ldata_p_new = getLevelDataPtr(lev, a_timestamp_new);
    ProblemSpecificFunctions::modify_ext_sources(
      getTime(lev, a_timestamp_eval), a_dt, ldata_p_eval->state,
      ldata_p_new->state, m_extSource[lev], geom[lev].data(), prob_parm_d);
  }
}

// Add the external sources evaluated from the a_timestamp_eval state into
// m_extSource. a_frozen: nullptr computes the frozen sources, otherwise its
// (possibly empty) stored snapshot is added instead.
void
PeleLM::assembleExternalSources(
  const int is_initIter,
  const PeleLM::TimeStamp a_timestamp_eval,
  const PeleLM::TimeStamp a_timestamp_new,
  const amrex::Vector<amrex::MultiFab>* a_frozen)
{
  if (m_n_sparks > 0) {
    addSpark(a_timestamp_eval);
  }
  if (a_frozen == nullptr) {
    addFrozenExternalSources(is_initIter, a_timestamp_eval);
  } else if (!a_frozen->empty()) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab::Add(
        *m_extSource[lev], (*a_frozen)[lev], 0, 0, NVAR,
        m_extSource[lev]->nGrowVect());
    }
  }
  addScalarVarianceSources(a_timestamp_eval);
  addUserExternalSources(a_timestamp_eval, a_timestamp_new, m_dt);
}

// ext_sources_SDC: S^{n+1,k} from the new state into diffData->extSrc_np1,
// and m_extSource <- 0.5 (S^n + S^{n+1,k})
void
PeleLM::updateExternalSourcesSDC(
  const std::unique_ptr<AdvanceDiffData>& diffData)
{
  BL_PROFILE("PeleLMeX::updateExternalSourcesSDC()");
  for (int lev = 0; lev <= finest_level; ++lev) {
    m_extSource[lev]->setVal(0.0);
  }
  assembleExternalSources(0, AmrNewTime, AmrNewTime, &diffData->extSrc_frozen);
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto& ext = *m_extSource[lev];
    std::swap(ext, diffData->extSrc_np1[lev]);
    amrex::MultiFab::LinComb(
      ext, 0.5, diffData->extSrc_n[lev], 0, 0.5, diffData->extSrc_np1[lev], 0,
      0, NVAR, ext.nGrowVect());
  }
}
