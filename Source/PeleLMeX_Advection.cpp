#include <PeleLMeX.H>
#include <PeleLMeX_K.H>
#include <PeleLMeX_Utils.H>
#include <hydro_utils.H>

void
PeleLM::computeVelocityAdvTerm(const std::unique_ptr<AdvanceAdvData>& advData)
{
  //----------------------------------------------------------------
  // Create temporary containers
  constexpr int nGrow_force = 1;
  amrex::Vector<amrex::MultiFab> divtau;
  divtau.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> velForces;
  velForces.reserve(finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes(
    finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> faces(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    divtau.emplace_back(
      grids[lev], dmap[lev], AMREX_SPACEDIM, 0, amrex::MFInfo(), Factory(lev));
    velForces.emplace_back(
      grids[lev], dmap[lev], AMREX_SPACEDIM, nGrow_force, amrex::MFInfo(),
      Factory(lev));
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      fluxes[lev][idim].define(
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
        dmap[lev], AMREX_SPACEDIM, 0, amrex::MFInfo(), Factory(lev));
      faces[lev][idim].define(
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
        dmap[lev], AMREX_SPACEDIM, 0, amrex::MFInfo(), Factory(lev));
    }
  }

  //----------------------------------------------------------------
  // Get viscous forces
  constexpr int use_density = 0;
  computeDivTau(AmrOldTime, GetVecOfPtrs(divtau), use_density);

  // Add compensating pressure gradient for periodic channel flow
  if (m_do_periodic_channel != 0) {
    m_background_gp[m_periodic_channel_dir] =
      MFSum(GetVecOfConstPtrs(divtau), m_periodic_channel_dir) / m_uncoveredVol;
    if (m_verbose > 2) {
      amrex::Print() << "   ... Adding background pressure gradient in dir "
                     << m_periodic_channel_dir << ": "
                     << m_background_gp[m_periodic_channel_dir] << "\n";
    }
  }

  //----------------------------------------------------------------
  // Gather all the velocity forces
  // F = [ (gravity+...) - gradP + divTau ] / rho
  constexpr int add_gradP = 1;
  getVelForces(
    AmrOldTime, GetVecOfPtrs(divtau), GetVecOfPtrs(velForces), nGrow_force,
    add_gradP);

  auto bcRecVel = fetchBCRecArray(VELX, AMREX_SPACEDIM);
  auto bcRecVel_d = convertToDeviceVector(bcRecVel);
  auto AdvTypeVel = fetchAdvTypeArray(VELX, AMREX_SPACEDIM);
  auto AdvTypeVel_d = convertToDeviceVector(AdvTypeVel);
  for (int lev = 0; lev <= finest_level; ++lev) {

    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);

    //----------------------------------------------------------------
    // Get divU
    amrex::MultiFab divu(
      grids[lev], dmap[lev], 1, m_nGrowdivu, amrex::MFInfo(), Factory(lev));
    if (m_incompressible != 0) {
      divu.setVal(0.0);
    } else {
      const amrex::Real time = getTime(lev, AmrOldTime);
      fillpatch_divu(lev, time, divu, m_nGrowdivu);
    }

    // Under mesh mapping the extrapolation is done in Xi with the
    // Xi-velocity u~ = umac / J_fc and the physical divU; the fluxes
    // u~.q are rescaled by J_fc below (see getExtrapUmac).
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> umac_xi;
    const auto umacE = getExtrapUmac(lev, advData, umac_xi);

    //----------------------------------------------------------------
#ifdef AMREX_USE_EB
    const auto& ebfact = EBFactory(lev);
#endif

    //----------------------------------------------------------------
    // Compute the velocity fluxes
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {

      amrex::Box const& bx = mfi.tilebox();
      AMREX_D_TERM(
        auto const& umac = umacE[0]->const_array(mfi);
        , auto const& vmac = umacE[1]->const_array(mfi);
        , auto const& wmac = umacE[2]->const_array(mfi);)
      AMREX_D_TERM(
        auto const& fx = fluxes[lev][0].array(mfi);
        , auto const& fy = fluxes[lev][1].array(mfi);
        , auto const& fz = fluxes[lev][2].array(mfi);)
      AMREX_D_TERM(
        auto const& facex = faces[lev][0].array(mfi);
        , auto const& facey = faces[lev][1].array(mfi);
        , auto const& facez = faces[lev][2].array(mfi);)
      auto const& divu_arr = divu.const_array(mfi);
      auto const& vel_arr = ldata_p->state.const_array(mfi, VELX);
      auto const& force_arr = velForces[lev].const_array(mfi);

      constexpr bool is_velocity = true;
      constexpr bool fluxes_are_area_weighted = false;
      constexpr bool knownEdgeState = false;
      HydroUtils::ComputeFluxesOnBoxFromState(
        bx, AMREX_SPACEDIM, mfi, vel_arr, AMREX_D_DECL(fx, fy, fz),
        AMREX_D_DECL(facex, facey, facez), knownEdgeState,
        AMREX_D_DECL(umac, vmac, wmac), divu_arr, force_arr, geom[lev], m_dt,
        bcRecVel, bcRecVel_d.dataPtr(), AdvTypeVel_d.dataPtr(),
#ifdef AMREX_USE_EB
        ebfact,
        (m_useEBinflow != 0)
          ? getEBState(mfi, lev, VELX, AMREX_SPACEDIM, AmrOldTime).const_array()
          : amrex::Array4<amrex::Real const>{},
#endif
        m_Godunov_ppm != 0, m_Godunov_ForceInTrans != 0, is_velocity,
        fluxes_are_area_weighted, m_advection_type, m_Godunov_ppm_limiter);
    }
    scaleMappedAdvFluxes(lev, fluxes[lev], 0, AMREX_SPACEDIM);
#ifdef AMREX_USE_EB
    EB_set_covered_faces(GetArrOfPtrs(fluxes[lev]), 0.);
    EB_set_covered_faces(GetArrOfPtrs(faces[lev]), 0.);
#endif
  }

  //----------------------------------------------------------------
  // Average down fluxes to ensure C/F consistency
  for (int lev = finest_level; lev > 0; --lev) {
#ifdef AMREX_USE_EB
    EB_average_down_faces(
      GetArrOfConstPtrs(fluxes[lev]), GetArrOfPtrs(fluxes[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
    EB_average_down_faces(
      GetArrOfConstPtrs(faces[lev]), GetArrOfPtrs(faces[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
#else
    average_down_faces(
      GetArrOfConstPtrs(fluxes[lev]), GetArrOfPtrs(fluxes[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
    average_down_faces(
      GetArrOfConstPtrs(faces[lev]), GetArrOfPtrs(faces[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
#endif
  }

  //----------------------------------------------------------------
  // Fluxes divergence to get the velocity advection term
  for (int lev = 0; lev <= finest_level; ++lev) {

    amrex::MultiFab divu(
      grids[lev], dmap[lev], 1, m_nGrowdivu, amrex::MFInfo(), Factory(lev));
    if (m_incompressible != 0) {
      divu.setVal(0.0);
    } else {
      const amrex::Real time = getTime(lev, AmrOldTime);
      fillpatch_divu(lev, time, divu, m_nGrowdivu);
    }
    if (m_mesh_mapping) {
      amrex::MultiFab::Multiply(
        divu, m_mesh_map->detJ_cc(lev), 0, 0, 1, m_nGrowdivu);
    }

    constexpr bool fluxes_are_area_weighted = false;
#ifdef AMREX_USE_EB
    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);
    //----------------------------------------------------------------
    // Use a temporary MF to hold divergence before redistribution
    constexpr int nGrow_divT = 3;
    amrex::MultiFab divTmp(
      grids[lev], dmap[lev], AMREX_SPACEDIM, nGrow_divT, amrex::MFInfo(),
      EBFactory(lev));
    divTmp.setVal(0.0);
    if (m_useEBinflow != 0) {
      advFluxDivergence(
        lev, divTmp, 0, divu, GetArrOfConstPtrs(fluxes[lev]), 0,
        GetArrOfConstPtrs(faces[lev]), 0,
        getEBState(lev, VELX, AMREX_SPACEDIM, AmrOldTime).get(),
        getEBState(lev, VELX, AMREX_SPACEDIM, AmrOldTime).get(), AMREX_SPACEDIM,
        AdvTypeVel_d.dataPtr(), geom[lev], -1.0, fluxes_are_area_weighted);
    } else {
      advFluxDivergence(
        lev, divTmp, 0, divu, GetArrOfConstPtrs(fluxes[lev]), 0,
        GetArrOfConstPtrs(faces[lev]), 0, AMREX_SPACEDIM,
        AdvTypeVel_d.dataPtr(), geom[lev], -1.0, fluxes_are_area_weighted);
    }

    divTmp.FillBoundary(geom[lev].periodicity());

    redistributeAofS(
      lev, m_dt, divTmp, 0, advData->AofS[lev], VELX, ldata_p->state, VELX,
      AMREX_SPACEDIM, bcRecVel_d.dataPtr(), geom[lev]);
    EB_set_covered(advData->AofS[lev], 0.0);
#else
    //----------------------------------------------------------------
    // Otherwise go directly into AofS
    advFluxDivergence(
      lev, advData->AofS[lev], VELX, divu, GetArrOfConstPtrs(fluxes[lev]), 0,
      GetArrOfConstPtrs(faces[lev]), 0, AMREX_SPACEDIM, AdvTypeVel_d.dataPtr(),
      geom[lev], -1.0, fluxes_are_area_weighted);
#endif

    // Under mesh mapping, the flux divergence was computed in uniform
    // (Xi) space with umac = u_bar, giving J . (physical divergence).
    // Restore the physical-space advection term by dividing by J.
    if (m_mesh_mapping) {
      for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        amrex::MultiFab::Divide(
          advData->AofS[lev], m_mesh_map->detJ_cc(lev), 0, VELX + n, 1, 0);
      }
    }
  }
}

void
PeleLM::updateVelocity(const std::unique_ptr<AdvanceAdvData>& advData)
{
  //----------------------------------------------------------------
  // Compute t^n divTau
  amrex::Vector<amrex::MultiFab> divtau;
  divtau.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    divtau.emplace_back(
      grids[lev], dmap[lev], AMREX_SPACEDIM, 0, amrex::MFInfo(), Factory(lev));
  }
  constexpr int use_density = 0;
  constexpr amrex::Real CrankNicholsonFactor = 0.5;
  computeDivTau(
    AmrOldTime, GetVecOfPtrs(divtau), use_density, CrankNicholsonFactor);

  //----------------------------------------------------------------
  // Get velocity forcing at half time including lagged grad P term
  constexpr int nGrow_force = 1;
  amrex::Vector<amrex::MultiFab> velForces;
  velForces.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    velForces.emplace_back(grids[lev], dmap[lev], AMREX_SPACEDIM, nGrow_force);
  }
  constexpr int add_gradP = 1;
  getVelForces(
    AmrHalfTime, GetVecOfPtrs(divtau), GetVecOfPtrs(velForces), nGrow_force,
    add_gradP);

  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get both old and new level_data
    auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
    auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);

    //----------------------------------------------------------------
    // Compute provisional new velocity
    // velForce holds: 1/\rho^{n+1/2} [(gravity+...)^{n+1/2} - \nabla pi^{n} +
    // 0.5 * divTau^{n}]
    auto const& state_old_ma = ldataOld_p->state.const_arrays();
    auto const& adv_aofs_ma = advData->AofS[lev].const_arrays();
    auto const& force_ma = velForces[lev].const_arrays();
    auto const& state_new_ma = ldataNew_p->state.arrays();
    amrex::ParallelFor(
      ldataOld_p->state, amrex::IntVect(0), AMREX_SPACEDIM,
      [state_old_ma, adv_aofs_ma, force_ma, state_new_ma,
       dt_loc =
         m_dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        amrex::Array4<amrex::Real> vel_new(state_new_ma[box_no], VELX);
        amrex::Array4<amrex::Real const> vel_old(state_old_ma[box_no], VELX);
        amrex::Array4<amrex::Real const> adv_aofs(adv_aofs_ma[box_no], VELX);
        vel_new(i, j, k, n) =
          vel_old(i, j, k, n) +
          dt_loc * (adv_aofs(i, j, k, n) + force_ma[box_no](i, j, k, n));
      });
  }
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::getScalarAdvForce(
  const std::unique_ptr<AdvanceAdvData>& advData,
  const std::unique_ptr<AdvanceDiffData>& diffData)
{

  const auto aux_diffuse_v = convertToDeviceVector(m_DiffTypeAux);
  const auto* aux_diffuse_d = aux_diffuse_v.dataPtr();
  auto const* leosparm = eos_parms.device_parm();

  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get t^{n} data pointer
    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);
    auto* ldataR_p = getLevelDataReactPtr(lev);

    auto const& state_ma = ldata_p->state.const_arrays();
    auto const& dn_ma = diffData->Dn[lev].const_arrays();
    auto const& adv_ma = advData->Forcing[lev].arrays();
    auto const& r_ma = ldataR_p->I_R.const_arrays();
    auto const& ext_ma = m_extSource[lev]->arrays();

    auto const& dn_aux_ma =
      (m_nAux > 0) ? diffData->Dn_aux[lev].const_arrays() : dn_ma;
    auto const& adv_aux_ma =
      (m_nAux > 0) ? advData->Forcing_aux[lev].arrays() : adv_ma;

    amrex::ParallelFor(
      advData->Forcing[lev],
      [state_ma, dn_ma, dn_aux_ma, r_ma, ext_ma, adv_ma, adv_aux_ma,
       aux_diffuse_d, leosparm, nAux = m_nAux, dp0dt = m_dp0dt,
       is_closed_ch = m_closed_chamber,
       do_react =
         m_do_react] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        amrex::Array4<const amrex::Real> rho(state_ma[box_no], DENSITY);
        amrex::Array4<const amrex::Real> rhoY(state_ma[box_no], FIRSTSPEC);
        amrex::Array4<const amrex::Real> T(state_ma[box_no], TEMP);
        amrex::Array4<const amrex::Real> dn(dn_ma[box_no], 0);
        amrex::Array4<const amrex::Real> ddn(dn_ma[box_no], NUM_SPECIES + 1);
        amrex::Array4<const amrex::Real> dn_aux(dn_aux_ma[box_no], 0);
        amrex::Array4<const amrex::Real> r(r_ma[box_no], 0);
        amrex::Array4<amrex::Real> extRhoY(ext_ma[box_no], FIRSTSPEC);
        amrex::Array4<amrex::Real> extRhoH(ext_ma[box_no], RHOH);
        amrex::Array4<amrex::Real> fY(adv_ma[box_no], 0);
        amrex::Array4<amrex::Real> fT(adv_ma[box_no], NUM_SPECIES);
        amrex::Array4<amrex::Real> fAux(adv_aux_ma[box_no], 0);
        buildAdvectionForcing(
          i, j, k, rho, rhoY, T, dn, ddn, r, extRhoY, extRhoH, dp0dt,
          is_closed_ch, do_react, fY, fT, fAux, dn_aux, aux_diffuse_d, nAux,
          leosparm);
      });
  }
  amrex::Gpu::streamSynchronize();

  // Fill forcing ghost cells
  if (advData->Forcing[0].nGrow() > 0) {
    fillpatch_forces(
      m_cur_time, GetVecOfPtrs(advData->Forcing), advData->Forcing[0].nGrow());
  }
  // Fill auxiliary forcing ghost cells
  if (m_nAux > 0) {
    if (advData->Forcing_aux[0].nGrow() > 0) {
      fillpatch_forces(
        m_cur_time, GetVecOfPtrs(advData->Forcing_aux),
        advData->Forcing_aux[0].nGrow());
    }
  }
}

void
PeleLM::computeScalarAdvTerms(const std::unique_ptr<AdvanceAdvData>& advData)
{

  //----------------------------------------------------------------
  // Get the BCRecs and AdvectionTypes
  auto bcRecSpec = fetchBCRecArray(FIRSTSPEC, NUM_SPECIES);
  auto bcRecSpec_d = convertToDeviceVector(bcRecSpec);
  auto AdvTypeSpec = fetchAdvTypeArray(FIRSTSPEC, NUM_SPECIES);
  auto AdvTypeSpec_d = convertToDeviceVector(AdvTypeSpec);
  auto bcRecTemp = fetchBCRecArray(TEMP, 1);
  auto bcRecTemp_d = convertToDeviceVector(bcRecTemp);
  auto AdvTypeTemp = fetchAdvTypeArray(TEMP, 1);
  auto AdvTypeTemp_d = convertToDeviceVector(AdvTypeTemp);
  auto bcRecRhoH = fetchBCRecArray(RHOH, 1);
  auto bcRecRhoH_d = convertToDeviceVector(bcRecRhoH);
  auto AdvTypeRhoH = fetchAdvTypeArray(RHOH, 1);
  auto AdvTypeRhoH_d = convertToDeviceVector(AdvTypeRhoH);
  auto bcRecAux = fetchBCRecAuxArray(0, m_nAux);
  auto bcRecAux_d = convertToDeviceVector(bcRecAux);
  auto AdvTypeAux = fetchAdvTypeAuxArray(0, m_nAux);
  auto AdvTypeAux_d = convertToDeviceVector(AdvTypeAux);
  //----------------------------------------------------------------
  // Create temporary containers
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes(
    finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes_aux(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      fluxes[lev][idim].define(
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
        dmap[lev], NUM_SPECIES + 1, 0, amrex::MFInfo(),
        Factory(lev)); // Species + RhoH
      if (m_nAux > 0) {
        fluxes_aux[lev][idim].define(
          amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
          dmap[lev], m_nAux, 0, amrex::MFInfo(),
          Factory(lev)); // auxiliary fields
      }
    }
  }

  //----------------------------------------------------------------
  // Loop over levels and get the fluxes
  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get level data ptr Old
    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);

    // Define edge state: Density + Species + RhoH + Temp
    constexpr int nGrow = 0;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> edgeState;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> edgeState_aux;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      edgeState[idim].define(
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
        dmap[lev], NUM_SPECIES + 3, nGrow, amrex::MFInfo(), Factory(lev));
      if (m_nAux > 0) {
        edgeState_aux[idim].define(
          amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
          dmap[lev], m_nAux, nGrow, amrex::MFInfo(), Factory(lev));
      }
    }

#ifdef PELE_USE_PLASMA
    //----------------------------------------------------------------
    // Assemble drift and mac velocities
    ionDriftAddUmac(lev, advData);
#endif

    //----------------------------------------------------------------
    // Get divU
    amrex::MultiFab divu(
      grids[lev], dmap[lev], 1, m_nGrowdivu, amrex::MFInfo(), Factory(lev));
    if (m_incompressible != 0) {
      divu.setVal(0.0);
    } else {
      amrex::MultiFab::Copy(divu, advData->mac_divu[lev], 0, 0, 1, m_nGrowdivu);
    }

    // Under mesh mapping the extrapolation is done in Xi with the
    // Xi-velocity u~ = umac / J_fc and the physical divU; the fluxes
    // u~.q are rescaled by J_fc below (see getExtrapUmac).
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> umac_xi;
    const auto umacE = getExtrapUmac(lev, advData, umac_xi);

    //----------------------------------------------------------------
#ifdef AMREX_USE_EB
    // Get EBFact & areafrac
    const auto& ebfact = EBFactory(lev);
    amrex::Array<const amrex::MultiCutFab*, AMREX_SPACEDIM> areafrac =
      ebfact.getAreaFrac();
#endif

    // Get the species edge state and advection term
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {

      amrex::Box const& bx = mfi.tilebox();
      AMREX_D_TERM(
        auto const& umac = umacE[0]->const_array(mfi);
        , auto const& vmac = umacE[1]->const_array(mfi);
        , auto const& wmac = umacE[2]->const_array(mfi);)
      AMREX_D_TERM(
        auto const& fx = fluxes[lev][0].array(mfi, 0);
        , auto const& fy = fluxes[lev][1].array(mfi, 0);
        , auto const& fz = fluxes[lev][2].array(mfi, 0);)
      AMREX_D_TERM(
        auto const& edgex = edgeState[0].array(mfi, 1);
        , auto const& edgey = edgeState[1].array(mfi, 1);
        , auto const& edgez = edgeState[2].array(mfi, 1);)
      auto const& divu_arr = divu.const_array(mfi);
      auto const& rhoY_arr = ldata_p->state.const_array(mfi, FIRSTSPEC);
      auto const& force_arr = advData->Forcing[lev].const_array(mfi, 0);

#ifdef PELE_USE_PLASMA
      // Uncharged species all at once
      constexpr bool is_velocity = false;
      constexpr bool fluxes_are_area_weighted = false;
      constexpr bool knownEdgeState = false;
      HydroUtils::ComputeFluxesOnBoxFromState(
        bx, NUM_SPECIES - NUM_IONS, mfi, rhoY_arr, AMREX_D_DECL(fx, fy, fz),
        AMREX_D_DECL(edgex, edgey, edgez), knownEdgeState,
        AMREX_D_DECL(umac, vmac, wmac), divu_arr, force_arr, geom[lev], m_dt,
        bcRecSpec, bcRecSpec_d.dataPtr(), AdvTypeSpec_d.dataPtr(),
#ifdef AMREX_USE_EB
        ebfact,
        (m_useEBinflow != 0)
          ? getEBState(mfi, lev, FIRSTSPEC, NUM_SPECIES - NUM_IONS, AmrOldTime)
              .const_array()
          : amrex::Array4<amrex::Real const>{},
#endif
        m_Godunov_ppm, m_Godunov_ForceInTrans, is_velocity,
        fluxes_are_area_weighted, m_advection_type, m_Godunov_ppm_limiter);

      // Ions one by one
      for (int n = 0; n < NUM_IONS; ++n) {
        const int ion_idx = NUM_SPECIES - NUM_IONS + n;
        auto bcRecIons = fetchBCRecArray(FIRSTSPEC + ion_idx, 1);
        auto bcRecIons_d = convertToDeviceVector(bcRecIons);
        auto AdvTypeIons = fetchAdvTypeArray(FIRSTSPEC + ion_idx, 1);
        auto AdvTypeIons_d = convertToDeviceVector(AdvTypeIons);
        AMREX_D_TERM(
          auto const& udrift = advData->uDrift[lev][0].const_array(mfi, n);
          , auto const& vdrift = advData->uDrift[lev][1].const_array(mfi, n);
          , auto const& wdrift = advData->uDrift[lev][2].const_array(mfi, n);)
        AMREX_D_TERM(
          auto const& fx_ions = fluxes[lev][0].array(mfi, ion_idx);
          , auto const& fy_ions = fluxes[lev][1].array(mfi, ion_idx);
          , auto const& fz_ions = fluxes[lev][2].array(mfi, ion_idx);)
        AMREX_D_TERM(
          auto const& edgex_ions = edgeState[0].array(mfi, 1 + ion_idx);
          , auto const& edgey_ions = edgeState[1].array(mfi, 1 + ion_idx);
          , auto const& edgez_ions = edgeState[2].array(mfi, 1 + ion_idx);)
        auto const& rhoYions_arr =
          ldata_p->state.const_array(mfi, FIRSTSPEC + ion_idx);
        auto const& forceions_arr =
          advData->Forcing[lev].const_array(mfi, ion_idx);
        HydroUtils::ComputeFluxesOnBoxFromState(
          bx, 1, mfi, rhoYions_arr, AMREX_D_DECL(fx_ions, fy_ions, fz_ions),
          AMREX_D_DECL(edgex_ions, edgey_ions, edgez_ions), knownEdgeState,
          AMREX_D_DECL(udrift, vdrift, wdrift), divu_arr, forceions_arr,
          geom[lev], m_dt, bcRecIons, bcRecIons_d.dataPtr(),
          AdvTypeIons_d.dataPtr(),
#ifdef AMREX_USE_EB
          ebfact,
          (m_useEBinflow != 0)
            ? getEBState(mfi, lev, FIRSTSPEC + ion_idx, 1, AmrOldTime)
                .const_array()
            : amrex::Array4<amrex::Real const>{},
#endif
          m_Godunov_ppm, m_Godunov_ForceInTrans, is_velocity,
          fluxes_are_area_weighted, m_advection_type, m_Godunov_ppm_limiter);
      }
#else
      constexpr bool is_velocity = false;
      constexpr bool fluxes_are_area_weighted = false;
      constexpr bool knownEdgeState = false;
      HydroUtils::ComputeFluxesOnBoxFromState(
        bx, NUM_SPECIES, mfi, rhoY_arr, AMREX_D_DECL(fx, fy, fz),
        AMREX_D_DECL(edgex, edgey, edgez), knownEdgeState,
        AMREX_D_DECL(umac, vmac, wmac), divu_arr, force_arr, geom[lev], m_dt,
        bcRecSpec, bcRecSpec_d.dataPtr(), AdvTypeSpec_d.dataPtr(),
#ifdef AMREX_USE_EB
        ebfact,
        (m_useEBinflow != 0)
          ? getEBState(mfi, lev, FIRSTSPEC, NUM_SPECIES, AmrOldTime)
              .const_array()
          : amrex::Array4<amrex::Real const>{},
#endif
        m_Godunov_ppm != 0, m_Godunov_ForceInTrans != 0, is_velocity,
        fluxes_are_area_weighted, m_advection_type, m_Godunov_ppm_limiter);
#endif
    }
    // get the edge state for auxiliary components
    if (m_nAux > 0) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(ldata_p->auxiliaries, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        AMREX_D_TERM(
          auto const& umac = umacE[0]->const_array(mfi);
          , auto const& vmac = umacE[1]->const_array(mfi);
          , auto const& wmac = umacE[2]->const_array(mfi);)
        AMREX_D_TERM(
          auto const& fx = fluxes_aux[lev][0].array(mfi, 0);
          , auto const& fy = fluxes_aux[lev][1].array(mfi, 0);
          , auto const& fz = fluxes_aux[lev][2].array(mfi, 0);)
        AMREX_D_TERM(
          auto const& edgex = edgeState_aux[0].array(mfi, 0);
          , auto const& edgey = edgeState_aux[1].array(mfi, 0);
          , auto const& edgez = edgeState_aux[2].array(mfi, 0);)
        auto const& divu_arr = divu.const_array(mfi);
        auto const& aux_arr = ldata_p->auxiliaries.const_array(mfi, 0);
        auto const& force_arr = advData->Forcing_aux[lev].const_array(mfi, 0);

        constexpr bool is_velocity = false;
        constexpr bool fluxes_are_area_weighted = false;
        constexpr bool knownEdgeState = false;

        HydroUtils::ComputeFluxesOnBoxFromState(
          bx, m_nAux, mfi, aux_arr, AMREX_D_DECL(fx, fy, fz),
          AMREX_D_DECL(edgex, edgey, edgez), knownEdgeState,
          AMREX_D_DECL(umac, vmac, wmac), divu_arr, force_arr, geom[lev], m_dt,
          bcRecAux, bcRecAux_d.dataPtr(), AdvTypeAux_d.dataPtr(),
#ifdef AMREX_USE_EB
          ebfact,
#endif
          m_Godunov_ppm != 0, m_Godunov_ForceInTrans != 0, is_velocity,
          fluxes_are_area_weighted, m_advection_type, m_Godunov_ppm_limiter);
      }
      // Zero out fluxes for non-advected auxiliaries
      for (int n = 0; n < m_nAux; ++n) {
        if (m_aux_advect[n] == 0) {
          for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            fluxes_aux[lev][idim].setVal(0.0, n, 1);
          }
        }
      }
    }

    // Get edge density by summing over the species
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {

      amrex::Box const& bx = mfi.tilebox();

#ifdef AMREX_USE_EB
      auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
#endif

      // Edge states
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const amrex::Box& ebx = amrex::surroundingNodes(bx, idim);
        auto const& rho_ed = edgeState[idim].array(mfi, 0);
        auto const& rhoY_ed = edgeState[idim].array(mfi, 1);
#ifdef AMREX_USE_EB
        if (flagfab.getType(ebx) == amrex::FabType::covered) { // Covered boxes
          amrex::ParallelFor(
            ebx, [rho_ed] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              rho_ed(i, j, k) = 0.0;
            });
        } else if (
          flagfab.getType(ebx) != amrex::FabType::regular) { // EB containing
                                                             // boxes
          const auto& afrac = areafrac[idim]->array(mfi);
          amrex::ParallelFor(
            ebx, [rho_ed, afrac,
                  rhoY_ed] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              rho_ed(i, j, k) = 0.0;
              if (afrac(i, j, k) > 0.0) { // Uncovered faces
                pele::physics::PhysicsType::eos_type::RY2R(
                  rhoY_ed.cellData(i, j, k), rho_ed(i, j, k));
              }
            });
        } else // Regular boxes
#endif
        {
          amrex::ParallelFor(
            ebx,
            [rhoY_ed, rho_ed] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              pele::physics::PhysicsType::eos_type::RY2R(
                rhoY_ed.cellData(i, j, k), rho_ed(i, j, k));
            });
        }
      }
    }

    // Get the edge temperature
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      amrex::Box const& bx = mfi.tilebox();
      AMREX_D_TERM(
        auto const& umac = umacE[0]->const_array(mfi);
        , auto const& vmac = umacE[1]->const_array(mfi);
        , auto const& wmac = umacE[2]->const_array(mfi);)
      AMREX_D_TERM(
        auto const& fx = fluxes[lev][0].array(mfi, NUM_SPECIES);
        , // Put temp fluxes in place of rhoH
        auto const& fy = fluxes[lev][1].array(mfi, NUM_SPECIES);
        , // will be overwritten later
        auto const& fz = fluxes[lev][2].array(mfi, NUM_SPECIES);)
      AMREX_D_TERM(
        auto const& edgex = edgeState[0].array(mfi, NUM_SPECIES + 2);
        , auto const& edgey = edgeState[1].array(mfi, NUM_SPECIES + 2);
        , auto const& edgez = edgeState[2].array(mfi, NUM_SPECIES + 2);)
      auto const& divu_arr = divu.const_array(mfi);
      auto const& temp_arr = ldata_p->state.const_array(mfi, TEMP);
      auto const& force_arr =
        advData->Forcing[lev].const_array(mfi, NUM_SPECIES);
      constexpr bool is_velocity = false;
      constexpr bool fluxes_are_area_weighted = false;
      constexpr bool knownEdgeState = false;
      HydroUtils::ComputeFluxesOnBoxFromState(
        bx, 1, mfi, temp_arr, AMREX_D_DECL(fx, fy, fz),
        AMREX_D_DECL(edgex, edgey, edgez), knownEdgeState,
        AMREX_D_DECL(umac, vmac, wmac), divu_arr, force_arr, geom[lev], m_dt,
        bcRecTemp, bcRecTemp_d.dataPtr(), AdvTypeTemp_d.dataPtr(),
#ifdef AMREX_USE_EB
        ebfact,
        (m_useEBinflow != 0)
          ? getEBState(mfi, lev, TEMP, 1, AmrOldTime).const_array()
          : amrex::Array4<amrex::Real const>{},
#endif
        m_Godunov_ppm != 0, m_Godunov_ForceInTrans != 0, is_velocity,
        fluxes_are_area_weighted, m_advection_type, m_Godunov_ppm_limiter);
    }

    // Get the edge RhoH states
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      amrex::Box const& bx = mfi.tilebox();
      auto const* leosparm = eos_parms.device_parm();

#ifdef AMREX_USE_EB
      auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
#endif

      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const amrex::Box& ebx = amrex::surroundingNodes(bx, idim);
        auto const& rho = edgeState[idim].const_array(mfi, 0);
        auto const& rhoY = edgeState[idim].const_array(mfi, 1);
        auto const& T = edgeState[idim].const_array(mfi, NUM_SPECIES + 2);
        auto const& rhoHm = edgeState[idim].array(mfi, NUM_SPECIES + 1);
#ifdef AMREX_USE_EB
        if (flagfab.getType(ebx) == amrex::FabType::covered) { // Covered boxes
          amrex::ParallelFor(
            ebx, [rhoHm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              rhoHm(i, j, k) = 0.0;
            });
        } else if (
          flagfab.getType(ebx) != amrex::FabType::regular) { // EB containing
                                                             // boxes
          const auto& afrac = areafrac[idim]->array(mfi);
          amrex::ParallelFor(
            ebx, [rhoHm, afrac, rhoY, T, rho,
                  leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              if (afrac(i, j, k) <= 0.0) { // Covered faces
                rhoHm(i, j, k) = 0.0;
              } else {
                getRHmixGivenTY(i, j, k, rho, rhoY, T, rhoHm, leosparm);
              }
            });
        } else // Regular boxes
#endif
        {
          amrex::ParallelFor(
            ebx, [rho, rhoY, T, rhoHm,
                  leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              getRHmixGivenTY(i, j, k, rho, rhoY, T, rhoHm, leosparm);
            });
        }
      }
    }

    // Finally get the RhoH advection term
    // Pass the Temp forces again here, but they aren't used.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      amrex::Box const& bx = mfi.tilebox();
      AMREX_D_TERM(
        auto const& umac = umacE[0]->const_array(mfi);
        , auto const& vmac = umacE[1]->const_array(mfi);
        , auto const& wmac = umacE[2]->const_array(mfi);)
      AMREX_D_TERM(
        auto const& fx = fluxes[lev][0].array(mfi, NUM_SPECIES);
        , auto const& fy = fluxes[lev][1].array(mfi, NUM_SPECIES);
        , auto const& fz = fluxes[lev][2].array(mfi, NUM_SPECIES);)
      AMREX_D_TERM(
        auto const& edgex = edgeState[0].array(mfi, NUM_SPECIES + 1);
        , auto const& edgey = edgeState[1].array(mfi, NUM_SPECIES + 1);
        , auto const& edgez = edgeState[2].array(mfi, NUM_SPECIES + 1);)
      auto const& divu_arr = divu.const_array(mfi);
      auto const& rhoh_arr = ldata_p->state.const_array(mfi, RHOH);
      auto const& force_arr =
        advData->Forcing[lev].const_array(mfi, NUM_SPECIES);
      constexpr bool is_velocity = false;
      constexpr bool fluxes_are_area_weighted = false;
      constexpr bool knownEdgeState = true;
      HydroUtils::ComputeFluxesOnBoxFromState(
        bx, 1, mfi, rhoh_arr, AMREX_D_DECL(fx, fy, fz),
        AMREX_D_DECL(edgex, edgey, edgez), knownEdgeState,
        AMREX_D_DECL(umac, vmac, wmac), divu_arr, force_arr, geom[lev], m_dt,
        bcRecRhoH, bcRecRhoH_d.dataPtr(), AdvTypeRhoH_d.dataPtr(),
#ifdef AMREX_USE_EB
        ebfact,
        (m_useEBinflow != 0)
          ? getEBState(mfi, lev, RHOH, 1, AmrOldTime).const_array()
          : amrex::Array4<amrex::Real const>{},
#endif
        m_Godunov_ppm != 0, m_Godunov_ForceInTrans != 0, is_velocity,
        fluxes_are_area_weighted, m_advection_type, m_Godunov_ppm_limiter);
    }
    scaleMappedAdvFluxes(lev, fluxes[lev], 0, NUM_SPECIES + 1);
    if (m_nAux > 0) {
      scaleMappedAdvFluxes(lev, fluxes_aux[lev], 0, m_nAux);
    }
#ifdef AMREX_USE_EB
    EB_set_covered_faces(GetArrOfPtrs(fluxes[lev]), 0.);
#endif
  }

  //----------------------------------------------------------------
  // Average down fluxes to ensure C/F consistency
  for (int lev = finest_level; lev > 0; --lev) {
#ifdef AMREX_USE_EB
    EB_average_down_faces(
      GetArrOfConstPtrs(fluxes[lev]), GetArrOfPtrs(fluxes[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
    if (m_nAux > 0) {
      EB_average_down_faces(
        GetArrOfConstPtrs(fluxes_aux[lev]), GetArrOfPtrs(fluxes_aux[lev - 1]),
        refRatio(lev - 1), geom[lev - 1]);
    }
#else
    average_down_faces(
      GetArrOfConstPtrs(fluxes[lev]), GetArrOfPtrs(fluxes[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
    if (m_nAux > 0) {
      average_down_faces(
        GetArrOfConstPtrs(fluxes_aux[lev]), GetArrOfPtrs(fluxes_aux[lev - 1]),
        refRatio(lev - 1), geom[lev - 1]);
    }
#endif
  }

  //----------------------------------------------------------------
  // If balances are required, compute face domain integrals
  // using level 0 since we've averaged down the fluxes already
  if (m_sdcIter == m_nSDCmax) {
    if (m_do_massBalance != 0) {
      addMassFluxes(GetArrOfConstPtrs(fluxes[0]), geom[0]);
    }
    if (m_do_energyBalance != 0) {
      addRhoHFluxes(GetArrOfConstPtrs(fluxes[0]), geom[0]);
    }
    if (m_do_speciesBalance != 0) {
      addRhoYFluxes(GetArrOfConstPtrs(fluxes[0]), geom[0]);
    }

    if (m_do_patch_mfr != 0) {
      addRhoYFluxesPatch(GetArrOfConstPtrs(fluxes[0]), geom[0]);
    }
  }
  // Compute face domain integral for U at every SDC iteration
  addUmacFluxes(advData, geom[0]);

#ifdef PELE_USE_PLASMA
  if (m_do_extraEFdiags) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      for (int n = 0; n < NUM_IONS; ++n) {
        const int spec_idx = NUM_SPECIES - NUM_IONS + n;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> ionFlux;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          ionFlux[idim].reset(new amrex::MultiFab(
            fluxes[lev][idim], amrex::make_alias, spec_idx, 1));
        }
        average_face_to_cellcenter(
          *m_ionsFluxes[lev], n * AMREX_SPACEDIM, GetArrOfConstPtrs(ionFlux));
      }
    }
  }
#endif

  //----------------------------------------------------------------
  // Fluxes divergence to get the scalars advection term
  auto AdvTypeAll =
    fetchAdvTypeArray(FIRSTSPEC, NUM_SPECIES + 1); // Species+RhoH
  auto AdvTypeAll_d = convertToDeviceVector(AdvTypeAll);
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab divu(
      grids[lev], dmap[lev], 1, m_nGrowdivu, amrex::MFInfo(), Factory(lev));
    if (m_incompressible != 0) {
      divu.setVal(0.0);
    } else {
      const amrex::Real time = getTime(lev, AmrOldTime);
      fillpatch_divu(lev, time, divu, m_nGrowdivu);
    }
    if (m_mesh_mapping) {
      amrex::MultiFab::Multiply(
        divu, m_mesh_map->detJ_cc(lev), 0, 0, 1, m_nGrowdivu);
    }

    bool fluxes_are_area_weighted = false;
#ifdef AMREX_USE_EB
    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);
    //----------------------------------------------------------------
    // Use a temporary MF to hold divergence before redistribution
    constexpr int nGrow_divTmp = 3;
    amrex::MultiFab divTmp(
      grids[lev], dmap[lev], NUM_SPECIES + 1, nGrow_divTmp, amrex::MFInfo(),
      EBFactory(lev));
    divTmp.setVal(0.0);
    if (m_useEBinflow != 0) {
      advFluxDivergence(
        lev, divTmp, 0, divu, GetArrOfConstPtrs(fluxes[lev]), 0,
        GetArrOfConstPtrs(fluxes[lev]),
        0, // This will not be used since none of rhoY/rhoH in convective
        getEBState(lev, VELX, AMREX_SPACEDIM, AmrOldTime).get(),
        getEBState(lev, FIRSTSPEC, NUM_SPECIES + 1, AmrOldTime).get(),
        NUM_SPECIES + 1, AdvTypeAll_d.dataPtr(), geom[lev], -1.0,
        fluxes_are_area_weighted);
    } else {
      advFluxDivergence(
        lev, divTmp, 0, divu, GetArrOfConstPtrs(fluxes[lev]), 0,
        GetArrOfConstPtrs(fluxes[lev]),
        0, // This will not be used since none of rhoY/rhoH in convective
        NUM_SPECIES + 1, AdvTypeAll_d.dataPtr(), geom[lev], -1.0,
        fluxes_are_area_weighted);
    }

    divTmp.FillBoundary(geom[lev].periodicity());

    // Need separate redistribution of rhoYs/rhoH
    redistributeAofS(
      lev, m_dt, divTmp, 0, advData->AofS[lev], FIRSTSPEC, ldata_p->state,
      FIRSTSPEC, NUM_SPECIES, bcRecSpec_d.dataPtr(), geom[lev]);

    redistributeAofS(
      lev, m_dt, divTmp, NUM_SPECIES, advData->AofS[lev], RHOH, ldata_p->state,
      RHOH, 1, bcRecRhoH_d.dataPtr(), geom[lev]);

    EB_set_covered(advData->AofS[lev], 0.0);

    // repeat process for auxiliaries
    if (m_nAux > 0) {
      amrex::MultiFab divTmp_aux(
        grids[lev], dmap[lev], m_nAux, nGrow_divTmp, amrex::MFInfo(),
        EBFactory(lev));
      divTmp_aux.setVal(0.0);
      advFluxDivergence(
        lev, divTmp_aux, 0, divu, GetArrOfConstPtrs(fluxes_aux[lev]), 0,
        GetArrOfConstPtrs(fluxes_aux[lev]), 0, m_nAux, AdvTypeAux_d.dataPtr(),
        geom[lev], -1.0, fluxes_are_area_weighted);

      divTmp_aux.FillBoundary(geom[lev].periodicity());

      redistributeAofS(
        lev, m_dt, divTmp_aux, 0, advData->AofS_aux[lev], 0,
        ldata_p->auxiliaries, 0, m_nAux, bcRecAux_d.dataPtr(), geom[lev]);

      EB_set_covered(advData->AofS_aux[lev], 0.0);
    }
#else
    //----------------------------------------------------------------
    // Otherwise go directly into AofS
    advFluxDivergence(
      lev, advData->AofS[lev], FIRSTSPEC, divu, GetArrOfConstPtrs(fluxes[lev]),
      0, GetArrOfConstPtrs(fluxes[lev]),
      0, // This will not be used since none of rhoY/rhoH in convective
      NUM_SPECIES + 1, AdvTypeAll_d.dataPtr(), geom[lev], -1.0,
      fluxes_are_area_weighted);
    // repeat for auxiliaries
    if (m_nAux > 0) {
      advFluxDivergence(
        lev, advData->AofS_aux[lev], 0, divu,
        GetArrOfConstPtrs(fluxes_aux[lev]), 0,
        GetArrOfConstPtrs(fluxes_aux[lev]), 0, m_nAux, AdvTypeAux_d.dataPtr(),
        geom[lev], -1.0, fluxes_are_area_weighted);
    }
#endif

    // Under mesh mapping, the flux divergence was computed in uniform
    // (Xi) space with umac = u_bar, giving J . (physical divergence).
    // Restore physical-space AofS for species (FIRSTSPEC..FIRSTSPEC+NSPEC-1)
    // and enthalpy (RHOH) by dividing by J.  The density component of AofS
    // is re-derived from the species sum below, so it inherits the scaling
    // automatically.
    if (m_mesh_mapping) {
      for (int n = 0; n < NUM_SPECIES; ++n) {
        amrex::MultiFab::Divide(
          advData->AofS[lev], m_mesh_map->detJ_cc(lev), 0, FIRSTSPEC + n, 1, 0);
      }
      amrex::MultiFab::Divide(
        advData->AofS[lev], m_mesh_map->detJ_cc(lev), 0, RHOH, 1, 0);
      if (m_nAux > 0) {
        for (int n = 0; n < m_nAux; ++n) {
          amrex::MultiFab::Divide(
            advData->AofS_aux[lev], m_mesh_map->detJ_cc(lev), 0, n, 1, 0);
        }
      }
    }
  }

  //----------------------------------------------------------------
  // Sum over the species AofS to get the density advection term
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto const& aofsma = advData->AofS[lev].arrays();
    amrex::ParallelFor(
      advData->AofS[lev],
      [aofsma] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        pele::physics::PhysicsType::eos_type::RY2R(
          aofsma[box_no].cellData(i, j, k), aofsma[box_no](i, j, k, DENSITY),
          FIRSTSPEC);
      });
  }
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::updateDensity(const std::unique_ptr<AdvanceAdvData>& advData)
{
  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get MultiArrays
    auto const& sma_o = getLevelDataPtr(lev, AmrOldTime)->state.arrays();
    auto const& sma_n = getLevelDataPtr(lev, AmrNewTime)->state.arrays();
    auto const& aofsma = advData->AofS[lev].const_arrays();
    auto const& extma = m_extSource[lev]->const_arrays();
    const auto dt = m_dt;

    amrex::ParallelFor(
      advData->AofS[lev], [sma_o, sma_n, aofsma, extma, dt] AMREX_GPU_DEVICE(
                            int box_no, int i, int j, int k) noexcept {
        sma_n[box_no](i, j, k, DENSITY) =
          sma_o[box_no](i, j, k, DENSITY) +
          dt * (aofsma[box_no](i, j, k, DENSITY) +
                extma[box_no](i, j, k, DENSITY));
      });
  }
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::computePassiveAdvTerms(
  const std::unique_ptr<AdvanceAdvData>& advData,
  const int state_comp,
  const int ncomp)
{
  //----------------------------------------------------------------
  // Get the BCRecs and AdvectionTypes
  auto bcRecPass = fetchBCRecArray(state_comp, ncomp);
  auto bcRecPass_d = convertToDeviceVector(bcRecPass);
  auto AdvTypePass = fetchAdvTypeArray(state_comp, ncomp);
  auto AdvTypePass_d = convertToDeviceVector(AdvTypePass);

  //----------------------------------------------------------------
  // Create temporary containers
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes(
    finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> edgeState(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      fluxes[lev][idim].define(
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
        dmap[lev], ncomp, 0, amrex::MFInfo(), Factory(lev));
      edgeState[lev][idim].define(
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
        dmap[lev], ncomp, 0, amrex::MFInfo(), Factory(lev));
    }
  }

  //----------------------------------------------------------------
  // Loop over levels and get the fluxes
  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get level data ptr Old
    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);

    //----------------------------------------------------------------
    // Get divU
    amrex::MultiFab divu(
      grids[lev], dmap[lev], 1, m_nGrowdivu, amrex::MFInfo(), Factory(lev));
    if (m_incompressible != 0) {
      divu.setVal(0.0);
    } else {
      amrex::MultiFab::Copy(divu, advData->mac_divu[lev], 0, 0, 1, m_nGrowdivu);
    }

    // Under mesh mapping the extrapolation is done in Xi with the
    // Xi-velocity u~ = umac / J_fc and the physical divU; the fluxes
    // u~.q are rescaled by J_fc below (see getExtrapUmac).
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> umac_xi;
    const auto umacE = getExtrapUmac(lev, advData, umac_xi);

    //----------------------------------------------------------------
#ifdef AMREX_USE_EB
    // Get EBFact
    const auto& ebfact = EBFactory(lev);
#endif

    // Get the passive variables edge state and advection term
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      amrex::Box const& bx = mfi.tilebox();
      AMREX_D_TERM(
        auto const& umac = umacE[0]->const_array(mfi);
        , auto const& vmac = umacE[1]->const_array(mfi);
        , auto const& wmac = umacE[2]->const_array(mfi);)
      AMREX_D_TERM(
        auto const& fx = fluxes[lev][0].array(mfi, 0);
        , auto const& fy = fluxes[lev][1].array(mfi, 0);
        , auto const& fz = fluxes[lev][2].array(mfi, 0);)
      AMREX_D_TERM(
        auto const& edgex = edgeState[lev][0].array(mfi, 0);
        , auto const& edgey = edgeState[lev][1].array(mfi, 0);
        , auto const& edgez = edgeState[lev][2].array(mfi, 0);)
      auto const& divu_arr = divu.const_array(mfi);
      auto const& pass_arr = ldata_p->state.const_array(mfi, state_comp);
      // TODO: Find way to include diffusive forces for passive scalars that
      // diffuse
      auto const& force_arr = m_extSource[lev]->const_array(mfi, state_comp);
      constexpr bool is_velocity = false;
      constexpr bool fluxes_are_area_weighted = false;
      constexpr bool knownEdgeState = false;
      HydroUtils::ComputeFluxesOnBoxFromState(
        bx, ncomp, mfi, pass_arr, AMREX_D_DECL(fx, fy, fz),
        AMREX_D_DECL(edgex, edgey, edgez), knownEdgeState,
        AMREX_D_DECL(umac, vmac, wmac), divu_arr, force_arr, geom[lev], m_dt,
        bcRecPass, bcRecPass_d.dataPtr(), AdvTypePass_d.dataPtr(),
#ifdef AMREX_USE_EB
        ebfact,
#endif
        m_Godunov_ppm != 0, m_Godunov_ForceInTrans != 0, is_velocity,
        fluxes_are_area_weighted, m_advection_type, m_Godunov_ppm_limiter);
    }
    scaleMappedAdvFluxes(lev, fluxes[lev], 0, ncomp);
#ifdef AMREX_USE_EB
    EB_set_covered_faces(GetArrOfPtrs(fluxes[lev]), 0.);
#endif
  }
  //----------------------------------------------------------------
  // Average down fluxes and edge state to ensure C/F consistency
  for (int lev = finest_level; lev > 0; --lev) {
#ifdef AMREX_USE_EB
    EB_average_down_faces(
      GetArrOfConstPtrs(fluxes[lev]), GetArrOfPtrs(fluxes[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
    EB_average_down_faces(
      GetArrOfConstPtrs(edgeState[lev]), GetArrOfPtrs(edgeState[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
#else
    average_down_faces(
      GetArrOfConstPtrs(fluxes[lev]), GetArrOfPtrs(fluxes[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
    average_down_faces(
      GetArrOfConstPtrs(edgeState[lev]), GetArrOfPtrs(edgeState[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
#endif
  }
  //----------------------------------------------------------------
  // Fluxes divergence to get the scalars advection term
  auto AdvTypeAll = fetchAdvTypeArray(state_comp, ncomp);
  auto AdvTypeAll_d = convertToDeviceVector(AdvTypeAll);
  for (int lev = 0; lev <= finest_level; ++lev) {

    amrex::MultiFab divu(
      grids[lev], dmap[lev], 1, m_nGrowdivu, amrex::MFInfo(), Factory(lev));
    if (m_incompressible != 0) {
      divu.setVal(0.0);
    } else {
      const amrex::Real time = getTime(lev, AmrOldTime);
      fillpatch_divu(lev, time, divu, m_nGrowdivu);
    }
    // Fluxes are J.u~.q: the non-conservative correction needs J.divU
    if (m_mesh_mapping) {
      amrex::MultiFab::Multiply(
        divu, m_mesh_map->detJ_cc(lev), 0, 0, 1, m_nGrowdivu);
    }

    constexpr bool fluxes_are_area_weighted = false;
#ifdef AMREX_USE_EB
    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);
    //----------------------------------------------------------------
    // Use a temporary MF to hold divergence before redistribution
    constexpr int nGrow_divTmp = 3;
    amrex::MultiFab divTmp(
      grids[lev], dmap[lev], ncomp, nGrow_divTmp, amrex::MFInfo(),
      EBFactory(lev));
    divTmp.setVal(0.0);
    advFluxDivergence(
      lev, divTmp, 0, divu, GetArrOfConstPtrs(fluxes[lev]), 0,
      GetArrOfConstPtrs(edgeState[lev]), 0, ncomp, AdvTypeAll_d.dataPtr(),
      geom[lev], -1.0, fluxes_are_area_weighted);

    divTmp.FillBoundary(geom[lev].periodicity());

    redistributeAofS(
      lev, m_dt, divTmp, 0, advData->AofS[lev], state_comp, ldata_p->state,
      state_comp, ncomp, bcRecPass_d.dataPtr(), geom[lev]);
#else
    //----------------------------------------------------------------
    // Otherwise go directly into AofS
    advFluxDivergence(
      lev, advData->AofS[lev], state_comp, divu, GetArrOfConstPtrs(fluxes[lev]),
      0, GetArrOfConstPtrs(edgeState[lev]), 0, ncomp, AdvTypeAll_d.dataPtr(),
      geom[lev], -1.0, fluxes_are_area_weighted);
#endif

    // Xi-space divergence is J.(physical divergence): restore physical units
    if (m_mesh_mapping) {
      for (int n = 0; n < ncomp; ++n) {
        amrex::MultiFab::Divide(
          advData->AofS[lev], m_mesh_map->detJ_cc(lev), 0, state_comp + n, 1,
          0);
      }
    }
  }
  // TODO: This assumes passive variables have no diffusive fluxes
  updateScalarComp(advData, state_comp, ncomp);
}

void
PeleLM::updateScalarComp(
  const std::unique_ptr<AdvanceAdvData>& advData,
  const int state_comp,
  const int ncomp)
{
  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get level data ptr
    auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
    auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);

    auto const& state_old_ma = ldataOld_p->state.const_arrays();
    auto const& adv_aofs_ma = advData->AofS[lev].const_arrays();
    auto const& ext_ma = m_extSource[lev]->const_arrays();
    auto const& state_new_ma = ldataNew_p->state.arrays();

    amrex::ParallelFor(
      ldataOld_p->state, amrex::IntVect(0), ncomp,
      [state_old_ma, adv_aofs_ma, ext_ma, state_new_ma, state_comp,
       dt =
         m_dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        amrex::Array4<amrex::Real> state_new_arr(
          state_new_ma[box_no], state_comp);
        amrex::Array4<amrex::Real const> state_old_arr(
          state_old_ma[box_no], state_comp);
        amrex::Array4<amrex::Real const> adv_aofs_arr(
          adv_aofs_ma[box_no], state_comp);
        amrex::Array4<amrex::Real const> ext_arr(ext_ma[box_no], state_comp);
        state_new_arr(i, j, k, n) =
          state_old_arr(i, j, k, n) +
          dt * (adv_aofs_arr(i, j, k, n) + ext_arr(i, j, k, n));
      });
  }
  amrex::Gpu::streamSynchronize();
  averageDown(AmrNewTime, state_comp, ncomp);
}

amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>
PeleLM::getExtrapUmac(
  int a_lev,
  const std::unique_ptr<AdvanceAdvData>& advData,
  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& a_scratch)
{
  amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM> r;
  if (!m_mesh_mapping) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      r[idim] = &advData->umac[a_lev][idim];
    }
    return r;
  }

  // umac = J.u~ after the MAC projection (macProject); the Godunov
  // extrapolation advances the state in Xi and needs the Xi-velocity
  // u~ = u_d / fac_d = umac / J_fc on each face.  The metric MultiFabs
  // carry at least as many ghost faces as umac.
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    const amrex::MultiFab& um = advData->umac[a_lev][idim];
    a_scratch[idim].define(
      um.boxArray(), um.DistributionMap(), 1, um.nGrowVect(), amrex::MFInfo(),
      um.Factory());
    amrex::MultiFab::Copy(a_scratch[idim], um, 0, 0, 1, um.nGrowVect());
    amrex::MultiFab::Divide(
      a_scratch[idim], m_mesh_map->detJ_fc(a_lev, idim), 0, 0, 1,
      um.nGrowVect());
    r[idim] = &a_scratch[idim];
  }
  return r;
}

void
PeleLM::scaleMappedAdvFluxes(
  int a_lev,
  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& a_fluxes,
  int a_comp,
  int a_ncomp)
{
  if (!m_mesh_mapping) {
    return;
  }
  // Fluxes were formed with u~: F = u~.q.  Multiply by J_fc so that the Xi
  // divergence of F is J.(physical divergence), the scaling advFluxDivergence
  // and the J.divU correction assume; AofS is divided by J afterwards.
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    for (int n = 0; n < a_ncomp; ++n) {
      amrex::MultiFab::Multiply(
        a_fluxes[idim], m_mesh_map->detJ_fc(a_lev, idim), 0, a_comp + n, 1, 0);
    }
  }
}
