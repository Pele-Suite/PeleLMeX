
#include <PeleLMeX.H>
#include <PeleLMeX_Utils.H>
#include <PeleLMeX_K.H>
#include <PeleLMeX_DiffusionOp.H>
#include <AMReX_PlotFileUtil.H>
#include <memory>

#ifdef AMREX_USE_EB
#include <AMReX_EB_utils.H>
#include <AMReX_EBFArrayBox.H>
#endif

DiffusionOp*
PeleLM::getDiffusionOp()
{
  if (!m_diffusion_op) {
    m_diffusion_op = std::make_unique<DiffusionOp>(this);
  }
  return m_diffusion_op.get();
}

DiffusionOp*
PeleLM::getMCDiffusionOp(const int ncomp)
{
  if (!m_mcdiffusion_op || m_mcdiffusion_op->m_ncomp != ncomp) {
    m_mcdiffusion_op = std::make_unique<DiffusionOp>(this, ncomp);
  }
  return m_mcdiffusion_op.get();
}

DiffusionTensorOp*
PeleLM::getDiffusionTensorOp()
{
  if (!m_diffusionTensor_op) {
    m_diffusionTensor_op = std::make_unique<DiffusionTensorOp>(this);
  }
  return m_diffusionTensor_op.get();
}

void
PeleLM::computeDifferentialDiffusionTerms(
  const TimeStamp a_time,
  const std::unique_ptr<AdvanceDiffData>& diffData,
  const int is_init,
  const amrex::Real flux_tracking_factor)
{
  BL_PROFILE("PeleLMeX::computeDifferentialDiffusionTerms()");

  //----------------------------------------------------------------
  // Checks
  AMREX_ASSERT((a_time == AmrOldTime) || (a_time == AmrNewTime));
  if (a_time == AmrOldTime) {
    AMREX_ASSERT(diffData->Dn.size() == finest_level + 1);
    AMREX_ASSERT(diffData->Dn[0].nComp() >= NUM_SPECIES + 2);
    if (m_nAux > 0) {
      AMREX_ASSERT(diffData->Dn_aux.size() == finest_level + 1);
      AMREX_ASSERT(diffData->Dn_aux[0].nComp() == m_nAux);
    }
  }
  if (a_time != AmrOldTime) {
    AMREX_ASSERT(diffData->Dnp1.size() == finest_level + 1);
    AMREX_ASSERT(diffData->Dnp1[0].nComp() >= NUM_SPECIES + 2);
    if (m_nAux > 0) {
      AMREX_ASSERT(diffData->Dnp1_aux.size() == finest_level + 1);
      AMREX_ASSERT(diffData->Dnp1_aux[0].nComp() == m_nAux);
    }
  }

  //----------------------------------------------------------------
  // Setup fluxes
  // [0:NUM_SPECIES-1] Species     : \Flux_k
  // [NUM_SPECIES]     Temperature : - \lambda \nabla T
  // [NUM_SPECIES+1]   DiffDiff    : \sum_k ( h_k * \Flux_k )
  constexpr int nGrow = 0; // No need for ghost face on fluxes
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes(
    finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes_aux(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto& ba = grids[lev];
    const auto& factory = Factory(lev);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      fluxes[lev][idim].define(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dmap[lev],
        NUM_SPECIES + 2, nGrow, amrex::MFInfo(), factory);
      if (m_nAux > 0) {
        fluxes_aux[lev][idim].define(
          amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)),
          dmap[lev], m_nAux, nGrow, amrex::MFInfo(), factory);
      }
    }
  }
#ifdef AMREX_USE_EB
  amrex::Vector<amrex::MultiFab> EBfluxes;
  if (m_isothermalEB != 0) {
    EBfluxes.reserve(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      EBfluxes.emplace_back(
        grids[lev], dmap[lev], 1, nGrow, amrex::MFInfo(), Factory(lev));
    }
  }
#endif

  //----------------------------------------------------------------
  // Compute differential diffusion fluxes including correction velocity and
  // wbar term During initialization, don't bother getting the wbar fluxes
  // separately
  amrex::Vector<std::array<amrex::MultiFab*, AMREX_SPACEDIM>> wbarFluxVec =
    ((is_init != 0) || (m_use_wbar == 0))
      ? amrex::Vector<std::array<amrex::MultiFab*, AMREX_SPACEDIM>>{}
      : GetVecOfArrOfPtrs(diffData->wbar_fluxes);
  amrex::Vector<std::array<amrex::MultiFab*, AMREX_SPACEDIM>> soretFluxVec =
    ((is_init != 0) || (m_use_soret == 0))
      ? amrex::Vector<std::array<amrex::MultiFab*, AMREX_SPACEDIM>>{}
      : GetVecOfArrOfPtrs(diffData->soret_fluxes);

#ifdef AMREX_USE_EB
  if (m_isothermalEB != 0) {
    computeDifferentialDiffusionFluxes(
      a_time, GetVecOfArrOfPtrs(fluxes), GetVecOfPtrs(EBfluxes), wbarFluxVec,
      soretFluxVec, GetVecOfArrOfPtrs(fluxes_aux));
  } else
#endif
  {
    computeDifferentialDiffusionFluxes(
      a_time, GetVecOfArrOfPtrs(fluxes), {}, wbarFluxVec, soretFluxVec,
      GetVecOfArrOfPtrs(fluxes_aux));
  }

  // If doing species balances, compute face domain integrals
  // using level 0 since we've averaged down the fluxes already
  // Factor for SDC is 0.5 is for Dn and -0.5 for Dnp1
  if ((flux_tracking_factor != 0.0) && (m_do_speciesBalance != 0)) {
    addRhoYFluxes(GetArrOfConstPtrs(fluxes[0]), geom[0], flux_tracking_factor);
  }

  //----------------------------------------------------------------
  // Compute divergence/fill a_viscTerm
  // [0:NUM_SPECIES-1] Species           : \nabla \cdot \Flux_k
  // [NUM_SPECIES]     Temperature       : \nabla \cdot (-\lambda \nabla T)
  // [NUM_SPECIES+1]   Differential diff : \nabla \cdot \sum_k ( h_k * \Flux_k )
  constexpr int intensiveFluxes = 1; // All the fluxes are intensive here
  amrex::Vector<amrex::MultiFab*> diffTermVec =
    (a_time == AmrOldTime) ? GetVecOfPtrs(diffData->Dn)
                           : GetVecOfPtrs(diffData->Dnp1);
  amrex::Vector<amrex::MultiFab*> diffTermAuxVec =
    (a_time == AmrOldTime) ? GetVecOfPtrs(diffData->Dn_aux)
                           : GetVecOfPtrs(diffData->Dnp1_aux);
#ifdef AMREX_USE_EB
  auto bcRecSpec = fetchBCRecArray(FIRSTSPEC, NUM_SPECIES);
  auto bcRecSpec_d = convertToDeviceVector(bcRecSpec);
  fluxDivergenceRD(
    GetVecOfConstPtrs(getSpeciesVect(a_time)), 0, diffTermVec, 0,
    GetVecOfArrOfPtrs(fluxes), 0, {}, 0, NUM_SPECIES, intensiveFluxes,
    bcRecSpec_d.dataPtr(), -1.0, m_dt);

  if (m_nAux > 0) {
    auto bcRecAux = fetchBCRecAuxArray(0, m_nAux);
    auto bcRecAux_d = convertToDeviceVector(bcRecAux);
    fluxDivergenceRD(
      GetVecOfConstPtrs(getAuxVect(a_time)), 0, diffTermAuxVec, 0,
      GetVecOfArrOfPtrs(fluxes_aux), 0, {}, 0, m_nAux, intensiveFluxes,
      bcRecAux_d.dataPtr(), -1.0, m_dt);
  }

  auto bcRecTemp = fetchBCRecArray(TEMP, 1);
  auto bcRecTemp_d = convertToDeviceVector(bcRecTemp);
  amrex::Vector<amrex::MultiFab*> EBFluxesVec =
    (m_isothermalEB) != 0 ? GetVecOfPtrs(EBfluxes)
                          : amrex::Vector<amrex::MultiFab*>{};
  fluxDivergenceRD(
    GetVecOfConstPtrs(getTempVect(a_time)), 0, diffTermVec, NUM_SPECIES,
    GetVecOfArrOfPtrs(fluxes), NUM_SPECIES, EBFluxesVec, 0, 1, intensiveFluxes,
    bcRecTemp_d.dataPtr(), -1.0, m_dt);

  auto bcRecRhoH = fetchBCRecArray(RHOH, 1);
  auto bcRecRhoH_d = convertToDeviceVector(bcRecRhoH);
  fluxDivergenceRD(
    GetVecOfConstPtrs(getRhoHVect(a_time)), 0, diffTermVec, NUM_SPECIES + 1,
    GetVecOfArrOfPtrs(fluxes), NUM_SPECIES + 1, {}, 0, 1, intensiveFluxes,
    bcRecRhoH_d.dataPtr(), -1.0, m_dt);
#else
  fluxDivergence(
    diffTermVec, 0, GetVecOfArrOfPtrs(fluxes), 0, NUM_SPECIES + 2,
    intensiveFluxes, -1.0);
  if (m_nAux > 0) {
    fluxDivergence(
      diffTermAuxVec, 0, GetVecOfArrOfPtrs(fluxes_aux), 0, m_nAux,
      intensiveFluxes, -1.0);
  }
#endif

  // Get the wbar term if appropriate
  if ((is_init == 0) && (m_use_wbar != 0)) {
#ifdef AMREX_USE_EB
    fluxDivergenceRD(
      GetVecOfConstPtrs(getSpeciesVect(a_time)), 0,
      GetVecOfPtrs(diffData->Dwbar), 0,
      GetVecOfArrOfPtrs(diffData->wbar_fluxes), 0, {}, 0, NUM_SPECIES,
      intensiveFluxes, bcRecSpec_d.dataPtr(), -1.0, m_dt);
#else
    fluxDivergence(
      GetVecOfPtrs(diffData->Dwbar), 0,
      GetVecOfArrOfPtrs(diffData->wbar_fluxes), 0, NUM_SPECIES, intensiveFluxes,
      -1.0);
#endif
  }

  // Get the Soret term if appropriate
  if ((is_init == 0) && (m_use_soret != 0)) {
#ifdef AMREX_USE_EB
    fluxDivergenceRD(
      GetVecOfConstPtrs(getSpeciesVect(a_time)), 0, GetVecOfPtrs(diffData->DT),
      0, GetVecOfArrOfPtrs(diffData->soret_fluxes), 0, {}, 0, NUM_SPECIES,
      intensiveFluxes, bcRecSpec_d.dataPtr(), -1.0, m_dt);
#else
    fluxDivergence(
      GetVecOfPtrs(diffData->DT), 0, GetVecOfArrOfPtrs(diffData->soret_fluxes),
      0, NUM_SPECIES, intensiveFluxes, -1.0);
#endif
  }

#ifdef AMREX_USE_EB
  // Set EB-covered diffusion terms here to avoid having to mask operations
  // using this data later on
  for (int lev = 0; lev <= finest_level; ++lev) {
    if (a_time == AmrOldTime) {
      EB_set_covered(diffData->Dn[lev], 0.0);
      if (m_nAux > 0) {
        EB_set_covered(diffData->Dn_aux[lev], 0.0);
      }
    } else {
      EB_set_covered(diffData->Dnp1[lev], 0.0);
      if (m_nAux > 0) {
        EB_set_covered(diffData->Dnp1_aux[lev], 0.0);
      }
    }
    if ((is_init == 0) && (m_use_wbar != 0)) {
      EB_set_covered(diffData->Dwbar[lev], 0.0);
    }
    if ((is_init == 0) && (m_use_soret != 0)) {
      EB_set_covered(diffData->DT[lev], 0.0);
    }
  }
#endif

  // Under mesh mapping the flux divergence was computed in Xi-space on
  // Xi-space fluxes (bcoeff carries the J/fac^2 factor).  That produces
  // J . (physical divergence).  Divide each divergence term by J so the
  // returned Dn / Dnp1 / Dwbar / DT are in physical-space units.
  if (m_mesh_mapping) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab& dterm =
        (a_time == AmrOldTime) ? diffData->Dn[lev] : diffData->Dnp1[lev];
      const int nc = dterm.nComp();
      for (int n = 0; n < nc; ++n) {
        amrex::MultiFab::Divide(dterm, m_mesh_map->detJ_cc(lev), 0, n, 1, 0);
      }
      if (m_nAux > 0) {
        amrex::MultiFab& dt_aux = (a_time == AmrOldTime)
                                    ? diffData->Dn_aux[lev]
                                    : diffData->Dnp1_aux[lev];
        for (int n = 0; n < dt_aux.nComp(); ++n) {
          amrex::MultiFab::Divide(dt_aux, m_mesh_map->detJ_cc(lev), 0, n, 1, 0);
        }
      }
      if ((is_init == 0) && (m_use_wbar != 0)) {
        for (int n = 0; n < diffData->Dwbar[lev].nComp(); ++n) {
          amrex::MultiFab::Divide(
            diffData->Dwbar[lev], m_mesh_map->detJ_cc(lev), 0, n, 1, 0);
        }
      }
      if ((is_init == 0) && (m_use_soret != 0)) {
        for (int n = 0; n < diffData->DT[lev].nComp(); ++n) {
          amrex::MultiFab::Divide(
            diffData->DT[lev], m_mesh_map->detJ_cc(lev), 0, n, 1, 0);
        }
      }
    }
  }
}

void
PeleLM::unscaleMappedDivergence(
  const amrex::Vector<amrex::MultiFab*>& a_div,
  const int a_comp,
  const int a_ncomp)
{
  // Under mesh mapping a flux divergence formed in Xi-space from Xi-space
  // fluxes (bcoeff carries J/fac^2) is J . (physical divergence); divide
  // by J to return to physical units.  No-op without mapping.
  if (!m_mesh_mapping) {
    return;
  }
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int n = 0; n < a_ncomp; ++n) {
      amrex::MultiFab::Divide(
        *a_div[lev], m_mesh_map->detJ_cc(lev), 0, a_comp + n, 1, 0);
    }
  }
}

template <typename EOSType>
void
PeleLM::adjustSpeciesFluxes(
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_spfluxes,
  amrex::Vector<amrex::MultiFab const*> const& a_spec)
{

  BL_PROFILE("PeleLMeX::adjustSpeciesFluxes()");

  // Get the species BCRec
  auto bcRecSpec = fetchBCRecArray(FIRSTSPEC, NUM_SPECIES);

  for (int lev = 0; lev <= finest_level; ++lev) {

    const amrex::Box& domain = geom[lev].Domain();

#ifdef AMREX_USE_EB
    //------------------------------------------------------------------------
    // Get the edge species state needed for EB
    constexpr int nGrow = 1;
    const auto& ba = a_spec[lev]->boxArray();
    const auto& dm = a_spec[lev]->DistributionMap();
    const auto& ebfact = EBFactory(lev);
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> edgstate{AMREX_D_DECL(
      amrex::MultiFab(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(0)), dm,
        NUM_SPECIES, nGrow, amrex::MFInfo(), ebfact),
      amrex::MultiFab(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(1)), dm,
        NUM_SPECIES, nGrow, amrex::MFInfo(), ebfact),
      amrex::MultiFab(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(2)), dm,
        NUM_SPECIES, nGrow, amrex::MFInfo(), ebfact))};
    EB_interp_CellCentroid_to_FaceCentroid(
      *a_spec[lev], GetArrOfPtrs(edgstate), 0, 0, NUM_SPECIES, geom[lev],
      bcRecSpec);
    auto const& areafrac = ebfact.getAreaFrac();
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*a_spec[lev], amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const amrex::Box& ebx = mfi.nodaltilebox(idim);
        const amrex::Box& edomain = amrex::surroundingNodes(domain, idim);
        auto const& rhoY = a_spec[lev]->const_array(mfi);
        auto const& flux_dir = a_spfluxes[lev][idim]->array(mfi);

        const auto bc_lo = bcRecSpec[0].lo(idim);
        const auto bc_hi = bcRecSpec[0].hi(idim);

#ifdef AMREX_USE_EB
        auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];

        if (flagfab.getType(amrex::grow(ebx, 0)) != amrex::FabType::covered) {
          // No cut cells in tile + nghost-cell width halo -> use non-eb routine
          if (
            flagfab.getType(amrex::grow(ebx, nGrow)) ==
            amrex::FabType::regular) {
            amrex::ParallelFor(
              ebx, [idim, rhoY, flux_dir, edomain, bc_lo,
                    bc_hi] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const int idx[3] = {i, j, k};
                const bool on_lo =
                  ((bc_lo == amrex::BCType::ext_dir) &&
                   (idx[idim] <= edomain.smallEnd(idim)));
                const bool on_hi =
                  ((bc_hi == amrex::BCType::ext_dir) &&
                   (idx[idim] >= edomain.bigEnd(idim)));
                repair_flux(i, j, k, idim, on_lo, on_hi, rhoY, flux_dir);
              });
          } else {
            auto const& rhoYed_ar = edgstate[idim].const_array(mfi);
            auto const& areafrac_ar = areafrac[idim]->const_array(mfi);
            amrex::ParallelFor(
              ebx,
              [idim, rhoY, flux_dir, rhoYed_ar, areafrac_ar, edomain, bc_lo,
               bc_hi] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const int idx[3] = {i, j, k};
                const bool on_lo =
                  ((bc_lo == amrex::BCType::ext_dir) &&
                   (idx[idim] <= edomain.smallEnd(idim)));
                const bool on_hi =
                  ((bc_hi == amrex::BCType::ext_dir) &&
                   (idx[idim] >= edomain.bigEnd(idim)));
                repair_flux_eb(
                  i, j, k, idim, on_lo, on_hi, rhoY, rhoYed_ar, areafrac_ar,
                  flux_dir);
              });
          }
        }
#else
        amrex::ParallelFor(
          ebx, [idim, rhoY, flux_dir, edomain, bc_lo,
                bc_hi] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const int idx[3] = {i, j, k};
            const bool on_lo =
              ((bc_lo == amrex::BCType::ext_dir) &&
               (idx[idim] <= edomain.smallEnd(idim)));
            const bool on_hi =
              ((bc_hi == amrex::BCType::ext_dir) &&
               (idx[idim] >= edomain.bigEnd(idim)));
            repair_flux(i, j, k, idim, on_lo, on_hi, rhoY, flux_dir);
          });
#endif
      }
    }
  }
}

template <>
void
PeleLM::adjustSpeciesFluxes<pele::physics::eos::Manifold>(
  const amrex::Vector<
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>& /*a_spfluxes*/,
  amrex::Vector<amrex::MultiFab const*> const& /*a_spec*/)
{
  // Manifold Model: "Species" don't sum to unity, so no need to adjust the
  // fluxes
  BL_PROFILE("PeleLM::adjustSpeciesFluxes()");
}

void
PeleLM::correctIsothermalBoundary(
  const TimeStamp a_time,
  const amrex::Vector<amrex::MultiFab*>& a_spec_boundary,
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_wbarfluxes,
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_soretfluxes)
{
  BL_PROFILE("PeleLMeX::correctIsothermalBoundary()");
  auto bcRecSpec = fetchBCRecArray(FIRSTSPEC, NUM_SPECIES);
  const bool need_explicit_fluxes = a_soretfluxes.empty();

  amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> soretfluxes(
    finest_level + 1);
  if (need_explicit_fluxes) { // need to fill the soret fluxes ourselves
    for (int lev = 0; lev <= finest_level; ++lev) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        soretfluxes[lev][idim] = new amrex::MultiFab(
          grids[lev], dmap[lev], NUM_SPECIES, 1, amrex::MFInfo(), Factory(lev));
        soretfluxes[lev][idim]->setVal(0.0);
      }
    }
    addSoretTerm(
      soretfluxes, soretfluxes, GetVecOfConstPtrs(getTempVect(a_time)),
      GetVecOfConstPtrs(getDiffusivityVect(a_time)));
  } else { // have the lagged ones, alias to them
    for (int lev = 0; lev <= finest_level; ++lev) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        soretfluxes[lev][idim] = new amrex::MultiFab(
          *a_soretfluxes[lev][idim], amrex::make_alias, 0, NUM_SPECIES);
      }
    }
  }
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_time);
    // Lets overwrite the boundaries with the fluxes for the inhomogeneous
    // Neumann solve
    // Get the edge centered diffusivities
    amrex::MultiFab& ldata_beta_cc = ldata_p->diff_cc;
    const amrex::Box& domain = geom[lev].Domain();
    constexpr int doZeroVisc = 1;
    constexpr int addTurbContribution = 0;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> beta_ec = getDiffusivity(
      lev, 0, NUM_SPECIES, doZeroVisc, bcRecSpec, ldata_beta_cc,
      addTurbContribution);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_beta_cc, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const auto bc_lo = m_phys_bc.lo(idim);
        const auto bc_hi = m_phys_bc.hi(idim);
        const amrex::Box& edomain = amrex::surroundingNodes(domain, idim);
        const amrex::Box& ebx = mfi.nodaltilebox(idim);
        auto const& flux_soret = soretfluxes[lev][idim]->const_array(mfi);
        auto const& rhoD_ec = beta_ec[idim].const_array(mfi);
        auto const& flux_wbar = (m_use_wbar != 0 && !need_explicit_fluxes)
                                  ? a_wbarfluxes[lev][idim]->const_array(mfi)
                                  : rhoD_ec;
        auto const& boundary_ar = a_spec_boundary[lev]->array(mfi);
        const auto use_wbar = m_use_wbar;
        amrex::ParallelFor(
          ebx, [bc_lo, bc_hi, idim, need_explicit_fluxes, edomain, flux_soret,
                rhoD_ec, flux_wbar, boundary_ar,
                use_wbar] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int idx[3] = {i, j, k};
            const bool on_lo =
              (bc_lo == BoundaryCondition::BCNoSlipWallIsotherm ||
               bc_lo == BoundaryCondition::BCSlipWallIsotherm) &&
              (idx[idim] <= edomain.smallEnd(idim));
            const bool on_hi =
              (bc_hi == BoundaryCondition::BCNoSlipWallIsotherm ||
               bc_hi == BoundaryCondition::BCSlipWallIsotherm) &&
              (idx[idim] >= edomain.bigEnd(idim));
            if (on_lo || on_hi) {
              if (on_lo) { // need to move -1 for lo boundary
                idx[idim] -= 1;
              }
              for (int n = 0; n < NUM_SPECIES; ++n) {
                boundary_ar(idx[0], idx[1], idx[2], n) = flux_soret(i, j, k, n);
                // add lagged wbar flux
                if (use_wbar != 0 && !need_explicit_fluxes) {
                  boundary_ar(idx[0], idx[1], idx[2], n) +=
                    flux_wbar(i, j, k, n);
                }
                boundary_ar(idx[0], idx[1], idx[2], n) /= rhoD_ec(i, j, k, n);
              }
            }
          });
      }
    }
  }
  // TODO: wbar fluxes disabled for this case - boundary system becomes complex
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      delete soretfluxes[lev][idim];
    }
  }
}

void
PeleLM::computeDifferentialDiffusionFluxes(
  const TimeStamp a_time,
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>& a_fluxes,
  const amrex::Vector<amrex::MultiFab*>&
#ifdef AMREX_USE_EB
    a_EBfluxes
#else
/*unused*/
#endif
  ,
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_wbarfluxes,
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_soretfluxes,
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_auxfluxes)
{
  BL_PROFILE("PeleLMeX::computeDifferentialDiffusionFluxes()");

  //----------------------------------------------------------------

  amrex::Vector<amrex::MultiFab> spec_boundary;
  if (m_soret_boundary_override != 0) {
    spec_boundary.reserve(finest_level + 1);
    // this is the same regardless of lagged or not
    for (int lev = 0; lev <= finest_level; ++lev) {
      spec_boundary.emplace_back(
        grids[lev], dmap[lev], NUM_SPECIES, 1, amrex::MFInfo(), Factory(lev));
      auto* ldata_p = getLevelDataPtr(lev, a_time);
      // if we have a mix of Dirichlet and Isothermal walls, we need to give
      // Dirichlet boundaries and divide by density since diffuse_scalar doesn't
      // touch this boundary MF

      auto const& sp_bound_ma = spec_boundary[lev].arrays();
      auto const& state_ma = ldata_p->state.const_arrays();
      amrex::ParallelFor(
        spec_boundary[lev], spec_boundary[lev].nGrowVect(), NUM_SPECIES,
        [sp_bound_ma, state_ma] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          amrex::Array4<amrex::Real const> rhoY(state_ma[box_no], FIRSTSPEC);
          amrex::Array4<amrex::Real const> rho(state_ma[box_no], DENSITY);
          sp_bound_ma[box_no](i, j, k, n) = rhoY(i, j, k, n) / rho(i, j, k);
        });
    }
    amrex::Gpu::streamSynchronize();
    // correct the boundary values, pass empty soret & wbar to trigger explicit
    // calculation
    correctIsothermalBoundary(a_time, GetVecOfPtrs(spec_boundary), {}, {});
  }
  // Species fluxes
  // Get the species BCRec
  auto bcRecSpec = fetchBCRecArray(FIRSTSPEC, NUM_SPECIES);
  auto bcRecAux = fetchBCRecAuxArray(0, m_nAux);
  constexpr int do_avgDown = 0;

#ifdef PELE_USE_PLASMA
  // Get the species diffusion fluxes from the DiffusionOp
  // Don't average down just yet
  getMCDiffusionOp(NUM_SPECIES - NUM_IONS)
    ->computeDiffFluxes(
      a_fluxes, 0, GetVecOfConstPtrs(getSpeciesVect(a_time)), 0,
      GetVecOfConstPtrs(getDensityVect(a_time)),
      GetVecOfConstPtrs(getDiffusivityVect(a_time)), 0, bcRecSpec,
      NUM_SPECIES - NUM_IONS, do_avgDown, {});
  // Ions one by one
  for (int n = 0; n < NUM_IONS; ++n) {
    auto bcRecIons = fetchBCRecArray(FIRSTSPEC + NUM_SPECIES - NUM_IONS + n, 1);
    getDiffusionOp()->computeDiffFluxes(
      a_fluxes, NUM_SPECIES - NUM_IONS + n,
      GetVecOfConstPtrs(getSpeciesVect(a_time)), NUM_SPECIES - NUM_IONS + n,
      GetVecOfConstPtrs(getDensityVect(a_time)),
      GetVecOfConstPtrs(getDiffusivityVect(a_time)), NUM_SPECIES - NUM_IONS + n,
      bcRecIons, 1, do_avgDown, {});
  }
#else
  // Get the species diffusion fluxes from the DiffusionOp
  // Don't average down just yet
  getMCDiffusionOp(NUM_SPECIES)
    ->computeDiffFluxes(
      a_fluxes, 0, GetVecOfConstPtrs(getSpeciesVect(a_time)), 0,
      GetVecOfConstPtrs(getDensityVect(a_time)),
      GetVecOfConstPtrs(getDiffusivityVect(a_time)), 0, bcRecSpec, NUM_SPECIES,
      do_avgDown, GetVecOfConstPtrs(spec_boundary));

  if (m_nAux > 0) {
    getMCDiffusionOp(m_nAux)->computeDiffFluxes(
      a_auxfluxes, 0, GetVecOfConstPtrs(getAuxVect(a_time)), 0, {},
      GetVecOfConstPtrs(getAuxDiffusivityVect(a_time)), 0, bcRecAux, m_nAux,
      do_avgDown, GetVecOfConstPtrs(spec_boundary));
  }
#endif

  // Add the wbar term
  if (m_use_wbar != 0) {
    const int need_wbar_fluxes = (a_wbarfluxes.empty()) ? 0 : 1;
    if (need_wbar_fluxes == 0) {
      addWbarTerm(
        a_fluxes, {}, GetVecOfConstPtrs(getSpeciesVect(a_time)),
        GetVecOfConstPtrs(getDensityVect(a_time)),
        GetVecOfConstPtrs(getDiffusivityVect(a_time)),
        GetVecOfConstPtrs(spec_boundary));
    } else {
      addWbarTerm(
        a_fluxes, a_wbarfluxes, GetVecOfConstPtrs(getSpeciesVect(a_time)),
        GetVecOfConstPtrs(getDensityVect(a_time)),
        GetVecOfConstPtrs(getDiffusivityVect(a_time)),
        GetVecOfConstPtrs(spec_boundary));
    }
  }

  // Add the Soret term
  if (m_use_soret != 0) {
    const int need_soret_fluxes = (a_soretfluxes.empty()) ? 0 : 1;
    if (need_soret_fluxes == 0) {
      addSoretTerm(
        a_fluxes, {}, GetVecOfConstPtrs(getTempVect(a_time)),
        GetVecOfConstPtrs(getDiffusivityVect(a_time)));
    } else {
      addSoretTerm(
        a_fluxes, a_soretfluxes, GetVecOfConstPtrs(getTempVect(a_time)),
        GetVecOfConstPtrs(getDiffusivityVect(a_time)));
    }
  }

  // Adjust species diffusion fluxes to ensure their sum is zero
  adjustSpeciesFluxes<pele::physics::PhysicsType::eos_type>(
    a_fluxes, GetVecOfConstPtrs(getSpeciesVect(a_time)));
  //----------------------------------------------------------------

  //----------------------------------------------------------------
  // Enthalpy fluxes
  // Get the temperature BCRec
  auto bcRecTemp = fetchBCRecArray(TEMP, 1);

  // Fourier: - \lambda \nabla T
#ifdef AMREX_USE_EB
  if (m_isothermalEB != 0) {
    AMREX_ASSERT(!a_EBfluxes.empty());
    // Set up EB dirichlet value and diffusivity
    amrex::Vector<amrex::MultiFab> EBdiff;
    EBdiff.reserve(finest_level + 1);

    for (int lev = 0; lev <= finest_level; ++lev) {
      EBdiff.emplace_back(
        grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), EBFactory(lev));
      getEBDiff(lev, a_time, EBdiff[lev], NUM_SPECIES);
    }
    getDiffusionOp()->computeDiffFluxes(
      a_fluxes, NUM_SPECIES, a_EBfluxes, 0,
      GetVecOfConstPtrs(getTempVect(a_time)), 0, {},
      GetVecOfConstPtrs(getDiffusivityVect(a_time)), NUM_SPECIES,
      GetVecOfConstPtrs(getEBState(TEMP, 1, a_time)), GetVecOfConstPtrs(EBdiff),
      bcRecTemp, 1, do_avgDown, {});
  } else
#endif
  {
    getDiffusionOp()->computeDiffFluxes(
      a_fluxes, NUM_SPECIES, GetVecOfConstPtrs(getTempVect(a_time)), 0, {},
      GetVecOfConstPtrs(getDiffusivityVect(a_time)), NUM_SPECIES, bcRecTemp, 1,
      do_avgDown, {});
  }

  // Differential diffusion term: \sum_k ( h_k * \Flux_k )
  computeSpeciesEnthalpyFlux(a_fluxes, GetVecOfConstPtrs(getTempVect(a_time)));
  //----------------------------------------------------------------

  //----------------------------------------------------------------
  // Get fluxes consistent across levels by averaging down all components
  getDiffusionOp()->avgDownFluxes(a_fluxes, 0, NUM_SPECIES + 2);
  if (m_nAux > 0) {
    getDiffusionOp()->avgDownFluxes(a_auxfluxes, 0, m_nAux);
  }
  //----------------------------------------------------------------

#ifdef AMREX_USE_EB
  //----------------------------------------------------------------
  // Set covered faces to large dummy values to catch any usage
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::EB_set_covered_faces(
      {AMREX_D_DECL(a_fluxes[lev][0], a_fluxes[lev][1], a_fluxes[lev][2])},
      1.234e40);
    if (m_nAux > 0) {
      amrex::EB_set_covered_faces(
        {AMREX_D_DECL(
          a_auxfluxes[lev][0], a_auxfluxes[lev][1], a_auxfluxes[lev][2])},
        1.234e40);
    }
  }
#endif
}

void
PeleLM::addWbarTerm(
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_spfluxes,
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_spwbarfluxes,
  amrex::Vector<amrex::MultiFab const*> const& a_spec,
  amrex::Vector<amrex::MultiFab const*> const& a_rho,
  amrex::Vector<amrex::MultiFab const*> const& a_beta,
  amrex::Vector<amrex::MultiFab const*> const& a_boundary)
{
  //------------------------------------------------------------------------
  // if a container for wbar fluxes is provided, fill it
  const int need_wbar_fluxes = (a_spwbarfluxes.empty()) ? 0 : 1;
  const int have_boundary = (a_boundary.empty()) ? 0 : 1;
  //------------------------------------------------------------------------
  // Compute Wbar on all the levels
  int nGrow = 1; // Need one ghost cell to compute gradWbar
  amrex::Vector<amrex::MultiFab> Wbar;
  Wbar.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> Wbar_boundary;
  if (have_boundary != 0) {
    Wbar_boundary.reserve(finest_level + 1);
  }
  auto const* leosparm = eos_parms.device_parm();
  for (int lev = 0; lev <= finest_level; ++lev) {
    Wbar.emplace_back(
      grids[lev], dmap[lev], 1, nGrow, amrex::MFInfo(), Factory(lev));
    if (have_boundary != 0) {
      Wbar_boundary.emplace_back(
        grids[lev], dmap[lev], 1, nGrow, amrex::MFInfo(), Factory(lev));
    }
  }
  for (int lev = 0; lev <= finest_level; ++lev) {
    const amrex::Box& domain = geom[lev].Domain();

    auto const& rho_ma = a_rho[lev]->const_arrays();
    auto const& rhoY_ma = a_spec[lev]->const_arrays();
    auto const& Wbar_ma = Wbar[lev].arrays();
    auto const& gradY_ma =
      (have_boundary != 0) ? a_boundary[lev]->const_arrays() : rhoY_ma;
    auto const& Wbar_boundary_ma =
      (have_boundary != 0) ? Wbar_boundary[lev].arrays() : Wbar_ma;

    amrex::ParallelFor(
      Wbar[lev], Wbar[lev].nGrowVect(),
      [rho_ma, rhoY_ma, Wbar_ma, gradY_ma, Wbar_boundary_ma, have_boundary,
       leosparm, domain,
       phys_bc =
         m_phys_bc] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        getMwmixGivenRY(
          i, j, k, rho_ma[box_no], rhoY_ma[box_no], Wbar_ma[box_no], leosparm);
        if (have_boundary != 0) { // need to impose gradWbar on boundary for
          // computeGradient
          // for dirichlet boundaries, we'll overwrite inhomog neumann ones
          // NOTE: for now, this is skipped since wbar disabled for
          // isothermal/soret
          Wbar_boundary_ma[box_no](i, j, k) = Wbar_ma[box_no](i, j, k);
          const int idx[3] = {i, j, k};
          for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const auto bc_lo = phys_bc.lo(idim);
            const auto bc_hi = phys_bc.hi(idim);
            const bool on_lo =
              (bc_lo == BoundaryCondition::BCNoSlipWallIsotherm ||
               bc_lo == BoundaryCondition::BCSlipWallIsotherm) &&
              (idx[idim] < domain.smallEnd(idim));
            const bool on_hi =
              (bc_hi == BoundaryCondition::BCNoSlipWallIsotherm ||
               bc_hi == BoundaryCondition::BCSlipWallIsotherm) &&
              (idx[idim] > domain.bigEnd(idim));

            if (on_lo || on_hi) {
              getGradMwmixGivengradYMwmix(
                i, j, k, gradY_ma[box_no], Wbar_ma[box_no],
                Wbar_boundary_ma[box_no], leosparm);
            }
          }
        }
      });
  }
  amrex::Gpu::streamSynchronize();

  //------------------------------------------------------------------------
  // Compute Wbar gradients and do average down to get gradients consistent
  // across levels Get the species BCRec
  constexpr int do_avgDown = 1;
  auto bcRecSpec = fetchBCRecArray(FIRSTSPEC, NUM_SPECIES);

  nGrow = 0; // No need for ghost face on fluxes
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> gradWbar(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto& ba = grids[lev];
    const auto& factory = Factory(lev);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      gradWbar[lev][idim].define(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dmap[lev],
        NUM_SPECIES, nGrow, amrex::MFInfo(), factory);
      gradWbar[lev][idim].setVal(0.0);
    }
  }
  getDiffusionOp()->computeGradient(
    GetVecOfArrOfPtrs(gradWbar), {}, // Don't need the laplacian out
    GetVecOfConstPtrs(Wbar), GetVecOfConstPtrs(Wbar_boundary), bcRecSpec[0],
    do_avgDown);

  //------------------------------------------------------------------------
  // add Wbar term to species fluxes
  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get edge diffusivity
    constexpr int doZeroVisc = 1;
    constexpr int addTurbContrib = 0;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> beta_ec = getDiffusivity(
      lev, 0, NUM_SPECIES, doZeroVisc, bcRecSpec, *a_beta[lev], addTurbContrib);

    const amrex::Box& domain = geom[lev].Domain();
    const bool use_harmonic_avg = m_harm_avg_cen2edge != 0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
      for (amrex::MFIter mfi(*a_beta[lev], amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

          // Get edge centered rhoYs
          const amrex::Box ebx = mfi.nodaltilebox(idim);
          amrex::FArrayBox rhoY_ed(ebx, NUM_SPECIES, amrex::The_Async_Arena());

          const amrex::Box& edomain = amrex::surroundingNodes(domain, idim);
          auto const& rhoY_arr = a_spec[lev]->const_array(mfi);
          const auto& rhoYed_arr = rhoY_ed.array(0);
          const auto bc_lo = bcRecSpec[0].lo(idim);
          const auto bc_hi = bcRecSpec[0].hi(idim);
          amrex::ParallelFor(
            ebx, [idim, bc_lo, bc_hi, use_harmonic_avg, rhoY_arr, rhoYed_arr,
                  edomain] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              const int idx[3] = {i, j, k};
              const bool on_lo =
                ((bc_lo == amrex::BCType::ext_dir) &&
                 (idx[idim] <= edomain.smallEnd(idim)));
              const bool on_hi =
                ((bc_hi == amrex::BCType::ext_dir) &&
                 (idx[idim] >= edomain.bigEnd(idim)));
              cen2edg_cpp(
                i, j, k, idim, NUM_SPECIES, use_harmonic_avg, on_lo, on_hi,
                rhoY_arr, rhoYed_arr);
            });

          auto const& rhoY = rhoY_ed.const_array(0);
          auto const& gradWbar_ar = gradWbar[lev][idim].const_array(mfi);
          auto const& beta_ar = beta_ec[idim].const_array(mfi);
          auto const& spFlux_ar = a_spfluxes[lev][idim]->array(mfi);
          auto const& spwbarFlux_ar =
            (need_wbar_fluxes) != 0
              ? a_spwbarfluxes[lev][idim]->array(mfi)
              : a_spfluxes[lev][idim]->array(mfi); // Dummy unused Array4

          // Wbar flux is : - \rho Y_m / W_k * D_m * \nabla
          // \overline{W} with beta_m = \rho * D_m * overline(W) / W_k below
          // need to divide by \overline(W)

          amrex::ParallelFor(
            ebx, [rhoY, gradWbar_ar, beta_ar, spFlux_ar, spwbarFlux_ar,
                  need_wbar_fluxes,
                  leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              auto eos = pele::physics::PhysicsType::eos(leosparm);
              // Get Wbar from rhoYs
              amrex::Real rho = 0.0;
              for (int n = 0; n < NUM_SPECIES; ++n) {
                rho += rhoY(i, j, k, n);
              }
              amrex::Real rho_inv = 1.0 / rho;
              amrex::Real y[NUM_SPECIES] = {0.0};
              for (int n = 0; n < NUM_SPECIES; ++n) {
                y[n] = rhoY(i, j, k, n) * rho_inv;
              }
              amrex::Real WBAR = 0.0;
              eos.Y2WBAR(y, WBAR);
              WBAR *= 0.001;
              for (int n = 0; n < NUM_SPECIES; ++n) {
                spFlux_ar(i, j, k, n) -=
                  y[n] / WBAR * beta_ar(i, j, k, n) * gradWbar_ar(i, j, k);
              }
              if (need_wbar_fluxes != 0) {
                for (int n = 0; n < NUM_SPECIES; ++n) {
                  spwbarFlux_ar(i, j, k, n) =
                    -y[n] / WBAR * beta_ar(i, j, k, n) * gradWbar_ar(i, j, k);
                }
              }
            });
        }
      }
    }
  }
}

void
PeleLM::addSoretTerm(
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_spfluxes,
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_spsoretfluxes,
  amrex::Vector<amrex::MultiFab const*> const& a_temp,
  amrex::Vector<amrex::MultiFab const*> const& a_beta)
{
  //------------------------------------------------------------------------
  // if a container for soret fluxes is provided, fill it
  const int need_soret_fluxes = (a_spsoretfluxes.empty()) ? 0 : 1;

  //------------------------------------------------------------------------
  // Compute T gradients and do average down to get gradients consistent across
  // levels Get the temperature BCRec
  constexpr int do_avgDown = 1;
  auto bcRecTemp = fetchBCRecArray(TEMP, 1);
  auto bcRecSpec = fetchBCRecArray(FIRSTSPEC, NUM_SPECIES);

  constexpr int nGrow = 0; // No need for ghost face on fluxes
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> gradT(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto& ba = grids[lev];
    const auto& factory = Factory(lev);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      gradT[lev][idim].define(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dmap[lev],
        1, nGrow, amrex::MFInfo(), factory);
      gradT[lev][idim].setVal(0.0);
    }
  }
  getDiffusionOp()->computeGradient(
    GetVecOfArrOfPtrs(gradT), {}, // Don't need the laplacian out
    a_temp, {}, bcRecTemp[0], do_avgDown);

  //------------------------------------------------------------------------
  // add Soret term to species fluxes
  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get edge diffusivity
    constexpr int doZeroVisc = 1;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> beta_ec = getDiffusivity(
      lev, NUM_SPECIES + 2, NUM_SPECIES, doZeroVisc, bcRecSpec, *a_beta[lev]);

    const amrex::Box& domain = geom[lev].Domain();
    const bool use_harmonic_avg = m_harm_avg_cen2edge != 0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
      amrex::FArrayBox T_ed;
      for (amrex::MFIter mfi(*a_beta[lev], amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

          // Get edge centered rhoYs
          const amrex::Box ebx = mfi.nodaltilebox(idim);
          const amrex::Box& edomain = amrex::surroundingNodes(domain, idim);

          // Get edge centered temps
          T_ed.resize(ebx, 1);
          amrex::Elixir T_el = T_ed.elixir();

          auto const& T_arr = a_temp[lev]->const_array(mfi);
          const auto& Ted_arr = T_ed.array(0);
          const auto bc_lo_temp = bcRecTemp[0].lo(idim);
          const auto bc_hi_temp = bcRecTemp[0].hi(idim);
          amrex::ParallelFor(
            ebx,
            [idim, bc_lo_temp, bc_hi_temp, use_harmonic_avg, T_arr, Ted_arr,
             edomain] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              const int idx[3] = {i, j, k};
              const bool on_lo =
                ((bc_lo_temp == amrex::BCType::ext_dir) &&
                 (idx[idim] <= edomain.smallEnd(idim)));
              const bool on_hi =
                ((bc_hi_temp == amrex::BCType::ext_dir) &&
                 (idx[idim] >= edomain.bigEnd(idim)));
              cen2edg_cpp(
                i, j, k, idim, 1, use_harmonic_avg, on_lo, on_hi, T_arr,
                Ted_arr);
            });

          auto const& T = T_ed.const_array(0);
          auto const& gradT_ar = gradT[lev][idim].const_array(mfi);
          auto const& beta_ar = beta_ec[idim].const_array(mfi);
          auto const& spFlux_ar = a_spfluxes[lev][idim]->array(mfi);
          auto const& spsoretFlux_ar =
            (need_soret_fluxes) != 0
              ? a_spsoretfluxes[lev][idim]->array(mfi)
              : a_spfluxes[lev][idim]->array(mfi); // Dummy unused Array4

          // Soret flux is : - rho * D_m * chi_m * \nabla T / T
          // with beta_m = rho * D_m * chi_m below
          amrex::ParallelFor(
            ebx, NUM_SPECIES,
            [gradT_ar, beta_ar, T,
             spFlux_ar] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
              spFlux_ar(i, j, k, n) -=
                beta_ar(i, j, k, n) * gradT_ar(i, j, k) / T(i, j, k);
            });

          if (need_soret_fluxes != 0) {
            amrex::ParallelFor(
              ebx, NUM_SPECIES,
              [gradT_ar, beta_ar, T, spsoretFlux_ar] AMREX_GPU_DEVICE(
                int i, int j, int k, int n) noexcept {
                spsoretFlux_ar(i, j, k, n) =
                  -beta_ar(i, j, k, n) * gradT_ar(i, j, k) / T(i, j, k);
              });
          }
        }
      }
    }
  }
}

void
PeleLM::computeSpeciesEnthalpyFlux(
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>& a_fluxes,
  amrex::Vector<amrex::MultiFab const*> const& a_temp)
{

  BL_PROFILE("PeleLMeX::computeSpeciesEnthalpyFlux()");

  // Get the species BCRec
  auto bcRecSpec = fetchBCRecArray(FIRSTSPEC, NUM_SPECIES);
  auto const* leosparm = eos_parms.device_parm();

  for (int lev = 0; lev <= finest_level; ++lev) {

#ifdef AMREX_USE_EB
    auto const& ebfact = EBFactory(lev);
#endif
    //------------------------------------------------------------------------
    // Compute the cell-centered species enthalpies
    constexpr int nGrow = 1;
    amrex::MultiFab Enth(
      grids[lev], dmap[lev], NUM_SPECIES, nGrow, amrex::MFInfo(), Factory(lev));

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(Enth, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box& gbx = mfi.growntilebox();
      auto const& Temp_arr = a_temp[lev]->const_array(mfi);
      auto const& Hi_arr = Enth.array(mfi);

#ifdef AMREX_USE_EB
      auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
      auto const& flag = flagfab.const_array();
      if (flagfab.getType(gbx) == amrex::FabType::covered) { // Covered boxes
        amrex::ParallelFor(
          gbx, [Hi_arr] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            Hi_arr(i, j, k) = 0.0;
          });
      } else if (
        flagfab.getType(gbx) != amrex::FabType::regular) { // EB containing
                                                           // boxes
        amrex::ParallelFor(
          gbx, [Hi_arr, flag, Temp_arr,
                leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (flag(i, j, k).isCovered()) {
              Hi_arr(i, j, k) = 0.0;
            } else {
              getHGivenT(i, j, k, Temp_arr, Hi_arr, leosparm);
            }
          });
      } else
#endif
      {
        amrex::ParallelFor(
          gbx, [Temp_arr, Hi_arr,
                leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            getHGivenT(i, j, k, Temp_arr, Hi_arr, leosparm);
          });
      }
    }

    //------------------------------------------------------------------------
    // Get the face-centered species enthalpies
    constexpr int doZeroVisc = 0;
    constexpr int addTurbContrib = 0;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> Enth_ec = getDiffusivity(
      lev, 0, NUM_SPECIES, doZeroVisc, bcRecSpec, Enth, addTurbContrib);

    //------------------------------------------------------------------------
    // Compute \sum_k { \Flux_k * h_k }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(Enth, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const amrex::Box& ebox = mfi.nodaltilebox(idim);
        auto const& spflux_ar = a_fluxes[lev][idim]->const_array(mfi, 0);
        auto const& enthflux_ar =
          a_fluxes[lev][idim]->array(mfi, NUM_SPECIES + 1);
        auto const& enth_ar = Enth_ec[idim].const_array(mfi);
        amrex::ParallelFor(
          ebox, [enthflux_ar, spflux_ar,
                 enth_ar] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            enthflux_ar(i, j, k) = 0.0;
            for (int n = 0; n < NUM_SPECIES; ++n) {
              enthflux_ar(i, j, k) +=
                spflux_ar(i, j, k, n) * enth_ar(i, j, k, n);
            }
          });
      }
    }
  }
}

void
PeleLM::differentialDiffusionUpdate(
  const std::unique_ptr<AdvanceAdvData>& advData,
  const std::unique_ptr<AdvanceDiffData>& diffData)
{
  BL_PROFILE("PeleLMeX::differentialDiffusionUpdate()");

  // Refill the AmrNewTime ghost cells of the scalars about to be
  // diffused.  See PeleLM::diffuseVelocity for the analogous fix and
  // the LidDrivenCavity regression for the discovery of this class
  // of bug: the implicit Crank-Nicholson solve in diffuse_scalar
  // pulls its Dirichlet boundary value from the ghost cells of its
  // input MultiFab when no explicit a_boundary is supplied.  Scalar
  // advection between the last fillPatchState (in oneSDC) and this
  // routine writes new interior values without refreshing the
  // ghosts, so the implicit solve would otherwise see stale (or, in
  // pathological cases, zero) Dirichlet data at Inflow faces.  For
  // typical Pele setups where Inflow scalars are static this would
  // be benign, but it is wrong as soon as bcnormal becomes time-
  // dependent and is hostile to anyone porting moving-wall- or
  // moving-Dirichlet-style problems to PeleLMeX.  Refilling now keeps
  // the implicit solve consistent with whatever bcnormal currently
  // returns at AmrNewTime.
  fillPatchSpecies(AmrNewTime);
  if (m_nAux > 0) {
    fillPatchAux(AmrNewTime);
  }
  // Temperature ghost cells are managed explicitly by the
  // deltaTIter_prepare / deltaTIter_update pair below (which sets T
  // and its ghost to zero before each deltaT solve and refills via
  // fillPatchTemp after).  No refill needed here.

  //------------------------------------------------------------------------
  // Setup fluxes
  // [0:NUM_SPECIES-1] Species     : \Flux_k
  // [NUM_SPECIES]     Temperature : - \lambda \nabla T
  // [NUM_SPECIES+1]   DiffDiff    : \sum_k ( h_k * \Flux_k )
  constexpr int nGrow = 0; // No need for ghost face on fluxes
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes(
    finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes_aux(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto& ba = grids[lev];
    const auto& factory = Factory(lev);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      fluxes[lev][idim].define(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dmap[lev],
        NUM_SPECIES + 2, nGrow, amrex::MFInfo(), factory);
      if (m_nAux > 0) {
        fluxes_aux[lev][idim].define(
          amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)),
          dmap[lev], m_nAux, nGrow, amrex::MFInfo(), factory);
      }
    }
  }
#ifdef AMREX_USE_EB
  amrex::Vector<amrex::MultiFab> EBfluxes;
  if (m_isothermalEB != 0) {
    EBfluxes.reserve(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      EBfluxes.emplace_back(
        grids[lev], dmap[lev], 1, nGrow, amrex::MFInfo(), Factory(lev));
    }
  }
#endif

  //------------------------------------------------------------------------
  // Convert species forcing into actual solve RHS by *dt and adding rhoY^{n}
  // Could have done it at the same time the forcing is built, but this is
  // clearer

  const auto dt = m_dt;
  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get t^{n} data pointer
    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);

    auto const& state_ma = ldata_p->state.const_arrays();
    auto const& fY_ma = advData->Forcing[lev].arrays();

    auto const& aux_ma =
      (m_nAux > 0) ? ldata_p->auxiliaries.const_arrays() : state_ma;
    auto const& fAux_ma =
      (m_nAux > 0) ? advData->Forcing_aux[lev].arrays() : fY_ma;

    amrex::ParallelFor(
      advData->Forcing[lev], amrex::IntVect(0), NUM_SPECIES,
      [state_ma, fY_ma,
       dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        amrex::Array4<amrex::Real const> rhoY(state_ma[box_no], FIRSTSPEC);
        fY_ma[box_no](i, j, k, n) *= dt;
        fY_ma[box_no](i, j, k, n) += rhoY(i, j, k, n);
      });
    amrex::ParallelFor(
      advData->Forcing[lev], amrex::IntVect(0), m_nAux,
      [aux_ma, fAux_ma,
       dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        fAux_ma[box_no](i, j, k, n) *= dt;
        fAux_ma[box_no](i, j, k, n) += aux_ma[box_no](i, j, k, n);
      });
  }
  amrex::Gpu::streamSynchronize();

  //------------------------------------------------------------------------
  // Species diffusion solve
  // Get the species BCRec
  auto bcRecSpec = fetchBCRecArray(FIRSTSPEC, NUM_SPECIES);
  auto bcRecAux = fetchBCRecAuxArray(0, m_nAux);
  amrex::Vector<amrex::MultiFab> spec_boundary;
  if (m_soret_boundary_override != 0) {
    spec_boundary.reserve(finest_level + 1);
    // this is the same regardless of lagged or not
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
      spec_boundary.emplace_back(
        grids[lev], dmap[lev], NUM_SPECIES, 1, amrex::MFInfo(), Factory(lev));

      // if we have a mix of Dirichlet and Isothermal walls, we need to give
      // Dirichlet boundaries and divide by density since diffuse_scalar doesn't
      // touch this boundary MF

      auto const& sp_bound_ma = spec_boundary[lev].arrays();
      auto const& state_ma = ldata_p->state.const_arrays();
      amrex::ParallelFor(
        spec_boundary[lev], spec_boundary[lev].nGrowVect(), NUM_SPECIES,
        [sp_bound_ma, state_ma] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          amrex::Array4<amrex::Real const> rhoY(state_ma[box_no], FIRSTSPEC);
          amrex::Array4<amrex::Real const> rho(state_ma[box_no], DENSITY);
          sp_bound_ma[box_no](i, j, k, n) = rhoY(i, j, k, n) / rho(i, j, k);
        });
    }
    amrex::Gpu::streamSynchronize();

    // correct boundary, we have lagged fluxes so lets use them
    correctIsothermalBoundary(
      AmrNewTime, GetVecOfPtrs(spec_boundary),
      GetVecOfArrOfPtrs(diffData->wbar_fluxes),
      GetVecOfArrOfPtrs(diffData->soret_fluxes));
  }

#ifdef PELE_USE_PLASMA
  // Solve for \widetilda{rhoY^{np1,kp1}}
  // -> return the uncorrected fluxes^{np1,kp1}
  // -> and the partially updated species (not including wbar or flux
  // correction)
  getMCDiffusionOp(NUM_SPECIES - NUM_IONS)
    ->diffuse_scalar(
      GetVecOfPtrs(getSpeciesVect(AmrNewTime)), 0,
      GetVecOfConstPtrs(advData->Forcing), 0, GetVecOfArrOfPtrs(fluxes), 0,
      GetVecOfConstPtrs(
        getDensityVect(AmrNewTime)), // this is the acoeff of LinOp
      GetVecOfConstPtrs(
        getDensityVect(AmrNewTime)), // this triggers proper scaling by density
      GetVecOfConstPtrs(getDiffusivityVect(AmrNewTime)), 0, bcRecSpec,
      NUM_SPECIES - NUM_IONS, 0, m_dt, {});
  // Ions one by one
  for (int n = 0; n < NUM_IONS; ++n) {
    auto bcRecIons = fetchBCRecArray(FIRSTSPEC + NUM_SPECIES - NUM_IONS + n, 1);
    getDiffusionOp()->diffuse_scalar(
      GetVecOfPtrs(getSpeciesVect(AmrNewTime)), NUM_SPECIES - NUM_IONS + n,
      GetVecOfConstPtrs(advData->Forcing), NUM_SPECIES - NUM_IONS + n,
      GetVecOfArrOfPtrs(fluxes), NUM_SPECIES - NUM_IONS + n,
      GetVecOfConstPtrs(
        getDensityVect(AmrNewTime)), // this is the acoeff of LinOp
      GetVecOfConstPtrs(
        getDensityVect(AmrNewTime)), // this triggers proper scaling by density
      GetVecOfConstPtrs(getDiffusivityVect(AmrNewTime)), 0, bcRecIons, 1, 0,
      m_dt, {});
  }
#else
  // Solve for \widetilda{rhoY^{np1,kp1}}
  // -> return the uncorrected fluxes^{np1,kp1}
  // -> and the partially updated species (not including wbar or flux
  // correction)
  getMCDiffusionOp(NUM_SPECIES)
    ->diffuse_scalar(
      GetVecOfPtrs(getSpeciesVect(AmrNewTime)), 0,
      GetVecOfConstPtrs(advData->Forcing), 0, GetVecOfArrOfPtrs(fluxes), 0,
      GetVecOfConstPtrs(
        getDensityVect(AmrNewTime)), // this is the acoeff of LinOp
      GetVecOfConstPtrs(
        getDensityVect(AmrNewTime)), // this triggers proper scaling by density
      GetVecOfConstPtrs(getDiffusivityVect(AmrNewTime)), 0, bcRecSpec,
      NUM_SPECIES, 0, m_dt, GetVecOfConstPtrs(spec_boundary));
  if (m_nAux > 0) {
    // Solve for \widetilde{c^{np1,kp1}}
    // Don't need to provide acoeff, or density.
    getMCDiffusionOp(m_nAux)->diffuse_scalar(
      GetVecOfPtrs(getAuxVect(AmrNewTime)), 0,
      GetVecOfConstPtrs(advData->Forcing_aux), 0, GetVecOfArrOfPtrs(fluxes_aux),
      0, {}, {}, GetVecOfConstPtrs(getAuxDiffusivityVect(AmrNewTime)), 0,
      bcRecAux, m_nAux, 0, m_dt, {});
  }
#endif

  // Add lagged Wbar term
  // Computed in computeDifferentialDiffusionTerms at t^{n} if first SDC
  // iteration, t^{np1,k} otherwise
  if (m_use_wbar != 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {

      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          const amrex::Box& ebx = mfi.nodaltilebox(idim);
          auto const& flux_spec = fluxes[lev][idim].array(mfi);
          auto const& flux_wbar =
            diffData->wbar_fluxes[lev][idim].const_array(mfi);
          amrex::ParallelFor(
            ebx, NUM_SPECIES,
            [flux_spec,
             flux_wbar] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
              flux_spec(i, j, k, n) += flux_wbar(i, j, k, n);
            });
        }
      }
    }
  }
  if (m_use_soret != 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(ldata_p->state, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          const amrex::Box& ebx = mfi.nodaltilebox(idim);
          auto const& flux_spec = fluxes[lev][idim].array(mfi);
          auto const& flux_soret =
            diffData->soret_fluxes[lev][idim].const_array(mfi);
          amrex::ParallelFor(
            ebx, NUM_SPECIES,
            [flux_spec,
             flux_soret] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
              flux_spec(i, j, k, n) += flux_soret(i, j, k, n);
            });
        }
      }
    }
  }

  // FillPatch the new species before computing flux correction terms
  fillPatchSpecies(AmrNewTime);

  // Adjust species diffusion fluxes to ensure their sum is zero
  adjustSpeciesFluxes<pele::physics::PhysicsType::eos_type>(
    GetVecOfArrOfPtrs(fluxes), GetVecOfConstPtrs(getSpeciesVect(AmrNewTime)));

  // Average down fluxes^{np1,kp1}
  getDiffusionOp()->avgDownFluxes(GetVecOfArrOfPtrs(fluxes), 0, NUM_SPECIES);

  // Compute diffusion term D^{np1,kp1} (or Dhat)
  fluxDivergence(
    GetVecOfPtrs(diffData->Dhat), 0, GetVecOfArrOfPtrs(fluxes), 0, NUM_SPECIES,
    1, -1.0);
  unscaleMappedDivergence(GetVecOfPtrs(diffData->Dhat), 0, NUM_SPECIES);

  // Repeat for auxiliaries
  if (m_nAux > 0) {
    // Average down fluxes^{np1,kp1}
    getDiffusionOp()->avgDownFluxes(GetVecOfArrOfPtrs(fluxes_aux), 0, m_nAux);
    fluxDivergence(
      GetVecOfPtrs(diffData->Dhat_aux), 0, GetVecOfArrOfPtrs(fluxes_aux), 0,
      m_nAux, 1, -1.0);
    unscaleMappedDivergence(GetVecOfPtrs(diffData->Dhat_aux), 0, m_nAux);
  }

  // Update species
  // Remove the Wbar and Soret terms because we included them both the dhat and
  // the forcing.
  if (m_use_wbar != 0 && m_use_soret != 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {

      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);

      auto const& state_ma = ldata_p->state.arrays();
      auto const& dhat_ma = diffData->Dhat[lev].const_arrays();
      auto const& force_ma = advData->Forcing[lev].const_arrays();
      auto const& dwbar_ma = diffData->Dwbar[lev].const_arrays();
      auto const& dT_ma = diffData->DT[lev].const_arrays();

      amrex::ParallelFor(
        ldata_p->state, amrex::IntVect(0), NUM_SPECIES,
        [state_ma, dhat_ma, force_ma, dwbar_ma, dT_ma,
         dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          amrex::Array4<amrex::Real> rhoY(state_ma[box_no], FIRSTSPEC);
          rhoY(i, j, k, n) =
            force_ma[box_no](i, j, k, n) +
            dt * (dhat_ma[box_no](i, j, k, n) - dwbar_ma[box_no](i, j, k, n) -
                  dT_ma[box_no](i, j, k, n));
        });
    }
  } else if (m_use_wbar != 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {

      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);

      auto const& state_ma = ldata_p->state.arrays();
      auto const& dhat_ma = diffData->Dhat[lev].const_arrays();
      auto const& force_ma = advData->Forcing[lev].const_arrays();

      auto const& dwbar_ma = diffData->Dwbar[lev].const_arrays();
      amrex::ParallelFor(
        ldata_p->state, amrex::IntVect(0), NUM_SPECIES,
        [state_ma, dhat_ma, force_ma, dwbar_ma,
         dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          amrex::Array4<amrex::Real> rhoY(state_ma[box_no], FIRSTSPEC);
          rhoY(i, j, k, n) =
            force_ma[box_no](i, j, k, n) +
            dt * (dhat_ma[box_no](i, j, k, n) - dwbar_ma[box_no](i, j, k, n));
        });
    }
  } else if (m_use_soret != 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {

      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);

      auto const& state_ma = ldata_p->state.arrays();
      auto const& dhat_ma = diffData->Dhat[lev].const_arrays();
      auto const& force_ma = advData->Forcing[lev].const_arrays();

      auto const& dT_ma = diffData->DT[lev].const_arrays();
      amrex::ParallelFor(
        ldata_p->state, amrex::IntVect(0), NUM_SPECIES,
        [state_ma, dhat_ma, force_ma, dT_ma,
         dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          amrex::Array4<amrex::Real> rhoY(state_ma[box_no], FIRSTSPEC);
          rhoY(i, j, k, n) =
            force_ma[box_no](i, j, k, n) +
            dt * (dhat_ma[box_no](i, j, k, n) - dT_ma[box_no](i, j, k, n));
        });
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {

      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);

      auto const& state_ma = ldata_p->state.arrays();
      auto const& dhat_ma = diffData->Dhat[lev].const_arrays();
      auto const& force_ma = advData->Forcing[lev].const_arrays();

      amrex::ParallelFor(
        ldata_p->state, amrex::IntVect(0), NUM_SPECIES,
        [state_ma, dhat_ma, force_ma,
         dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          amrex::Array4<amrex::Real> rhoY(state_ma[box_no], FIRSTSPEC);
          rhoY(i, j, k, n) =
            force_ma[box_no](i, j, k, n) + dt * dhat_ma[box_no](i, j, k, n);
        });
    }
  }
  if (m_nAux > 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {

      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);

      auto const& aux_ma = ldata_p->auxiliaries.arrays();
      auto const& dhat_aux_ma = diffData->Dhat_aux[lev].const_arrays();
      auto const& force_aux_ma = advData->Forcing_aux[lev].const_arrays();
      amrex::ParallelFor(
        ldata_p->state, amrex::IntVect(0), m_nAux,
        [aux_ma, dhat_aux_ma, force_aux_ma,
         dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          aux_ma[box_no](i, j, k, n) = force_aux_ma[box_no](i, j, k, n) +
                                       dt * dhat_aux_ma[box_no](i, j, k, n);
        });
    }
  }
  amrex::Gpu::streamSynchronize();

  // FillPatch species again before going into the enthalpy solve
  fillPatchSpecies(AmrNewTime);

  // If doing species balances, compute face domain integrals
  // using level 0 since we've averaged down the fluxes already
  if (m_sdcIter == m_nSDCmax && (m_do_speciesBalance != 0)) {
    addRhoYFluxes(GetArrOfConstPtrs(fluxes[0]), geom[0]);
  }
  //------------------------------------------------------------------------

  //------------------------------------------------------------------------
  if (pele::physics::PhysicsType::eos_type::identifier() != "Manifold") {
    // Enthalpy iterative diffusion solve
    // Get the temperature BCRec
    auto bcRecTemp = fetchBCRecArray(TEMP, 1);

    // Fourier: - \lambda \nabla T
    constexpr int do_avgDown = 0;
#ifdef AMREX_USE_EB
    if (m_isothermalEB != 0) {
      // Set up EB dirichlet value and diffusivity
      amrex::Vector<amrex::MultiFab> EBdiff;
      EBdiff.reserve(finest_level + 1);
      for (int lev = 0; lev <= finest_level; ++lev) {
        EBdiff.emplace_back(
          grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), EBFactory(lev));
        getEBDiff(lev, AmrNewTime, EBdiff[lev], NUM_SPECIES);
      }
      getDiffusionOp()->computeDiffFluxes(
        GetVecOfArrOfPtrs(fluxes), NUM_SPECIES, GetVecOfPtrs(EBfluxes), 0,
        GetVecOfConstPtrs(getTempVect(AmrNewTime)), 0, {},
        GetVecOfConstPtrs(getDiffusivityVect(AmrNewTime)), NUM_SPECIES,
        GetVecOfConstPtrs(getEBState(TEMP, 1, AmrNewTime)),
        GetVecOfConstPtrs(EBdiff), bcRecTemp, 1, do_avgDown, {});
    } else
#endif
    {
      getDiffusionOp()->computeDiffFluxes(
        GetVecOfArrOfPtrs(fluxes), NUM_SPECIES,
        GetVecOfConstPtrs(getTempVect(AmrNewTime)), 0, {},
        GetVecOfConstPtrs(getDiffusivityVect(AmrNewTime)), NUM_SPECIES,
        bcRecTemp, 1, do_avgDown, {});
    }

    // Differential diffusion term: \sum_k ( h_k * \Flux_k )
    computeSpeciesEnthalpyFlux(
      GetVecOfArrOfPtrs(fluxes), GetVecOfConstPtrs(getTempVect(AmrNewTime)));

    // average_down enthalpy fluxes
    getDiffusionOp()->avgDownFluxes(GetVecOfArrOfPtrs(fluxes), NUM_SPECIES, 2);

    // Compute diffusion term D^{np1,kp1} of Fourier and DifferentialDiffusion
#ifdef AMREX_USE_EB
    if (m_isothermalEB != 0) {
      // Do Fourier with EBflux first then differential diffusion
      fluxDivergence(
        GetVecOfPtrs(diffData->Dhat), NUM_SPECIES, GetVecOfArrOfPtrs(fluxes),
        NUM_SPECIES, GetVecOfPtrs(EBfluxes), 0, 1, 1, -1.0);
      fluxDivergence(
        GetVecOfPtrs(diffData->Dhat), NUM_SPECIES + 1,
        GetVecOfArrOfPtrs(fluxes), NUM_SPECIES + 1, 1, 1, -1.0);
    } else
#endif
    {
      fluxDivergence(
        GetVecOfPtrs(diffData->Dhat), NUM_SPECIES, GetVecOfArrOfPtrs(fluxes),
        NUM_SPECIES, 2, 1, -1.0);
    }
    unscaleMappedDivergence(GetVecOfPtrs(diffData->Dhat), NUM_SPECIES, 2);

    //------------------------------------------------------------------------
    // Recompute RhoH (before deltaT iters: we've updated species at the new
    // state already, we should modify enthalpy to reflect this update before
    // the iterative procedure to update enthalpy due to T diffusion)

    auto const* leosparm = eos_parms.device_parm();
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
      auto const& sma = ldata_p->state.arrays();
      amrex::ParallelFor(
        ldata_p->state, [sma, leosparm] AMREX_GPU_DEVICE(
                          int box_no, int i, int j, int k) noexcept {
          getRHmixGivenTY(
            i, j, k, amrex::Array4<amrex::Real const>(sma[box_no], DENSITY),
            amrex::Array4<amrex::Real const>(sma[box_no], FIRSTSPEC),
            amrex::Array4<amrex::Real const>(sma[box_no], TEMP),
            amrex::Array4<amrex::Real>(sma[box_no], RHOH), leosparm);
        });
    }
    amrex::Gpu::streamSynchronize();

    //------------------------------------------------------------------------
    // delta(T) iterations
    if (m_deltaT_verbose != 0) {
      amrex::Print() << " Iterative solve for deltaT \n";
    }

    //------------------------------------------------------------------------
    // Temporary data holders
    amrex::Vector<amrex::MultiFab> rhs;
    rhs.reserve(finest_level + 1); // Linear deltaT solve RHS
    amrex::Vector<amrex::MultiFab> Tsave;
    Tsave.reserve(finest_level + 1); // Storage of T while working on deltaT
    amrex::Vector<amrex::MultiFab> RhoCp;
    RhoCp.reserve(finest_level + 1); // Acoeff of the linear solve
    for (int lev = 0; lev <= finest_level; ++lev) {
      rhs.emplace_back(
        grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
      Tsave.emplace_back(
        grids[lev], dmap[lev], 1, 1, amrex::MFInfo(), Factory(lev));
      RhoCp.emplace_back(
        grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
    }

    // DeltaT norm
    amrex::Real deltaT_norm = 0.0;

    // Reset deltaT iteration counter at the start of deltaT loop for MLMG fail
    // tracking
    m_deltaTIter = 0;

    for (int dTiter = 0; dTiter < m_deltaTIterMax &&
                         (dTiter == 0 || deltaT_norm >= m_deltaT_norm_max);
         ++dTiter) {

      m_deltaTIter = dTiter;
      // Prepare the deltaT iteration linear solve:
      // -> Assemble the RHS
      // -> Compute current value of \rho * \Cp_{mix}
      // -> Save current T^{np1,kp1}
      // -> set T^{np1,kp1} to zero
      deltaTIter_prepare(
        GetVecOfPtrs(rhs), GetVecOfPtrs(Tsave), GetVecOfPtrs(RhoCp), advData,
        diffData);

      // Diffuse deltaT
#ifdef AMREX_USE_EB
      if (m_isothermalEB != 0) {
        // Set up EB dirichlet value and diffusivity
        // Dirichlet value is deltaT
        amrex::Vector<amrex::MultiFab> EBvalue;
        EBvalue.reserve(finest_level + 1);
        amrex::Vector<amrex::MultiFab> EBdiff;
        EBdiff.reserve(finest_level + 1);
        for (int lev = 0; lev <= finest_level; ++lev) {
          EBvalue.emplace_back(
            grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), EBFactory(lev));
          EBdiff.emplace_back(
            grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), EBFactory(lev));
          getEBDiff(lev, AmrNewTime, EBdiff[lev], NUM_SPECIES);
          EBvalue[lev].setVal(0.0);
        }
        getDiffusionOp()->diffuse_scalar(
          GetVecOfPtrs(getTempVect(AmrNewTime)), 0, GetVecOfConstPtrs(EBvalue),
          0, GetVecOfConstPtrs(rhs), 0, GetVecOfArrOfPtrs(fluxes), NUM_SPECIES,
          GetVecOfConstPtrs(RhoCp), {},
          GetVecOfConstPtrs(getDiffusivityVect(AmrNewTime)), NUM_SPECIES,
          GetVecOfConstPtrs(EBdiff), 0, bcRecTemp, 1, 0, m_dt, {});
      } else
#endif
      {
        getDiffusionOp()->diffuse_scalar(
          GetVecOfPtrs(getTempVect(AmrNewTime)), 0, GetVecOfConstPtrs(rhs), 0,
          GetVecOfArrOfPtrs(fluxes), NUM_SPECIES, GetVecOfConstPtrs(RhoCp), {},
          GetVecOfConstPtrs(getDiffusivityVect(AmrNewTime)), NUM_SPECIES,
          bcRecTemp, 1, 0, m_dt, {});
      }

      // Post deltaT iteration linear solve
      // -> evaluate deltaT_norm
      // -> add deltaT to T^{np1,kp1}
      // -> recompute enthalpy fluxes
      // -> recompute rhoH
#ifdef AMREX_USE_EB
      if (m_isothermalEB != 0) {
        deltaTIter_update(
          dTiter, GetVecOfArrOfPtrs(fluxes), GetVecOfPtrs(EBfluxes),
          GetVecOfConstPtrs(Tsave), diffData, deltaT_norm);
      } else
#endif
      {
        deltaTIter_update(
          dTiter, GetVecOfArrOfPtrs(fluxes), {}, GetVecOfConstPtrs(Tsave),
          diffData, deltaT_norm);
      }

      // Check for convergence failure
      if (
        (dTiter == m_deltaTIterMax - 1) && (deltaT_norm > m_deltaT_norm_max)) {
        if (m_crashOnDeltaTFail != 0) {
          amrex::Abort("deltaT_iters not converged !");
        } else {
          amrex::Print() << "deltaT_iters not converged !\n";
        }
      }
    }
  }
  //------------------------------------------------------------------------
}

void
PeleLM::deltaTIter_prepare(
  const amrex::Vector<amrex::MultiFab*>& a_rhs,
  const amrex::Vector<amrex::MultiFab*>& a_Tsave,
  const amrex::Vector<amrex::MultiFab*>& a_rhoCp,
  const std::unique_ptr<AdvanceAdvData>& advData,
  const std::unique_ptr<AdvanceDiffData>& diffData)
{
  auto const* leosparm = eos_parms.device_parm();
  const amrex::Real dtinv = 1.0 / m_dt;
  for (int lev = 0; lev <= finest_level; ++lev) {

    auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
    auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);

    auto const& state_old_ma = ldataOld_p->state.const_arrays();
    auto const& state_new_ma = ldataNew_p->state.const_arrays();
    auto const& force_ma = advData->Forcing[lev].const_arrays();
    auto const& dhat_ma = diffData->Dhat[lev].const_arrays();
    auto const& rhs_ma = a_rhs[lev]->arrays();
    auto const& rhocp_ma = a_rhoCp[lev]->arrays();
    auto const& tsave_ma = a_Tsave[lev]->arrays();

    amrex::ParallelFor(
      ldataNew_p->state,
      [state_old_ma, state_new_ma, force_ma, dhat_ma, rhs_ma, rhocp_ma,
       tsave_ma, dtinv, leosparm,
       dt = m_dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        amrex::Array4<amrex::Real const> rhoH_o(state_old_ma[box_no], RHOH);
        amrex::Array4<amrex::Real const> rhoH_n(state_new_ma[box_no], RHOH);
        amrex::Array4<amrex::Real const> force(force_ma[box_no], NUM_SPECIES);
        amrex::Array4<amrex::Real const> fourier(dhat_ma[box_no], NUM_SPECIES);
        amrex::Array4<amrex::Real const> diffDiff(
          dhat_ma[box_no], NUM_SPECIES + 1);
        // Assemble deltaT RHS
        rhs_ma[box_no](i, j, k) =
          dt * ((rhoH_o(i, j, k) - rhoH_n(i, j, k)) * dtinv + force(i, j, k) +
                fourier(i, j, k) + diffDiff(i, j, k));

        amrex::Array4<amrex::Real const> rho(state_new_ma[box_no], DENSITY);
        amrex::Array4<amrex::Real const> rhoY(state_new_ma[box_no], FIRSTSPEC);
        amrex::Array4<amrex::Real const> T(state_new_ma[box_no], TEMP);
        // Get \rho * Cp_{mix}
        getCpmixGivenRYT(i, j, k, rho, rhoY, T, rhocp_ma[box_no], leosparm);
        rhocp_ma[box_no](i, j, k) *= rho(i, j, k);

        // Save T
        tsave_ma[box_no](i, j, k) = T(i, j, k);
      });
    amrex::Gpu::streamSynchronize();
    // Set T^{np1} to zero
    // Include one ghost cell to ensure levelBC at zero for linear solve
    ldataNew_p->state.setVal(0.0, TEMP, 1, 1);
  }
}

void
PeleLM::deltaTIter_update(
  const int a_dtiter,
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>& a_fluxes,
  const amrex::Vector<amrex::MultiFab*>& a_ebfluxes,
  const amrex::Vector<amrex::MultiFab const*>& a_Tsave,
  const std::unique_ptr<AdvanceDiffData>& diffData,
  amrex::Real& a_deltaT_norm)
{
#ifndef AMREX_USE_EB
  amrex::ignore_unused(a_ebfluxes);
#endif

  //------------------------------------------------------------------------
  // Evaluate deltaT norm and add Tsave back into the new LevelData
  a_deltaT_norm = -1.0e12;
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
    a_deltaT_norm = amrex::max<amrex::Real>(
      a_deltaT_norm, ldata_p->state.norm0(TEMP, 0, false, true));
    amrex::MultiFab::Add(ldata_p->state, *a_Tsave[lev], 0, TEMP, 1, 0);
  }

  if (m_deltaT_verbose != 0) {
    amrex::Print() << "   DeltaT solve norm [" << a_dtiter
                   << "] = " << a_deltaT_norm << "\n";
  }

  // FillPatch the new temperature before going into the fluxe computation
  fillPatchTemp(AmrNewTime);

  //------------------------------------------------------------------------
  // Re-evaluate the enthalpy fluxes and divergence
  // Get the temperature BCRec
  auto bcRecTemp = fetchBCRecArray(TEMP, 1);

  // Fourier: - \lambda \nabla T
  constexpr int do_avgDown = 0;
#ifdef AMREX_USE_EB
  if (m_isothermalEB != 0) {
    // Set up EB dirichlet value and diffusivity
    amrex::Vector<amrex::MultiFab> EBdiff;
    EBdiff.reserve(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      EBdiff.emplace_back(
        grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), EBFactory(lev));
      getEBDiff(lev, AmrNewTime, EBdiff[lev], NUM_SPECIES);
    }
    getDiffusionOp()->computeDiffFluxes(
      a_fluxes, NUM_SPECIES, a_ebfluxes, 0,
      GetVecOfConstPtrs(getTempVect(AmrNewTime)), 0, {},
      GetVecOfConstPtrs(getDiffusivityVect(AmrNewTime)), NUM_SPECIES,
      GetVecOfConstPtrs(getEBState(TEMP, 1, AmrNewTime)),
      GetVecOfConstPtrs(EBdiff), bcRecTemp, 1, do_avgDown, {});
  } else
#endif
  {
    getDiffusionOp()->computeDiffFluxes(
      a_fluxes, NUM_SPECIES, GetVecOfConstPtrs(getTempVect(AmrNewTime)), 0, {},
      GetVecOfConstPtrs(getDiffusivityVect(AmrNewTime)), NUM_SPECIES, bcRecTemp,
      1, do_avgDown, {});
  }

  // Differential diffusion term: \sum_k ( h_k * \Flux_k )
  computeSpeciesEnthalpyFlux(
    a_fluxes, GetVecOfConstPtrs(getTempVect(AmrNewTime)));

  // average_down enthalpy fluxes
  getDiffusionOp()->avgDownFluxes(a_fluxes, NUM_SPECIES, 2);

  // Compute diffusion term D^{np1,kp1} of Fourier and DifferentialDiffusion
#ifdef AMREX_USE_EB
  if (m_isothermalEB != 0) {
    // Do Fourier with EBflux first then differential diffusion
    fluxDivergence(
      GetVecOfPtrs(diffData->Dhat), NUM_SPECIES, a_fluxes, NUM_SPECIES,
      a_ebfluxes, 0, 1, 1, -1.0);
    fluxDivergence(
      GetVecOfPtrs(diffData->Dhat), NUM_SPECIES + 1, a_fluxes, NUM_SPECIES + 1,
      1, 1, -1.0);
  } else
#endif
  {
    fluxDivergence(
      GetVecOfPtrs(diffData->Dhat), NUM_SPECIES, a_fluxes, NUM_SPECIES, 2, 1,
      -1.0);
  }
  unscaleMappedDivergence(GetVecOfPtrs(diffData->Dhat), NUM_SPECIES, 2);

  //------------------------------------------------------------------------
  // Recompute RhoH
  auto const* leosparm = eos_parms.device_parm();
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
    auto const& sma = ldata_p->state.arrays();
    amrex::ParallelFor(
      ldata_p->state, [sma, leosparm] AMREX_GPU_DEVICE(
                        int box_no, int i, int j, int k) noexcept {
        getRHmixGivenTY(
          i, j, k, amrex::Array4<amrex::Real const>(sma[box_no], DENSITY),
          amrex::Array4<amrex::Real const>(sma[box_no], FIRSTSPEC),
          amrex::Array4<amrex::Real const>(sma[box_no], TEMP),
          amrex::Array4<amrex::Real>(sma[box_no], RHOH), leosparm);
      });
  }
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::getScalarDiffForce(
  const std::unique_ptr<AdvanceAdvData>& advData,
  const std::unique_ptr<AdvanceDiffData>& diffData)
{

  const auto aux_advect_v = convertToDeviceVector(m_aux_advect);
  const auto aux_diffuse_v = convertToDeviceVector(m_DiffTypeAux);
  const auto* aux_advect_d = aux_advect_v.dataPtr();
  const auto* aux_diffuse_d = aux_diffuse_v.dataPtr();

  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get t^{n} data pointer
    auto* ldataR_p = getLevelDataReactPtr(lev);

    auto const& dn_ma = diffData->Dn[lev].const_arrays();
    auto const& dnp1_ma = diffData->Dnp1[lev].const_arrays();
    auto const& r_ma = ldataR_p->I_R.const_arrays();
    auto const& a_ma = advData->AofS[lev].const_arrays();
    auto const& ext_ma = m_extSource[lev]->const_arrays();
    auto const& f_ma = advData->Forcing[lev].arrays();

    auto const& dwbar_ma =
      (m_use_wbar != 0) ? diffData->Dwbar[lev].const_arrays() : dn_ma;
    auto const& dT_ma =
      (m_use_soret != 0) ? diffData->DT[lev].const_arrays() : dn_ma;
    auto const& f_aux_ma =
      (m_nAux > 0) ? advData->Forcing_aux[lev].arrays() : f_ma;
    auto const& a_aux_ma =
      (m_nAux > 0) ? advData->AofS_aux[lev].const_arrays() : dn_ma;
    auto const& dn_aux_ma =
      (m_nAux > 0) ? diffData->Dn_aux[lev].const_arrays() : dn_ma;
    auto const& dnp1_aux_ma =
      (m_nAux > 0) ? diffData->Dnp1_aux[lev].const_arrays() : dn_ma;

    amrex::ParallelFor(
      advData->Forcing[lev],
      [dn_ma, dnp1_ma, r_ma, a_ma, ext_ma, f_ma, dwbar_ma, dT_ma, f_aux_ma,
       a_aux_ma, dn_aux_ma, dnp1_aux_ma, do_react = m_do_react,
       use_wbar = m_use_wbar, use_soret = m_use_soret, dp0dt = m_dp0dt,
       is_closed_ch = m_closed_chamber, nAux = m_nAux, aux_advect_d,
       aux_diffuse_d] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        amrex::Array4<amrex::Real const> ddn(dn_ma[box_no], NUM_SPECIES + 1);
        amrex::Array4<amrex::Real const> ddnp1(
          dnp1_ma[box_no], NUM_SPECIES + 1);
        amrex::Array4<amrex::Real const> a(a_ma[box_no], FIRSTSPEC);
        amrex::Array4<amrex::Real const> extRhoY(ext_ma[box_no], FIRSTSPEC);
        amrex::Array4<amrex::Real const> extRhoH(ext_ma[box_no], RHOH);
        amrex::Array4<amrex::Real> fT(f_ma[box_no], NUM_SPECIES);
        buildDiffusionForcing(
          i, j, k, dn_ma[box_no], ddn, dnp1_ma[box_no], ddnp1, r_ma[box_no], a,
          dp0dt, is_closed_ch, do_react, f_ma[box_no], fT, dwbar_ma[box_no],
          dT_ma[box_no], extRhoY, extRhoH, use_wbar, use_soret,
          f_aux_ma[box_no], a_aux_ma[box_no], dn_aux_ma[box_no],
          dnp1_aux_ma[box_no], aux_advect_d, aux_diffuse_d, nAux);
      });
  }
  amrex::Gpu::streamSynchronize();

  // Fill forcing ghost cells
  if (advData->Forcing[0].nGrow() > 0) {
    fillpatch_forces(
      m_cur_time, GetVecOfPtrs(advData->Forcing), advData->Forcing[0].nGrow());
  }
  if (m_nAux > 0 && advData->Forcing_aux[0].nGrow() > 0) {
    fillpatch_forces(
      m_cur_time, GetVecOfPtrs(advData->Forcing_aux),
      advData->Forcing_aux[0].nGrow());
  }
}

void
PeleLM::computeDivTau(
  const TimeStamp a_time,
  const amrex::Vector<amrex::MultiFab*>& a_divtau,
  const int use_density,
  const amrex::Real scale)
{
  BL_PROFILE("PeleLMeX::computeDivTau()");
  // Get the density component BCRec to get viscosity on faces
  auto bcRec = fetchBCRecArray(DENSITY, 1);

  if (use_density != 0) {
    getDiffusionTensorOp()->compute_divtau(
      a_divtau, GetVecOfConstPtrs(getVelocityVect(a_time)),
      GetVecOfConstPtrs(getDensityVect(a_time)),
      GetVecOfConstPtrs(getViscosityVect(a_time)), bcRec[0], scale);
  } else {
    getDiffusionTensorOp()->compute_divtau(
      a_divtau, GetVecOfConstPtrs(getVelocityVect(a_time)), {},
      GetVecOfConstPtrs(getViscosityVect(a_time)), bcRec[0], scale);
  }
}

void
PeleLM::diffuseVelocity()
{
  BL_PROFILE("PeleLMeX::diffuseVelocity()");
  // Get the density component BCRec to get viscosity on faces
  auto bcRec = fetchBCRecArray(DENSITY, 1);

  // BUGFIX: the implicit Crank-Nicholson solve below calls setLevelBC
  // on the velocity MultiFab and pulls the Dirichlet boundary value
  // from its ghost cells.  After updateVelocity (the explicit
  // predictor) the new-time velocity ghost cells contain whatever
  // was there from the previous fillpatch, NOT the bcnormal value at
  // Inflow faces -- in particular they are zero for a moving-wall
  // tangential component that has not been refilled since.  Without
  // this refill the implicit corrector enforces u_face = 0 at the
  // Inflow boundary instead of the user-supplied value, so the only
  // wall-stress contribution comes from the explicit 0.5*divTau^n
  // half of the C-N split.  See the LidDrivenCavity regression test
  // for the test case that surfaces this.
  for (int lev = 0; lev <= finestLevel(); ++lev) {
    auto& vel_mf = getLevelDataPtr(lev, AmrNewTime)->state;
    setInflowBoundaryVel(vel_mf, lev, AmrNewTime);
  }

  // CrankNicholson 0.5 coeff
  const amrex::Real dt_lcl = 0.5 * m_dt;
  if (m_incompressible != 0) {
    getDiffusionTensorOp()->diffuse_velocity(
      GetVecOfPtrs(getVelocityVect(AmrNewTime)), {},
      GetVecOfConstPtrs(getViscosityVect(AmrNewTime)), bcRec[0], dt_lcl);
  } else {
    getDiffusionTensorOp()->diffuse_velocity(
      GetVecOfPtrs(getVelocityVect(AmrNewTime)),
      GetVecOfConstPtrs(getDensityVect(AmrHalfTime)),
      GetVecOfConstPtrs(getViscosityVect(AmrNewTime)), bcRec[0], dt_lcl);
  }
}

amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>
PeleLM::getDiffusionLinOpBC(
  amrex::Orientation::Side a_side, const amrex::BCRec& a_bc)
{
  amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> r;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (Geom(0).isPeriodic(idim)) {
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
        if (
          m_use_soret != 0 &&
          amrexbc == amrex::BCType::foextrap) { // should just catch
                                                // density/species/forcing for
                                                // isothermal walls
          auto phys_bc = (a_side == amrex::Orientation::low)
                           ? m_phys_bc.lo(idim)
                           : m_phys_bc.hi(idim);
          if (
            phys_bc == BoundaryCondition::BCNoSlipWallIsotherm ||
            phys_bc == BoundaryCondition::BCSlipWallIsotherm) {
            r[idim] = amrex::LinOpBCType::inhomogNeumann;
          }
        }
      } else if (amrexbc == amrex::BCType::reflect_odd) {
        r[idim] = amrex::LinOpBCType::reflect_odd;
      } else {
        r[idim] = amrex::LinOpBCType::bogus;
      }
    }
  }
  return r;
}

amrex::Vector<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>>
PeleLM::getDiffusionTensorOpBC(
  const amrex::Orientation::Side a_side,
  const amrex::Vector<amrex::BCRec>& a_bc)
{
  AMREX_ASSERT(a_bc.size() == AMREX_SPACEDIM);
  amrex::Vector<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>> r(
    AMREX_SPACEDIM);
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (Geom(0).isPeriodic(idim)) {
      AMREX_D_TERM(
        r[0][idim] = amrex::LinOpBCType::Periodic;
        , r[1][idim] = amrex::LinOpBCType::Periodic;
        , r[2][idim] = amrex::LinOpBCType::Periodic;);
    } else {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        auto amrexbc = (a_side == amrex::Orientation::low) ? a_bc[dir].lo(idim)
                                                           : a_bc[dir].hi(idim);
        if (amrexbc == amrex::BCType::ext_dir) {
          r[dir][idim] = amrex::LinOpBCType::Dirichlet;
        } else if (
          amrexbc == amrex::BCType::foextrap ||
          amrexbc == amrex::BCType::hoextrap ||
          amrexbc == amrex::BCType::reflect_even) {
          r[dir][idim] = amrex::LinOpBCType::Neumann;
        } else if (amrexbc == amrex::BCType::reflect_odd) {
          r[dir][idim] = amrex::LinOpBCType::reflect_odd;
        } else {
          r[dir][idim] = amrex::LinOpBCType::bogus;
        }
      }
    }
  }
  return r;
}
