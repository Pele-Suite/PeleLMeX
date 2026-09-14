#include <PeleLMeX.H>
#include <PeleLMeX_Utils.H>
#include <hydro_utils.H>
#include <AMReX_FillPatchUtil.H>
#include <PeleLMeX_BCfill.H>

void
PeleLM::predictVelocity(const std::unique_ptr<AdvanceAdvData>& advData)
{
  BL_PROFILE("PeleLMeX::predictVelocity()");

  // set umac boundaries to zero
  if (advData->umac[0][0].nGrow() > 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      AMREX_D_TERM(advData->umac[lev][0].setBndry(0.0);
                   , advData->umac[lev][1].setBndry(0.0);
                   , advData->umac[lev][2].setBndry(0.0););
    }
  }

  //----------------------------------------------------------------
  // Get viscous forces
  constexpr int nGrow_force = 1;
  amrex::Vector<amrex::MultiFab> divtau;
  divtau.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> velForces;
  velForces.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    divtau.emplace_back(
      grids[lev], dmap[lev], AMREX_SPACEDIM, 0, amrex::MFInfo(), Factory(lev));
    velForces.emplace_back(
      grids[lev], dmap[lev], AMREX_SPACEDIM, nGrow_force, amrex::MFInfo(),
      Factory(lev));
  }
  constexpr int use_density = 0;
  computeDivTau(AmrOldTime, GetVecOfPtrs(divtau), use_density);

  //----------------------------------------------------------------
  // Gather all the velocity forces
  // F = [ (gravity+...) - gradP + divTau ] / rho
  constexpr int add_gradP = 1;
  getVelForces(
    AmrOldTime, GetVecOfPtrs(divtau), GetVecOfPtrs(velForces), nGrow_force,
    add_gradP);

  //----------------------------------------------------------------
  // Mesh mapping: the Godunov extrapolation works on the uniform Xi grid,
  // so it is done on the Xi-velocity u~_d = u_d / fac_d (fac_d = dx_d/dXi_d
  // depends on Xi_d only), which is both the transported quantity and the
  // advection speed in Xi.  Its equation follows from the physical one:
  //   d u~_d/dt + sum_j u~_j d u~_d/dXi_j
  //       = F_d / fac_d - u~_d^2 (d fac_d/dXi_d) / fac_d
  // The predicted faces are converted back to physical velocities, which is
  // what macProject expects on entry.
  amrex::Vector<amrex::MultiFab> vel_xi;
  if (m_mesh_mapping) {
    vel_xi.reserve(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);
      const amrex::IntVect ngv = ldata_p->state.nGrowVect();
      vel_xi.emplace_back(
        grids[lev], dmap[lev], AMREX_SPACEDIM, ngv, amrex::MFInfo(),
        Factory(lev));
      AMREX_ASSERT(m_mesh_map->fac_cc(lev).nGrowVect().allGE(ngv));
      AMREX_ASSERT(m_mesh_map->fac_fc(lev, 0).nGrowVect().allGE(
        amrex::IntVect(nGrow_force + 1)));
      const auto dxinv = geom[lev].InvCellSizeArray();
      const auto& vel_ma = ldata_p->state.const_arrays();
      const auto& fac_ma = m_mesh_map->fac_cc(lev).const_arrays();
      const auto& uxi_ma = vel_xi[lev].arrays();
      amrex::ParallelFor(
        vel_xi[lev], ngv, AMREX_SPACEDIM,
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          uxi_ma[box_no](i, j, k, n) =
            vel_ma[box_no](i, j, k, VELX + n) / fac_ma[box_no](i, j, k, n);
        });
      const auto& force_ma = velForces[lev].arrays();
      AMREX_D_TERM(
        const auto& facx_ma = m_mesh_map->fac_fc(lev, 0).const_arrays();
        , const auto& facy_ma = m_mesh_map->fac_fc(lev, 1).const_arrays();
        , const auto& facz_ma = m_mesh_map->fac_fc(lev, 2).const_arrays();)
      amrex::ParallelFor(
        velForces[lev], amrex::IntVect(nGrow_force),
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
          auto f = force_ma[box_no];
          auto fac = fac_ma[box_no];
          auto uxi = uxi_ma[box_no];
          // d fac_d / dXi_d at the cell centre from the d-face values
          AMREX_D_TERM(
            const amrex::Real dfx =
              (facx_ma[box_no](i + 1, j, k, 0) - facx_ma[box_no](i, j, k, 0)) *
              dxinv[0];
            , const amrex::Real dfy = (facy_ma[box_no](i, j + 1, k, 1) -
                                       facy_ma[box_no](i, j, k, 1)) *
                                      dxinv[1];
            , const amrex::Real dfz = (facz_ma[box_no](i, j, k + 1, 2) -
                                       facz_ma[box_no](i, j, k, 2)) *
                                      dxinv[2];)
          AMREX_D_TERM(
            f(i, j, k, 0) =
              (f(i, j, k, 0) - uxi(i, j, k, 0) * uxi(i, j, k, 0) * dfx) /
              fac(i, j, k, 0);
            , f(i, j, k, 1) =
                (f(i, j, k, 1) - uxi(i, j, k, 1) * uxi(i, j, k, 1) * dfy) /
                fac(i, j, k, 1);
            , f(i, j, k, 2) =
                (f(i, j, k, 2) - uxi(i, j, k, 2) * uxi(i, j, k, 2) * dfz) /
                fac(i, j, k, 2););
        });
    }
    amrex::Gpu::streamSynchronize();
  }

  //----------------------------------------------------------------
  // Predict face velocities at t^{n+1/2} with Godunov
  auto bcRecVel = fetchBCRecArray(VELX, AMREX_SPACEDIM);
  auto bcRecVel_d = convertToDeviceVector(bcRecVel);
  for (int lev = 0; lev <= finest_level; ++lev) {

    auto* ldata_p = getLevelDataPtr(lev, AmrOldTime);

    //----------------------------------------------------------------
#ifdef AMREX_USE_EB
    const auto& ebfact = EBFactory(lev);
#endif

    HydroUtils::ExtrapVelToFaces(
      m_mesh_mapping ? vel_xi[lev] : ldata_p->state, velForces[lev],
      AMREX_D_DECL(
        advData->umac[lev][0], advData->umac[lev][1], advData->umac[lev][2]),
      bcRecVel, bcRecVel_d.dataPtr(), geom[lev], m_dt,
#ifdef AMREX_USE_EB
      ebfact,
      (m_useEBinflow != 0)
        ? getEBState(lev, VELX, AMREX_SPACEDIM, AmrOldTime).get()
        : nullptr,
#endif
      m_Godunov_ppm != 0, m_Godunov_ForceInTrans != 0, m_predict_advection_type,
      m_Godunov_ppm_limiter);

    // Back to physical face velocities: u_d = fac_d u~_d
    if (m_mesh_mapping) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        amrex::MultiFab::Multiply(
          advData->umac[lev][idim], m_mesh_map->fac_fc(lev, idim), idim, 0, 1,
          advData->umac[lev][idim].nGrowVect());
      }
    }
  }
}

void
PeleLM::createMACRHS(const std::unique_ptr<AdvanceAdvData>& advData)
{
  BL_PROFILE("PeleLMeX::createMACRHS()");

  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::Real halftime = 0.5 * (m_t_old[lev] + m_t_new[lev]);
    fillpatch_divu(lev, halftime, advData->mac_divu[lev], m_nGrowAdv);
  }
}

void
PeleLM::addChiIncrement(
  const int a_sdcIter,
  const TimeStamp a_time,
  const std::unique_ptr<AdvanceAdvData>& advData)
{
  BL_PROFILE("PeleLMeX::addChiIncrement()");

  const int nGrow = m_nGrowAdv;
  amrex::Vector<amrex::MultiFab> chiIncr;
  chiIncr.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    chiIncr.emplace_back(
      grids[lev], dmap[lev], 1, nGrow, amrex::MFInfo(), Factory(lev));
  }

  // Update the thermodynamic pressure
  setThermoPress(a_time);

  // Compute the pressure drift term
  calc_dPdt(a_time, GetVecOfPtrs(chiIncr));

  // Add chiIncr to chi and add chi to mac_divu
  // Both mac_divu and chiIncr have properly filled ghost cells -> work on
  // grownbox

  if (a_sdcIter == 1) {
    switch (m_chi_correction_type) {
    case ChiCorrectionType::DivuFirstIter: {
      for (int lev = 0; lev <= finest_level; ++lev) {
        auto const& chiInc_ma = chiIncr[lev].const_arrays();
        auto const& chi_ma = advData->chi[lev].arrays();
        auto const& mac_divu_ma = advData->mac_divu[lev].arrays();
        amrex::ParallelFor(
          advData->chi[lev], advData->chi[lev].nGrowVect(),
          [chi_ma, chiInc_ma, mac_divu_ma] AMREX_GPU_DEVICE(
            int box_no, int i, int j, int k) noexcept {
            chi_ma[box_no](i, j, k) =
              chiInc_ma[box_no](i, j, k) + mac_divu_ma[box_no](i, j, k);
            mac_divu_ma[box_no](i, j, k) = chi_ma[box_no](i, j, k);
          });
      }
      break;
    }
    case ChiCorrectionType::NoDivu: {
      for (int lev = 0; lev <= finest_level; ++lev) {
        auto const& chiInc_ma = chiIncr[lev].const_arrays();
        auto const& chi_ma = advData->chi[lev].arrays();
        auto const& mac_divu_ma = advData->mac_divu[lev].arrays();
        amrex::ParallelFor(
          advData->chi[lev], advData->chi[lev].nGrowVect(),
          [chi_ma, chiInc_ma, mac_divu_ma] AMREX_GPU_DEVICE(
            int box_no, int i, int j, int k) noexcept {
            chi_ma[box_no](i, j, k) = chiInc_ma[box_no](i, j, k);
            mac_divu_ma[box_no](i, j, k) = chi_ma[box_no](i, j, k);
          });
      }
      break;
    }
    default: {
      for (int lev = 0; lev <= finest_level; ++lev) {
        auto const& chiInc_ma = chiIncr[lev].const_arrays();
        auto const& chi_ma = advData->chi[lev].arrays();
        auto const& mac_divu_ma = advData->mac_divu[lev].arrays();
        amrex::ParallelFor(
          advData->chi[lev], advData->chi[lev].nGrowVect(),
          [chi_ma, chiInc_ma, mac_divu_ma] AMREX_GPU_DEVICE(
            int box_no, int i, int j, int k) noexcept {
            chi_ma[box_no](i, j, k) = chiInc_ma[box_no](i, j, k);
            mac_divu_ma[box_no](i, j, k) += chi_ma[box_no](i, j, k);
          });
      }
    }
    }
  } else {
    if (
      m_chi_correction_type == ChiCorrectionType::DivuFirstIter ||
      m_chi_correction_type == ChiCorrectionType::NoDivu) {
      for (int lev = 0; lev <= finest_level; ++lev) {
        auto const& chiInc_ma = chiIncr[lev].const_arrays();
        auto const& chi_ma = advData->chi[lev].arrays();
        auto const& mac_divu_ma = advData->mac_divu[lev].arrays();
        amrex::ParallelFor(
          advData->chi[lev], advData->chi[lev].nGrowVect(),
          [chi_ma, chiInc_ma, mac_divu_ma] AMREX_GPU_DEVICE(
            int box_no, int i, int j, int k) noexcept {
            chi_ma[box_no](i, j, k) += chiInc_ma[box_no](i, j, k);
            mac_divu_ma[box_no](i, j, k) = chi_ma[box_no](i, j, k);
          });
      }
    } else {
      for (int lev = 0; lev <= finest_level; ++lev) {
        auto const& chiInc_ma = chiIncr[lev].const_arrays();
        auto const& chi_ma = advData->chi[lev].arrays();
        auto const& mac_divu_ma = advData->mac_divu[lev].arrays();
        amrex::ParallelFor(
          advData->chi[lev], advData->chi[lev].nGrowVect(),
          [chi_ma, chiInc_ma, mac_divu_ma] AMREX_GPU_DEVICE(
            int box_no, int i, int j, int k) noexcept {
            chi_ma[box_no](i, j, k) += chiInc_ma[box_no](i, j, k);
            mac_divu_ma[box_no](i, j, k) += chi_ma[box_no](i, j, k);
          });
      }
    }
  }
  amrex::Gpu::streamSynchronize();

  if (m_print_chi_convergence) {
    const amrex::Real max_corr =
      MLNorm0(GetVecOfConstPtrs(chiIncr)) * m_dt / m_dpdtFactor;
    amrex::Print() << "      Before SDC " << a_sdcIter
                   << ": max relative P mismatch is " << max_corr << "\n";
  }
}

void
PeleLM::macProject(
  const TimeStamp a_time,
  const std::unique_ptr<AdvanceAdvData>& advData,
  const amrex::Vector<amrex::MultiFab*>& a_divu)
{
  BL_PROFILE("PeleLMeX::macProject()");

  const int has_divu = static_cast<int>(!a_divu.empty());

  // Get face rho inv
  auto bcRec = fetchBCRecArray(DENSITY, 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> rho_inv(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    if (m_incompressible != 0) {
      amrex::Real rhoInv = m_dt / (2.0 * m_rho);
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        rho_inv[lev][idim].define(
          amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
          dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
        rho_inv[lev][idim].setVal(rhoInv);
      }
    } else {
      auto* ldata_p = getLevelDataPtr(lev, a_time);
      constexpr int doZeroVisc = 0;
      rho_inv[lev] =
        getDiffusivity(lev, DENSITY, 1, doZeroVisc, {bcRec}, ldata_p->state);
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        rho_inv[lev][idim].invert(m_dt / 2.0, 0);
        rho_inv[lev][idim].FillBoundary(geom[lev].periodicity());
      }
    }
  }

  // For closed chamber, compute change in chamber pressure
  amrex::Real Sbar = 0.0;
  if ((m_closed_chamber != 0) && (m_incompressible == 0)) {
    Sbar = adjustPandDivU(advData);
  }

  //-------------------------------------------------------------------
  // Mesh-mapping scaling (MAC projection).  Solves
  //     div_Xi . ( beta grad phi ) = div_Xi . u_bar - J.S
  // with u_bar_i = (J/fac_i) u_i on face i and
  //      beta_i = (dt/2) . J / (fac_i^2 . rho) on face i.
  // Per the amr-wind convention, umac enters macProject in physical
  // space and leaves in uniform (Xi) space -- the scaling applied here
  // is NOT undone on exit.  Downstream consumers of umac must be aware
  // (see computeVelocityAdvTerm in PeleLMeX_Advection.cpp).
  //   amr-wind/docs/sphinx/theory/mapping.rst  (MAC projection section)
  //   amr-wind/equation_systems/icns/icns_advection.cpp ::
  //   mac_proj_to_uniform_space
  //
  // The caller-owned a_divu is NOT modified; a scratch MF holding
  // (J . divU) is built and passed to setDivU instead, so the caller's
  // divU remains in physical space across the call.
  //-------------------------------------------------------------------
  amrex::Vector<amrex::MultiFab> scaled_divu;
  amrex::Vector<const amrex::MultiFab*> divu_for_proj;
  if (m_mesh_mapping) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const auto& fac_ma = m_mesh_map->fac_fc(lev, idim).const_arrays();
        const auto& detJ_ma = m_mesh_map->detJ_fc(lev, idim).const_arrays();
        const auto& umac_ma = advData->umac[lev][idim].arrays();
        const auto& rhoinv_ma = rho_inv[lev][idim].arrays();
        const int nc = idim;
        amrex::ParallelFor(
          advData->umac[lev][idim],
          [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
            const amrex::Real f = fac_ma[box_no](i, j, k, nc);
            const amrex::Real dJ = detJ_ma[box_no](i, j, k);
            umac_ma[box_no](i, j, k) *= dJ / f;
            rhoinv_ma[box_no](i, j, k) *= dJ / (f * f);
          });
      }
    }
    amrex::Gpu::streamSynchronize();

    if (has_divu != 0) {
      scaled_divu.resize(finest_level + 1);
      divu_for_proj.resize(finest_level + 1);
      for (int lev = 0; lev <= finest_level; ++lev) {
        scaled_divu[lev].define(
          a_divu[lev]->boxArray(), a_divu[lev]->DistributionMap(), 1,
          a_divu[lev]->nGrow(), amrex::MFInfo(), a_divu[lev]->Factory());
        amrex::MultiFab::Copy(
          scaled_divu[lev], *a_divu[lev], 0, 0, 1, a_divu[lev]->nGrow());
        amrex::MultiFab::Multiply(
          scaled_divu[lev], m_mesh_map->detJ_cc(lev), 0, 0, 1,
          a_divu[lev]->nGrow());
        divu_for_proj[lev] = &scaled_divu[lev];
      }
    }
  }

  if (macproj->needInitialization()) {
    amrex::LPInfo lpInfo;
    lpInfo.setMaxCoarseningLevel(m_mac_mg_max_coarsening_level);
    macproj->initProjector(lpInfo, GetVecOfArrOfConstPtrs(rho_inv));
    macproj->setDomainBC(
      getMACProjectionBC(amrex::Orientation::low),
      getMACProjectionBC(amrex::Orientation::high));
#ifdef AMREX_USE_HYPRE
    if (
      macproj->getMLMG().getBottomSolver() ==
      amrex::MLMG::BottomSolver::hypre) {
      macproj->getMLMG().setHypreOptionsNamespace(m_hypre_namespace_mac);
    }
#endif
  } else {
    macproj->updateBeta(GetVecOfArrOfConstPtrs(rho_inv));
  }

  // set MAC velocity and projection RHS
  macproj->getLinOp().setMaxOrder(m_mac_max_order);
  macproj->setUMAC(GetVecOfArrOfPtrs(advData->umac));
  if (has_divu != 0) {
    if (m_mesh_mapping) {
      macproj->setDivU(divu_for_proj);
    } else {
      macproj->setDivU(GetVecOfConstPtrs(a_divu));
    }
  }

#ifdef AMREX_USE_EB
  if (m_useEBinflow != 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      macproj->setEBInflowVelocity(
        lev, *getEBState(lev, VELX, AMREX_SPACEDIM, a_time));
    }
  }
#endif

  // Project
  if (m_macproj_verbose > 0) {
    amrex::Print() << "MLMG: MAC Projection\n";
  }
  if (!m_mlmg_fail_plt_residuals) {
    macproj->project(m_mac_mg_rtol, m_mac_mg_atol);
  } else {
    macproj->getMLMG().setThrowException(true);
    macproj->getMLMG().setConvergenceNormType(amrex::MLMGNormType::bnorm);

    // Maximum MLMG iterations may change for debugging purposes
    if (
      m_mac_mg_fail_sdc_miniter >= 0 &&
      m_sdcIter >= m_mac_mg_fail_sdc_miniter) {
      if (m_mac_mg_fail_maxiter_after_sdc_miniter > 0) {
        macproj->getMLMG().setMaxIter(m_mac_mg_fail_maxiter_after_sdc_miniter);
        amrex::Print() << "      Limiting MAC MLMG max_iter to "
                       << m_mac_mg_fail_maxiter_after_sdc_miniter
                       << " (SDC iter [" << m_sdcIter
                       << "] >= " << m_mac_mg_fail_sdc_miniter << ")\n";
      }
    }

    try {
      macproj->project(m_mac_mg_rtol, m_mac_mg_atol);
    } catch (const std::exception& e) {
      amrex::Print() << "\n";
      amrex::Print() << "  *** MAC projection MLMG solve failed! ***\n";
      amrex::Print() << "  Error: " << e.what() << "\n";
      amrex::Print() << "  Dumping MAC projection residuals for debugging...\n";

      auto& mlmg = macproj->getMLMG();
      const auto& phi_ptrs = amrex::GetVecOfPtrs(
        const_cast<amrex::Vector<amrex::MultiFab>&>(macproj->getPhi()));
      const auto& rhs_ptrs = amrex::GetVecOfConstPtrs(macproj->getRHS());

      WriteMLMGResidual(mlmg, phi_ptrs, rhs_ptrs, "mac_projection", m_nstep);

      amrex::Abort("MLMG solve for mac_projection failed");
    }
  }

  // Restore mac_divu
  if ((m_closed_chamber != 0) && (m_incompressible == 0)) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      a_divu[lev]->plus(Sbar, 0, 1);
    }
  }

  // Intentionally no post-project umac / a_divu unscaling.  umac remains
  // in uniform (Xi) space -- downstream advection consumers must apply
  // their own mesh-mapping adjustments.

  // FillBoundary umac
  // Do coarse first
  AMREX_D_TERM(advData->umac[0][0].FillBoundary(geom[0].periodicity());
               , advData->umac[0][1].FillBoundary(geom[0].periodicity());
               , advData->umac[0][2].FillBoundary(geom[0].periodicity()));
  for (int lev = 1; lev <= finest_level; ++lev) {
    // We need to fill the MAC velocities outside the fine region so we can
    // use them in the Godunov method
    const amrex::IntVect rr =
      geom[lev].Domain().size() / geom[lev - 1].Domain().size();
    create_constrained_umac_grown(
      m_nGrowMAC, &geom[lev - 1], &geom[lev],
      GetArrOfPtrs(advData->umac[lev - 1]), GetArrOfPtrs(advData->umac[lev]),
      rr);
  }
}

void
PeleLM::create_constrained_umac_grown(
  const int a_nGrow,
  const amrex::Geometry* crse_geom,
  const amrex::Geometry* fine_geom,
  const amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>& u_mac_crse,
  const amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>& u_mac_fine,
  const amrex::IntVect& crse_ratio)
{
  // Divergence preserving interp
  amrex::Interpolater* mapper = &amrex::face_divfree_interp;

  // Set BCRec for Umac
  amrex::Vector<amrex::BCRec> bcrec(1);
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (crse_geom->isPeriodic(idim)) {
      bcrec[0].setLo(idim, amrex::BCType::int_dir);
      bcrec[0].setHi(idim, amrex::BCType::int_dir);
    } else {
      bcrec[0].setLo(idim, amrex::BCType::foextrap);
      bcrec[0].setHi(idim, amrex::BCType::foextrap);
    }
  }
  amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> bcrecArr = {
    AMREX_D_DECL(bcrec, bcrec, bcrec)};

  amrex::PhysBCFunct<amrex::GpuBndryFuncFab<umacFill>> crse_bndry_func(
    *crse_geom, bcrec, umacFill{});
  amrex::Array<
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<umacFill>>, AMREX_SPACEDIM>
    cbndyFuncArr = {
      AMREX_D_DECL(crse_bndry_func, crse_bndry_func, crse_bndry_func)};

  amrex::PhysBCFunct<amrex::GpuBndryFuncFab<umacFill>> fine_bndry_func(
    *fine_geom, bcrec, umacFill{});
  amrex::Array<
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<umacFill>>, AMREX_SPACEDIM>
    fbndyFuncArr = {
      AMREX_D_DECL(fine_bndry_func, fine_bndry_func, fine_bndry_func)};

  // Use piecewise constant interpolation in time, so create dummy variable for
  // time
  constexpr amrex::Real dummy = 0.;
  FillPatchTwoLevels(
    u_mac_fine, amrex::IntVect(a_nGrow), dummy, {u_mac_crse}, {dummy},
    {u_mac_fine}, {dummy}, 0, 0, 1, *crse_geom, *fine_geom, cbndyFuncArr, 0,
    fbndyFuncArr, 0, crse_ratio, mapper, bcrecArr, 0);
}

amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>
PeleLM::getMACProjectionBC(amrex::Orientation::Side a_side)
{

  amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> r;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (Geom(0).isPeriodic(idim)) {
      r[idim] = amrex::LinOpBCType::Periodic;
    } else {
      auto physbc = (a_side == amrex::Orientation::low) ? m_phys_bc.lo(idim)
                                                        : m_phys_bc.hi(idim);
      if (physbc == amrex::PhysBCType::outflow) {
        r[idim] = amrex::LinOpBCType::Dirichlet;
      } else {
        r[idim] = amrex::LinOpBCType::Neumann;
      }
    }
  }
  return r;
}
