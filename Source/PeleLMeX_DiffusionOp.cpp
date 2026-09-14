#include <PeleLMeX.H>
#include <PeleLMeX_DiffusionOp.H>
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>

#ifdef AMREX_USE_EB
#include <AMReX_EB_Redistribution.H>
#include <AMReX_EBFArrayBox.H>
#endif

//---------------------------------------------------------------------------------------
// Diffusion Operator

DiffusionOp::DiffusionOp(PeleLM* a_pelelm, const int ncomp)
  : m_pelelm(a_pelelm), m_ncomp(ncomp)
{
  BL_PROFILE("DiffusionOp::DiffusionOp()");
  readParameters();

  // Solve LPInfo
  amrex::LPInfo info_solve;
  info_solve.setAgglomeration(true);
  info_solve.setConsolidation(true);
  info_solve.setMaxCoarseningLevel(m_mg_max_coarsening_level);

  // Apply LPInfo (no coarsening)
  amrex::LPInfo info_apply;
  info_apply.setMaxCoarseningLevel(0);

#ifdef AMREX_USE_EB
  // Get vector of EB Factory
  amrex::Vector<amrex::EBFArrayBoxFactory const*> ebfactVec;
  for (int lev = 0; lev <= m_pelelm->finestLevel(); ++lev) {
    ebfactVec.push_back(&(m_pelelm->EBFactory(lev)));
  }
#endif

  // Scalar apply op.
#ifdef AMREX_USE_EB
  m_scal_apply_op = std::make_unique<amrex::MLEBABecLap>(
    m_pelelm->Geom(0, m_pelelm->finestLevel()),
    m_pelelm->boxArray(0, m_pelelm->finestLevel()),
    m_pelelm->DistributionMap(0, m_pelelm->finestLevel()), info_apply,
    ebfactVec, m_ncomp);
#else
  const amrex::Vector<amrex::FabFactory<amrex::FArrayBox> const*>&
    empty_factory = {};
  m_scal_apply_op = std::make_unique<amrex::MLABecLaplacian>(
    m_pelelm->Geom(0, m_pelelm->finestLevel()),
    m_pelelm->boxArray(0, m_pelelm->finestLevel()),
    m_pelelm->DistributionMap(0, m_pelelm->finestLevel()), info_apply,
    empty_factory, m_ncomp);
#endif
  m_scal_apply_op->setMaxOrder(m_mg_maxorder);

  // Scalar solve op.
#ifdef AMREX_USE_EB
  m_scal_solve_op = std::make_unique<amrex::MLEBABecLap>(
    m_pelelm->Geom(0, m_pelelm->finestLevel()),
    m_pelelm->boxArray(0, m_pelelm->finestLevel()),
    m_pelelm->DistributionMap(0, m_pelelm->finestLevel()), info_solve,
    ebfactVec, m_ncomp);
#else
  m_scal_solve_op = std::make_unique<amrex::MLABecLaplacian>(
    m_pelelm->Geom(0, m_pelelm->finestLevel()),
    m_pelelm->boxArray(0, m_pelelm->finestLevel()),
    m_pelelm->DistributionMap(0, m_pelelm->finestLevel()), info_solve,
    empty_factory, m_ncomp);
#endif
  m_scal_solve_op->setMaxOrder(m_mg_maxorder);

  // Gradient op. : scalar/coefficient already preset
#ifdef AMREX_USE_EB
  m_gradient_op = std::make_unique<amrex::MLEBABecLap>(
    m_pelelm->Geom(0, m_pelelm->finestLevel()),
    m_pelelm->boxArray(0, m_pelelm->finestLevel()),
    m_pelelm->DistributionMap(0, m_pelelm->finestLevel()), info_apply,
    ebfactVec, m_ncomp);
#else
  m_gradient_op = std::make_unique<amrex::MLABecLaplacian>(
    m_pelelm->Geom(0, m_pelelm->finestLevel()),
    m_pelelm->boxArray(0, m_pelelm->finestLevel()),
    m_pelelm->DistributionMap(0, m_pelelm->finestLevel()), info_apply,
    empty_factory, m_ncomp);
#endif
  m_gradient_op->setMaxOrder(m_mg_maxorder);
  m_gradient_op->setScalars(0.0, 1.0);
  for (int lev = 0; lev <= m_pelelm->finestLevel(); ++lev) {
    m_gradient_op->setBCoeffs(lev, -1.0);
  }
}

void
DiffusionOp::diffuse_scalar(
  amrex::Vector<amrex::MultiFab*> const& a_phi,
  const int phi_comp,
  amrex::Vector<amrex::MultiFab const*> const& a_rhs,
  const int rhs_comp,
  amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> const& a_flux,
  const int flux_comp,
  amrex::Vector<amrex::MultiFab const*> const& a_acoeff,
  amrex::Vector<amrex::MultiFab const*> const& a_density,
  amrex::Vector<amrex::MultiFab const*> const& a_bcoeff,
  const int bcoeff_comp,
  amrex::Vector<amrex::BCRec> a_bcrec,
  const int ncomp,
  const int isPoissonSolve,
  const amrex::Real a_dt,
  amrex::Vector<amrex::MultiFab const*> const& a_boundary)
{
  BL_PROFILE("DiffusionOp::diffuse_scalar()");

  //----------------------------------------------------------------
  // What are we dealing with ?
  const int have_density = (a_density.empty()) ? 0 : 1;
  const int have_fluxes = (a_flux.empty()) ? 0 : 1;
  const int have_acoeff = (a_acoeff.empty()) ? 0 : 1;
  const int have_bcoeff = (a_bcoeff.empty()) ? 0 : 1;
  const int have_boundary = (a_boundary.empty()) ? 0 : 1;

  //----------------------------------------------------------------
  // Checks
  AMREX_ASSERT(m_ncomp == 1 || m_ncomp == ncomp);
  AMREX_ASSERT(a_phi[0]->nComp() >= phi_comp + ncomp);
  AMREX_ASSERT(a_rhs[0]->nComp() >= rhs_comp + ncomp);
  if (have_fluxes != 0) {
    AMREX_ASSERT(a_flux[0][0]->nComp() >= flux_comp + ncomp);
  }
  if (have_bcoeff != 0) {
    AMREX_ASSERT(a_bcoeff[0]->nComp() >= bcoeff_comp + ncomp);
    AMREX_ASSERT(a_bcrec.size() >= ncomp);
  }

  const int finest_level = m_pelelm->finestLevel();

  //----------------------------------------------------------------
  // Duplicate phi_old to include rho scaling
  // include 1 ghost cell to provide levelBC
  // NOTE: this is a bit weird for species: since we already updated the density
  // after adv., when we divide by \rho, it is inconsistent. But it only matters
  // if it screws up the ghost cell values 'cause interiors are just an initial
  // solution for the solve.
  amrex::Vector<amrex::MultiFab> phi;
  phi.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    phi.emplace_back(
      a_phi[lev]->boxArray(), a_phi[lev]->DistributionMap(), ncomp, 1,
      amrex::MFInfo(), a_phi[lev]->Factory());
  }
  if (have_density == 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab::Copy(
        phi[lev], *a_phi[lev], phi_comp, 0, ncomp, phi[lev].nGrowVect());
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto const& a_phi_ma = a_phi[lev]->const_arrays();
      auto const& a_rho_ma = a_density[lev]->const_arrays();
      auto const& phi_ma = phi[lev].arrays();
      amrex::ParallelFor(
        phi[lev], phi[lev].nGrowVect(), ncomp,
        [a_phi_ma, a_rho_ma, phi_ma, phi_comp] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          amrex::Array4<amrex::Real const> a_phi_arr(
            a_phi_ma[box_no], phi_comp);
          phi_ma[box_no](i, j, k, n) =
            a_phi_arr(i, j, k, n) / a_rho_ma[box_no](i, j, k);
        });
    }
    amrex::Gpu::streamSynchronize();
  }
  //----------------------------------------------------------------
  // Setup solve LinearOp coefficients
  // LinOp is \alpha A \phi - \beta \nabla \cdot B \nabla \phi = rhs
  // => \alpha = 1.0, A is a_acoeff if provided, 1.0 otherwise
  // => \beta = a_dt, B face centered diffusivity bcoeff^{np1,k}
  //
  // Under mesh mapping: the equivalent operator in Xi-space is
  //     alpha A . J . phi - beta div_Xi . (B . J/fac^2 grad_Xi phi) = J . rhs
  // so we scale A by detJ_cc, B on face i by J/fac_i^2, and rhs by detJ_cc.

  const bool mesh_mapping = m_pelelm->hasMeshMapping();
  auto* mm = mesh_mapping ? m_pelelm->meshMap() : nullptr;

  // Build scaled A-coeff scratch (only when mesh_mapping).  Carries
  // either (a_acoeff . detJ) or (detJ) depending on whether the caller
  // provided an A-coefficient MF.
  amrex::Vector<amrex::MultiFab> acoeff_scaled;
  if (mesh_mapping) {
    acoeff_scaled.resize(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      acoeff_scaled[lev].define(
        a_phi[lev]->boxArray(), a_phi[lev]->DistributionMap(), 1, 0,
        amrex::MFInfo(), a_phi[lev]->Factory());
      if (have_acoeff != 0) {
        amrex::MultiFab::Copy(acoeff_scaled[lev], *a_acoeff[lev], 0, 0, 1, 0);
      } else {
        acoeff_scaled[lev].setVal(1.0);
      }
      amrex::MultiFab::Multiply(
        acoeff_scaled[lev], mm->detJ_cc(lev), 0, 0, 1, 0);
    }
  }

  const amrex::Real alpha = (isPoissonSolve != 0) ? 0.0 : 1.0;
  const amrex::Real beta = a_dt;
  m_scal_solve_op->setScalars(alpha, beta);
  for (int lev = 0; lev <= finest_level; ++lev) {
    if (mesh_mapping) {
      m_scal_solve_op->setACoeffs(lev, acoeff_scaled[lev]);
    } else if (have_acoeff != 0) {
      m_scal_solve_op->setACoeffs(lev, *a_acoeff[lev]);
    } else {
      m_scal_solve_op->setACoeffs(lev, 1.0);
    }
  }

  //----------------------------------------------------------------
  // Solve and get fluxes on a m_ncomp component basis
  for (int comp = 0; comp < ncomp; comp += m_ncomp) {

    // Aliases
    amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> fluxes(
      finest_level + 1);
    amrex::Vector<amrex::MultiFab> component;
    amrex::Vector<amrex::MultiFab> rhs;
    amrex::Vector<amrex::MultiFab> boundary;

    // Allow for component specific LinOp BC
    m_scal_solve_op->setDomainBC(
      m_pelelm->getDiffusionLinOpBC(amrex::Orientation::low, a_bcrec[comp]),
      m_pelelm->getDiffusionLinOpBC(amrex::Orientation::high, a_bcrec[comp]));

    // Set aliases and bcoeff comp
    for (int lev = 0; lev <= finest_level; ++lev) {
      if (have_fluxes != 0) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          fluxes[lev][idim] = new amrex::MultiFab(
            *a_flux[lev][idim], amrex::make_alias, flux_comp + comp, m_ncomp);
        }
      }

      if (have_bcoeff != 0) {
        constexpr int doZeroVisc = 1;
        constexpr int addTurbContrib = 1;
        amrex::Vector<amrex::BCRec> subBCRec = {
          a_bcrec.begin() + comp, a_bcrec.begin() + comp + m_ncomp};
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> bcoeff_ec =
          m_pelelm->getDiffusivity(
            lev, bcoeff_comp + comp, m_ncomp, doZeroVisc, subBCRec,
            *a_bcoeff[lev], addTurbContrib);
        if (mesh_mapping) {
          // bcoeff_i -> (J/fac_i^2) . bcoeff on face i
          for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const auto& fac_ma = mm->fac_fc(lev, idim).const_arrays();
            const auto& detJ_ma = mm->detJ_fc(lev, idim).const_arrays();
            const auto& b_ma = bcoeff_ec[idim].arrays();
            const int nc = idim;
            amrex::ParallelFor(
              bcoeff_ec[idim], amrex::IntVect(0), m_ncomp,
              [=] AMREX_GPU_DEVICE(
                int box_no, int i, int j, int k, int n) noexcept {
                const amrex::Real f = fac_ma[box_no](i, j, k, nc);
                const amrex::Real dJ = detJ_ma[box_no](i, j, k);
                b_ma[box_no](i, j, k, n) *= dJ / (f * f);
              });
          }
          amrex::Gpu::streamSynchronize();
        }
#ifdef AMREX_USE_EB
        m_scal_solve_op->setBCoeffs(
          lev, GetArrOfConstPtrs(bcoeff_ec),
          amrex::MLMG::Location::FaceCentroid);
#else
        m_scal_solve_op->setBCoeffs(lev, GetArrOfConstPtrs(bcoeff_ec));
#endif
      } else {
        m_scal_solve_op->setBCoeffs(lev, 1.0);
      }

      component.emplace_back(phi[lev], amrex::make_alias, comp, m_ncomp);
      if (mesh_mapping) {
        // Scale rhs by detJ: rhs = a_rhs . J.  Use scratch to leave
        // the caller-owned a_rhs bit-exactly untouched.
        rhs.emplace_back(
          a_rhs[lev]->boxArray(), a_rhs[lev]->DistributionMap(), m_ncomp, 0,
          amrex::MFInfo(), a_rhs[lev]->Factory());
        amrex::MultiFab::Copy(
          rhs.back(), *a_rhs[lev], rhs_comp + comp, 0, m_ncomp, 0);
        for (int n = 0; n < m_ncomp; ++n) {
          amrex::MultiFab::Multiply(rhs.back(), mm->detJ_cc(lev), 0, n, 1, 0);
        }
      } else {
        rhs.emplace_back(
          *a_rhs[lev], amrex::make_alias, rhs_comp + comp, m_ncomp);
      }
      if (have_boundary != 0) {
        boundary.emplace_back(
          *a_boundary[lev], amrex::make_alias, comp, m_ncomp);
      } else {
        boundary.emplace_back(phi[lev], amrex::make_alias, comp, m_ncomp);
      }
      m_scal_solve_op->setLevelBC(lev, &boundary[lev]);
    }

    // Setup linear solver
    amrex::MLMG mlmg(*m_scal_solve_op);

    std::string mg_variable =
      (m_ncomp == NUM_SPECIES) ? "Species" : "Temperature";
    if (m_mg_verbose > 0) {
      if (m_ncomp == NUM_SPECIES) {
        amrex::Print() << "MLMG: " << mg_variable << " Diffusion\n";
      } else {
        amrex::Print() << "MLMG: DeltaT solve [" << m_pelelm->m_deltaTIter
                       << "]\n";
      }
    }

    // Maximum iterations for MultiGrid / ConjugateGradients may change for
    // debugging purposes
    int max_iter = m_mg_max_iter;
    if (m_pelelm->m_mlmg_fail_plt_residuals) {

      bool sdc_iters_met = (m_pelelm->m_sdcIter >= m_mg_fail_sdc_miniter);
      bool limit_max_iter = false;

      if (m_ncomp == NUM_SPECIES) {
        // Species diffusion: only SDC iter matters
        if (
          sdc_iters_met && (m_mg_fail_species_maxiter_after_sdc_miniter > 0)) {
          limit_max_iter = true;
          max_iter = m_mg_fail_species_maxiter_after_sdc_miniter;
        }
      } else {
#ifndef USE_MANIFOLD_EOS
        // Temperature diffusion: check both SDC and deltaT iters
        bool dT_iters_met =
          (m_pelelm->m_deltaTIter >= m_mg_fail_deltaT_miniter);
        limit_max_iter = sdc_iters_met && dT_iters_met &&
                         (m_mg_fail_temp_maxiter_after_sdc_deltaT_miniter > 0);
        if (limit_max_iter) {
          max_iter = m_mg_fail_temp_maxiter_after_sdc_deltaT_miniter;
        }
#endif
      }

      // Print diagnostic message if limit was applied
      if (limit_max_iter) {
        if (m_ncomp == NUM_SPECIES) {
          amrex::Print() << "      Limiting species diffusion MLMG max_iter to "
                         << max_iter << " (SDC iter [" << m_pelelm->m_sdcIter
                         << "] >= " << m_mg_fail_sdc_miniter << ")\n";
        } else {
          amrex::Print()
            << "      Limiting temperature diffusion MLMG max_iter to "
            << max_iter << " (SDC iter [" << m_pelelm->m_sdcIter
            << "] >= " << m_mg_fail_sdc_miniter << ", deltaT solve ["
            << m_pelelm->m_deltaTIter << "] >= " << m_mg_fail_deltaT_miniter
            << ")\n";
        }
      }
    }

    mlmg.setMaxIter(max_iter);
    mlmg.setMaxFmgIter(m_mg_max_fmg_iter);
    mlmg.setBottomMaxIter(m_mg_bottom_maxiter);

    // Verbosity
    mlmg.setVerbose(m_mg_verbose);
    mlmg.setBottomVerbose(m_mg_bottom_verbose);

    mlmg.setPreSmooth(m_num_pre_smooth);
    mlmg.setPostSmooth(m_num_post_smooth);

    // Solve
    if (!m_pelelm->m_mlmg_fail_plt_residuals) {
      mlmg.solve(
        GetVecOfPtrs(component), GetVecOfConstPtrs(rhs), m_mg_rtol, m_mg_atol);
    } else {
      mlmg.setThrowException(true);
      mlmg.setConvergenceNormType(amrex::MLMGNormType::bnorm);

      try {
        mlmg.solve(
          GetVecOfPtrs(component), GetVecOfConstPtrs(rhs), m_mg_rtol,
          m_mg_atol);
      } catch (const std::exception& e) {
        amrex::Print() << "\n"
                       << "  *** " << mg_variable
                       << " diffusion MLMG solve failed (non-EB)! ***\n"
                       << "  Error: " << e.what() << "\n"
                       << "  Dumping residuals for debugging...\n";
        std::string m_solver_type = (m_ncomp == NUM_SPECIES)
                                      ? "species_diffusion"
                                      : "temperature_diffusion";
        m_pelelm->WriteMLMGResidual(
          mlmg, GetVecOfPtrs(component), GetVecOfConstPtrs(rhs), m_solver_type,
          m_pelelm->m_nstep);

        amrex::Abort("MLMG solve for scalar diffusion failed");
      }
    }

    // Need to get the fluxes
    if (have_fluxes != 0) {
#ifdef AMREX_USE_EB
      mlmg.getFluxes(fluxes, amrex::MLMG::Location::FaceCentroid);
#else
      mlmg.getFluxes(fluxes, amrex::MLMG::Location::FaceCenter);
#endif

      for (int lev = 0; lev <= finest_level; ++lev) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          delete fluxes[lev][idim];
        }
      }
    }
  }

  //----------------------------------------------------------------
  // Copy the results of the solve back into a_phi
  // Times rho{np1,kp1} if needed
  // Don't touch the ghost cells
  if (have_density == 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab::Copy(*a_phi[lev], phi[lev], 0, 0, ncomp, 0);
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto const& a_phi_ma = a_phi[lev]->arrays();
      auto const& phi_ma = phi[lev].const_arrays();
      auto const& a_rho_ma = a_density[lev]->const_arrays();
      amrex::ParallelFor(
        phi[lev], amrex::IntVect(0), ncomp,
        [a_phi_ma, a_rho_ma, phi_ma] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          a_phi_ma[box_no](i, j, k, n) =
            phi_ma[box_no](i, j, k, n) * a_rho_ma[box_no](i, j, k);
        });
    }
    amrex::Gpu::streamSynchronize();
  }
}

#ifdef AMREX_USE_EB
void
DiffusionOp::diffuse_scalar(
  amrex::Vector<amrex::MultiFab*> const& a_phi,
  const int phi_comp,
  amrex::Vector<amrex::MultiFab const*> const& a_phiEB,
  const int /*phiEB_comp*/,
  amrex::Vector<amrex::MultiFab const*> const& a_rhs,
  const int rhs_comp,
  amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> const& a_flux,
  const int flux_comp,
  amrex::Vector<amrex::MultiFab const*> const& a_acoeff,
  amrex::Vector<amrex::MultiFab const*> const& a_density,
  amrex::Vector<amrex::MultiFab const*> const& a_bcoeff,
  const int bcoeff_comp,
  amrex::Vector<amrex::MultiFab const*> const& a_bcoeffEB,
  const int /*bcoeffEB_comp*/,
  amrex::Vector<amrex::BCRec> a_bcrec,
  const int ncomp,
  const int isPoissonSolve,
  const amrex::Real a_dt,
  amrex::Vector<amrex::MultiFab const*> const& a_boundary)
{
  BL_PROFILE("DiffusionOp::diffuse_scalar()");

  //----------------------------------------------------------------
  // What are we dealing with ?
  const int have_density = (a_density.empty()) ? 0 : 1;
  const int have_fluxes = (a_flux.empty()) ? 0 : 1;
  const int have_acoeff = (a_acoeff.empty()) ? 0 : 1;
  const int have_bcoeff = (a_bcoeff.empty()) ? 0 : 1;
  const int have_boundary = (a_boundary.empty()) ? 0 : 1;

  //----------------------------------------------------------------
  // Checks
  AMREX_ASSERT(m_ncomp == 1 || m_ncomp == ncomp);
  AMREX_ASSERT(a_phi[0]->nComp() >= phi_comp + ncomp);
  AMREX_ASSERT(a_rhs[0]->nComp() >= rhs_comp + ncomp);
  if (have_fluxes != 0) {
    AMREX_ASSERT(a_flux[0][0]->nComp() >= flux_comp + ncomp);
  }
  if (have_bcoeff != 0) {
    AMREX_ASSERT(a_bcoeff[0]->nComp() >= bcoeff_comp + ncomp);
    AMREX_ASSERT(a_bcrec.size() >= ncomp);
  }

  const int finest_level = m_pelelm->finestLevel();

  //----------------------------------------------------------------
  // Duplicate phi_old to include rho scaling
  // include 1 ghost cell to provide levelBC
  // NOTE: this is a bit weird for species: since we already updated the density
  // after adv., when we divide by \rho, it is inconsistent. But it only matters
  // if it screws up the ghost cell values 'cause interiors are just an initial
  // solution for the solve.
  amrex::Vector<amrex::MultiFab> phi;
  phi.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    phi.emplace_back(
      a_phi[lev]->boxArray(), a_phi[lev]->DistributionMap(), ncomp, 1,
      amrex::MFInfo(), a_phi[lev]->Factory());
  }
  if (have_density == 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab::Copy(
        phi[lev], *a_phi[lev], phi_comp, 0, ncomp, phi[lev].nGrowVect());
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto const& a_phi_ma = a_phi[lev]->const_arrays();
      auto const& a_rho_ma = a_density[lev]->const_arrays();
      auto const& phi_ma = phi[lev].arrays();
      amrex::ParallelFor(
        phi[lev], phi[lev].nGrowVect(), ncomp,
        [a_phi_ma, a_rho_ma, phi_ma, phi_comp] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          amrex::Array4<amrex::Real const> a_phi_arr(
            a_phi_ma[box_no], phi_comp);
          phi_ma[box_no](i, j, k, n) =
            a_phi_arr(i, j, k, n) / a_rho_ma[box_no](i, j, k);
        });
    }
    amrex::Gpu::streamSynchronize();
  }
  //----------------------------------------------------------------
  // Setup solve LinearOp coefficients
  // LinOp is \alpha A \phi - \beta \nabla \cdot B \nabla \phi = rhs
  // => \alpha = 1.0, A is a_acoeff if provided, 1.0 otherwise
  // => \beta = a_dt, B face centered diffusivity bcoeff^{np1,k}

  const amrex::Real alpha = (isPoissonSolve != 0) ? 0.0 : 1.0;
  const amrex::Real beta = a_dt;
  m_scal_solve_op->setScalars(alpha, beta);
  if (have_acoeff != 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      m_scal_solve_op->setACoeffs(lev, *a_acoeff[lev]);
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      m_scal_solve_op->setACoeffs(lev, 1.0);
    }
  }

  //----------------------------------------------------------------
  // Solve and get fluxes on a m_ncomp component basis
  for (int comp = 0; comp < ncomp; comp += m_ncomp) {

    // Aliases
    amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> fluxes(
      finest_level + 1);
    amrex::Vector<amrex::MultiFab> component;
    amrex::Vector<amrex::MultiFab> rhs;
    amrex::Vector<amrex::MultiFab> boundary;

    // Allow for component specific LinOp BC
    m_scal_solve_op->setDomainBC(
      m_pelelm->getDiffusionLinOpBC(amrex::Orientation::low, a_bcrec[comp]),
      m_pelelm->getDiffusionLinOpBC(amrex::Orientation::high, a_bcrec[comp]));

    // Set aliases and bcoeff comp
    for (int lev = 0; lev <= finest_level; ++lev) {
      if (have_fluxes != 0) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          fluxes[lev][idim] = new amrex::MultiFab(
            *a_flux[lev][idim], amrex::make_alias, flux_comp + comp, m_ncomp);
        }
      }

      if (have_bcoeff != 0) {
        constexpr int doZeroVisc = 1;
        amrex::Vector<amrex::BCRec> subBCRec = {
          a_bcrec.begin() + comp, a_bcrec.begin() + comp + m_ncomp};
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> bcoeff_ec =
          m_pelelm->getDiffusivity(
            lev, bcoeff_comp + comp, m_ncomp, doZeroVisc, subBCRec,
            *a_bcoeff[lev]);
        m_scal_solve_op->setBCoeffs(
          lev, GetArrOfConstPtrs(bcoeff_ec),
          amrex::MLMG::Location::FaceCentroid);
      } else {
        m_scal_solve_op->setBCoeffs(lev, 1.0);
      }

      component.emplace_back(phi[lev], amrex::make_alias, comp, m_ncomp);
      rhs.emplace_back(
        *a_rhs[lev], amrex::make_alias, rhs_comp + comp, m_ncomp);
      if (have_boundary != 0) {
        boundary.emplace_back(
          *a_boundary[lev], amrex::make_alias, comp, m_ncomp);
      } else {
        boundary.emplace_back(phi[lev], amrex::make_alias, comp, m_ncomp);
      }
      m_scal_solve_op->setLevelBC(lev, &boundary[lev]);
      m_scal_solve_op->setEBDirichlet(lev, *a_phiEB[lev], *a_bcoeffEB[lev]);
    }

    // Setup linear solver
    amrex::MLMG mlmg(*m_scal_solve_op);

    std::string mg_variable =
      (m_ncomp == NUM_SPECIES) ? "Species" : "Temperature";
    if (m_mg_verbose > 0) {
      if (m_ncomp == NUM_SPECIES) {
        amrex::Print() << "MLMG: " << mg_variable << " Diffusion\n";
      } else {
        amrex::Print() << "MLMG: DeltaT solve [" << m_pelelm->m_deltaTIter
                       << "]\n";
      }
    }

    // Maximum iterations for MultiGrid / ConjugateGradients may change for
    // debugging purposes
    int max_iter = m_mg_max_iter;
    if (m_pelelm->m_mlmg_fail_plt_residuals) {

      bool sdc_iters_met = (m_pelelm->m_sdcIter >= m_mg_fail_sdc_miniter);
      bool limit_max_iter = false;

      if (m_ncomp == NUM_SPECIES) {
        // Species diffusion: only SDC iter matters
        if (
          sdc_iters_met && (m_mg_fail_species_maxiter_after_sdc_miniter > 0)) {
          limit_max_iter = true;
          max_iter = m_mg_fail_species_maxiter_after_sdc_miniter;
        }
      } else {
#ifndef USE_MANIFOLD_EOS
        // Temperature diffusion: check both SDC and deltaT iters
        bool dT_iters_met =
          (m_pelelm->m_deltaTIter >= m_mg_fail_deltaT_miniter);
        limit_max_iter = sdc_iters_met && dT_iters_met &&
                         (m_mg_fail_temp_maxiter_after_sdc_deltaT_miniter > 0);
        if (limit_max_iter) {
          max_iter = m_mg_fail_temp_maxiter_after_sdc_deltaT_miniter;
        }
#endif
      }

      // Print diagnostic message if limit was applied
      if (limit_max_iter) {
        if (m_ncomp == NUM_SPECIES) {
          amrex::Print() << "      Limiting species diffusion MLMG max_iter to "
                         << max_iter << " (SDC iter [" << m_pelelm->m_sdcIter
                         << "] >= " << m_mg_fail_sdc_miniter << ")\n";
        } else {
          amrex::Print()
            << "      Limiting temperature diffusion MLMG max_iter to "
            << max_iter << " (SDC iter [" << m_pelelm->m_sdcIter
            << "] >= " << m_mg_fail_sdc_miniter << ", deltaT solve ["
            << m_pelelm->m_deltaTIter << "] >= " << m_mg_fail_deltaT_miniter
            << ")\n";
        }
      }
    }

    mlmg.setMaxIter(max_iter);
    mlmg.setMaxFmgIter(m_mg_max_fmg_iter);
    mlmg.setBottomMaxIter(m_mg_bottom_maxiter);

    // Verbosity
    mlmg.setVerbose(m_mg_verbose);
    mlmg.setBottomVerbose(m_mg_bottom_verbose);

    mlmg.setPreSmooth(m_num_pre_smooth);
    mlmg.setPostSmooth(m_num_post_smooth);

    // Solve
    if (!m_pelelm->m_mlmg_fail_plt_residuals) {
      mlmg.solve(
        GetVecOfPtrs(component), GetVecOfConstPtrs(rhs), m_mg_rtol, m_mg_atol);
    } else {
      mlmg.setThrowException(true);
      mlmg.setConvergenceNormType(amrex::MLMGNormType::bnorm);

      try {
        mlmg.solve(
          GetVecOfPtrs(component), GetVecOfConstPtrs(rhs), m_mg_rtol,
          m_mg_atol);
      } catch (const std::exception& e) {
        amrex::Print() << "\n"
                       << "  *** " << mg_variable
                       << " diffusion MLMG solve failed (EB)! ***\n"
                       << "  Error: " << e.what() << "\n"
                       << "  Dumping residuals for debugging...\n";
        std::string m_solver_type = (m_ncomp == NUM_SPECIES)
                                      ? "species_diffusion"
                                      : "temperature_diffusion";
        m_pelelm->WriteMLMGResidual(
          mlmg, GetVecOfPtrs(component), GetVecOfConstPtrs(rhs), m_solver_type,
          m_pelelm->m_nstep);

        amrex::Abort("MLMG solve for scalar diffusion failed");
      }
    }

    // Need to get the fluxes
    if (have_fluxes != 0) {
      mlmg.getFluxes(fluxes, amrex::MLMG::Location::FaceCentroid);

      for (int lev = 0; lev <= finest_level; ++lev) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          delete fluxes[lev][idim];
        }
      }
    }
  }

  //----------------------------------------------------------------
  // Copy the results of the solve back into a_phi
  // Times rho{np1,kp1} if needed
  // Don't touch the ghost cells
  if (have_density == 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab::Copy(*a_phi[lev], phi[lev], 0, 0, ncomp, 0);
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto const& a_phi_ma = a_phi[lev]->arrays();
      auto const& phi_ma = phi[lev].const_arrays();
      auto const& a_rho_ma = a_density[lev]->const_arrays();
      amrex::ParallelFor(
        phi[lev], amrex::IntVect(0), ncomp,
        [a_phi_ma, a_rho_ma, phi_ma] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          a_phi_ma[box_no](i, j, k, n) =
            phi_ma[box_no](i, j, k, n) * a_rho_ma[box_no](i, j, k);
        });
    }
    amrex::Gpu::streamSynchronize();
  }
}
#endif

void
DiffusionOp::computeDiffLap(
  amrex::Vector<amrex::MultiFab*> const& a_laps,
  const int lap_comp,
  amrex::Vector<amrex::MultiFab const*> const& a_phi,
  const int phi_comp,
  amrex::Vector<amrex::MultiFab const*> const& a_bcoeff,
  const int bcoeff_comp,
  amrex::Vector<amrex::BCRec> a_bcrec,
  const int ncomp)
{
  BL_PROFILE("DiffusionOp::computeDiffLap()");

  //----------------------------------------------------------------
  // Checks
  AMREX_ASSERT(m_ncomp == 1 || m_ncomp == ncomp);
  AMREX_ASSERT(a_laps[0]->nComp() >= lap_comp + ncomp);
  AMREX_ASSERT(a_phi[0]->nComp() >= phi_comp + ncomp);
  AMREX_ASSERT(a_bcoeff[0]->nComp() >= bcoeff_comp + ncomp);
  AMREX_ASSERT(a_bcrec.size() >= ncomp);

  const int finest_level = m_pelelm->finestLevel();
  const bool mesh_mapping = m_pelelm->hasMeshMapping();
  auto* mm = mesh_mapping ? m_pelelm->meshMap() : nullptr;

  // Copy phi with 1 ghost cell
  amrex::Vector<amrex::MultiFab> phi;
  phi.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    phi.emplace_back(
      a_phi[lev]->boxArray(), a_phi[lev]->DistributionMap(), ncomp, 1,
      amrex::MFInfo(), a_phi[lev]->Factory());
    amrex::MultiFab::Copy(phi[lev], *a_phi[lev], phi_comp, 0, ncomp, 1);
  }

  // LinOp is \alpha A \phi - \beta \nabla \cdot B \nabla \phi
  // => \alpha = 0, A doesn't matter
  // => \beta = -1.0, B face centered diffusivity a_bcoeff

  // Set scalars \alpha & \beta
  amrex::Real alpha = 0.0;
  amrex::Real beta = -1.0;
  m_scal_apply_op->setScalars(alpha, beta);

  for (int comp = 0; comp < ncomp; comp += m_ncomp) {

    // Component based vector of data
    amrex::Vector<amrex::MultiFab> laps;
    amrex::Vector<amrex::MultiFab> component;

    for (int lev = 0; lev <= finest_level; ++lev) {
      laps.emplace_back(
        *a_laps[lev], amrex::make_alias, lap_comp + comp, m_ncomp);
      component.emplace_back(phi[lev], amrex::make_alias, comp, m_ncomp);
      constexpr int doZeroVisc = 0;
      constexpr int addTurbContrib = 0;
      amrex::Vector<amrex::BCRec> subBCRec = {
        a_bcrec.begin() + comp, a_bcrec.begin() + comp + m_ncomp};
      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> bcoeff_ec =
        m_pelelm->getDiffusivity(
          lev, bcoeff_comp + comp, m_ncomp, doZeroVisc, subBCRec,
          *a_bcoeff[lev], addTurbContrib);
      if (mesh_mapping) {
        // bcoeff_i -> (J/fac_i^2) . bcoeff on face i
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          const auto& fac_ma = mm->fac_fc(lev, idim).const_arrays();
          const auto& detJ_ma = mm->detJ_fc(lev, idim).const_arrays();
          const auto& b_ma = bcoeff_ec[idim].arrays();
          const int nc = idim;
          amrex::ParallelFor(
            bcoeff_ec[idim], amrex::IntVect(0), m_ncomp,
            [=] AMREX_GPU_DEVICE(
              int box_no, int i, int j, int k, int n) noexcept {
              const amrex::Real f = fac_ma[box_no](i, j, k, nc);
              const amrex::Real dJ = detJ_ma[box_no](i, j, k);
              b_ma[box_no](i, j, k, n) *= dJ / (f * f);
            });
        }
        amrex::Gpu::streamSynchronize();
      }

#ifdef AMREX_USE_EB
      m_scal_apply_op->setBCoeffs(
        lev, GetArrOfConstPtrs(bcoeff_ec), amrex::MLMG::Location::FaceCentroid);
#else
      m_scal_apply_op->setBCoeffs(lev, GetArrOfConstPtrs(bcoeff_ec));
#endif
      m_scal_apply_op->setLevelBC(lev, &component[lev]);
    }

    amrex::MLMG mlmg(*m_scal_apply_op);
    mlmg.apply(GetVecOfPtrs(laps), GetVecOfPtrs(component));

    // Under mesh mapping the applied operator produces J . (physical laps).
    // Divide by J so the returned laps is in physical-space units.
    if (mesh_mapping) {
      for (int lev = 0; lev <= finest_level; ++lev) {
        for (int n = 0; n < m_ncomp; ++n) {
          amrex::MultiFab::Divide(laps[lev], mm->detJ_cc(lev), 0, n, 1, 0);
        }
      }
    }
  }
}

void
DiffusionOp::computeDiffFluxes(
  amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> const& a_flux,
  const int flux_comp,
  amrex::Vector<amrex::MultiFab const*> const& a_phi,
  const int phi_comp,
  amrex::Vector<amrex::MultiFab const*> const& a_density,
  amrex::Vector<amrex::MultiFab const*> const& a_bcoeff,
  const int bcoeff_comp,
  amrex::Vector<amrex::BCRec> a_bcrec,
  const int ncomp,
  const int do_avgDown,
  amrex::Vector<amrex::MultiFab const*> const& a_boundary)
{
  BL_PROFILE("DiffusionOp::computeDiffFluxes()");

  //----------------------------------------------------------------
  // Checks
  AMREX_ASSERT(m_ncomp == 1 || m_ncomp == ncomp);
  AMREX_ASSERT(a_flux[0][0]->nComp() >= flux_comp + ncomp);
  AMREX_ASSERT(a_phi[0]->nComp() >= phi_comp + ncomp);
  AMREX_ASSERT(a_bcoeff[0]->nComp() >= bcoeff_comp + ncomp);
  AMREX_ASSERT(a_bcrec.size() >= ncomp);

  const int finest_level = m_pelelm->finestLevel();
  const bool mesh_mapping = m_pelelm->hasMeshMapping();
  auto* mm = mesh_mapping ? m_pelelm->meshMap() : nullptr;

  const int have_density = (a_density.empty()) ? 0 : 1;
  const int have_boundary = (a_boundary.empty()) ? 0 : 1;

  // Duplicate phi since it is modified by the LinOp
  // and if have_density -> divide by density
  amrex::Vector<amrex::MultiFab> phi;
  phi.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    phi.emplace_back(
      a_phi[lev]->boxArray(), a_phi[lev]->DistributionMap(), ncomp, 1,
      amrex::MFInfo(), a_phi[lev]->Factory());
  }
  if (have_density == 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab::Copy(
        phi[lev], *a_phi[lev], phi_comp, 0, ncomp, phi[lev].nGrowVect());
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto const& a_phi_ma = a_phi[lev]->const_arrays();
      auto const& a_rho_ma = a_density[lev]->const_arrays();
      auto const& phi_ma = phi[lev].arrays();
      amrex::ParallelFor(
        phi[lev], phi[lev].nGrowVect(), ncomp,
        [a_phi_ma, a_rho_ma, phi_ma, phi_comp] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          amrex::Array4<amrex::Real const> a_phi_arr(
            a_phi_ma[box_no], phi_comp);
          phi_ma[box_no](i, j, k, n) =
            a_phi_arr(i, j, k, n) / a_rho_ma[box_no](i, j, k);
        });
    }
    amrex::Gpu::streamSynchronize();
  }

  // LinOp is \alpha A \phi - \beta \nabla \cdot B \nabla \phi
  // => \alpha = 0, A doesn't matter
  // => \beta = -1.0, B face centered diffusivity a_bcoeff

  // Set scalars \alpha & \beta
  constexpr amrex::Real alpha = 0.0;
  constexpr amrex::Real beta = -1.0;
  m_scal_apply_op->setScalars(alpha, beta);

  // Get fluxes on a m_ncomp component(s) basis
  for (int comp = 0; comp < ncomp; comp += m_ncomp) {

    // Component based vector of data
    amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> fluxes(
      finest_level + 1);
    amrex::Vector<amrex::MultiFab> component;
    amrex::Vector<amrex::MultiFab> laps;
    amrex::Vector<amrex::MultiFab> boundary;

    // Allow for component specific LinOp BC
    m_scal_apply_op->setDomainBC(
      m_pelelm->getDiffusionLinOpBC(amrex::Orientation::low, a_bcrec[comp]),
      m_pelelm->getDiffusionLinOpBC(amrex::Orientation::high, a_bcrec[comp]));

    for (int lev = 0; lev <= finest_level; ++lev) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        fluxes[lev][idim] = new amrex::MultiFab(
          *a_flux[lev][idim], amrex::make_alias, flux_comp + comp, m_ncomp);
      }
      component.emplace_back(phi[lev], amrex::make_alias, comp, m_ncomp);
      if (have_boundary != 0) {
        boundary.emplace_back(
          *a_boundary[lev], amrex::make_alias, comp, m_ncomp);
      } else {
        boundary.emplace_back(phi[lev], amrex::make_alias, comp, m_ncomp);
      }

      constexpr int doZeroVisc = 1;
      constexpr int addTurbContrib = 1;
      amrex::Vector<amrex::BCRec> subBCRec = {
        a_bcrec.begin() + comp, a_bcrec.begin() + comp + m_ncomp};
      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> bcoeff_ec =
        m_pelelm->getDiffusivity(
          lev, bcoeff_comp + comp, m_ncomp, doZeroVisc, subBCRec,
          *a_bcoeff[lev], addTurbContrib);
      if (mesh_mapping) {
        // bcoeff_i -> (J/fac_i^2) . bcoeff on face i; returned fluxes
        // live in Xi-space (consumers apply their own /J on the
        // divergence to recover physical-space terms).
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          const auto& fac_ma = mm->fac_fc(lev, idim).const_arrays();
          const auto& detJ_ma = mm->detJ_fc(lev, idim).const_arrays();
          const auto& b_ma = bcoeff_ec[idim].arrays();
          const int nc = idim;
          amrex::ParallelFor(
            bcoeff_ec[idim], amrex::IntVect(0), m_ncomp,
            [=] AMREX_GPU_DEVICE(
              int box_no, int i, int j, int k, int n) noexcept {
              const amrex::Real f = fac_ma[box_no](i, j, k, nc);
              const amrex::Real dJ = detJ_ma[box_no](i, j, k);
              b_ma[box_no](i, j, k, n) *= dJ / (f * f);
            });
        }
        amrex::Gpu::streamSynchronize();
      }
      laps.emplace_back(
        a_phi[lev]->boxArray(), a_phi[lev]->DistributionMap(), m_ncomp, 1,
        amrex::MFInfo(), a_phi[lev]->Factory());
#ifdef AMREX_USE_EB
      m_scal_apply_op->setBCoeffs(
        lev, GetArrOfConstPtrs(bcoeff_ec), amrex::MLMG::Location::FaceCentroid);
#else
      m_scal_apply_op->setBCoeffs(lev, GetArrOfConstPtrs(bcoeff_ec));
#endif
      m_scal_apply_op->setLevelBC(lev, &boundary[lev]);
    }

    amrex::MLMG mlmg(*m_scal_apply_op);
    mlmg.apply(GetVecOfPtrs(laps), GetVecOfPtrs(component));
#ifdef AMREX_USE_EB
    mlmg.getFluxes(
      fluxes, GetVecOfPtrs(component), amrex::MLMG::Location::FaceCentroid);
#else
    mlmg.getFluxes(
      fluxes, GetVecOfPtrs(component), amrex::MLMG::Location::FaceCenter);
#endif
    for (int lev = 0; lev <= finest_level; ++lev) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        delete fluxes[lev][idim];
      }
    }
  }

  // Average down if requested
  if (do_avgDown != 0) {
    avgDownFluxes(a_flux, flux_comp, ncomp);
  }
}

#ifdef AMREX_USE_EB
void
DiffusionOp::computeDiffFluxes(
  amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> const& a_flux,
  const int flux_comp,
  amrex::Vector<amrex::MultiFab*> const& a_EBflux,
  const int ebflux_comp,
  amrex::Vector<amrex::MultiFab const*> const& a_phi,
  const int phi_comp,
  amrex::Vector<amrex::MultiFab const*> const& a_density,
  amrex::Vector<amrex::MultiFab const*> const& a_bcoeff,
  const int bcoeff_comp,
  amrex::Vector<amrex::MultiFab const*> const& a_EBvalue,
  amrex::Vector<amrex::MultiFab const*> const& a_EBbcoeff,
  amrex::Vector<amrex::BCRec> a_bcrec,
  const int ncomp,
  const int do_avgDown,
  amrex::Vector<amrex::MultiFab const*> const& a_boundary)
{
  BL_PROFILE("DiffusionOp::computeDiffFluxes()");

  //----------------------------------------------------------------
  // Checks
  AMREX_ASSERT(m_ncomp == 1 || m_ncomp == ncomp);
  AMREX_ASSERT(a_flux[0][0]->nComp() >= flux_comp + ncomp);
  AMREX_ASSERT(a_phi[0]->nComp() >= phi_comp + ncomp);
  AMREX_ASSERT(a_bcoeff[0]->nComp() >= bcoeff_comp + ncomp);
  AMREX_ASSERT(a_bcrec.size() >= ncomp);

  const int finest_level = m_pelelm->finestLevel();

  const int have_density = (a_density.empty()) ? 0 : 1;
  const int have_boundary = (a_boundary.empty()) ? 0 : 1;

  // Duplicate phi since it is modified by the LinOp
  // and if have_density -> divide by density
  amrex::Vector<amrex::MultiFab> phi;
  phi.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    phi.emplace_back(
      a_phi[lev]->boxArray(), a_phi[lev]->DistributionMap(), ncomp, 1,
      amrex::MFInfo(), a_phi[lev]->Factory());
  }
  if (have_density == 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab::Copy(
        phi[lev], *a_phi[lev], phi_comp, 0, ncomp, phi[lev].nGrowVect());
    }
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto const& a_phi_ma = a_phi[lev]->const_arrays();
      auto const& a_rho_ma = a_density[lev]->const_arrays();
      auto const& phi_ma = phi[lev].arrays();
      amrex::ParallelFor(
        phi[lev], phi[lev].nGrowVect(), ncomp,
        [a_phi_ma, a_rho_ma, phi_ma, phi_comp] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          amrex::Array4<amrex::Real const> a_phi_arr(
            a_phi_ma[box_no], phi_comp);
          phi_ma[box_no](i, j, k, n) =
            a_phi_arr(i, j, k, n) / a_rho_ma[box_no](i, j, k);
        });
    }
    amrex::Gpu::streamSynchronize();
  }

  // LinOp is \alpha A \phi - \beta \nabla \cdot B \nabla \phi
  // => \alpha = 0, A doesn't matter
  // => \beta = -1.0, B face centered diffusivity a_bcoeff

  // Set scalars \alpha & \beta
  constexpr amrex::Real alpha = 0.0;
  constexpr amrex::Real beta = -1.0;
  m_scal_apply_op->setScalars(alpha, beta);

  // Get fluxes on a m_ncomp component(s) basis
  for (int comp = 0; comp < ncomp; comp += m_ncomp) {

    // Component based vector of data
    amrex::Vector<
      amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>>
      fluxes(finest_level + 1);
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> ebfluxes;
    amrex::Vector<amrex::MultiFab> component;
    amrex::Vector<amrex::MultiFab> laps;
    amrex::Vector<amrex::MultiFab> boundary;

    // Allow for component specific LinOp BC
    m_scal_apply_op->setDomainBC(
      m_pelelm->getDiffusionLinOpBC(amrex::Orientation::low, a_bcrec[comp]),
      m_pelelm->getDiffusionLinOpBC(amrex::Orientation::high, a_bcrec[comp]));

    for (int lev = 0; lev <= finest_level; ++lev) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        fluxes[lev][idim] = std::make_unique<amrex::MultiFab>(
          *a_flux[lev][idim], amrex::make_alias, flux_comp + comp, m_ncomp);
      }
      ebfluxes.push_back(
        std::make_unique<amrex::MultiFab>(
          *a_EBflux[lev], amrex::make_alias, ebflux_comp + comp, m_ncomp));
      component.emplace_back(phi[lev], amrex::make_alias, comp, m_ncomp);
      if (have_boundary != 0) {
        boundary.emplace_back(
          *a_boundary[lev], amrex::make_alias, comp, m_ncomp);
      } else {
        boundary.emplace_back(phi[lev], amrex::make_alias, comp, m_ncomp);
      }
      constexpr int doZeroVisc = 1;
      amrex::Vector<amrex::BCRec> subBCRec = {
        a_bcrec.begin() + comp, a_bcrec.begin() + comp + m_ncomp};
      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> bcoeff_ec =
        m_pelelm->getDiffusivity(
          lev, bcoeff_comp + comp, m_ncomp, doZeroVisc, subBCRec,
          *a_bcoeff[lev]);
      laps.emplace_back(
        a_phi[lev]->boxArray(), a_phi[lev]->DistributionMap(), m_ncomp, 1,
        amrex::MFInfo(), a_phi[lev]->Factory());
      m_scal_apply_op->setBCoeffs(
        lev, GetArrOfConstPtrs(bcoeff_ec), amrex::MLMG::Location::FaceCentroid);
      m_scal_apply_op->setLevelBC(lev, &boundary[lev]);
      m_scal_apply_op->setEBDirichlet(lev, *a_EBvalue[lev], *a_EBbcoeff[lev]);
    }

    amrex::MLMG mlmg(*m_scal_apply_op);
    mlmg.apply(GetVecOfPtrs(laps), GetVecOfPtrs(component));
    mlmg.getFluxes(
      GetVecOfArrOfPtrs(fluxes), GetVecOfPtrs(component),
      amrex::MLMG::Location::FaceCentroid);
    mlmg.getEBFluxes(GetVecOfPtrs(ebfluxes), GetVecOfPtrs(component));
  }

  // Average down if requested
  if (do_avgDown != 0) {
    avgDownFluxes(a_flux, flux_comp, ncomp);
  }
}
#endif

void
DiffusionOp::computeGradient(
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>& a_grad,
  const amrex::Vector<amrex::MultiFab*>& a_laps,
  const amrex::Vector<amrex::MultiFab const*>& a_phi,
  const amrex::Vector<amrex::MultiFab const*>& a_boundary,
  const amrex::BCRec& a_bcrec,
  const int do_avgDown,
  const int comp) const
{
  BL_PROFILE("DiffusionOp::computeGradient()");

  // Do I need the Laplacian out ?
  const int need_laplacian = (a_laps.empty()) ? 0 : 1;
  // Force updating the operator
  for (int lev = 0; lev <= m_pelelm->finestLevel(); ++lev) {
    m_gradient_op->setBCoeffs(lev, -1.0);
  }

  // Checks: one components only and 1 ghost cell at least
  AMREX_ASSERT(a_phi[0]->nComp() > comp);
  AMREX_ASSERT(a_phi[0]->nGrow() >= 1);

  const int finest_level = m_pelelm->finestLevel();
  const int have_boundary = (a_boundary.empty()) ? 0 : 1;

  // Set domainBCs
  m_gradient_op->setDomainBC(
    m_pelelm->getDiffusionLinOpBC(amrex::Orientation::low, a_bcrec),
    m_pelelm->getDiffusionLinOpBC(amrex::Orientation::high, a_bcrec));

  // Duplicate phi since it is modified by the LinOp
  // and setup level BCs
  amrex::Vector<amrex::MultiFab> phi;
  phi.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> boundary;
  boundary.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> laps;
  for (int lev = 0; lev <= finest_level; ++lev) {
    phi.emplace_back(
      a_phi[lev]->boxArray(), a_phi[lev]->DistributionMap(), 1, 1,
      amrex::MFInfo(), a_phi[lev]->Factory());
    boundary.emplace_back(
      a_phi[lev]->boxArray(), a_phi[lev]->DistributionMap(), 1, 1,
      amrex::MFInfo(), a_phi[lev]->Factory());

    amrex::MultiFab::Copy(phi[lev], *a_phi[lev], comp, 0, 1, 1);

    if (have_boundary != 0) {
      amrex::MultiFab::Copy(boundary[lev], *a_boundary[lev], 0, 0, 1, 1);
    } else {
      amrex::MultiFab::Copy(boundary[lev], *a_phi[lev], comp, 0, 1, 1);
    }

    m_gradient_op->setLevelBC(lev, &boundary[lev]);
    if (need_laplacian != 0) {
      laps.emplace_back(*a_laps[lev], amrex::make_alias, 0, 1);
    } else {
      laps.emplace_back(
        a_phi[lev]->boxArray(), a_phi[lev]->DistributionMap(), 1, 1,
        amrex::MFInfo(), a_phi[lev]->Factory());
    }
  }

  amrex::MLMG mlmg(*m_gradient_op);
  mlmg.apply(GetVecOfPtrs(laps), GetVecOfPtrs(phi));
#ifdef AMREX_USE_EB
  mlmg.getFluxes(
    a_grad, GetVecOfPtrs(phi), amrex::MLMG::Location::FaceCentroid);
#else
  mlmg.getFluxes(a_grad, GetVecOfPtrs(phi), amrex::MLMG::Location::FaceCenter);
#endif
  if (do_avgDown != 0) {
    avgDownFluxes(a_grad, 0, 1);
  }
}

void
DiffusionOp::avgDownFluxes(
  const amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>>& a_fluxes,
  const int flux_comp,
  const int ncomp) const
{

  const int finest_level = m_pelelm->finestLevel();

  for (int lev = finest_level; lev > 0; --lev) {
    // Get the requested components only
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> flux_fine;
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> flux_crse;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      flux_fine[idim] = new amrex::MultiFab(
        *a_fluxes[lev][idim], amrex::make_alias, flux_comp, ncomp);
      flux_crse[idim] = new amrex::MultiFab(
        *a_fluxes[lev - 1][idim], amrex::make_alias, flux_comp, ncomp);
    }
#ifdef AMREX_USE_EB
    EB_average_down_faces(
      GetArrOfConstPtrs(flux_fine), flux_crse, m_pelelm->refRatio(lev - 1),
      flux_crse[0]->nGrow());
#else
    average_down_faces(
      GetArrOfConstPtrs(flux_fine), flux_crse, m_pelelm->refRatio(lev - 1),
      flux_crse[0]->nGrow());
#endif
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      delete flux_fine[idim];
      delete flux_crse[idim];
    }
  }
}

void
DiffusionOp::readParameters()
{
  amrex::ParmParse pp("diffusion");

  m_mg_verbose = amrex::max<int>(m_pelelm->getVerbose() - 2, m_mg_verbose);
  pp.query("verbose", m_mg_verbose);
  pp.query("atol", m_mg_atol);
  pp.query("rtol", m_mg_rtol);
  pp.query("max_iter", m_mg_max_iter);
  pp.query("bottom_solver", m_mg_bottom_solver);
  pp.query("max_order", m_mg_maxorder);
  pp.query("mlmg_fail_sdc_miniter", m_mg_fail_sdc_miniter);
  pp.query("mlmg_fail_deltaT_miniter", m_mg_fail_deltaT_miniter);
  pp.query(
    "mlmg_fail_species_maxiter_after_sdc_miniter",
    m_mg_fail_species_maxiter_after_sdc_miniter);
  pp.query(
    "mlmg_fail_temp_maxiter_after_sdc_deltaT_miniter",
    m_mg_fail_temp_maxiter_after_sdc_deltaT_miniter);
}

//---------------------------------------------------------------------------------------
// Tensor Operator

DiffusionTensorOp::DiffusionTensorOp(PeleLM* a_pelelm) : m_pelelm(a_pelelm)
{

  readParameters();

  const int finest_level = m_pelelm->finestLevel();

  auto bcRecVel = m_pelelm->fetchBCRecArray(VELX, AMREX_SPACEDIM);

  // Solve LPInfo
  amrex::LPInfo info_solve;
  info_solve.setMaxCoarseningLevel(m_mg_max_coarsening_level);

#ifdef AMREX_USE_EB
  // Get vector of EB Factory
  amrex::Vector<amrex::EBFArrayBoxFactory const*> ebfactVec;
  for (int lev = 0; lev <= finest_level; ++lev) {
    ebfactVec.push_back(&(m_pelelm->EBFactory(lev)));
  }
#endif

#ifdef AMREX_USE_EB
  m_solve_op = std::make_unique<amrex::MLEBTensorOp>(
    m_pelelm->Geom(0, finest_level), m_pelelm->boxArray(0, finest_level),
    m_pelelm->DistributionMap(0, finest_level), info_solve, ebfactVec);
#else
  m_solve_op = std::make_unique<amrex::MLTensorOp>(
    m_pelelm->Geom(0, finest_level), m_pelelm->boxArray(0, finest_level),
    m_pelelm->DistributionMap(0, finest_level), info_solve);
#endif
  m_solve_op->setMaxOrder(m_mg_maxorder);
  m_solve_op->setDomainBC(
    m_pelelm->getDiffusionTensorOpBC(amrex::Orientation::low, bcRecVel),
    m_pelelm->getDiffusionTensorOpBC(amrex::Orientation::high, bcRecVel));

  // Apply LPInfo (no coarsening)
  amrex::LPInfo info_apply;
  info_apply.setMaxCoarseningLevel(0);

#ifdef AMREX_USE_EB
  m_apply_op = std::make_unique<amrex::MLEBTensorOp>(
    m_pelelm->Geom(0, finest_level), m_pelelm->boxArray(0, finest_level),
    m_pelelm->DistributionMap(0, finest_level), info_apply, ebfactVec);
#else
  m_apply_op = std::make_unique<amrex::MLTensorOp>(
    m_pelelm->Geom(0, finest_level), m_pelelm->boxArray(0, finest_level),
    m_pelelm->DistributionMap(0, finest_level), info_apply);
#endif
  m_apply_op->setMaxOrder(m_mg_maxorder);
  m_apply_op->setDomainBC(
    m_pelelm->getDiffusionTensorOpBC(amrex::Orientation::low, bcRecVel),
    m_pelelm->getDiffusionTensorOpBC(amrex::Orientation::high, bcRecVel));
}

void
DiffusionTensorOp::computeGradientTensor(
  amrex::Vector<amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>> const&
    a_velgrad,
  amrex::Vector<amrex::MultiFab const*> const& a_vel)
{
  // This function returns the velocity gradient tensor at faces
  //
  // The derivatives are put in the array with the following order:
  // component: 0    ,  1    ,  2    ,  3    ,  4    , 5    ,  6    ,  7    ,  8
  // in 2D:     dU/dx,  dV/dx,  dU/dy,  dV/dy
  // in 3D:     dU/dx,  dV/dx,  dW/dx,  dU/dy,  dV/dy, dW/dy,  dU/dz,  dV/dz,
  // dW/dz

  const int finest_level = m_pelelm->finestLevel();

  // Duplicate vel since it may be modified by the TensorOp
  amrex::Vector<amrex::MultiFab> vel;
  vel.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    vel.emplace_back(
      a_vel[lev]->boxArray(), a_vel[lev]->DistributionMap(), AMREX_SPACEDIM, 1,
      amrex::MFInfo(), a_vel[lev]->Factory());
    amrex::MultiFab::Copy(vel[lev], *a_vel[lev], 0, 0, AMREX_SPACEDIM, 1);
  }

  // Set up some parameters
  m_apply_op->setMaxOrder(m_mg_maxorder);
  m_apply_op->setScalars(0.0, 1.0);
  for (int lev = 0; lev <= finest_level; ++lev) {
    m_apply_op->setShearViscosity(lev, 0.0);
#ifdef AMREX_USE_EB
    m_apply_op->setEBShearViscosity(lev, 0.0);
#endif
    m_apply_op->setLevelBC(lev, &vel[lev]);
  }

  // Dummy variable for applying MLMG operator
  amrex::Vector<amrex::MultiFab> divtau;
  for (int lev = 0; lev <= finest_level; ++lev) {
    divtau.emplace_back(
      a_vel[lev]->boxArray(), a_vel[lev]->DistributionMap(), AMREX_SPACEDIM, 1,
      amrex::MFInfo(), a_vel[lev]->Factory());
  }

  // Create MLMG object and apply to setup BCs, etc
  amrex::MLMG mlmg(*m_apply_op);
  mlmg.apply(GetVecOfPtrs(divtau), GetVecOfPtrs(vel));

  // Compute the velocity gradient on faces
  for (int lev = 0; lev <= finest_level; ++lev) {
    m_apply_op->compVelGrad(
      lev, a_velgrad[lev], vel[lev], amrex::MLLinOp::Location::FaceCentroid);
  }
}

void
DiffusionTensorOp::compute_divtau(
  amrex::Vector<amrex::MultiFab*> const& a_divtau,
  amrex::Vector<amrex::MultiFab const*> const& a_vel,
  amrex::Vector<amrex::MultiFab const*> const& a_density,
  amrex::Vector<amrex::MultiFab const*> const& a_beta,
  const amrex::BCRec& a_bcrec,
  const amrex::Real scale)
{
  const int finest_level = m_pelelm->finestLevel();

  const int have_density = (a_density.empty()) ? 0 : 1;
  const bool mesh_mapping = m_pelelm->hasMeshMapping();

  // Duplicate vel since it is modified by the TensorOp
  amrex::Vector<amrex::MultiFab> vel;
  vel.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    vel.emplace_back(
      a_vel[lev]->boxArray(), a_vel[lev]->DistributionMap(), AMREX_SPACEDIM, 2,
      amrex::MFInfo(), a_vel[lev]->Factory());
    amrex::MultiFab::Copy(vel[lev], *a_vel[lev], 0, 0, AMREX_SPACEDIM, 2);
  }

#ifdef AMREX_USE_EB
  // Need a temporary divTau to apply redistribution
  amrex::Vector<amrex::MultiFab> divtau_tmp;
  divtau_tmp.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    divtau_tmp.emplace_back(
      a_divtau[lev]->boxArray(), a_divtau[lev]->DistributionMap(),
      AMREX_SPACEDIM, 2, amrex::MFInfo(), a_divtau[lev]->Factory());
    divtau_tmp[lev].setVal(0.0);
  }

  m_apply_op->setScalars(0.0, -scale);
  for (int lev = 0; lev <= finest_level; ++lev) {
    if (have_density != 0) { // alpha being zero, not sure that this does
                             // anything.
      m_apply_op->setACoeffs(lev, *a_density[lev]);
    }
    constexpr int doZeroVisc = 0;
    constexpr int addTurbContrib = 1;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> beta_ec =
      m_pelelm->getDiffusivity(
        lev, 0, 1, doZeroVisc, {a_bcrec}, *a_beta[lev], addTurbContrib);
    m_apply_op->setShearViscosity(
      lev, GetArrOfConstPtrs(beta_ec), amrex::MLMG::Location::FaceCentroid);
    if (m_pelelm->m_useEBinflow != 0) {
      m_apply_op->setEBShearViscosityWithInflow(
        lev, *a_beta[lev],
        *(m_pelelm->getEBState(
          lev, VELX, AMREX_SPACEDIM, m_pelelm->AmrOldTime)));
    } else {
      m_apply_op->setEBShearViscosity(lev, *a_beta[lev]);
    }
    m_apply_op->setLevelBC(lev, &vel[lev]);
  }

  amrex::MLMG mlmg(*m_apply_op);
  mlmg.apply(GetVecOfPtrs(divtau_tmp), GetVecOfPtrs(vel));

  // Flux redistribute explicit diffusion fluxes into outgoing a_divtau
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::single_level_redistribute(
      divtau_tmp[lev], *a_divtau[lev], 0, AMREX_SPACEDIM, m_pelelm->Geom(lev));
  }

#else
  m_apply_op->setScalars(0.0, -scale);
  for (int lev = 0; lev <= finest_level; ++lev) {
    if (have_density != 0) { // alpha being zero, not sure that this does
                             // anything.
      m_apply_op->setACoeffs(lev, *a_density[lev]);
    }
    constexpr int doZeroVisc = 0;
    constexpr int addTurbContrib = 1;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> beta_ec =
      m_pelelm->getDiffusivity(
        lev, 0, 1, doZeroVisc, {a_bcrec}, *a_beta[lev], addTurbContrib);
    if (mesh_mapping) {
      // beta_i -> (J/fac_i^2) . beta on face i
      auto* mm = m_pelelm->meshMap();
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const auto& fac_ma = mm->fac_fc(lev, idim).const_arrays();
        const auto& detJ_ma = mm->detJ_fc(lev, idim).const_arrays();
        const auto& b_ma = beta_ec[idim].arrays();
        const int nc = idim;
        amrex::ParallelFor(
          beta_ec[idim],
          [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
            const amrex::Real f = fac_ma[box_no](i, j, k, nc);
            const amrex::Real dJ = detJ_ma[box_no](i, j, k);
            b_ma[box_no](i, j, k) *= dJ / (f * f);
          });
      }
      amrex::Gpu::streamSynchronize();
#if defined(AMREX_MLTENSOROP_HAS_MAPPING_FACTORS) && !defined(AMREX_USE_EB)
      // The scaled face viscosity handles the normal-derivative part; the
      // transpose and bulk (cross) terms need the per-direction factors.
      // (m_apply_op is an MLEBTensorOp with EB, which has no mapping API;
      // mesh mapping is rejected at runtime with EB anyway.)
      m_apply_op->setMappingFactors(
        lev, {AMREX_D_DECL(
               &mm->fac_fc(lev, 0), &mm->fac_fc(lev, 1), &mm->fac_fc(lev, 2))});
#endif
    }
    m_apply_op->setShearViscosity(lev, GetArrOfConstPtrs(beta_ec));
    m_apply_op->setLevelBC(lev, &vel[lev]);
  }

  amrex::MLMG mlmg(*m_apply_op);
  mlmg.apply(a_divtau, GetVecOfPtrs(vel));
#endif

  // Under mesh mapping the apply operator produces J . (physical divTau).
  // Divide by J to recover the physical-space viscous force.
  if (mesh_mapping) {
    auto* mm = m_pelelm->meshMap();
    for (int lev = 0; lev <= finest_level; ++lev) {
      for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        amrex::MultiFab::Divide(*a_divtau[lev], mm->detJ_cc(lev), 0, n, 1, 0);
      }
    }
  }

  if (have_density != 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto const& divtau_ma = a_divtau[lev]->arrays();
      auto const& rho_ma = a_density[lev]->const_arrays();
      amrex::ParallelFor(
        *a_divtau[lev], amrex::IntVect(0), AMREX_SPACEDIM,
        [divtau_ma, rho_ma] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          divtau_ma[box_no](i, j, k, n) /= rho_ma[box_no](i, j, k);
        });
    }
    amrex::Gpu::streamSynchronize();
  }
}

void
DiffusionTensorOp::diffuse_velocity(
  amrex::Vector<amrex::MultiFab*> const& a_vel,
  amrex::Vector<amrex::MultiFab const*> const& a_density,
  amrex::Vector<amrex::MultiFab const*> const& a_beta,
  const amrex::BCRec& a_bcrec,
  const amrex::Real a_dt)
{

  const int finest_level = m_pelelm->finestLevel();

  const int have_density = (a_density.empty()) ? 0 : 1;

  AMREX_ASSERT(
    (!m_pelelm->m_incompressible && have_density) ||
    (m_pelelm->m_incompressible && !have_density));

  const bool mesh_mapping = m_pelelm->hasMeshMapping();
  auto* mm = mesh_mapping ? m_pelelm->meshMap() : nullptr;

  // Under mesh mapping we need a cell-centered A-coeff of rho . J
  // (density . detJ).  Build scratch MFs that live as long as the solve.
  amrex::Vector<amrex::MultiFab> a_coeff_scaled;
  if (mesh_mapping) {
    a_coeff_scaled.resize(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      a_coeff_scaled[lev].define(
        a_vel[lev]->boxArray(), a_vel[lev]->DistributionMap(), 1, 0,
        amrex::MFInfo(), a_vel[lev]->Factory());
      if (have_density != 0) {
        amrex::MultiFab::Copy(a_coeff_scaled[lev], *a_density[lev], 0, 0, 1, 0);
      } else {
        a_coeff_scaled[lev].setVal(m_pelelm->m_rho);
      }
      amrex::MultiFab::Multiply(
        a_coeff_scaled[lev], mm->detJ_cc(lev), 0, 0, 1, 0);
    }
  }

  m_solve_op->setScalars(1.0, a_dt);
  for (int lev = 0; lev <= finest_level; ++lev) {
    if (mesh_mapping) {
      m_solve_op->setACoeffs(lev, a_coeff_scaled[lev]);
    } else if (have_density != 0) {
      m_solve_op->setACoeffs(lev, *a_density[lev]);
    } else {
      m_solve_op->setACoeffs(lev, m_pelelm->m_rho);
    }
    constexpr int doZeroVisc = 0;
    constexpr int addTurbContrib = 1;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> beta_ec =
      m_pelelm->getDiffusivity(
        lev, 0, 1, doZeroVisc, {a_bcrec}, *a_beta[lev], addTurbContrib);
    if (mesh_mapping) {
      // beta_i -> (J/fac_i^2) . beta on face i
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const auto& fac_ma = mm->fac_fc(lev, idim).const_arrays();
        const auto& detJ_ma = mm->detJ_fc(lev, idim).const_arrays();
        const auto& b_ma = beta_ec[idim].arrays();
        const int nc = idim;
        amrex::ParallelFor(
          beta_ec[idim],
          [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
            const amrex::Real f = fac_ma[box_no](i, j, k, nc);
            const amrex::Real dJ = detJ_ma[box_no](i, j, k);
            b_ma[box_no](i, j, k) *= dJ / (f * f);
          });
      }
      amrex::Gpu::streamSynchronize();
#if defined(AMREX_MLTENSOROP_HAS_MAPPING_FACTORS) && !defined(AMREX_USE_EB)
      // The scaled face viscosity handles the normal-derivative part; the
      // transpose and bulk (cross) terms need the per-direction factors.
      m_solve_op->setMappingFactors(
        lev, {AMREX_D_DECL(
               &mm->fac_fc(lev, 0), &mm->fac_fc(lev, 1), &mm->fac_fc(lev, 2))});
#endif
    }
#ifdef AMREX_USE_EB
    m_solve_op->setShearViscosity(
      lev, GetArrOfConstPtrs(beta_ec), amrex::MLMG::Location::FaceCentroid);
    if (m_pelelm->m_useEBinflow != 0) {
      m_solve_op->setEBShearViscosityWithInflow(
        lev, *a_beta[lev],
        *(m_pelelm->getEBState(
          lev, VELX, AMREX_SPACEDIM, m_pelelm->AmrOldTime)));
    } else {
      m_solve_op->setEBShearViscosity(lev, *a_beta[lev]);
    }
#else
    m_solve_op->setShearViscosity(lev, GetArrOfConstPtrs(beta_ec));
#endif
    m_solve_op->setLevelBC(lev, a_vel[lev]);
  }

  amrex::Vector<amrex::MultiFab> rhs;
  rhs.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    rhs.emplace_back(
      a_vel[lev]->boxArray(), a_vel[lev]->DistributionMap(), AMREX_SPACEDIM, 0);
  }
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto const& rhs_ma = rhs[lev].arrays();
    auto const& vel_ma = a_vel[lev]->const_arrays();
    if (m_pelelm->m_incompressible == 0) {
      auto const& rho_ma = a_density[lev]->const_arrays();
      amrex::ParallelFor(
        rhs[lev], amrex::IntVect(0), AMREX_SPACEDIM,
        [rhs_ma, vel_ma, rho_ma] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          rhs_ma[box_no](i, j, k, n) =
            rho_ma[box_no](i, j, k) * vel_ma[box_no](i, j, k, n);
        });
    } else {
      amrex::ParallelFor(
        rhs[lev], amrex::IntVect(0), AMREX_SPACEDIM,
        [rhs_ma, vel_ma, rho = m_pelelm->m_rho] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k, int n) noexcept {
          rhs_ma[box_no](i, j, k, n) = rho * vel_ma[box_no](i, j, k, n);
        });
    }
  }
  amrex::Gpu::streamSynchronize();

  // Under mesh mapping, rhs = rho . u -> rho . J . u  (to match the
  // A-coeff scaling above so the implicit operator remains consistent).
  if (mesh_mapping) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        amrex::MultiFab::Multiply(rhs[lev], mm->detJ_cc(lev), 0, n, 1, 0);
      }
    }
  }

  amrex::MLMG mlmg(*m_solve_op);

  if (m_mg_verbose > 0) {
    amrex::Print() << "MLMG: Velocity Diffusion\n";
  }

  // Maximum iterations for / ConjugateGradients may change for debugging
  // purposes
  int max_iter = m_mg_max_iter;
  if (m_pelelm->m_mlmg_fail_plt_residuals) {

    bool sdc_iters_met = (m_pelelm->m_sdcIter >= m_mg_fail_sdc_miniter);

    // Only check SDC iter
    if (sdc_iters_met && (m_mg_fail_maxiter_after_sdc_miniter > 0)) {
      max_iter = m_mg_fail_maxiter_after_sdc_miniter;
      amrex::Print() << "      Limiting velocity diffusion MLMG max_iter to "
                     << max_iter << " (SDC iter [" << m_pelelm->m_sdcIter
                     << "] >= " << m_mg_fail_sdc_miniter << ")\n";
    }
  }

  mlmg.setMaxIter(max_iter);
  mlmg.setMaxFmgIter(m_mg_max_fmg_iter);
  mlmg.setBottomMaxIter(m_mg_bottom_maxiter);

  // Verbosity for MultiGrid / ConjugateGradients
  mlmg.setVerbose(m_mg_verbose);
  mlmg.setBottomVerbose(m_mg_bottom_verbose);

  mlmg.setPreSmooth(m_num_pre_smooth);
  mlmg.setPostSmooth(m_num_post_smooth);

  // Solve
  if (!m_pelelm->m_mlmg_fail_plt_residuals) {
    mlmg.solve(a_vel, GetVecOfConstPtrs(rhs), m_mg_rtol, m_mg_atol);
  } else {
    mlmg.setThrowException(true);
    mlmg.setConvergenceNormType(amrex::MLMGNormType::bnorm);

    try {
      mlmg.solve(a_vel, GetVecOfConstPtrs(rhs), m_mg_rtol, m_mg_atol);
    } catch (const std::exception& e) {
      amrex::Print() << "\n"
                     << "  *** Velocity diffusion MLMG solve failed! ***\n"
                     << "  Error: " << e.what() << "\n"
                     << "  Dumping residuals for debugging...\n";
      m_pelelm->WriteMLMGResidual(
        mlmg, a_vel, GetVecOfConstPtrs(rhs), "vel_diffusion",
        m_pelelm->m_nstep);

      amrex::Abort("MLMG solve for velocity diffusion failed");
    }
  }
}

void
DiffusionTensorOp::readParameters()
{
  amrex::ParmParse pp("tensor_diffusion");

  m_mg_verbose = amrex::max<int>(m_pelelm->getVerbose() - 2, m_mg_verbose);
  pp.query("verbose", m_mg_verbose);
  pp.query("atol", m_mg_atol);
  pp.query("rtol", m_mg_rtol);
  pp.query("max_iter", m_mg_max_iter);
  pp.query("bottom_solver", m_mg_bottom_solver);
  pp.query("mg_max_coarsening_level", m_mg_max_coarsening_level);
  pp.query("mg_max_fmg_iter", m_mg_max_fmg_iter);
  pp.query("num_pre_smooth", m_num_pre_smooth);
  pp.query("num_post_smooth", m_num_post_smooth);
  pp.query("mlmg_fail_sdc_miniter", m_mg_fail_sdc_miniter);
  pp.query(
    "mlmg_maxiter_after_sdc_miniter", m_mg_fail_maxiter_after_sdc_miniter);
}
