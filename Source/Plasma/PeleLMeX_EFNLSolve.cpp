#include <PeleLMeX.H>
#include <PeleLMeX_K.H>
#include <PeleLMeX_EF_K.H>
#include <MLGMRES.H>
#include <PeleLMeX_DiffusionOp.H>
#include <PeleLMeX_Utils.H>
#include <hydro_utils.H>

PrecondOp*
PeleLM::getPrecondOp()
{
  if (!m_precond_op)
    m_precond_op.reset(new PrecondOp(this));
  return m_precond_op.get();
}

void
PeleLM::implicitNonLinearSolve(
  const int sdcIter,
  const amrex::Real a_dt,
  const std::unique_ptr<AdvanceDiffData>& diffData,
  const std::unique_ptr<AdvanceAdvData>& advData)
{
  BL_PROFILE_VAR("PeleLMeX::implicitNonLinearSolve()", implicitNonLinearSolve);

  const amrex::Real strt_time = amrex::ParallelDescriptor::second();

  //------------------------------------------------------------------------
  // Pre-solve

  // Get cell-centered charged species transport coefficients
  calcEFTransport(AmrNewTime);

  // Substepping of non-linear solve
  dtsub = a_dt / ef_substep;

  // Pass t^{n} nE/PhiV from leveldata to leveldatanlsolve
  // t^{n} have been fillpatched already
  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get t^{n} data pointer
    auto ldata_p = getLevelDataPtr(lev, AmrOldTime);
    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
    amrex::MultiFab::Copy(
      ldataNLs_p->nlState, ldata_p->state, NE, 0, 1, m_nGrowState);
    amrex::MultiFab::Copy(
      ldataNLs_p->nlState, ldata_p->state, PHIV, 1, 1, m_nGrowState);
  }

  // Gradient of PhiV at t^{n}
  constexpr int do_avgDown = 0; // TODO or should I ?
  auto bcRecPhiV = fetchBCRecArray(PHIV, 1);
  getDiffusionOp()->computeGradient(
    getNLgradPhiVVect(), {}, // don't need the laplacian out
    GetVecOfConstPtrs(getPhiVVect(AmrOldTime)), {}, bcRecPhiV[0], do_avgDown);

  // Stash away a copy of umac
  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      amrex::MultiFab::Copy(
        ldataNLs_p->umac[idim], advData->umac[lev][idim], 0, 0, 1, 0);
    }
  }

  // Setup MLGMRES
  MLGMRESSolver gmres;
  int GMRES_tot_count = 0;
  if (!m_ef_use_PETSC_direct) {
    gmres.define(this, 2, 1);
    MLJtimesVFunc jtv = &PeleLM::jTimesV;
    gmres.setJtimesV(jtv);
    MLNormFunc normF = &PeleLM::nlSolveNorm;
    gmres.setNorm(normF);
    MLPrecondFunc prec = &PeleLM::applyPrecond;
    gmres.setPrecond(prec);
  }

  //------------------------------------------------------------------------
  // Outer subcycling loop
  if (ef_substep > 1)
    amrex::Abort("Non-linear solve sub-stepping not re-implemented yet");
  int NK_tot_count = 0;
  for (int sstep = 0; sstep < ef_substep; ++sstep) {
    curtime = getTime(0, AmrOldTime) + (sstep + 1) * dtsub;

    // -----------------
    // Pre-Newton
    // Set up the NL state scaling
    getNLStateScaling(nE_scale, phiV_scale);
    nE_scale = (nE_scale > 1.0e-12) ? nE_scale : 1.0;
    phiV_scale = (phiV_scale > 1.0e-12) ? phiV_scale : 1.0;
    if (ef_verbose) {
      amrex::Print() << "(" << sstep << ") ne scaling: " << nE_scale << "\n";
      amrex::Print() << "(" << sstep << ") phiV scaling: " << phiV_scale
                     << "\n";
    }
    scaleNLState(nE_scale, phiV_scale);

    // Compute the background charge distribution at curtime
    computeBGcharge(curtime, diffData, advData);
    // WriteDebugPlotFile(GetVecOfConstPtrs(getNLBGChargeVect()),"NLBGCharge");

    // Newton initial guess
    // TODO newton initial guess
    nlSolveNorm(getNLstateVect(), nl_stateNorm);

    // Initial NL residual: update residual scaling and preconditioner
    int update_scaling = 1;
    int update_precond = 1;
    nonLinearResidual(
      dtsub, getNLstateVect(), getNLresidVect(), update_scaling,
      update_precond);
    nlSolveNorm(getNLresidVect(), nl_residNorm);
    // WriteDebugPlotFile(GetVecOfConstPtrs(getNLresidVect()),"NLResInit");

    // Check for convergence
    amrex::Real scaledResnE, scaledResphiV;
    getNLResidScaling(scaledResnE, scaledResphiV);
    amrex::Real max_nlres = amrex::max<amrex::Real>(scaledResnE, scaledResphiV);
    if (max_nlres <= m_ef_newtonTol) {
      if (ef_verbose) {
        amrex::Print() << " No Newton iteration needed, exiting. \n";
      }
      return;
    }

    // -----------------
    // Newton iteration
    int exit_newton = 0;
    int NK_ite = 0;
    do {
      NK_ite += 1;

      // Verbose
      if (ef_verbose) {
        amrex::Print() << " Newton it: " << NK_ite << " L2**2 residual: "
                       << 0.5 * nl_residNorm * nl_residNorm
                       << ". Linf residual: " << max_nlres << "\n";
      }

      // Solve for Newton direction
      amrex::Vector<amrex::MultiFab> newtonDir;
      newtonDir.reserve(finest_level + 1);
      for (int lev = 0; lev <= finest_level; ++lev) {
        newtonDir.emplace_back(
          grids[lev], dmap[lev], 2, 1, amrex::MFInfo(), Factory(lev));
        newtonDir[lev].setVal(0.0, 0, 2, 1);
      }
      if (!m_ef_use_PETSC_direct) {
        const amrex::Real S_tol = m_ef_GMRES_reltol;
        const amrex::Real S_tol_abs = m_ef_GMRES_abstol;
        GMRES_tot_count += gmres.solve(
          GetVecOfPtrs(newtonDir), getNLresidVect(), S_tol_abs, S_tol);
      } else {
      }
      // WriteDebugPlotFile(GetVecOfConstPtrs(newtonDir),"newtonDir_"+std::to_string(NK_ite));
      amrex::Real newtonDir_Norm;
      nlSolveNorm(GetVecOfPtrs(newtonDir), newtonDir_Norm);
      // Print() << " newtonDir_Norm " << newtonDir_Norm << "\n";

      // Linesearch & update state TODO
      updateNLState(GetVecOfPtrs(newtonDir));
      nlSolveNorm(getNLstateVect(), nl_stateNorm);
      update_scaling = 0;
      update_precond = 1;
      nonLinearResidual(
        dtsub, getNLstateVect(), getNLresidVect(), update_scaling,
        update_precond);
      nlSolveNorm(getNLresidVect(), nl_residNorm);
      getNLResidScaling(scaledResnE, scaledResphiV);
      max_nlres = amrex::max<amrex::Real>(scaledResnE, scaledResphiV);
      // WriteDebugPlotFile(GetVecOfConstPtrs(getNLstateVect()),"NLState_"+std::to_string(NK_ite));
      // WriteDebugPlotFile(GetVecOfConstPtrs(getNLresidVect()),"NLResid_"+std::to_string(NK_ite));

      // Exit condition
      exit_newton = testExitNewton(NK_ite, max_nlres, newtonDir_Norm);
    } while (!exit_newton);
    NK_tot_count += NK_ite;

    // -----------------
    // Post-Newton
    // Increment the forcing term
    incrementElectronForcing(sstep, advData);
    // Unscale nl_state and if not last subcycle update 'old' state
    for (int lev = 0; lev <= finest_level; ++lev) {
      m_leveldatanlsolve[lev]->nlState.mult(nE_scale, 0, 1, 0);
      m_leveldatanlsolve[lev]->nlState.mult(phiV_scale, 1, 1, 0);
    }
  }

  // Update the state
  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get t^{n} data pointer
    auto ldata_p = getLevelDataPtr(lev, AmrNewTime);
    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
    int nGrowNL = 0;
    amrex::MultiFab::Copy(
      ldata_p->state, ldataNLs_p->nlState, 0, NE, 1, nGrowNL);
    amrex::MultiFab::Copy(
      ldata_p->state, ldataNLs_p->nlState, 1, PHIV, 1, nGrowNL);
  }

  if (ef_verbose) {
    amrex::Real run_time = amrex::ParallelDescriptor::second() - strt_time;
    amrex::ParallelDescriptor::ReduceRealMax(
      run_time, amrex::ParallelDescriptor::IOProcessorNumber());
    if (!m_ef_use_PETSC_direct) {
      amrex::Real avgGMRES = (float)GMRES_tot_count / (float)NK_tot_count;
      amrex::Print() << "  [" << sdcIter << "] dt: " << a_dt
                     << " - Avg GMRES/Newton: " << avgGMRES << "\n";
    }
    amrex::Print() << "  >> PeleLMeX::implicitNLSolve() " << run_time << "\n";
  }

  // amrex::VisMF::Write(advData->Forcing[0],"ForcingNE");
}

int
PeleLM::testExitNewton(
  const int newtonIter,
  const amrex::Real max_res,
  const amrex::Real norm_NewtonDir)
{
  int exit = 0;
  if (max_res <= m_ef_newtonTol || norm_NewtonDir <= 1e-11) {
    exit = 1;
    if (ef_verbose) {
      amrex::Print() << " Newton iterations converged: \n";
      amrex::Print() << " Final Newton L2**2 res norm : "
                     << 0.5 * nl_residNorm * nl_residNorm << "\n";
      amrex::Print() << " Final Newton Linf res norm : " << max_res << "\n";
    }
  }

  if (newtonIter >= m_ef_maxNewtonIter && exit == 0) {
    exit = 1;
    amrex::Print()
      << " WARNING: Max Newton iteration reached without convergence !!! \n";
    amrex::Print() << " Final Newton L2**2 res norm : "
                   << 0.5 * nl_residNorm * nl_residNorm << "\n";
    amrex::Print() << " Final Newton Linf res norm : " << max_res << "\n";
  }
  return exit;
}

void
PeleLM::updateNLState(const amrex::Vector<amrex::MultiFab*>& a_update)
{
  // AverageDown the newton direction
  /*
  for (int lev = finest_level; lev > 0; --lev) {
#ifdef AMREX_USE_EB
     EB_average_down(*a_update[lev],
                     *a_update[lev-1],
                     0,2,refRatio(lev-1));
#else
     average_down(*a_update[lev],
                  *a_update[lev-1],
                  0,2,refRatio(lev-1));
#endif
  }
  */
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev); // NL data
    ldataNLs_p->nlState.plus(*a_update[lev], 0, 2, 0);
  }

  // AverageDown the new state
  for (int lev = finest_level; lev > 0; --lev) {
    auto ldataNLsFine_p = getLevelDataNLSolvePtr(lev);     // NL data
    auto ldataNLsCrse_p = getLevelDataNLSolvePtr(lev - 1); // NL data
#ifdef AMREX_USE_EB
    EB_average_down(
      ldataNLsFine_p->nlState, ldataNLsCrse_p->nlState, 0, 2,
      refRatio(lev - 1));
#else
    average_down(
      ldataNLsFine_p->nlState, ldataNLsCrse_p->nlState, 0, 2,
      refRatio(lev - 1));
#endif
  }

  // FillBoundary NLState
  /*
  for (int lev = 0; lev <= finest_level; ++lev) {
     auto ldataNLs_p = getLevelDataNLSolvePtr(lev);   // NL data
     ldataNLs_p->nlState.FillBoundary(geom[lev].periodicity());
  }
  */

  // Need to fillpatch the NL state
  // First unscale NLState
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev); // NL data
    ldataNLs_p->nlState.mult(nE_scale, 0, 1, m_nGrowState);
    ldataNLs_p->nlState.mult(phiV_scale, 1, 1, m_nGrowState);
  }

  // FillPatch
  amrex::Vector<amrex::MultiFab> nEState;
  amrex::Vector<amrex::MultiFab> phiVState;
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev); // NL data
    nEState.emplace_back(ldataNLs_p->nlState, amrex::make_alias, 0, 1);
    phiVState.emplace_back(ldataNLs_p->nlState, amrex::make_alias, 1, 1);
  }
  fillPatchNLnE(m_cur_time, GetVecOfPtrs(nEState), m_nGrowState);
  fillPatchNLphiV(m_cur_time, GetVecOfPtrs(phiVState), m_nGrowState);

  // Rescale
  scaleNLState(nE_scale, phiV_scale);
}

void
PeleLM::incrementElectronForcing(
  const int a_sstep, const std::unique_ptr<AdvanceAdvData>& advData)
{

  for (int lev = 0; lev <= finest_level; ++lev) {
    auto ldata_p = getLevelDataPtr(lev, AmrOldTime); // Old time electron
    auto ldataR_p = getLevelDataReactPtr(lev);       // Reaction
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);   // NL data

    auto const& state_o_ma = ldata_p->state.const_arrays();
    auto const& nE_n_ma = ldataNLs_p->nlState.const_arrays();
    auto const& I_R_ma = ldataR_p->I_R.const_arrays();
    auto const& F_ma = advData->Forcing[lev].arrays();
    const amrex::Real scaling = nE_scale;
    const amrex::Real dtinv = 1.0 / dtsub;
    if (a_sstep == 0) {
      amrex::ParallelFor(
        ldata_p->state,
        [state_o_ma, nE_n_ma, I_R_ma, F_ma, scaling,
         dtinv] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
          amrex::Array4<amrex::Real const> nE_o(state_o_ma[box_no], NE);
          amrex::Array4<amrex::Real const> I_R_nE(I_R_ma[box_no], NUM_SPECIES);
          amrex::Array4<amrex::Real> FnE(F_ma[box_no], NUM_SPECIES + 1);
          FnE(i, j, k) =
            (nE_n_ma[box_no](i, j, k) * scaling - nE_o(i, j, k)) * dtinv -
            I_R_nE(i, j, k);
        });
      // Shift outside?
      amrex::Gpu::streamSynchronize();
    } else {
      amrex::ParallelFor(
        ldata_p->state,
        [state_o_ma, nE_n_ma, I_R_ma, F_ma, scaling,
         dtinv] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
          amrex::Array4<amrex::Real const> nE_o(state_o_ma[box_no], NE);
          amrex::Array4<amrex::Real const> I_R_nE(I_R_ma[box_no], NUM_SPECIES);
          amrex::Array4<amrex::Real> FnE(F_ma[box_no], NUM_SPECIES + 1);
          FnE(i, j, k) +=
            (nE_n_ma[box_no](i, j, k) * scaling - nE_o(i, j, k)) * dtinv -
            I_R_nE(i, j, k);
        });
      // Shift outside?
      amrex::Gpu::streamSynchronize();
    }
  }
}

void
PeleLM::computeBGcharge(
  const amrex::Real a_time,
  const std::unique_ptr<AdvanceDiffData>& diffData,
  const std::unique_ptr<AdvanceAdvData>& advData)
{
  // Get integration dt
  const amrex::Real dt_int = a_time - getTime(0, AmrOldTime);

  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get data pointers
    auto ldata_p = getLevelDataPtr(lev, AmrOldTime); // Old time species
    auto ldataR_p = getLevelDataReactPtr(lev);       // Reaction
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);   // NL data

    auto const& state_old_ma = ldata_p->state.const_arrays();
    auto const& adv_ma = advData->AofS[lev].const_arrays();
    auto const& dn_ma = diffData->Dn[lev].const_arrays();
    auto const& dnp1_ma = diffData->Dnp1[lev].const_arrays();
    auto const& dhat_ma = diffData->Dhat[lev].const_arrays();
    auto const& rhoYdot_ma = ldataR_p->I_R.const_arrays();

    ldataNLs_p->backgroundCharge.setVal(0.0);
    auto const& charge_ma = ldataNLs_p->backgroundCharge.arrays();
    constexpr amrex::Real factor = 1.0 / elemCharge;

    amrex::ParallelFor(
      ldataNLs_p->backgroundCharge, amrex::IntVect(0.0), NUM_SPECIES,
      [state_old_ma, adv_ma, dn_ma, dnp1_ma, dhat_ma, rhoYdot_ma, charge_ma,
       dt_int,
       zk =
         zk] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        amrex::Array4<amrex::Real const> rhoYold(
          state_old_ma[box_no], FIRSTSPEC);
        amrex::Array4<amrex::Real const> adv(adv_ma[box_no], FIRSTSPEC);
        amrex::Real rhoYprov =
          rhoYold(i, j, k, n) +
          dt_int *
            (adv(i, j, k, n) +
             0.5 * (dn_ma[box_no](i, j, k, n) - dnp1_ma[box_no](i, j, k, n)) +
             dhat_ma[box_no](i, j, k, n) + rhoYdot_ma[box_no](i, j, k, n));
        rhoYprov = amrex::max<amrex::Real>(rhoYprov, 0.0);
        const amrex::Real val = zk[n] * rhoYprov;
        amrex::Gpu::Atomic::Add(&charge_ma[box_no](i, j, k), val);
      });
    amrex::Gpu::streamSynchronize();
    ldataNLs_p->backgroundCharge.mult(factor);
  }
}

void
PeleLM::nonLinearResidual(
  const amrex::Real a_dt,
  const amrex::Vector<amrex::MultiFab*>& a_nlstate,
  const amrex::Vector<amrex::MultiFab*>& a_nlresid,
  const int updateScaling,
  const int updatePrecond)
{
  // Get unscaled copy of the NL state
  amrex::Vector<amrex::MultiFab> nE;
  nE.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> phiV;
  phiV.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    nE.emplace_back(
      grids[lev], dmap[lev], 1, m_nGrowState, amrex::MFInfo(), Factory(lev));
    phiV.emplace_back(
      grids[lev], dmap[lev], 1, m_nGrowState, amrex::MFInfo(), Factory(lev));
    amrex::MultiFab::Copy(nE[lev], *a_nlstate[lev], 0, 0, 1, m_nGrowState);
    nE[lev].mult(nE_scale, 0, 1, m_nGrowState);
    amrex::MultiFab::Copy(phiV[lev], *a_nlstate[lev], 1, 0, 1, m_nGrowState);
    phiV[lev].mult(phiV_scale, 0, 1, m_nGrowState);
  }

  // FillPatch nE state here
  fillPatchNLnE(m_cur_time, GetVecOfPtrs(nE), m_nGrowState);

  // Get L(phiV) and Grad(phiV)
  amrex::Vector<amrex::MultiFab> laplacian;
  laplacian.reserve(finest_level + 1);
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> gradPhiVCur(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    laplacian.emplace_back(
      grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      const auto& fba =
        amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim));
      gradPhiVCur[lev][idim].define(
        fba, dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
    }
  }
  constexpr int do_avgDown = 0; // TODO or should I ?
  auto bcRecPhiV = fetchBCRecArray(PHIV, 1);
  getDiffusionOp()->computeGradient(
    GetVecOfArrOfPtrs(gradPhiVCur), GetVecOfPtrs(laplacian),
    GetVecOfConstPtrs(phiV), {}, bcRecPhiV[0], do_avgDown);

  // Get nE diffusion term
  amrex::Vector<amrex::MultiFab> diffnE;
  diffnE.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    diffnE.emplace_back(
      grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
  }
  auto bcRecnE = fetchBCRecArray(NE, 1);
  getDiffusionOp()->computeDiffLap(
    GetVecOfPtrs(diffnE), 0, GetVecOfConstPtrs(nE), 0,
    GetVecOfConstPtrs(getnEDiffusivityVect(AmrNewTime)), 0, bcRecnE, 1);
  // WriteDebugPlotFile(GetVecOfConstPtrs(diffnE),"diffnE");
  // amrex::VisMF::Write(nE[0],"nEForDiffnlResid");
  // amrex::VisMF::Write(diffnE[0],"diffnEnlResid");

  // Get nE advection term
  amrex::Vector<amrex::MultiFab> advnE;
  advnE.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    advnE.emplace_back(
      grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
  }
  getAdvectionTerm(
    GetVecOfConstPtrs(nE), GetVecOfPtrs(advnE),
    GetVecOfArrOfConstPtrs(gradPhiVCur));
  // WriteDebugPlotFile(GetVecOfConstPtrs(advnE),"advnE");

  // Assemble non-linear residual
  // res(ne(:)) = dt * ( diff(:) + conv(:) + I_R(:) ) - ( ne(:) - ne_old(:) )
  // res(phiv(:)) = \Sum z_k * \tilde Y_k / q_e - ne + Lapl_PhiV
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto ldataOld_p = getLevelDataPtr(lev, AmrOldTime); // Old time
    auto ldataR_p = getLevelDataReactPtr(lev);          // Reaction

    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);

    // Init the ghostcells too
    a_nlresid[lev]->setVal(0.0);

    auto const& I_R_ma = ldataR_p->I_R.const_arrays();
    auto const& lapPhiV_ma = laplacian[lev].const_arrays();
    auto const& ne_diff_ma = diffnE[lev].const_arrays();
    auto const& ne_adv_ma = advnE[lev].const_arrays();
    auto const& ne_curr_ma = nE[lev].const_arrays();
    auto const& state_old_ma = ldataOld_p->state.const_arrays();
    auto const& charge_ma = ldataNLs_p->backgroundCharge.const_arrays();
    auto const& res_ma = a_nlresid[lev]->arrays();
    amrex::ParallelFor(
      ldataNLs_p->nlResid,
      [I_R_ma, lapPhiV_ma, ne_diff_ma, ne_adv_ma, ne_curr_ma, state_old_ma,
       charge_ma, res_ma,
       a_dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        amrex::Array4<amrex::Real> res_nE(res_ma[box_no], 0);
        amrex::Array4<amrex::Real> res_phiV(res_ma[box_no], 1);
        amrex::Array4<amrex::Real const> I_R_nE(I_R_ma[box_no], NUM_SPECIES);
        amrex::Array4<amrex::Real const> ne_old(state_old_ma[box_no], NE);
        res_nE(i, j, k) = ne_old(i, j, k) - ne_curr_ma[box_no](i, j, k) +
                          a_dt * (ne_diff_ma[box_no](i, j, k) +
                                  ne_adv_ma[box_no](i, j, k) + I_R_nE(i, j, k));
        constexpr amrex::Real scalLap = eps0 * epsr / elemCharge;
        res_phiV(i, j, k) = lapPhiV_ma[box_no](i, j, k) * scalLap -
                            ne_curr_ma[box_no](i, j, k) +
                            charge_ma[box_no](i, j, k);
        res_nE(i, j, k) *= -1.0;   // NLresidual is -RHS
        res_phiV(i, j, k) *= -1.0; // NLresidual is -RHS
      });
    // Shift outside?
    amrex::Gpu::streamSynchronize();
  }
  // WriteDebugPlotFile(GetVecOfConstPtrs(a_nlresid),"UnscalednlResid");

  //   /*
  // AverageDown the NLresiduals
  for (int lev = finest_level; lev > 0; --lev) {
#ifdef AMREX_USE_EB
    EB_average_down(
      *a_nlresid[lev], *a_nlresid[lev - 1], 0, 2, refRatio(lev - 1));
#else
    average_down(*a_nlresid[lev], *a_nlresid[lev - 1], 0, 2, refRatio(lev - 1));
#endif
  }
  //   */

  // Update the residual scale if necessary
  if (updateScaling) {
    getNLResidScaling(FnE_scale, FphiV_scale);
    FnE_scale = (FnE_scale > 1.0e-12) ? FnE_scale : 1.0;
    FphiV_scale = (FphiV_scale > 1.0e-12) ? FphiV_scale : 1.0;
    if (ef_verbose) {
      amrex::Print() << "    F(ne) scaling: " << FnE_scale << "\n";
      amrex::Print() << "    F(PhiV) scaling: " << FphiV_scale << "\n";
    }
  }

  // Apply scaling
  scaleNLResid(a_nlresid, FnE_scale, FphiV_scale);

  // Update the preconditioning LinOps
  if (updatePrecond && !m_ef_use_PETSC_direct) {
    setUpPrecond(a_dt, GetVecOfConstPtrs(nE));
  }
}

void
PeleLM::getAdvectionTerm(
  const amrex::Vector<const amrex::MultiFab*>& a_nE,
  const amrex::Vector<amrex::MultiFab*>& a_advTerm,
  const amrex::Vector<amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>>&
    a_gPhiVCur)
{
  // Get advection fluxes on all levels
  int nGrow = 0;
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> fluxes(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto& ba = grids[lev];
    const auto& factory = Factory(lev);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      fluxes[lev][idim].define(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dmap[lev],
        1, nGrow, amrex::MFInfo(), factory);
    }
  }

  // nE BCRec
  auto bcRecnE = fetchBCRecArray(NE, 1);

  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get t^{n} data pointer
    auto ldata_p = getLevelDataPtr(lev, AmrNewTime);

    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);

    // Get the face centered electron mobility
    constexpr int doZeroVisc = 0;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> mobE_ec =
      getDiffusivity(lev, 0, 1, doZeroVisc, bcRecnE, ldata_p->mobE_cc);

    // Get the electron effective velocity
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      auto const& ueff_ma = ldataNLs_p->uEffnE[idim].arrays();
      auto const& umac_ma = ldataNLs_p->umac[idim].const_arrays();
      auto const& gphi_c_ma = a_gPhiVCur[lev][idim]->const_arrays();
      auto const& gphi_o_ma = ldataNLs_p->gPhiVOld[idim].const_arrays();
      auto const& kappa_e_ma = mobE_ec[idim].const_arrays();
      amrex::ParallelFor(
        ldataNLs_p->uEffnE[idim],
        [ueff_ma, umac_ma, gphi_c_ma, gphi_o_ma, kappa_e_ma] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k) noexcept {
          ueff_ma[box_no](i, j, k) =
            umac_ma[box_no](i, j, k) -
            kappa_e_ma[box_no](i, j, k) * -1.0 * 0.5 *
              (gphi_c_ma[box_no](i, j, k) + gphi_o_ma[box_no](i, j, k));
        });
      // Shift outside?
      amrex::Gpu::streamSynchronize();
    }
  }

  // Average down the Ueff
  /*
  for (int lev = finest_level; lev > 0; --lev) {
     auto ldataNLFine_p = getLevelDataNLSolvePtr(lev);
     auto ldataNLCrse_p = getLevelDataNLSolvePtr(lev-1);
#ifdef AMREX_USE_EB
     EB_average_down_faces(GetArrOfConstPtrs(ldataNLFine_p->uEffnE),
                           GetArrOfPtrs(ldataNLCrse_p->uEffnE),
                           refRatio(lev-1),nGrow);
#else
     average_down_faces(GetArrOfConstPtrs(ldataNLFine_p->uEffnE),
                        GetArrOfPtrs(ldataNLCrse_p->uEffnE),
                        refRatio(lev-1),nGrow);
#endif
  }
  */

  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
    // Compute advective fluxes
    getAdvectionFluxes(
      lev, GetArrOfPtrs(fluxes[lev]), *a_nE[lev],
      GetArrOfConstPtrs(ldataNLs_p->uEffnE), bcRecnE[0]);
    // getAdvectionFluxesMOL(lev, GetArrOfPtrs(fluxes[lev]), *a_nE[lev],
    // GetArrOfConstPtrs(ldataNLs_p->uEffnE), bcRecnE[0]);
    // amrex::VisMF::Write(fluxes[lev][1],"FluxYNewbefAvgDown_lev"+std::to_string(lev));
  }

  // Average down the fluxes
  for (int lev = finest_level; lev > 0; --lev) {
#ifdef AMREX_USE_EB
    EB_average_down_faces(
      GetArrOfConstPtrs(fluxes[lev]), GetArrOfPtrs(fluxes[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
#else
    average_down_faces(
      GetArrOfConstPtrs(fluxes[lev]), GetArrOfPtrs(fluxes[lev - 1]),
      refRatio(lev - 1), geom[lev - 1]);
#endif
  }

  // Compute divergence
  constexpr int intensiveFluxes = 1;
  fluxDivergence(
    a_advTerm, 0, GetVecOfArrOfPtrs(fluxes), 0, 1, intensiveFluxes, -1.0);
}

void
PeleLM::getAdvectionFluxesMOL(
  const int lev,
  const amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>& a_fluxes,
  const amrex::MultiFab& a_nE,
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_ueff,
  amrex::BCRec bcrec)
{
  auto bcRec_d = convertToDeviceVector(amrex::Vector<amrex::BCRec>(1, bcrec));
  auto AdvType = fetchAdvTypeArray(FIRSTSPEC, 1);
  auto AdvType_d = convertToDeviceVector(AdvType);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  {
    amrex::FArrayBox edgstate[AMREX_SPACEDIM];
    for (amrex::MFIter mfi(a_nE, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {

      amrex::Box const& bx = mfi.tilebox();
      AMREX_D_TERM(
        const amrex::Box& xbx = mfi.grownnodaltilebox(0, 0);
        , const amrex::Box& ybx = mfi.grownnodaltilebox(1, 0);
        , const amrex::Box& zbx = mfi.grownnodaltilebox(2, 0));

      AMREX_D_TERM(
        auto const& fx = a_fluxes[0]->array(mfi, 0);
        , auto const& fy = a_fluxes[1]->array(mfi, 0);
        , auto const& fz = a_fluxes[2]->array(mfi, 0);)
      AMREX_D_TERM(
        auto const& ueff = a_ueff[0]->const_array(mfi);
        , auto const& veff = a_ueff[1]->const_array(mfi);
        , auto const& weff = a_ueff[2]->const_array(mfi);)
      AMREX_D_TERM(
        edgstate[0].resize(xbx, 1);, edgstate[1].resize(ybx, 1);
        , edgstate[2].resize(zbx, 1));
      AMREX_D_TERM(
        amrex::Array4<amrex::Real> xstate = edgstate[0].array();
        , amrex::Array4<amrex::Real> ystate = edgstate[1].array();
        , amrex::Array4<amrex::Real> zstate = edgstate[2].array());
      AMREX_D_TERM(
        amrex::Elixir eli_edgex = edgstate[0].elixir();
        , amrex::Elixir eli_edgey = edgstate[1].elixir();
        , amrex::Elixir eli_edgez = edgstate[2].elixir());

      auto const& nE_arr = a_nE.const_array(mfi);
      auto const& divu_arr = a_nE.const_array(mfi);
      auto const& force_arr = a_nE.const_array(mfi);

      constexpr bool is_velocity = false;
      constexpr bool fluxes_are_area_weighted = false;
      constexpr bool knownEdgeState = false;
      std::string mol = "MOL";
      HydroUtils::ComputeFluxesOnBoxFromState(
        bx, 1, mfi, nE_arr, AMREX_D_DECL(fx, fy, fz),
        AMREX_D_DECL(xstate, ystate, zstate), knownEdgeState,
        AMREX_D_DECL(ueff, veff, weff), divu_arr, force_arr, geom[lev], m_dt,
        {bcrec}, bcRec_d.dataPtr(), AdvType_d.dataPtr(),
#ifdef AMREX_USE_EB
        EBFactory(lev),
#endif
        0, 0, is_velocity, fluxes_are_area_weighted, mol);
    }
  }
}

void
PeleLM::getAdvectionFluxes(
  const int lev,
  const amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM>& a_fluxes,
  const amrex::MultiFab& a_nE,
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_ueff,
  amrex::BCRec bcrec)
{
  const amrex::Box& domain = geom[lev].Domain();

  const int order = m_nEAdvOrder;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  {
    amrex::FArrayBox edgstate[AMREX_SPACEDIM];
    for (amrex::MFIter mfi(a_nE, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      amrex::Box const& bx = mfi.tilebox();
      AMREX_D_TERM(
        const amrex::Box& xbx = surroundingNodes(bx, 0);
        , const amrex::Box& ybx = surroundingNodes(bx, 1);
        , const amrex::Box& zbx = surroundingNodes(bx, 2));

      // data arrays
      auto const& ne_arr = a_nE.const_array(mfi);
      AMREX_D_TERM(
        amrex::Array4<amrex::Real> xflux = a_fluxes[0]->array(mfi);
        , amrex::Array4<amrex::Real> yflux = a_fluxes[1]->array(mfi);
        , amrex::Array4<amrex::Real> zflux = a_fluxes[2]->array(mfi));
      AMREX_D_TERM(
        amrex::Array4<amrex::Real const> u = a_ueff[0]->const_array(mfi);
        , amrex::Array4<amrex::Real const> v = a_ueff[1]->const_array(mfi);
        , amrex::Array4<amrex::Real const> w = a_ueff[2]->const_array(mfi););
      AMREX_D_TERM(
        edgstate[0].resize(xbx, 1);, edgstate[1].resize(ybx, 1);
        , edgstate[2].resize(zbx, 1));
      AMREX_D_TERM(
        amrex::Array4<amrex::Real> xstate = edgstate[0].array();
        , amrex::Array4<amrex::Real> ystate = edgstate[1].array();
        , amrex::Array4<amrex::Real> zstate = edgstate[2].array());
      AMREX_D_TERM(
        amrex::Elixir xstate_eli = edgstate[0].elixir();
        , amrex::Elixir ystate_eli = edgstate[1].elixir();
        , amrex::Elixir zstate_eli = edgstate[2].elixir());

      // Predict edge states
      // X
      {
        // BCs
        const amrex::Box& edomain = surroundingNodes(domain, 0);
        const auto bc_lo = bcrec.lo(0);
        const auto bc_hi = bcrec.hi(0);

        amrex::ParallelFor(
          xbx, [bc_lo, bc_hi, edomain, ne_arr, xstate, u, order, domain,
                xbx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const int idx[3] = {i, j, k};
            const bool on_lo =
              ((bc_lo == amrex::BCType::ext_dir) &&
               (idx[0] <= edomain.smallEnd(0)));
            const bool on_hi =
              ((bc_hi == amrex::BCType::ext_dir) &&
               (idx[0] >= edomain.bigEnd(0)));
            if (order == 1) {
              xstate(i, j, k) =
                ef_edge_state_extdir(i, j, k, 0, on_lo, on_hi, ne_arr, u);
            } else if (order == 2) {
              const bool extdir_or_ho_lo = (bc_lo == amrex::BCType::ext_dir) ||
                                           (bc_lo == amrex::BCType::hoextrap);
              const bool extdir_or_ho_hi = (bc_hi == amrex::BCType::ext_dir) ||
                                           (bc_hi == amrex::BCType::hoextrap);
              xstate(i, j, k) = ef_edge_state_2ndO_extdir(
                i, j, k, 0, on_lo, on_hi, extdir_or_ho_lo, extdir_or_ho_hi,
                domain.smallEnd(0), domain.bigEnd(0), ne_arr, u, xbx);
            }
          });
      }
#if (AMREX_SPACEDIM > 1)
      // Y
      {
        // BCs
        const amrex::Box& edomain = surroundingNodes(domain, 1);
        const auto bc_lo = bcrec.lo(1);
        const auto bc_hi = bcrec.hi(1);

        amrex::ParallelFor(
          ybx, [bc_lo, bc_hi, edomain, domain, order, ystate, ne_arr, v,
                ybx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const int idx[3] = {i, j, k};
            const bool on_lo =
              ((bc_lo == amrex::BCType::ext_dir) &&
               (idx[1] <= edomain.smallEnd(1)));
            const bool on_hi =
              ((bc_hi == amrex::BCType::ext_dir) &&
               (idx[1] >= edomain.bigEnd(1)));
            if (order == 1) {
              ystate(i, j, k) =
                ef_edge_state_extdir(i, j, k, 1, on_lo, on_hi, ne_arr, v);
            } else if (order == 2) {
              const bool extdir_or_ho_lo = (bc_lo == amrex::BCType::ext_dir) ||
                                           (bc_lo == amrex::BCType::hoextrap);
              const bool extdir_or_ho_hi = (bc_hi == amrex::BCType::ext_dir) ||
                                           (bc_hi == amrex::BCType::hoextrap);
              ystate(i, j, k) = ef_edge_state_2ndO_extdir(
                i, j, k, 1, on_lo, on_hi, extdir_or_ho_lo, extdir_or_ho_hi,
                domain.smallEnd(1), domain.bigEnd(1), ne_arr, v, ybx);
            }
          });
      }
#if (AMREX_SPACEDIM == 3)
      // Z
      {
        // BCs
        const amrex::Box& edomain = surroundingNodes(domain, 2);
        const auto bc_lo = bcrec.lo(2);
        const auto bc_hi = bcrec.hi(2);

        amrex::ParallelFor(
          zbx, [bc_lo, bc_hi, edomain, zstate, order, ne_arr, w, zbx,
                domain] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const int idx[3] = {i, j, k};
            const bool on_lo =
              ((bc_lo == amrex::BCType::ext_dir) &&
               (idx[2] <= edomain.smallEnd(2)));
            const bool on_hi =
              ((bc_hi == amrex::BCType::ext_dir) &&
               (idx[2] >= edomain.bigEnd(2)));
            if (order == 1) {
              zstate(i, j, k) =
                ef_edge_state_extdir(i, j, k, 2, on_lo, on_hi, ne_arr, w);
            } else if (order == 2) {
              const bool extdir_or_ho_lo = (bc_lo == amrex::BCType::ext_dir) ||
                                           (bc_lo == amrex::BCType::hoextrap);
              const bool extdir_or_ho_hi = (bc_hi == amrex::BCType::ext_dir) ||
                                           (bc_hi == amrex::BCType::hoextrap);
              zstate(i, j, k) = ef_edge_state_2ndO_extdir(
                i, j, k, 2, on_lo, on_hi, extdir_or_ho_lo, extdir_or_ho_hi,
                domain.smallEnd(2), domain.bigEnd(2), ne_arr, w, zbx);
            }
          });
      }
#endif
#endif

      // Computing fluxes
      amrex::ParallelFor(
        xbx, [xflux, u, xstate] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          xflux(i, j, k) = u(i, j, k) * xstate(i, j, k);
        });
#if (AMREX_SPACEDIM > 1)
      amrex::ParallelFor(
        ybx, [yflux, v, ystate] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          yflux(i, j, k) = v(i, j, k) * ystate(i, j, k);
        });
#if (AMREX_SPACEDIM == 3)
      amrex::ParallelFor(
        zbx, [zflux, w, zstate] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          zflux(i, j, k) = w(i, j, k) * zstate(i, j, k);
        });
#endif
#endif
    }
  }
}

void
PeleLM::setUpPrecond(
  const amrex::Real a_dt, const amrex::Vector<const amrex::MultiFab*>& a_nE)
{
  BL_PROFILE("PeleLMeX::setUpPrecond()");

  // Update LinearOps defs if needed -> done internally by the getPrecondOp()
  // func

  // nE BCRec
  auto bcRecnE = fetchBCRecArray(NE, 1);
  auto bcRecPhiV = fetchBCRecArray(PHIV, 1);

  //--------------------------------------------------------------------------
  // Set diff/drift operator
  getPrecondOp()->setDiffOpScalars(
    -nE_scale / FnE_scale, -a_dt * nE_scale / FnE_scale,
    a_dt * nE_scale / FnE_scale);
  amrex::Real omega = m_ABecCecOmega;
  getPrecondOp()->setDiffOpRelaxation(omega);
  getPrecondOp()->setDiffOpBCs(bcRecnE[0]);

  // Get and set coefficients on each level
  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get level data pointer
    auto ldata_p = getLevelDataPtr(lev, AmrNewTime);

    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);

    constexpr int doZeroVisc = 0;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> diffE_ec =
      getDiffusivity(lev, 0, 1, doZeroVisc, bcRecnE, ldata_p->diffE_cc);

    getPrecondOp()->setDiffOpACoeff(lev, 1.0);
    getPrecondOp()->setDiffOpBCoeff(lev, GetArrOfConstPtrs(diffE_ec));
    getPrecondOp()->setDiffOpCCoeff(lev, GetArrOfConstPtrs(ldataNLs_p->uEffnE));
  }

  // Stilda approx first-order
  amrex::Vector<amrex::MultiFab> diagDiffOp;
  diagDiffOp.reserve(finest_level + 1);
  if (m_ef_PC_approx == 2) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      diagDiffOp.emplace_back(grids[lev], dmap[lev], 1, 1);
      getPrecondOp()->getDiffOpDiagonal(lev, diagDiffOp[lev]);
      diagDiffOp[lev].mult(FnE_scale / nE_scale);
    }
    fillPatchExtrap(m_cur_time, GetVecOfPtrs(diagDiffOp), 1);
  }

  //--------------------------------------------------------------------------
  // Set Stilda and drift operators

  // Set scalars
  getPrecondOp()->setDriftOpScalars(0.0, 0.5 * phiV_scale / FnE_scale * a_dt);
  if (m_ef_PC_approx == 1 || m_ef_PC_approx == 2) {
    getPrecondOp()->setStildaOpScalars(0.0, -1.0);
  } else if (m_ef_PC_approx == 3 || m_ef_PC_approx == 4) {
    getPrecondOp()->setStildaOpScalars(0.0, 1.0);
  }

  // Set BCs
  getPrecondOp()->setDriftOpBCs(bcRecPhiV[0]);
  getPrecondOp()->setStildaOpBCs(bcRecPhiV[0]);

  // Get and set coefficients on each level
  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get nl solve data pointer
    auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
    auto ldata_p = getLevelDataPtr(lev, AmrNewTime);

    // CC neKe values
    amrex::MultiFab nEKe(grids[lev], dmap[lev], 1, 1);

    // CC Schur neKe values
    amrex::MultiFab Schur_nEKe;
    if (m_ef_PC_approx == 2) {
      Schur_nEKe.define(grids[lev], dmap[lev], 1, 1);
    }

    auto const& neke_ma = nEKe.arrays();
    auto const& kappaE_ma = ldata_p->mobE_cc.arrays();
    auto const& ne_arr_ma = a_nE[lev]->const_arrays();
    auto const& Schur_ma =
      (m_ef_PC_approx == 2) ? Schur_nEKe.arrays() : nEKe.arrays();
    auto const& diffOp_diag_ma =
      (m_ef_PC_approx == 2) ? diagDiffOp[lev].arrays() : nEKe.arrays();
    const int do_Schur = (m_ef_PC_approx == 2) ? 1 : 0;

    amrex::ParallelFor(
      nEKe, nEKe.nGrowVect(),
      [neke_ma, kappaE_ma,
       ne_arr_ma] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        neke_ma[box_no](i, j, k) =
          kappaE_ma[box_no](i, j, k) * ne_arr_ma[box_no](i, j, k);
      });
    amrex::Gpu::streamSynchronize();
    if (do_Schur == 1) {
      amrex::ParallelFor(
        nEKe, nEKe.nGrowVect(),
        [Schur_ma, a_dt, neke_ma, diffOp_diag_ma] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k) noexcept {
          Schur_ma[box_no](i, j, k) = -a_dt * 0.5 * neke_ma[box_no](i, j, k) /
                                      diffOp_diag_ma[box_no](i, j, k);
        });
      amrex::Gpu::streamSynchronize();
    }
    // Upwinded edge neKe values
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> neKe_ec = getUpwindedEdge(
      lev, 0, 1, bcRecnE, nEKe, GetArrOfConstPtrs(ldataNLs_p->uEffnE));

    // Set drift Op coefficients
    getPrecondOp()->setDriftOpBCoeff(lev, GetArrOfConstPtrs(neKe_ec));

    // Set Stilda Op coefficients
    if (m_ef_PC_approx == 1) { // Assuming identity of the inverse of DiffOp
      // Add Stilda pieces
      amrex::Real scalLap = eps0 * epsr / elemCharge;
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        neKe_ec[idim].mult(0.5 * a_dt, 0, 1);
        neKe_ec[idim].plus(scalLap, 0, 1);
      }
      getPrecondOp()->setStildaOpBCoeff(lev, GetArrOfConstPtrs(neKe_ec));
    } else if (m_ef_PC_approx == 2) { // Assuming inverse of the diag of DiffOp
      // Upwinded Schur edge neKe values
      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> Schur_neKe_ec =
        getUpwindedEdge(
          lev, 0, 1, bcRecnE, Schur_nEKe,
          GetArrOfConstPtrs(ldataNLs_p->uEffnE));
      amrex::Real scalLap = eps0 * epsr / elemCharge;
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        Schur_neKe_ec[idim].plus(scalLap, 0, 1);
      }
      getPrecondOp()->setStildaOpBCoeff(lev, GetArrOfConstPtrs(Schur_neKe_ec));
    } else {
      amrex::Abort("Preconditioner option /= 1 or 2 not available yet");
    }
  }
}

amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>
PeleLM::getUpwindedEdge(
  const int lev,
  const int edge_comp,
  const int ncomp,
  const amrex::Vector<amrex::BCRec>& bcrec,
  const amrex::MultiFab& ccMF,
  const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& ecVel)
{
  AMREX_ASSERT(bcrec.size() >= ncomp);
  AMREX_ASSERT(ccMF.nComp() >= edge_comp + ncomp);
  const auto& ba = ccMF.boxArray();
  const auto& dm = ccMF.DistributionMap();
  const auto& factory = ccMF.Factory();
  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> ecMFs{AMREX_D_DECL(
    amrex::MultiFab(
      amrex::convert(ba, amrex::IntVect::TheDimensionVector(0)), dm, ncomp, 0,
      amrex::MFInfo(), factory),
    amrex::MultiFab(
      amrex::convert(ba, amrex::IntVect::TheDimensionVector(1)), dm, ncomp, 0,
      amrex::MFInfo(), factory),
    amrex::MultiFab(
      amrex::convert(ba, amrex::IntVect::TheDimensionVector(2)), dm, ncomp, 0,
      amrex::MFInfo(), factory))};

#ifdef AMREX_USE_EB
  // TODO cen2edg_upwind EB
#else
  const amrex::Box& domain = geom[lev].Domain();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(ccMF, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      const amrex::Box ebx = mfi.nodaltilebox(idim);
      const amrex::Box& edomain = amrex::surroundingNodes(domain, idim);
      const auto& ccVal = ccMF.const_array(mfi, edge_comp);
      const auto& ecUeff = ecVel[idim]->const_array(mfi);
      const auto& ecVal = ecMFs[idim].array(mfi);
      const auto bc_lo = bcrec[0].lo(idim);
      const auto bc_hi = bcrec[0].hi(idim);
      amrex::ParallelFor(
        ebx, [bc_lo, bc_hi, idim, edomain, ecUeff, ccVal,
              ecVal] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          const int idx[3] = {i, j, k};
          const bool on_lo =
            ((bc_lo == amrex::BCType::ext_dir) &&
             (idx[idim] <= edomain.smallEnd(idim)));
          const bool on_hi =
            ((bc_hi == amrex::BCType::ext_dir) &&
             (idx[idim] >= edomain.bigEnd(idim)));
          cen2edg_upwind(i, j, k, idim, 1, on_lo, on_hi, ecUeff, ccVal, ecVal);
        });
    }
  }
#endif

  return ecMFs;
}

void
PeleLM::jTimesV(
  const amrex::Vector<amrex::MultiFab*>& a_v,
  const amrex::Vector<amrex::MultiFab*>& a_Jv)
{
  amrex::Real vNorm;
  nlSolveNorm(a_v, vNorm);

  // v is zero, Jv is zero and return
  if (vNorm == 0.0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      a_Jv[lev]->setVal(0.0);
    }
    return;
  }

  // TODO: only one-sided difference for now
  amrex::Real delta_pert =
    m_ef_lambda_jfnk * (m_ef_lambda_jfnk + nl_stateNorm / vNorm);

  if (m_ef_diffT_jfnk == 1) {
    amrex::Vector<amrex::MultiFab> statePert;
    statePert.reserve(finest_level + 1);
    amrex::Vector<amrex::MultiFab> residPert;
    residPert.reserve(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
      statePert.emplace_back(
        grids[lev], dmap[lev], 2, m_nGrowState, amrex::MFInfo(), Factory(lev));
      residPert.emplace_back(
        grids[lev], dmap[lev], 2, m_nGrowState, amrex::MFInfo(), Factory(lev));
      amrex::MultiFab::Copy(
        statePert[lev], ldataNLs_p->nlState, 0, 0, 2, m_nGrowState);
      amrex::MultiFab::Saxpy(statePert[lev], delta_pert, *a_v[lev], 0, 0, 2, 0);
    }

    constexpr int update_scaling = 0;
    constexpr int update_precond = 0;
    nonLinearResidual(
      dtsub, GetVecOfPtrs(statePert), GetVecOfPtrs(residPert), update_scaling,
      update_precond);

    for (int lev = 0; lev <= finest_level; ++lev) {
      auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
      amrex::MultiFab::LinComb(
        *a_Jv[lev], 1.0, residPert[lev], 0, -1.0, ldataNLs_p->nlResid, 0, 0, 2,
        0);
      a_Jv[lev]->mult(-1.0 / delta_pert);
    }
  } else if (m_ef_diffT_jfnk == 2) {
    amrex::Vector<amrex::MultiFab> statePertPls;
    statePertPls.reserve(finest_level + 1);
    amrex::Vector<amrex::MultiFab> residPertPls;
    residPertPls.reserve(finest_level + 1);
    amrex::Vector<amrex::MultiFab> statePertMns;
    statePertMns.reserve(finest_level + 1);
    amrex::Vector<amrex::MultiFab> residPertMns;
    residPertMns.reserve(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto ldataNLs_p = getLevelDataNLSolvePtr(lev);
      statePertPls.emplace_back(
        grids[lev], dmap[lev], 2, m_nGrowState, amrex::MFInfo(), Factory(lev));
      residPertPls.emplace_back(
        grids[lev], dmap[lev], 2, m_nGrowState, amrex::MFInfo(), Factory(lev));
      statePertMns.emplace_back(
        grids[lev], dmap[lev], 2, m_nGrowState, amrex::MFInfo(), Factory(lev));
      residPertMns.emplace_back(
        grids[lev], dmap[lev], 2, m_nGrowState, amrex::MFInfo(), Factory(lev));
      amrex::MultiFab::Copy(
        statePertPls[lev], ldataNLs_p->nlState, 0, 0, 2, m_nGrowState);
      amrex::MultiFab::Copy(
        statePertMns[lev], ldataNLs_p->nlState, 0, 0, 2, m_nGrowState);
      amrex::MultiFab::Saxpy(
        statePertPls[lev], delta_pert, *a_v[lev], 0, 0, 2, 0);
      amrex::MultiFab::Saxpy(
        statePertMns[lev], -delta_pert, *a_v[lev], 0, 0, 2, 0);
    }

    constexpr int update_scaling = 0;
    constexpr int update_precond = 0;
    nonLinearResidual(
      dtsub, GetVecOfPtrs(statePertPls), GetVecOfPtrs(residPertPls),
      update_scaling, update_precond);
    nonLinearResidual(
      dtsub, GetVecOfPtrs(statePertMns), GetVecOfPtrs(residPertMns),
      update_scaling, update_precond);

    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab::LinComb(
        *a_Jv[lev], 1.0, residPertPls[lev], 0, -1.0, residPertMns[lev], 0, 0, 2,
        0);
      a_Jv[lev]->mult(-0.5 / delta_pert);
    }
  }
}

void
PeleLM::applyPrecond(
  const amrex::Vector<amrex::MultiFab*>& a_v,
  const amrex::Vector<amrex::MultiFab*>& a_Pv)
{
  BL_PROFILE("PeleLMeX::applyPrecond()");

  // Setup aliases and temps
  amrex::Vector<amrex::MultiFab> nE_al;
  nE_al.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> phiV_al;
  phiV_al.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> PnE_al;
  PnE_al.reserve(finest_level + 1);
  amrex::Vector<amrex::MultiFab> PphiV_al;
  PphiV_al.reserve(finest_level + 1);
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> temp(finest_level + 1);
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> temp2(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    // Aliases
    nE_al.emplace_back(*a_v[lev], amrex::make_alias, 0, 1);
    phiV_al.emplace_back(*a_v[lev], amrex::make_alias, 1, 1);
    PnE_al.emplace_back(*a_Pv[lev], amrex::make_alias, 0, 1);
    PphiV_al.emplace_back(*a_Pv[lev], amrex::make_alias, 1, 1);
    PnE_al[lev].setVal(0.0, 0, 1, 1);
    PphiV_al[lev].setVal(0.0, 0, 1, 1);

    // Temporary data holder
    temp[lev].reset(new amrex::MultiFab(
      grids[lev], dmap[lev], 1, nE_al[lev].nGrow(), amrex::MFInfo(),
      Factory(lev)));
    temp[lev]->setVal(0.0);
    temp2[lev].reset(new amrex::MultiFab(
      grids[lev], dmap[lev], 1, nE_al[lev].nGrow(), amrex::MFInfo(),
      Factory(lev)));
    temp2[lev]->setVal(0.0);
  }

  // Set LinOps Level BCs
  for (int lev = 0; lev <= finest_level; ++lev) {
    getPrecondOp()->setDiffOpLevelBC(lev, &PnE_al[lev]);
    getPrecondOp()->setDriftOpLevelBC(lev, &PphiV_al[lev]);
    getPrecondOp()->setStildaOpLevelBC(lev, &PphiV_al[lev]);
  }

  // Most inner matrix
  amrex::Real S_tol = m_ef_PC_MG_Tol;
  amrex::Real S_tol_abs = MLNorm0(GetVecOfConstPtrs(nE_al)) * m_ef_PC_MG_Tol;
  getPrecondOp()->diffOpSolve(
    GetVecOfPtrs(PnE_al), GetVecOfConstPtrs(nE_al), S_tol, S_tol_abs);
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Copy(PphiV_al[lev], phiV_al[lev], 0, 0, 1, 0);
  }

  // Pivot second matrix
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Saxpy(
      PphiV_al[lev], nE_scale / FphiV_scale, PnE_al[lev], 0, 0, 1, 0);
  }

  // PhiV estimate matrix
  // First scale PphiV
  for (int lev = 0; lev <= finest_level; ++lev) {
    PphiV_al[lev].mult(FphiV_scale / phiV_scale);
  }
  S_tol_abs = MLNorm0(GetVecOfConstPtrs(PphiV_al)) * m_ef_PC_MG_Tol;
  getPrecondOp()->StildaOpSolve(
    GetVecOfPtrs(temp), GetVecOfConstPtrs(PphiV_al), S_tol, S_tol_abs);
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Copy(PphiV_al[lev], *temp[lev], 0, 0, 1, 0);
  }

  // Final matrix
  getPrecondOp()->driftOpApply(GetVecOfPtrs(temp), GetVecOfPtrs(PphiV_al));
  S_tol_abs = MLNorm0(GetVecOfConstPtrs(temp)) * m_ef_PC_MG_Tol;
  getPrecondOp()->diffOpSolve(
    GetVecOfPtrs(temp2), GetVecOfConstPtrs(temp), S_tol, S_tol_abs);
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Saxpy(PnE_al[lev], -1.0, *temp2[lev], 0, 0, 1, 0);
  }
}

void
PeleLM::nlSolveNorm(const amrex::Vector<amrex::MultiFab*>& a_MF, amrex::Real& r)
{
  r = 0.0;
  const int nComp = a_MF[0]->nComp();
  for (int comp = 0; comp < nComp; ++comp) {
    amrex::Real norm = 0.0;
    for (int lev = 0; lev < a_MF.size(); ++lev) {
      // TODO : norm not weighted by cell size, should it ?
      if (lev != a_MF.size() - 1) {
        norm += amrex::MultiFab::Dot(
          *m_coveredMask[lev], *a_MF[lev], comp, *a_MF[lev], comp, 1, 0);
      } else {
        norm += amrex::MultiFab::Dot(*a_MF[lev], comp, *a_MF[lev], comp, 1, 0);
      }
    }
    r += norm;
  }
  r = std::sqrt(r);
}
