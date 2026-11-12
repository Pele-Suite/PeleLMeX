#include <PeleLMeX.H>
#include <AMReX_PlotFileUtil.H>
#include <memory>

void
PeleLM::Evaluate()
{
  BL_PROFILE("PeleLMeX::Evaluate()");

  //----------------------------------------------------------------
  // Check that requested evaluate entries exist and determine the size
  // of the container and entries names
  int ncomp = 0;
  amrex::Vector<std::string> plt_VarsName;
  for (int ivar = 0; ivar < m_evaluatePlotVarCount; ++ivar) {
    bool itexists = derive_lst.canDerive(m_evaluatePlotVars[ivar]) ||
                    evaluate_lst.canDerive(m_evaluatePlotVars[ivar]) ||
                    isStateVariable(m_evaluatePlotVars[ivar]);
    if (!itexists) {
      amrex::Error(
        "PeleLM::evaluate(): unknown variable: " + m_evaluatePlotVars[ivar]);
    }
    if (derive_lst.canDerive(m_evaluatePlotVars[ivar])) {
      const PeleLMDeriveRec* rec = derive_lst.get(m_evaluatePlotVars[ivar]);
      ncomp += rec->numDerive();
      for (int dvar = 0; dvar < rec->numDerive(); ++dvar) {
        plt_VarsName.push_back(rec->variableName(dvar));
      }
    } else if (evaluate_lst.canDerive(m_evaluatePlotVars[ivar])) {
      const PeleLMDeriveRec* rec = evaluate_lst.get(m_evaluatePlotVars[ivar]);
      ncomp += rec->numDerive();
      for (int dvar = 0; dvar < rec->numDerive(); ++dvar) {
        plt_VarsName.push_back(rec->variableName(dvar));
      }
    } else if (isStateVariable(m_evaluatePlotVars[ivar])) {
      ncomp += 1;
      plt_VarsName.push_back(m_evaluatePlotVars[ivar]);
    }
  }

  //----------------------------------------------------------------
  // Define the outgoing container
  amrex::Vector<amrex::MultiFab> mf_plt;
  mf_plt.reserve(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    mf_plt.emplace_back(
      grids[lev], dmap[lev], ncomp, 0, amrex::MFInfo(), Factory(lev));
  }

  //----------------------------------------------------------------
  // Prepare a few things if not restarting from a chkfile
  if (m_restart_chkfile.empty()) {
    m_nstep = 0;
  }

  //----------------------------------------------------------------
  // Fill the outgoing container
  int cnt = 0;
  for (int ivar = 0; ivar < m_evaluatePlotVarCount; ++ivar) {
    int cntIncr = 0;

    amrex::Print() << " --> Evaluating " << m_evaluatePlotVars[ivar] << "\n";

    // Evaluate function calls actual PeleLM::Evolve pieces and may require
    // the entire multi-level hierarchy
    if (evaluate_lst.canDerive(m_evaluatePlotVars[ivar])) {
      MLevaluate(GetVecOfPtrs(mf_plt), cnt, cntIncr, m_evaluatePlotVars[ivar]);

      // Regular derived functions and State entries are called on a per level
      // basis derive function can handle both derived and state entries
    } else if (
      derive_lst.canDerive(m_evaluatePlotVars[ivar]) ||
      isStateVariable(m_evaluatePlotVars[ivar])) {
      for (int lev = 0; lev <= finest_level; ++lev) {
        std::unique_ptr<amrex::MultiFab> mf;
        mf = derive(m_evaluatePlotVars[ivar], m_cur_time, lev, 0);
        amrex::MultiFab::Copy(mf_plt[lev], *mf, 0, cnt, mf->nComp(), 0);
        cntIncr = mf->nComp();
      }
    }
    cnt += cntIncr;
  }

  //----------------------------------------------------------------
  // Write the evaluated variables to disc
  amrex::Vector<int> istep(finest_level + 1, 0);

  // Override m_cur_time to store the dt in pltEvaluate
  m_cur_time = m_dt;

  std::string plotfilename = "pltEvaluate";
  amrex::WriteMultiLevelPlotfile(
    plotfilename, finest_level + 1, GetVecOfConstPtrs(mf_plt), plt_VarsName,
    Geom(), m_cur_time, istep, refRatio());
}

void
PeleLM::MLevaluate(
  const amrex::Vector<amrex::MultiFab*>& a_MFVec,
  const int a_comp,
  int& nComp,
  const std::string& a_var)
{

  // This function manually maps the evaluate variables to the function calls
  // used in PeleLM:::Evolve

  if (a_var == "divU") {
    constexpr int is_initialization = 0;    // No, use IRR
    constexpr int computeDiffusionTerm = 1; // Needed here
    constexpr int do_avgDown = 1;           // Always

    // Light version of the diffusion data container
    std::unique_ptr<AdvanceDiffData> diffData;
    diffData = std::make_unique<AdvanceDiffData>(
      finest_level, grids, dmap, m_factory, m_nGrowAdv, m_use_wbar, m_use_soret,
      m_nAux, is_initialization);
    calcDivU(
      is_initialization, computeDiffusionTerm, do_avgDown, AmrNewTime,
      diffData);
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
      amrex::MultiFab::Copy(*a_MFVec[lev], ldata_p->divu, 0, a_comp, 1, 0);
    }
    nComp = 1;
  } else if (a_var == "velProj") {
    // Will need DivU
    constexpr int is_initialization = 0;    // No, use IRR
    constexpr int computeDiffusionTerm = 1; // Needed here
    constexpr int do_avgDown = 1;           // Always

    // Light version of the diffusion data container
    std::unique_ptr<AdvanceDiffData> diffData;
    diffData = std::make_unique<AdvanceDiffData>(
      finest_level, grids, dmap, m_factory, m_nGrowAdv, m_use_wbar, m_use_soret,
      m_nAux, is_initialization);
    calcDivU(
      is_initialization, computeDiffusionTerm, do_avgDown, AmrNewTime,
      diffData);

    // Do initProj
    initialProjection();

    // Copy into outgoing data holder
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
      amrex::MultiFab::Copy(
        *a_MFVec[lev], ldata_p->state, VELX, a_comp, AMREX_SPACEDIM, 0);
    }
    nComp = AMREX_SPACEDIM;
  } else if (a_var == "divTau") {
    // Velocity tensor components
    constexpr int use_density = 0;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> aliasDivTau(
      finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      aliasDivTau[lev] = std::make_unique<amrex::MultiFab>(
        *a_MFVec[lev], amrex::make_alias, a_comp, AMREX_SPACEDIM);
    }
    computeDivTau(AmrNewTime, GetVecOfPtrs(aliasDivTau), use_density);
#ifdef AMREX_USE_EB
    for (int lev = 0; lev <= finest_level; ++lev) {
      EB_set_covered(*aliasDivTau[lev], 0.0);
    }
#endif
    nComp = AMREX_SPACEDIM;
  } else if (a_var == "diffTerm") {
    // Use the diffusion data holder, get diffusivity and calc D
    // Finally, copy into a_MFVec
    std::unique_ptr<AdvanceDiffData> diffData;
    diffData = std::make_unique<AdvanceDiffData>(
      finest_level, grids, dmap, m_factory, m_nGrowAdv, m_use_wbar, m_use_soret,
      m_nAux);
    calcDiffusivity(AmrNewTime);
    // If doing LES, need to be able to add in turbulent component
    if (m_do_les) {
      calcTurbViscosity(AmrNewTime);
    }
    computeDifferentialDiffusionTerms(AmrNewTime, diffData);
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab::Copy(
        *a_MFVec[lev], diffData->Dnp1[lev], 0, a_comp, NUM_SPECIES + 2, 0);
    }
    nComp = NUM_SPECIES + 2;
  } else if (a_var == "advTerm") {
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> aliasMF(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      aliasMF[lev] = std::make_unique<amrex::MultiFab>(
        *a_MFVec[lev], amrex::make_alias, a_comp, NVAR - 2);
    }
    evaluateAdvectionTerms(GetVecOfPtrs(aliasMF));
    nComp = NVAR - 2;
  } else if (a_var == "chemTest") {
    // 'Old' chemical state (rhoYs + rhoH + T) and adv/diff forcing for chem.
    // integration Replicate most of the advance function Copy the state
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
      amrex::MultiFab::Copy(
        *a_MFVec[lev], ldata_p->state, FIRSTSPEC, a_comp, NUM_SPECIES + 2, 0);
    }
    // Initial velocity projection
    if (m_restart_chkfile.empty()) {
      projectInitSolution();
    }
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> aliasMF(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      aliasMF[lev] = std::make_unique<amrex::MultiFab>(
        *a_MFVec[lev], amrex::make_alias, a_comp + NUM_SPECIES + 2,
        NUM_SPECIES + 1);
    }
    evaluateChemExtForces(GetVecOfPtrs(aliasMF));
    nComp = 2 * (NUM_SPECIES + 1) + 1;
  } else if (a_var == "instRR") {
    for (int lev = 0; lev <= finest_level; ++lev) {
      std::unique_ptr<amrex::MultiFab> I_RR = std::make_unique<amrex::MultiFab>(
        *a_MFVec[lev], amrex::make_alias, a_comp, NUM_SPECIES);
      computeInstantaneousReactionRate(lev, AmrNewTime, I_RR.get());
    }
    nComp = NUM_SPECIES;
  } else if (a_var == "transportCC") {
    // Cell-centered transport coefficients functions go through the level
    // data container. Simply copy once the later has been filled.
    // No LES component here - this is just molecular transport coeffs
    calcViscosity(AmrNewTime);
    calcDiffusivity(AmrNewTime);
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
      amrex::MultiFab::Copy(
        *a_MFVec[lev], ldata_p->diff_cc, 0, a_comp, NUM_SPECIES + 1, 0);
      amrex::MultiFab::Copy(
        *a_MFVec[lev], ldata_p->visc_cc, 0, a_comp + NUM_SPECIES + 1, 1, 0);
      if (m_use_soret != 0) {
        amrex::MultiFab::Copy(
          *a_MFVec[lev], ldata_p->diff_cc, NUM_SPECIES + 2,
          a_comp + NUM_SPECIES + 2, NUM_SPECIES, 0);
      }
    }
    if (m_use_soret != 0) {
      nComp = 2 * NUM_SPECIES + 2;
    } else {
      nComp = NUM_SPECIES + 2;
    }
  } else if (a_var == "velForce") {
    // Velocity forces used in computing the velocity advance
    constexpr int add_gradP = 0;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> aliasMFVec(
      finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      aliasMFVec[lev] = std::make_unique<amrex::MultiFab>(
        *a_MFVec[lev], amrex::make_alias, a_comp, AMREX_SPACEDIM);
    }
    getVelForces(AmrNewTime, {}, GetVecOfPtrs(aliasMFVec), 0, add_gradP);
    nComp = AMREX_SPACEDIM;
  }
}

void
PeleLM::evaluateChemExtForces(
  const amrex::Vector<amrex::MultiFab*>& a_chemForces)
{
  //----------------------------------------------------------------
  // Copy old <- new state
  copyStateNewToOld(1);
  copyPressNewToOld();

  // The old state now holds the current solution: stamp it with the current
  // time so that boundary fills during the dt estimate use the right time
  for (int lev = 0; lev <= finest_level; ++lev) {
    m_t_old[lev] = m_cur_time;
  }

  //----------------------------------------------------------------
  // TIME
  // Compute time-step size
  m_dt = computeDt(0, AmrOldTime);

  // Update time vectors
  for (int lev = 0; lev <= finest_level; ++lev) {
    m_t_new[lev] = m_cur_time + m_dt;
  }
  //----------------------------------------------------------------

  //----------------------------------------------------------------
  // Data for the advance, only live for the duration of the advance
  std::unique_ptr<AdvanceDiffData> diffData;
  diffData = std::make_unique<AdvanceDiffData>(
    finest_level, grids, dmap, m_factory, m_nGrowAdv, m_use_wbar, m_use_soret,
    m_nAux);
  std::unique_ptr<AdvanceAdvData> advData;
  advData = std::make_unique<AdvanceAdvData>(
    finest_level, grids, dmap, m_factory, m_incompressible, m_nAux, m_nGrowAdv,
    m_nGrowMAC);

  //----------------------------------------------------------------
  // Advance setup
  // Pre-SDC
  m_sdcIter = 0;

  // fillpatch the t^{n} data
  averageDownState(AmrOldTime);
  fillPatchState(AmrOldTime);

  // compute t^{n} data
  calcViscosity(AmrOldTime);
  calcDiffusivity(AmrOldTime);
  if (m_do_les) {
    calcTurbViscosity(AmrOldTime);
  }

  floorSpecies(AmrOldTime);
  setThermoPress(AmrOldTime);
  computeDifferentialDiffusionTerms(AmrOldTime, diffData);

  // Init SDC iteration
  copyTransportOldToNew();
  copyDiffusionOldToNew(diffData);

  // Compute the instantaneous reaction rate
  // on Old state
  computeInstantaneousReactionRate(getIRVect(), AmrOldTime);

  //----------------------------------------------------------------
  // Perform a single SDC iteration
  m_sdcIter = 1;

  // Predict face velocity with Godunov
  predictVelocity(advData);

  // Create S^{n+1/2} by fillpatching t^{n} and t^{np1,k}
  createMACRHS(advData);

  // Re-evaluate thermo. pressure and add chi_increment
  addChiIncrement(m_sdcIter, AmrNewTime, advData);

  // MAC projection
  macProject(AmrOldTime, advData, GetVecOfPtrs(advData->mac_divu));

  // Get scalar advection SDC forcing
  getScalarAdvForce(advData, diffData);

  // Get AofS: (\nabla \cdot (\rho Y Umac))^{n+1/2,k}
  // and for density = \sum_k AofS_k
  computeScalarAdvTerms(advData);

  // Compute \rho^{np1,k+1} and fillpatch new density
  updateDensity(advData);
  fillPatchDensity(AmrNewTime);

  // Get scalar diffusion SDC RHS (stored in Forcing)
  getScalarDiffForce(advData, diffData);

  // Diffuse scalars
  differentialDiffusionUpdate(advData, diffData);

  // Get external forcing for chemistry
  getScalarReactForce(advData);

  // Copy external forcing for chemistry into outgoing container
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Copy(
      *a_chemForces[lev], advData->Forcing[lev], 0, 0, NUM_SPECIES + 1, 0);
  }

  // Reset state
  copyStateOldToNew();
}

void
PeleLM::evaluateAdvectionTerms(
  const amrex::Vector<amrex::MultiFab*>& a_advTerms)
{
  //----------------------------------------------------------------
  // Copy old <- new state
  copyStateNewToOld(1);
  copyPressNewToOld();

  // The old state now holds the current solution: stamp it with the current
  // time so that boundary fills during the dt estimate use the right time
  for (int lev = 0; lev <= finest_level; ++lev) {
    m_t_old[lev] = m_cur_time;
  }

  //----------------------------------------------------------------
  // TIME
  // Compute time-step size
  m_dt = computeDt(0, AmrOldTime);

  // Update time vectors
  for (int lev = 0; lev <= finest_level; ++lev) {
    m_t_new[lev] = m_cur_time + m_dt;
  }
  //----------------------------------------------------------------

  //----------------------------------------------------------------
  // Data for the advance, only live for the duration of the advance
  std::unique_ptr<AdvanceDiffData> diffData;
  diffData = std::make_unique<AdvanceDiffData>(
    finest_level, grids, dmap, m_factory, m_nGrowAdv, m_use_wbar, m_use_soret,
    m_nAux);
  std::unique_ptr<AdvanceAdvData> advData;
  advData = std::make_unique<AdvanceAdvData>(
    finest_level, grids, dmap, m_factory, m_incompressible, m_nAux, m_nGrowAdv,
    m_nGrowMAC);

  //----------------------------------------------------------------
  // Advance setup
  // Pre-SDC
  m_sdcIter = 0;

  // fillpatch the t^{n} data
  averageDownState(AmrOldTime);
  fillPatchState(AmrOldTime);

  // compute t^{n} data
  calcViscosity(AmrOldTime);
  calcDiffusivity(AmrOldTime);
  if (m_do_les) {
    calcTurbViscosity(AmrOldTime);
  }

  floorSpecies(AmrOldTime);
  setThermoPress(AmrOldTime);
  computeDifferentialDiffusionTerms(AmrOldTime, diffData);

  // Init SDC iteration
  copyTransportOldToNew();
  copyDiffusionOldToNew(diffData);

  // Compute the instantaneous reaction rate
  // on Old state
  computeInstantaneousReactionRate(getIRVect(), AmrOldTime);

  //----------------------------------------------------------------
  // Perform a single SDC iteration
  m_sdcIter = 1;

  // Predict face velocity with Godunov
  predictVelocity(advData);

  // Create S^{n+1/2} by fillpatching t^{n} and t^{np1,k}
  createMACRHS(advData);
  WriteDebugPlotFile(GetVecOfConstPtrs(advData->mac_divu), "macdivU");

  // Re-evaluate thermo. pressure and add chi_increment
  addChiIncrement(m_sdcIter, AmrNewTime, advData);
  WriteDebugPlotFile(GetVecOfConstPtrs(advData->mac_divu), "macdivUwithIncr");

  // MAC projection
  macProject(AmrOldTime, advData, GetVecOfPtrs(advData->mac_divu));

  // Get scalar advection SDC forcing
  getScalarAdvForce(advData, diffData);
  WriteDebugPlotFile(GetVecOfConstPtrs(advData->Forcing), "advForcing");

  // Get AofS: (\nabla \cdot (\rho Y Umac))^{n+1/2,k}
  // and for density = \sum_k AofS_k
  computeScalarAdvTerms(advData);

  // Compute t^{n+1/2} velocity advection term
  computeVelocityAdvTerm(advData);

  // Copy AofS into outgoing container, skip Temperature and RhoRT
  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::MultiFab::Copy(
      *a_advTerms[lev], advData->AofS[lev], 0, 0, NVAR - 2, 0);
  }

  // Reset state
  copyStateOldToNew();
}
