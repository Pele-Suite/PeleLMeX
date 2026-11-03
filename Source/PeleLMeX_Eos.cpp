#include <PeleLMeX.H>
#include <PeleLMeX_K.H>
#include <memory>

void
PeleLM::setThermoPress(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::setThermoPress()");

  AMREX_ASSERT(a_time == AmrOldTime || a_time == AmrNewTime);

  for (int lev = 0; lev <= finest_level; ++lev) {
    setThermoPress(lev, a_time);
  }
}

void
PeleLM::setThermoPress(const int lev, const TimeStamp a_time)
{

  AMREX_ASSERT(a_time == AmrOldTime || a_time == AmrNewTime);

  auto* ldata_p = getLevelDataPtr(lev, a_time);
  auto const& sma = ldata_p->state.arrays();
  auto const* leosparm = eos_parms.device_parm();

  amrex::ParallelFor(
    ldata_p->state,
    [sma, leosparm] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      getPGivenRTY(
        i, j, k, amrex::Array4<amrex::Real const>(sma[box_no], DENSITY),
        amrex::Array4<amrex::Real const>(sma[box_no], FIRSTSPEC),
        amrex::Array4<amrex::Real const>(sma[box_no], TEMP),
        amrex::Array4<amrex::Real>(sma[box_no], RHORT), leosparm);
    });
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::calcDivU(
  const int is_init,
  const int computeDiff,
  const int do_avgDown,
  const TimeStamp a_time,
  const std::unique_ptr<AdvanceDiffData>& diffData)
{
  BL_PROFILE("PeleLMeX::calcDivU()");

  AMREX_ASSERT(a_time == AmrOldTime || a_time == AmrNewTime);

  // If requested, compute diffusion terms
  // otherwise assumes it has already been computed and stored in the proper
  // container of diffData
  if (computeDiff != 0) {
    calcDiffusivity(a_time);
    computeDifferentialDiffusionTerms(a_time, diffData, is_init);
  }

  // Assemble divU on each level
  for (int lev = 0; lev <= finest_level; ++lev) {

    auto* ldata_p = getLevelDataPtr(lev, a_time);

    amrex::MultiFab RhoYdot;
    if ((m_do_react != 0) && (m_skipInstantRR == 0)) {
      if (is_init != 0) { // Either pre-divU, divU or press initial iterations
        if (m_dt > 0.0) { // divU ite   -> use I_R
          auto* ldataR_p = getLevelDataReactPtr(lev);
          RhoYdot.define(grids[lev], dmap[lev], nCompIR(), 0);
          amrex::MultiFab::Copy(RhoYdot, ldataR_p->I_R, 0, 0, nCompIR(), 0);
        } else { // press ite  -> set to zero
          RhoYdot.define(grids[lev], dmap[lev], nCompIR(), 0);
          RhoYdot.setVal(0.0);
        }
      } else { // Regular    -> use instantaneous RR
        RhoYdot.define(grids[lev], dmap[lev], nCompIR(), 0);
#ifdef PELE_USE_PLASMA
        computeInstantaneousReactionRateEF(lev, a_time, &RhoYdot);
#else
        computeInstantaneousReactionRate(lev, a_time, &RhoYdot);
#endif
      }
    }

    //----------------------------------------------------------------
#ifdef AMREX_USE_EB
    // Get EBFact
    const auto& ebfact = EBFactory(lev);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ldata_p->divu, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();

#ifdef AMREX_USE_EB
      auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
      auto const& flag = flagfab.const_array();
#endif

      auto const& rhoY = ldata_p->state.const_array(mfi, FIRSTSPEC);
      auto const& T = ldata_p->state.const_array(mfi, TEMP);
      auto const& SpecD = (a_time == AmrOldTime)
                            ? diffData->Dn[lev].const_array(mfi, 0)
                            : diffData->Dnp1[lev].const_array(mfi, 0);
      auto const& Fourier =
        (a_time == AmrOldTime)
          ? diffData->Dn[lev].const_array(mfi, NUM_SPECIES)
          : diffData->Dnp1[lev].const_array(mfi, NUM_SPECIES);
      auto const& DiffDiff =
        (a_time == AmrOldTime)
          ? diffData->Dn[lev].const_array(mfi, NUM_SPECIES + 1)
          : diffData->Dnp1[lev].const_array(mfi, NUM_SPECIES + 1);
      auto const& r =
        ((m_do_react != 0) && (m_skipInstantRR == 0))
          ? RhoYdot.const_array(mfi)
          : ldata_p->state.const_array(mfi, FIRSTSPEC); // Dummy unused Array4
      auto const& extRhoY = m_extSource[lev]->const_array(mfi, FIRSTSPEC);
      auto const& extRhoH = m_extSource[lev]->const_array(mfi, RHOH);
      auto const& divu = ldata_p->divu.array(mfi);
      int use_react = ((m_do_react != 0) && (m_skipInstantRR == 0)) ? 1 : 0;
      auto const* leosparm = eos_parms.device_parm();

#ifdef AMREX_USE_EB
      if (flagfab.getType(bx) == amrex::FabType::covered) { // Covered boxes
        amrex::ParallelFor(
          bx, [divu] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            divu(i, j, k) = 0.0;
          });
      } else if (flagfab.getType(bx) != amrex::FabType::regular) { // EB
                                                                   // containing
                                                                   // boxes
        amrex::ParallelFor(
          bx,
          [flag, divu, rhoY, T, SpecD, Fourier, DiffDiff, r, extRhoY, extRhoH,
           use_react, leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (flag(i, j, k).isCovered()) {
              divu(i, j, k) = 0.0;
            } else {
              compute_divu<pele::physics::PhysicsType::eos_type>(
                i, j, k, rhoY, T, SpecD, Fourier, DiffDiff, r, extRhoY, extRhoH,
                divu, use_react, leosparm);
            }
          });
      } else
#endif
      {
        amrex::ParallelFor(
          bx,
          [divu, rhoY, T, SpecD, Fourier, DiffDiff, r, extRhoY, extRhoH,
           use_react, leosparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            compute_divu<pele::physics::PhysicsType::eos_type>(
              i, j, k, rhoY, T, SpecD, Fourier, DiffDiff, r, extRhoY, extRhoH,
              divu, use_react, leosparm);
          });
      }
    }
  }

  // Average down divU
  if (do_avgDown != 0) {
    for (int lev = finest_level; lev > 0; --lev) {
      auto* ldataFine_p = getLevelDataPtr(lev, a_time);
      auto* ldataCrse_p = getLevelDataPtr(lev - 1, a_time);
#ifdef AMREX_USE_EB
      EB_average_down(
        ldataFine_p->divu, ldataCrse_p->divu, 0, 1, refRatio(lev - 1));
#else
      average_down(
        ldataFine_p->divu, ldataCrse_p->divu, 0, 1, refRatio(lev - 1));
#endif
    }
  }

  // fillPatch a_time divu to get properly filled ghost cells
  for (int lev = 0; lev <= finest_level; ++lev) {
    const amrex::Real time = getTime(lev, a_time);
    auto* ldata_p = getLevelDataPtr(lev, a_time);
    fillpatch_divu(lev, time, ldata_p->divu, m_nGrowdivu);
  }
}

void
PeleLM::setRhoToSumRhoY(const int lev, const TimeStamp a_time)
{

  AMREX_ASSERT(a_time == AmrOldTime || a_time == AmrNewTime);

  auto* ldata_p = getLevelDataPtr(lev, a_time);
  auto const& sma = ldata_p->state.arrays();

  amrex::ParallelFor(
    ldata_p->state,
    [sma] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      pele::physics::PhysicsType::eos_type::RY2R(
        sma[box_no].cellData(i, j, k), sma[box_no](i, j, k, DENSITY),
        FIRSTSPEC);
    });
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::setTemperature(const TimeStamp a_time)
{
  BL_PROFILE_VAR("PeleLMeX::setTemperature()", setTemperature);

  AMREX_ASSERT(a_time == AmrOldTime || a_time == AmrNewTime);

  for (int lev = 0; lev <= finest_level; ++lev) {
    setTemperature(lev, a_time);
  }
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::setTemperature(const int lev, const TimeStamp a_time)
{

  AMREX_ASSERT(a_time == AmrOldTime || a_time == AmrNewTime);

  auto* ldata_p = getLevelDataPtr(lev, a_time);
  auto const& sma = ldata_p->state.arrays();
  auto const* leosparm = eos_parms.device_parm();

  amrex::ParallelFor(
    ldata_p->state,
    [sma, leosparm] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      getTfromHY(
        i, j, k, amrex::Array4<amrex::Real const>(sma[box_no], DENSITY),
        amrex::Array4<amrex::Real const>(sma[box_no], FIRSTSPEC),
        amrex::Array4<amrex::Real const>(sma[box_no], RHOH),
        amrex::Array4<amrex::Real>(sma[box_no], TEMP), leosparm);
    });
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::calc_dPdt(
  const TimeStamp a_time, const amrex::Vector<amrex::MultiFab*>& a_dPdt)
{
  BL_PROFILE("PeleLMeX::calc_dPdt()");

  AMREX_ASSERT(a_time == AmrOldTime || a_time == AmrNewTime);

  for (int lev = 0; lev <= finest_level; ++lev) {
    calc_dPdt(lev, a_time, a_dPdt[lev]);
#ifdef AMREX_USE_EB
    EB_set_covered(*a_dPdt[lev], 0.0);
#endif
  }

  // Fill ghost cell(s)
  if (a_dPdt[0]->nGrow() > 0) {
    fillpatch_forces(m_cur_time, a_dPdt, a_dPdt[0]->nGrow());
  }
}

void
PeleLM::calc_dPdt(
  const int lev, const TimeStamp a_time, amrex::MultiFab* a_dPdt)
{
  auto const& sma = getLevelDataPtr(lev, a_time)->state.arrays();
  auto const& dPdtma = a_dPdt->arrays();

  // Use new ambient pressure to compute dPdt
  const amrex::Real p_amb = m_pNew;
  const auto dt = m_dt;
  const auto dpdt_fac = m_dpdtFactor;
  amrex::ParallelFor(
    *a_dPdt, [dPdtma, sma, p_amb, dt, dpdt_fac] AMREX_GPU_DEVICE(
               int box_no, int i, int j, int k) noexcept {
      auto dPdta = dPdtma[box_no];
      auto sa = sma[box_no];
      dPdta(i, j, k) =
        (sa(i, j, k, RHORT) - p_amb) / (dt * sa(i, j, k, RHORT)) * dpdt_fac;
    });
  amrex::Gpu::streamSynchronize();
}

amrex::Real
PeleLM::adjustPandDivU(const std::unique_ptr<AdvanceAdvData>& advData)
{
  BL_PROFILE("PeleLMeX::adjustPandDivU()");

  amrex::Vector<std::unique_ptr<amrex::MultiFab>> ThetaHalft(finest_level + 1);

  // Get theta = 1 / (\Gamma * P_amb) at half time
  for (int lev = 0; lev <= finest_level; ++lev) {
    ThetaHalft[lev] = std::make_unique<amrex::MultiFab>(
      grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), *m_factory[lev]);
  }
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto const& tma = ThetaHalft[lev]->arrays();
    auto const& sma_o = getLevelDataPtr(lev, AmrOldTime)->state.const_arrays();
    auto const& sma_n = getLevelDataPtr(lev, AmrNewTime)->state.const_arrays();
    auto const* leosparm = eos_parms.device_parm();

    const auto pOld = m_pOld;
    const auto pNew = m_pNew;
    amrex::ParallelFor(
      *ThetaHalft[lev],
      [tma, sma_o, sma_n, pOld, pNew,
       leosparm] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        auto theta = tma[box_no];
        const amrex::Real gammaInv_o = getGammaInv(
          i, j, k, amrex::Array4<amrex::Real const>(sma_o[box_no], FIRSTSPEC),
          amrex::Array4<amrex::Real const>(sma_o[box_no], TEMP), leosparm);
        const amrex::Real gammaInv_n = getGammaInv(
          i, j, k, amrex::Array4<amrex::Real const>(sma_n[box_no], FIRSTSPEC),
          amrex::Array4<amrex::Real const>(sma_n[box_no], TEMP), leosparm);
        theta(i, j, k) = 0.5 * (gammaInv_o / pOld + gammaInv_n / pNew);
      });
  }
  amrex::Gpu::streamSynchronize();

  // Get the mean mac_divu (Sbar) and mean theta
  const amrex::Real Sbar =
    MFSum(GetVecOfConstPtrs(advData->mac_divu), 0) / m_uncoveredVol;
  const amrex::Real Thetabar =
    MFSum(GetVecOfConstPtrs(ThetaHalft), 0) / m_uncoveredVol;

  // Adjust
  for (int lev = 0; lev <= finest_level; ++lev) {
    // ThetaHalft is now delta_theta
    ThetaHalft[lev]->plus(-Thetabar, 0, 1);
    // mac_divu is now delta_S
    advData->mac_divu[lev].plus(-Sbar, 0, 1);
  }

  // Net outward volume flux through the domain boundary, per unit volume,
  // from the predicted face velocity (inflow faces carry their boundary
  // values).  This has to be the flux the MAC projection will see: it was
  // previously taken from m_domainUmacFlux, which is accumulated after the
  // projection (so the first SDC iteration saw zero) with the opposite sign.
  // Under mesh mapping the projector works with (J/fac).u on the uniform
  // grid, so the flux is taken from that scaled velocity.
  amrex::Real umacFluxBalance = 0.0;
  if (m_mesh_mapping) {
    amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> umac_xi(
      finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        umac_xi[lev][idim].define(
          advData->umac[lev][idim].boxArray(),
          advData->umac[lev][idim].DistributionMap(), 1, 0, amrex::MFInfo(),
          advData->umac[lev][idim].Factory());
        const auto& fac_ma = m_mesh_map->fac_fc(lev, idim).const_arrays();
        const auto& detJ_ma = m_mesh_map->detJ_fc(lev, idim).const_arrays();
        const auto& umac_ma = advData->umac[lev][idim].const_arrays();
        const auto& uxi_ma = umac_xi[lev][idim].arrays();
        const int nc = idim;
        amrex::ParallelFor(
          umac_xi[lev][idim],
          [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
            uxi_ma[box_no](i, j, k) = umac_ma[box_no](i, j, k) *
                                      detJ_ma[box_no](i, j, k) /
                                      fac_ma[box_no](i, j, k, nc);
          });
      }
    }
    amrex::Gpu::streamSynchronize();
    umacFluxBalance = domainFaceFlux(GetVecOfArrOfConstPtrs(umac_xi));
  } else {
    umacFluxBalance = domainFaceFlux(GetVecOfArrOfConstPtrs(advData->umac));
  }
  const amrex::Real divu_vol = umacFluxBalance / m_uncoveredVol;

  // Advance the ambient pressure
  m_pNew = m_pOld + m_dt * (Sbar - divu_vol) / Thetabar;
  m_dp0dt = (Sbar - divu_vol) / Thetabar;

  // subtract \tilde{theta} * Sbar / Thetabar from divu
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto const& tma = ThetaHalft[lev]->arrays();
    auto const& uma = advData->mac_divu[lev].arrays();
    amrex::ParallelFor(
      *ThetaHalft[lev], [uma, Sbar, Thetabar, divu_vol, tma] AMREX_GPU_DEVICE(
                          int box_no, int i, int j, int k) noexcept {
        auto theta = tma[box_no];
        uma[box_no](i, j, k) -=
          (theta(i, j, k) * Sbar / Thetabar -
           divu_vol * (1 + theta(i, j, k) / Thetabar));
      });
  }
  amrex::Gpu::streamSynchronize();

  if (m_verbose > 2 || m_debug_divu_mismatch != 0) {
    amrex::Print() << " >> Closed chamber pOld: " << m_pOld
                   << ", pNew: " << m_pNew << ", dp0dt: " << m_dp0dt
                   << ", Sbar: " << Sbar << ", Thetabar: " << Thetabar
                   << ", divu_vol: " << divu_vol << "\n";
    amrex::Print() << " >> Total mass old: " << m_massOld
                   << ", mass new: " << m_massNew << "\n";
  }

  // Return Sbar so that we'll add it back to mac_divu after the MAC projection
  return Sbar;
}
