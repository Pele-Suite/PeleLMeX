#include <PeleLMeX.H>
#include <PeleLMeX_K.H>
#include <pelelmex_prob.H>
#include <PeleLMeX_DiffusionOp.H>
#ifdef PELE_USE_PLASMA
#include <PeleLMeX_EF_K.H>
#endif

void
PeleLM::calcTurbViscosity(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::calcTurbViscosity()");

  // We shouldn't be here unless we're doing LES
  AMREX_ALWAYS_ASSERT(m_do_les);

  if (m_les_verbose > 0) {
    amrex::Print() << "   Computing Turbulent Viscosity with LES model: "
                   << m_les_model << " for time " << getTime(0, a_time) << "\n";
  }

  // Create temporary multifab to store velocity gradient tensor
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> GradVel(
    finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_time);
    const auto& ba = ldata_p->state.boxArray();
    const auto& dm = ldata_p->state.DistributionMap();
    const auto& factory = ldata_p->state.Factory();
    constexpr int ncomp = AMREX_SPACEDIM * AMREX_SPACEDIM;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      GradVel[lev][idim].define(
        amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, ncomp,
        0, amrex::MFInfo(), factory);
      GradVel[lev][idim].setVal(0.0);
    }
  }

  // compute the velocity gradient
  getDiffusionTensorOp()->computeGradientTensor(
    GetVecOfArrOfPtrs(GradVel), GetVecOfConstPtrs(getVelocityVect(a_time)));

  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_time);

    // Get density and cp (if needed) at faces for computing turbulent transport
    // properties
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> dens_fc;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> cp_fc;
    amrex::MultiFab cp_cc;
    const auto& ba = ldata_p->state.boxArray();
    const auto& dm = ldata_p->state.DistributionMap();
    const auto& factory = ldata_p->state.Factory();
    if (m_incompressible != 0) {
      // just set density as a constant; don't need to worry about cp
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        dens_fc[idim].define(
          amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, 1,
          0, amrex::MFInfo(), factory);
        dens_fc[idim].setVal(m_rho);
      }
    } else {
      // get cp_cc (valid in 1 grow cell for interpolation to FCs)
      constexpr int ngrow = 1;
      auto const* leosparm = eos_parms.device_parm();
      cp_cc.define(ba, dm, 1, ngrow, amrex::MFInfo(), factory);
      auto const& state_arr = ldata_p->state.const_arrays();
      auto const& cp_arr = cp_cc.arrays();
      amrex::ParallelFor(
        cp_cc, cp_cc.nGrowVect(),
        [state_arr, cp_arr,
         leosparm] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
          getCpmixGivenRYT(
            i, j, k,
            amrex::Array4<amrex::Real const>(state_arr[box_no], DENSITY),
            amrex::Array4<amrex::Real const>(state_arr[box_no], FIRSTSPEC),
            amrex::Array4<amrex::Real const>(state_arr[box_no], TEMP),
            amrex::Array4<amrex::Real>(cp_arr[box_no]), leosparm);
        });
      amrex::Gpu::streamSynchronize();

      // this function really just interpolates CCs to FCs in this case
      constexpr int doZeroVisc = 0;
      auto bcRec = fetchBCRecArray(DENSITY, 1);
      dens_fc =
        getDiffusivity(lev, DENSITY, 1, doZeroVisc, {bcRec}, ldata_p->state);
      cp_fc = getDiffusivity(lev, 0, 1, doZeroVisc, {bcRec}, cp_cc);
    }

    // Now we compute the turbulent viscosity at faces
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      auto const& velgrad_arr = GradVel[lev][idim].const_arrays();
      auto const& dens_arr = dens_fc[idim].const_arrays();
      auto const& mut_arr = ldata_p->visc_turb_fc[idim].arrays();
      const amrex::Real vol = AMREX_D_TERM(
        geom[lev].CellSize(0), *geom[lev].CellSize(1), *geom[lev].CellSize(2));

      const amrex::Real l_scale =
        AMREX_D_PICK(vol, std::sqrt(vol), std::cbrt(vol));

#ifdef AMREX_USE_EB
      auto const& ebfact = EBFactory(lev);
      auto const& vfrac = ebfact.getVolFrac().const_arrays();
#endif
      if (m_les_model == "Smagorinsky") {
        const amrex::Real prefact =
          m_les_cs_smag * m_les_cs_smag * l_scale * l_scale;
        amrex::ParallelFor(
          ldata_p->visc_turb_fc[idim], ldata_p->visc_turb_fc[idim].nGrowVect(),
          [prefact, velgrad_arr, dens_arr, mut_arr
#ifdef AMREX_USE_EB
           ,
           vfrac, idim
#endif
        ] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
            getTurbViscSmagorinsky(
              i, j, k, prefact,
              amrex::Array4<amrex::Real const>(velgrad_arr[box_no]),
              amrex::Array4<amrex::Real const>(dens_arr[box_no]),
              amrex::Array4<amrex::Real>(mut_arr[box_no]));
#ifdef AMREX_USE_EB
            int idx[3] = {i, j, k};
            idx[idim] -= 1;
            const amrex::Real vfr_m = AMREX_D_PICK(
              1.0, vfrac[box_no](idx[0], idx[1], idx[2]),
              std::cbrt(
                vfrac[box_no](idx[0], idx[1], idx[2]) *
                vfrac[box_no](idx[0], idx[1], idx[2])));
            const amrex::Real vfr_p = AMREX_D_PICK(
              1.0, vfrac[box_no](i, j, k),
              std::cbrt(vfrac[box_no](i, j, k) * vfrac[box_no](i, j, k)));
            mut_arr[box_no](i, j, k) *= amrex::min(vfr_m, vfr_p);
#endif
          });
      } else if (m_les_model == "WALE") {
        const amrex::Real prefact =
          m_les_cm_wale * m_les_cm_wale * l_scale * l_scale;
        amrex::ParallelFor(
          ldata_p->visc_turb_fc[idim], ldata_p->visc_turb_fc[idim].nGrowVect(),
          [prefact, velgrad_arr, dens_arr, mut_arr
#ifdef AMREX_USE_EB
           ,
           vfrac, idim
#endif
        ] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
            getTurbViscWALE(
              i, j, k, prefact,
              amrex::Array4<amrex::Real const>(velgrad_arr[box_no]),
              amrex::Array4<amrex::Real const>(dens_arr[box_no]),
              amrex::Array4<amrex::Real>(mut_arr[box_no]));
#ifdef AMREX_USE_EB
            int idx[3] = {i, j, k};
            idx[idim] -= 1;
            const amrex::Real vfr_m = AMREX_D_PICK(
              1.0, vfrac[box_no](idx[0], idx[1], idx[2]),
              std::cbrt(
                vfrac[box_no](idx[0], idx[1], idx[2]) *
                vfrac[box_no](idx[0], idx[1], idx[2])));
            const amrex::Real vfr_p = AMREX_D_PICK(
              1.0, vfrac[box_no](i, j, k),
              std::cbrt(vfrac[box_no](i, j, k) * vfrac[box_no](i, j, k)));
            mut_arr[box_no](i, j, k) *= amrex::min(vfr_m, vfr_p);
#endif
          });
      } else if (m_les_model == "Sigma") {
        const amrex::Real prefact =
          m_les_cs_sigma * m_les_cs_sigma * l_scale * l_scale;
        amrex::ParallelFor(
          ldata_p->visc_turb_fc[idim], ldata_p->visc_turb_fc[idim].nGrowVect(),
          [prefact, velgrad_arr, dens_arr, mut_arr
#ifdef AMREX_USE_EB
           ,
           vfrac, idim
#endif
        ] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
            getTurbViscSigma(
              i, j, k, prefact,
              amrex::Array4<amrex::Real const>(velgrad_arr[box_no]),
              amrex::Array4<amrex::Real const>(dens_arr[box_no]),
              amrex::Array4<amrex::Real>(mut_arr[box_no]));
#ifdef AMREX_USE_EB
            int idx[3] = {i, j, k};
            idx[idim] -= 1;
            const amrex::Real vfr_m = AMREX_D_PICK(
              1.0, vfrac[box_no](idx[0], idx[1], idx[2]),
              std::cbrt(
                vfrac[box_no](idx[0], idx[1], idx[2]) *
                vfrac[box_no](idx[0], idx[1], idx[2])));
            const amrex::Real vfr_p = AMREX_D_PICK(
              1.0, vfrac[box_no](i, j, k),
              std::cbrt(vfrac[box_no](i, j, k) * vfrac[box_no](i, j, k)));
            mut_arr[box_no](i, j, k) *= amrex::min(vfr_m, vfr_p);
#endif
          });
      }
    }
    amrex::Gpu::streamSynchronize();
    // Compute lambda_turb = alpha_t * cp = mu_t / Pr_t * cp
    if (m_incompressible == 0) {
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        auto const& l_turb_ma = ldata_p->lambda_turb_fc[idim].arrays();
        auto const& v_turb_ma = ldata_p->visc_turb_fc[idim].const_arrays();
        auto const& cp_ma = cp_fc[idim].const_arrays();
        amrex::ParallelFor(
          ldata_p->lambda_turb_fc[idim],
          [l_turb_ma, v_turb_ma, cp_ma, inv_pr = m_Prandtl_inv] AMREX_GPU_DEVICE(
            int box_no, int i, int j, int k) noexcept {
            l_turb_ma[box_no](i, j, k) =
              cp_ma[box_no](i, j, k) * v_turb_ma[box_no](i, j, k) * inv_pr;
          });
      }
      amrex::Gpu::streamSynchronize();
    }
  }
}

void
PeleLM::calcViscosity(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::calcViscosity()");

  for (int lev = 0; lev <= finest_level; ++lev) {

    auto* ldata_p = getLevelDataPtr(lev, a_time);

    if (m_incompressible != 0) {
      ldata_p->visc_cc.setVal(m_mu);
    } else {

      // Transport data pointer
      auto const* ltransparm = trans_parms.device_parm();

      // MultiArrays
      auto const& sma = ldata_p->state.const_arrays();
      auto const& vma = ldata_p->visc_cc.arrays();

      amrex::ParallelFor(
        ldata_p->visc_cc, ldata_p->visc_cc.nGrowVect(),
        [sma, vma, ltransparm] AMREX_GPU_DEVICE(
          int box_no, int i, int j, int k) noexcept {
          getVelViscosity(
            i, j, k, amrex::Array4<amrex::Real const>(sma[box_no], FIRSTSPEC),
            amrex::Array4<amrex::Real>(sma[box_no], TEMP),
            amrex::Array4<amrex::Real>(vma[box_no], 0), ltransparm);
        });
    }
  }
  amrex::Gpu::streamSynchronize();
}

void
PeleLM::calcDiffusivity(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::calcDiffusivity()");

  const amrex::Real Pr_inv = m_Prandtl_inv;
  const amrex::Real Le_inv = m_Lewis_inv;
  const bool do_fixed_Le = (m_fixed_Le != 0);
  const bool do_fixed_Pr = (m_fixed_Pr != 0);
  const bool do_soret = (m_use_soret != 0);
  // pass soret array, or pass mu as dummy (won't do anything)
  const int soret_idx = do_soret ? 1 : 0;

  // Transport data pointer
  auto const* ltransparm = trans_parms.device_parm();
  auto const* leosparm = eos_parms.device_parm();
#ifdef PELE_USE_PLASMA
  amrex::GpuArray<amrex::Real, NUM_SPECIES> mwt{0.0};
  {
    auto eos = pele::physics::PhysicsType::eos(leosparm);
    eos.molecular_weight(mwt.arr);
  }
#endif

  for (int lev = 0; lev <= finest_level; ++lev) {

    auto* ldata_p = getLevelDataPtr(lev, a_time);

    // MultiArrays
    auto const& sma = ldata_p->state.const_arrays();
    auto const& dma = ldata_p->diff_cc.arrays();
#ifdef PELE_USE_PLASMA
    auto const& kma = ldata_p->mob_cc.arrays();
#endif

    amrex::ParallelFor(
      ldata_p->diff_cc, ldata_p->diff_cc.nGrowVect(),
      [sma, dma, soret_idx, ltransparm, do_fixed_Le, do_fixed_Pr, do_soret,
       Le_inv, Pr_inv, leosparm
#if PELE_USE_PLASMA
       ,
       kma, mwt, zk = zk
#endif
    ] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        getTransportCoeff<pele::physics::PhysicsType::eos_type>(
          i, j, k, do_fixed_Le, do_fixed_Pr, do_soret, Le_inv, Pr_inv,
          amrex::Array4<amrex::Real const>(sma[box_no], FIRSTSPEC),
          amrex::Array4<amrex::Real const>(sma[box_no], TEMP),
          amrex::Array4<amrex::Real>(dma[box_no], 0),
          amrex::Array4<amrex::Real>(dma[box_no], NUM_SPECIES + 1 + soret_idx),
          amrex::Array4<amrex::Real>(dma[box_no], NUM_SPECIES),
          amrex::Array4<amrex::Real>(dma[box_no], NUM_SPECIES + 1), ltransparm,
          leosparm);
#ifdef PELE_USE_PLASMA
        getKappaSp(
          i, j, k, mwt.arr, zk,
          amrex::Array4<amrex::Real const>(sma[box_no], FIRSTSPEC),
          amrex::Array4<amrex::Real>(dma[box_no], 0),
          amrex::Array4<amrex::Real const>(sma[box_no], TEMP),
          amrex::Array4<amrex::Real>(kma[box_no], 0));
#endif
      });

    // Fill the diff_aux MF with specified Schmidt number
    for (int n = 0; n < m_nAux; ++n) {
      auto const& diff_aux_arr = ldata_p->diff_aux_cc.arrays();
      auto const& diff_arr = ldata_p->diff_cc.const_arrays();
      if (m_aux_Schmidt[n] > 0) {
        // Compute diffusivity with Schmidt number
        const amrex::Real inv_sc = 1.0 / m_aux_Schmidt[n];
        amrex::ParallelFor(
          ldata_p->diff_aux_cc, ldata_p->diff_aux_cc.nGrowVect(),
          [diff_aux_arr, diff_arr, inv_sc,
           n] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
            diff_aux_arr[box_no](i, j, k, n) =
              diff_arr[box_no](i, j, k, NUM_SPECIES + 1) * inv_sc;
          });
      } else {
        // Otherwise, assume unity Lewis number
        const auto& ba = ldata_p->state.boxArray();
        const auto& dm = ldata_p->state.DistributionMap();
        const auto& factory = ldata_p->state.Factory();
        const int ngrow = ldata_p->diff_cc.nGrow();
        amrex::MultiFab cp_cc(ba, dm, 1, ngrow, amrex::MFInfo(), factory);
        auto const& state_arr = ldata_p->state.const_arrays();
        auto const& cp_arr = cp_cc.arrays();
        amrex::ParallelFor(
          ldata_p->diff_aux_cc, ldata_p->diff_aux_cc.nGrowVect(),
          [state_arr, cp_arr, diff_aux_arr, diff_arr, leosparm,
           n] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
            getCpmixGivenRYT(
              i, j, k,
              amrex::Array4<amrex::Real const>(state_arr[box_no], DENSITY),
              amrex::Array4<amrex::Real const>(state_arr[box_no], FIRSTSPEC),
              amrex::Array4<amrex::Real const>(state_arr[box_no], TEMP),
              amrex::Array4<amrex::Real>(cp_arr[box_no]), leosparm);
            diff_aux_arr[box_no](i, j, k, n) =
              diff_arr[box_no](i, j, k, NUM_SPECIES) / cp_arr[box_no](i, j, k);
          });
      }
    }
  }
  amrex::Gpu::streamSynchronize();
}

amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>
PeleLM::getDiffusivity(
  const int lev,
  const int beta_comp,
  const int ncomp,
  const int doZeroVisc,
  amrex::Vector<amrex::BCRec> const& bcrec,
  amrex::MultiFab const& beta_cc,
  const int addTurbContrib)
{
  BL_PROFILE("PeleLMeX::getDiffusivity()");

  AMREX_ASSERT(bcrec.size() >= ncomp);
  AMREX_ASSERT(beta_cc.nComp() >= beta_comp + ncomp);

  const auto& ba = beta_cc.boxArray();
  const auto& dm = beta_cc.DistributionMap();
  const auto& factory = beta_cc.Factory();
  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> beta_ec{AMREX_D_DECL(
    amrex::MultiFab(
      amrex::convert(ba, amrex::IntVect::TheDimensionVector(0)), dm, ncomp, 0,
      amrex::MFInfo(), factory),
    amrex::MultiFab(
      amrex::convert(ba, amrex::IntVect::TheDimensionVector(1)), dm, ncomp, 0,
      amrex::MFInfo(), factory),
    amrex::MultiFab(
      amrex::convert(ba, amrex::IntVect::TheDimensionVector(2)), dm, ncomp, 0,
      amrex::MFInfo(), factory))};

  const amrex::Box& domain = geom[lev].Domain();

#ifdef AMREX_USE_EB
  // EB : use EB CCentroid -> FCentroid
  EB_interp_CellCentroid_to_FaceCentroid(
    beta_cc, GetArrOfPtrs(beta_ec), beta_comp, 0, ncomp, geom[lev], bcrec);
  EB_set_covered_faces(GetArrOfPtrs(beta_ec), 1.234e40);
#else
  // NON-EB : use cen2edg_cpp
  const bool use_harmonic_avg = m_harm_avg_cen2edge != 0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(beta_cc, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      const amrex::Box ebx = mfi.nodaltilebox(idim);
      const amrex::Box& edomain = amrex::surroundingNodes(domain, idim);
      const auto& diff_c = beta_cc.const_array(mfi, beta_comp);
      const auto& diff_ec = beta_ec[idim].array(mfi);
      const auto bc_lo = bcrec[0].lo(idim);
      const auto bc_hi = bcrec[0].hi(idim);
      amrex::ParallelFor(
        ebx, [bc_lo, bc_hi, diff_ec, use_harmonic_avg, ncomp, idim, edomain,
              diff_c] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          const int idx[3] = {i, j, k};
          const bool on_lo =
            ((bc_lo == amrex::BCType::ext_dir) &&
             (idx[idim] <= edomain.smallEnd(idim)));
          const bool on_hi =
            ((bc_hi == amrex::BCType::ext_dir) &&
             (idx[idim] >= edomain.bigEnd(idim)));
          cen2edg_cpp(
            i, j, k, idim, ncomp, use_harmonic_avg, on_lo, on_hi, diff_c,
            diff_ec);
        });
    }
  }
#endif

  // Add the turbulent component if desired
  // Use the following relationships to translate from the ncomp and beta_comp
  // to the component ncomp = NUM_SPECIES, beta_comp = 0           --> SPECIES
  // DIFFUSIVITY ncomp = 1          , beta_comp = NUM_SPECIES --> THERMAL
  // CONDUCTIVITY ncomp = 1          , beta_comp = 0           --> VISCOSITY
  // (auxiliary diffusivities are recognised separately). If
  // PELE_USE_PLASMA is active, these relationships will not hold and LES is not
  // supported
  if ((addTurbContrib != 0) and m_do_les) {

    // Turbulent viscosity - always in old data except during initialization
    // (not updated because velocity doesn't get updated until end of advance)
    TimeStamp tstamp;
    if (getTime(lev, AmrNewTime) == 0.0) {
      tstamp = AmrNewTime;
      if (lev == 0) {
        calcTurbViscosity(tstamp);
      }
    } else {
      tstamp = AmrOldTime;
    }
    auto* ldata_p = getLevelDataPtr(lev, tstamp);

    // Auxiliary scalars: identified by their own diffusivity container, as
    // (ncomp, beta_comp) can coincide with the viscosity or species cases
    const bool is_aux =
      (m_nAux > 0) && (&beta_cc == &m_leveldata_old[lev]->diff_aux_cc ||
                       &beta_cc == &m_leveldata_new[lev]->diff_aux_cc);

    // Identify and add the correct turbulent contribution
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      if (is_aux) { // Auxiliary diffusivity: mu_t / Sc_t, as for species
        for (int n = 0; n < ncomp; ++n) {
          amrex::MultiFab::Saxpy(
            beta_ec[idim], m_Schmidt_inv, ldata_p->visc_turb_fc[idim], 0, n, 1,
            0);
        }
      } else if ((ncomp == 1) and (beta_comp == 0)) { // Viscosity
        amrex::MultiFab::Add(
          beta_ec[idim], ldata_p->visc_turb_fc[idim], 0, 0, 1, 0);
      } else if ((ncomp == NUM_SPECIES) and (beta_comp == 0)) { // Species
                                                                // diffusivity
        for (int ispec = 0; ispec < NUM_SPECIES; ++ispec) {
          amrex::MultiFab::Saxpy(
            beta_ec[idim], m_Schmidt_inv, ldata_p->visc_turb_fc[idim], 0, ispec,
            1, 0);
        }
      } else if ((ncomp == 1) and (beta_comp == NUM_SPECIES)) { // Thermal
                                                                // conductivity
        amrex::MultiFab::Add(
          beta_ec[idim], ldata_p->lambda_turb_fc[idim], 0, 0, 1, 0);
      } else { // Invalid
        amrex::Abort(
          "getDiffusivity(): LES model is on but cannot provide a "
          "turbulent transport coefficient");
      }
    }
  }

  // Enable zeroing diffusivity on faces to produce walls
  if (doZeroVisc != 0) {
    ProbParm const* lprobparm = prob_parm_d;
    const auto geomdata = geom[lev].data();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      const amrex::Box& edomain = amrex::surroundingNodes(domain, idim);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(beta_ec[idim], amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
        const amrex::Box ebx = mfi.tilebox();
        const auto& diff_ec = beta_ec[idim].array(mfi);
        amrex::ParallelFor(
          ebx, [diff_ec, geomdata, edomain, idim, beta_comp, ncomp,
                lprobparm] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            ProblemSpecificFunctions::zero_visc(
              i, j, k, diff_ec, geomdata, edomain, idim, beta_comp, ncomp,
              *lprobparm);
          });
      }
    }
  }

  return beta_ec;
}
