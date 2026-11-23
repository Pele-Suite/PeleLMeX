#include <PeleLMeX.H>
#include <PeleLMeX_Derive.H>
#include <PeleLMRad.H>

void
PeleLM::RadInit()
{
  amrex::Print() << "Radiation model is activated " << '\n';
  PeleRad::RadComps rc;
  amrex::Vector<std::string> names;
  pele::physics::eos::speciesNames<pele::physics::PhysicsType::eos_type>(names);
  for (int i = 0; i < NUM_SPECIES; ++i) {
    if (names[i] == "CO2") {
      rc.co2Indx = i;
      continue;
    }
    if (names[i] == "H2O") {
      rc.h2oIndx = i;
      continue;
    }
    if (names[i] == "CO") {
      rc.coIndx = i;
    }
  }
  amrex::ParmParse mlmgpp("pelerad");

#ifdef AMREX_USE_EB
  amrex::Print() << "The EB support is activated in the radiation module \n";
  rad_model = std::make_unique<PeleRad::Radiation>(
    geom, grids, dmap, rc, mlmgpp, m_factory);
#else
  rad_model =
    std::make_unique<PeleRad::Radiation>(geom, grids, dmap, rc, mlmgpp);
#endif
}

void
PeleLM::computeRadSource(const TimeStamp a_timestamp)
{
  auto const radIndx = rad_model->readRadIndices();
  amrex::GpuArray<int, 3> const xradIndx{
    radIndx.co2Indx, radIndx.h2oIndx, radIndx.coIndx};

  BL_PROFILE_VAR("PeleLM::advance::rad::spec", PLM_RAD_SPEC);
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = PeleLM::getLevelDataPtr(lev, a_timestamp);

    // The Planck-mean absorption coefficient is P * sum_i X_i kp_i(T): the
    // model takes mole fractions (CO2, H2O, CO), while the state holds rho*Y.
    // A species the mechanism lacks (index -1), or a cell with no density
    // (EB covered), gets X = 0.
    amrex::MultiFab xrad(grids[lev], dmap[lev], 3, 0);
    auto const& state_ma = ldata_p->state.const_arrays();
    auto const& xrad_ma = xrad.arrays();
    auto const* leosparm = eos_parms.device_parm();
    amrex::ParallelFor(
      xrad, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        auto const& state = state_ma[box_no];
        amrex::Real Yt[NUM_SPECIES] = {0.0};
        amrex::Real Xt[NUM_SPECIES] = {0.0};
        amrex::Real const rho = state(i, j, k, DENSITY);
        if (rho > 0.0) {
          for (int n = 0; n < NUM_SPECIES; ++n) {
            Yt[n] = state(i, j, k, FIRSTSPEC + n) / rho;
          }
          auto eos = pele::physics::PhysicsType::eos(leosparm);
          eos.Y2X(Yt, Xt);
        }
        for (int c = 0; c < 3; ++c) {
          xrad_ma[box_no](i, j, k, c) =
            xradIndx[c] >= 0 ? Xt[xradIndx[c]] : 0.0;
        }
      });

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*(m_extSource[lev]), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      auto const& q_xin_co2 = xrad.const_array(mfi, 0);
      auto const& q_xin_h2o = xrad.const_array(mfi, 1);
      auto const& q_xin_co = xrad.const_array(mfi, 2);
      auto const& q_Tin = ldata_p->state.const_array(mfi, TEMP);
      auto const& q_Pin = ldata_p->state.const_array(mfi, RHORT);
#ifdef PELE_USE_SOOT
      auto const& q_fvin = ldata_p->state.const_array(mfi, FIRSTSOOT + 1);
      rad_model->updateSpecProp(
        mfi, q_xin_co2, q_xin_h2o, q_xin_co, q_Tin, q_Pin, q_fvin, lev);
#else
      rad_model->updateSpecProp(
        mfi, q_xin_co2, q_xin_h2o, q_xin_co, q_Tin, q_Pin, lev);
#endif
    }
  }
  BL_PROFILE_VAR_STOP(PLM_RAD_SPEC);

  BL_PROFILE_VAR("PeleLM::advance::rad::solve", PLM_RAD_SOLV);
  rad_model->evaluateRad();

  for (int lev = 0; lev <= finest_level; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*(m_extSource[lev]), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
      auto const& source_arr = m_extSource[lev]->array(mfi, RHOH);
      rad_model->calcRadSource(mfi, source_arr, lev);
    }
  }
  BL_PROFILE_VAR_STOP(PLM_RAD_SOLV);
}
