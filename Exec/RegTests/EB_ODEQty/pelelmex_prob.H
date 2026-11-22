#ifndef PELELM_PROB_H
#define PELELM_PROB_H

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#include <PeleLMeX_ProblemSpecificFunctions.H>
#include <PeleLMeX_Index.H>
#include <PMFData.H>
#include <PelePhysics.H>

struct MyProbParm : public ProbParmDefault
{
  // Shared params
  amrex::Real T_mean = 298.0;
  amrex::Real meanFlowMag = 0.0;
  int meanFlowDir = 1;
  bool ode_qty_test = false;
  amrex::Real ode_IC = 0.0;
  amrex::Real ode_xy_lo = 0.0;
  amrex::Real ode_length = 0.0;
  amrex::Real ode_height = 0.0;
  amrex::Real ode_srcstrength = 0.0;
  bool composition_test = false;
  amrex::Real Y_CO2_0 = 0.0;
  amrex::Real Y_N2_0 = 0.25;
  amrex::Real Y_AR_0 = 0.75;
  amrex::Real extRhoYCO2 = 0.0;
  amrex::Real extRhoYCO2_ts = 0.0;
  amrex::Real extRhoYCO2_ramp = 0.0;
  // Isothermal injection: Gaussian CO2 mass source carrying h_CO2(T_mean),
  // so T must stay at T_mean while the composition change drives a flow
  bool isothermal_test = false;
  amrex::Real iso_amp = 0.0;
  amrex::Real iso_sigma = 0.0;
};

struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void initdata(
    int i,
    int j,
    int k,
    int is_incompressible,
    amrex::Array4<amrex::Real> const& state,
    amrex::Array4<amrex::Real> const& /*aux*/,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    auto eos = pele::physics::PhysicsType::eos();

    amrex::Real massfrac[NUM_SPECIES] = {0.0};
    massfrac[CO2_ID] = prob_parm.Y_CO2_0;
    massfrac[N2_ID] = prob_parm.Y_N2_0;
    massfrac[AR_ID] = prob_parm.Y_AR_0;

    switch (prob_parm.meanFlowDir) {
    case 1:
      AMREX_D_TERM(state(i, j, k, VELX) = prob_parm.meanFlowMag;
                   , state(i, j, k, VELY) = 0.0;, state(i, j, k, VELZ) = 0.0);
      break;
    case -1:
      AMREX_D_TERM(state(i, j, k, VELX) = -prob_parm.meanFlowMag;
                   , state(i, j, k, VELY) = 0.0;, state(i, j, k, VELZ) = 0.0);
      break;
    case 2:
      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;
                   , state(i, j, k, VELY) = prob_parm.meanFlowMag;
                   , state(i, j, k, VELZ) = 0.0);
      break;
    case -2:
      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;
                   , state(i, j, k, VELY) = -prob_parm.meanFlowMag;
                   , state(i, j, k, VELZ) = 0.0);
      break;
    case 3:
      AMREX_D_TERM(state(i, j, k, VELX) = prob_parm.meanFlowMag;
                   , state(i, j, k, VELY) = prob_parm.meanFlowMag;
                   , state(i, j, k, VELZ) = 0.0);
      break;
    case -3:
      AMREX_D_TERM(state(i, j, k, VELX) = -prob_parm.meanFlowMag;
                   , state(i, j, k, VELY) = -prob_parm.meanFlowMag;
                   , state(i, j, k, VELZ) = 0.0);
      break;
    default:
      break;
    }

    if (!(is_incompressible == 0)) {
      return;
    }

    state(i, j, k, TEMP) = prob_parm.T_mean;

    amrex::Real P_cgs = prob_parm.P_mean * 10.0;

    // Density
    amrex::Real rho_cgs = 0.0;
    eos.PYT2R(P_cgs, massfrac, state(i, j, k, TEMP), rho_cgs);
    state(i, j, k, DENSITY) = rho_cgs * 1.0e3;

    // Enthalpy
    amrex::Real h_cgs = 0.0;
    eos.TY2H(state(i, j, k, TEMP), massfrac, h_cgs);
    state(i, j, k, RHOH) = h_cgs * 1.0e-4 * state(i, j, k, DENSITY);

    // Species mass
    for (int n = 0; n < NUM_SPECIES; n++) {
      state(i, j, k, FIRSTSPEC + n) = massfrac[n] * state(i, j, k, DENSITY);
    }

    // Initial ODE quantity
#if NUM_ODE > 0
    amrex::Real ode_qty[NUM_ODE] = {0.0};

    if (prob_parm.ode_qty_test) {
      // Initial state for ode quantities
      amrex::Real ode_qty_0[NUM_ODE] = {0.0};
      for (int n = 0; n < NUM_ODE; n++) {
        ode_qty_0[n] = (n + 1) * prob_parm.ode_IC;
      }

      // Current x,y,z locations
      const amrex::Real* prob_lo = geomdata.ProbLo();
      const amrex::Real* dx = geomdata.CellSize();
      const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
        prob_lo[0] + static_cast<amrex::Real>(i + 0.5) * dx[0],
        prob_lo[1] + static_cast<amrex::Real>(j + 0.5) * dx[1],
        prob_lo[2] + static_cast<amrex::Real>(k + 0.5) * dx[2])};

      // Create NUM_ODE boxes from (x_strt,y_strt) to (x_end,y_end)
      for (int n = 0; n < NUM_ODE; n++) {
        amrex::Real x_strt =
          prob_parm.ode_xy_lo + n * 4. * prob_parm.ode_length;
        amrex::Real x_end = x_strt + prob_parm.ode_length;
        amrex::Real y_strt = prob_parm.ode_xy_lo;
        amrex::Real y_end = y_strt + prob_parm.ode_height;
        if ((x[0] > x_strt) && (x[1] > y_strt)) {
          if ((x[0] < x_end) && (x[1] < y_end)) {
            ode_qty[n] = ode_qty_0[n];
          }
        }
      }
    }
    // Add to state variable
    for (int n = 0; n < NUM_ODE; n++) {
      state(i, j, k, FIRSTODE + n) = ode_qty[n];
    }
#endif
  }

  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bcnormal(
    const amrex::Real* /*x[AMREX_SPACEDIM]*/,
    const amrex::CellData<amrex::Real>& /*s_in*/,
    amrex::Real s_ext[NVAR],
    const int /*idir*/,
    const int /*sgn*/,
    const amrex::Real /*time*/,
    amrex::GeometryData const& /*geomdata*/,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    amrex::Real massfrac[NUM_SPECIES] = {0.0};
    massfrac[CO2_ID] = prob_parm.Y_CO2_0;
    massfrac[N2_ID] = prob_parm.Y_N2_0;
    massfrac[AR_ID] = prob_parm.Y_AR_0;

    auto eos = pele::physics::PhysicsType::eos();

    switch (prob_parm.meanFlowDir) {
    case 1:
      AMREX_D_TERM(
        s_ext[VELX] = prob_parm.meanFlowMag;, s_ext[VELY] = 0.0;
        , s_ext[VELZ] = 0.0);
      break;
    case 2:
      AMREX_D_TERM(
        s_ext[VELY] = prob_parm.meanFlowMag;, s_ext[VELX] = 0.0;
        , s_ext[VELZ] = 0.0);
      break;
    case 3:
      AMREX_D_TERM(
        s_ext[VELZ] = prob_parm.meanFlowMag;, s_ext[VELX] = 0.0;
        , s_ext[VELY] = 0.0);
      break;
    default:
      break;
    }

    s_ext[TEMP] = prob_parm.T_mean;

    amrex::Real rho_cgs, P_cgs, RhoH_temp;
    P_cgs = prob_parm.P_mean * 10.0;

    eos.PYT2R(P_cgs, massfrac, s_ext[TEMP], rho_cgs);
    s_ext[DENSITY] = rho_cgs * 1.0e3;

    eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
    s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion

    for (int n = 0; n < NUM_SPECIES; n++) {
      s_ext[FIRSTSPEC + n] = massfrac[n] * s_ext[DENSITY];
    }
    for (int n = 0; n < NUM_ODE; n++) {
      s_ext[FIRSTODE + n] = 0.0;
    }
  }

  static void set_ode_names(amrex::Vector<std::string>& a_ode_names)
  {
    a_ode_names.resize(NUM_ODE);
    for (int n = 0; n < NUM_ODE; n++) {
      a_ode_names[n] = "MY_ODE_" + std::to_string(n);
    }
  }

  static void modify_ext_sources(
    amrex::Real time,
    amrex::Real /*dt*/,
    const amrex::MultiFab& state_old,
    const amrex::MultiFab& /*state_new*/,
    std::unique_ptr<amrex::MultiFab>& ext_src,
    const amrex::GeometryData& geomdata,
    const MyProbParm* prob_parm_d)
  {
    /*
    Notes:
      1) ext_src contains sources from velocity forcing coming in.
         This function should add to rather than overwrite ext_src.
      2) Requires "peleLM.user_defined_ext_sources = true" in input file
    */

    auto ext_src_arr = ext_src->arrays();
    auto const& state_old_arr = state_old.const_arrays();

    ParallelFor(
      *ext_src, [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        if (prob_parm_d->ode_qty_test) {
          for (int n = 0; n < NUM_ODE; n++) {
            // Source terms for ODE qty test

            amrex::Real B_n = state_old_arr[box_no](i, j, k, FIRSTODE + n);
            amrex::Real src =
              prob_parm_d->ode_srcstrength * pow(10.0, n + 1) * B_n;
            ext_src_arr[box_no](i, j, k, FIRSTODE + n) += src;
          }
        }

        // Source terms for composition test
        if (prob_parm_d->composition_test) {
          amrex::Real src = 0.0;

          if (time >= prob_parm_d->extRhoYCO2_ts) {
            // Switch on over extRhoYCO2_ramp with a smoothstep (a step
            // when 0), so the source is smooth in time
            amrex::Real ramp = 1.0;
            if (prob_parm_d->extRhoYCO2_ramp > 0.0) {
              const amrex::Real s = amrex::min(
                (time - prob_parm_d->extRhoYCO2_ts) /
                  prob_parm_d->extRhoYCO2_ramp,
                amrex::Real(1.0));
              ramp = s * s * (3.0 - 2.0 * s);
            }
            amrex::Real CO2 =
              state_old_arr[box_no](i, j, k, FIRSTSPEC + CO2_ID);
            src = prob_parm_d->extRhoYCO2 * ramp * CO2;
          }

          ext_src_arr[box_no](i, j, k, FIRSTSPEC + CO2_ID) += src;
          ext_src_arr[box_no](i, j, k, DENSITY) += src;
        }

        // Source terms for the isothermal injection test
        if (prob_parm_d->isothermal_test) {
          const amrex::Real* plo = geomdata.ProbLo();
          const amrex::Real* phi = geomdata.ProbHi();
          const amrex::Real* dx = geomdata.CellSize();
          const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
          amrex::Real r2 = 0.0;
          for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            const amrex::Real x =
              plo[d] + (iv[d] + 0.5) * dx[d] - 0.5 * (plo[d] + phi[d]);
            r2 += x * x;
          }
          const amrex::Real sig = prob_parm_d->iso_sigma;
          const amrex::Real src =
            prob_parm_d->iso_amp * std::exp(-r2 / (2.0 * sig * sig));
          auto eos = pele::physics::PhysicsType::eos();
          amrex::Real hi_cgs[NUM_SPECIES] = {0.0};
          eos.T2Hi(prob_parm_d->T_mean, hi_cgs);
          ext_src_arr[box_no](i, j, k, FIRSTSPEC + CO2_ID) += src;
          ext_src_arr[box_no](i, j, k, DENSITY) += src;
          ext_src_arr[box_no](i, j, k, RHOH) +=
            src * hi_cgs[CO2_ID] * 1.0e-4; // CGS -> MKS conversion
        }
      });
    amrex::Gpu::streamSynchronize();
  }
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
