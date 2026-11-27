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
  amrex::Real T_mean = 298.0;
  amrex::Real rvort = 0.001;
  amrex::Real xvort = 0.0;
  amrex::Real yvort = 0.0;
  amrex::Real forcevort = 3.0;
};

struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{
  // All boundaries are walls. The fill calls bcnormal on any face with an
  // ext_dir component (the wall-normal velocity), and the default aborts.
  // s_ext arrives zeroed, which is already the wall value.
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bcnormal(
    const amrex::Real* /*x[AMREX_SPACEDIM]*/,
    const amrex::CellData<amrex::Real>& /*s_in*/,
    amrex::Real /*s_ext*/[NVAR],
    const int /*idir*/,
    const int /*sgn*/,
    const amrex::Real /*time*/,
    amrex::GeometryData const& /*geomdata*/,
    MyProbParm const& /*prob_parm*/,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
  }

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
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();
    const amrex::Real* dx = geomdata.CellSize();

    const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
    const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];

    const amrex::Real Lx = prob_hi[0] - prob_lo[0];
    const amrex::Real xc = prob_lo[0] + 0.5 * Lx;

    const amrex::Real deltax = x - prob_parm.xvort;
    const amrex::Real deltay = y - prob_parm.yvort;
    const amrex::Real d_sq = deltax * deltax + deltay * deltay;
    const amrex::Real r_sq = prob_parm.rvort * prob_parm.rvort;
    AMREX_D_TERM(
      const amrex::Real u_vort =
        -prob_parm.forcevort * deltay / r_sq * exp(-d_sq / r_sq / 2.);
      , const amrex::Real v_vort =
          prob_parm.forcevort * deltax / r_sq * exp(-d_sq / r_sq / 2.);
      , const amrex::Real w_vort = 0.0;)

    AMREX_D_TERM(state(i, j, k, VELX) = u_vort;, state(i, j, k, VELY) = v_vort;
                 , state(i, j, k, VELZ) = w_vort;)

    if (!(is_incompressible == 0)) {
      return;
    }

    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real massfrac[NUM_SPECIES] = {0.0};
    massfrac[O2_ID] = 0.233;
    massfrac[N2_ID] = 0.767;

    if (std::abs(x - xc) < 0.1 * Lx) {
      massfrac[O2_ID] = 0.283;
      massfrac[N2_ID] = 0.717;
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
  }
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
