#ifndef PELELM_PROB_H
#define PELELM_PROB_H

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#include <PeleLMeX_ProblemSpecificFunctions.H>
#include <PeleLMeX_Index.H>
#include <PMF.H>
#include <PMFData.H>
#include <PelePhysics.H>

struct MyProbParm : public ProbParmDefault
{
  amrex::Real standoff = 0.0;
  amrex::Real pertmag = 0.0004;
  amrex::Real pertLz = 0.0;
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
    pele::physics::PMF::PmfData::DataContainer const* pmf_data)
  {
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();
    const amrex::Real* dx = geomdata.CellSize();

    AMREX_D_TERM(
      const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
      , const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
      , const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];);

    AMREX_D_TERM(
      const amrex::Real Lx = prob_hi[0] - prob_lo[0];
      , const amrex::Real Ly = prob_hi[1] - prob_lo[1];
      , const amrex::Real Lz = prob_hi[2] - prob_lo[2]);

    AMREX_D_TERM(
      const amrex::Real xc = prob_lo[0] + 0.5 * Lx;
      , const amrex::Real yc = prob_lo[1] + 0.5 * Ly;
      , const amrex::Real zc = prob_lo[2] + 0.5 * Lz;);

    constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;

    auto eos = pele::physics::PhysicsType::eos();
    amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
    amrex::Real molefrac[NUM_SPECIES] = {0.0};
    amrex::Real massfrac[NUM_SPECIES] = {0.0};
    amrex::Real pert = 0.0;

    amrex::Real radius = std::sqrt(AMREX_D_TERM(
      (x - xc) * (x - xc), +(y - yc) * (y - yc), +(z - zc) * (z - zc)));

    if (prob_parm.pertmag > 0.0) {

#if (AMREX_SPACEDIM == 2)
      amrex::Real angle = std::atan2((y - yc), (x - xc));
      pert = prob_parm.pertmag *
             (0.7 * std::sin(2 * angle) + 0.9 * std::sin(8 * angle) +
              1.4 * std::sin(4 * angle));
    }

    amrex::Real y1 =
      3.6 - (radius - prob_parm.standoff - 1.42 * dx[1] + pert) * 100;
    amrex::Real y2 =
      3.6 - (radius - prob_parm.standoff + 1.42 * dx[1] + pert) * 100;

#elif (AMREX_SPACEDIM == 3)
      amrex::Real angle = std::atan2((y - yc), (x - xc));
      amrex::Real angle2 = std::atan2((x - xc), (z - zc));
      pert = prob_parm.pertmag *
             (0.974 * std::sin(3 * angle) * 0.995 * std::sin(5 * angle2) +
              0.983 * std::sin(11 * (angle - 0.63578)) * 0.982 *
                std::sin(5 * (angle2 - 0.3345)) +
              1.022 * std::sin(7 * angle) * 0.945 * std::sin(2 * angle2));
    }

    amrex::Real y1 =
      3.6 - (radius - prob_parm.standoff - 1.42 * dx[1] + pert) * 100;
    amrex::Real y2 =
      3.6 - (radius - prob_parm.standoff + 1.42 * dx[1] + pert) * 100;
#endif

    pele::physics::PMF::pmf(pmf_data, y1, y1, pmf_vals);

    state(i, j, k, TEMP) = pmf_vals[0];

    for (int n = 0; n < NUM_SPECIES; n++) {
      molefrac[n] = pmf_vals[3 + n];
    }
    eos.X2Y(molefrac, massfrac);

    AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) = 0.0;
                 , state(i, j, k, VELZ) = 0.0);

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
