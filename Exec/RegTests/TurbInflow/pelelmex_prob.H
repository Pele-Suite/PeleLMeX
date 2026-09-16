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
  amrex::Real meanFlowMag = 0.0;
  int meanFlowDir = 1;
  int bothSides = 0;
  // Mesh mapping: geomdata's ProbLo/ProbHi are Xi-space extents while the
  // coordinates handed to bcnormal are physical, so the physical domain
  // length is the Xi length times the ConstantMap factor (1 for the other
  // maps, which preserve the total length, and without mapping).
  amrex::Real fac_x = 1.0;
  amrex::Real fac_y = 1.0;
  amrex::Real fac_z = 1.0;
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
    amrex::GeometryData const& /*geomdata*/,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    auto eos = pele::physics::PhysicsType::eos();

    amrex::Real massfrac[NUM_SPECIES] = {0.0};
    massfrac[O2_ID] = 0.233;
    massfrac[N2_ID] = 0.767;

    AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) = 0.0;
                 , state(i, j, k, VELZ) = 0.0);
    state(i, j, k, VELX + prob_parm.meanFlowDir) = prob_parm.meanFlowMag;

    if (is_incompressible != 0) {
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
  }

  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bcnormal(
    const amrex::Real x[AMREX_SPACEDIM],
    const amrex::CellData<amrex::Real>& /*s_in*/,
    amrex::Real s_ext[NVAR],
    const int /*idir*/,
    const int sgn,
    const amrex::Real /*time*/,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();

    // Physical extents and centre (x[] is physical; see fac_* in MyProbParm)
    AMREX_D_TERM(
      const amrex::Real Lx = (prob_hi[0] - prob_lo[0]) * prob_parm.fac_x;
      , const amrex::Real Ly = (prob_hi[1] - prob_lo[1]) * prob_parm.fac_y;
      , const amrex::Real Lz = (prob_hi[2] - prob_lo[2]) * prob_parm.fac_z);

    AMREX_D_TERM(
      const amrex::Real x_c = prob_lo[0] + 0.5 * Lx;
      , const amrex::Real y_c = prob_lo[1] + 0.5 * Ly;
      , const amrex::Real z_c = prob_lo[2] + 0.5 * Lz);

    amrex::Real massfrac[NUM_SPECIES] = {0.0};

    auto eos = pele::physics::PhysicsType::eos();

    // When using PELE_USE_TURBINFLOW, s_ext comes in with
    // turbulence data. Only add the mean flow here.
    if (prob_parm.bothSides != 0) {
      if (sgn == 1) {
        s_ext[prob_parm.meanFlowDir] += std::abs(prob_parm.meanFlowMag);
      } else {
        s_ext[prob_parm.meanFlowDir] -= std::abs(prob_parm.meanFlowMag);
      }
    } else {
      s_ext[prob_parm.meanFlowDir] += prob_parm.meanFlowMag;
    }

    massfrac[O2_ID] = 0.233;
    massfrac[N2_ID] = 0.767;

    amrex::Real rad = 0.0;
    if (prob_parm.meanFlowDir == 0) {
      rad =
        std::sqrt((x[1] - y_c) * (x[1] - y_c) + (x[2] - z_c) * (x[2] - z_c));
    } else if (prob_parm.meanFlowDir == 1) {
      rad =
        std::sqrt((x[0] - x_c) * (x[0] - x_c) + (x[2] - z_c) * (x[2] - z_c));
    } else if (prob_parm.meanFlowDir == 2) {
      rad =
        std::sqrt((x[0] - x_c) * (x[0] - x_c) + (x[1] - y_c) * (x[1] - y_c));
    }

    if (rad < 0.15 * Lx) {
      massfrac[O2_ID] = 0.5;
      massfrac[N2_ID] = 0.5;
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
  }
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
