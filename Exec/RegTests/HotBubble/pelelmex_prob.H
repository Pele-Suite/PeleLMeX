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
  amrex::Real T_mean = 300.0;
  amrex::Real T_bubble = 600.0;
  amrex::Real bubble_rad = 0.005;
  amrex::Real bubble_y0 = 0.01;
  int bubble_is_mix = 0;
  int is_sym = 0;
  // Mesh-mapping scaling factors (read from ConstantMap.scaling_factor).
  // When mesh mapping is active the AMReX grid is Xi-space and physical
  // coordinates are  x_phys = prob_lo + (i+0.5) * dx * fac.  With the
  // defaults (fac = 1) this reduces to the standard AMReX-space IC.
  amrex::Real fac_x = 1.0;
  amrex::Real fac_y = 1.0;
  amrex::Real fac_z = 1.0;
};

struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void initdata_mapped(
    int i,
    int j,
    int k,
    int /*is_incompressible*/,
    amrex::Array4<amrex::Real> const& state,
    amrex::Array4<amrex::Real> const& /*aux*/,
    amrex::GeometryData const& geomdata,
    MeshMapEvaluator const& mmap,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();

    // Convert AMReX indices to *physical* coordinates using the mesh-
    // mapping factors; for the default fac = 1 this is identical to
    // the standard AMReX-coordinate IC.
    AMREX_D_TERM(
      const amrex::Real x = mmap.x_phys_cc(0, i, geomdata);
      , const amrex::Real y = mmap.x_phys_cc(1, j, geomdata);
      , const amrex::Real z = mmap.x_phys_cc(2, k, geomdata););

    AMREX_D_TERM(
      const amrex::Real Lx = (prob_hi[0] - prob_lo[0]) * prob_parm.fac_x;
      , , const amrex::Real Lz = (prob_hi[2] - prob_lo[2]) * prob_parm.fac_z);

    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real massfrac[NUM_SPECIES] = {0.0};

    massfrac[O2_ID] = 0.233;
    massfrac[N2_ID] = 0.767;

    state(i, j, k, TEMP) = prob_parm.T_mean;
    amrex::Real x_c =
      prob_parm.is_sym != 0 ? prob_lo[0] : prob_lo[0] + 0.5 * Lx;
#if AMREX_SPACEDIM == 3
    amrex::Real z_c =
      prob_parm.is_sym != 0 ? prob_lo[2] : prob_lo[2] + 0.5 * Lz;
#endif
    amrex::Real rad = std::sqrt(AMREX_D_TERM(
      (y - prob_parm.bubble_y0) * (y - prob_parm.bubble_y0),
      +(x - x_c) * (x - x_c), +(z - z_c) * (z - z_c)));
    if (prob_parm.bubble_is_mix != 0) {
      if (rad <= prob_parm.bubble_rad) {
        massfrac[O2_ID] = 0.0;
        massfrac[N2_ID] = 1.0;
      }
    } else {
      if (rad <= prob_parm.bubble_rad) {
        state(i, j, k, TEMP) = prob_parm.T_bubble;
      }
    }

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
    MyProbParm const& /*prob_parm*/,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    // NoSlipWallAdiab is ext_dir on velocity.
    AMREX_D_TERM(s_ext[VELX] = 0.0;, s_ext[VELY] = 0.0;, s_ext[VELZ] = 0.0;)
  }
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
