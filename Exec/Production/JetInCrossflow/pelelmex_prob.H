#ifndef PELELM_PROB_H
#define PELELM_PROB_H

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_SPACE.H>

#include <PeleLMeX_ProblemSpecificFunctions.H>
#include <PeleLMeX_Index.H>
#include <PelePhysics.H>
#include <PMFData.H>

struct MyProbParm : public ProbParmDefault
{
  amrex::Real P_mean = 2431800.0;

  // Cross-Flow Conditions
  amrex::Real cf_velocity = 1.1e2;
  amrex::Real cf_temp = 7.5e2;
  amrex::Real cf_massfracs[NUM_SPECIES]{0.0};
  int cf_dir = 2;

  // Jet Conditions
  amrex::Real jet_velocity = 1.1e2;
  amrex::Real jet_temp = 5.5e2;
  amrex::Real jet_massfracs[NUM_SPECIES]{0.0};
  amrex::Real jet_bl_thickness = -1.0;
  amrex::Real jet_rad = 2.5e-4;
  int jet_dir = 0;
  amrex::Real jet_init_time = -1;
  amrex::Real jet_start_time = -1;
  int double_jet = 0;
};

struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void initdata(
    int i,
    int j,
    int k,
    int /*is_incompressible*/,
    amrex::Array4<amrex::Real> const& state,
    amrex::Array4<amrex::Real> const& /*aux*/,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    auto eos = pele::physics::PhysicsType::eos();

    state(i, j, k, TEMP) = prob_parm.cf_temp;

    // Set the initial velocity field
    amrex::RealVect velocity{0.0};
    velocity[prob_parm.cf_dir] = prob_parm.cf_velocity;
    state(i, j, k, VELX) = velocity[0];
    state(i, j, k, VELY) = velocity[1];
    state(i, j, k, VELZ) = velocity[2];

    amrex::Real P_cgs = prob_parm.P_mean * 10.0;

    // Density
    amrex::Real rho_cgs = 0.0;
    eos.PYT2R(P_cgs, prob_parm.cf_massfracs, state(i, j, k, TEMP), rho_cgs);
    state(i, j, k, DENSITY) = rho_cgs * 1.0e3;

    // Enthalpy
    amrex::Real h_cgs = 0.0;
    eos.TY2H(state(i, j, k, TEMP), prob_parm.cf_massfracs, h_cgs);

    state(i, j, k, RHOH) = h_cgs * 1.0e-4 * state(i, j, k, DENSITY);

    // Species mass
    for (int n = 0; n < NUM_SPECIES; n++) {
      state(i, j, k, FIRSTSPEC + n) =
        prob_parm.cf_massfracs[n] * state(i, j, k, DENSITY);
    }
  }

  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bcnormal(
    const amrex::Real x[AMREX_SPACEDIM],
    const amrex::CellData<amrex::Real>& /*s_in*/,
    amrex::Real s_ext[NVAR],
    const int idir,
    const int sgn,
    const amrex::Real time,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();
    const amrex::Real* dx = geomdata.CellSize();
    constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;

    amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
    amrex::Real massfrac[NUM_SPECIES] = {0.0};

    const amrex::Real cf_velocity = prob_parm.cf_velocity;
    const amrex::Real jet_rad = prob_parm.jet_rad;
    const int cf_dir = prob_parm.cf_dir;
    const int jet_dir = prob_parm.jet_dir;

    amrex::Real U_air[AMREX_SPACEDIM] = {0.0};
    amrex::Real U_jet[AMREX_SPACEDIM] = {0.0};
    amrex::Real rho_air = 0.0;
    amrex::Real rho_jet = 0.0;
    amrex::Real radsq = 0.0;
    amrex::Real vel_factor = 0.0;
    amrex::Real comp_factor = 0.0;
    amrex::Real area = 0.0;
    const amrex::Real jet_vel = prob_parm.jet_velocity;

    auto eos = pele::physics::PhysicsType::eos();

    // Find the directions
    int other_dir = 0;
    bool found_dir = false;

    while (!found_dir) {
      if (other_dir == jet_dir or other_dir == cf_dir) {
        other_dir += 1;
      } else {
        found_dir = true;
      }
    }

    // Initialize jet velocity BCs
#if (AMREX_SPACEDIM < 3)
    amrex::Abort("JICF only implemented in 3D");
#endif
    if (cf_dir == 0) {
      U_air[0] = cf_velocity;
    } else if (cf_dir == 1) {
      U_air[1] = cf_velocity;
    } else {
      U_air[2] = cf_velocity;
    }

    if (jet_dir == 0) {
      U_jet[0] = jet_vel;
    } else if (jet_dir == 1) {
      U_jet[1] = jet_vel;
    } else {
      U_jet[2] = jet_vel;
    }

    if (idir == jet_dir and (prob_parm.double_jet == 1 or sgn == 1)) {

      if (prob_parm.jet_bl_thickness > 0.0) {
        // specified BL thickness - use tanh profile at jet boundary
        radsq = x[other_dir] * x[other_dir] + x[cf_dir] * x[cf_dir];
        vel_factor =
          0.5 *
          (1.0 -
           tanh((sqrt(radsq) - jet_rad) / (prob_parm.jet_bl_thickness / 4.0)));
        comp_factor = vel_factor;
      } else {
        // scale inlet conditions by how much of cell is covered by the jet
        radsq = x[other_dir] * x[other_dir] + x[cf_dir] * x[cf_dir];
        if (
          radsq <
          jet_rad + geomdata.CellSize(cf_dir) + geomdata.CellSize(other_dir)) {
          amrex::Real area =
            pele::physics::utilities::rectangle_circle_intersection_area(
              x[other_dir] - 0.5 * geomdata.CellSize(other_dir),
              x[other_dir] + 0.5 * geomdata.CellSize(other_dir),
              x[cf_dir] - 0.5 * geomdata.CellSize(cf_dir),
              x[cf_dir] + 0.5 * geomdata.CellSize(cf_dir), 0.0, 0.0, jet_rad);
          vel_factor =
            area / (geomdata.CellSize(cf_dir) * geomdata.CellSize(other_dir));
          vel_factor = (vel_factor > 1e-9) ? vel_factor : 0.0;
          comp_factor = (vel_factor > 0.5e-9) ? 1.0 : 0.0;
        } else {
          vel_factor = 0.0;
          comp_factor = 0.0;
        }
      }

      amrex::Real time_factor = 0.0;
      if (time < prob_parm.jet_start_time) {
        time_factor = 0.0;
      } else if (time < prob_parm.jet_start_time + prob_parm.jet_init_time) {
        time_factor =
          (time - prob_parm.jet_start_time) / prob_parm.jet_init_time;
      } else {
        time_factor = 1.0;
      }
      vel_factor *= time_factor;
      comp_factor *= time_factor;

      for (int n = 0; n < NUM_SPECIES; n++) {
        massfrac[n] = comp_factor * prob_parm.jet_massfracs[n] +
                      (1.0 - comp_factor) * prob_parm.cf_massfracs[n];
      }

      amrex::Real rho_cgs, P_cgs, RhoH_temp;
      P_cgs = prob_parm.P_mean * 10.0;

      s_ext[TEMP] = comp_factor * prob_parm.jet_temp +
                    (1.0 - comp_factor) * prob_parm.cf_temp;

      eos.PYT2R(P_cgs, massfrac, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;

      eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
      s_ext[RHOH] =
        RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion

      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = massfrac[n] * s_ext[DENSITY];
      }

      // Velocity
      s_ext[VELX] =
        (1 - 0.5 * prob_parm.double_jet) * sgn * vel_factor * U_jet[0];
      s_ext[VELY] =
        (1 - 0.5 * prob_parm.double_jet) * sgn * vel_factor * U_jet[1];
      s_ext[VELZ] =
        (1 - 0.5 * prob_parm.double_jet) * sgn * vel_factor * U_jet[2];
    }

    if ((idir == cf_dir) and (sgn == 1)) {
      // Density
      eos.PYT2R(
        prob_parm.P_mean * 10.0, prob_parm.cf_massfracs, prob_parm.cf_temp,
        rho_air);
      rho_air *= 1.0e3;
      s_ext[DENSITY] = rho_air;
      // Mass fraction
      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = prob_parm.cf_massfracs[n] * s_ext[DENSITY];
      }
      // Temperature
      s_ext[TEMP] = prob_parm.cf_temp;
      amrex::Real RhoH_temp;
      // Enthalpy
      eos.TY2H(s_ext[TEMP], prob_parm.cf_massfracs, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY];
      // Velocity
      s_ext[VELX] += U_air[0];
      s_ext[VELY] += U_air[1];
      s_ext[VELZ] += U_air[2];
    }
  }

  // Zero diffusion coefficients for walls on inflow face so we don't get
  // diffusion through them
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void zero_visc(
    int i,
    int j,
    int k,
    amrex::Array4<amrex::Real> const& beta,
    amrex::GeometryData const& /*geomdata*/,
    amrex::Box const& domainBox,
    const int dir,
    const int beta_comp,
    const int nComp,
    MyProbParm const& prob_parm)
  {
    // We treat species when beta_comp == 0 and nComp == NUM_SPECIES
    // otherwise this routine could be called for other face diffusivity (Temp,
    // velocity, ...)
    if (beta_comp == 0 && nComp == NUM_SPECIES) {
      amrex::IntVect idx{i, j, k};
      if (dir == prob_parm.jet_dir && idx[dir] <= domainBox.smallEnd(dir)) {
        // Just over whole inlet
        for (int n = 0; n < nComp; n++) {
          beta(i, j, k, n) = 0.0;
        }
      }
    }
  }
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
