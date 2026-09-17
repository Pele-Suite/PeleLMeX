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
#include <Utilities.H>

namespace m2c = pele::physics::utilities::mks2cgs;
namespace c2m = pele::physics::utilities::cgs2mks;

AMREX_ENUM(PmfReferenceFrame, lab, flame, fixed_velocity);

struct MyProbParm : public ProbParmDefault
{
  amrex::Real standoff = 0.0;
  amrex::Real pertmag = 0.0004;
  amrex::Real pertlength = 0.008;
  int meanFlowDir = 1;
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm;
  PmfReferenceFrame ref_frame = PmfReferenceFrame::flame;
  amrex::Real moving_frame_velocity = 0.0;
};

struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{
  /**
   * Initial state from the PMF profile at a *physical* position.  Shared by
   * initdata() (uniform grid: physical == AMReX coordinates) and
   * initdata_mapped() (mesh mapping: the AMReX grid is Xi space, so the
   * position, the domain lengths used for the perturbation wavelengths and
   * the cell half-width used to average the PMF over the cell are all taken
   * in physical units from the map).
   */
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void initdata_at(
    int i,
    int j,
    int k,
    amrex::Array4<amrex::Real> const& state,
    AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z),
    AMREX_D_DECL(amrex::Real /*Lx_unused*/, amrex::Real Lx, amrex::Real Ly),
    amrex::Real half_dn,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* pmf_data)
  {
    constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;

    auto eos = pele::physics::PhysicsType::eos(prob_parm.eosparm);
    amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
    amrex::Real molefrac[NUM_SPECIES] = {0.0};
    amrex::Real massfrac[NUM_SPECIES] = {0.0};
    amrex::Real pert = 0.0;

    if (prob_parm.pertmag > 0.0) {
#if (AMREX_SPACEDIM == 2)
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM - 1> lpert{Lx};
      if (prob_parm.pertlength > 0.0) {
        lpert[0] = prob_parm.pertlength;
      }
      pert = prob_parm.pertmag *
             (1.0 * std::sin(2 * Pi * 4 * x / lpert[0]) +
              1.023 * std::sin(2 * Pi * 2 * (x - 0.004598) / lpert[0]) +
              0.945 * std::sin(2 * Pi * 3 * (x - 0.00712435) / lpert[0]) +
              1.017 * std::sin(2 * Pi * 5 * (x - 0.0033) / lpert[0]) +
              0.982 * std::sin(2 * Pi * 5 * (x - 0.014234) / lpert[0]));
    }
    amrex::Real y1 = (y - prob_parm.standoff - half_dn + pert) * 100;
    amrex::Real y2 = (y - prob_parm.standoff + half_dn + pert) * 100;
#elif (AMREX_SPACEDIM == 3)
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM - 1> lpert{Lx, Ly};
      if (prob_parm.pertlength > 0.0) {
        lpert[0] = prob_parm.pertlength;
        lpert[1] = prob_parm.pertlength;
      }

      pert = prob_parm.pertmag *
             (1.0 * std::sin(2 * Pi * 4 * x / lpert[0]) *
                std::sin(2 * Pi * 5 * y / lpert[1]) +
              1.023 * std::sin(2 * Pi * 2 * (x - 0.004598) / lpert[0]) *
                std::sin(2 * Pi * 4 * (y - 0.0053765) / lpert[1]) +
              0.945 * std::sin(2 * Pi * 3 * (x - 0.00712435) / lpert[0]) *
                std::sin(2 * Pi * 3 * (y - 0.02137) / lpert[1]) +
              1.017 * std::sin(2 * Pi * 5 * (x - 0.0033) / lpert[0]) *
                std::sin(2 * Pi * 6 * (y - 0.018) / lpert[1]) +
              0.982 * std::sin(2 * Pi * 5 * (x - 0.014234) / lpert[0]));
    }
    amrex::Real y1 = (z - prob_parm.standoff - half_dn + pert) * 100;
    amrex::Real y2 = (z - prob_parm.standoff + half_dn + pert) * 100;
#endif

    pele::physics::PMF::pmf(pmf_data, y1, y2, pmf_vals);
    state(i, j, k, TEMP) = pmf_vals[0];
    for (int dim = 0; dim < AMREX_SPACEDIM - 1; dim++) {
      state(i, j, k, VELX + dim) = 0.0;
    }
    state(i, j, k, VELX + AMREX_SPACEDIM - 1) = pmf_vals[1] * 1e-2;

#ifdef USE_MANIFOLD_EOS
    for (int n = 0; n < NUM_SPECIES - 1; n++) {
      massfrac[n] = pmf_vals[3 + n];
    }
    amrex::Real rho_temp, tmp1, tmp2;
    eos.PYT2R(tmp1, massfrac, tmp2, rho_temp);
    rho_temp = c2m::Rho(rho_temp);
    state(i, j, k, DENSITY) = rho_temp;
    state(i, j, k, RHOH) = 0.0; // No RhoH transport in manifold model
    for (int n = 0; n < NUM_SPECIES - 1; n++) {
      state(i, j, k, FIRSTSPEC + n) = massfrac[n] * rho_temp;
    }
    state(i, j, k, FIRSTSPEC + NUM_SPECIES - 1) = rho_temp;
#else
    for (int n = 0; n < NUM_SPECIES; n++) {
      molefrac[n] = pmf_vals[3 + n];
    }
    eos.X2Y(molefrac, massfrac);

    amrex::Real P_cgs = m2c::P(prob_parm.P_mean);

    // Density
    amrex::Real rho_cgs = 0.0;
    eos.PYT2R(P_cgs, massfrac, state(i, j, k, TEMP), rho_cgs);
    state(i, j, k, DENSITY) = c2m::Rho(rho_cgs);

    // Enthalpy
    amrex::Real h_cgs = 0.0;
    eos.TY2H(state(i, j, k, TEMP), massfrac, h_cgs);
    state(i, j, k, RHOH) = state(i, j, k, DENSITY) * c2m::H(h_cgs);

    // Species mass
    for (int n = 0; n < NUM_SPECIES; n++) {
      state(i, j, k, FIRSTSPEC + n) = massfrac[n] * state(i, j, k, DENSITY);
    }
#endif
  }

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
      , const amrex::Real Lx = prob_hi[0] - prob_lo[0];
      , const amrex::Real Ly = prob_hi[1] - prob_lo[1]);

    initdata_at(
      i, j, k, state, AMREX_D_DECL(x, y, z), AMREX_D_DECL(0.0, Lx, Ly),
      0.5 * dx[AMREX_SPACEDIM - 1], prob_parm, pmf_data);
  }

  // Mesh mapping: same IC at the physical position of the cell.  Selected
  // automatically (has_initdata_mapped_v<>) when a mapping is active.
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
    pele::physics::PMF::PmfData::DataContainer const* pmf_data)
  {
    AMREX_D_TERM(
      const amrex::Real x = mmap.x_phys_cc(0, i, geomdata);
      , const amrex::Real y = mmap.x_phys_cc(1, j, geomdata);
      , const amrex::Real z = mmap.x_phys_cc(2, k, geomdata););

    AMREX_D_TERM(
      , const amrex::Real Lx = mmap.domain_size_phys(0, geomdata);
      , const amrex::Real Ly = mmap.domain_size_phys(1, geomdata));

    // Physical half-width of the cell in the flame-normal direction.  With
    // the identity map use the grid spacing directly so that unmapped runs
    // stay bit-identical to the legacy initdata() (dx_phys_cc differences
    // two face positions, which is not exactly dx in floating point).
    const int idx_n = AMREX_D_PICK(i, j, k);
    const amrex::Real half_dn =
      (mmap.m_kind == MeshMapEvaluator::Kind::Identity)
        ? 0.5 * geomdata.CellSize()[AMREX_SPACEDIM - 1]
        : 0.5 * mmap.dx_phys_cc(AMREX_SPACEDIM - 1, idx_n, geomdata);

    initdata_at(
      i, j, k, state, AMREX_D_DECL(x, y, z), AMREX_D_DECL(0.0, Lx, Ly), half_dn,
      prob_parm, pmf_data);
  }

  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bcnormal(
    const amrex::Real* /*x[AMREX_SPACEDIM]*/,
    const amrex::CellData<amrex::Real>& /*s_in*/,
    amrex::Real s_ext[NVAR],
    const int idir,
    const int sgn,
    const amrex::Real /*time*/,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* pmf_data)
  {
    const amrex::Real* prob_lo = geomdata.ProbLo();
    amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
    amrex::Real molefrac[NUM_SPECIES] = {0.0};
    amrex::Real massfrac[NUM_SPECIES] = {0.0};

    auto eos = pele::physics::PhysicsType::eos(prob_parm.eosparm);
    if (sgn == 1) {
      pele::physics::PMF::pmf(pmf_data, prob_lo[idir], prob_lo[idir], pmf_vals);

      for (int dim = 0; dim < AMREX_SPACEDIM - 1; dim++) {
        s_ext[VELX + dim] = 0.0;
      }
      if (prob_parm.ref_frame == PmfReferenceFrame::flame) {
        s_ext[VELX + AMREX_SPACEDIM - 1] = pmf_vals[1] * 1e-2;
      } else if (prob_parm.ref_frame == PmfReferenceFrame::fixed_velocity) {
        s_ext[VELX + AMREX_SPACEDIM - 1] = prob_parm.moving_frame_velocity;
      } else {
        s_ext[VELX + AMREX_SPACEDIM - 1] = 0.0;
      }
      s_ext[TEMP] = pmf_vals[0];

#ifdef USE_MANIFOLD_EOS
      for (int n = 0; n < NUM_SPECIES; n++) {
        massfrac[n] = pmf_vals[3 + n];
      }
      amrex::Real P_dummy, rho_cgs;
      eos.PYT2R(P_dummy, massfrac, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;
      s_ext[RHOH] = 0.0;
      for (int n = 0; n < NUM_SPECIES - 1; n++) {
        s_ext[FIRSTSPEC + n] = massfrac[n] * s_ext[DENSITY];
      }
      s_ext[FIRSTSPEC + NUM_SPECIES - 1] = s_ext[DENSITY];
#else
      for (int n = 0; n < NUM_SPECIES; n++) {
        molefrac[n] = pmf_vals[3 + n];
      }
      eos.X2Y(molefrac, massfrac);

      amrex::Real rho_cgs, P_cgs, RhoH_temp;
      P_cgs = prob_parm.P_mean * 10.0;

      eos.PYT2R(P_cgs, massfrac, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;

      eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
      s_ext[RHOH] =
        RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion

      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = massfrac[n] * s_ext[DENSITY];
      }
#endif
    }
  }
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
