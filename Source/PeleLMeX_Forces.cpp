#include <PeleLMeX.H>
#include <PeleLMeX_K.H>
#include <PelePhysics.H>

#include <fstream>
#include <limits>
#include <sstream>

namespace c2m = pele::physics::utilities::cgs2mks;

// Return velocity forces scaled by rhoInv
// including grapP term if add_gradP = 1
// including divTau if input amrex::Vector not empty
// a_extRhoExplicit: fraction of the -u extrho external source term included
void
PeleLM::getVelForces(
  const TimeStamp a_time,
  const amrex::Vector<amrex::MultiFab*>& a_divTau,
  const amrex::Vector<amrex::MultiFab*>& a_velForce,
  const int nGrowForce,
  const int add_gradP,
  const amrex::Real a_extRhoExplicit)
{
  BL_PROFILE("PeleLMeX::getVelForces()");
  const int has_divTau = static_cast<int>(!a_divTau.empty());

  for (int lev = 0; lev <= finest_level; ++lev) {
    if (has_divTau != 0) {
      getVelForces(
        a_time, lev, a_divTau[lev], a_velForce[lev], add_gradP,
        a_extRhoExplicit);
    } else {
      getVelForces(
        a_time, lev, nullptr, a_velForce[lev], add_gradP, a_extRhoExplicit);
    }
  }

  // FillPatch forces
  if (nGrowForce > 0) {
    fillpatch_forces(m_cur_time, a_velForce, nGrowForce);
  }
}

void
PeleLM::getVelForces(
  const TimeStamp a_time,
  const int lev,
  amrex::MultiFab* a_divTau,
  amrex::MultiFab* a_velForce,
  const int add_gradP,
  const amrex::Real a_extRhoExplicit)
{

  // Get level data
  // TODO: the 1 here bypass getting halftime vel and return oldtime vel
  auto* ldata_p = getLevelDataPtr(lev, a_time, 1);

  // Get gradp: if m_t_old < 0.0, we are during initialization -> only NewTime
  // data initialized at this point
  auto* ldataGP_p = (m_t_old[lev] < 0.0) ? getLevelDataPtr(lev, AmrNewTime)
                                         : getLevelDataPtr(lev, AmrOldTime);

  const amrex::Real time = getTime(lev, a_time);

  const int is_incomp = m_incompressible;
  const amrex::Real rho_incomp = m_rho;

  const int pseudo_gravity = m_ctrl_pseudoGravity;
  const amrex::Real dV_control = m_ctrl_dV;
  const auto grav = m_gravity;
  const auto gp0 = m_background_gp;
  const int ps_dir = m_ctrl_flameDir;

  auto const& state_ma = ldata_p->state.const_arrays();
  auto const& ext_ma = m_extSource[lev]->const_arrays();
  auto const& force_ma = a_velForce->arrays();

  amrex::ParallelFor(
    *a_velForce,
    [state_ma, ext_ma, force_ma, grav, gp0, ps_dir, is_incomp, rho_incomp,
     pseudo_gravity, dV_control,
     a_extRhoExplicit] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) {
      amrex::Array4<amrex::Real const> vel(state_ma[box_no], VELX);
      amrex::Array4<amrex::Real const> extmom(ext_ma[box_no], VELX);
      amrex::Array4<amrex::Real const> rho(state_ma[box_no], DENSITY);
      amrex::Array4<amrex::Real const> extrho(ext_ma[box_no], DENSITY);
      // Incompressible: no density in the state and no mass source, so
      // point rho/extrho at something and drop the -u extrho term
      amrex::Real extRhoExplicit = a_extRhoExplicit;
      if (is_incomp != 0) {
        rho = vel;
        extrho = extmom;
        extRhoExplicit = 0.0;
      }
      // background gp, pseudo grav, ext sources
      makeVelForce(
        i, j, k, is_incomp, rho_incomp, pseudo_gravity, ps_dir, grav, gp0,
        dV_control, vel, rho, extmom, extrho, extRhoExplicit, force_ma[box_no]);
    });
  amrex::Gpu::streamSynchronize();

  if (add_gradP != 0) {
    if (m_mesh_mapping) {
      // gp is stored in Xi-space (component i = fac_i . grad_phys p_i).
      // When subtracting from the physical-space velocity force we need
      // the physical gradient, hence the 1/fac_i per-component factor.
      auto const& gp_ma = ldataGP_p->gp.const_arrays();
      auto const& fac_ma = m_mesh_map->fac_cc(lev).const_arrays();
      auto const& fm_ma = a_velForce->arrays();
      amrex::ParallelFor(
        *a_velForce, amrex::IntVect(0), AMREX_SPACEDIM,
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
          fm_ma[box_no](i, j, k, n) -=
            gp_ma[box_no](i, j, k, n) / fac_ma[box_no](i, j, k, n);
        });
      amrex::Gpu::streamSynchronize();
    } else {
      amrex::MultiFab::Subtract(
        *a_velForce, ldataGP_p->gp, 0, 0, AMREX_SPACEDIM, 0);
    }
  }
  const int has_divTau = static_cast<int>(a_divTau != nullptr);
  if (has_divTau != 0) {
    amrex::MultiFab::Add(*a_velForce, *a_divTau, 0, 0, AMREX_SPACEDIM, 0);
  }

#ifndef PELE_USE_PLASMA
  if (m_do_turbulent_forcing)
#endif
  {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*a_velForce, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const auto& bx = mfi.tilebox();
      amrex::FArrayBox DummyFab(bx, 1);
      const auto& rho_arr = (m_incompressible != 0)
                              ? DummyFab.array()
                              : ldata_p->state.const_array(mfi, DENSITY);
      const auto& force_arr = a_velForce->array(mfi);
#ifdef PELE_USE_PLASMA
      const auto& rhoY_arr = (m_incompressible != 0)
                               ? DummyFab.array()
                               : ldata_p->state.const_array(mfi, FIRSTSPEC);
      const auto& phiV_arr = ldata_p->state.const_array(mfi, PHIV);
      const auto& ne_arr = ldata_p->state.const_array(mfi, NE);
      addLorentzVelForces(lev, bx, time, force_arr, rhoY_arr, phiV_arr, ne_arr);
      if (m_do_turbulent_forcing)
#endif
      {
        // Add forcing terms to maintain turbulence
        // note: if m_incompressible == 0 then m_rho is unused by
        // addTurbVelForce
        turb_forcing.addTurbVelForces(
          geom[lev].data(), bx, time, force_arr, rho_arr, is_incomp,
          rho_incomp);
      }
    }
  }

  if (is_incomp != 0) {
    a_velForce->mult(1.0 / rho_incomp, 0, AMREX_SPACEDIM, 0);
  } else {
    amrex::ParallelFor(
      *a_velForce, amrex::IntVect(0), AMREX_SPACEDIM,
      [force_ma, state_ma] AMREX_GPU_DEVICE(
        int box_no, int i, int j, int k, int n) noexcept {
        amrex::Array4<amrex::Real const> rho(state_ma[box_no], DENSITY);
        force_ma[box_no](i, j, k, n) /= rho(i, j, k);
      });
    amrex::Gpu::streamSynchronize();
  }
}

namespace {
// Spark profiles: a cosine kernel edge in space and linear power ramps in
// time.  Both are odd about the nominal edge / switch time with slope
// 1/(2 w) there, so smoothing a top-hat with them keeps its zeroth and first
// moments: the solution differs from the sharp-source one by O(w^2).  Both
// are compact; the cosine edge is C1, and the linear ramp (C0) is integrated
// exactly by the trapezoidal source once steps land on its kinks
// (limitDtForSparks).

// Smoothed top-hat spark kernel: 1 inside the radius, falling to 0 over an
// edge (radius - r) / delta in [-pi/2, pi/2] as 0.5 (1 + sin) (a sharp
// top-hat when delta = 0).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
spark_kernel(amrex::Real r, amrex::Real radius, amrex::Real delta) noexcept
{
  if (delta <= 0.0) {
    return (r < radius) ? 1.0 : 0.0;
  }
  const amrex::Real s = (radius - r) / delta;
  const amrex::Real smax = 0.5 * amrex::Math::pi<amrex::Real>();
  if (s <= -smax) {
    return 0.0;
  }
  if (s >= smax) {
    return 1.0;
  }
  return 0.5 * (1.0 + std::sin(s));
}

// Linear switch: 0 before t - tau, 1 after t + tau
amrex::Real
spark_ramp(amrex::Real t, amrex::Real tau)
{
  return amrex::Clamp(0.5 * (1.0 + t / tau), 0.0, 1.0);
}

// Power shape in time, normalised to integrate to t1 - t0: switched on at
// t0 and off at t1 with linear ramps of half-width tau (sharp when tau = 0).
amrex::Real
spark_power_shape(
  amrex::Real t, amrex::Real t0, amrex::Real t1, amrex::Real tau)
{
  if (tau <= 0.0) {
    return (t >= t0 && t < t1) ? 1.0 : 0.0;
  }
  return spark_ramp(t - t0, tau) - spark_ramp(t - t1, tau);
}

// Integral of spark_power_shape from -infinity to t, from the integral of
// the ramp: 0 before -tau, (x + tau)^2 / (4 tau) across it, x after
amrex::Real
spark_power_integral(
  amrex::Real t, amrex::Real t0, amrex::Real t1, amrex::Real tau)
{
  auto ramp_integral = [tau](amrex::Real x) -> amrex::Real {
    if (tau <= 0.0) {
      return amrex::max(x, 0.0);
    }
    if (x <= -tau) {
      return 0.0;
    }
    if (x >= tau) {
      return x;
    }
    return (x + tau) * (x + tau) / (4.0 * tau);
  };
  return ramp_integral(t - t0) - ramp_integral(t - t1);
}

// Physical distance from the centre of cell (i,j,k) to the spark location.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
spark_distance(
  int i,
  int j,
  int k,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& x0,
  MeshMapEvaluator const& mmap,
  amrex::GeometryData const& gd) noexcept
{
  const int ic[3] = {i, j, k};
  amrex::Real d2 = 0.0;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    const amrex::Real dxc = mmap.x_phys_cc(d, ic[d], gd) - x0[d];
    d2 += dxc * dxc;
  }
  amrex::ignore_unused(j, k);
  return std::sqrt(d2);
}

} // namespace

amrex::Real
PeleLM::sparkVolume(int n) const
{
  // Volume of the sphere of the spark radius clipped to the physical domain
  // (so a spark on a symmetry plane gets the matching fraction of its
  // energy).  Chords along x are exact; the remaining directions use midpoint
  // quadrature.  RZ (x = r): dV = 2 pi r dr dz, integrated exactly in r.
  const amrex::GeometryData gd = geom[0].data();
  const amrex::Real radius = m_spark_radius[n];
  const auto& xs = m_spark_location[n];
  constexpr int nq = (AMREX_SPACEDIM == 3) ? 512 : 4096;

  amrex::Real lo[AMREX_SPACEDIM];
  amrex::Real hi[AMREX_SPACEDIM];
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    lo[d] = std::max(
      xs[d] - radius, m_map_eval.x_phys_from_xi(d, gd.ProbLo()[d], gd));
    hi[d] = std::min(
      xs[d] + radius, m_map_eval.x_phys_from_xi(d, gd.ProbHi()[d], gd));
    if (hi[d] <= lo[d]) {
      return 0.0;
    }
  }

  const bool is_rz = geom[0].IsRZ();
  const amrex::Real hy = (hi[1] - lo[1]) / nq;
#if AMREX_SPACEDIM == 3
  const int nz = nq;
  const amrex::Real hz = (hi[2] - lo[2]) / nq;
#else
  const int nz = 1;
  const amrex::Real hz = 1.0;
#endif
  amrex::Real vol = 0.0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < nq; ++j) {
      amrex::Real rho2 = 0.0;
      const amrex::Real dy = lo[1] + (j + 0.5) * hy - xs[1];
      rho2 += dy * dy;
#if AMREX_SPACEDIM == 3
      const amrex::Real dz = lo[2] + (k + 0.5) * hz - xs[2];
      rho2 += dz * dz;
#endif
      if (rho2 >= radius * radius) {
        continue;
      }
      const amrex::Real half = std::sqrt(radius * radius - rho2);
      const amrex::Real xa = std::max(xs[0] - half, lo[0]);
      const amrex::Real xb = std::min(xs[0] + half, hi[0]);
      if (xb > xa) {
        vol += is_rz ? amrex::Math::pi<amrex::Real>() * (xb * xb - xa * xa)
                     : (xb - xa);
      }
    }
  }
  return vol * hy * hz;
}

amrex::Real
PeleLM::limitDtForSparks(amrex::Real a_dt)
{
  amrex::Real dt = a_dt;
  for (int n = 0; n < m_n_sparks; ++n) {
    const amrex::Real t0 = m_spark_time[n];
    const amrex::Real t1 = t0 + m_spark_duration[n];
    // Freeze a ramp given in steps from the step size once the ramp is
    // within two steps, so it does not follow dt while it is sampled.
    if (m_spark_tau[n] < 0.0) {
      const amrex::Real tau = m_spark_ramp_steps[n] * dt;
      if (m_cur_time + 2.0 * dt >= t0 - tau) {
        m_spark_tau[n] = tau;
        if (m_cur_time > t0 - tau) {
          amrex::Warning(
            m_spark[n] + ": ramp frozen after it should have started; part "
                         "of the switch-on is lost (start the spark later)");
        }
        if (m_spark_verbose > 0) {
          amrex::Print() << m_spark[n] << " ramp half-width frozen at " << tau
                         << " s\n";
        }
      }
    }
    // Land a step on each end of the linear ramps, where the power has a
    // kink: the trapezoidal source is then exact
    const amrex::Real tau = m_spark_tau[n];
    if (tau <= 0.0) {
      continue;
    }
    // The kinks are not in time order when tau > duration / 2: take the
    // nearest one ahead.  Within rounding of m_cur_time counts as landed.
    const amrex::Real kinks[4] = {t0 - tau, t0 + tau, t1 - tau, t1 + tau};
    amrex::Real left = std::numeric_limits<amrex::Real>::max();
    for (const amrex::Real kink : kinks) {
      const amrex::Real tol = amrex::max(
        1.0e-10 * dt,
        16.0 * std::numeric_limits<amrex::Real>::epsilon() * std::abs(kink));
      if (kink - m_cur_time > tol) {
        left = amrex::min(left, kink - m_cur_time);
      }
    }
    // Ensure ~O(dt) steps by checking a little in advance
    if (2.0 * dt > left && left > dt) {
      dt = 0.5 * left;
    } else {
      dt = amrex::min(dt, left);
    }
  }
  return dt;
}

void
PeleLM::writeSparkCheckpoint(const std::string& a_dir) const
{
  if (m_n_sparks == 0 || !amrex::ParallelDescriptor::IOProcessor()) {
    return;
  }
  std::ofstream os(a_dir + "/Sparks");
  if (!os.good()) {
    amrex::FileOpenFailed(a_dir + "/Sparks");
  }
  os.precision(17);
  // One line per spark: name, energy frozen at ignition and ramp half-width
  // frozen before the ramp (< 0: not yet), energy deposited so far
  for (int n = 0; n < m_n_sparks; ++n) {
    os << m_spark[n] << " " << m_spark_energy[n] << " " << m_spark_tau[n] << " "
       << m_spark_deposited[n] << "\n";
  }
}

void
PeleLM::readSparkCheckpoint(const std::string& a_dir)
{
  const std::string file = a_dir + "/Sparks";
  if (m_n_sparks == 0 || !amrex::FileExists(file)) {
    return;
  }
  amrex::Vector<char> buf;
  amrex::ParallelDescriptor::ReadAndBcastFile(file, buf);
  std::istringstream is(buf.dataPtr());
  std::string name;
  amrex::Real energy = 0.0;
  amrex::Real tau = 0.0;
  amrex::Real deposited = 0.0;
  while (is >> name >> energy >> tau >> deposited) {
    for (int n = 0; n < m_n_sparks; ++n) {
      if (m_spark[n] != name) {
        continue;
      }
      // Only values derived during the run are state: a given energy or
      // ramp time comes from the inputs, which may have changed on restart
      if (m_spark_temp[n] > 0.0 && energy >= 0.0) {
        m_spark_energy[n] = energy;
      }
      if (m_spark_ramp_time[n] < 0.0 && tau >= 0.0) {
        m_spark_tau[n] = tau;
      }
      m_spark_deposited[n] = deposited;
    }
  }
}

void
PeleLM::addSpark(const TimeStamp a_timestamp, const int a_is_initIter)
{
  BL_PROFILE("PeleLM::addSpark()");
  const amrex::Real time = getTime(0, a_timestamp);
  const bool verb = m_spark_verbose > 1;
  // The t^n evaluation of a regular step
  const bool step_start = (a_timestamp == AmrOldTime) && (a_is_initIter == 0);
  for (int n = 0; n < m_n_sparks; ++n) {
    amrex::Real tau = m_spark_tau[n];
    if (tau < 0.0) {
      // Not frozen yet (limitDtForSparks freezes it before the ramp): the
      // spark is still far off, so this only sizes the activity window
      tau = m_spark_ramp_steps[n] * m_dt;
    }
    const amrex::Real t0 = m_spark_time[n];
    const amrex::Real t1 = t0 + m_spark_duration[n];
    // Power shape at this time (a point value). Without ext_sources_SDC the
    // t^n source is held over the step: use the exact step average instead,
    // so the deposited energy stays exact
    const bool step_average = step_start && !m_ext_sources_SDC;
    const amrex::Real p = step_average
                            ? (spark_power_integral(time + m_dt, t0, t1, tau) -
                               spark_power_integral(time, t0, t1, tau)) /
                                m_dt
                            : spark_power_shape(time, t0, t1, tau);
    // A step starting here may still overlap the spark even if p(t^n) = 0:
    // the energy must then be set (temperature input) and counted
    const bool in_step =
      step_start && time + m_dt > t0 - tau && time < t1 + tau;
    if (p <= 0.0 && !in_step) {
      if (verb) {
        amrex::Print() << m_spark[n] << " not active \n";
      }
      continue;
    }

    if (m_spark_volume[n] < 0.0) {
      m_spark_volume[n] = sparkVolume(n);
    }
    if (m_spark_volume[n] <= 0.0) {
      amrex::Warning(m_spark[n] + " not in domain!");
      continue;
    }

    const amrex::Real radius = m_spark_radius[n];
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> x0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      x0[d] = m_spark_location[n][d];
    }
    const MeshMapEvaluator mmap = m_map_eval;

    // Edge width from the grid: width_cells x the physical cell size (largest
    // direction) at the spark centre, on the finest level whose grids hold
    // the centre.  The edge then always spans the same number of cells of
    // the grid resolving the spark; the normalisation below keeps the energy
    // independent of the width.
    int spark_lev = 0;
    amrex::Real dx_spark = 0.0;
    for (int lev = finest_level; lev >= 0; --lev) {
      const amrex::GeometryData gd = geom[lev].data();
      const amrex::Box& domain = geom[lev].Domain();
      amrex::IntVect idx;
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        const amrex::Real xi = mmap.xi_from_x_phys(d, x0[d], gd);
        idx[d] = amrex::max(
          domain.smallEnd(d),
          amrex::min(
            domain.bigEnd(d), static_cast<int>(std::floor(
                                (xi - gd.ProbLo()[d]) / gd.CellSize()[d]))));
      }
      if (lev == 0 || grids[lev].contains(idx)) {
        spark_lev = lev;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
          dx_spark = amrex::max(dx_spark, mmap.dx_phys_cc(d, idx[d], gd));
        }
        break;
      }
    }
    const amrex::Real delta = (m_spark_width[n] > 0.0)
                                ? m_spark_width[n]
                                : m_spark_width_cells[n] * dx_spark;

    // Spark energy given as a temperature: frozen at ignition as the energy
    // to take the gas inside the radius from its state then to the spark
    // temperature, sphere_volume x <rho (h(T_spark, Y) - h(T, Y))>_sphere.
    // Frozen so the rate does not decay as the gas heats.  The temperature
    // only sets the energy: the gas left in the kernel ends up hotter, as
    // heated gas expands out of it.
    if (m_spark_energy[n] < 0.0) {
      // {sum_in q dV, sum_in dV, sum w q dV, sum w dV}; the kernel-weighted
      // pair is the fallback when no cell centre lies inside the radius.
      amrex::Real sums[4] = {0.0, 0.0, 0.0, 0.0};
      auto const* leosparm = eos_parms.device_parm();
      for (int lev = 0; lev <= finest_level; ++lev) {
        const amrex::GeometryData gd = geom[lev].data();
        auto const& statema =
          getLevelDataPtr(lev, a_timestamp)->state.const_arrays();
        auto const& volma = m_cellVol[lev].const_arrays();
        const bool has_mask = lev != finest_level;
        auto const& maskma = has_mask ? m_coveredMask[lev]->const_arrays()
                                      : amrex::MultiArray4<int const>{};
        const amrex::Real spark_temp = m_spark_temp[n];
        auto r = amrex::ParReduce(
          amrex::TypeList<
            amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
            amrex::ReduceOpSum>{},
          amrex::TypeList<amrex::Real, amrex::Real, amrex::Real, amrex::Real>{},
          m_cellVol[lev], amrex::IntVect(0),
          [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept
            -> amrex::GpuTuple<
              amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
            const amrex::Real r_c = spark_distance(i, j, k, x0, mmap, gd);
            const amrex::Real w = spark_kernel(r_c, radius, delta);
            const amrex::Real in = (r_c < radius) ? 1.0 : 0.0;
            amrex::Real dV = volma[box_no](i, j, k);
            if (has_mask) {
              dV *= static_cast<amrex::Real>(maskma[box_no](i, j, k));
            }
            if (w <= 0.0 && in <= 0.0) {
              return {0.0, 0.0, 0.0, 0.0};
            }
            auto eos = pele::physics::PhysicsType::eos(leosparm);
            auto const& s = statema[box_no];
            const amrex::Real rho = s(i, j, k, DENSITY);
            amrex::Real Y[NUM_SPECIES];
            for (int ns = 0; ns < NUM_SPECIES; ++ns) {
              Y[ns] = s(i, j, k, FIRSTSPEC + ns) / rho;
            }
            amrex::Real h_spark = 0.0;
            amrex::Real h_gas = 0.0;
            eos.TY2H(spark_temp, Y, h_spark);
            eos.TY2H(s(i, j, k, TEMP), Y, h_gas);
            const amrex::Real q = rho * c2m::H(h_spark - h_gas);
            return {in * q * dV, in * dV, w * q * dV, w * dV};
          });
        sums[0] += amrex::get<0>(r);
        sums[1] += amrex::get<1>(r);
        sums[2] += amrex::get<2>(r);
        sums[3] += amrex::get<3>(r);
      }
      amrex::ParallelDescriptor::ReduceRealSum(sums, 4);
      amrex::Real q_mean = 0.0;
      if (sums[1] > 0.0) {
        q_mean = sums[0] / sums[1];
      } else if (sums[3] > 0.0) {
        q_mean = sums[2] / sums[3];
      }
      m_spark_energy[n] = amrex::max(0.0, q_mean * m_spark_volume[n]);
      if (q_mean <= 0.0) {
        amrex::Warning(
          m_spark[n] + ": gas inside the radius is already at or above the "
                       "spark temperature; no energy added");
      }
      if (time > t0 - tau) {
        amrex::Warning(
          m_spark[n] + ": energy evaluated after the spark started (restart "
                       "during the spark?); it is based on the current state");
      }
      if (m_spark_verbose > 0) {
        amrex::Print() << m_spark[n] << " energy [J";
#if AMREX_SPACEDIM == 2
        amrex::Print() << (geom[0].IsRZ() ? "" : "/m");
#endif
        amrex::Print() << "]: " << m_spark_energy[n] << "\n";
      }
    }
    if (m_spark_energy[n] <= 0.0) {
      continue;
    }

    // Deposited energy as the scalar update sees it: trapezoidal with
    // ext_sources_SDC, else the step average over the step
    if (step_start) {
      const amrex::Real p_end =
        m_ext_sources_SDC ? spark_power_shape(time + m_dt, t0, t1, tau) : p;
      m_spark_deposited[n] +=
        m_spark_energy[n] / m_spark_duration[n] * m_dt * 0.5 * (p + p_end);
      if (time + m_dt >= t1 + tau && m_spark_verbose > 0) {
        amrex::Print() << m_spark[n]
                       << " deposited energy: " << m_spark_deposited[n]
                       << ", target " << m_spark_energy[n] << ", rel. error "
                       << m_spark_deposited[n] / m_spark_energy[n] - 1.0
                       << "\n";
      }
    }
    if (p <= 0.0) {
      continue;
    }

    // Discrete integral of the kernel over the composite (uncovered) grid,
    // so the deposited power is exactly energy / duration whatever the
    // resolution and edge width.
    amrex::Vector<amrex::MultiFab> kernel(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
      kernel[lev].define(
        grids[lev], dmap[lev], 1, 0, amrex::MFInfo(), Factory(lev));
      auto const& kma = kernel[lev].arrays();
      const amrex::GeometryData gd = geom[lev].data();
      amrex::ParallelFor(
        kernel[lev],
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
          kma[box_no](i, j, k) =
            spark_kernel(spark_distance(i, j, k, x0, mmap, gd), radius, delta);
        });
    }
    const amrex::Real kernel_sum = MFSum(amrex::GetVecOfConstPtrs(kernel), 0);
    if (kernel_sum <= 0.0) {
      amrex::Warning(m_spark[n] + " not resolved by any cell!");
      continue;
    }
    // Source [W/m^3]: power at this time x kernel / kernel integral
    const amrex::Real coef =
      m_spark_energy[n] / m_spark_duration[n] * p / kernel_sum;
    if (verb) {
      amrex::Print() << m_spark[n] << " active on level " << spark_lev
                     << ", edge width " << delta << ", power fraction " << p
                     << ", sphere volume " << m_spark_volume[n]
                     << ", kernel integral " << kernel_sum << "\n";
    }

    for (int lev = 0; lev <= finest_level; ++lev) {
      auto const& extma = m_extSource[lev]->arrays();
      auto const& kma = kernel[lev].const_arrays();
      amrex::ParallelFor(
        kernel[lev],
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
          const amrex::Real w = kma[box_no](i, j, k);
          if (w > 0.0) {
            extma[box_no](i, j, k, RHOH) += coef * w;
          }
        });
    }
  }
  amrex::Gpu::streamSynchronize();
}

// Manifold model - dissipation rate sources for variances
void
PeleLM::addScalarVarianceSources(const TimeStamp a_timestamp)
{
  BL_PROFILE("PeleLM::addScalarVarianceSources");
  // no scalar dissipation sources if not using a manifold model
#ifndef USE_MANIFOLD_EOS
  amrex::ignore_unused(a_timestamp);
#else

  if (m_add_variance_sources) {
    auto const& leosparm = eos_parms.host_parm();

    // Determine if we have any scalar variances that need sources added
    // Could be moved elsewhere to not do every timestep
    int nvariances = 0;
    int var_of_scalar = -1;
    for (int n = 0; n < MANIFOLD_DIM; ++n) {
      if (leosparm.is_variance_of[n] >= 0) {
        if (!m_do_les) {
          amrex::Abort(
            "PeleLM::addScalarVarianceSources(): cannot add a "
            "scalar dissipation without an active LES model");
        }
        nvariances += 1;
        var_of_scalar = FIRSTSPEC + leosparm.is_variance_of[n];
      }
    }

    if (nvariances > 1) {
      amrex::Abort(
        "PeleLM::addScalarVarianceSources(): currently we only support "
        "manifold models with 0 or 1 variances");
    } else if (nvariances > 0) {

      // Compute scalar gradients (no need to average down here)
      constexpr int do_avgDown = 0;
      auto bcRecScalar = fetchBCRecArray(var_of_scalar, 1);
      constexpr int nGrow = 0; // No need for ghost face on fluxes
      amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> grad_fc(
        finest_level + 1);
      for (int lev = 0; lev <= finest_level; ++lev) {
        const auto& ba = grids[lev];
        const auto& factory = Factory(lev);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          grad_fc[lev][idim].define(
            amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)),
            dmap[lev], 1, nGrow, amrex::MFInfo(), factory);
          grad_fc[lev][idim].setVal(0.0); // Required?
        }
      }
      getDiffusionOp()->computeGradient(
        GetVecOfArrOfPtrs(grad_fc), {},
        GetVecOfConstPtrs(getStateVect(a_timestamp)), {}, bcRecScalar[0],
        do_avgDown, var_of_scalar);

      constexpr amrex::Real fact =
        0.5 / static_cast<amrex::Real>(AMREX_SPACEDIM);
      const amrex::Real C_chi = m_les_c_chi;
      const amrex::Real ScInv = m_Schmidt_inv;

      // Add in Production and Dissipation source terms for subfilter variances
      for (int lev = 0; lev <= finest_level; ++lev) {

        auto* ldata_p = getLevelDataPtr(lev, a_timestamp);

        // Require the turbulent viscosity to be pre-computed
        // it always is stored at AmrOldTime, so we just use that
        // We need cell-centered mu_t but have it at faces
        // The simple interpolation below probably isn't valid for EB
#ifdef AMREX_USE_EB
        amrex::Abort(
          "PeleLM::addScalarVarianceSources(): this is not supported with "
          "EB");
#endif
        // l_scale will also need modification for EB
        const amrex::Real vol = AMREX_D_TERM(
          geom[lev].CellSize(0), *geom[lev].CellSize(1),
          *geom[lev].CellSize(2));
        const amrex::Real l_scale =
          AMREX_D_PICK(vol, std::sqrt(vol), std::cbrt(vol));
        const amrex::Real inv_l_scale2 = 1.0 / (l_scale * l_scale);

        for (int n = 0; n < MANIFOLD_DIM; ++n) {
          if (leosparm.is_variance_of[n] >= 0) {

            AMREX_D_TERM(
              auto const& mut_arr_x =
                m_leveldata_old[lev]->visc_turb_fc[0].const_arrays();
              , auto const& mut_arr_y =
                  m_leveldata_old[lev]->visc_turb_fc[1].const_arrays();
              , auto const& mut_arr_z =
                  m_leveldata_old[lev]->visc_turb_fc[2].const_arrays();)
            AMREX_D_TERM(
              auto const& gx = grad_fc[lev][0].const_arrays();
              , auto const& gy = grad_fc[lev][1].const_arrays();
              , auto const& gz = grad_fc[lev][2].const_arrays();)
            auto const& extma = m_extSource[lev]->arrays();
            auto const& statema = ldata_p->state.const_arrays();

            amrex::ParallelFor(
              *m_extSource[lev], [extma, statema, n, C_chi, ScInv, inv_l_scale2,
                                  mut_arr_x, gx, mut_arr_y, gy
#if (AMREX_SPACEDIM == 3)
                                  ,
                                  mut_arr_z, gz
#endif
            ] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
                // Subfilter Scalar Dissipation: Linear Relaxation model
                // rho chi_sgs = C_chi * mu_t / Delta^2 * Variance
                const amrex::Real mu_t =
                  fact *
                  (AMREX_D_TERM(
                    mut_arr_x[bx](i, j, k) + mut_arr_x[bx](i + 1, j, k),
                    +mut_arr_y[bx](i, j, k) + mut_arr_y[bx](i, j + 1, k),
                    +mut_arr_z[bx](i, j, k) + mut_arr_z[bx](i, j, k + 1)));

                extma[bx](i, j, k, FIRSTSPEC + n) -=
                  C_chi * mu_t * inv_l_scale2 *
                  statema[bx](i, j, k, FIRSTSPEC + n);

                // Production term (w/ Smagorinsky closure for turbulent flux)
                // -2 (rho <u_j C> - rho <u_j><C>) d<C>/dx_j
                // = 2 *mu_t/Sc_t * d<C>/dx_j * d<C>/dx_j
                const amrex::Real mu_grad2 =
                  fact *
                  (AMREX_D_TERM(
                    mut_arr_x[bx](i, j, k) * gx[bx](i, j, k) * gx[bx](i, j, k) +
                      mut_arr_x[bx](i + 1, j, k) * gx[bx](i + 1, j, k) *
                        gx[bx](i + 1, j, k),
                    +mut_arr_y[bx](i, j, k) * gy[bx](i, j, k) *
                        gy[bx](i, j, k) +
                      mut_arr_y[bx](i, j + 1, k) * gy[bx](i, j + 1, k) *
                        gy[bx](i, j + 1, k),
                    +mut_arr_z[bx](i, j, k) * gz[bx](i, j, k) *
                        gz[bx](i, j, k) +
                      mut_arr_z[bx](i, j, k + 1) * gz[bx](i, j, k + 1) *
                        gz[bx](i, j, k + 1)));

                extma[bx](i, j, k, FIRSTSPEC + n) += 2.0 * ScInv * mu_grad2;
              });
          }
        }
      }
      amrex::Gpu::streamSynchronize();
    }
  }
#endif
}

// Calculate additional external sources (soot, radiation, user defined, etc.)
// at t^n. With ext_sources_SDC, also keep S^n and its frozen part in diffData
// for the SDC re-evaluations.
void
PeleLM::getExternalSources(
  const int is_initIter,
  const PeleLM::TimeStamp a_timestamp_old,
  const PeleLM::TimeStamp a_timestamp_new,
  const std::unique_ptr<AdvanceDiffData>& diffData)
{
  if (!m_ext_sources_SDC) {
    assembleExternalSources(
      is_initIter, a_timestamp_old, a_timestamp_new, nullptr);
    return;
  }

  auto& frozen = diffData->extSrc_frozen;
  // m_extSource is zero on entry: snapshot the frozen part on its own
  if (addFrozenExternalSources(is_initIter, a_timestamp_old)) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto& ext = *m_extSource[lev];
      frozen.emplace_back(
        grids[lev], dmap[lev], NVAR, ext.nGrowVect(), amrex::MFInfo(),
        *m_factory[lev]);
      amrex::MultiFab::Copy(frozen[lev], ext, 0, 0, NVAR, ext.nGrowVect());
      ext.setVal(0.0);
    }
  }
  assembleExternalSources(
    is_initIter, a_timestamp_old, a_timestamp_new, &frozen);
  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto& ext = *m_extSource[lev];
    diffData->extSrc_n.emplace_back(
      grids[lev], dmap[lev], NVAR, ext.nGrowVect(), amrex::MFInfo(),
      *m_factory[lev]);
    amrex::MultiFab::Copy(
      diffData->extSrc_n[lev], ext, 0, 0, NVAR, ext.nGrowVect());
    // Filled by the first updateExternalSourcesSDC
    diffData->extSrc_np1.emplace_back(
      grids[lev], dmap[lev], NVAR, ext.nGrowVect(), amrex::MFInfo(),
      *m_factory[lev]);
  }
}

// Spray advances its own particles (MKD), soot integrates its source over dt
// internally and radiation needs a global solve, so these are only evaluated
// once per step, from t^n. Returns whether any of them is active.
bool
PeleLM::addFrozenExternalSources(
  const int is_initIter, const PeleLM::TimeStamp a_timestamp)
{
  amrex::ignore_unused(is_initIter, a_timestamp);
  bool added = false;
#ifdef PELE_USE_SPRAY
  if (is_initIter == 0) {
    SprayMKD(m_cur_time, m_dt);
    added = added || do_spray_particles;
  }
#endif
#ifdef PELE_USE_SOOT
  if (do_soot_solve) {
    computeSootSource(a_timestamp, m_dt);
    added = true;
  }
#endif
#ifdef PELE_USE_RADIATION
  if (do_rad_solve) {
    BL_PROFILE_VAR("PeleLM::advance::rad", PLM_RAD);
    computeRadSource(a_timestamp);
    BL_PROFILE_VAR_STOP(PLM_RAD);
    added = true;
  }
#endif
  return added;
}

// Add the external sources evaluated from the a_timestamp_eval state into
// m_extSource. a_frozen: nullptr computes the frozen sources, otherwise its
// (possibly empty) stored snapshot is added instead.
void
PeleLM::assembleExternalSources(
  const int is_initIter,
  const PeleLM::TimeStamp a_timestamp_eval,
  const PeleLM::TimeStamp a_timestamp_new,
  const amrex::Vector<amrex::MultiFab>* a_frozen)
{
  if (m_n_sparks > 0) {
    addSpark(a_timestamp_eval, is_initIter);
  }
  if (a_frozen == nullptr) {
    addFrozenExternalSources(is_initIter, a_timestamp_eval);
  } else if (!a_frozen->empty()) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      amrex::MultiFab::Add(
        *m_extSource[lev], (*a_frozen)[lev], 0, 0, NVAR,
        m_extSource[lev]->nGrowVect());
    }
  }
  addScalarVarianceSources(a_timestamp_eval);

  if (m_user_defined_ext_sources) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      auto* ldata_p_eval = getLevelDataPtr(lev, a_timestamp_eval);
      auto* ldata_p_new = getLevelDataPtr(lev, a_timestamp_new);
      ProblemSpecificFunctions::modify_ext_sources(
        getTime(lev, a_timestamp_eval), m_dt, ldata_p_eval->state,
        ldata_p_new->state, m_extSource[lev], geom[lev].data(), prob_parm_d);
    }
  }
}

// ext_sources_SDC: S^{n+1,k} from the new state into diffData->extSrc_np1,
// and m_extSource <- 0.5 (S^n + S^{n+1,k})
void
PeleLM::updateExternalSourcesSDC(
  const std::unique_ptr<AdvanceDiffData>& diffData)
{
  BL_PROFILE("PeleLMeX::updateExternalSourcesSDC()");
  for (int lev = 0; lev <= finest_level; ++lev) {
    m_extSource[lev]->setVal(0.0);
  }
  assembleExternalSources(0, AmrNewTime, AmrNewTime, &diffData->extSrc_frozen);
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto& ext = *m_extSource[lev];
    std::swap(ext, diffData->extSrc_np1[lev]);
    amrex::MultiFab::LinComb(
      ext, 0.5, diffData->extSrc_n[lev], 0, 0.5, diffData->extSrc_np1[lev], 0,
      0, NVAR, ext.nGrowVect());
  }
}
