#include "PeleLMeX.H"
#include "pelelmex_prob.H"

#include <sys/stat.h>

void
PeleLM::initActiveControl()
{
  amrex::ParmParse pp("active_control");

  pp.query("on", m_ctrl_active);

  if (m_ctrl_active == 0) {
    return;
  }

  amrex::Print() << " Initialization of active control \n";

  pp.query("use_temp", m_ctrl_useTemp);
  pp.query("temperature", m_ctrl_temperature);
  pp.query("tau", m_ctrl_tauControl);
  pp.query("v", m_ctrl_verbose);
  pp.get("height", m_ctrl_h);
  pp.query("changeMax", m_ctrl_changeMax);
  pp.query("velMax", m_ctrl_velMax);
  pp.query("pseudo_gravity", m_ctrl_pseudoGravity);
  pp.query("flow_dir", m_ctrl_flameDir);
  pp.query("AC_history", m_ctrl_AChistory);
  pp.query("npoints_average", m_ctrl_NavgPts);
  pp.query("method", m_ctrl_method);

  // Active control checks
  if ((m_ctrl_useTemp != 0) && (m_ctrl_temperature <= 0.0)) {
    amrex::Error(
      "active_control.temperature MUST be set with "
      "active_control.use_temp = 1");
  }

  if ((m_ctrl_active != 0) && (m_ctrl_tauControl <= 0.0)) {
    amrex::Error("active_control.tau MUST be set when using active_control");
  }

  if (m_ctrl_flameDir > 2) {
    amrex::Error(
      "active_control.flow_dir MUST be 0, 1 or 2 for X, Y and Z resp.");
  }

  // Initialize flow controller
  if (!hasFlowControllerData<ProbParm>::value) {
    amrex::Abort(
      "ProbParm doesn't have a FCData FlowControllerData member variable");
  }

  // Get FlowControllerData from ProbParm if it exists
  FlowControllerData* fcdata_host{nullptr};
  fcdata_host = getFCDataPtr(*prob_parm, hasFlowControllerData<ProbParm>{});

  if (fcdata_host != nullptr) {
    // Resize vector for temporal average
    m_ctrl_time_pts.resize(m_ctrl_NavgPts + 1, -1.0);
    m_ctrl_velo_pts.resize(m_ctrl_NavgPts + 1, -1.0);
    m_ctrl_cntl_pts.resize(m_ctrl_NavgPts + 1, -1.0);

    // Extract data from BC: assumes flow comes in from lo side of ctrl_flameDir
    ProbParm const* lprobparm = prob_parm_d;
    auto const* lpmfdata = pmf_data.device_parm();

    amrex::Gpu::DeviceVector<amrex::Real> s_ext_v(NVAR);
    amrex::Real* s_ext_d = s_ext_v.data();
    amrex::Real x[AMREX_SPACEDIM] = {
      AMREX_D_DECL(Geom(0).ProbLo(0), Geom(0).ProbLo(1), Geom(0).ProbLo(2))};
    x[m_ctrl_flameDir] -= 1.0;
    const int ctrl_flameDir_l = m_ctrl_flameDir;
    const amrex::Real time_l = -1.0;
    const auto geomdata = Geom(0).data();
    amrex::Dim3 begin{0, 0, 0};
    amrex::Dim3 end{1, 1, 1};
    auto fake_state = amrex::Array4<amrex::Real>(nullptr, begin, end, NVAR);

    amrex::Box dumbx({AMREX_D_DECL(0, 0, 0)}, {AMREX_D_DECL(0, 0, 0)});
    amrex::ParallelFor(
      dumbx,
      [x, fake_state, s_ext_d, ctrl_flameDir_l, time_l, geomdata, lprobparm,
       lpmfdata] AMREX_GPU_DEVICE(int /*i*/, int /*j*/, int /*k*/) noexcept {
        const auto s_in = fake_state.cellData(0, 0, 0);
        ProblemSpecificFunctions::bcnormal(
          x, s_in, s_ext_d, ctrl_flameDir_l, 1, time_l, geomdata, *lprobparm,
          lpmfdata);
      });
    amrex::Vector<amrex::Real> s_ext(NVAR);
    amrex::Gpu::copy(
      amrex::Gpu::deviceToHost, s_ext_v.begin(), s_ext_v.end(), s_ext.begin());

    if (m_ctrl_useTemp == 0) {
      // Get the fuel rhoY
      if (fuelID < 0) {
        amrex::Abort(
          "Using activeControl based on fuel mass requires peleLM.fuelName !");
      }

      // Physical transverse area (the fuel-mass integral is physical).
      amrex::Real area_tot = 1.0;
      for (int idim{0}; idim < AMREX_SPACEDIM; ++idim) {
        if (idim != m_ctrl_flameDir) {
          area_tot *= m_map_eval.domain_size_phys(idim, Geom(0).data());
        }
      }

      m_ctrl_scale = area_tot * s_ext[FIRSTSPEC + fuelID];
      m_ctrl_cfix = m_ctrl_h * m_ctrl_scale;
    } else {
      // If using temp, scale is 1.0
      m_ctrl_scale = 1.0;
      m_ctrl_cfix = m_ctrl_h;
    }

    // Extract initial ctrl_V_in from bc
    m_ctrl_V_in = s_ext[m_ctrl_flameDir];
    m_ctrl_V_in_old = m_ctrl_V_in;

    // Fill host ProbParm FCData
    fcdata_host->ctrl_active = 1;
    fcdata_host->ctrl_V_in = m_ctrl_V_in;

    // Update device copy of FCData
    FlowControllerData* fcdata_dev{nullptr};
    fcdata_dev = getFCDataPtr(*prob_parm_d, hasFlowControllerData<ProbParm>{});
    if (fcdata_dev != nullptr) {
      amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, fcdata_host, fcdata_host + 1, fcdata_dev);
    }

    if (m_ctrl_verbose != 0) {
      if (m_ctrl_useTemp != 0) {
        amrex::Print()
          << " Active control based on temperature iso-level activated."
          << " Maintaining the flame at " << m_ctrl_h << " in "
          << m_ctrl_flameDir << " direction. \n";
      } else {
        amrex::Print() << " Active control based on fuel mass activated."
                       << " Maintaining the flame at " << m_ctrl_h << " in "
                       << m_ctrl_flameDir << " direction. \n";
      }
    }
  }
}

void
PeleLM::activeControl(const int is_restart)
{
  if (m_ctrl_active == 0) {
    return;
  }

  // -------------------------------------------
  // Get the current target state (either amount of fuel or T-iso position)
  // -------------------------------------------
  amrex::Real coft = 0.0;
  if (m_ctrl_useTemp == 0) {
    // Compute the integral of the fuel mass in the domain
    coft = MFSum(GetVecOfConstPtrs(getSpeciesVect(AmrNewTime)), fuelID);
    ;

  } else {
    // Get the low T position
    coft = 1.e37;
    getActiveControlLowT(coft);
  }

  // If first time, set old target state
  if (m_ctrl_coftOld < 0.0) {
    m_ctrl_coftOld = coft;
  }

  if (m_nstep == 0) {
    m_ctrl_nfilled = m_ctrl_NavgPts + 1;
  }

  // Manage AC history I/O
  if (is_restart != 0) {
    loadActiveControlHistory();
  }

  m_ctrl_V_in_old = m_ctrl_V_in;
  if (m_dt > 0.0) {
    m_ctrl_V_in += m_dt * m_ctrl_dV;
    // Accumulate the convected distance (integral of V_in dt) for consistent
    // turbulent-inflow marching. Only on forward steps; on restart the value
    // is reconstructed from AC history in loadActiveControlHistory().
    if (is_restart == 0) {
      m_turb_conv_dist += 0.5 * (m_ctrl_V_in_old + m_ctrl_V_in) * m_dt;
    }
  }

  amrex::Real slocal = 0.5 * (m_ctrl_V_in_old + m_ctrl_V_in) -
                       (coft - m_ctrl_coftOld) / (m_dt * m_ctrl_scale);

  // -------------------------------------------
  // Get s_est averaged from previous N steps
  // -------------------------------------------
  // Update m_ctrl_* Vectors if not restarting
  if (is_restart == 0) {
#pragma GCC diagnostic ignored "-Wnull-dereference"
    m_ctrl_time_pts.insert(m_ctrl_time_pts.begin(), m_cur_time);
#pragma GCC diagnostic ignored "-Wnull-dereference"
    m_ctrl_velo_pts.insert(m_ctrl_velo_pts.begin(), m_ctrl_V_in);
#pragma GCC diagnostic ignored "-Wnull-dereference"
    m_ctrl_cntl_pts.insert(m_ctrl_cntl_pts.begin(), coft);
    if (m_ctrl_time_pts.size() > m_ctrl_NavgPts) { // Pop_back only if it's
                                                   // filled
      m_ctrl_time_pts.pop_back();
      m_ctrl_velo_pts.pop_back();
      m_ctrl_cntl_pts.pop_back();
    }
  }

  if (m_ctrl_nfilled <= 0) {
    amrex::Real velIntegral = 0.0;
    for (int n = 1; n <= m_ctrl_NavgPts;
         ++n) { // Piecewise constant velocity over NavgPts last steps
      velIntegral += 0.5 * (m_ctrl_velo_pts[n - 1] + m_ctrl_velo_pts[n]) *
                     (m_ctrl_time_pts[n - 1] - m_ctrl_time_pts[n]);
    }
    m_ctrl_sest =
      (velIntegral -
       (m_ctrl_cntl_pts[0] - m_ctrl_cntl_pts[m_ctrl_NavgPts]) / m_ctrl_scale) /
      (m_ctrl_time_pts[0] - m_ctrl_time_pts[m_ctrl_NavgPts]);

  } else {
    m_ctrl_nfilled -= 1;
    if (m_nstep != 0) {
      m_ctrl_sest = (1.0 - m_ctrl_corr) * m_ctrl_sest + m_ctrl_corr * slocal;
    }
  }

  // -------------------------------------------
  // Compute Vnew
  // -------------------------------------------
  amrex::Real Vnew = 0.0;

  if (m_ctrl_method == 1) { // linear
    amrex::Real vslope =
      2.0 *
      ((m_ctrl_cfix - coft) / (m_ctrl_scale * m_ctrl_tauControl) + m_ctrl_sest -
       m_ctrl_V_in) /
      m_ctrl_tauControl;
    Vnew = m_ctrl_V_in + m_dt * vslope;
  } else if (m_ctrl_method == 2) { // Quadratic 1
    amrex::Real vslope =
      3.0 *
      ((m_ctrl_cfix - coft) / (m_ctrl_scale * m_ctrl_tauControl) + m_ctrl_sest -
       m_ctrl_V_in) /
      m_ctrl_tauControl;
    Vnew =
      m_ctrl_V_in + (m_dt - 0.5 * m_dt * m_dt / m_ctrl_tauControl) * vslope;
  } else if (m_ctrl_method == 3) { // Quadratic 2
    amrex::Real rhs2 =
      2.0 *
      ((m_ctrl_cfix - coft) / (m_ctrl_scale * m_ctrl_tauControl) + m_ctrl_sest -
       m_ctrl_V_in) /
      m_ctrl_tauControl;
    amrex::Real rhs1 = (m_ctrl_sest - m_ctrl_V_in) / m_ctrl_tauControl;
    amrex::Real vt_tay = 3.0 * rhs2 - 2.0 * rhs1;
    amrex::Real vtt_tay = 6.0 * (rhs1 - rhs2) / m_ctrl_tauControl;
    Vnew = m_ctrl_V_in + m_dt * vt_tay + 0.5 * m_dt * m_dt * vtt_tay;
  }

  // Limit Vnew
  amrex::Real dVmax = m_ctrl_changeMax * 1.0;
  amrex::Real dVmin =
    m_ctrl_changeMax * amrex::max<amrex::Real>(1.0, m_ctrl_V_in);
  Vnew = amrex::max<amrex::Real>(Vnew, 0.0);
  Vnew = amrex::Clamp(Vnew, m_ctrl_V_in - dVmin, m_ctrl_V_in + dVmax);
  if (m_ctrl_velMax > 0.0) { // Only limit Vnew to velMax if velMax > 0.0
    Vnew = amrex::min<amrex::Real>(Vnew, m_ctrl_velMax);
  }

  if ((is_restart == 0) && m_nstep > 0) {
    m_ctrl_tBase = m_cur_time;

    // Compute
    m_ctrl_dV = (Vnew - m_ctrl_V_in) / m_dt;
  }

  m_ctrl_coftOld = coft;

  // Get FlowControllerData from ProbParm if it exists
  FlowControllerData* fcdata_h{nullptr};
  FlowControllerData* fcdata_d{nullptr};
  fcdata_h = getFCDataPtr(*prob_parm, hasFlowControllerData<ProbParm>{});
  fcdata_d = getFCDataPtr(*prob_parm_d, hasFlowControllerData<ProbParm>{});

  if ((fcdata_h != nullptr) && (fcdata_d != nullptr)) {
    // Pass dV and ctrl_V_in to FCData
    fcdata_h->ctrl_V_in = m_ctrl_V_in;
    fcdata_h->ctrl_dV = m_ctrl_dV;
    fcdata_h->ctrl_tBase = m_ctrl_tBase;

    // Update Device version
    amrex::Gpu::copy(
      amrex::Gpu::hostToDevice, fcdata_h, fcdata_h + 1, fcdata_d);
  }

  if ((m_ctrl_verbose != 0) && (is_restart == 0)) {
    amrex::Print()
      << "\n------------------------AC CONTROL------------------------ \n";
    amrex::Print() << " |     Time: " << m_cur_time << " -     dt: " << m_dt
                   << "\n";
    amrex::Print() << " | Position: " << coft / m_ctrl_scale
                   << " - Target: " << m_ctrl_cfix / m_ctrl_scale << "\n";
    amrex::Print() << " |    V_new: " << Vnew << " -   V_in: " << m_ctrl_V_in
                   << " - dV: " << m_ctrl_dV << "\n";
    amrex::Print()
      << "---------------------------------------------------------- \n";
  }

  // Append to (or create) AC history file
  if (is_restart == 0) {
    std::ofstream ACfile(
      m_ctrl_AChistory.c_str(), std::ofstream::out | std::ofstream::app);
    amrex::Print(ACfile).SetPrecision(15)
      << m_nstep << "  " << m_cur_time << "  " << m_ctrl_V_in << "  " << slocal
      << "  " << m_ctrl_dV << "  " << m_ctrl_sest << "  " << m_ctrl_coftOld
      << "\n";
  }
}

namespace {
// Physical position of the T_cross crossing between cell (i,j,k) and its
// upstream neighbour along flame_dir, or 1e37 if there is none there.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
ac_crossing_position(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real const> const& T_arr,
  int flame_dir,
  amrex::Real T_cross,
  MeshMapEvaluator const& mmap,
  amrex::GeometryData const& gd) noexcept
{
  if (T_arr(i, j, k, TEMP) <= T_cross) {
    return 1.e37;
  }
  int idx[3] = {i, j, k};
  idx[flame_dir] -= 1;
  const amrex::Real T_lo = T_arr(idx[0], idx[1], idx[2], TEMP);
  if (T_lo >= T_cross) {
    return 1.e37;
  }
  const int ic[3] = {i, j, k};
  const amrex::Real x_hi = mmap.x_phys_cc(flame_dir, ic[flame_dir], gd);
  const amrex::Real x_lo = mmap.x_phys_cc(flame_dir, idx[flame_dir], gd);
  const amrex::Real slope = (T_arr(i, j, k, TEMP) - T_lo) / (x_hi - x_lo);
  return x_lo + (T_cross - T_lo) / slope;
}
} // namespace

void
PeleLM::getActiveControlLowT(amrex::Real& a_coft)
{
  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get t^{n+1} data pointer
    auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
    const auto geomdata = Geom(lev).data();
    const MeshMapEvaluator mmap = m_map_eval;

    // local FC data
    int AC_FlameDir = m_ctrl_flameDir;
    amrex::Real AC_Tcross = m_ctrl_temperature;

    amrex::Real lowT = 1.e37;

    if (lev != finest_level) {
      lowT = amrex::ReduceMin(
        ldata_p->state, *m_coveredMask[lev], 0,
        [geomdata, AC_Tcross, AC_FlameDir, mmap] AMREX_GPU_HOST_DEVICE(
          amrex::Box const& bx, amrex::Array4<amrex::Real const> const& T_arr,
          amrex::Array4<int const> const& covered_arr) -> amrex::Real {
          const auto lo = amrex::lbound(bx);
          const auto hi = amrex::ubound(bx);
          amrex::Real tmp_pos = 1.e37;
          for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
              for (int i = lo.x; i <= hi.x; ++i) {
                if (covered_arr(i, j, k) > 0) {
                  tmp_pos = amrex::min<amrex::Real>(
                    tmp_pos,
                    ac_crossing_position(
                      i, j, k, T_arr, AC_FlameDir, AC_Tcross, mmap, geomdata));
                }
              }
            }
          }
          return tmp_pos;
        });
    } else {
      lowT = amrex::ReduceMin(
        ldata_p->state, 0,
        [geomdata, AC_Tcross, AC_FlameDir, mmap] AMREX_GPU_HOST_DEVICE(
          amrex::Box const& bx,
          amrex::Array4<amrex::Real const> const& T_arr) -> amrex::Real {
          const auto lo = amrex::lbound(bx);
          const auto hi = amrex::ubound(bx);
          amrex::Real tmp_pos = 1.e37;
          for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
              for (int i = lo.x; i <= hi.x; ++i) {
                tmp_pos = amrex::min<amrex::Real>(
                  tmp_pos,
                  ac_crossing_position(
                    i, j, k, T_arr, AC_FlameDir, AC_Tcross, mmap, geomdata));
              }
            }
          }
          return tmp_pos;
        });
    }
    a_coft = amrex::min<amrex::Real>(a_coft, lowT);
  }
  amrex::ParallelDescriptor::ReduceRealMin(a_coft);
}

void
PeleLM::loadActiveControlHistory()
{
  m_ctrl_nfilled = m_ctrl_NavgPts + 1;
  struct stat buffer;
  bool have_history = (stat(m_ctrl_AChistory.c_str(), &buffer) == 0);
  if (have_history) {
    if (m_ctrl_verbose != 0) {
      amrex::Print() << " Setting AC from history from " << m_ctrl_AChistory
                     << "\n";
    }
    std::fstream ACfile(m_ctrl_AChistory.c_str(), std::fstream::in);
    if (ACfile.is_open()) {
      // Reconstruct the convected distance (integral of V_in dt) up to the
      // restart step by trapezoidal quadrature of the (time, V_in) columns, so
      // turbulent-inflow marching resumes smoothly. Assumes the history is
      // contiguous up to m_nstep; since the turbfile is periodic in the
      // convection direction, a small offset would be harmless anyway.
      m_turb_conv_dist = 0.0;
      bool have_prev = false;
      amrex::Real prev_time = 0.0;
      amrex::Real prev_vel = 0.0;
      while (ACfile.good()) {
        int step_io;
        amrex::Real time_io, vel_io, slocal_io, dV_io, s_est_io, coft_old_io;
        ACfile >> step_io >> time_io >> vel_io >> slocal_io >> dV_io >>
          s_est_io >> coft_old_io;
        if (step_io <= m_nstep) {
          if (have_prev && (time_io >= prev_time)) {
            m_turb_conv_dist +=
              0.5 * (prev_vel + vel_io) * (time_io - prev_time);
          }
          prev_time = time_io;
          prev_vel = vel_io;
          have_prev = true;
        }
        if (
          ((m_nstep - step_io) >= 0) &&
          ((m_nstep - step_io) <=
           m_ctrl_NavgPts)) { // Fill the previous step data
          int pos = (m_nstep - step_io);
          m_ctrl_nfilled -= 1;
          m_ctrl_time_pts[pos] = time_io;
          m_ctrl_velo_pts[pos] = vel_io;
          m_ctrl_cntl_pts[pos] = coft_old_io;
        }
        if (step_io == m_nstep) { // Found the right step
          m_ctrl_V_in = vel_io;
          m_ctrl_tBase = time_io;
          m_ctrl_dV = dV_io;
          m_ctrl_sest = s_est_io;
          m_ctrl_coftOld = coft_old_io;
        }
      }
      ACfile.close();
    }
    if (m_ctrl_verbose != 0) {
      amrex::Print() << " AC history arrays: \n";
      for (long int n = 0; n < m_ctrl_time_pts.size(); ++n) {
        amrex::Print() << "  [" << n << "] time: " << m_ctrl_time_pts[n]
                       << ", velo: " << m_ctrl_velo_pts[n]
                       << ", coft: " << m_ctrl_cntl_pts[n] << "\n";
      }
    }
  } else {
    if (m_ctrl_verbose != 0) {
      amrex::Print() << " AC history file " << m_ctrl_AChistory
                     << " not found, restarting from scratch \n";
    }
  }
}
