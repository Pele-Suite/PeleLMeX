#ifndef PeleLMEF_K_H
#define PeleLMEF_K_H

#include <PeleLMeX_EF_Constants.H>
#include <PelePhysics.H>
#include <AMReX_Slopes_K.H>
#include <PeleLMeX_EOS_Extension.H>
#include <mechanism.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getKappaE(
  int i,
  int j,
  int k,
  int isKeTabulated,
  const amrex::Real& cstKe,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxinv,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& phiV,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<amrex::Real> const& Ke) noexcept
{
  if (!isKeTabulated) {
    Ke(i, j, k) = cstKe;
  } else {
    auto eos = pele::physics::PhysicsType::eos();

    // Get rho & Y from rhoY
    amrex::Real rho = 0.0;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      rho += rhoY(i, j, k, n);
    }
    amrex::Real rhoinv = 1.0 / rho;
    amrex::Real y[NUM_SPECIES] = {0.0};
    for (int n = 0; n < NUM_SPECIES; ++n) {
      y[n] = rhoY(i, j, k, n) * rhoinv;
    }

    // Get the reduced electric field
    amrex::Real Wbar = 0.0;
    eos.Y2WBAR(y, Wbar);
    amrex::Real Ngas = rho * Na / Wbar * 1000.0; // CGS -> MKS conversion
    amrex::Real EnormSq = AMREX_D_TERM(
      0.5 * (phiV(i + 1, j, k) - phiV(i - 1, j, k)) * dxinv[0] * 0.5 *
        (phiV(i + 1, j, k) - phiV(i - 1, j, k)) * dxinv[0],
      +0.5 * (phiV(i, j + 1, k) - phiV(i, j - 1, k)) * dxinv[1] * 0.5 *
        (phiV(i, j + 1, k) - phiV(i, j - 1, k)) * dxinv[1],
      +0.5 * (phiV(i, j, k + 1) - phiV(i, j, k - 1)) * dxinv[2] * 0.5 *
        (phiV(i, j, k + 1) - phiV(i, j, k - 1)) * dxinv[2]);
    amrex::Real Enorm = std::sqrt(EnormSq);
    amrex::Real redE = Enorm / Ngas * 1e21; // Td conversion
    amrex::ignore_unused(redE);             // Not used by the linear fit below

    amrex::Real C = (T(i, j, k) - 298.0) / (2140.0 - 298.0);

    // Linear values along C axis at ~1Td
    amrex::Real keLow = 0.8;
    amrex::Real keHigh = 0.035;
    amrex::Real kappaC = keLow + C * (keHigh - keLow);
    Ke(i, j, k) = kappaC;
  }
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getKappaSp(
  int i,
  int j,
  int k,
  amrex::Real const mwt[NUM_SPECIES],
  amrex::GpuArray<amrex::Real, NUM_SPECIES> a_zk,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<amrex::Real> const& rhoD,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<amrex::Real> const& Ksp) noexcept
{
  amrex::Real rho = 0.0;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    rho += rhoY(i, j, k, n);
  }
  int ionCnt = 0;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    if (a_zk[n] != 0.0) {
      Ksp(i, j, k, ionCnt) = rhoD(i, j, k, n) * mwt[n] * 0.001 * a_zk[n] /
                             (rho * PP_RU_MKS * T(i, j, k));
      if (n == E_ID)
        Ksp(i, j, k, ionCnt) = 0.0; // Set the E species mobility to zero.
                                    // Electron are treated separately.
      ionCnt += 1;
    }
  }
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getDiffE(
  int i,
  int j,
  int k,
  const amrex::Real& factor,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<const amrex::Real> const& Ke,
  amrex::Array4<amrex::Real> const& De) noexcept
{
  De(i, j, k) = Ke(i, j, k) * T(i, j, k) * factor;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
ef_edge_state_extdir(
  int i,
  int j,
  int k,
  int dir,
  const bool on_lo,
  const bool on_hi,
  amrex::Array4<amrex::Real const> const& state,
  amrex::Array4<amrex::Real const> const& effvel) noexcept
{
  // Default behavior: id_l -1 in dir
  // Handle the BCs
  // Need -1 in id_l and id_h in dir on low Dirichlet BC
  // Need nothing on high Dirichlet BC
  int id_l[3] = {i, j, k};
  int id_h[3] = {i, j, k};
  if (!on_lo) {
    if (!on_hi) {
      id_l[dir] -= 1;
    }
  } else {
    id_l[dir] -= 1;
    id_h[dir] -= 1;
  }

  amrex::Real state_ed;
  if (effvel(i, j, k) > smallVel) {
    state_ed = state(id_l[0], id_l[1], id_l[2]);
  } else if (effvel(i, j, k) < -smallVel) {
    state_ed = state(id_h[0], id_h[1], id_h[2]);
  } else {
    state_ed = 0.5 * (state(id_l[0], id_l[1], id_l[2]) +
                      state(id_h[0], id_h[1], id_h[2]));
  }
  return state_ed;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
ef_edge_state_2ndO_extdir(
  int i,
  int j,
  int k,
  int dir,
  const bool on_lo,
  const bool on_hi,
  const bool extdir_or_ho_lo,
  const bool extdir_or_ho_hi,
  int domlo,
  int domhi,
  amrex::Array4<amrex::Real const> const& state,
  amrex::Array4<amrex::Real const> const& effvel,
  const amrex::Box& /*box*/) noexcept
{
  // Default behavior: id_l -1 in dir
  // Handle the BCs
  // Need -1 in id_l and id_h in dir on low Dirichlet BC
  // Need nothing on high Dirichlet BC
  int id_l[3] = {i, j, k};
  int id_h[3] = {i, j, k};
  if (!on_lo) {
    if (!on_hi) {
      id_l[dir] -= 1;
    }
  } else {
    id_l[dir] -= 1;
    id_h[dir] -= 1;
  }

  int order = 2;

  amrex::Real state_ed;
  // Just zero out edge state on hi/lo external boundaries (these fluxes are
  // updated later)
  if (on_hi || on_lo) {
    state_ed = 0.0;
  } else {
    amrex::Real state_pls = state(id_h[0], id_h[1], id_h[2]);
    if (dir == 0) {
      state_pls -= 0.5 * amrex_calc_xslope_extdir(
                           id_h[0], id_h[1], id_h[2], 0, order, state,
                           extdir_or_ho_lo, extdir_or_ho_hi, domlo, domhi);
    } else if (dir == 1) {
      state_pls -= 0.5 * amrex_calc_yslope_extdir(
                           id_h[0], id_h[1], id_h[2], 0, order, state,
                           extdir_or_ho_lo, extdir_or_ho_hi, domlo, domhi);
#if (AMREX_SPACEDIM == 3)
    } else {
      state_pls -= 0.5 * amrex_calc_zslope_extdir(
                           id_h[0], id_h[1], id_h[2], 0, order, state,
                           extdir_or_ho_lo, extdir_or_ho_hi, domlo, domhi);
#endif
    }
    amrex::Real state_mns = state(id_l[0], id_l[1], id_l[2]);
    if (dir == 0) {
      state_mns += 0.5 * amrex_calc_xslope_extdir(
                           id_l[0], id_l[1], id_l[2], 0, order, state,
                           extdir_or_ho_lo, extdir_or_ho_hi, domlo, domhi);
    } else if (dir == 1) {
      state_mns += 0.5 * amrex_calc_yslope_extdir(
                           id_l[0], id_l[1], id_l[2], 0, order, state,
                           extdir_or_ho_lo, extdir_or_ho_hi, domlo, domhi);
#if (AMREX_SPACEDIM == 3)
    } else {
      state_mns += 0.5 * amrex_calc_zslope_extdir(
                           id_l[0], id_l[1], id_l[2], 0, order, state,
                           extdir_or_ho_lo, extdir_or_ho_hi, domlo, domhi);
#endif
    }
    if (effvel(i, j, k) > smallVel) {
      state_ed = state_mns;
    } else if (effvel(i, j, k) < -smallVel) {
      state_ed = state_pls;
    } else {
      state_ed = 0.5 * (state_pls + state_mns);
    }
  }
  return state_ed;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
cen2edg_upwind(
  int i,
  int j,
  int k,
  int dir,
  int ncomp,
  const bool on_lo,
  const bool on_hi,
  amrex::Array4<const amrex::Real> const& ueff,
  amrex::Array4<const amrex::Real> const& cfab,
  amrex::Array4<amrex::Real> const& efab) noexcept
{
  // Default behavior: id_l -1 in dir
  // Handle the BCs
  // Need -1 in id_l and id_h in dir on low Dirichlet BC
  // Need nothing on high Dirichlet BC
  int id_l[3] = {i, j, k};
  int id_h[3] = {i, j, k};
  if (!on_lo) {
    if (!on_hi) {
      id_l[dir] -= 1;
    }
  } else {
    id_l[dir] -= 1;
    id_h[dir] -= 1;
  }

  for (int n = 0; n < ncomp; ++n) {
    if (ueff(i, j, k) > smallVel) {
      efab(i, j, k, n) = cfab(id_l[0], id_l[1], id_l[2], n);
    } else if (ueff(i, j, k) < -smallVel) {
      efab(i, j, k, n) = cfab(id_h[0], id_h[1], id_h[2], n);
    } else {
      efab(i, j, k, n) = 0.5 * (cfab(id_l[0], id_l[1], id_l[2], n) +
                                cfab(id_h[0], id_h[1], id_h[2], n));
    }
  }
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
reactionRateRhoY_EF(
  int i,
  int j,
  int k,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& rhoH,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<const amrex::Real> const& nE,
  amrex::Array4<amrex::Real> const& rhoYdot,
  amrex::Array4<amrex::Real> const& nEdot) noexcept
{
  // Get rho from rhoY. Is there a better way ?
  amrex::Real rho = 0.0;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    rho += rhoY(i, j, k, n);
  }
  amrex::Real rhoinv = 1.0 / rho;
  amrex::Real y[NUM_SPECIES] = {0.0};
  for (int n = 0; n < NUM_SPECIES; ++n) {
    y[n] = rhoY(i, j, k, n) * rhoinv;
  }

  // Get Y[e] from nE
  auto eos = pele::physics::PhysicsType::eos();
  amrex::Real mw[NUM_SPECIES] = {0.0};
  eos.molecular_weight(mw);
  y[E_ID] = nE(i, j, k) / Na * mw[E_ID] * 0.001 * rhoinv;

  // Get T from Y/H. Was done like that before.
  amrex::Real Tloc = T(i, j, k);
  amrex::Real H =
    rhoH(i, j, k) * rhoinv * 1.0e4; // Include MKS -> CGS conversion
  eos.HY2T(H, y, Tloc);

  // Get wdot. Don't like the temporary wdot there ...
  rho = rho * 0.001; // rho MKS -> CGS
  amrex::Real wdot[NUM_SPECIES] = {0.0};
  eos.RTY2WDOT(rho, Tloc, y, wdot);
  for (int n = 0; n < NUM_SPECIES; ++n) {
    rhoYdot(i, j, k, n) = wdot[n] * 1000.0; // CGS -> MKS conversion
  }
  nEdot(i, j, k) = wdot[E_ID] * Na * 1.0e6;
  rhoYdot(i, j, k, E_ID) = 0.0;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
addLorentzForce(
  int i,
  int j,
  int k,
  amrex::GpuArray<int, 3> const bxlo,
  amrex::GpuArray<int, 3> const bxhi,
  const amrex::Real& /*time*/,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
  amrex::GpuArray<amrex::Real, NUM_SPECIES> const a_zk,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& ne,
  amrex::Array4<const amrex::Real> const& phiV,
  amrex::Array4<amrex::Real> const& force) noexcept
{
  // Electric field. Use 1-sided stencil in the force ghost cells.
  amrex::Real EFx;
  if (i == bxlo[0]) {
    EFx = -1.0 / dx[0] * (phiV(i + 1, j, k) - phiV(i, j, k));
  } else if (i == bxhi[0]) {
    EFx = -1.0 / dx[0] * (phiV(i, j, k) - phiV(i - 1, j, k));
  } else {
    EFx = -0.5 / dx[0] * (phiV(i + 1, j, k) - phiV(i - 1, j, k));
  }
  amrex::Real EFy;
  if (j == bxlo[1]) {
    EFy = -1.0 / dx[1] * (phiV(i, j + 1, k) - phiV(i, j, k));
  } else if (j == bxhi[1]) {
    EFy = -1.0 / dx[1] * (phiV(i, j, k) - phiV(i, j - 1, k));
  } else {
    EFy = -0.5 / dx[1] * (phiV(i, j + 1, k) - phiV(i, j - 1, k));
  }
#if (AMREX_SPACEDIM == 3)
  amrex::Real EFz;
  if (k == bxlo[2]) {
    EFz = -1.0 / dx[2] * (phiV(i, j, k + 1) - phiV(i, j, k));
  } else if (k == bxhi[2]) {
    EFz = -1.0 / dx[2] * (phiV(i, j, k) - phiV(i, j, k - 1));
  } else {
    EFz = -0.5 / dx[2] * (phiV(i, j, k + 1) - phiV(i, j, k - 1));
  }
#endif
  for (int n = 0; n < NUM_SPECIES; ++n) {
    force(i, j, k, 0) += a_zk[n] * rhoY(i, j, k, n) * EFx;
    force(i, j, k, 1) += a_zk[n] * rhoY(i, j, k, n) * EFy;
#if (AMREX_SPACEDIM == 3)
    force(i, j, k, 2) += a_zk[n] * rhoY(i, j, k, n) * EFz;
#endif
  }
  force(i, j, k, 0) -= ne(i, j, k) * elemCharge * EFx;
  force(i, j, k, 1) -= ne(i, j, k) * elemCharge * EFy;
#if (AMREX_SPACEDIM == 3)
  force(i, j, k, 2) -= ne(i, j, k) * elemCharge * EFz;
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
initElecNeutral(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& rho,
  amrex::Array4<amrex::Real> const& rhoY,
  amrex::Array4<amrex::Real> const& rhoH,
  amrex::Array4<amrex::Real> const& temp,
  amrex::Array4<amrex::Real> const& nE,
  ProbParm const& prob_parm)
{
  auto eos = pele::physics::PhysicsType::eos();

  // Get species charge/mass
  amrex::Real zk[NUM_SPECIES] = {0.0};
  pele::physics::eos::charge_mass(zk);
  for (int n = 0; n < NUM_SPECIES; ++n) {
    zk[n] *= 1000.0; // CGS->MKS
  }

  // Get mass fractions
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  amrex::Real rho_inv = 1.0 / rho(i, j, k);
  for (int n = 0; n < NUM_SPECIES; ++n) {
    massfrac[n] = rhoY(i, j, k, n) * rho_inv;
  }

  // Set such that mixture is electro-neutral
  // by adding nE or remove ions if needed
  // First remove Y_nE
  massfrac[E_ID] = 0.0;
  amrex::Real sum = 0.0;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    massfrac[n] = std::max(amrex::Real(0.0), massfrac[n]);
    if (n != N2_ID)
      sum += massfrac[n];
  }
  massfrac[N2_ID] = 1.0 - sum;

  // Change ions massfrac where charge already < 0.0
  amrex::Real chargeDist = 0.0;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    chargeDist += massfrac[n] * zk[n];
  }
  if (chargeDist < 0.0) {
    amrex::Real sum_pos_ion = 0.0;
    amrex::Real mean_zk_pos_ion = 0.0;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      if (zk[n] > 0.0) {
        sum_pos_ion += massfrac[n];
        mean_zk_pos_ion += massfrac[n] * zk[n];
      }
    }
    if (sum_pos_ion > 0.0) {
      for (int n = 0; n < NUM_SPECIES; ++n) {
        if (zk[n] > 0.0) {
          massfrac[N2_ID] += massfrac[n] / sum_pos_ion * chargeDist /
                             mean_zk_pos_ion * sum_pos_ion;
          massfrac[n] -= massfrac[n] / sum_pos_ion * chargeDist /
                         mean_zk_pos_ion * sum_pos_ion;
        }
      }
    }
  }
  // Add Electrons
  chargeDist = 0.0;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    chargeDist += massfrac[n] * zk[n];
  }
  nE(i, j, k) = std::max(1.0e-24, chargeDist / elemCharge);

  // We keep the temperature constant
  // recompute rho
  amrex::Real P_cgs = prob_parm.P_mean * 10.0;
  amrex::Real rho_cgs = 0.0;
  eos.PYT2R(P_cgs, massfrac, temp(i, j, k), rho_cgs);
  rho(i, j, k) = rho_cgs * 1.0e3;

  // recompute enthalpy
  amrex::Real h_cgs = 0.0;
  eos.TY2H(temp(i, j, k), massfrac, h_cgs);
  rhoH(i, j, k) = h_cgs * 1.0e-4 * rho(i, j, k);

  // Species mass
  for (int n = 0; n < NUM_SPECIES; ++n) {
    rhoY(i, j, k, n) = massfrac[n] * rho(i, j, k);
  }

  nE(i, j, k) *= rho(i, j, k);
}
#endif
