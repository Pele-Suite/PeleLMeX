#ifndef PeleLMeX_K_H
#define PeleLMeX_K_H

#include <mechanism.H>
#include <PelePhysics.H>
#include <PeleLMeX_MeshMapEvaluator.H>

namespace m2c = pele::physics::utilities::mks2cgs;
namespace c2m = pele::physics::utilities::cgs2mks;

template <typename EOSType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
getTransportCoeff(
  const int i,
  const int j,
  const int k,
  const bool do_fixed_Le,
  const bool do_fixed_Pr,
  const bool do_soret,
  amrex::Real LeInv,
  amrex::Real PrInv,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<amrex::Real> const& rhoDi,
  amrex::Array4<amrex::Real> const& rhotheta,
  amrex::Array4<amrex::Real> const& lambda,
  amrex::Array4<amrex::Real> const& mu,
  pele::physics::transport::TransParm<
    pele::physics::PhysicsType::eos_type,
    pele::physics::PhysicsType::transport_type> const* trans_parm,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm) noexcept
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);
  amrex::Real mwtinv[NUM_SPECIES] = {0.0};
  eos.inv_molecular_weight(mwtinv);

  // Get rho & Y from rhoY
  amrex::Real rho = 0.0;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    rho += rhoY(i, j, k, n);
  }
  const amrex::Real rhoinv = 1.0 / rho;
  amrex::Real y[NUM_SPECIES] = {0.0};
  for (int n = 0; n < NUM_SPECIES; ++n) {
    y[n] = rhoY(i, j, k, n) * rhoinv;
  }

  amrex::Real Wbar = 0.0;
  eos.Y2WBAR(y, Wbar);

  const amrex::Real rho_cgs = m2c::Rho(rho); // MKS -> CGS conversion
  amrex::Real rhoDi_cgs[NUM_SPECIES] = {0.0};
  amrex::Real lambda_cgs = 0.0;
  amrex::Real mu_cgs = 0.0;
  amrex::Real dummy_xi = 0.0;
  amrex::Real chi_loc[NUM_SPECIES] = {0.0};
  amrex::Real Tloc = T(i, j, k);

  constexpr bool get_xi = false; // always ignore bulk viscosity
  constexpr bool get_mu = true;  // always get dynamic viscosity
  const bool get_lam = !do_fixed_Pr;
  const bool get_Ddiag = !do_fixed_Le;
  const bool get_chi = do_soret;
  auto trans = pele::physics::PhysicsType::transport();
  trans.transport(
    get_xi, get_mu, get_lam, get_Ddiag, get_chi, Tloc, rho_cgs, y, rhoDi_cgs,
    chi_loc, mu_cgs, dummy_xi, lambda_cgs, trans_parm);

  // Do CGS -> MKS conversions
  mu(i, j, k) = c2m::Mu(mu_cgs);
  if (do_fixed_Pr && do_fixed_Le) { // fixed Pr and Le, transport all dependent
                                    // on visc
    amrex::Real cpmix_cgs = 0.0;
    eos.TY2Cp(T(i, j, k), y, cpmix_cgs);
    lambda(i, j, k) = PrInv * c2m::Cp(cpmix_cgs) * mu(i, j, k);
    const amrex::Real ScInv = PrInv * LeInv;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      rhoDi(i, j, k, n) = ScInv * mu(i, j, k);
    }
  } else if (do_fixed_Pr) { // fixed Pr, still use species diffs
    amrex::Real cpmix_cgs = 0.0;
    eos.TY2Cp(T(i, j, k), y, cpmix_cgs);
    lambda(i, j, k) = PrInv * c2m::Cp(cpmix_cgs) * mu(i, j, k);
    for (int n = 0; n < NUM_SPECIES; ++n) {
      rhoDi(i, j, k, n) = rhoDi_cgs[n] * Wbar * mwtinv[n] * 1.0e-1;
    }
  } else if (do_fixed_Le) { // fixed Le, still use thermal cond
    lambda(i, j, k) = c2m::Lambda(lambda_cgs);
    amrex::Real cpmix_cgs = 0.0;
    eos.TY2Cp(T(i, j, k), y, cpmix_cgs);
    for (int n = 0; n < NUM_SPECIES; ++n) {
      rhoDi(i, j, k, n) = lambda_cgs * 1.0e-1 * LeInv / cpmix_cgs;
    }
  } else { // full MA model, with Soret if needed
    lambda(i, j, k) = c2m::Lambda(lambda_cgs);
    for (int n = 0; n < NUM_SPECIES; ++n) {
      rhoDi(i, j, k, n) = rhoDi_cgs[n] * Wbar * mwtinv[n] * 1.0e-1;
      if (do_soret) {
        rhotheta(i, j, k, n) = -rhoDi_cgs[n] * chi_loc[n] * 1.0e-1;
      }
    }
    if (do_soret) {
      // Soret coefficient scaling, see Howarth, Day, Pitsch & Aspden
#if defined(H2_ID)
      rhotheta(i, j, k, H2_ID) *= 0.664;
#endif
#if defined(H_ID)
      rhotheta(i, j, k, H_ID) *= 0.58;
#endif
#if !defined(H2_ID) && !defined(H_ID)
      amrex::Abort(
        "Running with Soret without light species, waste of time "
        "and memory...");
#endif
    }
  }
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
getTransportCoeff<pele::physics::eos::Manifold>(
  const int i,
  const int j,
  const int k,
  const bool /*do_fixed_Le*/,
  const bool /*do_fixed_Pr*/,
  const bool do_soret,
  amrex::Real /*LeInv*/,
  amrex::Real /*PrInv*/,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<amrex::Real> const& rhoDi,
  amrex::Array4<amrex::Real> const& /*rhotheta*/,
  amrex::Array4<amrex::Real> const& lambda,
  amrex::Array4<amrex::Real> const& mu,
  pele::physics::transport::TransParm<
    pele::physics::PhysicsType::eos_type,
    pele::physics::PhysicsType::transport_type> const* trans_parm,
  const pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>*
    eosparm) noexcept
{
  // No Soret with manifold moels
  AMREX_ASSERT(!do_soret);

  // Get rho & Y from rhoY
  amrex::Real rho, rhoinv;
  amrex::Real y[NUM_SPECIES] = {0.0};
  amrex::Real massdens[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; ++n) {
    massdens[n] = rhoY(i, j, k, n);
  }
  auto eos = pele::physics::PhysicsType::eos(eosparm);
  eos.RY2RRinvY(massdens, rho, rhoinv, y);

  const amrex::Real rho_cgs = m2c::Rho(rho); // MKS -> CGS conversion
  amrex::Real rhoDi_cgs[NUM_SPECIES] = {0.0};
  amrex::Real lambda_cgs = 0.0;
  amrex::Real mu_cgs = 0.0;
  amrex::Real dummy_xi = 0.0;
  amrex::Real* dummy_chi = nullptr;
  amrex::Real Tloc = T(i, j, k);

  constexpr bool get_xi = false;
  constexpr bool get_mu = true;
  constexpr bool get_lam = false;
  constexpr bool get_Ddiag = true;
  const bool get_chi = do_soret;
  auto trans = pele::physics::PhysicsType::transport();
  trans.transport(
    get_xi, get_mu, get_lam, get_Ddiag, get_chi, Tloc, rho_cgs, y, rhoDi_cgs,
    dummy_chi, mu_cgs, dummy_xi, lambda_cgs, trans_parm);

  // Do CGS -> MKS conversions
  for (int n = 0; n < NUM_SPECIES; ++n) {
    rhoDi(i, j, k, n) = rhoDi_cgs[n] * 1.0e-1;
  }
  lambda(i, j, k) = 0.0; // No need to carry lambda for manifold
  mu(i, j, k) = c2m::Mu(mu_cgs);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
getVelViscosity(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<amrex::Real> const& mu,
  pele::physics::transport::TransParm<
    pele::physics::PhysicsType::eos_type,
    pele::physics::PhysicsType::transport_type> const* trans_parm) noexcept
{
  // Get rho & Y from rhoY
  amrex::Real massdens[NUM_SPECIES];
  for (int n = 0; n < NUM_SPECIES; ++n) {
    massdens[n] = rhoY(i, j, k, n);
  }
  amrex::Real rho = 0.0, rhoinv = 0.0, y[NUM_SPECIES] = {0.0};
  pele::physics::PhysicsType::eos_type::RY2RRinvY(massdens, rho, rhoinv, y);

  const amrex::Real rho_cgs = m2c::Rho(rho); // MKS -> CGS conversion
  amrex::Real temp = T(i, j, k);
  amrex::Real dummy_rhoDi[NUM_SPECIES] = {0.0};
  amrex::Real dummy_chi[NUM_SPECIES] = {0.0};
  amrex::Real dummy_lambda = 0.0;
  amrex::Real mu_cgs = 0.0;
  amrex::Real dummy_xi = 0.0;

  constexpr bool get_xi = false;
  constexpr bool get_mu = true;
  constexpr bool get_lam = false;
  constexpr bool get_Ddiag = false;
  constexpr bool get_chi = false;
  auto trans = pele::physics::PhysicsType::transport();
  trans.transport(
    get_xi, get_mu, get_lam, get_Ddiag, get_chi, temp, rho_cgs, y, dummy_rhoDi,
    dummy_chi, mu_cgs, dummy_xi, dummy_lambda, trans_parm);

  // Do CGS -> MKS conversions
  mu(i, j, k) = c2m::Mu(mu_cgs);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
getPGivenRTY(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& rho,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<amrex::Real> const& P,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm) noexcept
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);
  const amrex::Real rhoinv = 1.0 / rho(i, j, k);
  const amrex::Real rho_cgs = m2c::Rho(rho(i, j, k));

  amrex::Real y[NUM_SPECIES] = {0.0};
  for (int n = 0; n < NUM_SPECIES; ++n) {
    y[n] = rhoY(i, j, k, n) * rhoinv;
  }
  amrex::Real P_cgs = 0.0;
  eos.RTY2P(rho_cgs, T(i, j, k), y, P_cgs);
  P(i, j, k) = c2m::P(P_cgs); // CGS -> MKS conversion
}

template <typename EOSType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
compute_divu(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<const amrex::Real> const& specDiff,
  amrex::Array4<const amrex::Real> const& tempDiff,
  amrex::Array4<const amrex::Real> const& specEnthDiff,
  amrex::Array4<const amrex::Real> const& rhoYdot,
  amrex::Array4<const amrex::Real> const& extRhoY,
  amrex::Array4<const amrex::Real> const& extRhoH,
  amrex::Array4<amrex::Real> const& divu,
  const int do_react,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm) noexcept
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);
  amrex::Real mwtinv[NUM_SPECIES] = {0.0};
  eos.inv_molecular_weight(mwtinv);

  // Get rho & Y from rhoY
  amrex::Real rho = 0.0;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    rho += rhoY(i, j, k, n);
  }
  const amrex::Real rhoinv = 1.0 / rho;
  amrex::Real y[NUM_SPECIES] = {0.0};
  for (int n = 0; n < NUM_SPECIES; ++n) {
    y[n] = rhoY(i, j, k, n) * rhoinv;
  }

  amrex::Real cpmix_cgs = 0.0;
  eos.TY2Cp(T(i, j, k), y, cpmix_cgs);
  amrex::Real Wbar = 0.0;
  eos.Y2WBAR(y, Wbar);
  amrex::Real hi_cgs[NUM_SPECIES] = {0.0};
  eos.T2Hi(T(i, j, k), hi_cgs);
  amrex::Real cpmix = c2m::Cp(cpmix_cgs); // CGS -> MKS conversion
  amrex::Real hi[NUM_SPECIES] = {0.0};
  for (int n = 0; n < NUM_SPECIES; ++n) {
    hi[n] = c2m::H(hi_cgs[n]); // CGS -> MKS conversion
  }

  // Note: divu does not depend on extRho. See docs and PR #428 for details.
  const amrex::Real denominv = 1.0 / (rho * cpmix * T(i, j, k));
  divu(i, j, k) =
    (specEnthDiff(i, j, k) + tempDiff(i, j, k) + extRhoH(i, j, k)) * denominv;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    amrex::Real specTerm = specDiff(i, j, k, n) + extRhoY(i, j, k, n);
    if (do_react != 0) {
      specTerm += rhoYdot(i, j, k, n);
    }
    divu(i, j, k) += specTerm * (mwtinv[n] * Wbar * rhoinv - hi[n] * denominv);
  }
}

template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
compute_divu<pele::physics::eos::Manifold>(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& /*T*/,
  amrex::Array4<const amrex::Real> const& specDiff,
  amrex::Array4<const amrex::Real> const& /*tempDiff*/,
  amrex::Array4<const amrex::Real> const& /*specEnthDiff*/,
  amrex::Array4<const amrex::Real> const& rhoYdot,
  amrex::Array4<const amrex::Real> const& extRhoY,
  amrex::Array4<const amrex::Real> const& /*extRhoH*/,
  amrex::Array4<amrex::Real> const& divu,
  const int do_react,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm) noexcept
{
  // Get rho & Ys from rhoYs.
  amrex::Real rho, rhoinv;
  amrex::Real y[NUM_SPECIES] = {0.0};
  amrex::Real massdens[NUM_SPECIES] = {0.0};
  for (int n = 0; n < NUM_SPECIES; ++n) {
    massdens[n] = rhoY(i, j, k, n);
  }

  auto eos = pele::physics::PhysicsType::eos(eosparm);
  eos.RY2RRinvY(massdens, rho, rhoinv, y);
  amrex::Real derivs[MANIFOLD_DIM] = {0.0};
  amrex::Real rho_cgs = m2c::Rho(rho);

  eos.RY2dRdY(rho_cgs, y, derivs);

  divu(i, j, k) = 0.0;

  for (int n = 0; n < MANIFOLD_DIM; ++n) {
    derivs[n] *= 1.0e3; // Unit conversion CGS -> MKS
    divu(i, j, k) += derivs[n] * specDiff(i, j, k, n);
    if (do_react != 0) {
      divu(i, j, k) += derivs[n] * (rhoYdot(i, j, k, n));
    }
    divu(i, j, k) += derivs[n] * (extRhoY(i, j, k, n));
  }
  // First -rhoinv: from divu = -1/rho Drho/Dt
  // Second rhoinv: convert from rho*Dxi/Dt (which RHS terms are evaluated for)
  // to Dxi/Dt
  divu(i, j, k) *= -rhoinv * rhoinv;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
extFluxDivergence_K(
  const int i,
  const int j,
  const int k,
  const int ncomp,
  AMREX_D_DECL(
    amrex::Array4<const amrex::Real> const& fluxX,
    amrex::Array4<const amrex::Real> const& fluxY,
    amrex::Array4<const amrex::Real> const& fluxZ),
  amrex::Array4<const amrex::Real> const& vol,
  const amrex::Real scaling,
  amrex::Array4<amrex::Real> const& div) noexcept
{
  const amrex::Real factor = scaling / vol(i, j, k);

  for (int n = 0; n < ncomp; ++n) {
    div(i, j, k, n) = AMREX_D_TERM(
      (fluxX(i + 1, j, k, n) - fluxX(i, j, k, n)),
      +(fluxY(i, j + 1, k, n) - fluxY(i, j, k, n)),
      +(fluxZ(i, j, k + 1, n) - fluxZ(i, j, k, n)));
    div(i, j, k, n) *= factor;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
EB_intFluxDivergence_K(
  const int i,
  const int j,
  const int k,
  const int ncomp,
  AMREX_D_DECL(
    amrex::Array4<const amrex::Real> const& fluxX,
    amrex::Array4<const amrex::Real> const& fluxY,
    amrex::Array4<const amrex::Real> const& fluxZ),
  AMREX_D_DECL(
    amrex::Array4<const amrex::Real> const& afracX,
    amrex::Array4<const amrex::Real> const& afracY,
    amrex::Array4<const amrex::Real> const& afracZ),
  AMREX_D_DECL(amrex::Real areax, amrex::Real areay, amrex::Real areaz),
  amrex::Array4<const amrex::Real> const& vol,
  const amrex::Real scaling,
  amrex::Array4<amrex::Real> const& div) noexcept
{
  const amrex::Real factor = scaling / vol(i, j, k);

  for (int n = 0; n < ncomp; ++n) {
    div(i, j, k, n) = AMREX_D_TERM(
      (areax * afracX(i + 1, j, k) * fluxX(i + 1, j, k, n) -
       areax * afracX(i, j, k) * fluxX(i, j, k, n)),
      +(areay * afracY(i, j + 1, k) * fluxY(i, j + 1, k, n) -
        areay * afracY(i, j, k) * fluxY(i, j, k, n)),
      +(areaz * afracZ(i, j, k + 1) * fluxZ(i, j, k + 1, n) -
        areaz * afracZ(i, j, k) * fluxZ(i, j, k, n)));
    div(i, j, k, n) *= factor;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
EB_intFluxDivergence_K(
  const int i,
  const int j,
  const int k,
  const int ncomp,
  AMREX_D_DECL(
    amrex::Array4<const amrex::Real> const& fluxX,
    amrex::Array4<const amrex::Real> const& fluxY,
    amrex::Array4<const amrex::Real> const& fluxZ),
  AMREX_D_DECL(
    amrex::Array4<const amrex::Real> const& afracX,
    amrex::Array4<const amrex::Real> const& afracY,
    amrex::Array4<const amrex::Real> const& afracZ),
  AMREX_D_DECL(amrex::Real areax, amrex::Real areay, amrex::Real areaz),
  amrex::Array4<const amrex::Real> const& ebFlux,
  amrex::Array4<const amrex::Real> const& ebAreaFrac,
  amrex::Array4<const amrex::Real> const& vol,
  const amrex::Real a_dx,
  const amrex::Real scaling,
  amrex::Array4<amrex::Real> const& div) noexcept
{
  const amrex::Real factor = scaling / vol(i, j, k);
  const amrex::Real eb_area =
    AMREX_D_TERM(1.0, *a_dx, *a_dx) * ebAreaFrac(i, j, k);

  for (int n = 0; n < ncomp; ++n) {
    div(i, j, k, n) = AMREX_D_TERM(
      (areax * afracX(i + 1, j, k) * fluxX(i + 1, j, k, n) -
       areax * afracX(i, j, k) * fluxX(i, j, k, n)),
      +(areay * afracY(i, j + 1, k) * fluxY(i, j + 1, k, n) -
        areay * afracY(i, j, k) * fluxY(i, j, k, n)),
      +(areaz * afracZ(i, j, k + 1) * fluxZ(i, j, k + 1, n) -
        areaz * afracZ(i, j, k) * fluxZ(i, j, k, n)));
    div(i, j, k, n) += eb_area * ebFlux(i, j, k, n);
    div(i, j, k, n) *= factor;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
intFluxDivergence_K(
  const int i,
  const int j,
  const int k,
  const int ncomp,
  AMREX_D_DECL(
    amrex::Array4<const amrex::Real> const& fluxX,
    amrex::Array4<const amrex::Real> const& fluxY,
    amrex::Array4<const amrex::Real> const& fluxZ),
  AMREX_D_DECL(amrex::Real areax, amrex::Real areay, amrex::Real areaz),
  amrex::Array4<const amrex::Real> const& vol,
  const amrex::Real scaling,
  amrex::Array4<amrex::Real> const& div) noexcept
{
  const amrex::Real factor = scaling / vol(i, j, k);

  for (int n = 0; n < ncomp; ++n) {
    div(i, j, k, n) = AMREX_D_TERM(
      (areax * fluxX(i + 1, j, k, n) - areax * fluxX(i, j, k, n)),
      +(areay * fluxY(i, j + 1, k, n) - areay * fluxY(i, j, k, n)),
      +(areaz * fluxZ(i, j, k + 1, n) - areaz * fluxZ(i, j, k, n)));
    div(i, j, k, n) *= factor;
  }
}

#if (AMREX_SPACEDIM == 2)
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
intFluxDivergence_rz_K(
  const int i,
  const int j,
  const int k,
  const int ncomp,
  AMREX_D_DECL(
    amrex::Array4<const amrex::Real> const& fluxX,
    amrex::Array4<const amrex::Real> const& fluxY,
    amrex::Array4<const amrex::Real> const& fluxZ),
  amrex::Array4<const amrex::Real> const& ax,
  amrex::Array4<const amrex::Real> const& ay,
  amrex::Array4<const amrex::Real> const& vol,
  const amrex::Real scaling,
  amrex::Array4<amrex::Real> const& div) noexcept
{
  const amrex::Real factor = scaling / vol(i, j, k);

  for (int n = 0; n < ncomp; ++n) {
    div(i, j, k, n) = (ax(i + 1, j, k) * fluxX(i + 1, j, k, n) -
                       ax(i, j, k) * fluxX(i, j, k, n)) +
                      (ay(i, j + 1, k) * fluxY(i, j + 1, k, n) -
                       ay(i, j, k) * fluxY(i, j, k, n));
    div(i, j, k, n) *= factor;
  }
}
#endif

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
getMwmixGivenRY(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& rho,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<amrex::Real> const& Mwmix,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm) noexcept
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);
  const amrex::Real rhoinv = 1.0 / rho(i, j, k);
  amrex::Real y[NUM_SPECIES] = {0.0};
  for (int n = 0; n < NUM_SPECIES; ++n) {
    y[n] = rhoY(i, j, k, n) * rhoinv;
  }

  eos.Y2WBAR(y, Mwmix(i, j, k));
  Mwmix(i, j, k) = Mwmix(i, j, k) * 0.001; // CGS -> MKS conversion
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
getGradMwmixGivengradYMwmix(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& gradY,
  amrex::Array4<const amrex::Real> const& Mwmix,
  amrex::Array4<amrex::Real> const& gradMwmix,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm) noexcept
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);
  amrex::Real imw[NUM_SPECIES] = {0.0};
  eos.inv_molecular_weight(imw);
  gradMwmix(i, j, k) = 0.0;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    imw[n] *= 1000.0; // CGS -> MKS
    gradMwmix(i, j, k) += gradY(i, j, k, n) * imw[n];
  }
  // gradWbar = -Wbar^2 * sum(grad Y_k / W_k)
  gradMwmix(i, j, k) *= -Mwmix(i, j, k) * Mwmix(i, j, k);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
cen2edg_cpp(
  const int i,
  const int j,
  const int k,
  const int dir,
  const int ncomp,
  const bool do_harmonic,
  const bool on_lo,
  const bool on_hi,
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
  if (do_harmonic) {
    for (int n = 0; n < ncomp; ++n) {
      if (
        (cfab(id_l[0], id_l[1], id_l[2], n) *
         cfab(id_h[0], id_h[1], id_h[2], n)) > 0.0) {
        efab(i, j, k, n) = 2.0 *
                           (cfab(id_l[0], id_l[1], id_l[2], n) *
                            cfab(id_h[0], id_h[1], id_h[2], n)) /
                           (cfab(id_l[0], id_l[1], id_l[2], n) +
                            cfab(id_h[0], id_h[1], id_h[2], n));
      } else {
        efab(i, j, k, n) = 0.0;
      }
    }
  } else {
    for (int n = 0; n < ncomp; ++n) {
      efab(i, j, k, n) = 0.5 * (cfab(id_l[0], id_l[1], id_l[2], n) +
                                cfab(id_h[0], id_h[1], id_h[2], n));
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
repair_flux(
  const int i,
  const int j,
  const int k,
  const int dir,
  const bool on_lo,
  const bool on_hi,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<amrex::Real> const& flux) noexcept
{
  // Handle the BCs : need a -1 on id_l in dir away from the BC
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

  amrex::Real sumFlux = 0.0;
  amrex::Real sumRhoYe = 0.0;
  amrex::Real RhoYe[NUM_SPECIES] = {0.0};
  for (int n = 0; n < NUM_SPECIES; ++n) {
    sumFlux += flux(i, j, k, n);
    RhoYe[n] = 0.5 * (rhoY(id_l[0], id_l[1], id_l[2], n) +
                      rhoY(id_h[0], id_h[1], id_h[2], n));
    sumRhoYe += RhoYe[n];
  }
  sumRhoYe = 1.0 / sumRhoYe;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    flux(i, j, k, n) -= sumFlux * RhoYe[n] * sumRhoYe;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
repair_flux_eb(
  const int i,
  const int j,
  const int k,
  const int dir,
  const bool on_lo,
  const bool on_hi,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& rhoYed,
  amrex::Array4<const amrex::Real> const& areafrac,
  amrex::Array4<amrex::Real> const& flux) noexcept
{
  // Handle the BCs : need a -1 on id_l in dir on Dirichlet BC
  int id_l[3] = {i, j, k};
  id_l[dir] -= 1;
  if (areafrac(i, j, k) > 0.0) {
    amrex::Real sumFlux = 0.0;
    amrex::Real sumRhoYe = 0.0;
    amrex::Real RhoYe[NUM_SPECIES] = {0.0};
    for (int n = 0; n < NUM_SPECIES; ++n) {
      sumFlux += flux(i, j, k, n);
      RhoYe[n] = rhoYed(i, j, k, n);
      if (on_lo) {
        RhoYe[n] = rhoY(id_l[0], id_l[1], id_l[2], n);
      }
      if (on_hi) {
        RhoYe[n] = rhoY(i, j, k, n);
      }
      sumRhoYe += RhoYe[n];
    }
    sumRhoYe = 1.0 / sumRhoYe;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      flux(i, j, k, n) -= sumFlux * RhoYe[n] * sumRhoYe;
    }
  } else {
    for (int n = 0; n < NUM_SPECIES; ++n) {
      flux(i, j, k, n) = 0.0;
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
getHGivenT(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<amrex::Real> const& Hi,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm) noexcept
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);
  amrex::Real hi_spec_cgs[NUM_SPECIES] = {0.0};
  eos.T2Hi(T(i, j, k), hi_spec_cgs);
  for (int n = 0; n < NUM_SPECIES; ++n) {
    Hi(i, j, k, n) = c2m::H(hi_spec_cgs[n]); // CGS -> MKS conversion
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
makeVelForce(
  const int i,
  const int j,
  const int k,
  const int is_incomp,
  const amrex::Real rho_incomp,
  const int pseudo_gravity,
  const int pseudo_gravity_dir,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& gravity,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& gp0,
  const amrex::Real dV_control,
  amrex::Array4<const amrex::Real> const& vel,
  amrex::Array4<const amrex::Real> const& rho,
  amrex::Array4<const amrex::Real> const& extmom,
  amrex::Array4<const amrex::Real> const& extrho,
  amrex::Array4<amrex::Real> const& force) noexcept
{
  // Switch between incompressible/low-Mach rhos
  amrex::Real rho_lcl = 0.0;
  if (is_incomp != 0) {
    rho_lcl = rho_incomp;
  } else {
    rho_lcl = rho(i, j, k);
  }

  // Gravity and background pressure gradient
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    force(i, j, k, idim) = -gp0[idim] + gravity[idim] * rho_lcl;
  }

  // Pseudo-gravity for active control
  if (pseudo_gravity != 0) {
    force(i, j, k, pseudo_gravity_dir) += dV_control * rho_lcl;
  }

  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    force(i, j, k, idim) +=
      extmom(i, j, k, idim) - vel(i, j, k, idim) * extrho(i, j, k);
  }
}

// This kernel returns directly rhoHmix
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
getRHmixGivenTY(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& rho,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<amrex::Real> const& Hmix,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm) noexcept
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);
  const amrex::Real rhoinv = 1.0 / rho(i, j, k);
  amrex::Real y[NUM_SPECIES] = {0.0};
  for (int n = 0; n < NUM_SPECIES; ++n) {
    y[n] = rhoY(i, j, k, n) * rhoinv;
  }
  amrex::Real hmix_cgs = 0.0;
  eos.TY2H(T(i, j, k), y, hmix_cgs);
  Hmix(i, j, k) = c2m::H(hmix_cgs) * rho(i, j, k); // CGS -> MKS conversion
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
getTfromHY(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& rho,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& rhoH,
  amrex::Array4<amrex::Real> const& T,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm) noexcept
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);
  const amrex::Real rhoinv = 1.0 / rho(i, j, k);
  amrex::Real y[NUM_SPECIES] = {0.0};
  for (int n = 0; n < NUM_SPECIES; ++n) {
    y[n] = rhoY(i, j, k, n) * rhoinv;
  }
  const amrex::Real h_cgs = m2c::H(rhoH(i, j, k) * rhoinv);
  amrex::Real T_loc = T(i, j, k);
  eos.HY2T(h_cgs, y, T_loc);
  T(i, j, k) = T_loc;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
getCpmixGivenRYT(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& rho,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<amrex::Real> const& cpmix,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm) noexcept
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);
  const amrex::Real rhoinv = 1.0 / rho(i, j, k);
  amrex::Real y[NUM_SPECIES] = {0.0};
  for (int n = 0; n < NUM_SPECIES; ++n) {
    y[n] = rhoY(i, j, k, n) * rhoinv;
  }
  amrex::Real cpmix_cgs = 0.0;
  eos.TY2Cp(T(i, j, k), y, cpmix_cgs);
  cpmix(i, j, k) = c2m::Cp(cpmix_cgs); // CGS -> MKS conversion
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
buildAdvectionForcing(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& rho,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<const amrex::Real> const& dn,
  amrex::Array4<const amrex::Real> const& ddn,
  amrex::Array4<const amrex::Real> const& r,
  amrex::Array4<const amrex::Real> const& extRhoY,
  amrex::Array4<const amrex::Real> const& extRhoH,
  const amrex::Real dp0dt,
  const int closed_chamber,
  const int do_react,
  amrex::Array4<amrex::Real> const& forceY,
  amrex::Array4<amrex::Real> const& forceT,
  amrex::Array4<amrex::Real> const& forceAux,
  amrex::Array4<const amrex::Real> const& dn_aux,
  const int* aux_diffuse,
  const int nAux,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm) noexcept
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);
  // Get species enthalpy
  amrex::Real hi_spec_cgs[NUM_SPECIES] = {0.0};
  eos.T2Hi(T(i, j, k), hi_spec_cgs);

  // For species mass fractions
  const amrex::Real rhoinv = 1.0 / rho(i, j, k);
  amrex::Real y[NUM_SPECIES] = {0.0};

  forceT(i, j, k) = dn(i, j, k, NUM_SPECIES) + ddn(i, j, k);
  for (int n = 0; n < NUM_SPECIES; ++n) {
    y[n] = rhoY(i, j, k, n) * rhoinv;    // get y
    forceY(i, j, k, n) = dn(i, j, k, n); // forceY = Dnk
    if (do_react != 0) {
      forceY(i, j, k, n) += r(i, j, k, n); //         + Rk
    }
    forceT(i, j, k) -=
      forceY(i, j, k, n) *
      c2m::H(hi_spec_cgs[n]); // forceT -= Sum{ hk. (Rk + Dnk) } with CGS ->
                              // MKS conversion of hk
    forceY(i, j, k, n) += extRhoY(i, j, k, n);
  }

  forceT(i, j, k) += extRhoH(i, j, k);
  if (closed_chamber == 1) {
    forceT(i, j, k) += dp0dt; // Add dp0/dt
  }

  for (int n = 0; n < nAux; ++n) {
    if (aux_diffuse[n] != 0) {
      forceAux(i, j, k, n) = dn_aux(i, j, k, n);
    } else {
      forceAux(i, j, k, n) = 0.0;
    }
    // TO DO: source terms for Aux
  }

  amrex::Real cpmix_cgs = 0.0;
  eos.TY2Cp(T(i, j, k), y, cpmix_cgs);
  const amrex::Real cpmixinv =
    1.0 / c2m::Cp(cpmix_cgs); // CGS -> MKS conversion
  forceT(i, j, k) *= rhoinv * cpmixinv;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
buildDiffusionForcing(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& dn,
  amrex::Array4<const amrex::Real> const& ddn,
  amrex::Array4<const amrex::Real> const& dnp1k,
  amrex::Array4<const amrex::Real> const& ddnp1k,
  amrex::Array4<const amrex::Real> const& r,
  amrex::Array4<const amrex::Real> const& a,
  const amrex::Real dp0dt,
  const int closed_chamber,
  const int do_react,
  amrex::Array4<amrex::Real> const& forceY,
  amrex::Array4<amrex::Real> const& forceT,
  amrex::Array4<const amrex::Real> const& dwbar,
  amrex::Array4<const amrex::Real> const& dT,
  amrex::Array4<const amrex::Real> const& extRhoY,
  amrex::Array4<const amrex::Real> const& extRhoH,
  const int use_wbar,
  const int use_soret,
  amrex::Array4<amrex::Real> const& forceAux,
  amrex::Array4<const amrex::Real> const& a_aux,
  amrex::Array4<const amrex::Real> const& dn_aux,
  amrex::Array4<const amrex::Real> const& dnp1k_aux,
  const int* aux_advect,
  const int* aux_diffuse,
  const int nAux) noexcept
{
  for (int n = 0; n < NUM_SPECIES; ++n) {
    forceY(i, j, k, n) = a(i, j, k, n) +
                         0.5 * (dn(i, j, k, n) - dnp1k(i, j, k, n)) +
                         extRhoY(i, j, k, n);
    if (do_react != 0) {
      forceY(i, j, k, n) += r(i, j, k, n);
    }
    if (use_wbar != 0) {
      forceY(i, j, k, n) += dwbar(i, j, k, n);
    }
    if (use_soret != 0) {
      forceY(i, j, k, n) += dT(i, j, k, n);
    }
  }
  forceT(i, j, k) = a(i, j, k, NUM_SPECIES) +
                    0.5 * (dn(i, j, k, NUM_SPECIES) + ddn(i, j, k) -
                           dnp1k(i, j, k, NUM_SPECIES) - ddnp1k(i, j, k)) +
                    extRhoH(i, j, k);

  if (closed_chamber == 1) {
    forceT(i, j, k) += dp0dt;
  }

  for (int n = 0; n < nAux; ++n) {
    forceAux(i, j, k, n) = 0.0;
    if (aux_diffuse[n] != 0) {
      forceAux(i, j, k, n) +=
        0.5 * (dn_aux(i, j, k, n) - dnp1k_aux(i, j, k, n));
    }
    if (aux_advect[n] != 0) {
      forceAux(i, j, k, n) += a_aux(i, j, k, n);
    }
    // To do: source terms for aux
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
reactionRateRhoY(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& rhoH,
  amrex::Array4<const amrex::Real> const& T,
  amrex::Array4<amrex::Real> const& rhoYdot,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm) noexcept
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);

  // Get rho & Ys from rhoYs.
  amrex::Real rho = 0.0, rhoinv = 0.0, y[NUM_SPECIES] = {0.0};
  eos.RY2RRinvY(rhoY.cellData(i, j, k), rho, rhoinv, y);

  // Get T from Y/H.
  amrex::Real Tloc = T(i, j, k);
  const amrex::Real h_cgs =
    m2c::H(rhoH(i, j, k) * rhoinv); // Include MKS -> CGS conversion
  eos.HY2T(h_cgs, y, Tloc);

  // Get wdot.
  const amrex::Real rho_cgs = m2c::Rho(rho); // rho MKS -> CGS
  amrex::Real wdot[NUM_SPECIES] = {0.0};
  eos.RTY2WDOT(rho_cgs, Tloc, y, wdot);
  for (int n = 0; n < NUM_SPECIES; ++n) {
    rhoYdot(i, j, k, n) = wdot[n] * 1000.0; // CGS -> MKS conversion
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
fabMinMax(
  const int i,
  const int j,
  const int k,
  const int ncomp,
  const amrex::Real fmin,
  const amrex::Real fmax,
  amrex::Array4<amrex::Real> const& fab) noexcept
{
  for (int n = 0; n < ncomp; ++n) {
    fab(i, j, k, n) = amrex::Clamp(fab(i, j, k, n), fmin, fmax);
  }
}

// Type 1 est_divu_dt function
// Check based on divU
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
est_divu_dt_1(
  const int i,
  const int j,
  const int k,
  const amrex::Real dtfactor,
  const amrex::Real rhomin,
  amrex::Array4<const amrex::Real> const& rho,
  amrex::Array4<const amrex::Real> const& divu) noexcept
{
  amrex::Real dtcell = 1.0e20;
  const bool rho_is_okay = (rho(i, j, k) > rhomin);

  if (divu(i, j, k) > 0.0) {
    dtcell = (rho_is_okay) ? (1.0 - rhomin / rho(i, j, k)) / divu(i, j, k)
                           : 1.0 / divu(i, j, k);
    dtcell *= dtfactor;
  }
  return dtcell;
}

// Type 2 est_divu_dt function
// Check based rho * divU + u \cdot grad rho
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
est_divu_dt_2(
  const int i,
  const int j,
  const int k,
  const amrex::Real dtfactor,
  const amrex::Real rhomin,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxinv_uniform,
  MeshMapEvaluator const& mmap,
  amrex::GeometryData const& gd,
  amrex::Array4<const amrex::Real> const& rho,
  amrex::Array4<const amrex::Real> const& vel,
  amrex::Array4<const amrex::Real> const& divu) noexcept
{
  amrex::Real dtcell = 1.0e20;
  amrex::Real denom = 0.0;
  const bool rho_is_okay = (rho(i, j, k) > rhomin);

  const auto dxinv = mapped_inv_dx_array(i, j, k, dxinv_uniform, mmap, gd);

  denom =
    rho(i, j, k) * divu(i, j, k) +
    AMREX_D_TERM(
      vel(i, j, k, 0) * (rho(i + 1, j, k) - rho(i - 1, j, k)) * dxinv[0],
      +vel(i, j, k, 1) * (rho(i, j + 1, k) - rho(i, j - 1, k)) * dxinv[1],
      +vel(i, j, k, 2) * (rho(i, j, k + 1) - rho(i, j, k - 1)) * dxinv[2]);
  if (denom > 0.0) {
    dtcell = (rho_is_okay) ? (rho(i, j, k) - rhomin) / denom
                           : std::abs(rho(i, j, k)) / denom;
    dtcell *= dtfactor;
  }
  return dtcell;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
check_divu_dt(
  const int i,
  const int j,
  const int k,
  const int check_type,
  const amrex::Real dtfactor,
  const amrex::Real rhomin,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxinv_uniform,
  MeshMapEvaluator const& mmap,
  amrex::GeometryData const& gd,
  amrex::Array4<const amrex::Real> const& rho,
  amrex::Array4<const amrex::Real> const& vel,
  amrex::Array4<const amrex::Real> const& divu,
  amrex::Real const& a_dt) noexcept
{
  amrex::Real dtcell = 1.0e12;
  amrex::Real denom = 0.0;
  const bool rho_is_okay = (rho(i, j, k) > rhomin);
  const auto dxinv = mapped_inv_dx_array(i, j, k, dxinv_uniform, mmap, gd);

  if (check_type == 1) { // Check based on divU
    if (divu(i, j, k) > 0.0) {
      dtcell = (rho_is_okay) ? (1.0 - rhomin / rho(i, j, k)) / divu(i, j, k)
                             : 1.0 / divu(i, j, k);
    }
  } else if (check_type == 2) { // Check based rho * divU + u \cdot grad rho
    denom =
      rho(i, j, k) * divu(i, j, k) +
      AMREX_D_TERM(
        vel(i, j, k, 0) * (rho(i + 1, j, k) - rho(i - 1, j, k)) * dxinv[0],
        +vel(i, j, k, 1) * (rho(i, j + 1, k) - rho(i, j - 1, k)) * dxinv[1],
        +vel(i, j, k, 2) * (rho(i, j, k + 1) - rho(i, j, k - 1)) * dxinv[2]);
    if (denom > 0.0) {
      dtcell = (rho_is_okay) ? (rho(i, j, k) - rhomin) / denom
                             : std::abs(rho(i, j, k)) / denom;
    }
  }
  if (a_dt > dtcell) {
    amrex::Abort("Error: divU dtcell larger than actual dt !");
  } else if (a_dt > dtcell * dtfactor) {
    amrex::Warning("Warning: divU dtcell close to actual dt !");
  }
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
getGammaInv(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& rhoY,
  amrex::Array4<const amrex::Real> const& T,
  pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type> const*
    eosparm) noexcept
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);
  // Get rho & Y from rhoY
  amrex::Real rho = 0.0;
  for (int n = 0; n < NUM_SPECIES; ++n) {
    rho += rhoY(i, j, k, n);
  }
  const amrex::Real rhoinv = 1.0 / rho;
  amrex::Real y[NUM_SPECIES] = {0.0};
  for (int n = 0; n < NUM_SPECIES; ++n) {
    y[n] = rhoY(i, j, k, n) * rhoinv;
  }

  amrex::Real cpmix = 0.0;
  eos.TY2Cp(T(i, j, k), y, cpmix);
  amrex::Real cvmix = 0.0;
  eos.TY2Cv(T(i, j, k), y, cvmix);

  const amrex::Real gammainv = cvmix / cpmix;

  return gammainv;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
getSijSij(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& velgrad) noexcept
{
  // This function returns SijSij, not strain rate magnitude |S| = sqrt(2
  // SijSij)
  amrex::Real SijSij = 0.0;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
      // Sij = 0.5(dui/dxj + duj/dxi)
      amrex::Real Sij = 0.5 * (velgrad(i, j, k, idim * AMREX_SPACEDIM + jdim) +
                               velgrad(i, j, k, jdim * AMREX_SPACEDIM + idim));
      SijSij += Sij * Sij;
    }
  }
  return SijSij;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
getVelGradTerm(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& velgrad) noexcept
{
  // Compute S^d_ij*S^d_ij
  // S^d_ij = 0.5(g2_ij + g2_ji) - (1/3)*delta_ij*g2_kk
  // g2_ij = du_i/dx_l * du_l/dx_j
  // Therefore, g2_kk = du_k/dx_l * du_l/dx_k
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM * AMREX_SPACEDIM> g2_ij;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
      g2_ij[idim * AMREX_SPACEDIM + jdim] = 0.0;
      for (int ldim = 0; ldim < AMREX_SPACEDIM; ++ldim) {
        g2_ij[idim * AMREX_SPACEDIM + jdim] +=
          (velgrad(i, j, k, idim * AMREX_SPACEDIM + ldim) *
           velgrad(i, j, k, ldim * AMREX_SPACEDIM + jdim));
      }
    }
  }
  const amrex::Real g2_kk_third =
    (1.0 / AMREX_SPACEDIM) *
    (AMREX_D_TERM(
      g2_ij[0], +g2_ij[AMREX_SPACEDIM + 1], +g2_ij[2 * AMREX_SPACEDIM + 2]));

  amrex::Real SdijSdij = 0.0;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
      amrex::Real Sdij = (idim == jdim)
                           ? g2_ij[idim * AMREX_SPACEDIM + jdim] - g2_kk_third
                           : 0.5 * (g2_ij[idim * AMREX_SPACEDIM + jdim] +
                                    g2_ij[jdim * AMREX_SPACEDIM + idim]);
      SdijSdij += Sdij * Sdij;
    }
  }
  return SdijSdij;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getTurbViscSmagorinsky(
  const int i,
  const int j,
  const int k,
  const amrex::Real prefactor,
  amrex::Array4<const amrex::Real> const& velgrad,
  amrex::Array4<const amrex::Real> const& rho,
  amrex::Array4<amrex::Real> const& mu_t) noexcept
{
  // mu_t = rho * Cs * Delta^2 * |S|    where |S| = sqrt(2 SijSij)

  const amrex::Real Smag = std::sqrt(2.0 * getSijSij(i, j, k, velgrad));
  mu_t(i, j, k) = prefactor * rho(i, j, k) * Smag;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getTurbViscWALE(
  const int i,
  const int j,
  const int k,
  const amrex::Real prefactor,
  amrex::Array4<const amrex::Real> const& velgrad,
  amrex::Array4<const amrex::Real> const& rho,
  amrex::Array4<amrex::Real> const& mu_t) noexcept
{
  // From Ducros, Nicoud, and Poinsot 1999
  // mu_t = rho * Cs * Delta^2 * (S^d_ijS^d_ij)^(3/2)
  //        / (SijSij^(5/2) + (S^d_ijS^d_ij)^(5/4) + smallnum)
  // smallnum included to make sure denom is nonzero

  const amrex::Real SijSij = getSijSij(i, j, k, velgrad);
  const amrex::Real Sterm52 = std::pow(SijSij, 2.5);
  const amrex::Real Sdterm = getVelGradTerm(i, j, k, velgrad); // (S^d_ijS^d_ij)
  const amrex::Real Sdterm14 = std::pow(Sdterm, 0.25);
  const amrex::Real Sdterm54 = Sdterm * Sdterm14;
  const amrex::Real Sdterm32 = Sdterm54 * Sdterm14;
  mu_t(i, j, k) =
    prefactor * rho(i, j, k) * Sdterm32 / (Sterm52 + Sdterm54 + 1.0e-12);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getTurbViscSigma(
  const int i,
  const int j,
  const int k,
  const amrex::Real prefactor,
  amrex::Array4<const amrex::Real> const& velgrad,
  amrex::Array4<const amrex::Real> const& rho,
  amrex::Array4<amrex::Real> const& mu_t) noexcept
{
#if (AMREX_SPACEDIM == 2)
  amrex::ignore_unused(i, j, k, prefactor, velgrad, rho, mu_t);
  amrex::Abort("getTurbViscSigma not implemented in 2D");
#elif (AMREX_SPACEDIM == 3)
  // From Nicoud, et al . 2011
  // mu_t = rho * Cs * Delta^2 * \sig_3(\sig_1-\sig_2)(\sig_2-\sig_3)
  //        / (\sig_1*\sig_1 + smallnum)
  // smallnum included to make sure denom is nonzero
  // where sig_* are the singular values of G = g^tg

  constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;

  amrex::Real D_sig = 0.0;

  // Get G
  amrex::Array2D<amrex::Real, 0, AMREX_SPACEDIM - 1, 0, AMREX_SPACEDIM - 1> gtg;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
      gtg(idim, jdim) = 0.0;
      for (int ldim = 0; ldim < AMREX_SPACEDIM; ++ldim) {
        gtg(idim, jdim) +=
          (velgrad(i, j, k, idim * AMREX_SPACEDIM + ldim) *
           velgrad(i, j, k, ldim * AMREX_SPACEDIM + jdim));
      }
    }
  }

  // Invariants: tr(G), 0.5*(tr(G)^2 - tr(G^2)), det(G)
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> I{0.0};
  I[0] = gtg(0, 0) + gtg(1, 1) + gtg(2, 2);
  I[1] = gtg(0, 0) * gtg(1, 1) - gtg(0, 1) * gtg(0, 1) + gtg(0, 0) * gtg(2, 2) -
         gtg(0, 2) * gtg(0, 2) + gtg(1, 1) * gtg(2, 2) - gtg(1, 2) * gtg(1, 2);
  I[2] = gtg(0, 0) * (gtg(1, 1) * gtg(2, 2) - gtg(1, 2) * gtg(2, 1)) -
         gtg(1, 0) * (gtg(0, 1) * gtg(2, 2) - gtg(0, 2) * gtg(2, 1)) +
         gtg(2, 0) * (gtg(0, 1) * gtg(1, 2) - gtg(0, 2) * gtg(1, 1));

  // Rotation angles:
  amrex::Real alpha1 = 0.0;
  amrex::Real alpha2 = 0.0;
  amrex::Real alpha3 = 0.0;

  alpha1 =
    amrex::max<amrex::Real>(alpha1, (I[0] / 3.0) * (I[0] / 3.0) - I[1] / 3.0);
  if (alpha1 != 0.0) {
    alpha2 = (I[0] / 3.0) * (I[0] / 3.0) * (I[0] / 3.0) - I[0] * I[1] / 6.0 +
             I[2] / 2.0;
    const amrex::Real arcCosArg =
      amrex::Clamp(alpha2 / std::pow(alpha1, 1.5), -1.0, 1.0);
    alpha3 = std::acos(arcCosArg) / 3.0;

    // Singular values:
    amrex::Real sigma1 = std::sqrt(
      amrex::max<amrex::Real>(
        0.0, I[0] / 3.0 + 2.0 * std::sqrt(alpha1) * std::cos(alpha3)));
    amrex::Real sigma2 = std::sqrt(
      amrex::max<amrex::Real>(
        0.0,
        I[0] / 3.0 - 2.0 * std::sqrt(alpha1) * std::cos(Pi / 3.0 + alpha3)));
    amrex::Real sigma3 = std::sqrt(
      amrex::max<amrex::Real>(
        0.0,
        I[0] / 3.0 - 2.0 * std::sqrt(alpha1) * std::cos(Pi / 3.0 - alpha3)));

    // Sort
    sigma2 = amrex::max<amrex::Real>(sigma3, sigma2);
    sigma1 = amrex::max<amrex::Real>(sigma2, sigma1);
    sigma1 = amrex::max<amrex::Real>(1e-16, sigma1);

    D_sig = sigma3 * (sigma1 - sigma2) * (sigma2 - sigma3) / (sigma1 * sigma1);
  }

  mu_t(i, j, k) = prefactor * rho(i, j, k) * D_sig;
#endif
}

#endif
