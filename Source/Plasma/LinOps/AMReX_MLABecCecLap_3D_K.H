#ifndef AMREX_MLABECCECLAP_3D_K_H
#define AMREX_MLABECCECLAP_3D_K_H

namespace amrex {

constexpr Real eps = 1.0e-6;

AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
getFaceStateUpwind(
  Real const& edgeVel, Real const& leftState, Real const& rightState)
{
  Real edstate = 0.0;
  if (edgeVel > eps) {
    edstate = leftState;
  }
  if (edgeVel < -eps) {
    edstate = rightState;
  }
  if (fabs(edgeVel) <= eps) {
    edstate = 0.5 * (leftState + rightState);
  }
  return edstate;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
getCellUpwindCoeff(Real const& leftedgeVel, Real const& rightedgeVel)
{

  Real coeff = 0.0;
  if (leftedgeVel * rightedgeVel < 0.0) { // Case where velocity change sign,
                                          // use centered + art. visc
    Real maxvel = std::max(fabs(leftedgeVel), fabs(rightedgeVel));
    coeff += 0.5 * (rightedgeVel - leftedgeVel); // centered part
    coeff += maxvel;                             // art. visc.
  } else { // same sign velocity, use upwind
    if (leftedgeVel < -eps)
      coeff -= leftedgeVel;
    else if (fabs(leftedgeVel) <= eps)
      coeff -= 0.5 * leftedgeVel;
    if (rightedgeVel > eps)
      coeff += rightedgeVel;
    else if (fabs(rightedgeVel) <= eps)
      coeff += 0.5 * rightedgeVel;
  }

  return coeff;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
getLeftCellUpwindCoeff(Real const& leftedgeVel, Real const& rightedgeVel)
{

  Real coeff = 0.0;
  if (leftedgeVel * rightedgeVel < 0.0) {
    Real maxvel = std::max(fabs(leftedgeVel), fabs(rightedgeVel));
    coeff -= 0.5 * leftedgeVel;
    coeff -= 0.5 * maxvel;
  } else {
    if (leftedgeVel > eps)
      coeff -= leftedgeVel;
    else if (fabs(leftedgeVel) < eps)
      coeff -= 0.5 * leftedgeVel;
  }
  return coeff;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
getRightCellUpwindCoeff(Real const& leftedgeVel, Real const& rightedgeVel)
{

  Real coeff = 0.0;
  if (leftedgeVel * rightedgeVel < 0.0) {
    Real maxvel = std::max(fabs(leftedgeVel), fabs(rightedgeVel));
    coeff += 0.5 * rightedgeVel;
    coeff += 0.5 * maxvel;
  } else {
    if (rightedgeVel < -eps)
      coeff += rightedgeVel;
    else if (fabs(rightedgeVel) < eps)
      coeff += 0.5 * rightedgeVel;
  }
  return coeff;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
getCellUpwindCoeff(
  Real const& leftedgeVel,
  Real const& rightedgeVel,
  Real const& relaxleft,
  Real const& relaxright)
{

  Real coeff = 0.0;
  if (
    leftedgeVel * relaxleft * rightedgeVel * relaxright <
    0.0) { // Case where velocity change sign, use centered + art. visc
    Real maxvel =
      std::max(fabs(leftedgeVel * relaxleft), fabs(rightedgeVel * relaxright));
    coeff += 0.5 * (rightedgeVel * relaxright -
                    leftedgeVel * relaxleft); // centered part
    coeff += maxvel;                          // art. visc.
  } else {                                    // same sign velocity, use upwind
    if (leftedgeVel < -eps)
      coeff -= leftedgeVel * relaxleft;
    else if (fabs(leftedgeVel) < eps)
      coeff -= 0.5 * leftedgeVel * relaxleft;
    if (rightedgeVel > eps)
      coeff += rightedgeVel * relaxright;
    else if (fabs(rightedgeVel) < eps)
      coeff += 0.5 * rightedgeVel * relaxright;
  }

  return coeff;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE Real
getFaceUpwindFlux(
  Real const& edgeVel, Real const& leftState, Real const& rightState)
{

  Real Flux = 0.0;
  if (edgeVel > eps)
    Flux = edgeVel * leftState;
  else if (edgeVel < -eps)
    Flux = edgeVel * rightState;
  else
    Flux = edgeVel * 0.5 * (leftState + rightState);
  return Flux;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_adotx(
  Box const& box,
  Array4<Real> const& y,
  Array4<Real const> const& x,
  Array4<Real const> const& a,
  Array4<Real const> const& bX,
  Array4<Real const> const& bY,
  Array4<Real const> const& bZ,
  Array4<Real const> const& cX,
  Array4<Real const> const& cY,
  Array4<Real const> const& cZ,
  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
  Real alpha,
  Real beta,
  Real eta,
  int ncomp) noexcept
{
  const Real dhxsq = beta * dxinv[0] * dxinv[0];
  const Real dhysq = beta * dxinv[1] * dxinv[1];
  const Real dhzsq = beta * dxinv[2] * dxinv[2];
  const Real dhx = eta * dxinv[0];
  const Real dhy = eta * dxinv[1];
  const Real dhz = eta * dxinv[2];

  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        AMREX_PRAGMA_SIMD
        for (int i = lo.x; i <= hi.x; ++i) {
          y(i, j, k, n) =
            alpha * a(i, j, k) * x(i, j, k, n) -
            dhxsq * (bX(i + 1, j, k, n) * (x(i + 1, j, k, n) - x(i, j, k, n)) -
                     bX(i, j, k, n) * (x(i, j, k, n) - x(i - 1, j, k, n))) -
            dhysq * (bY(i, j + 1, k, n) * (x(i, j + 1, k, n) - x(i, j, k, n)) -
                     bY(i, j, k, n) * (x(i, j, k, n) - x(i, j - 1, k, n))) -
            dhzsq * (bZ(i, j, k + 1, n) * (x(i, j, k + 1, n) - x(i, j, k, n)) -
                     bZ(i, j, k, n) * (x(i, j, k, n) - x(i, j, k - 1, n))) -
            dhx * (cX(i + 1, j, k, n) *
                     getFaceStateUpwind(
                       cX(i + 1, j, k, n), x(i, j, k, n), x(i + 1, j, k, n)) -
                   cX(i, j, k, n) *
                     getFaceStateUpwind(
                       cX(i, j, k, n), x(i - 1, j, k, n), x(i, j, k, n))) -
            dhy * (cY(i, j + 1, k, n) *
                     getFaceStateUpwind(
                       cY(i, j + 1, k, n), x(i, j, k, n), x(i, j + 1, k, n)) -
                   cY(i, j, k, n) *
                     getFaceStateUpwind(
                       cY(i, j, k, n), x(i, j - 1, k, n), x(i, j, k, n))) -
            dhz * (cZ(i, j, k + 1, n) *
                     getFaceStateUpwind(
                       cZ(i, j, k + 1, n), x(i, j, k, n), x(i, j, k + 1, n)) -
                   cZ(i, j, k, n) *
                     getFaceStateUpwind(
                       cZ(i, j, k, n), x(i, j, k - 1, n), x(i, j, k, n)));
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_adotx_os(
  Box const& box,
  Array4<Real> const& y,
  Array4<Real const> const& x,
  Array4<Real const> const& a,
  Array4<Real const> const& bX,
  Array4<Real const> const& bY,
  Array4<Real const> const& bZ,
  Array4<Real const> const& cX,
  Array4<Real const> const& cY,
  Array4<Real const> const& cZ,
  Array4<int const> const& osm,
  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
  Real alpha,
  Real beta,
  Real eta,
  int ncomp) noexcept
{
  const Real dhxsq = beta * dxinv[0] * dxinv[0];
  const Real dhysq = beta * dxinv[1] * dxinv[1];
  const Real dhzsq = beta * dxinv[2] * dxinv[2];
  const Real dhx = eta * dxinv[0];
  const Real dhy = eta * dxinv[1];
  const Real dhz = eta * dxinv[2];

  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        AMREX_PRAGMA_SIMD
        for (int i = lo.x; i <= hi.x; ++i) {
          if (osm(i, j, k) == 0) {
            y(i, j, k, n) = 0.0;
          } else {
            y(i, j, k, n) =
              alpha * a(i, j, k) * x(i, j, k, n) -
              dhxsq *
                (bX(i + 1, j, k, n) * (x(i + 1, j, k, n) - x(i, j, k, n)) -
                 bX(i, j, k, n) * (x(i, j, k, n) - x(i - 1, j, k, n))) -
              dhysq *
                (bY(i, j + 1, k, n) * (x(i, j + 1, k, n) - x(i, j, k, n)) -
                 bY(i, j, k, n) * (x(i, j, k, n) - x(i, j - 1, k, n))) -
              dhzsq *
                (bZ(i, j, k + 1, n) * (x(i, j, k + 1, n) - x(i, j, k, n)) -
                 bZ(i, j, k, n) * (x(i, j, k, n) - x(i, j, k - 1, n))) -
              dhx * (cX(i + 1, j, k, n) *
                       getFaceStateUpwind(
                         cX(i + 1, j, k, n), x(i, j, k, n), x(i + 1, j, k, n)) -
                     cX(i, j, k, n) *
                       getFaceStateUpwind(
                         cX(i, j, k, n), x(i - 1, j, k, n), x(i, j, k, n))) -
              dhy * (cY(i, j + 1, k, n) *
                       getFaceStateUpwind(
                         cY(i, j + 1, k, n), x(i, j, k, n), x(i, j + 1, k, n)) -
                     cY(i, j, k, n) *
                       getFaceStateUpwind(
                         cY(i, j, k, n), x(i, j - 1, k, n), x(i, j, k, n))) -
              dhz * (cZ(i, j, k + 1, n) *
                       getFaceStateUpwind(
                         cZ(i, j, k + 1, n), x(i, j, k, n), x(i, j, k + 1, n)) -
                     cZ(i, j, k, n) *
                       getFaceStateUpwind(
                         cZ(i, j, k, n), x(i, j, k - 1, n), x(i, j, k, n)));
          }
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_normalize(
  Box const& box,
  Array4<Real> const& x,
  Array4<Real const> const& a,
  Array4<Real const> const& bX,
  Array4<Real const> const& bY,
  Array4<Real const> const& bZ,
  Array4<Real const> const& cX,
  Array4<Real const> const& cY,
  Array4<Real const> const& cZ,
  GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
  Real alpha,
  Real beta,
  Real eta,
  int ncomp) noexcept
{
  const Real dhxsq = beta * dxinv[0] * dxinv[0];
  const Real dhysq = beta * dxinv[1] * dxinv[1];
  const Real dhzsq = beta * dxinv[2] * dxinv[2];
  const Real dhx = eta * dxinv[0];
  const Real dhy = eta * dxinv[1];
  const Real dhz = eta * dxinv[2];

  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        AMREX_PRAGMA_SIMD
        for (int i = lo.x; i <= hi.x; ++i) {
          x(i, j, k, n) /=
            alpha * a(i, j, k) + dhxsq * (bX(i, j, k, n) + bX(i + 1, j, k, n)) +
            dhysq * (bY(i, j, k, n) + bY(i, j + 1, k, n)) +
            dhzsq * (bZ(i, j, k, n) + bZ(i, j, k + 1, n)) -
            dhx * getCellUpwindCoeff(cX(i, j, k, n), cX(i + 1, j, k, n)) -
            dhy * getCellUpwindCoeff(cY(i, j, k, n), cY(i, j + 1, k, n)) -
            dhz * getCellUpwindCoeff(cZ(i, j, k, n), cZ(i, j, k + 1, n));
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_flux_x(
  Box const& box,
  Array4<Real> const& fx,
  Array4<Real const> const& sol,
  Array4<Real const> const& bx,
  Array4<Real const> const& cx,
  Real fac,
  Real facc,
  int ncomp) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        AMREX_PRAGMA_SIMD
        for (int i = lo.x; i <= hi.x; ++i) {
          fx(i, j, k, n) =
            -fac * bx(i, j, k, n) * (sol(i, j, k, n) - sol(i - 1, j, k, n)) -
            facc * getFaceUpwindFlux(
                     cx(i, j, k, n), sol(i - 1, j, k, n), sol(i, j, k, n));
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_flux_xface(
  Box const& box,
  Array4<Real> const& fx,
  Array4<Real const> const& sol,
  Array4<Real const> const& bx,
  Array4<Real const> const& cx,
  Real fac,
  Real facc,
  int xlen,
  int ncomp) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        int i = lo.x;
        fx(i, j, k, n) =
          -fac * bx(i, j, k, n) * (sol(i, j, k, n) - sol(i - 1, j, k, n)) -
          facc * getFaceUpwindFlux(
                   cx(i, j, k, n), sol(i - 1, j, k, n), sol(i, j, k, n));
        i += xlen;
        fx(i, j, k, n) =
          -fac * bx(i, j, k, n) * (sol(i, j, k, n) - sol(i - 1, j, k, n)) -
          facc * getFaceUpwindFlux(
                   cx(i, j, k, n), sol(i - 1, j, k, n), sol(i, j, k, n));
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_flux_y(
  Box const& box,
  Array4<Real> const& fy,
  Array4<Real const> const& sol,
  Array4<Real const> const& by,
  Array4<Real const> const& cy,
  Real fac,
  Real facc,
  int ncomp) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        AMREX_PRAGMA_SIMD
        for (int i = lo.x; i <= hi.x; ++i) {
          fy(i, j, k, n) =
            -fac * by(i, j, k, n) * (sol(i, j, k, n) - sol(i, j - 1, k, n)) -
            facc * getFaceUpwindFlux(
                     cy(i, j, k, n), sol(i, j - 1, k, n), sol(i, j, k, n));
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_flux_yface(
  Box const& box,
  Array4<Real> const& fy,
  Array4<Real const> const& sol,
  Array4<Real const> const& by,
  Array4<Real const> const& cy,
  Real fac,
  Real facc,
  int ylen,
  int ncomp) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      int j = lo.y;
      AMREX_PRAGMA_SIMD
      for (int i = lo.x; i <= hi.x; ++i) {
        fy(i, j, k, n) =
          -fac * by(i, j, k, n) * (sol(i, j, k, n) - sol(i, j - 1, k, n)) -
          facc * getFaceUpwindFlux(
                   cy(i, j, k, n), sol(i, j - 1, k, n), sol(i, j, k, n));
      }
      j += ylen;
      AMREX_PRAGMA_SIMD
      for (int i = lo.x; i <= hi.x; ++i) {
        fy(i, j, k, n) =
          -fac * by(i, j, k, n) * (sol(i, j, k, n) - sol(i, j - 1, k, n)) -
          facc * getFaceUpwindFlux(
                   cy(i, j, k, n), sol(i, j - 1, k, n), sol(i, j, k, n));
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_flux_z(
  Box const& box,
  Array4<Real> const& fz,
  Array4<Real const> const& sol,
  Array4<Real const> const& bz,
  Array4<Real const> const& cz,
  Real fac,
  Real facc,
  int ncomp) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        AMREX_PRAGMA_SIMD
        for (int i = lo.x; i <= hi.x; ++i) {
          fz(i, j, k, n) =
            -fac * bz(i, j, k, n) * (sol(i, j, k, n) - sol(i, j, k - 1, n)) -
            facc * getFaceUpwindFlux(
                     cz(i, j, k, n), sol(i, j, k - 1, n), sol(i, j, k, n));
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
mlabecceclap_flux_zface(
  Box const& box,
  Array4<Real> const& fz,
  Array4<Real const> const& sol,
  Array4<Real const> const& bz,
  Array4<Real const> const& cz,
  Real fac,
  Real facc,
  int zlen,
  int ncomp) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < ncomp; ++n) {
    int k = lo.z;
    for (int j = lo.y; j <= hi.y; ++j) {
      AMREX_PRAGMA_SIMD
      for (int i = lo.x; i <= hi.x; ++i) {
        fz(i, j, k, n) =
          -fac * bz(i, j, k, n) * (sol(i, j, k, n) - sol(i, j, k - 1, n)) -
          facc * getFaceUpwindFlux(
                   cz(i, j, k, n), sol(i, j, k - 1, n), sol(i, j, k, n));
      }
    }

    k += zlen;
    for (int j = lo.y; j <= hi.y; ++j) {
      AMREX_PRAGMA_SIMD
      for (int i = lo.x; i <= hi.x; ++i) {
        fz(i, j, k, n) =
          -fac * bz(i, j, k, n) * (sol(i, j, k, n) - sol(i, j, k - 1, n)) -
          facc * getFaceUpwindFlux(
                   cz(i, j, k, n), sol(i, j, k - 1, n), sol(i, j, k, n));
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
abeccec_gsrb(
  Box const& box,
  Array4<Real> const& phi,
  Array4<Real const> const& rhs,
  Real alpha,
  Real omega,
  Array4<Real const> const& a,
  Real dhxsq,
  Real dhysq,
  Real dhzsq,
  Real dhx,
  Real dhy,
  Real dhz,
  Array4<Real const> const& bX,
  Array4<Real const> const& bY,
  Array4<Real const> const& bZ,
  Array4<Real const> const& cX,
  Array4<Real const> const& cY,
  Array4<Real const> const& cZ,
  Array4<int const> const& m0,
  Array4<int const> const& m2,
  Array4<int const> const& m4,
  Array4<int const> const& m1,
  Array4<int const> const& m3,
  Array4<int const> const& m5,
  Array4<Real const> const& f0,
  Array4<Real const> const& f2,
  Array4<Real const> const& f4,
  Array4<Real const> const& f1,
  Array4<Real const> const& f3,
  Array4<Real const> const& f5,
  Box const& vbox,
  int redblack,
  int nc) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);
  const auto vlo = amrex::lbound(vbox);
  const auto vhi = amrex::ubound(vbox);

  for (int n = 0; n < nc; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        AMREX_PRAGMA_SIMD
        for (int i = lo.x; i <= hi.x; ++i) {
          if ((i + j + k + redblack) % 2 == 0) {
            Real cf0 = (i == vlo.x and m0(vlo.x - 1, j, k) > 0)
                         ? f0(vlo.x, j, k, n)
                         : 0.0;
            Real cf1 = (j == vlo.y and m1(i, vlo.y - 1, k) > 0)
                         ? f1(i, vlo.y, k, n)
                         : 0.0;
            Real cf2 = (k == vlo.z and m2(i, j, vlo.z - 1) > 0)
                         ? f2(i, j, vlo.z, n)
                         : 0.0;
            Real cf3 = (i == vhi.x and m3(vhi.x + 1, j, k) > 0)
                         ? f3(vhi.x, j, k, n)
                         : 0.0;
            Real cf4 = (j == vhi.y and m4(i, vhi.y + 1, k) > 0)
                         ? f4(i, vhi.y, k, n)
                         : 0.0;
            Real cf5 = (k == vhi.z and m5(i, j, vhi.z + 1) > 0)
                         ? f5(i, j, vhi.z, n)
                         : 0.0;

            Real gamma =
              alpha * a(i, j, k) +
              dhxsq * (bX(i, j, k, n) + bX(i + 1, j, k, n)) +
              dhysq * (bY(i, j, k, n) + bY(i, j + 1, k, n)) +
              dhzsq * (bZ(i, j, k, n) + bZ(i, j, k + 1, n)) -
              dhx * getCellUpwindCoeff(cX(i, j, k, n), cX(i + 1, j, k, n)) -
              dhy * getCellUpwindCoeff(cY(i, j, k, n), cY(i, j + 1, k, n)) -
              dhz * getCellUpwindCoeff(cZ(i, j, k, n), cZ(i, j, k + 1, n));

            Real g_m_d =
              gamma -
              (dhxsq * (bX(i, j, k, n) * cf0 + bX(i + 1, j, k, n) * cf3) +
               dhysq * (bY(i, j, k, n) * cf1 + bY(i, j + 1, k, n) * cf4) +
               dhzsq * (bZ(i, j, k, n) * cf2 + bZ(i, j, k + 1, n) * cf5));

            Real rho =
              dhxsq * (bX(i, j, k, n) * phi(i - 1, j, k, n) +
                       bX(i + 1, j, k, n) * phi(i + 1, j, k, n)) +
              dhysq * (bY(i, j, k, n) * phi(i, j - 1, k, n) +
                       bY(i, j + 1, k, n) * phi(i, j + 1, k, n)) +
              dhzsq * (bZ(i, j, k, n) * phi(i, j, k - 1, n) +
                       bZ(i, j, k + 1, n) * phi(i, j, k + 1, n)) +
              dhx *
                (getLeftCellUpwindCoeff(cX(i, j, k, n), cX(i + 1, j, k, n)) *
                   phi(i - 1, j, k, n) +
                 getRightCellUpwindCoeff(cX(i, j, k, n), cX(i + 1, j, k, n)) *
                   phi(i + 1, j, k, n)) +
              dhy *
                (getLeftCellUpwindCoeff(cY(i, j, k, n), cY(i, j + 1, k, n)) *
                   phi(i, j - 1, k, n) +
                 getRightCellUpwindCoeff(cY(i, j, k, n), cY(i, j + 1, k, n)) *
                   phi(i, j + 1, k, n)) +
              dhz *
                (getLeftCellUpwindCoeff(cZ(i, j, k, n), cZ(i, j, k + 1, n)) *
                   phi(i, j, k - 1, n) +
                 getRightCellUpwindCoeff(cZ(i, j, k, n), cZ(i, j, k + 1, n)) *
                   phi(i, j, k + 1, n));

            Real res = rhs(i, j, k, n) - (gamma * phi(i, j, k, n) - rho);
            phi(i, j, k, n) = phi(i, j, k, n) + omega / g_m_d * res;
          }
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
abeccec_gsrb_os(
  Box const& box,
  Array4<Real> const& phi,
  Array4<Real const> const& rhs,
  Real alpha,
  Real omega,
  Array4<Real const> const& a,
  Real dhxsq,
  Real dhysq,
  Real dhzsq,
  Real dhx,
  Real dhy,
  Real dhz,
  Array4<Real const> const& bX,
  Array4<Real const> const& bY,
  Array4<Real const> const& bZ,
  Array4<Real const> const& cX,
  Array4<Real const> const& cY,
  Array4<Real const> const& cZ,
  Array4<int const> const& m0,
  Array4<int const> const& m2,
  Array4<int const> const& m4,
  Array4<int const> const& m1,
  Array4<int const> const& m3,
  Array4<int const> const& m5,
  Array4<Real const> const& f0,
  Array4<Real const> const& f2,
  Array4<Real const> const& f4,
  Array4<Real const> const& f1,
  Array4<Real const> const& f3,
  Array4<Real const> const& f5,
  Array4<int const> const& osm,
  Box const& vbox,
  int redblack,
  int nc) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);
  const auto vlo = amrex::lbound(vbox);
  const auto vhi = amrex::ubound(vbox);

  for (int n = 0; n < nc; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        AMREX_PRAGMA_SIMD
        for (int i = lo.x; i <= hi.x; ++i) {
          if ((i + j + k + redblack) % 2 == 0) {
            if (osm(i, j, k) == 0) {
              phi(i, j, k, n) = 0.0;
            } else {
              Real cf0 = (i == vlo.x and m0(vlo.x - 1, j, k) > 0)
                           ? f0(vlo.x, j, k, n)
                           : 0.0;
              Real cf1 = (j == vlo.y and m1(i, vlo.y - 1, k) > 0)
                           ? f1(i, vlo.y, k, n)
                           : 0.0;
              Real cf2 = (k == vlo.z and m2(i, j, vlo.z - 1) > 0)
                           ? f2(i, j, vlo.z, n)
                           : 0.0;
              Real cf3 = (i == vhi.x and m3(vhi.x + 1, j, k) > 0)
                           ? f3(vhi.x, j, k, n)
                           : 0.0;
              Real cf4 = (j == vhi.y and m4(i, vhi.y + 1, k) > 0)
                           ? f4(i, vhi.y, k, n)
                           : 0.0;
              Real cf5 = (k == vhi.z and m5(i, j, vhi.z + 1) > 0)
                           ? f5(i, j, vhi.z, n)
                           : 0.0;

              Real gamma =
                alpha * a(i, j, k) +
                dhxsq * (bX(i, j, k, n) + bX(i + 1, j, k, n)) +
                dhysq * (bY(i, j, k, n) + bY(i, j + 1, k, n)) +
                dhzsq * (bZ(i, j, k, n) + bZ(i, j, k + 1, n)) -
                dhx * getCellUpwindCoeff(cX(i, j, k, n), cX(i + 1, j, k, n)) -
                dhy * getCellUpwindCoeff(cY(i, j, k, n), cY(i, j + 1, k, n)) -
                dhz * getCellUpwindCoeff(cZ(i, j, k, n), cZ(i, j, k + 1, n));

              Real g_m_d =
                gamma -
                (dhxsq * (bX(i, j, k, n) * cf0 + bX(i + 1, j, k, n) * cf3) +
                 dhysq * (bY(i, j, k, n) * cf1 + bY(i, j + 1, k, n) * cf4) +
                 dhzsq * (bZ(i, j, k, n) * cf2 + bZ(i, j, k + 1, n) * cf5));

              Real rho =
                dhxsq * (bX(i, j, k, n) * phi(i - 1, j, k, n) +
                         bX(i + 1, j, k, n) * phi(i + 1, j, k, n)) +
                dhysq * (bY(i, j, k, n) * phi(i, j - 1, k, n) +
                         bY(i, j + 1, k, n) * phi(i, j + 1, k, n)) +
                dhzsq * (bZ(i, j, k, n) * phi(i, j, k - 1, n) +
                         bZ(i, j, k + 1, n) * phi(i, j, k + 1, n)) +
                dhx *
                  (getLeftCellUpwindCoeff(cX(i, j, k, n), cX(i + 1, j, k, n)) *
                     phi(i - 1, j, k, n) +
                   getRightCellUpwindCoeff(cX(i, j, k, n), cX(i + 1, j, k, n)) *
                     phi(i + 1, j, k, n)) +
                dhy *
                  (getLeftCellUpwindCoeff(cY(i, j, k, n), cY(i, j + 1, k, n)) *
                     phi(i, j - 1, k, n) +
                   getRightCellUpwindCoeff(cY(i, j, k, n), cY(i, j + 1, k, n)) *
                     phi(i, j + 1, k, n)) +
                dhz *
                  (getLeftCellUpwindCoeff(cZ(i, j, k, n), cZ(i, j, k + 1, n)) *
                     phi(i, j, k - 1, n) +
                   getRightCellUpwindCoeff(cZ(i, j, k, n), cZ(i, j, k + 1, n)) *
                     phi(i, j, k + 1, n));

              Real res = rhs(i, j, k, n) - (gamma * phi(i, j, k, n) - rho);
              phi(i, j, k, n) = phi(i, j, k, n) + omega / g_m_d * res;
            }
          }
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
tridiagonal_solve(
  Array1D<Real, 0, 31>& a_ls,
  Array1D<Real, 0, 31>& b_ls,
  Array1D<Real, 0, 31>& c_ls,
  Array1D<Real, 0, 31>& r_ls,
  Array1D<Real, 0, 31>& u_ls,
  Array1D<Real, 0, 31>& gam,
  int ilen) noexcept
{
  Real bet = b_ls(0);
  u_ls(0) = r_ls(0) / bet;

  for (int i = 1; i <= ilen - 1; ++i) {
    gam(i) = c_ls(i - 1) / bet;
    bet = b_ls(i) - a_ls(i) * gam(i);
    if (bet == 0)
      amrex::Abort(">>>TRIDIAG FAILED");
    u_ls(i) = (r_ls(i) - a_ls(i) * u_ls(i - 1)) / bet;
  }
  for (int i = ilen - 2; i >= 0; --i) {
    u_ls(i) = u_ls(i) - gam(i + 1) * u_ls(i + 1);
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
coarsen_overset_mask(
  Box const& bx,
  Array4<int> const& cmsk,
  Array4<int const> const& fmsk) noexcept
{
  int nerrors = 0;
  const auto lo = amrex::lbound(bx);
  const auto hi = amrex::ubound(bx);
  for (int k = lo.z; k <= hi.z; ++k) {
    for (int j = lo.y; j <= hi.y; ++j) {
      for (int i = lo.x; i <= hi.x; ++i) {
        int ii = 2 * i;
        int jj = 2 * j;
        int kk = 2 * k;
        cmsk(i, j, k) = fmsk(ii, jj, kk) + fmsk(ii + 1, jj, kk) +
                        fmsk(ii, jj + 1, kk) + fmsk(ii + 1, jj + 1, kk) +
                        fmsk(ii, jj, kk + 1) + fmsk(ii + 1, jj, kk + 1) +
                        fmsk(ii, jj + 1, kk + 1) + fmsk(ii + 1, jj + 1, kk + 1);
        if (cmsk(i, j, k) == 8) {
          cmsk(i, j, k) = 1;
        } else if (cmsk(i, j, k) != 0) {
          ++nerrors;
        }
      }
    }
  }
  return nerrors;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
overset_rescale_bcoef_x(
  Box const& box,
  Array4<Real> const& bX,
  Array4<int const> const& osm,
  int ncomp,
  Real osfac) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);
  for (int n = 0; n < ncomp; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {
          if ((osm(i - 1, j, k) + osm(i, j, k)) == 1) {
            bX(i, j, k, n) *= osfac;
          }
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
overset_rescale_bcoef_y(
  Box const& box,
  Array4<Real> const& bY,
  Array4<int const> const& osm,
  int ncomp,
  Real osfac) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);
  for (int n = 0; n < ncomp; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {
          if ((osm(i, j - 1, k) + osm(i, j, k)) == 1) {
            bY(i, j, k, n) *= osfac;
          }
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
overset_rescale_bcoef_z(
  Box const& box,
  Array4<Real> const& bZ,
  Array4<int const> const& osm,
  int ncomp,
  Real osfac) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);
  for (int n = 0; n < ncomp; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {
          if ((osm(i, j, k - 1) + osm(i, j, k)) == 1) {
            bZ(i, j, k, n) *= osfac;
          }
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE void
abeccec_chkdiag(
  Box const& box,
  Real alpha,
  Array4<Real const> const& a,
  Real dhxsq,
  Real dhysq,
  Real dhzsq,
  Real dhx,
  Real dhy,
  Real dhz,
  Array4<Real const> const& bX,
  Array4<Real const> const& bY,
  Array4<Real const> const& bZ,
  Array4<Real const> const& cX,
  Array4<Real const> const& cY,
  Array4<Real const> const& cZ,
  int nc) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < nc; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        AMREX_PRAGMA_SIMD
        for (int i = lo.x; i <= hi.x; ++i) {
          // Complete diag
          Real diag =
            alpha * a(i, j, k) -
            dhxsq * (-bX(i, j, k, n) - bX(i + 1, j, k, n)) -
            dhysq * (-bY(i, j, k, n) - bY(i, j + 1, k, n)) -
            dhzsq * (-bZ(i, j, k, n) - bZ(i, j, k + 1, n)) -
            dhx * getCellUpwindCoeff(cX(i, j, k, n), cX(i + 1, j, k, n)) -
            dhy * getCellUpwindCoeff(cY(i, j, k, n), cY(i, j + 1, k, n)) -
            dhz * getCellUpwindCoeff(cZ(i, j, k, n), cZ(i, j, k + 1, n));

          // Offdiag
          Real offdiagdiff = -dhxsq * (bX(i, j, k, n) + bX(i + 1, j, k, n)) -
                             dhysq * (bY(i, j, k, n) + bY(i, j + 1, k, n)) -
                             dhzsq * (bZ(i, j, k, n) + bZ(i, j, k + 1, n));

          Real offdiagadv =
            -dhx *
              (getLeftCellUpwindCoeff(cX(i, j, k, n), cX(i + 1, j, k, n)) +
               getRightCellUpwindCoeff(cX(i, j, k, n), cX(i + 1, j, k, n))) -
            dhy *
              (getLeftCellUpwindCoeff(cY(i, j, k, n), cY(i, j + 1, k, n)) +
               getRightCellUpwindCoeff(cY(i, j, k, n), cY(i, j + 1, k, n))) -
            dhz * (getLeftCellUpwindCoeff(cZ(i, j, k, n), cZ(i, j, k + 1, n)) +
                   getRightCellUpwindCoeff(cZ(i, j, k, n), cZ(i, j, k + 1, n)));

          if (fabs(offdiagdiff + offdiagadv) > fabs(diag)) {
            amrex::Abort("LinOp not diagonal dominant");
          }
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE void
abeccec_getdiag(
  Box const& box,
  Real alpha,
  Array4<Real const> const& a,
  Real dhxsq,
  Real dhysq,
  Real dhzsq,
  Real dhx,
  Real dhy,
  Real dhz,
  Array4<Real const> const& bX,
  Array4<Real const> const& bY,
  Array4<Real const> const& bZ,
  Array4<Real const> const& cX,
  Array4<Real const> const& cY,
  Array4<Real const> const& cZ,
  int nc,
  Array4<Real> const& diag) noexcept
{
  const auto lo = amrex::lbound(box);
  const auto hi = amrex::ubound(box);

  for (int n = 0; n < nc; ++n) {
    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        AMREX_PRAGMA_SIMD
        for (int i = lo.x; i <= hi.x; ++i) {
          diag(i, j, k) =
            alpha * a(i, j, k) -
            dhxsq * (-bX(i, j, k, n) - bX(i + 1, j, k, n)) -
            dhysq * (-bY(i, j, k, n) - bY(i, j + 1, k, n)) -
            dhzsq * (-bZ(i, j, k, n) - bZ(i, j, k + 1, n)) -
            dhx * getCellUpwindCoeff(cX(i, j, k, n), cX(i + 1, j, k, n)) -
            dhy * getCellUpwindCoeff(cY(i, j, k, n), cY(i, j + 1, k, n)) -
            dhz * getCellUpwindCoeff(cZ(i, j, k, n), cZ(i, j, k + 1, n));
        }
      }
    }
  }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE void
abeccec_getnetfcoeff(
  Box const& box,
  Real /*alpha*/,
  Array4<Real const> const& /*a*/,
  Real dhxsq,
  Real dhysq,
  Real dhzsq,
  Real dhx,
  Real dhy,
  Real dhz,
  Array4<Real const> const& bX,
  Array4<Real const> const& bY,
  Array4<Real const> const& bZ,
  Array4<Real const> const& cX,
  Array4<Real const> const& cY,
  Array4<Real const> const& cZ,
  int nc,
  int dir,
  Array4<Real> const& coeffd) noexcept
{
  auto lo = amrex::lbound(box);
  auto hi = amrex::ubound(box);
  // Getting face coefficient
  if (dir == 0) {
    for (int n = 0; n < nc; ++n) {
      for (int k = lo.z; k <= hi.z; ++k) {
        for (int j = lo.y; j <= hi.y; ++j) {
          AMREX_PRAGMA_SIMD
          for (int i = lo.x; i <= hi.x; ++i) {
            coeffd(i, j, k) = -dhxsq * bX(i, j, k, n) - dhx * cX(i, j, k, n);
          }
        }
      }
    }
  } else if (dir == 1) {
    for (int n = 0; n < nc; ++n) {
      for (int k = lo.z; k <= hi.z; ++k) {
        for (int j = lo.y; j <= hi.y; ++j) {
          AMREX_PRAGMA_SIMD
          for (int i = lo.x; i <= hi.x; ++i) {
            coeffd(i, j, k) = -dhysq * bY(i, j, k, n) - dhy * cY(i, j, k, n);
          }
        }
      }
    }
  } else if (dir == 2) {
    for (int n = 0; n < nc; ++n) {
      for (int k = lo.z; k <= hi.z; ++k) {
        for (int j = lo.y; j <= hi.y; ++j) {
          AMREX_PRAGMA_SIMD
          for (int i = lo.x; i <= hi.x; ++i) {
            coeffd(i, j, k) = -dhzsq * bZ(i, j, k, n) - dhz * cZ(i, j, k, n);
          }
        }
      }
    }
  }
}

} // namespace amrex
#endif
