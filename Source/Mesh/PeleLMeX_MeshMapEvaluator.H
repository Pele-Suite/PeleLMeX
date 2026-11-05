#ifndef PELELMEX_MESHMAP_EVALUATOR_H
#define PELELMEX_MESHMAP_EVALUATOR_H

// The mesh-mapping descriptor lives in PelePhysics
// (Source/Utility/MeshMap/MeshMapEvaluator.H) so that the turbulent-inflow
// reader and the inflow-file generators share the solver's definition of
// the mapping.  PeleLMeX code keeps referring to it by its unqualified name.
#include <MeshMapEvaluator.H>

using MeshMapEvaluator = pele::physics::MeshMapEvaluator;

/**
 * \brief Physical cell-centre coordinate along \p idir for a possibly
 *        out-of-domain index, with periodic directions wrapped.
 *
 * MeshMapEvaluator::x_phys_cc continues the map past the domain end.  For
 * a ghost cell across a periodic boundary the physical position must be
 * the periodic image of the interior cell it represents (x of that cell
 * shifted by the physical period); a non-linear map continued past the
 * end gives a different point, and inflow values written by position at
 * such ghost cells (bcnormal, turbulent inflow) then differ between the
 * two periodic copies of the same cell, which the nodal projection
 * cannot reconcile.  Non-periodic directions keep the continued map,
 * which inflow faces rely on.  Identical to x_phys_cc for indices inside
 * the domain, and bypassed (bit-identical positions) for the identity map,
 * where the continued map already is the periodic image.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
x_phys_cc_periodic(
  MeshMapEvaluator const& mmap,
  int idir,
  int idx,
  amrex::GeometryData const& gd) noexcept
{
  const int lo = gd.Domain().smallEnd(idir);
  const int n = gd.Domain().length(idir);
  if (
    mmap.m_kind != MeshMapEvaluator::Kind::Identity &&
    gd.isPeriodic(idir) != 0 && (idx < lo || idx >= lo + n)) {
    // shift is the number of periods between idx and its image in the domain
    const int k = idx - lo;
    const int shift = (k >= 0) ? k / n : -((-k - 1) / n + 1);
    return mmap.x_phys_cc(idir, idx - shift * n, gd) +
           static_cast<amrex::Real>(shift) * mmap.domain_size_phys(idir, gd);
  }
  return mmap.x_phys_cc(idir, idx, gd);
}

/// True for the identity evaluator (no mesh mapping active).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
is_identity_map(MeshMapEvaluator const& mmap) noexcept
{
  return mmap.m_kind == MeshMapEvaluator::Kind::Identity;
}

/**
 * \brief Inverse physical spacing for a centred difference at cell \p idx
 *        along \p idir: 2 / (x_{idx+1} - x_{idx-1}), so that
 *        (a_{i+1} - a_{i-1}) * 0.5 * result is the physical derivative.
 *
 * Returns \p inv_dx_uniform for the identity map, keeping unmapped stencils
 * bit-identical.  Neighbour positions come from x_phys_cc_periodic, so the
 * spacing across a periodic seam is the distance to the periodic image.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
mapped_inv_dx(
  int idir,
  int idx,
  amrex::Real inv_dx_uniform,
  MeshMapEvaluator const& mmap,
  amrex::GeometryData const& gd) noexcept
{
  if (is_identity_map(mmap)) {
    return inv_dx_uniform;
  }
  return amrex::Real(2.0) / (x_phys_cc_periodic(mmap, idir, idx + 1, gd) -
                             x_phys_cc_periodic(mmap, idir, idx - 1, gd));
}

/// mapped_inv_dx in every direction at cell (i,j,k).
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
mapped_inv_dx_array(
  int i,
  int j,
  int k,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& inv_dx_uniform,
  MeshMapEvaluator const& mmap,
  amrex::GeometryData const& gd) noexcept
{
  amrex::ignore_unused(j, k);
  return {AMREX_D_DECL(
    mapped_inv_dx(0, i, inv_dx_uniform[0], mmap, gd),
    mapped_inv_dx(1, j, inv_dx_uniform[1], mmap, gd),
    mapped_inv_dx(2, k, inv_dx_uniform[2], mmap, gd))};
}

#endif // PELELMEX_MESHMAP_EVALUATOR_H
