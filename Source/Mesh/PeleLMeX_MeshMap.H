#ifndef PELELMEX_MESHMAP_H
#define PELELMEX_MESHMAP_H

#include <cmath>
#include <memory>
#include <string>

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FabFactory.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H>
#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>

#include <PeleLMeX_MeshMapEvaluator.H>

/**
 * \brief Abstract mesh-mapping base class.
 *
 * The MeshMap object owns per-level storage for the metric fields associated
 * with a single-block, diagonal, axis-aligned mesh mapping
 *     X = Phi(Xi) = (x(chi), y(eta), z(zeta))
 * on top of the uniform AMReX mesh.  Faces remain axis-aligned; the only
 * new quantities are the per-direction stretch factors fac_i and the
 * Jacobian detJ = prod_i fac_i.
 *
 * Each level stores, at each of CC / ND / XF / YF / ZF locations:
 *   - fac  : AMREX_SPACEDIM-component field giving per-direction stretch
 *   - detJ : 1-component field giving the Jacobian determinant
 *
 * The concrete class implements create_map(lev, geom) which fills these
 * metric fields from the geometry.
 *
 * Mirrors the design of amr-wind/core/MeshMap.H (PR #545 in Exawind/amr-wind)
 * but stores MultiFabs directly on the object rather than going through a
 * FieldRepo, since PeleLMeX uses raw AMReX MultiFabs for its state.
 */
class MeshMap
{
public:
  MeshMap() = default;
  virtual ~MeshMap() = default;
  MeshMap(const MeshMap&) = delete;
  MeshMap& operator=(const MeshMap&) = delete;
  MeshMap(MeshMap&&) = delete;
  MeshMap& operator=(MeshMap&&) = delete;

  /**
   * \brief Resize the per-level storage vectors to hold \p nlev entries.
   *
   * Called by PeleLM when the level count changes (after MakeNewLevel,
   * ClearLevel, or regrid).  Individual levels are populated via define().
   */
  void resize(int nlev);

  /**
   * \brief Clear the MultiFabs stored at level \p lev.
   *
   * Equivalent to amrex::MultiFab::clear() on every per-location field.
   * Invoked from PeleLM::ClearLevel().
   */
  void clear_level(int lev);

  /**
   * \brief Define (allocate) all metric MultiFabs on level \p lev.
   *
   * \p nghost is the number of ghost cells to request; matches the
   * maximum ghost-cell requirement of the consumers (projectors,
   * advection, diffusion).
   */
  void define(
    int lev,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm,
    const amrex::FabFactory<amrex::FArrayBox>& factory,
    int nghost);

  /**
   * \brief Fill the metric fields on level \p lev from \p geom.
   *
   * Must be implemented by every concrete derived class.
   */
  virtual void create_map(int lev, const amrex::Geometry& geom) = 0;

  /**
   * \brief Fill a nodal MultiFab with the per-direction displacement
   *        from the Xi-space node position to the physical-space node
   *        position.
   *
   * Used when writing plotfiles with mesh-mapping metadata so that
   * ParaView / VisIt's AMReX reader can render the solution on the
   * curvilinear physical grid (see PeleLMeX_Plot.cpp's
   * writePlotfileWithMapping).  The convention is
   *     disp_i(node) = x_phys_i(node) - x_xi_i(node)
   *                  = int_{prob_lo_i}^{x_xi_i(node)} (fac_i(xi) - 1) dxi_i
   *
   * \p disp_nd must be a nodal MultiFab with at least AMREX_SPACEDIM
   * components.  Only the first AMREX_SPACEDIM components are written.
   *
   * This is pure virtual by design.  It used to carry a default
   * implementation that assumed a piecewise-constant stretch factor,
   *     disp_i(node) = (fac_i(node) - 1) * (x_xi_i(node) - prob_lo_i),
   * which is exact only for ConstantMap; any map with spatially varying
   * fac that forgot to override it inherited a silently wrong plotfile
   * geometry (off by 85% of the axis length for TanhStretchMap at
   * beta = 2, 156% for ExpStretchMap at beta = 2.3).  A wrong grid in a
   * plotfile looks plausible, so the mistake is worth a compile error
   * rather than a runtime surprise.  \c fill_nodal_disp_from() below makes
   * the correct implementation a two-liner.
   */
  virtual void fill_nodal_displacement(
    int lev, const amrex::Geometry& geom, amrex::MultiFab& disp_nd) const = 0;

  /**
   * \brief Return a device-callable evaluator capable of producing the
   *        physical-space coordinate of any cell centre or face from
   *        its integer index plus the corresponding GeometryData.
   *
   * The base implementation returns the identity (\c Kind::Identity);
   * each concrete map populates a richer payload.  Held by value by
   * the PeleLM driver and captured by value into device lambdas, this
   * is the recommended way for user \c initdata_mapped /
   * \c bcnormal code to obtain physical positions in a mapping-aware
   * but mapping-agnostic fashion.
   */
  [[nodiscard]] virtual MeshMapEvaluator make_evaluator() const
  {
    return MeshMapEvaluator{};
  }

  /** \return the number of active levels currently allocated. */
  [[nodiscard]] int num_levels() const
  {
    return static_cast<int>(m_fac_cc.size());
  }

  // ---- Cell-centered accessors -------------------------------------------
  amrex::MultiFab& fac_cc(int lev) { return m_fac_cc[lev]; }
  amrex::MultiFab& detJ_cc(int lev) { return m_detJ_cc[lev]; }
  [[nodiscard]] const amrex::MultiFab& fac_cc(int lev) const
  {
    return m_fac_cc[lev];
  }
  [[nodiscard]] const amrex::MultiFab& detJ_cc(int lev) const
  {
    return m_detJ_cc[lev];
  }

  // ---- Nodal accessors ---------------------------------------------------
  amrex::MultiFab& fac_nd(int lev) { return m_fac_nd[lev]; }
  amrex::MultiFab& detJ_nd(int lev) { return m_detJ_nd[lev]; }
  [[nodiscard]] const amrex::MultiFab& fac_nd(int lev) const
  {
    return m_fac_nd[lev];
  }
  [[nodiscard]] const amrex::MultiFab& detJ_nd(int lev) const
  {
    return m_detJ_nd[lev];
  }

  // ---- Face-centered accessors (per direction) --------------------------
  amrex::MultiFab& fac_fc(int lev, int idim) { return m_fac_fc[lev][idim]; }
  amrex::MultiFab& detJ_fc(int lev, int idim) { return m_detJ_fc[lev][idim]; }
  [[nodiscard]] const amrex::MultiFab& fac_fc(int lev, int idim) const
  {
    return m_fac_fc[lev][idim];
  }
  [[nodiscard]] const amrex::MultiFab& detJ_fc(int lev, int idim) const
  {
    return m_detJ_fc[lev][idim];
  }

  /// Array accessor: returns references to the per-direction face MFs on a
  /// level.
  amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> fac_fc(int lev)
  {
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> r;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      r[d] = &m_fac_fc[lev][d];
    }
    return r;
  }
  amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> detJ_fc(int lev)
  {
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> r;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      r[d] = &m_detJ_fc[lev][d];
    }
    return r;
  }

  /**
   * \brief Factory method: build a concrete MeshMap by name.
   *
   * Recognised names:
   *   - "ConstantMap"        : uniform per-direction stretching
   *   - "ExpStretchMap"      : one-sided (single-axis) exponential mesh
   *                            stretching
   *   - "TanhStretchMap"     : two-sided tanh-based mesh stretching
   *                            (clusters at both ends of an axis)
   *   - "InteriorStretchMap" : interior-clustered stretching with
   *                            independent per-side coarsening rates
   *
   * Aborts with an informative message if the name is not recognised.
   */
  static std::unique_ptr<MeshMap> create(const std::string& name);

  /**
   * \brief Fill \c fac and \c detJ at every storage location on level
   *        \p lev from a single per-axis stretch-factor callable.
   *
   * \p fac_at must be device-callable with the signature
   * \code
   *   amrex::Real fac_at(int idir, amrex::Real xi)
   * \endcode
   * returning \f$dx_{\mathrm{phys}}/d\xi\f$ along axis \p idir at the
   * Xi-space coordinate \p xi.  This helper owns the whole
   * CC / ND / XF / YF / ZF traversal, the per-location index-to-\f$\xi\f$
   * offsets (0 at a node, 0.5 at a cell centre, and node-along-the-normal /
   * centre-transverse on faces), and
   * \f$\det J = \prod_i \mathrm{fac}_i\f$ --- so a concrete map's
   * \c create_map() is just the formula plus one call.
   */
  template <typename FacAt>
  void fill_metrics(int lev, const amrex::Geometry& geom, FacAt const& fac_at);

  /**
   * \brief Implement \c fill_nodal_displacement() from a per-axis
   *        physical-coordinate callable.
   *
   * \p x_phys_at must be device-callable with the signature
   * \code
   *   amrex::Real x_phys_at(int idir, amrex::Real xi)
   * \endcode
   * returning the physical coordinate along axis \p idir mapped from the
   * Xi-space coordinate \p xi.  The displacement written is
   * \c x_phys_at(d, xi_node) - xi_node.
   */
  template <typename XPhysAt>
  void fill_nodal_disp_from(
    int lev,
    const amrex::Geometry& geom,
    amrex::MultiFab& disp_nd,
    XPhysAt const& x_phys_at) const;

protected:
  amrex::Vector<amrex::MultiFab> m_fac_cc;
  amrex::Vector<amrex::MultiFab> m_detJ_cc;

  amrex::Vector<amrex::MultiFab> m_fac_nd;
  amrex::Vector<amrex::MultiFab> m_detJ_nd;

  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> m_fac_fc;
  amrex::Vector<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>> m_detJ_fc;
};

// ---- Template helper definitions ---------------------------------------

template <typename FacAt>
void
MeshMap::fill_metrics(int lev, const amrex::Geometry& geom, FacAt const& fac_at)
{
  AMREX_ASSERT(lev >= 0 && lev < num_levels());

  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{}, dxi{};
  amrex::GpuArray<int, AMREX_SPACEDIM> dlo{}, ncell{}, per{};
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    plo[d] = geom.ProbLo(d);
    dxi[d] = geom.CellSize(d);
    dlo[d] = geom.Domain().smallEnd(d);
    ncell[d] = geom.Domain().length(d);
    per[d] = geom.isPeriodic(d) ? 1 : 0;
  }

  // fac on axis d at integer index n with in-cell offset off
  // (0 = lower node, 0.5 = cell centre, 1 = upper node).
  //
  // Across a periodic boundary the ghost cells are the periodic images of
  // interior cells and must carry their metric, not the map continued past
  // the domain end: consumers that read metric ghost cells directly (the
  // scaling of the inflow ghost velocity for the nodal projection, the
  // Godunov predictor and advective extrapolation) would otherwise see a
  // different value on one side of the seam.  The wrap is done here, on
  // the index, so that it also covers cells outside the domain in more
  // than one direction (inflow-layer corners), which a FillBoundary of
  // the metric MultiFabs cannot reach.  Indices inside the domain (and
  // the upper node/face N) are untouched; non-periodic directions keep
  // the analytic extension, which inflow faces rely on.
  auto at = [=] AMREX_GPU_HOST_DEVICE(
              int d, int n, amrex::Real off) noexcept -> amrex::Real {
    if (per[d] != 0) {
      const int nmax = (off > amrex::Real(0.25)) ? ncell[d] - 1 : ncell[d];
      int k = n - dlo[d];
      if (k < 0 || k > nmax) {
        k -= ncell[d] * ((k < 0) ? -((-k - 1) / ncell[d] + 1) : k / ncell[d]);
      }
      n = k + dlo[d];
    }
    return fac_at(d, plo[d] + (static_cast<amrex::Real>(n) + off) * dxi[d]);
  };

  // ---- Cell-centered: every axis at CC (offset 0.5) ----
  {
    const auto& fac_ma = m_fac_cc[lev].arrays();
    const auto& det_ma = m_detJ_cc[lev].arrays();
    amrex::ParallelFor(
      m_fac_cc[lev], m_fac_cc[lev].nGrowVect(),
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        auto f = fac_ma[box_no];
        AMREX_D_TERM(
          const amrex::Real fx = at(0, i, amrex::Real(0.5));
          , const amrex::Real fy = at(1, j, amrex::Real(0.5));
          , const amrex::Real fz = at(2, k, amrex::Real(0.5)););
        AMREX_D_TERM(f(i, j, k, 0) = fx;, f(i, j, k, 1) = fy;
                     , f(i, j, k, 2) = fz;);
        det_ma[box_no](i, j, k) = AMREX_D_TERM(fx, *fy, *fz);
      });
  }

  // ---- Nodal: every axis at ND (offset 0.0) ----
  {
    const auto& fac_ma = m_fac_nd[lev].arrays();
    const auto& det_ma = m_detJ_nd[lev].arrays();
    amrex::ParallelFor(
      m_fac_nd[lev], m_fac_nd[lev].nGrowVect(),
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        auto f = fac_ma[box_no];
        AMREX_D_TERM(
          const amrex::Real fx = at(0, i, amrex::Real(0.0));
          , const amrex::Real fy = at(1, j, amrex::Real(0.0));
          , const amrex::Real fz = at(2, k, amrex::Real(0.0)););
        AMREX_D_TERM(f(i, j, k, 0) = fx;, f(i, j, k, 1) = fy;
                     , f(i, j, k, 2) = fz;);
        det_ma[box_no](i, j, k) = AMREX_D_TERM(fx, *fy, *fz);
      });
  }

  // ---- Face-centered, per face direction dface.  On axis dface the
  // ---- position is at a node (offset 0); on the other axes it is cc
  // ---- (offset 0.5). ----
  for (int dface = 0; dface < AMREX_SPACEDIM; ++dface) {
    const int dface_cap = dface;
    const auto& fac_ma = m_fac_fc[lev][dface].arrays();
    const auto& det_ma = m_detJ_fc[lev][dface].arrays();
    amrex::ParallelFor(
      m_fac_fc[lev][dface], m_fac_fc[lev][dface].nGrowVect(),
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        auto f = fac_ma[box_no];
        const amrex::Real off_x =
          (dface_cap == 0) ? amrex::Real(0.0) : amrex::Real(0.5);
#if (AMREX_SPACEDIM >= 2)
        const amrex::Real off_y =
          (dface_cap == 1) ? amrex::Real(0.0) : amrex::Real(0.5);
#endif
#if (AMREX_SPACEDIM == 3)
        const amrex::Real off_z =
          (dface_cap == 2) ? amrex::Real(0.0) : amrex::Real(0.5);
#endif
        AMREX_D_TERM(
          const amrex::Real fx = at(0, i, off_x);
          , const amrex::Real fy = at(1, j, off_y);
          , const amrex::Real fz = at(2, k, off_z););
        AMREX_D_TERM(f(i, j, k, 0) = fx;, f(i, j, k, 1) = fy;
                     , f(i, j, k, 2) = fz;);
        det_ma[box_no](i, j, k) = AMREX_D_TERM(fx, *fy, *fz);
      });
  }

  amrex::Gpu::streamSynchronize();

  // A map used across a periodic direction must give the same cell size on
  // both sides of the seam: fac at Xi_lo and at Xi_hi.  A two-sided
  // (wall-clustering) map does; a one-sided one leaves a jump of
  // fac_hi/fac_lo in cell size at the seam, which is consistent (the seam
  // then looks like an abrupt coarsening) but first-order there.
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    if (geom.isPeriodic(d)) {
      const amrex::Real flo = fac_at(d, geom.ProbLo(d));
      const amrex::Real fhi = fac_at(d, geom.ProbHi(d));
      if (
        std::abs(fhi - flo) >
        amrex::Real(1.e-10) * amrex::max(std::abs(flo), std::abs(fhi))) {
        amrex::Print() << " WARNING: mesh mapping: cell size jumps by a factor "
                       << fhi / flo << " across the periodic seam in direction "
                       << d << " (fac = " << flo << " at lo, " << fhi
                       << " at hi)\n";
      }
    }
  }
}

template <typename XPhysAt>
void
MeshMap::fill_nodal_disp_from(
  int lev,
  const amrex::Geometry& geom,
  amrex::MultiFab& disp_nd,
  XPhysAt const& x_phys_at) const
{
  AMREX_ALWAYS_ASSERT(lev >= 0 && lev < num_levels());
  AMREX_ASSERT(disp_nd.nComp() >= AMREX_SPACEDIM);
  AMREX_ASSERT_WITH_MESSAGE(
    disp_nd.boxArray().ixType().nodeCentered(),
    "MeshMap::fill_nodal_disp_from: output MultiFab must be nodal");

  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{}, dxi{};
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    plo[d] = geom.ProbLo(d);
    dxi[d] = geom.CellSize(d);
  }

  auto disp_at =
    [=] AMREX_GPU_HOST_DEVICE(int d, int n) noexcept -> amrex::Real {
    const amrex::Real xi = plo[d] + static_cast<amrex::Real>(n) * dxi[d];
    return x_phys_at(d, xi) - xi;
  };

  const auto& disp_ma = disp_nd.arrays();
  amrex::ParallelFor(
    disp_nd, amrex::IntVect(0),
    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      auto d = disp_ma[box_no];
      AMREX_D_TERM(d(i, j, k, 0) = disp_at(0, i);
                   , d(i, j, k, 1) = disp_at(1, j);
                   , d(i, j, k, 2) = disp_at(2, k););
    });
  amrex::Gpu::streamSynchronize();
}

#endif /* PELELMEX_MESHMAP_H */
