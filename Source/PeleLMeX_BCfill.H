#ifndef PELELM_BCFILL
#define PELELM_BCFILL
#include <PeleLMeX_Index.H>
#include <PeleLMeX_MeshMapEvaluator.H>
#include <pelelmex_prob.H>
#include <PMFData.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

using namespace amrex::literals;

// Compute the boundary-face coordinate array x[] that gets passed to
// the user's bcnormal.  The boundary-direction component (d == idir)
// is taken at the actual physical face; the tangential components
// (d != idir) are taken at the ghost-cell centre (which has the same
// tangential index as the matching interior cell).
//
// When the supplied MeshMapEvaluator is Kind::Identity, x[idir] is
// `prob_lo[idir] + face_idx*dx[idir]` and x[d!=idir] is
// `prob_lo[d] + (iv[d]+0.5)*dx[d]`.  When a mapping is active the
// returned values are the physical-space coordinates under that
// mapping, so user bcnormal code that branches on x[] sees the
// physical face position rather than the xi-space position.
//
// Note: this corrects a latent half-cell offset in the legacy
// computation, which used the ghost-cell centre for all components
// (i.e. `x[idir] = prob_lo + (iv[idir]+0.5)*dx`, half a cell outside
// the actual wall).  User code that branched on `x[idir]` via a
// loose threshold (e.g. `x[idir] > 0.99*L`) is unaffected; code that
// compares to an exact half-cell offset will see the wall position.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
pelelm_bc_face_x(
  amrex::IntVect const& iv,
  int idir,
  int sgn,
  amrex::GeometryData const& geom,
  MeshMapEvaluator const& mmap,
  amrex::Real x[AMREX_SPACEDIM]) noexcept
{
  // sgn = +1 means low side: ghost cell at iv[idir] = domlo-1, face at
  //                          integer index iv[idir] + 1 = domlo.
  // sgn = -1 means high side: ghost cell at iv[idir] = domhi+1, face at
  //                          integer index iv[idir] = domhi+1.
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    if (d == idir) {
      const int face_idx = (sgn > 0) ? iv[d] + 1 : iv[d];
      x[d] = mmap.x_phys_face(d, face_idx, geom);
    } else {
      x[d] = x_phys_cc_periodic(mmap, d, iv[d], geom);
    }
  }
}

template <typename T>
struct PeleLMCCFillExtDirState
{

  ProbParm const* lprobparm;
  pele::physics::PMF::PmfData::DataContainer const* lpmfdata;
  const int m_nAux;
  int m_do_turbInflow = 0;
  MeshMapEvaluator m_map_eval{}; // defaults to Identity (no mesh mapping)
  int m_do_inlet_from_plane = 0;
  int m_inlet_from_plane_dir = -1;
  // When != 0 (peleLM.recycling_mode = full), the pre-loaded ghost value on
  // the recycling face is the complete inflow velocity and is preserved
  // against whatever bcnormal writes to the velocity components there.
  int m_recycling_full = 0;

  AMREX_GPU_HOST
  constexpr PeleLMCCFillExtDirState(
    ProbParm const* a_prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* a_pmf_data,
    int a_nAux,
    int do_turbInflow,
    int do_inletFromPlane,
    int inlet_plane_dir,
    int recycling_full,
    MeshMapEvaluator a_map_eval = MeshMapEvaluator{})
    : lprobparm(a_prob_parm),
      lpmfdata(a_pmf_data),
      m_nAux(a_nAux),
      m_do_turbInflow(do_turbInflow),
      m_map_eval(a_map_eval),
      m_do_inlet_from_plane(do_inletFromPlane),
      m_inlet_from_plane_dir(inlet_plane_dir),
      m_recycling_full(recycling_full)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& state,
    const int /*dcomp*/,
    const int numcomp,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {

    // Get geometry data
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();

    // Get a 3D version of iv to access first cell in domain
    auto iv_3d = amrex::IntVectND<3>(0);
    AMREX_D_TERM(iv_3d[0] = iv[0];, iv_3d[1] = iv[1];, iv_3d[2] = iv[2];);

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

      // Low
      for (int n = 0; n < amrex::min<int>(numcomp, NVAR); ++n) {

        // bcnormal handles all the state components at once
        amrex::Real s_ext[NVAR] = {0.0};
        // Get current state component BC
        const int* bc = bcr[n].data();

        if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {

          amrex::Real x[AMREX_SPACEDIM];
          pelelm_bc_face_x(iv, idir, 1, geom, m_map_eval, x);

          // Get the state in the first cell in the domain
          auto iv_in = iv_3d;
          iv_in[idir] = iv_in[idir] + 1;
          const auto s_in = state.cellData(iv_in[0], iv_in[1], iv_in[2]);

          // If using TurbInflow or inlet_from_plane, pass in the sampled data.
          // User can overwrite or increment, as needed. In full recycling
          // mode the pre-loaded value is the complete inflow velocity and is
          // kept regardless of what bcnormal does to it.
          bool keep_recycled = false;
          if (n >= VELX && n < DENSITY) {
            if (
              m_do_turbInflow != 0 || ((m_do_inlet_from_plane != 0) &&
                                       (idir == m_inlet_from_plane_dir))) {
              if (!isnan(
                    state(iv, VELX + n))) { // During fillcoarsepatch, get here
                                            // with nan. TODO: find a better fix
                s_ext[VELX + n] = state(iv, VELX + n);
                keep_recycled = (m_recycling_full != 0) &&
                                (m_do_inlet_from_plane != 0) &&
                                (idir == m_inlet_from_plane_dir);
              } else {
                s_ext[VELX + n] = 0;
              }
            }
          }
          // bcnormal() is defined in pelelmex_prob.H in problem directory in
          // /Exec
          T::bcnormal(
            x, s_in, s_ext, idir, 1, time, geom, *lprobparm, lpmfdata);
          if (!keep_recycled) {
            state(iv, n) = s_ext[n];
          }
        }
      }

      // High
      for (int n = 0; n < amrex::min<int>(numcomp, NVAR); ++n) {

        // bcnormal handles all the state components at once
        amrex::Real s_ext[NVAR] = {0.0};

        // Get current state component BC
        const int* bc = bcr[n].data();

        if (
          (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
          (iv[idir] > domhi[idir])) {

          amrex::Real x[AMREX_SPACEDIM];
          pelelm_bc_face_x(iv, idir, -1, geom, m_map_eval, x);

          // Get the state in the first cell in the domain
          auto iv_in = iv_3d;
          iv_in[idir] = iv_in[idir] - 1;
          const auto s_in = state.cellData(iv_in[0], iv_in[1], iv_in[2]);

          // If using TurbInflow or inlet_from_plane, pass in the sampled data.
          // User can overwrite or increment, as needed. In full recycling
          // mode the pre-loaded value is the complete inflow velocity and is
          // kept regardless of what bcnormal does to it.
          bool keep_recycled = false;
          if (n >= VELX && n < DENSITY) {
            if (
              m_do_turbInflow != 0 || ((m_do_inlet_from_plane != 0) &&
                                       (idir == m_inlet_from_plane_dir))) {
              if (!isnan(
                    state(iv, VELX + n))) { // During fillcoarsepatch, get here
                                            // with nan. TODO: find a better fix
                s_ext[VELX + n] = state(iv, VELX + n);
                keep_recycled = (m_recycling_full != 0) &&
                                (m_do_inlet_from_plane != 0) &&
                                (idir == m_inlet_from_plane_dir);
              } else {
                s_ext[VELX + n] = 0;
              }
            }
          }
          // bcnormal() is defined in pelelmex_prob.H in problem directory in
          // /Exec
          T::bcnormal(
            x, s_in, s_ext, idir, -1, time, geom, *lprobparm, lpmfdata);
          if (!keep_recycled) {
            state(iv, n) = s_ext[n];
          }
        }
      }
    }
  }
};

template <typename T>
struct PeleLMCCFillExtDirAux
{

  ProbParm const* lprobparm;
  pele::physics::PMF::PmfData::DataContainer const* lpmfdata;
  const int m_nAux;
  MeshMapEvaluator m_map_eval{};

  AMREX_GPU_HOST
  constexpr PeleLMCCFillExtDirAux(
    ProbParm const* a_prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* a_pmf_data,
    int a_nAux,
    MeshMapEvaluator a_map_eval = MeshMapEvaluator{})
    : lprobparm(a_prob_parm),
      lpmfdata(a_pmf_data),
      m_nAux(a_nAux),
      m_map_eval(a_map_eval)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& aux,
    const int /*dcomp*/,
    const int /*numcomp*/,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {
    // Get geometry data
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();

#if AMREX_SPACEDIM == 2
    auto aux_ext = aux.cellData(iv[0], iv[1], 0);
#else
    auto aux_ext = aux.cellData(iv[0], iv[1], iv[2]);
#endif

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

      // Low
      for (int n = 0; n < m_nAux; ++n) {

        // Get current state component BC
        const int* bc = bcr[n].data();

        if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
          amrex::Real x[AMREX_SPACEDIM];
          pelelm_bc_face_x(iv, idir, 1, geom, m_map_eval, x);
          // bcnormal_aux() is defined in pelelmex_prob.H in problem directory
          // in /Exec
          T::bcnormal_aux(
            x, aux_ext, idir, 1, time, geom, *lprobparm, lpmfdata);
        }
      }
      // High
      for (int n = 0; n < m_nAux; ++n) {

        // Get current state component BC
        const int* bc = bcr[n].data();

        if (
          (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
          (iv[idir] > domhi[idir])) {
          amrex::Real x[AMREX_SPACEDIM];
          pelelm_bc_face_x(iv, idir, -1, geom, m_map_eval, x);
          T::bcnormal_aux(
            x, aux_ext, idir, -1, time, geom, *lprobparm, lpmfdata);
        }
      }
    }
  }
};

template <typename T>
struct PeleLMCCFillExtDirSpec
{

  ProbParm const* lprobparm;
  pele::physics::PMF::PmfData::DataContainer const* lpmfdata;
  const int m_nAux;
  MeshMapEvaluator m_map_eval{};

  AMREX_GPU_HOST
  constexpr PeleLMCCFillExtDirSpec(
    ProbParm const* a_prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* a_pmf_data,
    int a_nAux,
    MeshMapEvaluator a_map_eval = MeshMapEvaluator{})
    : lprobparm(a_prob_parm),
      lpmfdata(a_pmf_data),
      m_nAux(a_nAux),
      m_map_eval(a_map_eval)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& rhoY,
    const int dcomp,
    const int /*numcomp*/,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {

    // Get geometry data
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();

    // Get a 3D version of iv to access first cell in domain
    auto iv_3d = amrex::IntVectND<3>(0);
    AMREX_D_TERM(iv_3d[0] = iv[0];, iv_3d[1] = iv[1];, iv_3d[2] = iv[2];);

    for (int n = 0; n < NUM_SPECIES; ++n) {

      // Get the species component BC
      const int* bc = bcr[n].data();

      // bcnormal handles all the state components at once
      amrex::Real s_ext[NVAR] = {0.0};

      for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
        // Low
        if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {

          amrex::Real x[AMREX_SPACEDIM];
          pelelm_bc_face_x(iv, idir, 1, geom, m_map_eval, x);

          // Get the state on the first cell in the domain
          auto iv_in = iv_3d;
          iv_in[idir] = iv_in[idir] + 1;
          const auto s_in = rhoY.cellData(iv_in[0], iv_in[1], iv_in[2]);

          // bcnormal() is defined in pelelmex_prob.H in problem directory in
          // /Exec
          T::bcnormal(
            x, s_in, s_ext, idir, 1, time, geom, *lprobparm, lpmfdata);

          rhoY(iv, dcomp + n) = s_ext[FIRSTSPEC + n];

          // High
        } else if (
          (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
          (iv[idir] > domhi[idir])) {

          amrex::Real x[AMREX_SPACEDIM];
          pelelm_bc_face_x(iv, idir, -1, geom, m_map_eval, x);

          // Get the state on the first cell in the domain
          auto iv_in = iv_3d;
          iv_in[idir] = iv_in[idir] - 1;
          const auto s_in = rhoY.cellData(iv_in[0], iv_in[1], iv_in[2]);

          // bcnormal() is defined in pelelmex_prob.H in problem directory in
          // /Exec
          T::bcnormal(
            x, s_in, s_ext, idir, -1, time, geom, *lprobparm, lpmfdata);

          rhoY(iv, dcomp + n) = s_ext[FIRSTSPEC + n];
        }
      }
    }
  }
};

template <typename T>
struct PeleLMCCFillExtDirDens
{

  ProbParm const* lprobparm;
  pele::physics::PMF::PmfData::DataContainer const* lpmfdata;
  const int m_nAux;
  MeshMapEvaluator m_map_eval{};

  AMREX_GPU_HOST
  constexpr PeleLMCCFillExtDirDens(
    ProbParm const* a_prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* a_pmf_data,
    int a_nAux,
    MeshMapEvaluator a_map_eval = MeshMapEvaluator{})
    : lprobparm(a_prob_parm),
      lpmfdata(a_pmf_data),
      m_nAux(a_nAux),
      m_map_eval(a_map_eval)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& state,
    const int dcomp,
    const int /*numcomp*/,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {

    // Get geometry data
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();

    // Get a 3D version of iv to access first cell in domain
    auto iv_3d = amrex::IntVectND<3>(0);
    AMREX_D_TERM(iv_3d[0] = iv[0];, iv_3d[1] = iv[1];, iv_3d[2] = iv[2];);

    // Get the first species component BC
    const int* bc = bcr->data();

    // bcnormal handles all the state components at once
    amrex::Real s_ext[NVAR] = {0.0};

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
      // Low
      if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {

        amrex::Real x[AMREX_SPACEDIM];
        pelelm_bc_face_x(iv, idir, 1, geom, m_map_eval, x);

        // Get the state on the first cell in the domain
        auto iv_in = iv_3d;
        iv_in[idir] = iv_in[idir] + 1;
        const auto s_in = state.cellData(iv_in[0], iv_in[1], iv_in[2]);

        // bcnormal() is defined in pelelmex_prob.H in problem directory in
        // /Exec
        T::bcnormal(x, s_in, s_ext, idir, 1, time, geom, *lprobparm, lpmfdata);

        state(iv, dcomp) = s_ext[DENSITY];

        // High
      } else if (
        (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
        (iv[idir] > domhi[idir])) {

        amrex::Real x[AMREX_SPACEDIM];
        pelelm_bc_face_x(iv, idir, -1, geom, m_map_eval, x);

        // Get the state on the first cell in the domain
        auto iv_in = iv_3d;
        iv_in[idir] = iv_in[idir] - 1;
        const auto s_in = state.cellData(iv_in[0], iv_in[1], iv_in[2]);

        // bcnormal() is defined in pelelmex_prob.H in problem directory in
        // /Exec
        T::bcnormal(x, s_in, s_ext, idir, -1, time, geom, *lprobparm, lpmfdata);

        state(iv, dcomp) = s_ext[DENSITY];
      }
    }
  }
};

template <typename T>
struct PeleLMCCFillExtDirRhoH
{

  ProbParm const* lprobparm;
  pele::physics::PMF::PmfData::DataContainer const* lpmfdata;
  const int m_nAux;
  MeshMapEvaluator m_map_eval{};

  AMREX_GPU_HOST
  constexpr PeleLMCCFillExtDirRhoH(
    ProbParm const* a_prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* a_pmf_data,
    int a_nAux,
    MeshMapEvaluator a_map_eval = MeshMapEvaluator{})
    : lprobparm(a_prob_parm),
      lpmfdata(a_pmf_data),
      m_nAux(a_nAux),
      m_map_eval(a_map_eval)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& rhoH,
    const int /*dcomp*/,
    const int /*numcomp*/,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {

    // Get geometry data
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();

    // Get a 3D version of iv to access first cell in domain
    auto iv_3d = amrex::IntVectND<3>(0);
    AMREX_D_TERM(iv_3d[0] = iv[0];, iv_3d[1] = iv[1];, iv_3d[2] = iv[2];);

    // Get the first species component BC
    const int* bc = bcr->data();

    // bcnormal handles all the state components at once
    amrex::Real s_ext[NVAR] = {0.0};

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
      // Low
      if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {

        amrex::Real x[AMREX_SPACEDIM];
        pelelm_bc_face_x(iv, idir, 1, geom, m_map_eval, x);

        // Get the state on the first cell in the domain
        auto iv_in = iv_3d;
        iv_in[idir] = iv_in[idir] + 1;
        const auto s_in = rhoH.cellData(iv_in[0], iv_in[1], iv_in[2]);

        // bcnormal() is defined in pelelmex_prob.H in problem directory in
        // /Exec
        T::bcnormal(x, s_in, s_ext, idir, 1, time, geom, *lprobparm, lpmfdata);

        rhoH(iv) = s_ext[RHOH];

        // High
      } else if (
        (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
        (iv[idir] > domhi[idir])) {

        amrex::Real x[AMREX_SPACEDIM];
        pelelm_bc_face_x(iv, idir, -1, geom, m_map_eval, x);

        // Get the state on the first cell in the domain
        auto iv_in = iv_3d;
        iv_in[idir] = iv_in[idir] - 1;
        const auto s_in = rhoH.cellData(iv_in[0], iv_in[1], iv_in[2]);

        // bcnormal() is defined in pelelmex_prob.H in problem directory in
        // /Exec
        T::bcnormal(x, s_in, s_ext, idir, -1, time, geom, *lprobparm, lpmfdata);

        rhoH(iv) = s_ext[RHOH];
      }
    }
  }
};

template <typename T>
struct PeleLMCCFillExtDirTemp
{

  ProbParm const* lprobparm;
  pele::physics::PMF::PmfData::DataContainer const* lpmfdata;
  const int m_nAux;
  MeshMapEvaluator m_map_eval{};

  AMREX_GPU_HOST
  constexpr PeleLMCCFillExtDirTemp(
    ProbParm const* a_prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* a_pmf_data,
    int a_nAux,
    MeshMapEvaluator a_map_eval = MeshMapEvaluator{})
    : lprobparm(a_prob_parm),
      lpmfdata(a_pmf_data),
      m_nAux(a_nAux),
      m_map_eval(a_map_eval)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& state,
    const int dcomp,
    const int /*numcomp*/,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {

    // Get geometry data
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();

    // Get a 3D version of iv to access first cell in domain
    auto iv_3d = amrex::IntVectND<3>(0);
    AMREX_D_TERM(iv_3d[0] = iv[0];, iv_3d[1] = iv[1];, iv_3d[2] = iv[2];);

    // Get the first species component BC
    const int* bc = bcr->data();

    // bcnormal handles all the state components at once
    amrex::Real s_ext[NVAR] = {0.0};
    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
      // Low
      if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {

        amrex::Real x[AMREX_SPACEDIM];
        pelelm_bc_face_x(iv, idir, 1, geom, m_map_eval, x);

        // Get the state on the first cell in the domain
        auto iv_in = iv_3d;
        iv_in[idir] = iv_in[idir] + 1;
        const auto s_in = state.cellData(iv_in[0], iv_in[1], iv_in[2]);

        // bcnormal() is defined in pelelmex_prob.H in problem directory in
        // /Exec
        T::bcnormal(x, s_in, s_ext, idir, 1, time, geom, *lprobparm, lpmfdata);

        state(iv, dcomp) = s_ext[TEMP];

        // High
      } else if (
        (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
        (iv[idir] > domhi[idir])) {

        amrex::Real x[AMREX_SPACEDIM];
        pelelm_bc_face_x(iv, idir, -1, geom, m_map_eval, x);

        // Get the state on the first cell in the domain
        auto iv_in = iv_3d;
        iv_in[idir] = iv_in[idir] - 1;
        const auto s_in = state.cellData(iv_in[0], iv_in[1], iv_in[2]);

        // bcnormal() is defined in pelelmex_prob.H in problem directory in
        // /Exec
        T::bcnormal(x, s_in, s_ext, idir, -1, time, geom, *lprobparm, lpmfdata);

        state(iv, dcomp) = s_ext[TEMP];
      }
    }
  }
};

#ifdef PELE_USE_PLASMA
template <typename T>
struct PeleLMCCFillExtDirnE
{

  ProbParm const* lprobparm;
  pele::physics::PMF::PmfData::DataContainer const* lpmfdata;
  const int m_nAux;
  MeshMapEvaluator m_map_eval{};

  AMREX_GPU_HOST
  constexpr PeleLMCCFillExtDirnE(
    ProbParm const* a_prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* a_pmf_data,
    int a_nAux,
    MeshMapEvaluator a_map_eval = MeshMapEvaluator{})
    : lprobparm(a_prob_parm),
      lpmfdata(a_pmf_data),
      m_nAux(a_nAux),
      m_map_eval(a_map_eval)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& nE,
    const int /*dcomp*/,
    const int /*numcomp*/,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {

    // Get geometry data
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();

    // Get a 3D version of iv to access first cell in domain
    auto iv_3d = amrex::IntVectND<3>(0);
    AMREX_D_TERM(iv_3d[0] = iv[0];, iv_3d[1] = iv[1];, iv_3d[2] = iv[2];);

    // Get the first species component BC
    const int* bc = bcr->data();

    // bcnormal handles all the state components at once
    amrex::Real s_ext[NVAR] = {0.0};

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
      // Low
      if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {

        amrex::Real x[AMREX_SPACEDIM];
        pelelm_bc_face_x(iv, idir, 1, geom, m_map_eval, x);

        // Get the state on the first cell in the domain
        auto iv_in = iv_3d;
        iv_in[idir] = iv_in[idir] + 1;
        const auto s_in = nE.cellData(iv_in[0], iv_in[1], iv_in[2]);

        // bcnormal() is defined in pelelmex_prob.H in problem directory in
        // /Exec
        T::bcnormal(x, s_in, s_ext, idir, 1, time, geom, *lprobparm, lpmfdata);

        nE(iv) = s_ext[NE];

        // High
      } else if (
        (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
        (iv[idir] > domhi[idir])) {

        amrex::Real x[AMREX_SPACEDIM];
        pelelm_bc_face_x(iv, idir, -1, geom, m_map_eval, x);

        // Get the state on the first cell in the domain
        auto iv_in = iv_3d;
        iv_in[idir] = iv_in[idir] - 1;
        const auto s_in = nE.cellData(iv_in[0], iv_in[1], iv_in[2]);

        // bcnormal() is defined in pelelmex_prob.H in problem directory in
        // /Exec
        T::bcnormal(x, s_in, s_ext, idir, -1, time, geom, *lprobparm, lpmfdata);

        nE(iv) = s_ext[NE];
      }
    }
  }
};
#endif

#ifdef PELE_USE_PLASMA
template <typename T>
struct PeleLMCCFillExtDirPhiV
{

  ProbParm const* lprobparm;
  pele::physics::PMF::PmfData::DataContainer const* lpmfdata;
  const int m_nAux;
  MeshMapEvaluator m_map_eval{};

  AMREX_GPU_HOST
  constexpr PeleLMCCFillExtDirPhiV(
    ProbParm const* a_prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* a_pmf_data,
    int a_nAux,
    MeshMapEvaluator a_map_eval = MeshMapEvaluator{})
    : lprobparm(a_prob_parm),
      lpmfdata(a_pmf_data),
      m_nAux(a_nAux),
      m_map_eval(a_map_eval)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& phiV,
    const int dcomp,
    const int /*numcomp*/,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {

    // Get geometry data
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();

    // Get a 3D version of iv to access first cell in domain
    auto iv_3d = amrex::IntVectND<3>(0);
    AMREX_D_TERM(iv_3d[0] = iv[0];, iv_3d[1] = iv[1];, iv_3d[2] = iv[2];);

    // Get the first species component BC
    const int* bc = bcr->data();

    // bcnormal handles all the state components at once
    amrex::Real s_ext[NVAR] = {0.0};

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
      // Low
      if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {

        amrex::Real x[AMREX_SPACEDIM];
        pelelm_bc_face_x(iv, idir, 1, geom, m_map_eval, x);

        // Get the state on the first cell in the domain
        auto iv_in = iv_3d;
        iv_in[idir] = iv_in[idir] + 1;
        const auto s_in = phiV.cellData(iv_in[0], iv_in[1], iv_in[2]);

        // bcnormal() is defined in pelelmex_prob.H in problem directory in
        // /Exec
        T::bcnormal(x, s_in, s_ext, idir, 1, time, geom, *lprobparm, lpmfdata);

        phiV(iv, dcomp) = s_ext[PHIV];

        // High
      } else if (
        (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
        (iv[idir] > domhi[idir])) {

        amrex::Real x[AMREX_SPACEDIM];
        pelelm_bc_face_x(iv, idir, -1, geom, m_map_eval, x);

        // Get the state on the first cell in the domain
        auto iv_in = iv_3d;
        iv_in[idir] = iv_in[idir] - 1;
        const auto s_in = phiV.cellData(iv_in[0], iv_in[1], iv_in[2]);

        // bcnormal() is defined in pelelmex_prob.H in problem directory in
        // /Exec
        T::bcnormal(x, s_in, s_ext, idir, -1, time, geom, *lprobparm, lpmfdata);

        phiV(iv, dcomp) = s_ext[PHIV];
      }
    }
  }
};
#endif

struct PeleLMCCFillExtDirDummy
{
  const int m_nAux;

  AMREX_GPU_HOST
  constexpr PeleLMCCFillExtDirDummy(int a_nAux) : m_nAux(a_nAux) {}

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& /*iv*/,
    amrex::Array4<amrex::Real> const& /*dummy*/,
    const int /*dcomp*/,
    const int /*numcomp*/,
    amrex::GeometryData const& /*geom*/,
    const amrex::Real /*time*/,
    const amrex::BCRec* /*bcr*/,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {
  }
};

//
// A dummy function because FillPatch requires something to exist for filling
// dirichlet boundary conditions, even if we know we cannot have an ext_dir BC.
// u_mac BCs are only either periodic (amrex::BCType::int_dir) or first order
// extrapolation (amrex::BCType::foextrap).
//
struct umacFill
{
  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& /*iv*/,
    amrex::Array4<amrex::Real> const& /*dummy*/,
    const int /*dcomp*/,
    const int numcomp,
    amrex::GeometryData const& /*geom*/,
    const amrex::Real /*time*/,
    const amrex::BCRec* bcr,
    const int bcomp,
    const int /*orig_comp*/) const
  {
    // Abort if this function is expected to fill an ext_dir BC.
    for (int n = bcomp; n < bcomp + numcomp; ++n) {
      const amrex::BCRec& bc = bcr[n];
      if (AMREX_D_TERM(
            bc.lo(0) == amrex::BCType::ext_dir ||
              bc.hi(0) == amrex::BCType::ext_dir,
            || bc.lo(1) == amrex::BCType::ext_dir ||
              bc.hi(1) == amrex::BCType::ext_dir,
            || bc.lo(2) == amrex::BCType::ext_dir ||
              bc.hi(2) == amrex::BCType::ext_dir)) {
        amrex::Abort("PeleLM::umacFill: umac should not have BCType::ext_dir");
      }
    }
  }
};
#endif
