// Problem-specific functions (PSF)
#ifndef PROBLEM_FUNC_H
#define PROBLEM_FUNC_H

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <PMFData.H>

#include <PeleLMeX_MeshMapEvaluator.H>

#include <type_traits>

// Compile-time detection: does the user-supplied problem-specific
// struct provide a mapping-aware `initdata_mapped` static method
// (in addition to, or instead of, the legacy `initdata`)?  When true,
// PeleLM::initLevelData dispatches to `initdata_mapped` and passes
// the active MeshMapEvaluator; when false, it falls back to the
// legacy `initdata` signature.  See PeleLMeX_Init.cpp for the
// `if constexpr` switch.  Existing problem setups need no edits to
// continue working; mapping-aware problems opt in by defining
// `initdata_mapped` instead of `initdata`.
template <typename T, typename = void>
struct has_initdata_mapped : std::false_type
{
};

template <typename T>
struct has_initdata_mapped<T, std::void_t<decltype(&T::initdata_mapped)>>
  : std::true_type
{
};

template <typename T>
inline constexpr bool has_initdata_mapped_v = has_initdata_mapped<T>::value;

// Template helper that dispatches between legacy `initdata` and the
// mapping-aware `initdata_mapped` at compile time.  Wrapping the
// dispatch in a function template makes the calls dependent on the
// template parameter `PSF`, so the `if constexpr` branch that doesn't
// apply for the particular `PSF` is never instantiated -- which is
// what allows existing problem setups (that define only the legacy
// `initdata`) to keep compiling without providing an
// `initdata_mapped` stub.
template <typename PSF, typename ProbParm>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
call_initdata(
  int i,
  int j,
  int k,
  int is_incomp,
  amrex::Array4<amrex::Real> const& state,
  amrex::Array4<amrex::Real> const& aux,
  amrex::GeometryData const& geomdata,
  MeshMapEvaluator const& mmap,
  ProbParm const& prob_parm,
  pele::physics::PMF::PmfData::DataContainer const* pmf_data) noexcept
{
  if constexpr (has_initdata_mapped_v<PSF>) {
    PSF::initdata_mapped(
      i, j, k, is_incomp, state, aux, geomdata, mmap, prob_parm, pmf_data);
  } else {
    (void)mmap; // unused in the legacy path; suppress -Wunused
    PSF::initdata(
      i, j, k, is_incomp, state, aux, geomdata, prob_parm, pmf_data);
  }
}

struct ProbParmDefault
{
  /**
   * PeleLMeX always needs a mean thermodynamic pressure
   */
  amrex::Real P_mean = 101325.0;
};

struct DefaultProblemSpecificFunctions
{
  /**
   * \brief Initialize data at the beginning of the simulation
   *
   * \param i                 i index.
   * \param j                 j index.
   * \param k                 k index.
   * \param is_incompressible incompressible flag.
   * \param state             state Array4.
   * \param aux               auxiliary Array4.
   * \param geomdata          geometry data.
   * \param prob_parm         problem specific parameters.
   * \param pmf_data          PMF (Premixed Flame) data
   */
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void initdata(
    int /*i*/,
    int /*j*/,
    int /*k*/,
    int /*is_incompressible*/,
    amrex::Array4<amrex::Real> const& /*state*/,
    amrex::Array4<amrex::Real> const& /*aux*/,
    amrex::GeometryData const& /*geomdata*/,
    ProbParmDefault const& /*prob_parm*/,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    amrex::Abort(
      "Default empty initdata called, please override in your problem specific "
      "struct !");
    /** Notes:
     *   1. This function is called at the beginning of the simulation
     *   2. The default implementation does nothing
     *   3. Specific problem implementations should override this function
     */
  }

  /**
   * \brief Define the domain face boundary conditions. Note: If turbInflow or
   * recycling is used, s_in arrives here with the velocity fields preloaded
   * with those fluctuations. If neither option is activated, the fields are
   * preloaded with zero. Thus a += should be used for those fields to add
   * fluctuating fields with the user-desired mean values. Important: when
   * this happens, user code MUST use s_ext[VELX+n] += ... rather than =
   * for the velocity components, otherwise the preloaded fluctuation is
   * overwritten.
   *
   * \param x                 the face center coordinates.
   * \param s_in              state in first cell (as ptr into Array4)
   * \param s_ext             outgoing state (as a standard C++ array)
   * \param idir              direction of the face.
   * \param sgn               sign of the face normal.
   * \param time              current time
   * \param geomdata          geometry data.
   * \param prob_parm         problem specific parameters.
   * \param pmf_data          PMF (Premixed Flame) data
   */
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void bcnormal(
    const amrex::Real* /*x[AMREX_SPACEDIM]*/,
    const amrex::CellData<amrex::Real>& /*s_in*/,
    amrex::Real* /*s_ext[NVAR]*/,
    const int /*idir*/,
    const int /*sgn*/,
    const amrex::Real /*time*/,
    amrex::GeometryData const& /*geomdata*/,
    ProbParmDefault const& /*prob_parm*/,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    amrex::Abort(
      "Default empty bcnormal called, please override in your problem specific "
      "struct !");
    /** Notes:
     *   1. This function handles the boundary conditions on the domain faces
     *   2. The default implementation does nothing
     *   3. Specific problem implementations should override this function
     */
  }

  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bcnormal_aux(
    const amrex::Real x[AMREX_SPACEDIM],
    amrex::CellData<amrex::Real>& aux_ext,
    const int idir,
    const int sgn,
    const amrex::Real time,
    amrex::GeometryData const& geomdata,
    ProbParmDefault const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* pmf_data)
  {
    amrex::ignore_unused(
      x, aux_ext, idir, sgn, time, geomdata, prob_parm, pmf_data);
    amrex::Abort(
      "Default empty bcnormal called, please override in your "
      "problem specific struct !");
    // For filling auxiliary variable boundary conditions
    // Retrieve nAux with aux_ext.ncomp, then aux_ext[n] = ...
  }

  /**
   * \brief Alter the scalars face-centered diffusion coefficients
   *
   * \param i                 i index.
   * \param j                 j index.
   * \param k                 k index.
   * \param beta              face-centered Beta Array4.
   * \param geomdata          geometry data.
   * \param domainBox         the computational domain box.
   * \param dir               direction of the face.
   * \param beta_comp         first components in Beta MultiFab.
   * \param nComp             number of components to change.
   * \param prob_parm         problem specific parameters.
   */
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void zero_visc(
    int /*i*/,
    int /*j*/,
    int /*k*/,
    amrex::Array4<amrex::Real> const& /*beta*/,
    amrex::GeometryData const& /*geomdata*/,
    amrex::Box const& /*domainBox*/,
    const int /*dir*/,
    const int /*beta_comp*/,
    const int /*nComp*/,
    ProbParmDefault const& /*prob_parm*/)
  {
    /** Notes:
     *   1. This function can modify the diffusion coefficient
     *      on the domain faces in order to represent walls on a domain
     *      face otherwise marked as inflow
     *   2. To know for which state component the function is called, the
     *      logic is as follows:
     *       - if beta_comp == 0 and nComp == NUM_SPECIES => call for species
     *       - if beta_comp == 0 and nComp == 1 => call for temperature
     *   3. The default implementation does nothing
     *   4. To only modify at the domain faces, one needs to compare the i, j or
     * k index matching the idir direction against the domainBox
     */
  }

  /**
   * \brief Generate a custom Embedded Boundary geometry
   *
   * \param geom                        level at which to generate the geometry
   * \param required_coarsening_level   the minimum number of coarse levels
   * needed \param max_coarsening_level        the maximum number of coarse
   * levels
   */
  static void EBUserDefined(
    const amrex::Geometry& /*geom*/,
    const int /*required_coarsening_level*/,
    const int /*max_coarsening_level*/)
  {
    amrex::Abort(
      "Default EBUserDefined function called! Redefine it in your "
      "local ProblemSpecificFunctions struct !");
    /** Notes:
     *   1. ParmParse any parameter for your geometry
     *   2. Construct your geometry assembling EB2::* methods
     *   3. Build the geometry shop using EB2::makeShop
     *   4. Build the geometry using EB2::Build
     */
  }

  /**
   * \brief Set the ODE Qty names
   *
   * \param a_ode_names       the vector of ODE names to fill.
   */
  static void set_ode_names(amrex::Vector<std::string>& a_ode_names)
  {
#if NUM_ODE > 0
    a_ode_names.resize(NUM_ODE);
    for (int n = 0; n < NUM_ODE; ++n) {
      a_ode_names[n] = "ODE_" + std::to_string(n);
    }
#else
    amrex::ignore_unused(a_ode_names);
#endif
  }

  /**
   * \brief Modify the external source term in the state conservation eq.
   *
   * \param time              time at which to evaluate the sources.
   * \param dt                current time-step.
   * \param state_old         the state at which to evaluate the sources.
   * \param state_new         the new state multifab.
   * \param ext_src           a pointer to the external src term multifab.
   * \param geomdata          geometry data.
   * \param prob_parm         problem specific parameters.
   */
  static void modify_ext_sources(
    amrex::Real /*time*/,
    amrex::Real /*dt*/,
    const amrex::MultiFab& /*state_old*/,
    const amrex::MultiFab& /*state_new*/,
    std::unique_ptr<amrex::MultiFab>& /*ext_src*/,
    const amrex::GeometryData& /*geomdata*/,
    const ProbParmDefault* /*prob_parm_d*/)
  {
    /** Notes:
     *   1. ext_src contains sources from velocity forcing coming in.
     *      This function should add to rather than overwrite ext_src.
     *   2. Requires peleLM.user_defined_ext_sources = true in input file
     *   3. Called once per step at t^n with the t^n state. With
     *      peleLM.ext_sources_SDC = true it is also called at t^{n+1}
     *      with state_old = state_new = the t^{n+1,k} iterate, and the
     *      scalar update uses 0.5 (S^n + S^{n+1,k}).
     *
     * // Example: Exponential decay ode quantity
     * auto ext_src_arr = ext_src->arrays();
     * auto const& state_old_arr = state_old.const_arrays();
     * ParallelFor(
     *   *ext_src,
     *   [ext_src_arr] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)
     * noexcept { for (int n = 0; n < NUM_ODE; ++n) { amrex::Real B_n =
     * state_old_arr[box_no](i, j, k, FIRSTODE + n); amrex::Real src = -1.0 *
     * pow(10.0, n + 1) * B_n; ext_src_arr[box_no](i, j, k, FIRSTODE + n) +=
     * src;
     *     }
     *   });
     * amrex::Gpu::streamSynchronize();
     */
  }

  static void patchFlowVariables(
    const amrex::Geometry& /*geom*/,
    ProbParmDefault const& /*prob_parm*/,
    amrex::MultiFab& /*a_mf*/)
  {
    amrex::Abort(
      "Using patchFlowVariables requires providing a definition in local "
      "ProblemSpecificFunctions in pelelmex_prob.H");
  }

  /**
   * \brief Perform problem-specific operations after advance
   *
   * \param time              current time after advance.
   * \param dt                time-step size used in advance.
   * \param finest_level      finest AMR level.
   * \param state             vector of state multifabs for all levels.
   * \param geom              vector of geometry objects for all levels.
   * \param prob_parm         host copy of problem specific parameters.
   * \param prob_parm_d       device copy of problem specific parameters.
   */
  static void postAdvance(
    amrex::Real /*time*/,
    amrex::Real /*dt*/,
    int /*finest_level*/,
    const amrex::Vector<amrex::MultiFab*>& /*state*/,
    const amrex::Vector<const amrex::Geometry*>& /*geom*/,
    const ProbParmDefault& /*prob_parm*/,
    const ProbParmDefault* /*prob_parm_d*/)
  {
    /** Notes:
     *   1. This function is called once after each time advance
     *   2. Receives state data for all AMR levels
     *   3. prob_parm is the host copy, useful for CPU-side logic and checks
     *   4. prob_parm_d is the device copy, useful for GPU kernel launches
     *   5. Any post-advance operations must be performend on each level
     *   6. The default implementation does nothing
     *   7. Specific problem implementations can override this function
     */
  }

}; // namespace DefaultProblemSpecificFunctions

#endif
