#ifndef PELELM_BCFILL_EB
#define PELELM_BCFILL_EB
#include "PeleLMeX_Index.H"
#include "pelelmex_prob.H"

#include <AMReX_PhysBCFunct.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

using namespace amrex::literals;

template <typename, typename T>
struct matchBCNormEB_sign
{
  static_assert(
    std::integral_constant<T, false>::value,
    "Second template parameter needs to be of function type.");
};

template <typename T, typename Ret, typename... Args>
struct matchBCNormEB_sign<T, Ret(Args...)>
{
private:
  template <typename C>
  static constexpr auto check(T*) -> typename std::is_same<
    decltype(std::declval<C>().bcnormal_eb(std::declval<Args>()...)),
    Ret>::type;
  template <typename>
  static constexpr std::false_type check(...);

  using type = decltype(check<T>(nullptr));

public:
  static constexpr bool value = type::value;
};

template <typename T>
struct hasBCNormalEB
{
private:
  template <typename C>
  static constexpr auto check(T*) -> matchBCNormEB_sign<
    T,
    void(
      const amrex::Real*,                        // Face center coordinate
      const amrex::Real*,                        // Face normal (Outward)
      const amrex::CellData<const amrex::Real>&, // Cell-centered state (in)
      amrex::Real*,                              // Face-centered state (out)
      const amrex::Real,
      amrex::GeometryData const&,
      ProbParm const&)>;
  template <typename>
  static constexpr std::false_type check(...);

  using type = decltype(check<T>(nullptr));

public:
  static constexpr bool value = type::value;
};

// A default empty struct when the we don't provide a valid EB functions in
// ProblemSpecificFunctions
template <typename T, bool B = false>
struct PeleLMFillBCStateEB
{
  ProbParm const* lprobparm;
  T const ebh;

  AMREX_GPU_HOST
  constexpr PeleLMFillBCStateEB(ProbParm const* a_prob_parm, T const& a_ebh)
    : lprobparm(a_prob_parm), ebh(a_ebh)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    int /*i*/,
    int /*j*/,
    int /*k*/,
    amrex::Array4<const amrex::Real> const& /*state*/,
    amrex::Array4<amrex::Real> const& /*ebState*/,
    const int /*firstcomp*/,
    const int /*numcomp*/,
    amrex::Array4<const amrex::Real> const& /*ebbc*/,
    amrex::Array4<amrex::Real const> const& /*ebnorm*/,
    amrex::GeometryData const& /*geom*/,
    const amrex::Real /*time*/) const
  {
  }
};

template <typename T>
struct PeleLMFillBCStateEB<T, true>
{
  ProbParm const* lprobparm;
  T const ebh;

  AMREX_GPU_HOST
  constexpr PeleLMFillBCStateEB(ProbParm const* a_prob_parm, T const& a_ebh)
    : lprobparm(a_prob_parm), ebh(a_ebh)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    int i,
    int j,
    int k,
    amrex::Array4<const amrex::Real> const& state,
    amrex::Array4<amrex::Real> const& ebState,
    const int firstcomp,
    const int numcomp,
    amrex::Array4<const amrex::Real> const& ebbc,
    amrex::Array4<amrex::Real const> const& ebnorm,
    amrex::GeometryData const& geom,
    const amrex::Real time) const
  {
    // Get the EB boundary centroid coordinates
    const amrex::Real* dx = geom.CellSize();
    const amrex::Real* prob_lo = geom.ProbLo();
    const amrex::Real xcell[AMREX_SPACEDIM] = {AMREX_D_DECL(
      prob_lo[0] + (i + 0.5_rt) * dx[0], prob_lo[1] + (j + 0.5_rt) * dx[1],
      prob_lo[2] + (k + 0.5_rt) * dx[2])};
    const amrex::Real xface[AMREX_SPACEDIM] = {AMREX_D_DECL(
      xcell[0] + ebbc(i, j, k, 0) * dx[0], xcell[1] + ebbc(i, j, k, 1) * dx[1],
      xcell[2] + ebbc(i, j, k, 2) * dx[2])};
    const amrex::Real bcnorm[AMREX_SPACEDIM] = {
      AMREX_D_DECL(ebnorm(i, j, k, 0), ebnorm(i, j, k, 1), ebnorm(i, j, k, 2))};

    // TODO : would be practical to have the current state at the EBface ...
    // Let's start with the state at the cell-centroid
    const amrex::CellData<const amrex::Real> stateIn = state.cellData(i, j, k);

    amrex::Real stateExt[NVAR] = {0.0};

    // User-defined fill function
    ebh.bcnormal_eb(xface, bcnorm, stateIn, stateExt, time, geom, *lprobparm);

    // Extract requested entries
    for (int n = 0; n < numcomp; ++n) {
      ebState(i, j, k, n) = stateExt[firstcomp + n];
    }
  }
};

template <typename, typename T>
struct matchBCTypeEB_sign
{
  static_assert(
    std::integral_constant<T, false>::value,
    "Second template parameter needs to be of function type.");
};

template <typename T, typename Ret, typename... Args>
struct matchBCTypeEB_sign<T, Ret(Args...)>
{
private:
  template <typename C>
  static constexpr auto check(T*) -> typename std::is_same<
    decltype(std::declval<C>().bctype_eb(std::declval<Args>()...)),
    Ret>::type;
  template <typename>
  static constexpr std::false_type check(...);

  using type = decltype(check<T>(nullptr));

public:
  static constexpr bool value = type::value;
};

template <typename T>
struct hasBCTypeEB
{
private:
  template <typename C>
  static constexpr auto check(T*) -> matchBCTypeEB_sign<
    T,
    void(
      const amrex::Real*,
      int&,
      amrex::Real&,
      amrex::GeometryData const&,
      ProbParm const&)>;
  template <typename>
  static constexpr std::false_type check(...);

  using type = decltype(check<T>(nullptr));

public:
  static constexpr bool value = type::value;
};

// A default empty struct when the we don't provide a valid EB functions in
// ProblemSpecificFunctions
template <typename T, bool B = false>
struct PeleLMFillBCTypeEB
{
  ProbParm const* lprobparm;
  T const ebh;

  AMREX_GPU_HOST
  constexpr PeleLMFillBCTypeEB(ProbParm const* a_prob_parm, T const& a_ebh)
    : lprobparm(a_prob_parm), ebh(a_ebh)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    int /*i*/,
    int /*j*/,
    int /*k*/,
    int& /*EBFlagType*/,
    amrex::Real& /*EBfaceFrac*/,
    amrex::Array4<const amrex::Real> const& /*ebbc*/,
    amrex::GeometryData const& /*geom*/,
    ProbParm const& /*prob_parm*/) const
  {
  }
};

template <typename T>
struct PeleLMFillBCTypeEB<T, true>
{
  ProbParm const* lprobparm;
  T const ebh;

  AMREX_GPU_HOST
  constexpr PeleLMFillBCTypeEB(ProbParm const* a_prob_parm, T const& a_ebh)
    : lprobparm(a_prob_parm), ebh(a_ebh)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    int i,
    int j,
    int k,
    int& EBFlagType,
    amrex::Real& EBfaceFrac,
    amrex::Array4<const amrex::Real> const& ebbc,
    amrex::GeometryData const& geom,
    ProbParm const& /*prob_parm*/) const
  {
    // Get the EB boundary centroid coordinates
    const amrex::Real* dx = geom.CellSize();
    const amrex::Real* prob_lo = geom.ProbLo();
    const amrex::Real xcell[AMREX_SPACEDIM] = {AMREX_D_DECL(
      prob_lo[0] + (i + 0.5_rt) * dx[0], prob_lo[1] + (j + 0.5_rt) * dx[1],
      prob_lo[2] + (k + 0.5_rt) * dx[2])};
    const amrex::Real xface[AMREX_SPACEDIM] = {AMREX_D_DECL(
      xcell[0] + ebbc(i, j, k, 0) * dx[0], xcell[1] + ebbc(i, j, k, 1) * dx[1],
      xcell[2] + ebbc(i, j, k, 2) * dx[2])};

    // User-defined fill type function, default is wall adiab
    EBFlagType = pelelmex::BCTypeEB::wall_adiab;
    ebh.bctype_eb(xface, EBFlagType, EBfaceFrac, geom, *lprobparm);
  }
};
#endif
