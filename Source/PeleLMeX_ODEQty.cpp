#include <PeleLMeX.H>
#include <PeleLMeX_K.H>

#if NUM_ODE > 0
void
PeleLM::predictODEQty()
{
  // B^{np1} = B^{n} + dt * S_ext: forward Euler with S_ext = S^n; with
  // m_ext_sources_SDC, re-applied in SDC iterations > 1 with
  // S_ext = 0.5 (S^n + S^{n+1,k}) (trapezoidal)
  const auto dt = m_dt;
  for (int lev = 0; lev <= finest_level; ++lev) {
    auto const& old_arrs =
      getLevelDataPtr(lev, AmrOldTime)->state.const_arrays();
    auto const& state_arrs = getLevelDataPtr(lev, AmrNewTime)->state.arrays();
    auto const& ext_src_arrs = m_extSource[lev]->const_arrays();
    amrex::ParallelFor(
      *m_extSource[lev], amrex::IntVect(0), NUM_ODE,
      [old_arrs, state_arrs, ext_src_arrs,
       dt] AMREX_GPU_DEVICE(int box_no, int i, int j, int k, int n) noexcept {
        const amrex::Real S_ext = ext_src_arrs[box_no](i, j, k, FIRSTODE + n);
        state_arrs[box_no](i, j, k, FIRSTODE + n) =
          old_arrs[box_no](i, j, k, FIRSTODE + n) + dt * S_ext;
      });
  }
  amrex::Gpu::streamSynchronize();
}
#endif
