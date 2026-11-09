#include <PeleLMeX.H>

amrex::Real
PeleLM::estEFIonsDt(const TimeStamp a_time)
{
  amrex::Real estdt = 1.0e200;
  constexpr amrex::Real small = 1.0e-8;

  auto bcRecPhiV = fetchBCRecArray(PHIV, 1);

  // Need the mobility of the ions
  calcDiffusivity(a_time);

  // The E-field stencil below reads phiV one cell outside each box
  fillPatchPhiV(a_time);

  for (int lev = 0; lev <= finest_level; ++lev) {

    amrex::Real estdt_lev = 1.0e200;

    // Get cell centered gradient of phiV
    auto ldata_p = getLevelDataPtr(lev, a_time);
    amrex::MultiFab efield_cc(grids[lev], dmap[lev], AMREX_SPACEDIM, 0);
    amrex::MultiFab driftVelMax_cc(grids[lev], dmap[lev], 1, 0);

    const auto dxinv = Geom(lev).InvCellSizeArray();
    const auto domain = Geom(lev).Domain();

    auto const& state_ma = ldata_p->state.const_arrays();
    auto const& efield_ma = efield_cc.arrays();

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      auto bc_lo = bcRecPhiV[0].lo(idim);
      auto bc_hi = bcRecPhiV[0].hi(idim);
      const amrex::Real factor = -0.5 * dxinv[idim];
      amrex::ParallelFor(
        ldata_p->state,
        [bc_lo, bc_hi, efield_ma, state_ma, factor, domain,
         idim] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
          int idx[3] = {i, j, k};
          amrex::Array4<amrex::Real const> phiV(state_ma[box_no], PHIV);
          const bool on_lo =
            ((bc_lo == amrex::BCType::ext_dir) &&
             idx[idim] <= domain.smallEnd(idim));
          const bool on_hi =
            ((bc_hi == amrex::BCType::ext_dir) &&
             idx[idim] >= domain.bigEnd(idim));
          // use idx for idxp
          idx[idim] += 1;
          int idxm[3] = {i, j, k};
          idxm[idim] -= 1;
          efield_ma[box_no](i, j, k, idim) =
            factor *
            (phiV(idx[0], idx[1], idx[2]) - phiV(idxm[0], idxm[1], idxm[2]));
          if (on_lo) {
            efield_ma[box_no](i, j, k, idim) =
              factor * (phiV(idx[0], idx[1], idx[2]) + phiV(i, j, k) -
                        2.0 * phiV(idxm[0], idxm[1], idxm[2]));
          }
          if (on_hi) {
            efield_ma[box_no](i, j, k, idim) =
              factor * (2.0 * phiV(idx[0], idx[1], idx[2]) - phiV(i, j, k) -
                        phiV(idxm[0], idxm[1], idxm[2]));
          }
        });
      // Shift outside?
      amrex::Gpu::streamSynchronize();
    }

    auto const& efield_const_ma = efield_cc.const_arrays();
    auto const& mob_cc_ma = ldata_p->mob_cc.const_arrays();
    auto const& uDrMax_ma = driftVelMax_cc.arrays();

    // Get cell centered max effective velocities across
    // all dimension/ions
    amrex::ParallelFor(
      ldata_p->state,
      [state_ma, efield_const_ma, mob_cc_ma,
       uDrMax_ma] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        amrex::Array4<amrex::Real const> vel(state_ma[box_no], VELX);
        amrex::Real maxVel = 0.0;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          for (int n = 0; n < NUM_IONS; ++n) {
            amrex::Real ueff =
              vel(i, j, k, idim) + mob_cc_ma[box_no](i, j, k, n) *
                                     efield_const_ma[box_no](i, j, k, idim);
            maxVel = amrex::max<amrex::Real>(maxVel, std::abs(ueff));
          }
        }
        uDrMax_ma[box_no](i, j, k) = maxVel;
      });
    amrex::Gpu::streamSynchronize();
    const auto dx = Geom(lev).CellSizeArray();
    const amrex::Real cfl_lcl = m_cfl;
    estdt_lev = amrex::ReduceMin(
      driftVelMax_cc, 0,
      [dx, cfl_lcl] AMREX_GPU_HOST_DEVICE(
        amrex::Box const& bx,
        amrex::Array4<amrex::Real const> const& ueffm) noexcept -> amrex::Real {
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);
#if !defined(__CUDACC__) || (__CUDACC_VER_MAJOR__ != 9) || \
  (__CUDACC_VER_MINOR__ != 2)
        amrex::Real velmax = std::numeric_limits<amrex::Real>::min();
#else
        amrex::Real velmax = -1.e37;
#endif
        for (int k = lo.z; k <= hi.z; ++k) {
          for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
              velmax = amrex::max<amrex::Real>(velmax, ueffm(i, j, k));
            }
          }
        }
        return dx[0] / velmax * cfl_lcl;
      });

    // Min across levels
    estdt = amrex::min<amrex::Real>(estdt, estdt_lev);
  }

  // Min across processors
  amrex::ParallelDescriptor::ReduceRealMin(estdt);

  return estdt;
}
