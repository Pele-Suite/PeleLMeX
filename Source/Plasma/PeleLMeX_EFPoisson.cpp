#include <PeleLMeX.H>
#include <PeleLMeX_EF_Constants.H>
#include <PeleLMeX_DiffusionOp.H>

void
PeleLM::poissonSolveEF(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::poissonSolveEF()");
  if (ef_verbose) {
    amrex::Print() << " EF Poisson solve \n";
  }

  // Refill PhiV ghost cells before the solve so the implicit
  // Dirichlet boundary value picked up via setLevelBC reflects
  // bcnormal at the current time.  See PeleLM::diffuseVelocity and
  // PeleLM::differentialDiffusionUpdate for the analogous fixes and
  // the LidDrivenCavity regression for the discovery of this class
  // of bug.
  fillPatchPhiV(a_time);

  // Get the phiV BCRec
  auto bcRecPhiV = fetchBCRecArray(PHIV, 1);

  // Build Poisson RHS: charge distribution
  constexpr int nGhost = 0;
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> rhsPoisson(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    rhsPoisson[lev].reset(new amrex::MultiFab(
      grids[lev], dmap[lev], 1, nGhost, amrex::MFInfo(), *m_factory[lev]));

    auto ldata_p = getLevelDataPtr(lev, a_time);

    auto const& state_ma = ldata_p->state.const_arrays();
    auto const& rhs_ma = rhsPoisson[lev]->arrays();

    constexpr amrex::Real factor = -1.0;
    amrex::ParallelFor(
      ldata_p->state, [state_ma, rhs_ma, zk = zk] AMREX_GPU_DEVICE(
                        int box_no, int i, int j, int k) noexcept {
        amrex::Array4<amrex::Real const> nE(state_ma[box_no], NE);
        amrex::Array4<amrex::Real const> rhoY(state_ma[box_no], FIRSTSPEC);
        rhs_ma[box_no](i, j, k) = -nE(i, j, k) * elemCharge * factor;
        for (int n = 0; n < NUM_SPECIES; ++n) {
          rhs_ma[box_no](i, j, k) += zk[n] * rhoY(i, j, k, n) * factor;
        }
      });
    amrex::Gpu::streamSynchronize();
  }
  // Solve for PhiV
  getDiffusionOp()->diffuse_scalar(
    GetVecOfPtrs(getPhiVVect(a_time)), 0, GetVecOfConstPtrs(rhsPoisson), 0, {},
    0, {}, {}, {}, 0, bcRecPhiV, 1, 1, -eps0 * epsr, {}, "PhiV");
}
