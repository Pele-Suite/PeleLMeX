#!/usr/bin/env bash
#
# Physical-volume mean regression for mesh mapping.
#
# Every domain mean PeleLMeX forms through MFSum must be a physical-volume
# (detJ-weighted) mean, not a mean over the uniform Xi grid.  Two
# algorithmic consumers of that mean are exercised here on a wall-clustered
# TanhStretchMap against the same physical case on a uniform grid:
#
#  channel : PipeFlow, periodic channel with peleLM.do_periodic_channel=1.
#            The background pressure gradient gp0 = <div tau>_x.  The IC
#            (prob.problem_type=3) is u = U (1 - yhat^4), whose viscous
#            force mu u'' varies across the channel, so a Xi-weighted mean
#            is off by O(1) on the stretched mesh.  Exact: -16 mu U / Ly^2.
#            (A parabolic profile would hide the bug: u'' is constant.)
#
#  chamber : HotBubble, sealed box (walls/symmetry, periodic x), heat
#            conduction on (transport.* in CGS).  The closed-chamber
#            algorithm removes the mean of divU before each projection;
#            if that mean is not the detJ-weighted one, the all-Neumann
#            projection RHS is incompatible and dp0/dt is wrong.  The run prints the
#            projection compatibility residual (peleLM.v >= 2) and aborts
#            above peleLM.debug_compatibility_tol when
#            peleLM.debug_check_compatibility=1.
#
# ConstantMap cannot detect either: a constant detJ cancels in every mean.
#
# Usage:  [BIN3D=...] [BIN2D=...] [BETA=2.0] [RESULTS_DIR=...] ./run_mean_integrals.sh
# Then:   python3 check_mean_integrals.py [RESULTS_DIR]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$HERE/../../.."
PIPE_DIR="$PROJECT_ROOT/Exec/RegTests/PipeFlow"
BUBBLE_DIR="$PROJECT_ROOT/Exec/RegTests/HotBubble"

pick_bin() {
  # Any compiler/parallel flavour (gnu, llvm, MPI, OMP, ...); the newest
  # executable wins when several are present.  Override with BIN3D/BIN2D.
  local dir="$1" dim="$2"
  ls -t "$dir"/PeleLMeX"${dim}"d.*.ex 2>/dev/null | head -n 1 || true
}
: "${BIN3D:=$(pick_bin "$PIPE_DIR" 3)}"
: "${BIN2D:=$(pick_bin "$BUBBLE_DIR" 2)}"
: "${BETA:=2.0}"
: "${RESULTS_DIR:=${HERE}/results/mean_integrals}"
: "${MPIRUN:=}"   # e.g. "mpiexec -n 2" for an MPI build

mkdir -p "$RESULTS_DIR"
NFAIL=0
NRUN=0

run_case() {
  local bin="$1" inp="$2" tag="$3"; shift 3
  local outdir="$RESULTS_DIR/$tag"
  mkdir -p "$outdir"
  rm -f "$outdir/FAILED" "$outdir/run.log"
  echo "=== [$(date +%T)] $tag"
  NRUN=$((NRUN + 1))
  # Keep going after a failed run so the other cases still produce logs,
  # but remember it: the script exits non-zero and the checker refuses a
  # log that ends in an abort, so a run that printed its first-step value
  # and then crashed cannot pass.
  if ! ( cd "$outdir" && $MPIRUN "$bin" "$inp" "$@" \
      amrex.fpe_trap_invalid=0 amrex.fpe_trap_zero=0 amrex.fpe_trap_overflow=0 \
      > run.log 2>&1 ); then
    echo "   ... FAILED, see $outdir/run.log"
    NFAIL=$((NFAIL + 1))
    touch "$outdir/FAILED"
  fi
}

# ---------------------------------------------------------------- channel
if [[ -n "$BIN3D" ]]; then
  MU=0.0576; U=18.0
  COMMON3D=(amr.n_cell="64 32 8" amr.max_level=0 amr.max_step=2 amr.stop_time=1.0
            amr.fixed_dt=1.0e-6 amr.plot_int=-1 amr.check_int=-1
            geometry.is_periodic="1 0 1"
            geometry.prob_lo="0.0 -0.01 -0.0025" geometry.prob_hi="0.04 0.01 0.0025"
            peleLM.lo_bc="Interior NoSlipWallAdiab Interior"
            peleLM.hi_bc="Interior NoSlipWallAdiab Interior"
            peleLM.incompressible=1 peleLM.rho=1.0 peleLM.mu=$MU
            peleLM.gradP0="0.0 0.0 0.0"
            peleLM.do_periodic_channel=1 peleLM.periodic_channel_dir=0
            prob.problem_type=3 prob.meanFlowMag=$U prob.perturbMag=0.0
            peleLM.v=3 peleLM.do_temporals=0 nodal_proj.verbose=0)
  run_case "$BIN3D" "$PIPE_DIR/input.3d-Poiseuille" channel_uniform "${COMMON3D[@]}"
  run_case "$BIN3D" "$PIPE_DIR/input.3d-Poiseuille" channel_tanh "${COMMON3D[@]}" \
      geometry.mesh_mapping=TanhStretchMap "TanhStretchMap.beta=0.0 $BETA 0.0"
  # Same stretched case at twice the resolution: the remaining offset from
  # the exact mean is the first-order wall-gradient truncation of the
  # Dirichlet treatment and must shrink with N.
  run_case "$BIN3D" "$PIPE_DIR/input.3d-Poiseuille" channel_tanh_N64 "${COMMON3D[@]}" \
      amr.n_cell="128 64 16" amr.fixed_dt=5.0e-7 \
      geometry.mesh_mapping=TanhStretchMap "TanhStretchMap.beta=0.0 $BETA 0.0"
  echo "channel mu=$MU U=$U Ly=0.02" > "$RESULTS_DIR/channel.params"
else
  echo "WARNING: no PipeFlow 3D executable found in $PIPE_DIR; skipping channel"
fi

# ---------------------------------------------------------------- chamber
if [[ -n "$BIN2D" ]]; then
  COMMON2D=(amr.n_cell="32 64" amr.max_level=0 amr.max_step=2 amr.stop_time=1.0
            amr.fixed_dt=2.0e-5 amr.plot_int=-1 amr.check_int=-1
            geometry.is_periodic="1 0"
            geometry.prob_lo="0.0 0.0" geometry.prob_hi="0.016 0.032"
            peleLM.lo_bc="Interior NoSlipWallAdiab"
            peleLM.hi_bc="Interior NoSlipWallAdiab"
            peleLM.gravity="0.0 0.0 0.0" peleLM.sdc_iterMax=1
            prob.T_bubble=600.0 prob.bubble_radius=0.004 prob.bubble_y0=0.016
            transport.const_viscosity=1.8e-4 transport.const_conductivity=2.6e3
            transport.const_diffusivity=0.2 transport.const_bulk_viscosity=0.0
            peleLM.v=3 peleLM.do_temporals=0 peleLM.debug_check_compatibility=1
            peleLM.debug_compatibility_tol=1.0e-9
            mac_proj.bottom_solver=bicgcg mac_proj.mg_max_coarsening_level=2)
  # The MAC knobs: AMReX GMG with the default bottom solver diverges on the
  # beta=2 wall-clustered mesh (a solver limitation, see run_stretch_sweep.sh;
  # the projection RHS itself is compatible to round-off).
  run_case "$BIN2D" "$BUBBLE_DIR/input.2d-regt" chamber_uniform "${COMMON2D[@]}"
  run_case "$BIN2D" "$BUBBLE_DIR/input.2d-regt" chamber_tanh "${COMMON2D[@]}" \
      geometry.mesh_mapping=TanhStretchMap "TanhStretchMap.beta=0.0 $BETA"
else
  echo "WARNING: no HotBubble 2D executable found in $BUBBLE_DIR; skipping chamber"
fi

echo "=== done.  $NRUN runs, $NFAIL failed.  Results in $RESULTS_DIR"
echo "    python3 $HERE/check_mean_integrals.py $RESULTS_DIR"
if [[ "$NRUN" -eq 0 ]]; then
  echo "ERROR: no executable found, nothing was run"
  exit 1
fi
[[ "$NFAIL" -eq 0 ]]
