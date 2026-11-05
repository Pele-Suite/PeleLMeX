#!/usr/bin/env bash
#
# Mapping-awareness of the derived fields and positions.
#
#  constant : PipeFlow on a ConstantMap (fac = 2 in one direction, AMReX
#             extent halved) against the unmapped reference on the same
#             physical mesh.  Every plotfile field must agree to round-off,
#             including the gradient derives (mag_vort, vorticity, Qcrit,
#             enstrophy) and the x/y/z coordinate derives: a derive that
#             differences on the Xi grid is off by exactly the factor 2 in
#             the mapped direction, so compare_plotfiles.py (no --skip) is a
#             sharp test.
#  tanh     : the same case on a TanhStretchMap; the coordinate derives
#             must reproduce the analytic map (check_coordinates.py).
#  spark    : (if a HotBubble 2D executable exists) a spark in a sealed box,
#             uniform vs ConstantMap y-factor 2: the heated temperature field
#             must match cell for cell, i.e. the ignition sphere has its
#             physical radius.  The radius is chosen off the cell lattice so
#             no cell sits exactly on the sphere.
#
# Usage:  [BIN3D=...] [BIN2D=...] [RESULTS_DIR=...] [MPIRUN="mpiexec -n 2"] ./run_derives_equivalence.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE_DIR="$HERE/../PipeFlow"
BUBBLE_DIR="$HERE/../HotBubble"
: "${BIN3D:=$(ls -t "$PIPE_DIR"/PeleLMeX3d.*.ex 2>/dev/null | head -n 1 || true)}"
: "${BIN2D:=$(ls -t "$BUBBLE_DIR"/PeleLMeX2d.*.ex 2>/dev/null | head -n 1 || true)}"
: "${RESULTS_DIR:=${HERE}/results/derives}"
: "${MPIRUN:=}"
INP="$PIPE_DIR/input.3d-Poiseuille"

if [[ -z "$BIN3D" ]]; then
  echo "ERROR: no PipeFlow 3D executable found in $PIPE_DIR"
  exit 1
fi
mkdir -p "$RESULTS_DIR"

COMMON=(amr.max_level=0 amr.max_step=2 amr.stop_time=1.0 amr.plot_int=2
        amr.check_int=-1 amr.plot_file=plt_ amr.n_cell="32 16 16"
        prob.perturbMag=0.5 peleLM.do_temporals=0
        "amr.derive_plot_vars=avg_pressure mag_vort vorticity Qcrit enstrophy coordinates"
        amrex.fpe_trap_invalid=0 amrex.fpe_trap_zero=0 amrex.fpe_trap_overflow=0)

run_case() {
  local tag="$1"; shift
  local outdir="$RESULTS_DIR/$tag"
  mkdir -p "$outdir"
  rm -rf "$outdir"/plt_* "$outdir/run.log"
  echo "=== [$(date +%T)] $tag"
  ( cd "$outdir" && $MPIRUN "$BIN3D" "$INP" "${COMMON[@]}" "$@" > run.log 2>&1 )
}

run_spark() {
  local tag="$1"; shift
  local outdir="$RESULTS_DIR/$tag"
  mkdir -p "$outdir"
  rm -rf "$outdir"/plt_* "$outdir/run.log"
  echo "=== [$(date +%T)] $tag"
  ( cd "$outdir" && $MPIRUN "$BIN2D" "$BUBBLE_DIR/input.2d-regt" \
      amr.n_cell="32 64" amr.max_level=0 amr.max_step=1 amr.stop_time=1.0 \
      amr.fixed_dt=1.0e-5 amr.dt_shrink=1.0 amr.plot_int=1 amr.check_int=-1 \
      amr.plot_file=plt_ geometry.is_periodic="1 0" \
      peleLM.lo_bc="Interior NoSlipWallAdiab" peleLM.hi_bc="Interior NoSlipWallAdiab" \
      peleLM.gravity="0.0 0.0 0.0" prob.T_bubble=300.0 peleLM.do_temporals=0 \
      peleLM.sparks=sp1 "peleLM.sp1.location=0.008 0.016" peleLM.sp1.radius=0.0031 \
      peleLM.sp1.time=0.0 peleLM.sp1.duration=1.0e-5 peleLM.sp1.temp=1500.0 \
      mac_proj.bottom_solver=bicgcg \
      amrex.fpe_trap_invalid=0 amrex.fpe_trap_zero=0 amrex.fpe_trap_overflow=0 \
      "$@" > run.log 2>&1 )
}

FAIL=0
run_case ref geometry.prob_lo="0.0 -0.01 -0.01" geometry.prob_hi="0.04 0.01 0.01"
run_case mapped_x geometry.prob_lo="0.0 -0.01 -0.01" geometry.prob_hi="0.02 0.01 0.01" \
    geometry.mesh_mapping=ConstantMap "ConstantMap.scaling_factor=2.0 1.0 1.0"
run_case mapped_y geometry.prob_lo="0.0 -0.01 -0.01" geometry.prob_hi="0.04 0.0 0.01" \
    geometry.mesh_mapping=ConstantMap "ConstantMap.scaling_factor=1.0 2.0 1.0"
run_case mapped_z geometry.prob_lo="0.0 -0.01 -0.01" geometry.prob_hi="0.04 0.01 0.0" \
    geometry.mesh_mapping=ConstantMap "ConstantMap.scaling_factor=1.0 1.0 2.0"
for d in x y z; do
  case $d in x) fac=2,1,1;; y) fac=1,2,1;; z) fac=1,1,2;; esac
  echo "--- compare mapped_$d vs ref"
  python3 "$HERE/compare_plotfiles.py" "$RESULTS_DIR/ref/plt_00002" \
      "$RESULTS_DIR/mapped_$d/plt_00002" --fac "$fac" --tol 1e-10 || FAIL=1
done

run_case tanh geometry.prob_lo="0.0 -0.01 -0.01" geometry.prob_hi="0.04 0.01 0.01" \
    geometry.mesh_mapping=TanhStretchMap "TanhStretchMap.beta=0.0 2.0 1.0"
echo "--- coordinates of the tanh run vs the analytic map"
python3 "$HERE/check_coordinates.py" "$RESULTS_DIR/tanh/plt_00002" --beta 0,2,1 || FAIL=1

if [[ -n "$BIN2D" ]]; then
  run_spark spark_ref geometry.prob_lo="0.0 0.0" geometry.prob_hi="0.016 0.032"
  run_spark spark_mapped_y geometry.prob_lo="0.0 0.0" geometry.prob_hi="0.016 0.016" \
      geometry.mesh_mapping=ConstantMap "ConstantMap.scaling_factor=1.0 2.0"
  echo "--- spark: mapped_y vs ref (temperature field after one heated step)"
  python3 "$HERE/compare_plotfiles.py" "$RESULTS_DIR/spark_ref/plt_00001" \
      "$RESULTS_DIR/spark_mapped_y/plt_00001" --fac 1,2 --tol 1e-10 --only temp || FAIL=1
else
  echo "WARNING: no HotBubble 2D executable in $BUBBLE_DIR; spark check skipped"
fi

if [[ "$FAIL" -eq 0 ]]; then echo "ALL OK"; else echo "FAILURES"; fi
exit "$FAIL"
