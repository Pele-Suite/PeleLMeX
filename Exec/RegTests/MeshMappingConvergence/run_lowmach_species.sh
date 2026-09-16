#!/usr/bin/env bash
#
# Mesh-mapping round-off equivalence for the species-diffusion operators.
#
# Drives Exec/RegTests/HotBubble (2D, gravity off) with a hot, H2-doped
# N2 bubble in air, so that with Simple transport all of
#   - mixture-averaged species diffusion (composition gradient),
#   - the Wbar correction flux (use_wbar),
#   - the Soret flux (use_soret: light species + temperature gradient),
# are non-zero.  The unmapped reference and the ConstantMap runs share one
# physical mesh on index-identical grids, so every field of the final
# plotfile must agree to round-off (compare_plotfiles.py, tol 1e-10).
#
# Build HotBubble with a hydrogen mechanism and Simple transport, e.g.
#   cd ../HotBubble && make realclean
#   make -j8 COMP=llvm USE_MPI=TRUE Chemistry_Model=LiDryer Transport_Model=Simple
# and pass the executable as BIN=...  (the GNUmakefile defaults, air +
# Constant transport, run but exercise none of the above: the bubble is then
# plain N2 and Soret is identically zero).
#
# Knobs (env): NS (default "32"), MAX_STEP (10), RESULTS_DIR, TOL (1e-10),
#              YH2 (0.02), T_BUBBLE (600), MPIRUN (e.g. "mpirun -n 2").

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$HERE/../../.."
HOTBUBBLE_DIR="$PROJECT_ROOT/Exec/RegTests/HotBubble"
if [[ -z "${BIN:-}" ]]; then
  if [[ -x "${HOTBUBBLE_DIR}/PeleLMeX2d.gnu.OMP.ex" ]]; then
    BIN="${HOTBUBBLE_DIR}/PeleLMeX2d.gnu.OMP.ex"
  else
    BIN="${HOTBUBBLE_DIR}/PeleLMeX2d.gnu.ex"
  fi
fi
INP="${HOTBUBBLE_DIR}/input.2d-regt"

: "${NS:=32}"
: "${MAX_STEP:=10}"
: "${STOP_TIME:=0.05}"
: "${CFL:=0.5}"
: "${YH2:=0.02}"
: "${T_BUBBLE:=600.0}"
: "${TOL:=1e-10}"
: "${MPIRUN:=}"
: "${RESULTS_DIR:=${HERE}/results/lowmach_species}"

mkdir -p "$RESULTS_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: $BIN not found.  Build HotBubble first (see header) and pass BIN=..."
  exit 1
fi

COMMON_OVERRIDES=(
  "peleLM.gravity=0.0 0.0 0.0"
  "peleLM.v=1"
  "peleLM.do_react=0"
  "peleLM.use_wbar=1"
  "transport.use_soret=1"
  "prob.use_mix_bubble=1"
  "prob.mix_bubble_hot=1"
  "prob.bubble_YH2=${YH2}"
  "prob.T_bubble=${T_BUBBLE}"
)

run_case() {
  local tag="$1"
  local ncell="$2"
  local plo="$3"
  local phi="$4"
  local mm_args="$5"
  local outdir="$RESULTS_DIR/${tag}"
  mkdir -p "$outdir"
  cd "$outdir"

  echo "=== [$(date +%T)] $tag  n_cell=${ncell}  mapping='${mm_args}'"
  #shellcheck disable=SC2086
  $MPIRUN "$BIN" "$INP" \
      amr.n_cell="$ncell" \
      "geometry.prob_lo=$plo" "geometry.prob_hi=$phi" \
      amr.max_level=0 \
      amr.max_step="$MAX_STEP" \
      amr.stop_time="$STOP_TIME" \
      amr.plot_file=plt_ \
      amr.check_file=chk_ \
      amr.plot_int="$MAX_STEP" \
      amr.check_int=-1 \
      amr.cfl="$CFL" \
      amr.dt_shrink=1.0 \
      amr.init_dt=1.0e-5 \
      amr.max_dt=5.0e-4 \
      amr.dt_change_max=1.3 \
      amrex.fpe_trap_invalid=0 \
      amrex.fpe_trap_zero=0 \
      amrex.fpe_trap_overflow=0 \
      "${COMMON_OVERRIDES[@]}" \
      $mm_args \
      > run.log 2>&1
  cd "$HERE"
  echo "   ... done"
}

status=0
for N in $NS; do
  Ny=$((2 * N))
  s="N${N}"
  run_case "ref_${s}"      "$N $Ny" "0.0 0.0" "0.016 0.032" ""
  run_case "ident_${s}"    "$N $Ny" "0.0 0.0" "0.016 0.032" \
           "geometry.mesh_mapping=ConstantMap ConstantMap.scaling_factor=1.0 1.0"
  run_case "mapped_x_${s}" "$N $Ny" "0.0 0.0" "0.008 0.032" \
           "geometry.mesh_mapping=ConstantMap ConstantMap.scaling_factor=2.0 1.0"
  run_case "mapped_y_${s}" "$N $Ny" "0.0 0.0" "0.016 0.016" \
           "geometry.mesh_mapping=ConstantMap ConstantMap.scaling_factor=1.0 2.0"

  # The reference must have seen the operators this sweep is about.
  if ! grep -q "Soret" "$RESULTS_DIR/ref_${s}/run.log"; then
    echo "WARNING: run.log does not mention Soret; is the executable built with Simple transport?"
  fi

  last=$(printf "plt_%05d" "$MAX_STEP")
  for tag in ident mapped_x mapped_y; do
    case $tag in
      ident)    fac="1,1" ;;
      mapped_x) fac="2,1" ;;
      mapped_y) fac="1,2" ;;
    esac
    echo "--- ${tag}_${s} vs ref_${s}"
    if ! python3 "$HERE/compare_plotfiles.py" \
           "$RESULTS_DIR/ref_${s}/${last}" "$RESULTS_DIR/${tag}_${s}/${last}" \
           --fac "$fac" --tol "$TOL"; then
      status=1
    fi
  done
done

echo "=== lowmach_species sweep done.  Results in $RESULTS_DIR"
exit $status
