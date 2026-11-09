#!/bin/bash
# Decomposition-independence check for the ion time-step estimate.
#
# Runs the case twice on one rank, once as a single box and once split into
# many boxes, and requires identical ion dt estimates. Stencils that read
# unfilled ghost cells (e.g. the phiV gradient in estEFIonsDt) give different
# values at interior box edges, so the two runs disagree.
#
# Usage: ./check_decomposition.sh [path/to/PeleLMeX2d...PLASMA.ex] [input]
set -u
cd "$(dirname "$0")"
EXE=${1:-$(ls -t PeleLMeX2d*.ex 2>/dev/null | head -1)}
INP=${2:-input.2d-regt}
if [ -z "$EXE" ] || [ ! -x "$EXE" ]; then
  echo "No executable found: build with make first, or pass it as \$1"
  exit 2
fi
OPTS="amr.max_step=1 amr.plot_int=-1 amr.check_int=-1 amr.blocking_factor=16"

run() { # max_grid_size logfile
  "$EXE" "$INP" $OPTS amr.max_grid_size="$1" > "$2" 2>&1
  local rc=$?
  grep -o "ions: [0-9.e+-]*" "$2" > "$2.dt"
  return $rc
}

run 128 decomp_1box.log; rc1=$?
run 32 decomp_nbox.log; rc2=$?

if [ ! -s decomp_1box.log.dt ]; then
  echo "FAIL: no ion dt estimates in decomp_1box.log (exit $rc1)"
  exit 1
fi
if [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] &&
  cmp -s decomp_1box.log.dt decomp_nbox.log.dt; then
  echo "PASS: ion dt estimates match ($(wc -l < decomp_1box.log.dt) estimates)"
  exit 0
fi
echo "FAIL: ion dt estimates depend on the box decomposition"
echo "  exit codes: 1 box $rc1, many boxes $rc2"
echo "  1 box            many boxes"
paste decomp_1box.log.dt decomp_nbox.log.dt | awk '$2 != $4'
grep -m1 "Abort" decomp_nbox.log
exit 1
