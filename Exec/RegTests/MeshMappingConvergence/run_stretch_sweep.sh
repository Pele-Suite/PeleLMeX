#!/usr/bin/env bash
#
# Exponential-stretch convergence sweep.
#
# Drives Exec/RegTests/PipeFlow with ExpStretchMap in y at several beta
# values and resolutions.  Reports:
#   - whether the MAC + nodal MLMG solves converged within budget;
#   - total MLMG iteration counts per run (summed across all MAC and
#     nodal solves);
#   - final plotfiles so the analyzer can compute self-convergence
#     slopes (||u_N - u_{N/2}||_L2) as a function of beta.
#
# Solver knobs are deliberately loosened vs. PeleLMeX defaults (rtol =
# 1e-8, maxiter = 5000) so that slow-converging mapped problems finish
# instead of aborting, and we can see *where* the solver starts to
# suffer rather than just "it aborted".
#
# The stretched axis is y (direction = 1), wall at the low end,
# matching PipeFlow's no-slip walls.  Physical domain and n_cell per
# direction are identical across runs -- only the spatial distribution
# of cells along y changes.
#
# Note: If compile settings other gnu += OMP were used to build the
# executables to be run, pass a full path as BIN=... in the call to this 
# script.  Also note, a number of other settings can be modified, including
# NS, MAX_STEP, STOP_TIME, RESULTS_DIR, etc


set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$HERE/../../.."
PIPEFLOW_DIR="$PROJECT_ROOT/Exec/RegTests/PipeFlow"

if [[ -z "${BIN:-}" ]]; then
  if [[ -x "${PIPEFLOW_DIR}/PeleLMeX3d.gnu.OMP.ex" ]]; then
    BIN="${PIPEFLOW_DIR}/PeleLMeX3d.gnu.OMP.ex"
  else
    BIN="${PIPEFLOW_DIR}/PeleLMeX3d.gnu.ex"
  fi
fi
INP="${PIPEFLOW_DIR}/input.3d-Poiseuille"

: "${BETAS:=0 1 2 3 4 5 6}"
: "${NS:=32 64}"
: "${MAX_STEP:=10}"
: "${MEAN_FLOW:=1.0}"             # dialed down from the stock 18 so the
                                  # MAC problem conditioning is dominated
                                  # by the mapping rather than by the
                                  # velocity perturbations in the IC.
: "${PERTURB_MAG:=0.5}"           # magnitude of the IC sin*sin transverse
                                  # perturbations.  Default 5.0 in the
                                  # PipeFlow IC has wavenumber-11
                                  # content that's under-resolved at
                                  # modest N and dominates self-
                                  # convergence comparisons.  0.5 is
                                  # mild enough that the cell-aligned
                                  # error metric reflects time-evolution
                                  # discretization rather than IC
                                  # under-resolution.
: "${MAC_MAXITER:=5000}"
: "${MAC_RTOL:=1e-8}"
: "${NODAL_MAXITER:=5000}"
: "${NODAL_RTOL:=1e-8}"
# The defaults below need an executable built with hypre:
#   cd ../PipeFlow && make -j8 USE_HYPRE=TRUE   (HYPRE_DIR must point at a hypre install)
# PipeFlow's GNUmakefile defaults to USE_HYPRE=FALSE, which is enough for the
# other sweeps; without hypre set USE_MLHYPRE=0 and NODAL_BOTTOM=bicgcg here,
# expecting failures beyond beta ~ 2.
: "${USE_MLHYPRE:=1}"             # 1: use BoomerAMG for MAC projection via
                                  # HypreMLABecLap (bypasses MLMG).  Requires
                                  # the executable to have been built with
                                  # USE_HYPRE=TRUE.  At 0, the stock MLMG-
                                  # with-bicgstab-bottom path is used.
: "${NODAL_BOTTOM:=hypre}"        # bicg, bicgcg, cg, cgbicg, smoother,
                                  # or hypre (only valid if USE_HYPRE=TRUE
                                  # at build time).  Affects the nodal
                                  # projector's MLMG bottom solver only.
: "${RESULTS_DIR:=${HERE}/results/stretch}"

mkdir -p "$RESULTS_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: $BIN not found.  Build PipeFlow first:"
  echo "  cd $PIPEFLOW_DIR"
  echo "  make TPL && make -j"
  exit 1
fi

run_case() {
  local tag="$1"
  local ncell="$2"        # "Nx Ny Nz"
  local beta="$3"
  local outdir="$RESULTS_DIR/${tag}"
  mkdir -p "$outdir"
  cd "$outdir"
  echo "=== [$(date +%T)] ${tag}  n_cell=${ncell}  beta=${beta}"
  #shellcheck disable=SC2086
  "$BIN" "$INP" \
      amr.n_cell="$ncell" \
      geometry.prob_lo="0.0 -0.01 -0.01" \
      geometry.prob_hi="0.04 0.01 0.01" \
      amr.max_level=0 \
      amr.max_step="$MAX_STEP" \
      amr.plot_file=plt_ \
      amr.check_file=chk_ \
      amr.plot_int="$MAX_STEP" \
      amr.check_int=-1 \
      amrex.fpe_trap_invalid=0 \
      amrex.fpe_trap_zero=0 \
      amrex.fpe_trap_overflow=0 \
      prob.meanFlowMag="$MEAN_FLOW" \
      prob.perturbMag="$PERTURB_MAG" \
      geometry.mesh_mapping=ExpStretchMap \
      ExpStretchMap.direction=1 \
      ExpStretchMap.wall_end=lo \
      ExpStretchMap.beta="$beta" \
      mac_proj.verbose=1 \
      mac_proj.maxiter="$MAC_MAXITER" \
      mac_proj.rtol="$MAC_RTOL" \
      nodal_proj.verbose=1 \
	  mac_proj.bottom_solver         = hypre \
      mac_proj.hypre_namespace       = mac_proj.hypre \
      mac_proj.hypre.hypre_solver    = GMRES \
      mac_proj.hypre.hypre_preconditioner = BoomerAMG \
      mac_proj.hypre.bamg_coarsen_type    = 9 \
      mac_proj.hypre.bamg_interp_type     = 4 \
      mac_proj.hypre.bamg_relax_type      = 7 \
      mac_proj.use_mlhypre = 1 \
      nodal_proj.bottom_solver = hypre \
      nodal_proj.maxiter       = 20000 \
      nodal_proj.rtol          = 1.0e-7 \
      nodal_proj.verbose       = 1 \
      > run.log 2>&1 || true        # keep going even if a run aborts
  cd "$HERE"
  echo "   ... done"
}

for N in $NS; do
  for beta in $BETAS; do
    tag_suffix="N${N}_b${beta}"
    run_case "stretch_${tag_suffix}" "$N $N $N" "$beta"
  done
done

echo "=== stretch sweep done.  Results in $RESULTS_DIR"
echo "    Analyze with:"
echo "      python3 $HERE/analyze_stretch.py $RESULTS_DIR"
