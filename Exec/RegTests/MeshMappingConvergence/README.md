# Mesh-mapping convergence tests

This harness drives PeleLMeX with several mesh-mapping configurations
and reports velocity-field norms so we can check that mapping produces
numerically consistent results.

Three sweeps are available:

| driver                       | physical case | regime       | primary purpose |
|------------------------------|---------------|--------------|-----------------|
| `run_incompressible.sh`      | `PipeFlow`    | incompressible, inviscid | clean convergence check on the mechanical scaling (MAC proj, nodal proj, advection, CFL) |
| `run_lowmach.sh`             | `HotBubble`   | low-Mach, inert, gravity ON | end-to-end low-Mach path, *including* buoyancy-feedback amplification |
| `run_lowmach_nograv.sh`      | `HotBubble`   | low-Mach, inert, gravity OFF, conductivity + viscosity ON | low-Mach path WITHOUT buoyancy amplification: deviation plateaus instead of growing |
| `run_stretch_sweep.sh`       | `PipeFlow`    | incompressible, inviscid | Sweep through mesh stretching factors (requires an executable built with `make USE_HYPRE=TRUE`) |

## Protocol (all three sweeps)

For each of the first three cases, five runs are executed at each resolution `N`:

1. **ref** — `fac = (1,1,…)` on an AMReX grid matching the physical domain.
2. **ident** — `fac = (1,1,…)` with mesh mapping *enabled* (sanity / bit-identity check).
3. **mapped_x** — `fac_x = 2`, AMReX x-extent halved, physical extent preserved.
4. **mapped_y** — `fac_y = 2`, AMReX y-extent halved, physical extent preserved.
5. **mapped_z** — `fac_z = 2`, only in 3D.

Every run has the same physical domain, the same `n_cell` per direction,
and therefore the same physical cell spacing.  Cells at index `(i,j,k)`
occupy the same physical location in every configuration, so
cell-by-cell comparison is meaningful.

Initial conditions in both `PipeFlow/pelelmex_prob.H` and
`HotBubble/pelelmex_prob.{H,cpp}` are *mapping-aware*: they evaluate the
IC from physical coordinates (`x_phys = prob_lo + (i + 0.5) * dx *
fac`), so the starting state agrees bit-for-bit across ref and mapped
runs when the AMReX grid spans the same physical region.

The driver scripts pin `amr.max_level = 0`: the sweeps compare
discretisations on identical single-level grids.  Multi-level mesh
mapping itself is exercised by the `TurbInflow` regression
`input.3d_TanhStretch_rt_amr`.

For the stretch sweep, 32 and 64 cell cases are tried with the mesh stretching
beta = 0,1,2,3,4,5,6.  Its defaults use hypre for the MAC projection and the
nodal bottom solver, so build PipeFlow with `make USE_HYPRE=TRUE` (the
GNUmakefile default is `FALSE`, which is enough for the other sweeps).  With
AMReX GMG alone most of these cases fail; the knobs in the script enable
running with beta <=~ 4.

## Physical-space plotfile rendering

When mesh mapping is active, PeleLMeX's `WritePlotFile` emits a per-level
nodal `MultiFab` of node-displacement (`x_phys − x_xi`) alongside the
standard cell-centered data and appends the AMReX ParaView/VisIt
plugin's `amrexvec` trailer/block to the plotfile `Header`, with the
vector components written as `nu_x`, `nu_y`, and `nu_z`.  A
ParaView/VisIt session loading these plotfiles via the AMReX reader
automatically renders the solution on the curvilinear physical grid —
no user-side state file, calculator, or warp filter required.  This is
the same on-disk protocol used by ERF for its terrain-following output
(`WriteGenericPlotfileHeaderWithTerrain`).

Kill-switch: set `peleLM.plot_mesh_mapping = 0` to fall back to the
standard `WriteMultiLevelPlotfile` call, in which case the plotfile
is byte-compatible with the pre-mapping format and renders in
&xi;-space.

`yt` and other tools that only look at the standard cell-centered
variables (including `analyze.py` in this directory) are unaffected by
the extra metadata; they continue to read the plotfile in its native
&xi;-space coordinates.

## Running

```bash
cd Exec/RegTests/MeshMappingConvergence

# Build dependencies first:
#   cd ../PipeFlow && make TPL && make -j
#   cd ../HotBubble && make TPL && make -j

./run_incompressible.sh       # PipeFlow sweep
./run_lowmach.sh              # HotBubble sweep (buoyancy ON)
./run_lowmach_nograv.sh       # HotBubble sweep (buoyancy OFF)
./run_stretch_sweep.sh        # PipeFlow sweep

python3 analyze.py results/   # reads plotfiles, prints tables
```

`analyze.py` takes the *root* that holds the suite directories
(`results/incompressible`, `results/lowmach`, `results/lowmach_nograv`),
which is where the drivers write by default.  If a sweep was run with
`RESULTS_DIR` pointing somewhere else, give `analyze.py` a root with the
suite name under it, e.g. `mkdir -p root && ln -s ../my_run root/incompressible`.

Requires a Python environment with `yt` and `numpy`.

The `NS` environment variable overrides the resolution sweep
(default `"32 64"`); use `NS="32 64 128"` for a full study.  Likewise
`MAX_STEP`, `STOP_TIME`, `CFL`, `VISC`, `COND`, and `RESULTS_DIR` can
be overridden per-invocation.  See the stretch sweep file for additional
controls.

## Expected results

`ref` and every `mapped_*` run discretise the same physical problem on
the same physical mesh, so once the mapped operators are complete they
must agree to round-off, not merely to truncation order.  With the
mapping fixes of September 2026 in place (see below) that is what the
harness shows: at `N = 32`, after one step, every stage of the advance
(viscous forces, Godunov predictor, MAC projection, advection terms,
scalar diffusion, velocity update and nodal projection) agrees between
`ref` and `mapped_{x,y,z}` to 1e-14 relative in the incompressible
sweep and to ~1e-12 in the low-Mach sweeps, and `ident` is bit-identical
to `ref`.

Three mapped-operator errors, each `O(dt)` per step and therefore
invisible in a fixed-dt convergence study, used to produce 0.05 %
(incompressible), 2–3 % (low Mach with gravity) and ~5 %
(`lowmach_nograv`) deviations after 20 steps:

- the viscous tensor operator scaled the face viscosity by `J/fac_i²`,
  which is right for the normal derivatives but leaves the transpose
  and bulk terms off by `fac_i/fac_j` (fixed in AMReX
  `MLTensorOp::setMappingFactors`, AMReX #5909, and PeleLMeX passing
  `fac_fc`; active once PelePhysics pins an amrex containing it —
  until then the incompressible sweep still shows the 0.05 %);
- the `Dhat` diffusion terms of the SDC scalar update were not divided
  by `J` after the ξ-space flux divergence;
- the Godunov predictor and the state extrapolation advected with the
  physical velocity / with `J·ũ` instead of the ξ-velocity `ũ = u/fac`.

They were found with a stage-by-stage comparison of the first step
between `ref` and `mapped_x` (`peleLM.debug_dump_advance`, on the
`mesh/constantmap-drift-debug` branch), not with the end-state norms
`analyze.py` reports: the earlier "higher-order truncation amplified by buoyancy"
reading of the 2–3 % came from L2 ratios dominated by the mean flow,
and the `O(dt⁴)` step-1 scaling was the ratio of two `O(dt)` errors.
End-state norms are a convergence check, not a correctness check; use
the stage dumps for the latter.

For the stretch sweep, even with hypre installed, cases with beta>4 will
fail in one of the projection steps.  Leading up to this failure in beta
one should observe dramatically increasing numbers of solver iterations,
particularly for the mac.  Larger runs at the smaller beta values will be
required to verify convergence order.

The `lowmach_nograv` sweep is the cleanest low-Mach check: it exercises
the divU-aware projection and scalar-advection paths through thermal
expansion of a diffusing hot bubble, without the buoyancy feedback of
the standard HotBubble case.  Run with `Transport_Model = Simple` it
also covers temperature-dependent viscosity and conductivity in the
mapped tensor and Fourier operators.

Species diffusion with a composition gradient (mixture-averaged `D` and
the Wbar correction) is covered by the CI gate below rather than by a
sweep here: the `TurbInflow` ConstantMap pair carries an O2-rich inflow
core and runs with `use_wbar = 1`.  The Soret flux is not yet covered by
any mapped-vs-unmapped comparison (it needs a light species in a
temperature gradient; a FlameSheet-based pair is the natural case).

The same field-by-field check (`compare_plotfiles.py`, tol 1e-10) runs in
CI on that pair, so a change that reintroduces a mapped-operator drift
fails CI rather than showing up as a slow convergence-study surprise.

## Historical caveat (resolved)

Earlier versions of this harness suspected a σ_x-only bug in
`MLNodeLaplacian::updateVelocity`/`getFluxes` in AMReX.  That fix landed
in AMReX (`mlndlap_mknewu_ha`, feature macro
`AMREX_MLNODELAP_HAS_MKNEWU_HA`) and the PeleLMeX post-hoc σ_x
correction is gated on the macro; it did not change the deviation,
which had the three causes listed above.

## Output structure

```
results/
  incompressible/
    ref_N32/        ident_N32/    mapped_{x,y,z}_N32/
    ref_N64/ ...
  lowmach/
    (same layout, HotBubble with gravity ON)
  lowmach_nograv/
    (same layout, HotBubble with gravity OFF + k, μ ON)
  stretch/
    stretch_N32_b0/
    stretch_N32_b1/
	...
    stretch_N64_b0/
    stretch_N64_b1/
	...
```

Each leaf directory contains PeleLMeX plotfiles and the run log.
