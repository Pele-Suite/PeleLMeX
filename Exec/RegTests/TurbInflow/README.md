## TurbInflow
Injection of a 3D precomputed turbulent velocity field at the boundary of a
PeleLMeX simulation. Testing the turbulence injection functionality. Use the
TurbInflowGenerator support code from PelePhysics to generate the turbulent
boundary data (see PelePhysics documentation for more details).

### Mesh-mapped meshes

`input.3d_ConstantMap` / `input.3d_ConstantMap_ref` (scored by
`compare_inflow_plane.py`) check that a physically-uniform file is sampled at
the physical positions of a mapped target; `input.3d_TanhStretch` is the
smoke case for a wall-clustered map.

Files extracted from a mesh-mapped precursor are uniform in that run's
computational (Xi) coordinate and carry a `MESHMAP` trailer written by the
PelePhysics `TurbInflowGenerator`.  `input.3d_TanhStretch_gen` dumps the tanh
run's inflow ghost layer every step (flat 3D `ghostTanh_*` plotfiles); the
generator turns them into `TurbTanhRT` with the run's map; then
`input.3d_TanhStretch_rt` injects it on the same mesh (the ghost layer must
reproduce the source to round-off) and `input.3d_Uniform_xmap` on a uniform
mesh (must match the source interpolated in the file's Xi at the physical
cell centres), and `input.3d_TanhStretch_rt_amr` on the same tanh mesh with
a refined level covering the inflow face (level 0 exact, level 1 checked
against interpolation in the file's Xi at the fine positions).
`check_inflow_roundtrip.py` scores all of them (`--level` selects the AMR
level of the ghost dump); the exact commands are in
`.github/workflows/linux.yml`.
