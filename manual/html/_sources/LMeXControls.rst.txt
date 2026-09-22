PeleLMeX controls
=================

.. _sec:control:

The input file specified on the command line is a free-format text file, one entry per row, that specifies input data processed by the AMReX ``ParmParse`` module.
This file needs to specified along with the executable as an `argv` option, for example:

::

    mpirun -np 64 ./PeleLMeX2d.xxx.ex inputs

Also, any entry that can be specified in the inputs file can also be specified on the command line; values specified on the command line override values in the inputs file, e.g.:

::

    mpirun -np 64 ./PeleLMeX2d.xxx.ex inputs amr.max_level=2

The available options are divided into groups: those that control primarily AMReX are prefaced with `amr.`, those that are specific to the PeleLMeX are prefaced by `peleLM.`, while those corresponding to the various pieces of the algorithm are prefaced with specific keys, such that `diffusion`, `nodal_proj`, ... as described below.

Computational domain definition
-------------------------------

::

    #--------------------GEOMETRY DEFINITION-----------------------
    geometry.is_periodic = 1 1 0              # For each dir, 0: non-perio, 1: periodic
    geometry.coord_sys   = 0                  # 0 => cart, 1 => RZ
    geometry.prob_lo     = 0.0   0.0   0.0    # x_lo y_lo (z_lo)
    geometry.prob_hi     = 0.016 0.016 0.032  # x_hi y_hi (z_hi)

    #-----------------------BNDY CONDITIONS------------------------
    # Interior, Inflow, Outflow, Symmetry, SlipWallAdiab, NoSlipWallAdiab
    # SlipWallIsotherm, NoSlipWallIsotherm
    # Periodic direction must be set as Interior
    peleLM.lo_bc = Interior Interior Inflow
    peleLM.hi_bc = Interior Interior Inflow

If specifying boundaries as ``Inflow``, the `MyProblemSpecificFunctions::bcnormal` function must be defined
in the ``pelelmex_prob.H`` file for the case to define the inflow conditions.
Note that the ``bcnormal`` function is also called for wall boundaries but should
not modify them (except setting temperature on isothermal walls). Therefore, if
there are a mixture of inflows with other boundary types, the bcnormal function
should check which face is being set when applying the inflow boundary condition.
``Inflow`` boundaries may also be augmented with spatially and temporally
varying turbulent fluctuations using the ``TurbInflow`` utility from
PelePhysics. See the ``Exec/RegTests/TurbInflow`` test for an example of how
to use this capability and the
`PelePhysics  documentation <https://pele-suite.github.io/PelePhysics/Utility.html#turbulent-inflows>`_
for the relevant input file flags.

Grid/AMR parameters
-------------------

::

    #-------------------------AMR CONTROL--------------------------
    amr.n_cell          = 64 64 128        # Number of cells on Level 0 in each direction
    amr.v               = 1                # [OPT, DEF=0] AMR verbose
    amr.max_level       = 1                # maximum level number allowed
    amr.ref_ratio       = 2 2 2 2          # refinement ratio, one per refinement level
    amr.regrid_int      = 5                # how often to regrid
    amr.n_error_buf     = 1 1 2 2          # number of buffer cells in error est, one per refinement level
    amr.grid_eff        = 0.7              # what constitutes an efficient grid
    amr.blocking_factor = 16               # block factor in grid generation (min box size)
    amr.max_grid_size   = 64               # max box size

    peleLM.max_grid_size_chem = 32         # [OPT, DEF="None"] Max box size for the Chemistry BoxArray

Load balancing
--------------

*PeleLMeX* relies on two distribution maps (see :doc:`Implementation` for more details).

::

    peleLM.do_load_balancing = 1                    # [OPT, DEF=0] Activate load balancing
    peleLM.load_balancing_method = sfc              # [OPT, DEF="sfc"] AmrCore dmap load balancing method
    peleLM.load_balancing_cost_estimate = ncell     # [OPT, DEF="ncell"] AmrCore dmap balancing cost
    peleLM.chem_load_balancing_method = knapsack    # [OPT, DEF="knapsack"] Chemistry dmap load balancing method
    peleLM.chem_load_balancing_cost_estimate = chemfunctcall_sum # [OPT, DEF="chemfunctcall_sum"] Chemistry dmap balancing cost
    peleLM.load_balancing_efficiency_threshold = 1.05  # What constitute a better dmap ?

The balancing method can be one of `sfc`, `roundrobin` or `knapsack`, while the cost estimate can be one of
`ncell`, `chemfunctcall_avg`, `chemfunctcall_max`, `chemfunctcall_sum`, `userdefined_avg` or `userdefined_sum`. When
using either of the last to option, the user must provide a definition for the `derUserDefined`. If multiple components
are defined in the `derUserDefined` function, the first one is used for load balancing.

Time stepping parameters
------------------------

::

    #-------------------------TIME STEPPING------------------------
    amr.max_step      = 20                 # Maximum number of steps
    amr.stop_time     = 0.001              # Maximum simulation time [s]
    amr.max_wall_time = 1.0                # Maximum wall clock time [hr]
    amr.cfl           = 0.5                # [OPT, DEF=0.7] CFL for advection-controlled dt estimate
    amr.fixed_dt      = 1e-6               # [OPT] optional fixed dt (override CFL condition)
    amr.min_dt        = 1e-11              # [OPT, DEF=1e-12] small time step size limit triggering simulation termination
    amr.init_dt       = 1e-6               # [OPT] optional initial dt (override CFL condition upon initialization)
    amr.dt_shrink     = 0.0001             # [OPT, DEF=1.0] dt factor upon initialization
    amr.dt_change_max = 1.1                # [OPT, DEF=1.1] maximum dt change between consecutive steps

.. note::
   Note that one of `amr.max_step`, `amr.stop_time`, or `amr.max_wall_time` is required, and if more than one is specified,
   the first stopping criterion encountered will lead to termination of the simulation.

IO parameters
-------------

::

    #--------------------------IO CONTROL--------------------------
    peleLM.base_output_prefix = outdir/    # [OPT, DEF=""] Prefix applied across all output types (plt, chk, diagnostics, temporals)
    amr.plot_int         = 20              # [OPT, DEF=-1] Frequency (as step #) for writing plot file
    amr.plot_overwrite   = false           # [OPT, DEF=false] Overwrite plot files with same name if present
    amr.plot_init_state  = false           # [OPT, DEF=false] Create a plot file during initialization before the initial projections
    amr.plot_per         = 0.002           # [OPT, DEF=-1] Period (time in s) for writing plot file
    amr.plot_per_exact   = 1               # [OPT, DEF=0] Flag to enforce exactly plt_per by shortening dt
    amr.plot_file        = "plt_"          # [OPT, DEF="plt_"] Plot file prefix
    amr.check_int        = 100             # [OPT, DEF=-1] Frequency (as step #) for writing checkpoint file
    amr.check_overwrite  = false           # [OPT, DEF=false] Overwrite checkpoint files with same name if present
    amr.check_per        = 0.05            # [OPT, DEF=-1] Period (time in s) for writing checkpoint file
    amr.check_file       = "chk"           # [OPT, DEF="chk"] Checkpoint file prefix
    amr.file_stepDigits  = 6               # [OPT, DEF=5] Number of digits when adding nsteps to plt and chk names
    amr.derive_plot_vars = avg_pressure ...# [OPT, DEF=""] List of derived variable included in the plot files
    amr.plot_speciesState = 0              # [OPT, DEF=0] Force adding state rhoYs to the plot files
    peleLM.plot_extSource = false          # [OPT, DEF=false] Force adding state external sources to the plot files

    amr.restart          = chk00100        # [OPT, DEF=""] Checkpoint from which to restart the simulation
    amr.initDataPlt      = plt01000        # [OPT, DEF=""] Provide a plotfile from which to extract initial data
    peleLM.initDataPlt_reset_time = 1      # [OPT, DEF=1] Resets time and nsteps to 0 after restarting from a plot file. (Warning: plot file will be rewritten if not renamed and argument value = 0)
    peleLM.initDataPlt_coarsen = 0         # [OPT, DEF=0] Enable restarting from plotfile with 2x coarser base mesh (non-EB only). Uses multi-level mapping.
    peleLM.initDataPlt_specname_map = spec1 spec2 # [OPT, DEF=""] If specified, lookup these entries instead of species names when populating species from the init plot file (length must match NUM_SPECIES)
    peleLM.initDataPlt_patch_flow_variables = false # [OPT, DEF=false] Enable user-defined flow variable patching after reading a plot solution file
    amr.regrid_on_restart = 1              # [OPT, DEF="0"] Trigger a regrid after the data from checkpoint are loaded
    amr.n_files          = 64              # [OPT, DEF="min(256,NProcs)"] Number of files to write per level

Refinement controls
-------------------

Refinement in PeleLMeX is controlled by a set of 'Tagging' criterion listed under the `amr.refinement_indicators`
key. For each criteriq, the user needs to supply a definition. For example, the following provides a complete
overview of the available controls:

::

    amr.refinement_indicators gthan lthan adjd box1

    amr.gthan.max_level     = 3
    amr.gthan.value_greater = 0.005
    amr.gthan.field_name    = x_velocity

    amr.lthan.max_level     = 4
    amr.lthan.value_less    = 400.0
    amr.lthan.field_name    = temp
    amr.lthan.start_time    = 0.001
    amr.lthan.end_time      = 0.005

    amr.adjd.max_level                   = 2
    amr.adjd.adjacent_difference_greater = 0.05
    amr.adjd.field_name                  = density

    amr.box1.max_level      = 1
    amr.box1.in_box_lo      = 0.0 0.0 0.0
    amr.box1.in_box_hi      = 0.01 0.01 0.05

The `field_name` can be any of the state or derived variables (see below) component. Additional controls specific
to embedded boundaries are discussed below.

PeleLMeX derived variables
--------------------------

The following list of derived variables are available in PeleLMeX:

.. list-table:: PeleLMeX derived variables
    :widths: 25 25 100
    :header-rows: 1

    * - Key
      - Size (nComp)
      - Brief
    * - `mass_fractions`
      - NUM_SPECIES
      - Species mass fractions
    * - `mole_fractions`
      - NUM_SPECIES
      - Species mole fractions
    * - `element_fractions`
      - NUM_ELEMENTS
      - Element mass fractions
    * - `diffcoeff`
      - NUM_SPECIES
      - Species mixture-averaged diffusion coefficients
    * - `lambda`
      - 1
      - Thermal diffusivity
    * - `viscosity`
      - 1
      - Mixture viscosity
    * - `mixture_fraction`
      - 1
      - Mixture fraction based on Bilger's element formulation
    * - `progress_variable`
      - 1
      - Progress variable based on a linear combination of Ys, T
    * - `avg_pressure`
      - 1
      - Cell-averaged pressure (from the node-centered pressure)
    * - `mag_vort`
      - 1
      - Vorticity magnitude
    * - `vorticity`
      - AMREX_SPACEDIM*2-3
      - VortZ (2D) or VortX, VortY, VortZ (3D)
    * - `Qcrit`
      - 1
      - Q-Criterion : :math:`0.5(|\boldsymbol{\Omega}|^2 - |\boldsymbol{S}|^2)`
    * - `kinetic_energy`
      - 1
      - Kinetic energy: 0.5 * rho * (u^2+v^2+w^2)
    * - `enstrophy`
      - 1
      - enstrophy: 0.5 * rho * (\omega_x^2+\omega_y^2+\omega_z^2)
    * - `HeatRelease`
      - 1
      - Heat release rate from chem. reactions
    * - `rhominsumrhoY`
      - 1
      - Rho minus sum of rhoYs, for debug purposes
    * - `coordinates`
      - AMREX_SPACEDIM
      - Cell-center coordinates
    * - `DistributionMap`
      - 1
      - The MPI-rank of each box
    * - `derUserDefined`
      - ?
      - A user-defined derived which number of components is provided by the user (see below).

Note that `mixture_fraction` and `progress_variable` requires additional inputs from the users as described below.
The `derUserDefined` allow the user to define its own derived variable which can comprise several components. To do
so, the user need to copy the Source/DeriveUserDefined.cpp file into their run folder and update the file. The number of
components is defined based on the size of the vector returned by `pelelmex_setuserderives()`.  Be sure to add the
user derived variables to the input file via `amr.derive_plot_vars`.

PeleLMeX algorithm
------------------

::

    #-----------------------PELE CONTROL-----------------------
    peleLM.v = 1                           # [OPT, DEF=0] Verbose (1 - verbose for core algorithm, 2+ adds default verbosity for deltaTiters, projections, etc)
    peleLM.run_mode = normal               # [OPT, DEF=normal] Switch between time-advance mode (normal) or UnitTest (evaluate)
    peleLM.use_wbar = 1                    # [OPT, DEF=1] Enable Wbar correction in diffusion fluxes
    peleLM.sdc_iterMax = 2                 # [OPT, DEF=1] Number of SDC iterations
    peleLM.num_init_iter = 2               # [OPT, DEF=3] Number of iterations to get initial pressure
    peleLM.num_divu_iter = 1               # [OPT, DEF=1] Number of divU iterations to get initial dt estimate
    peleLM.do_init_proj = 1                # [OPT, DEF=1] Control over initial projection
    peleLM.advection_scheme = Godunov_BDS  # [OPT, DEF=Godunov_PLM] Advection scheme: Godunov_PLM, Godunov_PPM or Godunov_BDS
    peleLM.chi_correction_type = DivuFirstIter  # [OPT, DEF=DivuEveryIter] When to compute divu for MAC proj divu constraint [DivuEveryIter, DivuFirstIter, NoDivu]
    peleLM.print_chi_convergence = 1            # [OPT, DEF=(peleLM.v > 1)] Boolean flag on whether to print size of chi correction on each SDC iter
    peleLM.incompressible = 0              # [OPT, DEF=0] Enable to run fully incompressible, scalar advance is bypassed
    peleLM.rho = 1.17                      # [OPT, DEF=-1] If incompressible, density value [MKS]
    peleLM.mu = 1.8e-5                     # [OPT, DEF=-1] If incompressible, dynamic visc. value [MKS]
    peleLM.gravity = 0.0 0.0 -9.81         # [OPT, DEF=Vec{0.0}] Gravity vector [MKS]
    peleLM.gradP0 = 0.0 0.0 10.0           # [OPT, DEF=Vec{0.0}] Average background pressure gradient [Pa/m]
    peleLM.do_periodic_channel = 0         # [OPT, DEF= 0] Add an automatic pressure gradient to maintain initial condition mass flow rate in periodic channel
    peleLM.periodic_channel_dir = 2        # [OPT, DEF= -1] Required if do_periodic_channel != 0. Direction to apply pressure gradient.
    peleLM.closed_chamber = 0              # [OPT] Override the automatic detection of closed chamber (based on Outflow(s))
    peleLM.floor_species = 0               # [OPT, DEF=0] Crudely enforce mass fraction positivity
    peleLM.deltaT_verbose = 0              # [OPT, DEF=0] Verbose of the deltaT iterative solve algorithm
    peleLM.deltaT_iterMax = 5              # [OPT, DEF=10] Maximum number of deltaT iterations
    peleLM.deltaT_tol = 1e-10              # [OPT, DEF=1.e-10] Tolerance of the deltaT solve
    peleLM.evaluate_vars =...              # [OPT, DEF=""] In evaluate mode, list unitTest: diffTerm, divU, instRR, transportCC

    peleLM.spark_verbose = 0               # [OPT, DEF=0] Verbosity of spark ignition
    peleLM.sparks = spark1 spark2 ...      # [OPT] List of spark names - multiple can be given
    peleLM.spark1.location = 0.0 0.0 0.0   # [OPT] Spark location (in x,y,z coordinates) [m]
    peleLM.spark1.temp = 2000.0            # [OPT] Temperature of the spark [K]
    peleLM.spark1.radius = 1e-3            # [OPT] Radius of the spark [m]
    peleLM.spark1.duration = 1e-3          # [OPT] Duration of the spark [s]
    peleLM.spark1.time = 1e-2              # [OPT] Time when spark starts [s]

    peleLM.aux_vars = a b ...              # [OPT] Names of auxiliary variables
    peleLM.a.advect = 1                    # [OPT, DEF = 1] Flag whether this variable is advected
    peleLM.a.conservative = 1              # [OPT, DEF = 1] Flag whether this variable is conservative
    peleLM.a.diffuse = 0                   # [OPT, DEF = 1] Flag whether this variable is diffused
    peleLM.a.Schmidt = 0.7                 # [OPT, DEF = -1] Schmidt number for auxiliary variable. If unspecified or negative, assumed to diffuse with unity Lewis number.

    peleLM.user_defined_ext_sources = 0    # [OPT, DEF=0] Enable user defined source terms. Requires local ProblemSpecificFunctions.cpp.
    peleLM.add_variance_sources = 1        # [OPT, DEF=1] (only relevant for Manifold EOS) include production/dissipation for variances if present in model


Transport coefficients and LES
------------------------------

::

    #-----------------------DIFFUSION AND LES MODEL CONTROL-----------------------
    peleLM.fixed_Le = 0                    # [OPT, DEF=0] Use a fixed Lewis number approximation for species diffusivities
    peleLM.fixed_Pr = 0                    # [OPT, DEF=0] Use a fixed Prandtl number approximation for thermal diffusivity
    peleLM.Prandtl = 0.7                   # [OPT, DEF=0.7] If fixed_Pr or doing LES, specifies the Prandtl number
    peleLM.Schmidt = 0.7                   # [OPT, DEF=0.7] If doing LES, specifies the Schmidt number
    peleLM.Lewis = 1.0                     # [OPT, DEF=1.0] If fixed_Le, specifies the Lewis number

    peleLM.les_model = "None"              # [OPT, DEF="None"] Model to compute turbulent viscosity: None, Smagorinsky, WALE, Sigma
    peleLM.les_cs_smag = 0.18              # [OPT, DEF=0.18] If using Smagorinsky LES model, provides model coefficient
    peleLM.les_cm_wale = 0.60              # [OPT, DEF=0.60] If using WALE LES model, provides model coefficient
    peleLM.les_cs_sigma = 1.35             # [OPT, DEF=1.35] If using Sigma LES model, provides model coefficient
    peleLM.les_v = 0                       # [OPT, DEF=0] Verbosity level for LES model
    peleLM.plot_les = 0                    # [OPT, DEF=0] If doing LES, whether to plot the turbulent viscosity
    transport.use_soret = 0                # [OPT, DEF=0] Compute diffusion including the Soret effect (note, this option is inherited from PelePhysics)

.. note::
   When using the Soret effect, boundary condition corrections are needed at isothermal boundaries,
   which are not fully supported. Currently a correction for all terms except the wbar term
   is applied at isothermal domain boundaries (this is likely sufficient), while no corrections are applied
   at isothermal embedded boundaries (so use caution for isothermal EBs with Soret diffusion active).


.. _ssec:meshMappingParameters:

Mesh Mapping
---------------------------------------

::

    #-----------------------Mesh Mapping -----------------------
    geometry.mesh_mapping      = TanhStretchMap [OPT, DEF=unused] Can be ConstantMap, ExpStretchMap, TanhStretchMap
    TanhStretchMap.beta        = 2.0 2.0   # one value per AMREX_SPACEDIM axis
    ExpStretchMap.direction    = 0 # Can be 0,1,2
    ExpStretchMap.wall         = lo # Can be lo, hi
    ExpStretchMap.beta         = 1 # Must be non-negative
    ConstantMap.scaling_factor = 2.0 2.0 2.0 # one value per AMREX_SPACEDIM axis

.. note::
   Three mesh maps are provided with `PeleLMeX.  If `mesh_mapping` is not specified, no mapping will be applied. Each map has associated
   parameters shown above.

.. note::
   **Physical vs. computational coordinates.** With a mapping active, the AMReX
   grid is the uniform computational (:math:`\xi`) grid and the physical grid is
   its image under the map. Inputs that name a *position* are physical
   coordinates and are converted internally: ``peleLM.inlet_plane_position``
   (the recycling source plane) is inverted through the map before being turned
   into an index, and its bounds check is against the physical domain extent,
   which for ``ConstantMap`` differs from ``geometry.prob_lo`` /
   ``geometry.prob_hi``. Diagnostics inherited from `PelePhysics`, notably
   ``DiagFramePlane``, are the exception: their ``center`` is located on the
   AMReX grid and is therefore a :math:`\xi` coordinate by default; set
   ``peleLM.<diag>.center_is_physical = 1`` to give a physical position
   instead (the diagnostic reads ``geometry.mesh_mapping`` and prints the
   :math:`\xi` it resolved to).

.. note::
   **Turbulent inflow.** ``turbinflow`` may be combined with ``mesh_mapping``:
   the injection face's physical cell-centre positions are handed to
   `PelePhysics` explicitly, so the turbulence file is sampled at the right
   place on a stretched grid. This requires a `PelePhysics` new enough to
   provide the coordinate-based ``TurbInflow::add_turb()`` overload; PeleLMeX
   aborts at setup if the two are combined without it, rather than silently
   sampling the file at :math:`\xi` positions.

   The turbulence *file* is uniformly spaced in *some* coordinate -- its
   header carries only a point count and a domain size per direction.
   Synthetic data and planes from a uniform-mesh precursor are uniform in
   physical position. Planes extracted from a precursor that itself ran with
   ``mesh_mapping`` are uniform in that run's :math:`\xi` coordinate instead;
   the `PelePhysics` ``TurbInflowGenerator`` tags such files with a
   ``MESHMAP`` trailer in the ``HDR`` when its input carries the precursor's
   ``geometry.mesh_mapping`` block (copy those lines verbatim), and the reader
   then inverts the file's map for every target cell and interpolates in the
   file's :math:`\xi`. Either kind of file can be injected on either kind of
   target grid; when the file's map and :math:`\xi` grid coincide with the
   target's, the file is reproduced exactly. For a mapped file
   ``turb_center`` may be omitted (the file sits where the precursor had it).
   At ``peleLM.v > 0`` PeleLMeX prints, per inflow face, which coordinate the
   file is uniform in and whether it is injected by exact index or by
   interpolation, followed by a resolution check comparing the grid's
   physical spacing range against the file's; a ratio well above one means
   the file cannot fill the scales the stretched grid resolves near its
   clustering, and well below one means the injected field is aliased.
   See ``Exec/RegTests/TurbInflow`` for the round-trip and cross-map cases.

Turbulent Forcing and Velocity Plotfile
---------------------------------------

::

    #-----------------------Turbulent Forcing-----------------------
    peleLM.do_turbulent_forcing    = 0           # [OPT, DEF=0] Use turbulent forcing (note, all turbforce. options are inherited from PelePhysics).
    turbforce.v                    = 0         # [OPT, DEF=0] Verbosity level for turbulent forcing.
    turbforce.urms                 = 1.0       # [OPT, DEF=1.0] Target urms.
    turbforce.time_offset          = 0.0       # [OPT, DEF=0] Offset of the forcing function.
    turbforce.hack_lz              = 0         # [OPT, DEF=0] Allow periodic reproduction in z.
    turbforce.force_scale_fudge    = 1.0       # [OPT, DEF=1.0] Used for fine scale tuning of the forcing function.
    turbforce.rho_incompressible   = -1.0      # [OPT, DEF=-1.0] Density if running in incompressible mode (otherwise this is not used).

    peleLM.velocity_plotfile       = plt?????  # [OPT, DEF=""] Name of a plotfile that you want to provide velocities to add to the initial conditions (must match base grid and domain size exactly).
    peleLM.velocity_plotfile_scale = 1.0       # [OPT, DEF=1.0] Scaling of the velocities provided from the velocity_plotfile

.. note::
   For ff_factor, nmodes, forcing_epsilon, spectrum_type and moderate_zero_modes, it is best to leave
   these as defaults unless you are confident on the consiquences.

Recycling-plane inflow
----------------------

`PeleLMeX` can synthesize a time-varying inflow by sampling velocities from an
interior cross-section ("source plane") and feeding the resulting fluctuations
back onto the inflow ghost cells. The injection is a zero-mean perturbation
added on top of the standard ``ext_dir`` inflow profile, i.e.

.. math::

   \mathbf{u}_{\text{inlet}}(t) = \mathbf{u}_{\text{target}}(t) +
   \bigl[\mathbf{u}_{\text{src}}(t) - \langle \mathbf{u}_{\text{src}} \rangle\bigr]

where :math:`\mathbf{u}_{\text{target}}` is the prescribed inflow profile from
``bcnormal`` and :math:`\langle \cdot \rangle` denotes a running average over
samples taken at every timestep. The mean-flow profile must still be set by
the user through the standard ``Inflow`` boundary condition; this feature only
supplies the bracketed fluctuation. The running mean is preserved in
checkpoints and restored across restarts.

Alternatively, ``peleLM.recycling_mode = full`` injects the entire sampled
velocity, :math:`\mathbf{u}_{\text{inlet}}(t) = \gamma(t)\,
\mathbf{u}_{\text{src}}(t)`, with no running mean maintained. The recycled
velocity is enforced by the boundary-fill machinery itself: whatever
``bcnormal`` writes to the velocity components on the recycling face is
ignored in this mode, so existing problem setups (including ones that add a
mean profile to the velocity) run unmodified; ``bcnormal`` still controls
the thermodynamic state. Full mode carries no restart state (the injection
buffer reseeds from the first snapshot after initialization or restart,
taken before the first boundary fill), and the ``inlet_plane_avg_window`` /
``inlet_plane_warmup_steps`` controls are ignored.

The factor :math:`\gamma(t)` above is the full-mode **mass-flux control**
(on by default). Because full mode feeds the sampled velocity straight back
to the inlet, the bulk flow rate is a closed feedback loop that carries no
memory of the intended inflow: it is only neutrally stable in an adiabatic
constant-area duct, and any net heat transfer between the inlet and the
source plane biases it. In the low-Mach setting, wall heat loss makes
:math:`\nabla \cdot \mathbf{u} < 0` in the duct, so the plane always
returns a smaller volume flux than the inlet received (mass flux, not
volume flux, is what convection conserves) and the recycled bulk flow
decays monotonically until the inlet reverses — cooled walls collapse
within a few tens of flow-through times. The controller closes the loop:
every snapshot, the sample's normal velocity component is rescaled by
:math:`\gamma = Q_0 / Q_{\text{src}}`, where both the target :math:`Q_0`
and the sampled :math:`Q_{\text{src}}` are EB-masked volumetric flow rates
through the same cross-section. :math:`Q_0` is measured once from the
``bcnormal`` inflow profile (or set explicitly via
``peleLM.inlet_plane_target_flow_rate``). The rescaling is multiplicative
and applied to all velocity components: the sampled fluctuation field is
solenoidal to low-Mach accuracy, with continuity coupling the normal and
tangential components mode by mode, so the uniform scaling preserves that
structure and the injected turbulence passes through the inlet-adjacent
projection intact (a normal-only scaling would leave an irrotational
residual for the projection to remove, distorting the tangential
fluctuations). No-slip walls, the sampled profile shape, and the sample's
relative turbulence intensity are all preserved. :math:`\gamma` is clipped
to :math:`[0.5, 2]` (with a warning), and the run aborts if the sampled
through-flow collapses below 5% of the target or reverses, since a
collapsed sample can no longer be rescaled into a meaningful inflow.

::

    #-----------------------Recycling-plane inflow-----------------------
    peleLM.use_inlet_from_plane     = 0       # [OPT, DEF=0] Master switch (0 disables)
    peleLM.inlet_plane_dir          = -1      # [REQ if active] Sampling axis: 0=x, 1=y, 2=z
    peleLM.inlet_plane_position     = 0.0     # [REQ if active] Physical coordinate of the source plane along inlet_plane_dir
    peleLM.recycling_mode           = fluctuations # [OPT, DEF=fluctuations] 'fluctuations' injects u - <u> on top of the
                                              #                 bcnormal profile; 'full' injects the sampled velocity itself
    peleLM.inlet_plane_avg_window   = -1.0    # [OPT, DEF=-1.0] Time-window (s) for the running mean (EMA);
                                              #                 <=0 falls back to a cumulative 1/N average.
                                              #                 Fluctuations mode only.
    peleLM.inlet_plane_warmup_steps = 0       # [OPT, DEF=0] Number of samples to accumulate before any
                                              #              fluctuation is injected (lets the running mean settle).
                                              #              Fluctuations mode only.
    peleLM.inlet_plane_flux_control = 1       # [OPT, DEF=1] Full mode only: rescale the injected sample to the
                                              #              target inlet flow rate every step. Disable (0) only for
                                              #              adiabatic constant-area ducts, and at your own risk.
    peleLM.inlet_plane_target_flow_rate = ""  # [OPT] Full mode only: target volumetric flow rate (m^3/s, EB-masked,
                                              #       signed along +inlet_plane_dir). Default: measured from the
                                              #       bcnormal inflow profile on first use.

The source plane is one cell thick along ``inlet_plane_dir`` and spans the full
transverse cross-section at ``inlet_plane_position``. Storage is allocated
per AMR level and rebuilt automatically on regrid; previously-accumulated
running means are carried across rebuilds via a parallel-copy. EB-covered cells
on the source plane are excluded from the running mean and produce a zero
fluctuation contribution; the inflow face must therefore be ``Inflow``-typed
on every velocity component (the implementation aborts if the components
disagree).

.. note::
   For best results, place the source plane far enough downstream of any
   geometric feature that the recycled signal is broadband rather than
   dominated by a single coherent shedding mode. A loop delay
   :math:`(x_{\text{plane}} - x_{\text{inlet}})/U_{\text{bulk}}` close to
   one flow-through time is a reasonable starting point; placing the plane
   too close to the inlet (delay :math:`\ll \tau`) produces near-laminar
   recycled signal and effectively disables the feature, while placing it
   in a region of strong coherent shedding can produce self-resonant
   oscillations in the integrated flow statistics. The
   ``Exec/RegTests/EB_BFS_Recycling`` case demonstrates the trade-offs.

Chemistry integrator
--------------------

::

    #-----------------------CHEMISTRY CONTROL----------------------
    peleLM.chem_integrator   = "ReactorCvode"   # Chemistry integrator, from PelePhysics available list
    peleLM.use_typ_vals_chem = 1                # [OPT, DEF=1] Use Typical values to scale components in the reactors
    peleLM.typical_values_reset_int = 5         # [OPT, DEF=10] Frequency at which the typical values are updated
    ode.rtol = 1.0e-6                           # [OPT, DEF=1e-10] Relative tolerance of the chem. reactor
    ode.atol = 1.0e-6                           # [OPT, DEF=1e-10] Absolute tolerance of the chem. reactor, or pre-factor of the typical values when used
    cvode.solve_type = denseAJ_direct           # [OPT, DEF=GMRES] Linear solver employed for CVODE Newton direction
    cvode.max_order  = 4                        # [OPT, DEF=2] Maximum order of the BDF method in CVODE
    cvode.max_substeps = 10000                  # [OPT, DEF=10000] Maximum number of substeps for the linear solver in CVODE

Note that the last five parameters belong to the Reactor class of PelePhysics but are specified here for completeness. In particular, CVODE is the adequate choice of integrator to tackle PeleLMeX large time step sizes. Several linear solvers are available depending on whether or not GPU are employed: on CPU, `dense_direct` is a finite-difference direct solver, `denseAJ_direct` is an analytical-jacobian direct solver (preferred choice), `sparse_direct` is an analytical-jacobian sparse direct solver based on the KLU library and `GMRES` is a matrix-free iterative solver; on GPU `GMRES` is a matrix-free iterative solver (available on all the platforms), `sparse_direct` is a batched block-sparse direct solve based on NVIDIA's cuSparse (only with CUDA), `magma_direct` is a batched block-dense direct solve based on the MAGMA library (available with CUDA and HIP. Different `cvode.solve_type` should be tried before increasing the `cvode.max_substeps`.

.. note::
   The default chemistry integrator is 'ReactorNull' which do not include the chemical source terms.

Embedded Geometry
-----------------

`PeleLMeX` geometry relies on AMReX implementation of the EB method. Simple geometrical objects
can thus be constructed using `AMReX internal parser <https://amrex-codes.github.io/amrex/docs_html/EB.html>`_.
For instance, setting up a sphere of radius 5 mm can be achieved:

::

    eb2.geom_type = sphere
    eb2.sphere_radius = 0.005
    eb2.sphere_center = 0.0 0.0 0.0
    eb2.sphere_has_fluid_inside = 0
    eb2.small_volfrac = 1.0e-4
    eb2.maxiter = 200

The `eb2.small_volfrac` controls volume fraction that are deemed too small and eliminated from the EB representation.
This operation is done iteratively and the maximum number of iteration is prescribed by `eb2.maxiter`.
For most applications, a single AMReX object is insufficient to represent the geometry. AMReX enable to combine
objects using constructive solid geometry (CSG) in order to create complex geometry. It is up to users to define
the combination of basic elements leading to their desired geometry. To switch to a user-defined EB definition, one
must set:

::

    eb2.geom_type = UserDefined

and then implement the actual geometry definition in the `EBUserDefined` member function of the `ProblemSpecificFunctions` struct in
``pelelmex_prob.H``. An example of such implementation is available in the
``Exec/Case/ChallengeProblem`` folder. Example of more generic EB problems are also found in the ``Exec/RegTest/EB_*``
folders.

In addition to the input keys presented above, a set of `PeleLMeX`-specific keys are available in order to control refinement at the EB:

::

    peleLM.refine_EB_type = Static
    peleLM.refine_EB_max_level = 1
    peleLM.refine_EB_buffer = 2.0

By default, the EB is refined to the `amr.max_level`, which can lead to undesirably high number of cells
close to the EB when the physics of interest might be elsewhere. The above lines enable to limit the
EB-level to level 1 (must be below `amr.max_level`) and a derefinement strategy is adopted to ensure
that fine-grid patches do not cross the EB boundary. The last parameter set a safety margin to increase
how far the derefinement is applied in order to account for grid-patches diagonals and proper nesting constrains.
Note that the parameter do not ensure explicitly coarse-fine/EB crossings are avoided and the code will fail when this happens.

AMReX generates the EB at the finest level (specified by `amr.max_level`) and subsequently coarsen the resulting
EB data to coarser AMR levels (as well as multigrid levels) in order to ensure consistency across levels. As a consequence,
increasing `amr.max_level` during the course of a simulation can lead in small changes to the EB, potentially uncovering
previously covered regions for which the solver cannot provide initial conditions upon restart. It is thus advised to
set `amr.max_level` to the desired maximum level you ever plan to use in the simulation, and limit the level at which
the EB is refined (with `peleLM.refine_EB_max_level`) and refinement criteria are applied (with `amr.*****.max_level`)
to a lower value while starting the simulation. A side effect of this process is that the generation of the EB can
be excessively long if `amr.max_level` is large. To speed up the EB generation, one can use the following keys:

::

    eb2.num_coarsen_opt = 3
    eb2.max_grid_size = 64

Setting `eb2.num_coarsen_opt = 3` effectively speed the EB generation by coarsening the level at which the recursive EB intersection
algorithm starts by a factor 2^3. A large value of this parameter can lead to an erroneous EB representation, missing small features,
and should thus be kept at or below 4. The EB intersection algorithm relies on chopping the domain into boxes and finding those
fully or partially covered. The size of these boxes can be controlled with `eb2.max_grid_size`, and can be adjusted to better
match the number of MPI ranks used in the simulation.


It is also possible to change the default adiabatic EB wall condition to an isothermal EB. To do so, one need to switch the following
flag:

::

    peleLM.isothermal_EB = 1

The user is now responsible for providing the wall temperature *on all the EB walls*, but adiabtic wall can still be specified.
Control over the local EB thermal boundary condition is provided through the `bcnormal_eb` and `bctype_eb` functions, also
defined in the `ProblemSpecificFunctions` struct in ``pelelmex_prob.H`` already used above to provide a user-defined EB geometry.
Examples of isothermal EBs are provided
in ``Exec/RegTest/EB_BackwardStepFlame`` and ``Exec/RegTest/EB_FlowPastCylinder`` tests.

.. note::
   Note that when using isothermal EB in combination with LES, the thermal diffusion coefficient employed to compute the EB boundary thermal flux only uses the molecular contribution.

Inflow boundaries on EB surfaces may also be specified through experimental capability by setting the following flag:

::

   peleLM.EBinflow = 1

Similarly to isothermal boundaries, the user is responsible for specifying the inflow conditions using the `bcnormal_eb` function,
but this function must not specify all state variables in the region of the inflow. Again, an example is provided in the
``Exec/RegTest/EB_BackwardStepFlame`` test case.

.. note::
   EB inflow capability is still experimental. Scalar diffusion at these boundaries is not yet supported. This capability and the user interface
   are subject to potential change in the future and bugs/unanticipated behavior are more likely with this capability for now. Potential changes
   may also impact the isothermal EB capability.

Lastly, it is possible to change the default redistribution scheme described in the :ref:`geometry with embedded boundaries section: <ssec:geoEB>`
::

    peleLM.adv_redist_type = StateRedist  # [OPT, DEF=StateRedist] Redistribution scheme for advection [StateRedist, FluxRedist, NoRedist]
    peleLM.diff_redist_type = FluxRedist  # [OPT, DEF=FluxRedist]  Redistribution scheme for diffusion [StateRedist, FluxRedist, NoRedist]

Linear solvers
--------------

Linear solvers are a key component of PeleLMeX algorithm, separate controls are dedicated to the various solver (MAC projection, nodal projection, diffusion, ...)

::

    #-------------------------LINEAR SOLVERS-----------------------
    nodal_proj.verbose = 1                    # [OPT, DEF=0] Verbose of the nodal projector
    nodal_proj.rtol = 1.0e-11                 # [OPT, DEF=1e-11] Relative tolerance of the nodal projection
    nodal_proj.atol = 1.0e-12                 # [OPT, DEF=1e-14] Absolute tolerance of the nodal projection
    nodal_proj.maxiter = 100                  # [OPT, DEF=100] Maximum number of iterations of the nodal projection
    nodal_proj.mg_max_coarsening_level = 5    # [OPT, DEF=100] Maximum number of MG levels (useful when using EB)
    nodal_proj.bottom_verbose = 1             # [OPT, DEF=0] Verbose of the bottom solve for nodal projector
    nodal_proj.bottom_rtol = 1e-3             # [OPT, DEF=1e-4] Relative tolerance of the bottom solve for nodal projection
    nodal_proj.bottom_atol = 1e-10            # [OPT, DEF=0] Absolute tolerance of the bottom solve for nodal projection
    nodal_proj.bottom_maxiter = 200           # [OPT, DEF=100] Maximum number of iterations of the bottom solve for nodal projection

    mac_proj.verbose = 1                      # [OPT, DEF=0] Verbose of the MAC projector
    mac_proj.rtol = 1.0e-11                   # [OPT, DEF=1e-11] Relative tolerance of the MAC projection
    mac_proj.atol = 1.0e-12                   # [OPT, DEF=1e-14] Absolute tolerance of the MAC projection
    mac_proj.maxiter = 200                    # [OPT, DEF=200] Maximum number of iterations of the MAC projection
    mac_proj.mg_max_coarsening_level = 5      # [OPT, DEF=100] Maximum number of MG levels (useful when using EB)
    mac_proj.bottom_verbose = 1               # [OPT, DEF=0] Verbose of the bottom solve for MAC projector
    mac_proj.bottom_rtol = 1e-3               # [OPT, DEF=1e-4] Relative tolerance of the bottom solve for MAC projection
    mac_proj.bottom_atol = 1e-10              # [OPT, DEF=0] Absolute tolerance of the bottom solve for MAC projection
    mac_proj.bottom_maxiter = 200             # [OPT, DEF=100] Maximum number of iterations of the bottom solve for MAC projection

    diffusion.verbose = 1                     # [OPT, DEF=0] Verbose of the scalar diffusion solve
    diffusion.rtol = 1.0e-11                  # [OPT, DEF=1e-11] Relative tolerance of the scalar diffusion solve
    diffusion.atol = 1.0e-12                  # [OPT, DEF=1e-14] Absolute tolerance of the scalar diffusion solve
    diffusion.max_iter = 100                  # [OPT, DEF=100] Maximum number of iterations of the scalar diffusion solve

    tensor_diffusion.verbose = 1              # [OPT, DEF=0] Verbose of the velocity tensor diffusion solve
    tensor_diffusion.rtol = 1.0e-11           # [OPT, DEF=1e-11] Relative tolerance of the velocity tensor diffusion solve
    tensor_diffusion.atol = 1.0e-12           # [OPT, DEF=1e-14] Absolute tolerance of the velocity tensor diffusion solve
    tensor_diffusion.max_iter = 100           # [OPT, DEF=100] Maximum number of iterations of the velocity tensor diffusion solve

Hypre support
^^^^^^^^^^^^^

Through AMReX, PeleLMeX provides interfaces to the `Hypre <https://github.com/hypre-space/hypre>`_
preconditioners and solvers. These can be called as bottom solvers for the MLMG linear
solvers, for both cell-centered and node-based problems.  The Hypre solvers are particularly
useful if the geometry includes thin elements (such as tube or plate) or narrow channels,
as coarsening of the geometry is rapidly limited by the occurrence of multi-cut cells
(not supported by AMReX) and the linear solvers are no longer able to robustly
tackle projections and implicit diffusion solves.

To build Hypre, follow the steps outlined in the
`AMReX documentation <https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html#external-solvers>`_.

Next, in the ``GNUmakefile``, enable Hypre and define the path to the Hypre directory:

::

    USE_HYPRE = TRUE
    HYPRE_HOME = /path_to_hypre_dir/hypre/src/hypre


Select input file controls are provided below for the ``mac_proj`` bottom solver, which
can be applied similarly for the ``nodal_proj``. Additional information on the Hypre
solvers and parameters can be found in the `Hypre documentation <https://hypre.readthedocs.io/en/latest/>`_.

::

    #----------------------HYPRE LINEAR SOLVERS--------------------
    mac_proj.bottom_solver = "hypre"
    mac_proj.hypre_namespace = mac_proj.hypre
    mac_proj.hypre.verbose = 1
    mac_proj.hypre.hypre_solver = GMRES
    mac_proj.hypre.hypre_preconditioner = BoomerAMG
    mac_proj.hypre.bamg_verbose = 0
    mac_proj.hypre.bamg_coarsen_type = 9
    mac_proj.hypre.bamg_interp_type = 4
    mac_proj.hypre.bamg_relax_type = 7


Active control
--------------

`PeleLMeX` includes an active control mechanism to enable statistically steady simulations of flames
maintaining the flame at a fixed position in the domain. An example of this feature is provided in
the triple flame tutorial :doc:`Tutorials_TripleFlame`.

.. note::
   To enable active control, a ``FlowControllerData FCData`` object must be added to the problem ``ProbParm``!

During the course of the simulation, the FlowControllerData is updated based on the flame position to allow
the user to set the inflow velocity. The following options are available when using active control: ::

    #---------------------- AC CONTROL -------------------------------
    active_control.on = 1                     # [OPT, DEF=0] Use AC ?
    active_control.use_temp = 1               # [OPT, DEF=1] Default in fuel mass, rather use iso-T position ?
    active_control.temperature = 1400.0       # [OPT, DEF=-1] Value of iso-T ?
    active_control.tau = 5.0e-4               # [OPT, DEF=0.0] Control tau (should ~ 10 dt)
    active_control.height = 0.01              # [OPT, DEF=0.0] Where is the flame held ?
    active_control.v = 1                      # [OPT, DEF=0] verbose
    active_control.method = 1                 # [OPT, DEF=2] Controller: 1 - Linear, 2 - Quadratic, 3 - Weighted quadratic
    active_control.velMax = 2.0               # [OPT, DEF=-1.0] limit inlet velocity, only used when positive
    active_control.changeMax = 0.1            # [OPT, DEF=1.0] limit inlet velocity changes (absolute m/s)
    active_control.flow_dir  = 1              # [OPT, DEF=AMREX_SPACEDIM-1] flame main direction
    active_control.AC_history  = AChist       # [OPT, DEF=AC_history] Control history file, read upon restart
    active_control.npoints_average = 5        # [OPT, DEF=3] Number of previous steps using to estimate new velocity
    active_control.pseudo_gravity = 1         # [OPT, DEF=0] add density proportional force to compensate for the acceleration
                                              #           of the gas due to inlet velocity changes

Run-time diagnostics
--------------------

`PeleLMeX` provides a few diagnostics to check you simulations while it is running as well as adding basic analysis ingredients.

It is often useful to have an estimate of integrated quantities (kinetic energy, heat release rate, ,..), state extremas
or other overall balance information to get a sense of the status and sanity of the simulation. To this end, it is possible
to activate `temporal` diagnostics performing these reductions at given intervals:

::

    #-------------------------TEMPORALS---------------------------
    peleLM.do_temporals = 1                     # [OPT, DEF=0] Activate temporal diagnostics
    peleLM.temporal_dir = temporals             # [OPT, DEF="temporals"] Directory to store temporal diagnostics
    peleLM.temporal_int = 10                    # [OPT, DEF=5] Temporal freq.
    peleLM.do_extremas = 1                      # [OPT, DEF=0] Trigger extremas, if temporals activated
    peleLM.do_mass_balance = 1                  # [OPT, DEF=0] Compute mass balance, if temporals activated
    peleLM.do_energy_balance = 1                # [OPT, DEF=0] Compute enthalpy (rho*h) balance, if temporals activated
    peleLM.do_species_balance = 1               # [OPT, DEF=0] Compute species mass balance, if temporals activated
    peleLM.do_patch_mfr=1                       # [OPT, DEF=0] Activate patch based species flux diagbostics
    peleLM.bpatch.patchnames= <patch_name1 patch_name2 ..> # List of patchnames

    bpatch.patch_name1.patchtype=full-boundary             # patchtype one of "full-boundary", "circle, "rectangle", "circle-annular" or "rectangle-annular"
    bpatch.patch_name1.boundary_direction=2                # patch normal direction
    bpatch.patch_name1.boundary_lo_or_hi=0                 # patch in low or high side of boundary
    bpatch.patch_name1.species= {Air:O2,N2} O2 N2                      # list of species names

    bpatch.patch_name2.patchtype=circle                    # patchtype one of "full-boundary", "circle, "rectangle", "circle-annular" or "rectangle-annular"
    bpatch.patch_name2.boundary_direction=2                # patch normal direction
    bpatch.patch_name2.boundary_lo_or_hi=0                 # patch in low or high side of boundary
    bpatch.patch_name2.patch_circle_radius=0.1             # radius of the patch
    bpatch.patch_name2.patch_circle_center=0.0 0.0 0.0     # coordinates of patch center
    bpatch.patch_name2.species= O2 N2                      # list of species names

    bpatch.patch_name3.patchtype=rectangle
    bpatch.patch_name3.boundary_direction=2                # patch normal direction
    bpatch.patch_name3.boundary_lo_or_hi=0                 # patch in low or high side of boundary
    bpatch.patch_name3.patch_rectangle_lo=0.0 0.0 0.0      # coordinates of low corner of rectangle
    bpatch.patch_name3.patch_rectangle_hi=1.0 1.0 1.0      # coordinates of high corner of rectangle
    bpatch.patch_name3.species= O2 N2                      # list of species names

    bpatch.patch_name4.patchtype=circle-annular
    bpatch.patch_name4.boundary_direction=2                # patch normal direction
    bpatch.patch_name4.boundary_lo_or_hi=0                 # patch in low or high side of boundary
    bpatch.patch_name4.patch_circ_ann_center= 0.0 0.0 0.0  # center of annular circle
    bpatch.patch_name4.patch_circ_ann_inner_radius=0.1     # coordinates of patch center
    bpatch.patch_name4.patch_circ_ann_outer_radius=0.2     # coordinates of patch center
    bpatch.patch_name4.species= O2 N2                      # list of species names

    bpatch.patch_name5.patchtype=rectangle-annular
    bpatch.patch_name5.boundary_direction=2                     # patch normal direction
    bpatch.patch_name5.boundary_lo_or_hi=0                      # patch in low or high side of boundary
    bpatch.patch_name5.patch_rect_ann_outer_lo = -1.0 -1.0 -1.0 # coordinates of low corner of outer rectangle
    bpatch.patch_name5.patch_rect_ann_outer_hi =  1.0  1.0  1.0 # coordinates of high corner of outer rectangle
    bpatch.patch_name5.patch_rect_ann_inner_lo = -0.5 -0.5 -0.5 # coordinates of low corner of inner rectangle
    bpatch.patch_name5.patch_rect_ann_inner_hi =  0.5  0.5  0.5 # coordinates of high corner of inner rectangle
    bpatch.patch_name5.species= O2 N2                           # list of species names

The `do_temporal` flag will trigger the creation of a `temporals` folder in your run directory and the following entries
will be appended to an ASCII `temporals/tempState` file: step, time, dt, kin. energy integral, enstrophy integral, mean pressure
, fuel consumption rate integral, heat release rate integral. Please note that for simulations using finite-rate chemistry, the 
heat-release rate integral is computed from the species reaction rates and enthalpies (:math:`-\sum_n h_n \dot\omega_n`). For the 
manifold EOS, it is instead obtained by integrating the tabulated `HRR` variable looked up from the manifold table, which therefore 
requires the table to include an `HRR` column (volumetric heat release rate, in the table's units). If the table has no `HRR` column 
the reported integral is zero. Additionally, if the `do_temporal` flag is activated, one can turn on state extremas 
(stored in `temporals/tempExtremas` as min/max for each state entry), mass balance (stored in `temporals/tempMass`) computing the total mass, dMdt and advective mass fluxes across the domain boundaries as well as the error in
the balance (dMdt - sum of fluxes), energy balance (stored in `temporals/tempEnergy`) computing the total enthalpy :math:`\int \rho h`,
its time derivative, the net advective \& diffusive (Fourier and differential diffusion) enthalpy fluxes across the domain boundaries
and isothermal EB surfaces, and the error in the balance (d(rho h)/dt - sum of fluxes), and species balance (stored in `temporals/tempSpec`) computing each species total mass, dM_Ydt,
advective \& diffusive fluxes across the domain boundaries, consumption rate integral and the error (dMdt - sum of fluxes - reaction).
Users can also monitor species advective fluxes through specific regions of the domain boundaries (called as boundary patches).
Patches can be defined on the low or high sides of non-embedded boundaries through the use of pre-defined shapes such as `circle`,
`rectangle`,`circle-annular`, `rectangle-annular` and `full-boundary`. The zero AMR level, advective fluxes of each of the user-specified species will be
reported in the ASCII `temppatchmfr` file in the temporals folder. User can also group species to define a mixture in the species list, in the format {<mixture-name>:<species1>,<species2>,..,<speciesN>}. For example, to get the total air mass flow rate,
the user can define {Air:O2,N2} in the species list. The temporal function would then print total species fluxes of O2 and N2 through the specified patch.

Combustion diagnostics often involve the use of a mixture fraction and/or a progress variable, both of which can be defined
at run time and added to the derived variables included in the plotfile. If `mixture_fraction` or `progress_variable` is
added to the `amr.derive_plot_vars` list, one need to provide input for defining those. The mixture fraction is based on
Bilger's element definition and one needs to provide the composition of the 'fuel' and 'oxidizer' tanks using a Cantera-like
format (<species>:<value>) which assumes unspecified species at zero, or a list of floats, in which case all the species must
be specified in the order they appear in the mechanism file.
The progress variable definition in based on a linear combination of the species mass fractions and temperature, and can be
specified in a manner similar to the mixture fraction, providing a list of weights and the prescription of a 'cold' and 'hot'
state:

::

    # ------------------- INPUTS DERIVED DIAGS ------------------
    peleLM.fuel_name = CH4
    peleLM.mixtureFraction.format = Cantera
    peleLM.mixtureFraction.type   = mass
    peleLM.mixtureFraction.oxidTank = O2:0.233 N2:0.767
    peleLM.mixtureFraction.fuelTank = H2:0.5 CH4:0.5
    peleLM.progressVariable.format = Cantera
    peleLM.progressVariable.weights = CO:1.0 CO2:1.0
    peleLM.progressVariable.coldState = CO:0.0 CO2:0.0
    peleLM.progressVariable.hotState = CO:0.000002 CO2:0.0666


Analysing the data a-posteriori can become extremely cumbersome when dealing with extreme datasets.
PeleLMeX offers a set of diagnostics available at runtime (supported through PelePhysics) and more are under development.
Currently, the list of diagnostic contains:

* ``DiagFramePlane`` : extract a plane aligned in the 'x','y' or 'z' direction across the AMR hierarchy, writing
  a 2D plotfile compatible with Amrvis, Paraview or yt. Only available for 3D simulations.
* ``DiagPDF`` : extract the PDF of a given variable and write it to an ASCII file.
* ``DiagConditional`` : extract statistics (average and standard deviation, integral or sum) of a
  set of variables conditioned on the value of given variable and write it to an ASCII file.

When using `DiagPDF` or `DiagConditional`, it is possible to narrow down the diagnostic to a region of interest
by specifying a set of filters, defining a range of interest for a variable. Note also the for these two diagnostics,
fine-covered regions are masked. An arbitrary number of these diagnostics may be specified in a list by setting
``peleLM.diagnostics`` in the input file and then specifying the diagnostic type and relevant inputs for each
diagnostic listed. See the
`PelePhysics Diagnostics documentation <https://pele-suite.github.io/PelePhysics/Utility.html#diagnostics>`_ for full
details on the options that must be specified for each diagnostic type.

::

    #--------------------------DIAGNOSTICS------------------------

    peleLM.diagnostics = xnormP condT pdfTest

    peleLM.xnormP.type = DiagFramePlane                             # Diagnostic type
    ...

Run-time control
--------------------

Following some of AMReX's AmrLevel class implementation, PeleLMeX provides a couple of triggers to interact with the code while
it is running. This can be done by adding an empty file to the folder where the simulation is currently running using for
example:

::

    touch plt_and_continue

The list of available triggers is:

.. list-table:: PeleLMeX run-time triggers
    :widths: 50 100
    :header-rows: 1

    * - File
      - Function
    * - plt_and_continue
      - Write a pltfile to disk and pursue the simulation
    * - chk_and_continue
      - Write a chkfile to disk and pursue the simulation
    * - dump_and_stop
      - Write both pltfile and chkfile to disk and stop the simulation

By default, the code checks if these files exist every 10 time steps, but the user can either increase or decrease the
frequency using:

::

    amr.message_int      = 20                # [OPT, DEF=10] Frequency for checking the presence of trigger files

Selection of Helpful GNUmakefile Flags
--------------------------------------

When compiling PeleLMeX, several compilation flags can be selected to help with debugging, improving performance, etc. Below is a selection of some of the most common flags users may wish to add to their `GNUmakefile`:
::

    # Models
    USE_PLASMA = TRUE                    # [OPT, DEF=FALSE] Enable plasma model
    USE_PARTICLES = TRUE                 # [OPT, DEF=FALSE] Enable Lagrangian particle spray model
    USE_SOOT = TRUE                      # [OPT, DEF=FALSE] Enable soot model
    USE_RADIATION = TRUE                 # [OPT, DEF=FALSE] Enable radiation model
    PELELM_NUM_ODE = INT                 # [OPT, DEF=0] Number of user-defined ODEs to solve

    # If USE_PARTICLES = TRUE
    SPRAY_FUEL_NUM = INT                 # [REQD] Number of fuel species in spray model
    SPRAY_GCM = TRUE                     # [OPT, DEF=FALSE] Enable group contribution method for liquid property prediction in spray model

    # If USE_SOOT = TRUE
    NUM_SOOT_MOMENTS = 3                 # [REQD] Number of soot moments (must be 3 or 6)

    # Equation of State
    Eos_Model = Fuego                    # [OPT, DEF=Fuego] EOS model: GammaLaw, Fuego, Soave-Redlich-Kwong, Manifold, or Null

    # If Eos_Model = Manifold
    Manifold_Dim = INT                   # [REQD] Manifold dimension
    Manifold_Type = Table                # [REQD] Manifold type: Table or Network (if unspecified, will support both but at a performance cost)

    # Transport Model
    Transport_Model = Simple             # [OPT, DEF=Simple] Transport model: Simple, EGLib, Constant, Sutherland, or Manifold

    # Chemistry
    Chemistry_Model = drm19              # [REQD] Chemistry mechanism name (from PelePhysics/Mechanisms) or path if USE_CUSTOM_CHEMISTRY=TRUE
    USE_CUSTOM_CHEMISTRY = TRUE          # [OPT, DEF=FALSE] Chemistry_Model is interpreted as a path to an arbitrary location

    # Embedded Boundaries
    USE_EB = TRUE                        # [OPT, DEF=FALSE] Enable embedded boundary support

    # HPC System Specifics and GPU/Accelerator Support
    COMP = gnu                           # [OPT, DEF=gnu] Compiler choice: gnu, intel, cray, etc.
    USE_MPI = TRUE                       # [OPT, DEF=TRUE] Compile with MPI support
    USE_OMP = TRUE                       # [OPT, DEF=FALSE] Compile with OpenMP support
    USE_CUDA = TRUE                      # [OPT, DEF=FALSE] Compile for NVIDIA GPUs
    USE_HIP = TRUE                       # [OPT, DEF=FALSE] Compile for AMD GPUs
    USE_SYCL = TRUE                      # [OPT, DEF=FALSE] Compile for Intel GPUs

    # Case Specific files
    CEXE_sources += <file_name>          # [OPT] Add additional source files to the build

    # Other Quality of Life Flags
    USERSuffix = String                  # [OPT, DEF=""] Suffix to append to executable name, useful when multiple builds are needed
    TINY_PROFILE = TRUE                  # [OPT, DEF=FALSE] Enable lightweight profiling with AMReX's TinyProfiler
    DEBUG = TRUE                         # [OPT, DEF=FALSE] Compile with debug flags (disables optimizations)
