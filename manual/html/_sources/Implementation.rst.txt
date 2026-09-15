.. highlight:: rst

.. _sec:code:

Source code
===========

The following provides an overview of *PeleLMeX* source code, basic information on the data structure and is
useful for any user intending to use and/or do some development in the code.

Overview of source code
-----------------------

*PeleLMeX* is based upon AMReX's `AmrCore <https://amrex-codes.github.io/amrex/docs_html/AmrCore.html>`_ from which it inherits
an AMR hierarchy data structure and basic regridding functionalities. The code is entirely written in C++, with low level
compute-intensive kernels implemented as lambda functions to seamlessly run on CPU and various GPU backends through AMReX
high performance portatbility abstraction.

The core of the algorithm is implementation in the ``advance()`` function which acts on all the levels concurrently.
Projection operators and advection scheme functions are imported the `AMReX-Hydro library <https://amrex-fluids.github.io/amrex-hydro/docs_html/index.html>`_
while the core of the thermo-chemistry functionalities comes from `PelePhysics <https://amrex-combustion.github.io/PelePhysics/>`_ .
Users are responsible for providing initial and boundary conditions in the local subfolder implementing their case, i.e. it is
not possible to compile and run *PeleLMeX* without actually writing a few lines of codes. However, numerous example are provided
in ``Exec/RegTests`` from which new users can pull for their new case.

The source code contains a few dozen files, organized around the pieces of the algorithm and major functionalities:

* ``PeleLMeX_Evolve``: top level time advance loop, with IO/exit controls
* ``PeleLMeX_Setup``: setting up the simulation parameters, parsing the input file
* ``PeleLMeX_Init``: generating the initial solution from scratch or checkpoint file, performing initial projections/iteration(s)
* ``PeleLMeX_Advance``: top level implementation of the time step algorithm
* ``PeleLMeX_Projection``: implement the various flavors of the nodal projection
* ``PeleLMeX_Umac``: implement the construction and projection of MAC-velocities
* ``PeleLMeX_Advection``: functions to compute the explicit advection terms
* ``PeleLMeX_Diffusion``: functions to compute the diffusion terms, using the operators defined in ``PeleLMex_DiffusionOp``
* ``PeleLMeX_Reaction``: function using *PelePhysics* reactors to integrate the chemistry and linearized advection/diffusion
* ``PeleLMeX_Plot``: implementation of plotfile and checkpoint file IOs
* ``PeleLMeX_BC``: functions filling the ghost cells (at fine/fine, coarse/fine and domain boundaries)
* ``PeleLMeX_Regrid``: creating new AMR level or remaking modified AMR level during adaptive refinement
* ``PeleLMeX_Tagging``: mark cells for refinement
* ``PeleLMeX__K.H``: low-level kernel functions

Data structure and containers
-----------------------------

AMReX 101
^^^^^^^^^

The basic AMReX`s data structure is the `MultiFab <https://amrex-codes.github.io/amrex/docs_html/Basics.html#fabarray-multifab-and-imultifab>`_
(historically, multi Fortran Array Box (FAB)).
Within the block-structured AMR approach of AMReX, the domain is decomposed into non-overlapping rectangular `boxes`,
which can be assembled into a `boxArray`. Each AMR level has a `boxArray` providing the list of `boxes` of that level.
The `boxes` are distributed across the MPI ranks, the mapping of which is described by a `DistributionMap`. Given a
`boxArray` and a `DistributionMap`, one can define an actual data container (`boxes` are only lightweight descriptor
of the geometrical rectangular object, containing bounds and centering information only), where each rank will
allocate a FAB for the boxes it owns in the `boxArray`, resulting in a collection of FABs or a MultiFab, distributed
across the MPI ranks.

To access the data in a MultiFab, one uses a `MFIter <https://amrex-codes.github.io/amrex/docs_html/Basics.html#mfiter-and-tiling>`_
(or MultiFab iterator), which provides each MPI rank access to the FABs it owns within the MultiFab. Actual access to the data in
memory is then provided by the lightweight `Array4` structure and it is strongly advised to rely on AMReX
`ParallelFor <https://amrex-codes.github.io/amrex/docs_html/Basics.html#parallelfor>`_ function template to loop through the logical `i,j,k` indexes.
For example, to set the velocity data stored in a MultiFab called `NewState` with data from an `OldState` and an increment
from a third MultiFab `advTerm`: ::

    for (MFIter mfi(State,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        Box const& bx = mfi.tilebox();
        auto const& velo_old = OldState.const_array(mfi,VELOCITY_INDEX);
        auto const& incr = advTerm.const_array(mfi);
        auto const& velo_new = NewState.array(mfi,VELOCITY_INDEX);
        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            velo_new(i,j,k) = velo_old(i,j,k) + incr(i,j,k);
        });
    }

.. note::
   For the example above to function, all three MultiFabs must have the same `boxArray` and `DistributionMap`

Users are strongly encouraged to review of the content of `AMReX documentation <https://amrex-codes.github.io/amrex/docs_html/Basics.html>`_
to get more familiar with AMReX data structures and environment.

*PeleLMeX* state and advance containers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The state vector of *PeleLMeX* contains the 2 or 3 components of velocity, the mixture density, species density (rhoYs),
rhoH, temperature and the thermodynamic pressure. The state components are stored in a cell-centered MultiFab with
`NVAR` components. Additionally, the perturbational pressure stored at the nodes is contained in a separate MultiFab.
Together with the cell-centered pressure gradient, the cell-centered divergence constraint and cell-centered
transport properties, these MultiFabs are assembled into a `LevelData` struct.

Each level in the AMR hierarchy have two versions of the `LevelData` at any point during the simulation: one
for the old state and one for the new state. The developer can get a pointer to the `LevelData` struct by
calling : ::

    auto ldata_p = getLevelDataPtr(lev,AmrOldTime);

with either `AmrOldTime` or `AmrNewTime` on level `lev`. Additionally, calling this function with
`AmrHalfTime` with return a `LevelData` struct whose `state` is a linearly interpolated between the old and new
states (but the other MultiFab in `LevelData` are empty !).
It is also often useful to have access to a vector of a state component across the entire AMR hierarchy. To do so, *PeleLMeX*
provides a set of functions returning a vector of MultiFab `std::unique_ptr` aliased into the `LevelData`
MultiFab on each level: ::

    getStateVect(time);         # Return the entire state (ncomp: NVAR)
    getVelocityVect(time);      # Return the velocity only (ncomp: AMREX_SPACEDIM)
    getDensityVect(time);       # Return the mixture density (ncomp: 1)
    getSpeciesVect(time);       # Return the species density (ncomp: NUM_SPECIES)
    getRhoHVect(time);          # Return rhoH (ncomp: 1)
    getTempVect(time);          # Return temperature (ncomp: 1)
    getDivUVect(time);          # Return divergence constraint (ncomp: 1)
    getDiffusivityVect(time);   # Return diffusivity (ncomp: NUM_SPECIES+2)
    getViscosityVect(time);     # Return viscosity (ncomp: 1)

where ``time`` can either be `AmrOldTime` or `AmrNewTime`.
Also available at any point during the simulation is the `LevelDataReact` which contains the species
chemical source terms. A single version of the container is available on each level and can be accessed
using: ::

    auto ldataR_p = getLevelDataReactPtr(lev);

Within the time-advance function, the *PeleLMeX* algorithm calls for the computation of the advection,
diffusion and reaction source terms iteratively using SDC. At each step, the results of other steps
can be used as part of the numerical scheme (e.g. the explicit advection with a Godunov scheme uses
the diffusion term). These temporary variables, only useful in the scope of the advance function, are
assembled into two structs: ``AdvanceDiffData`` and ``AdvanceAdvData``. The former contains three
MultiFabs for the separate diffusion term evaluations described in :numref:`LMeX_Algo`: :math:`D^n`,
:math:`D^{n+1,k}` and :math:`D^{n+1,k+1}`, as well as additional containers for the :math:`\overline{W}`
and Soret contributions. The later encapsulate the face-centered MAC velocities :math:`U_{ADV}`, the
advection term :math:`A_{n+1/2,(k+1)}`, the pressure correction :math:`\chi` and a forcing container
used in the RHS of advection/diffusion/reaction solves. In contrast with the `LevelData`, these two containers
are freed at the end of the advance function, and are passed around in the functions called in `advance()`.

Parallelism
-----------

*PeleLMeX* inherits the MPI+X approach from the AMReX library, where X can be any of OpenMP on many-cores machines,
and CUDA, HIP or SYCL for heterogeneous architectures.
The reader is referred to `AMReX GPU documentation <https://amrex-codes.github.io/amrex/docs_html/GPU.html>`_ for more details on
the thread parallelism.

As mentioned above, the top-level spatial decomposition arises from AMReX's block-structured approach. On each level, non-overlapping
`boxes` are assembled into `boxArray` and distributed across MPI rank with `DistributionMap` (or `DMap`).
It is in our best interest to ensure that all the MultiFab in the code use the same `boxArray` and `DMap`,
such that operation using `MFIter` can be performed and data copy across MPI ranks is minimized.
However, it is also important to maintain a good load balancing, i.e. ensure that each MPI rank has the same amount
of work, to avoid wasting computational resource. Reactive flow simulation are challenging, because the chemistry
integration is very spatially heterogeneous, with stiff ODE integration required within the flame front and non-stiff
integration of the linearized advection/diffusion required in the cold gases or burnt mixture. Additionally, because
a non-subcycling approach is used in *PeleLMeX*, the chemistry doesn't have to be integrated in fine-covered region.
Two `boxArray` and associated `DMap` are thus available in *PeleLMeX*:

1. The first one is inherited from `AmrCore` and is available as ``grid[lev]`` (`boxArray`) and ``dmap[lev]`` (`DMap`) throughout the code. Most
   of *PeleLMeX* MultiFabs use these two, and the `boxes` sizes are dictated by the `amr.max_grid_size` and `amr.blocking_factor` from the input
   file. These are employed for all the operations in the code except the chemistry integration. The default load balancing approach is to use
   space curve filling (SCF) with each box weighted by the number of cells in each box. Advanced users can try alternate approach using the
   keys listed in :doc:`LMeXControls`.
2. A second one is created, masking fine-covered regions and updated during regrid operations. It is used to perform the chemistry integration,
   and because this is a purely local integration (in contrast with implicit diffusion solve for instance, which require communications
   to solve the linear problem using GMG), a Knapsack load balancing approach is used by default, where the weight of each box is based
   on the total number of chemistry RHS calls in the box. The size of the `boxes` in the chemistry `boxArray` (accessible with ``m_baChem[lev]``)
   is controlled by the `peleLM.max_grid_size_chem` in the input file. Once again, advanced users can try alternate approaches to load
   balancing the chemistry `DMap` using the keys described in :doc:`LMeXControls`.

After each regrid operation, even if the grids did not actually change, *PeleLMeX* will try to find a better load balancing for the
`AmrCore` `DMap`. Because changing the load balancing requires copying data across MPI ranks, we only want to change the `DMap`
only if a significantly better new `DMap` can be obtained, with the threshold for a better `DMap` defined based on the value of
`peleLM.load_balancing_efficiency_threshold`.

Debugging
---------

The first step to debug any addition or undefined behavior of *PeleLMeX* is to turn the ``DEBUG`` flag ``ON`` in the
GNUmakefile and activate AMReX`s floating point exception traps in the input file: ::

    amrex.fpe_trap_invalid = 1
    amrex.fpe_trap_zero = 1
    amrex.fpe_trap_overflow = 1

This will slow down the code considerably, but will enable bound checks on all AMReX low-level data structure,
catch floating point errors (using nans, dividing by zero, ...) and any ``AMREX_ASSERT`` statement added to the
code base. It is also often useful to visualize data in order to understand the erroneous results the solver can
return. Developers can write to disk a single MultiFab using AMReX `VisMF`: ::

    VisMF::Write(myMF,"VisMyMF");

and can be visualized with `Amrvis` using `amrvis -mf`. Alternatively, visualizing the entire AMR hierarchy is also
useful. *PeleLMeX* provides a simple function to write a vector of MultiFab: ::

    WriteDebugPlotFile(GetVecOfConstPtrs(getTempVect()),"TempDebug");

which can be opened with `Amrvis` or any other visualization software. This last function will function providing that
the MultiFabs in the vector all have the same number of components.
Finally, another way of checking individual pieces of the algorithm is to use *PeleLMeX* evaluate mode ``peleLM.run_mode=evaluate``
and specify a list of fields with ``peleLM.evaluate_vars`` as described in :doc:`LMeXControls`. Note that not all of the
algorithm is available in this mode yet.
