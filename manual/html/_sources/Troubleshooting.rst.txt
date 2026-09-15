Troubleshooting
===============

This section is intended to gather information on common failure mode of PeleLMeX.
Additional information can be found in GitHub issues of `PeleLM <https://github.com/AMReX-Combustion/PeleLM/issues>`_ and `PeleLMeX <https://github.com/AMReX-Combustion/PeleLMeX/issues>`_


Linear solver failure
---------------------

The PeleLMeX algorithm involves multiple linear solves to handle projections
and implicit diffusion. In the event of the solver is enable to solve the
problem, the code will abort with the following message:

```
amrex::Abort::0::MLMG failed !!!
```

or

```
amrex::Abort::0::MLMG failing so lets stop here !!!
```

appearing multiple times when using more than one MPI rank. The first thing to do
is to identify which linear solve is failing and how. To do so, one needs to increase
PeleLMeX, as well as the projection and diffusion solve's verbosity (see the `Control <https://amrex-combustion.github.io/PeleLMeX/LMeXControls.html>`_
section for more details on LMeX controls):

::

    peleLM.v = 4

which will increase the default verbosity for all linear solvers to 2 (more generally to peleLM.v - 2),
although the solver verbosity may also be controlled for the solvers individually:

::
    nodal_proj.verbose = 2
    mac_proj.verbose = 2
    diffusion.verbose = 2
    tensor_diffusion.verbose = 2

Note that we focused on the projection solves here because they are generally more
prone to failure than the diffusion ones. You can then restart the simulation
again and identify if the code is failing in the nodal projection, either during the
initial projection (following *Initial velocity projection*) or during the time step
one (following *- oneSDC()::ScalarReaction()  -->*), or in the MAC-projection (right after
*SDC iter [1]*). Then, the linear solver verbose is useful to understand how the solver
fails. If the solver hangs around a small value following an initial reduction of the
residual:

::

    MLMG: # of AMR levels: 1
      # of MG levels on the coarsest AMR level: 9
    MLMG: Initial rhs               = 2666.243975
    MLMG: Initial residual (resid0) = 2666.243975
    MLMG: Iteration   1 Fine resid/bnorm = 0.03858916872
    MLMG: Iteration   2 Fine resid/bnorm = 0.001142880258
    MLMG: Iteration   3 Fine resid/bnorm = 3.300053779e-04
    MLMG: Iteration   4 Fine resid/bnorm = 9.433906375e-06
    MLMG: Iteration   5 Fine resid/bnorm = 2.665697369e-07
    MLMG: Iteration   6 Fine resid/bnorm = 7.40910596e-09
    MLMG: Iteration   7 Fine resid/bnorm = 2.071981144e-10
    MLMG: Iteration   8 Fine resid/bnorm = 2.66772528e-11
    MLMG: Iteration   9 Fine resid/bnorm = 2.568558082e-11
    MLMG: Iteration  10 Fine resid/bnorm = 2.713587827e-11
    MLMG: Iteration  11 Fine resid/bnorm = 2.490776046e-11
    MLMG: Iteration  12 Fine resid/bnorm = 2.41198728e-11
    MLMG: Iteration  13 Fine resid/bnorm = 2.527429436e-11
    MLMG: Iteration  14 Fine resid/bnorm = 2.431036667e-11
    MLMG: Iteration  15 Fine resid/bnorm = 2.479456555e-11
    MLMG: Iteration  16 Fine resid/bnorm = 2.28960372e-11
    MLMG: Iteration  17 Fine resid/bnorm = 2.541484652e-11
    MLMG: Iteration  18 Fine resid/bnorm = 2.522691579e-11
    MLMG: Iteration  19 Fine resid/bnorm = 2.508988366e-11
    ...

it generally means that the required solver tolerance is too small for the problem. The
default relative tolerances of all solvers in PeleLMeX is `1e-11`, but increasing the
resolution, using a small `amr.blocking_factor` (<16) or large flow divergence across
coarse-fine interfaces can lead to the example above. In this case, one can increase the
tolerance of the faulty solver using one of:

::

    nodal_proj.rtol = 5e-11
    mac_proj.rtol   = 5e-11
    diffusion.rtol  = 5e-11

It is sometimes necessary to increase the tolerance up `5e-10`. If you need to go higher
than this ballpark value, it probably indicates that something is wrong in the problem
setup and one should take a closer look at the solution to understand the problem.
Alternatively, the solver can fail as follows:

::

    Nodal Projection:
    >> Before projection:
    * On lev 0 max(abs(rhs)) = 32972.4697

    MLMG: # of AMR levels: 1
        # of MG levels on the coarsest AMR level: 5
    MLMG: Initial rhs               = 29416.37315
    MLMG: Initial residual (resid0) = 29416.37315
    MLMG: Iteration   1 Fine resid/bnorm = 0.003754141785
    MLMG: Iteration   2 Fine resid/bnorm = 0.104435142
    MLMG: Iteration   3 Fine resid/bnorm = 2.918558728
    MLMG: Iteration   4 Fine resid/bnorm = 81.56015694
    MLMG: Iteration   5 Fine resid/bnorm = 2279.227794
    MLMG: Iteration   6 Fine resid/bnorm = 63693.83698
    MLMG: Iteration   7 Fine resid/bnorm = 1779947.086
    MLMG: Iteration   8 Fine resid/bnorm = 49741258.78
    MLMG: Iteration   9 Fine resid/bnorm = 1390037291
    MLMG: Iteration  10 Fine resid/bnorm = 3.884508992e+10
    MLMG: Iteration  11 Fine resid/bnorm = 1.08553995e+12
    MLMG: Iteration  12 Fine resid/bnorm = 3.033580267e+13
    MLMG: Iteration  13 Fine resid/bnorm = 8.477448701e+14
    MLMG: Iteration  14 Fine resid/bnorm = 2.369053401e+16
    MLMG: Iteration  15 Fine resid/bnorm = 6.620404574e+17
    MLMG: Iteration  16 Fine resid/bnorm = 1.850095768e+19
    MLMG: Iteration  17 Fine resid/bnorm = 5.170158884e+20
    MLMG: Failing to converge after 17 iterations. resid, resid/bnorm = 1.52087323e+25, 5.170158884e+20


In this case, the solver diverges and it is generally a clear indication that the problem
is not properly setup. To aid in debugging, it may be useful to dump plotfiles of the residual
and solution at the time of the failure to visually inspect where the solution is diverging.
This can be accomplished by enabling residual plotting on MLMG failure:

::

    peleLM.mlmg_fail_plt_residuals = true       # [OPT, DEF=false] Dump MLMG residuals plotfiles on MLMG failure

In the example above, plotfiles named ``<plot_dir>/pltMLMGResidual_<solver>_<step>_<iters>``
would be created after the failure in the 17th iteration of the nodal projection at the given time step.
However, given that ``resid/bnorm = 5.170158884e+20`` here, it is likely that the
residual is large everywhere in the domain. Therefore, it may be more useful to dump the residuals
after a smaller number of iterations, e.g., iteration 3 or 4 where the residual is still relatively small,
but clearly growing. Since the nodal projection is always the last MLMG solve for a given time step,
this can be accomplihed by setting `nodal_proj.maxiter=4`. This results in the following output upon failure:

::

    MLMG: # of AMR levels: 1
        # of MG levels on the coarsest AMR level: 5
    MLMG: Initial rhs               = 29416.37318
    MLMG: Initial residual (resid0) = 29416.37318
    MLMG: Iteration   1 Fine resid/bnorm = 0.003754106743
    MLMG: Iteration   2 Fine resid/bnorm = 0.1044336961
    MLMG: Iteration   3 Fine resid/bnorm = 2.918500162
    MLMG: Iteration   4 Fine resid/bnorm = 81.55801241
    MLMG: Failed to converge after 4 iterations. resid, resid/bnorm = 2399140.929, 81.55801241

    *** Nodal projection MLMG solve failed! ***
    Error: MLMG failed to converge.
    Dumping residuals for debugging...

For the other solvers (MAC projection, species/temperature diffusion, and tensor diffusion),
users can further control when to dump the residuals for debugging based on
a minimum number of SDC and, when appropriate, deltaT iterations.  This is done
on a per-solver basis using the following options:

::

    # MAC projection controls
    mac_proj.mlmg_fail_sdc_miniter = 2                             # [OPT, DEF=-1] Minimum SDC iterations before dumping residuals on MLMG failure
    mac_proj.mlmg_fail_maxiter_after_sdc_miniter = 3               # [OPT, DEF=-1] Maximum MLMG solver iters after minimum SDC iters have been reached

    # Species/Temperature diffusion controls
    diffusion.mlmg_fail_sdc_miniter = 1                            # [OPT, DEF=-1] Minimum SDC iterations before dumping residuals on MLMG failure
    diffusion.mlmg_fail_deltaT_miniter = 4                         # [OPT, DEF=-1] Minimum deltaT iterations before dumping residuals on MLMG failure (only for temperature diffusion)
    diffusion.mlmg_fail_species_maxiter_after_sdc_miniter = 5      # [OPT, DEF=-1] Maximum species MLMG solver iters after minimum SDC iters have been reached
    diffusion.mlmg_fail_temp_maxiter_after_sdc_deltaT_miniter = 3  # [OPT, DEF=-1] Maximum temp MLMG solver iters after minimum SDC and deltaT iters have been reached

    # Velocity diffusion controls
    tensor_diffusion.mlmg_fail_sdc_miniter = 1                     # [OPT, DEF=-1] Minimum SDC iterations before dumping residuals on MLMG failure
    tensor_diffusion.mlmg_maxiter_after_sdc_miniter = 6            # [OPT, DEF=-1] Maximum MLMG solver iters after minimum SDC iters have been reached

As an example, setting ``diffusion.mlmg_fail_sdc_miniter = 2``, ``diffusion.mlmg_fail_deltaT_miniter = 2``,
and ``diffusion.mlmg_fail_temp_maxiter_after_sdc_deltaT_miniter = 3`` would result in dumping the temperature diffusion
residuals for debugging only if the failure occurs after at least 2 SDC iterations and 2 deltaT iterations, and
the MLMG solver has performed at least 3 iterations after those minimums have been reached.


Chemistry integration failure
-----------------------------

PeleLMeX relies on `Sundials CVODE <https://computing.llnl.gov/projects/sundials/cvode>`_ to
integrate the stiff ODE resulting of the chemical system (along with advection/diffusion
forcing). CVODE has multiple failure modes, but the most common one appearing in PeleLMeX
will promp a message similar to one of the following:

::

    From CVODE: At t = 0 and h = 6.01889e-195, the corrector convergence test failed repeatedly or with |h| = hmin.```
    From CVODE: At t = 2.459e-6 and h = 6.01889e-16, the corrector convergence test failed repeatedly or with |h| = hmin.```
    [CVODE ERROR]  CVode
        At t = 5.09606e-09, mxstep steps taken before reaching tout.

All of which indicate that the internal sub-stepping algorithm of CVODE did not managed to integrate
the system of ODEs up to the CFL-constrained time step requested by PeleLMeX because CVODE logic
reauired awfully small substep size.

In the case of the first message, one can see that CVODE failed right away (`At t = 0`) which suggests
that the state given to CVODE was wrong. If this happens right at the start of the simulation, your
initial solution is most likely erroneous.

In the case of the second message, the system was integrated up to 2.459e-6 s, but CVODE was not able
to proceed any further as its internal step size dropped to a small value. This could indicates that your
CFL condition is too loose and the chemical stiffness can't be properly handled by
CVODE. You can consider reduce your CFL number:

::

    peleLM.cfl = 0.1

if your CFL step size is too large (generally >1e-5 s). e.g. as for a slow, laminar case. This message
can also appear if your state contains species mass fraction undershoots due to poor spatial resolution.
In this case, one can use the following option:

::

    ode.clean_init_massfrac = 1

where the ODE integration is then computed as an increment where the initial species mass fractions
[0-1] bounds are enforced.
