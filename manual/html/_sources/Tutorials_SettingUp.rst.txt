Setting-up your environment
---------------------------

Getting a functioning environment in which to compile and run `PeleLMeX` is the first step of this tutorial.
Follow the steps listed below to get to this point:

#. The first step is to get `PeleLMeX` and its dependencies. To do so, use a recursive *git clone*: ::

    git clone --recursive --shallow-submodules --single-branch https://github.com/AMReX-Combustion/PeleLMeX.git

   The ``--shallow-submodules`` and ``--single-branch`` flags are recommended for most users as they
   substantially reduce the size of the download by skipping extraneous parts of the git history.
   Developers may wish to omit these flags in order download the complete git history of PeleLMeX
   and its submodules, though standard ``git`` commands may also be used after a shallow clone to
   obtain the skipped portions if needed.

#. Move into the Exec folder containing your tutorial. To do so: ::

    cd PeleLMeX/Exec/RegTests/<CaseName>

   where <CaseName> is the name of your tutorial, e.g. ``HotBubble``, ``FlameSheet``,
   ``EB_BackwardStepFlame``, ``EB_FlowPastCylinder``, or ``TripleFlame``.

You're good to go!

.. Note::

   The makefile system is set up such that default paths are automatically set to the
   submodules obtained with the recursive *git clone*, however advanced users can set their own dependencies
   in the `GNUmakefile` for each case by updating the top-most lines as follows: ::

       PELE_HOME     = <path_to_PeleLMeX>
       AMREX_HOME        = <path_to_MyAMReX>
       AMREX_HYDRO_HOME  = <path_to_MyAMReXHydro>
       PELE_PHYSICS_HOME = <path_to_MyPelePhysics>
       SUNDIALS_HOME     = <path_to_MySUNDIALS>

   or directly through shell environment variables (using *bash* for instance): ::

       export PELE_HOME=<path_to_PeleLMeX>
       export AMREX_HOME=<path_to_MyAMReX>
       export AMREX_HYDRO_HOME=<path_to_MyAMReXHydro>
       export PELE_PHYSICS_HOME=<path_to_MyPelePhysics>
       export SUNDIALS_HOME=<path_to_MySUNDIALS>

   Note that using the first option will overwrite any
   environment variables you might have previously defined when using this `GNUmakefile`.
