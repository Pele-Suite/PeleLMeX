.. PeleLMeX documentation master file, created by
   sphinx-quickstart on Fri Mar 18 15:03:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PeleLMeX's documentation!
====================================

`PeleLMeX` is the non-subcycling version of `PeleLM <https://amrex-combustion.github.io/PeleLM/>`_, an adaptive-mesh low Mach number hydrodynamics
code for reacting flows. If you need help or have questions, please use the `GitHub discussion <https://github.com/AMReX-Combustion/PeleLMeX/discussions>`_.
The documentation pages appearing here are distributed with the code in the ``Docs`` folder as "restructured text" files.  The html is built
automatically with certain pushes to the `PeleLMeX` GibHub repository. A local version can also be built as follows ::

    cd ${PELE_HOME}/Docs
    cmake -S . -B build
    cmake --build build

where ``PELE_HOME`` is the location of your clone of the `PeleLMeX` repository.  To view the local pages,
point your web browser at the file ``${PELE_HOME}/Docs/build/html/index.html``.

.. toctree::
   :maxdepth: 2
   :caption: Theory:

   Model.rst
   Implementation.rst
   Validation.rst
   Performances.rst

.. toctree::
   :maxdepth: 2
   :caption: Usage:

   LMeXControls.rst
   Troubleshooting.rst

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   Tutorials.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
