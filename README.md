<img width="236" height="300" alt="RD100_2026_Winner_Logo" src="https://github.com/user-attachments/assets/e70b1412-ca42-4ca3-9436-acdea37ef722" />

# PeleLMeX

[![AMReX Badge](https://img.shields.io/static/v1?label=%22powered%20by%22&message=%22AMReX%22&color=%22blue%22)](https://amrex-codes.github.io/amrex/)
[![Exascale Computing Project](https://img.shields.io/badge/supported%20by-ECP-blue)](https://www.exascaleproject.org/research-project/combustion-pele/)
[![Language: C++20](https://img.shields.io/badge/language-C%2B%2B20-blue)](https://isocpp.org/)
[![Citing](https://joss.theoj.org/papers/10.21105/joss.05450/status.svg)](https://joss.theoj.org/papers/10.21105/joss.05450)
[![Archive](https://zenodo.org/badge/DOI/10.5281/zenodo.10056232.svg)](https://doi.org/10.5281/zenodo.10056232)

![CI](https://github.com/Pele-Suite/PeleLMeX/workflows/PeleLMeX-CI/badge.svg)

## Overview

*PeleLMeX* is a solver for high fidelity reactive flow simulations, namely direct numerical simulation (DNS) and large eddy simulation (LES).
The solver combines a low Mach number approach, adaptive mesh refinement (AMR), embedded boundary (EB) geometry treatment and high performance
computing (HPC) to provide
a flexible tool to address research questions on platforms ranging from small workstations to the world's largest GPU-accelerated supercomputers.
*PeleLMeX* has been used to study complex flame/turbulence interactions in [RCCI engines](https://gfm.aps.org/meetings/dfd-2022/63236765199e4c2c0873f9f6) and
[hydrogen combustion](https://www.sciencedirect.com/science/article/pii/S001021802300192X) or [the effect of sustainable aviation fuel
on gas turbine combustion](https://www.osti.gov/biblio/1995457).

*PeleLMeX* is part of the [Pele combustion Suite](https://pele-suite.github.io/).

## Documentation

![Documentation](https://github.com/Pele-Suite/PeleLMeX/workflows/PeleLMeX-Docs/badge.svg)

*PeleLMeX* is a non-subcycling version of [*PeleLM*](https://github.com/AMReX-Combustion/PeleLM) based on AMReX's
[AmrCore](https://amrex-codes.github.io/amrex/docs_html/AmrCore.html) and borrowing from the incompressible
solver [incflo](https://github.com/AMReX-Codes/incflo). It solves of the multispecies reactive Navier-Stokes equations
in the low Mach number limit as described in the [documentation](https://pele-suite.github.io/PeleLMeX/manual/html/index.html).
It inherits most of *PeleLM* algorithmic features, but differs significantly in its implementation stemming from the non-subcycling approach.
*PeleLM* is no longer under active development; *PeleLMeX* should be used for simulations of low Mach number reacting flows and
[*PeleC*](https://github.com/Pele-Suite/PeleC) for simulations of flows with higher Mach numbers where compressibility effects are
significant.

A overview of *PeleLMeX* controls is provided in the [documentation](https://pele-suite.github.io/PeleLMeX/manual/html/LMeXControls.html).

### Core Algorithm

The *PeleLMeX* governing equations and core algorithms are described in:

https://pele-suite.github.io/PeleLMeX/manual/html/Model.html#mathematical-background

https://pele-suite.github.io/PeleLMeX/manual/html/Model.html#pelelmex-algorithm

### Tutorials

A set of self-contained tutorials describing more complex problems is also provided:

https://pele-suite.github.io/PeleLMeX/manual/html/Tutorials.html

## Installation

### Requirements

Compiling *PeleLMeX* requires a C++20 compatible compiler as
well as [CMake](https://cmake.org/download/) >= 3.23 for compiling the [SUNDIALS](https://github.com/LLNL/sundials) third party library.

Most of the examples provided hereafter and in the [tutorials](https://pele-suite.github.io/PeleLMeX/manual/html/Tutorials.html)
will use MPI to run in parallel. Although not mandatory, it is advised to build *PeleLMeX* with MPI support from the get go if
more than a single core is available to you. Any of [mpich](https://www.mpich.org/downloads/) or
[open-mpi](https://www.open-mpi.org/software/ompi/v4.1/) is a suitable option if MPI is not already available on your platform.

Finally, when building with GPU support, CUDA >= 12 is required with NVIDIA GPUs and ROCm >= 6 is required with AMD GPUs.

### Download

The preferred method consists of cloning *PeleLMeX* and its submodules
([PelePhysics](https://github.com/Pele-Suite/PelePhysics),
[amrex](https://github.com/AMReX-Codes/amrex),
[AMReX-Hydro](https://github.com/AMReX-Fluids/AMReX-Hydro), and
[SUNDIALS](https://github.com/LLNL/sundials) using a recursive `git clone`:

```
git clone --recursive --shallow-submodules --single-branch https://github.com/Pele-Suite/PeleLMeX.git
```

The `--shallow-submodules` and `--single-branch` flags are recommended for most users as they substantially reduce the size of the download by
skipping extraneous parts of the git history. Developers may wish to omit these flags in order download the complete git history of PeleLMeX
and its submodules, though standard `git` commands may also be used after a shallow clone to obtain the skipped portions if needed. By default,
you will download the latest version of PeleLMeX. If you'd like to use a specific [released version](https://github.com/Pele-Suite/PeleLMeX/releases/)
of PeleLMeX, you can add the option `--branch=<version>`, e.g. `--branch=v25.04`, to your clone command. See more on PeleLMeX version numbers
in the [Versioning](#Versioning) section below.

Alternatively, you can use a separate `git clone` of each of the submodules.
The default location for *PeleLMeX* dependencies is the `Submodules` folder but you optionally
setup the following environment variables (e.g. using bash) to any other location:

```
export PELE_HOME=<path_to_PeleLMeX>
export AMREX_HYDRO_HOME=${PELE_HOME}/Submodules/AMReX-Hydro
export PELE_PHYSICS_HOME=${PELE_HOME}/Submodules/PelePhysics
export AMREX_HOME=${PELE_PHYSICS_HOME}/Submodules/amrex
export SUNDIALS_HOME=${PELE_PHYSICS_HOME}/Submodules/sundials
```

### Compilation

Both GNUmake and CMake can be used to build *PeleLMeX* executables. GNUmake is the preferred choice for single executables when running production simulations. While CMake is the preferred method for automatically building and testing most available executables.
The code handling the initial condition and boundary conditions is unique to each case,
and subfolders in the `Exec` directory provide a number of examples.

For instance, to compile the executable for the case of a rising hot bubble,
move into the `HotBubble` folder:

```
cd PeleLMeX/Exec/RegTests/HotBubble
```

If this is a clean install, you will need to make the third party libraries with: `make TPL` (note: if on macOS, you might need to specify `COMP=llvm` in the `make` statements).

Finally, make with: `make -j`, or if on macOS: `make -j COMP=llvm`. To clean the installation, use either `make clean` or `make realclean`. If running into compile errors after changing compile time options in PeleLMeX (e.g., the chemical mechanism), the first thing to try is to clean your build by running `make TPLrealclean && make realclean`, then try to rebuild the third party libraries and PeleLMeX with `make TPL && make -j`. See the [Tutorial](https://pele-suite.github.io/PeleLMeX/manual/html/Tutorials_HotBubble.html) for this case for instructions on how to compile with different options (for example, to compile without MPI support or to compile for GPUs) and how to run the code once compiled.

To compile and test using CMake, refer to the example `cmake.sh` script in the `Build` directory, or reference the GitHub Actions workflows in the `.github/workflows` directory.

## Getting help, contributing

Do you have a question ? Found an issue ? Please use the [GitHub Discussions](https://github.com/Pele-Suite/PeleLMeX/discussions) to engage
with the development team or open a new [GitHub issue](https://github.com/Pele-Suite/PeleLMeX/issues) to report a bug. The development team
also encourages users to take an active role in respectfully answering each other's questions in these spaces. When reporting a bug, it is helpful
to provide as much detail as possible, including a case description and the major compile and runtime options being used. Though not required,
it is most effective to create a fork of this repository and share a branch of that fork with a case that minimally reproduces the error.

New contributions to *PeleLMeX* are welcome ! Contributing Guidelines are provided in [CONTRIBUTING.md](CONTRIBUTING.md).

## Versioning

PeleLMeX now uses uses a type of semantic versioning to help users navigate different versions of the code,
which are labeled with [GitHub tags](https://github.com/Pele-Suite/PeleLMeX/releases/). These tagged versions are not exhaustive, but they adhere to
the following convention. Given a version number MAJOR.MINOR.PATCH:
1. MAJOR version for changes to key aspects of the solver affecting input/source files for all cases, when a key model is changed to significantly affect results of simulations, when a major new capability is added
2. MINOR version for when a significant feature is added (in a backward compatible manner), accumulation of smaller features, or changes to input file compatibility for less central aspects of the solver (e.g., post-processing) or aspects not affecting all cases
3. PATCH version for backward compatible bug fixes and minor features

PeleLMeX previously used YY.MM formatting for versions. These should be interpreted as version 0 subversions,
e.g. v25.04 is equivalent to v0.25.04. Significant breaking changes were added for the release of v1.0.0. See
[CHANGES.md](CHANGES.md) for updates needed to transition case files designed for earlier versions of PeleLMeX
to this newer version.

## Acknowledgment

This research was supported by the Exascale Computing Project (ECP), Project
Number: 17-SC-20-SC, a collaborative effort of two DOE organizations -- the
Office of Science and the National Nuclear Security Administration --
responsible for the planning and preparation of a capable exascale ecosystem --
including software, applications, hardware, advanced system engineering, and
early testbed platforms -- to support the nation's exascale computing
imperative.

## Citation

To cite PeleLMeX, please use [![Citing](https://joss.theoj.org/papers/10.21105/joss.05450/status.svg)](https://joss.theoj.org/papers/10.21105/joss.05450) and [the Pele software suite](https://doi.org/10.1137/1.9781611977967.2)

```
@article{PeleLMeX_JOSS,
  doi = {10.21105/joss.05450},
  url = {https://doi.org/10.21105/joss.05450},
  year = {2023},
  month = october,
  publisher = {The Open Journal},
  volume = {8},
  number = {90},
  pages = {5450},
  author = {Lucas Esclapez and Marc Day and John Bell and Anne Felden and Candace Gilet and Ray Grout and Marc {Henry de Frahan} and Emmanuel Motheau and Andrew Nonaka and Landon Owen and Bruce Perry and Jon Rood and Nicolas Wimer and Weiqun Zhang},
  journal = {Journal of Open Source Software},
  title= {{PeleLMeX: an AMR Low Mach Number Reactive Flow Simulation Code without level sub-cycling}}
}

@article{PeleSoftware,
  author = {Marc T. {Henry de Frahan} and Lucas Esclapez and Jon Rood and Nicholas T. Wimer and Paul Mullowney and Bruce A. Perry and Landon Owen and Hariswaran Sitaraman and Shashank Yellapantula and Malik Hassanaly and Mohammad J. Rahimi and Michael J. Martin and Olga A. Doronina and Sreejith N. A. and Martin Rieth and Wenjun Ge and Ramanan Sankaran and Ann S. Almgren and Weiqun Zhang and John B. Bell and Ray Grout and Marc S. Day and Jacqueline H. Chen},
  title = {The Pele Simulation Suite for Reacting Flows at Exascale},
  booktitle = {Proceedings of the 2024 SIAM Conference on Parallel Processing for Scientific Computing},
  journal = {Proceedings of the 2024 SIAM Conference on Parallel Processing for Scientific Computing},
  chapter = {},
  pages = {13-25},
  doi = {10.1137/1.9781611977967.2},
  URL = {https://epubs.siam.org/doi/abs/10.1137/1.9781611977967.2},
  eprint = {https://epubs.siam.org/doi/pdf/10.1137/1.9781611977967.2},
  year = {2024},
  publisher = {Proceedings of the 2024 SIAM Conference on Parallel Processing for Scientific Computing}
}
```

A full list of publications documenting the development of the Pele suite and its
application to various reacting flow and other simulations is available on the main
[Pele suite page](https://pele-suite.github.io/pubs.html). After publication,
if you'd like your work to be included on that list, you can request to have it added
[here](https://github.com/Pele-Suite/Pele-Suite.github.io/discussions/3).
