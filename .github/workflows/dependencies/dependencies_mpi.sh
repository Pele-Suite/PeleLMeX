#!/usr/bin/env bash
#
# Copyright 2020 The AMReX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

# OpenMPI (mpich's mpiexec does not work on ubuntu-24.04). The GNUmake jobs
# build with the runner's default g++.

set -eu -o pipefail

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    libopenmpi-dev \
    openmpi-bin
