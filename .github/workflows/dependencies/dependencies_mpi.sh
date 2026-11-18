#!/usr/bin/env bash
#
# Copyright 2020 The AMReX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

# OpenMPI for the GNUmake jobs; they build with the runner's default g++.

set -eu -o pipefail

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    libopenmpi-dev \
    openmpi-bin
