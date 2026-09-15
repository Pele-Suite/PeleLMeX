# AMReX
DIM = 3
COMP = gnu
PRECISION = DOUBLE
USE_EB = FALSE
# MeshMappingConvergence/run_stretch_sweep.sh needs 'make USE_HYPRE=TRUE'
USE_HYPRE = FALSE

# Profiling
PROFILE = FALSE
TINY_PROFILE = FALSE
COMM_PROFILE = FALSE
TRACE_PROFILE = FALSE
MEM_PROFILE = FALSE
USE_GPROF = FALSE

# Performance
USE_MPI = TRUE
USE_OMP = FALSE
USE_CUDA = FALSE
USE_HIP = FALSE
USE_SYCL = FALSE

# Debugging
DEBUG = FALSE
FSANITIZER = FALSE
THREAD_SANITIZER = FALSE

# PelePhysics
Chemistry_Model = air
Eos_Model = Fuego
Transport_Model = Simple

PELE_HOME ?= ../../..
include $(PELE_HOME)/Exec/Make.PeleLMeX
