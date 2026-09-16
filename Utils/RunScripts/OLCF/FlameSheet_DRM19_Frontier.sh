#!/bin/bash -l

#SBATCH -A CMB138_crusher
#SBATCH -J LMEX_FSDRM19
#SBATCH -o %x-%j.out
#SBATCH -t 01:00:00
#SBATCH -N 1

MYPWD=${PWD}

cmd() {
  echo "+ $@"
  eval "$@"
}

set -e

export https_proxy="http://proxy.ccs.ornl.gov:3128"

cmd "module purge"
cmd "module load PrgEnv-cray"
cmd "module load cmake cray-python craype-x86-trento craype-accel-amd-gfx90a rocm/5.4.0"
cmd "module load cray-libsci/22.12.1.1"

cmd "git clone --recursive https://github.com/Pele-Suite/PeleLMeX.git || true"
cmd "cd ${MYPWD}/PeleLMeX/Exec/RegTests/FlameSheet"

#Build SUNDIALS (requires internet connection because a clone happens)
cmd "make -j 20 DIM=3 USE_HIP=TRUE USE_MPI=TRUE TINY_PROFILE=TRUE PELE_USE_MAGMA=TRUE Chemistry_Model=drm19 TPLrealclean"
cmd "make -j 20 DIM=3 USE_HIP=TRUE USE_MPI=TRUE TINY_PROFILE=TRUE PELE_USE_MAGMA=TRUE Chemistry_Model=drm19 TPL"

#Build PeleLMeX
cmd "make -j 20 DIM=3 USE_HIP=TRUE USE_MPI=TRUE TINY_PROFILE=TRUE PELE_USE_MAGMA=TRUE Chemistry_Model=drm19 realclean"
cmd "make -j 20 DIM=3 USE_HIP=TRUE USE_MPI=TRUE TINY_PROFILE=TRUE PELE_USE_MAGMA=TRUE Chemistry_Model=drm19"

ARGS="input.3d-regt geometry.prob_lo=0.0 0.0 0.0 geometry.prob_hi=0.032 0.032 0.032 amr.n_cell=64 64 64 amr.max_level=3 amr.max_grid_size=128 amr.blocking_factor=16 prob.P_mean=101325.0 prob.standoff=-.023 prob.pertmag=0.00045 prob.pertlength=0.016 peleLM.num_init_iter=1 peleLM.do_temporals=0 amr.max_step=16 amr.dt_shrink=0.25 amr.fixed_dt=2.0e-6 peleLM.v=3 cvode.solve_type=magma_direct amr.plot_int=-1 peleLM.diagnostics=xnormal peleLM.xnormal.int=500 amrex.abort_on_out_of_gpu_memory=1"

cmd "export FI_MR_CACHE_MONITOR=memhooks"
cmd "export FI_CXI_RX_MATCH_MODE=software"
cmd "srun -N1 -n8 --gpus-per-node=8 --gpu-bind=closest ./PeleLMeX3d.hip.x86-trento.TPROF.MPI.HIP.ex ${ARGS}"
