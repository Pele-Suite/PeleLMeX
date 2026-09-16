#!/bin/bash -l

# This is run on an interactive node
# Summit
# bsub -W 1:00 -nnodes 1 -P CMB138 -Is /bin/bash

# Spock: missing internet connection to clone !
# salloc -t 1:00:00 -N 1 -A CMB138 -p ecp

# Tulip
# srun -N 1 -t 1:00:00 -p amdMI100 --pty bash -i

# Crusher: missing internet connection to clone !
# salloc -J LMeXTest -t 1:00:00 -N 1 -A CMB138_crusher -p batch

set -e

MYPWD=${PWD}

testMachine=crusher

if [ "$testMachine" = "summit" ]; then
   module load cmake gcc cuda python
elif [ "$testMachine" = "spock" ]; then
   module load PrgEnv-amd cmake rocm/4.5.0
elif [ "$testMachine" = "crusher" ]; then
   module load PrgEnv-amd cmake rocm/5.1.0 craype-accel-amd-gfx90a cray-libsci/21.08.1.2
elif [ "$testMachine" = "frontier" ]; then
   module load PrgEnv-cray cmake rocm/5.2.0 craype-x86-trento craype-accel-amd-gfx90a cray-libsci/21.08.1.2
elif [ "$testMachine" = "tulip" ]; then
   module unload craype-x86-naples rocm
   module load craype-x86-rome rocm cmake
fi

set -x

# Get LMeX and deps
git clone --recursive https://github.com/Pele-Suite/PeleLMeX.git PeleLMeX

# Get P.Mullowney & S.Thomas Hypre
git clone --branch AMD-ILU-2022-12-06 https://github.com/PaulMullowney/hypre.git

# Patch amrex
cd ${MYPWD}/PeleLMeX/Submodules/amrex
patch -p1 <<'EOF'
diff --git a/Src/Extern/HYPRE/AMReX_HypreIJIface.cpp b/Src/Extern/HYPRE/AMReX_HypreIJIface.cpp
index c2e4f1262..76e293a97 100644
--- a/Src/Extern/HYPRE/AMReX_HypreIJIface.cpp
+++ b/Src/Extern/HYPRE/AMReX_HypreIJIface.cpp
@@ -304,6 +304,12 @@ void HypreIJIface::boomeramg_precond_configure (const std::string& prefix)
             hpp("bamg_ilu_type", HYPRE_BoomerAMGSetILUType);
             hpp("bamg_ilu_level", HYPRE_BoomerAMGSetILULevel);
             hpp("bamg_ilu_max_iter", HYPRE_BoomerAMGSetILUMaxIter);
+
+            // These only work with SThomas IterTriSolve branch
+            hpp("bamg_ilu_tri_solve", HYPRE_BoomerAMGSetILUTriSolve);
+            hpp("bamg_ilu_lower_jacobi_iters", HYPRE_BoomerAMGSetILULowerJacobiIters);
+            hpp("bamg_ilu_upper_jacobi_iters", HYPRE_BoomerAMGSetILUUpperJacobiIters);
+            hpp("bamg_ilu_reordering_type", HYPRE_BoomerAMGSetILULocalReordering);
         }
         else if (smooth_type == 7) { // Pilut
             hpp("bamg_smooth_num_sweeps", HYPRE_BoomerAMGSetSmoothNumSweeps);
EOF
cd -

# Build Hypre
cd ${MYPWD}/hypre/src
if [ "$testMachine" = "summit" ]; then
   export CFLAGS="-g O2"
   export CXXFLAG="-g O2"
   export FC=$(which mpif90)
   export CXX=$(which mpicxx)
   export CC=$(which mpicc)
   make distclean
   ./configure --prefix=${MYPWD}/hypre/install/ --without-superlu --disable-bigint --without-openmp --with-cuda --enable-unified-memory --enable-curand --enable-cusolver --enable-cusparse --disable-cublas --enable-gpu-profiling --enable-shared
   make -j24
   make install
elif [ "$testMachine" = "crusher" ]; then
   export CFLAGS="-g -O2 -I$MPICH_DIR/include"
   export CXXFLAGS="-g -O2 -I$MPICH_DIR/include"
   export FC=$(which ftn)
   export CXX=$(which CC)
   export CC=$CXX
   make distclean
   ./configure --prefix=${MYPWD}/hypre/install/ --with-gpu-arch=gfx90a --without-superlu --disable-bigint --without-openmp --with-hip --enable-rocsparse --enable-rocrand --enable-shared --with-MPI-lib-dirs=/opt/cray/pe/mpich/8.1.16/ofi/crayclang/10.0/lib /opt/cray/pe/mpich/8.1.16/gtl/lib --with-MPI-libs=mpi mpi_gtl_hsa --with-MPI-include=/opt/cray/pe/mpich/8.1.16/ofi/crayclang/10.0/include
   make -j24
   make install
elif [ "$testMachine" = "frontier" ]; then
   export CFLAGS="-g -O2 -I$MPICH_DIR/include"
   export CXXFLAGS="-g -O2 -I$MPICH_DIR/include"
   export FC=$(which ftn)
   export CXX=$(which CC)
   export CC=$CXX
   make distclean
   ./configure --prefix=${MYPWD}/hypre/install/ --with-gpu-arch=gfx90a --without-superlu --disable-bigint --without-openmp --with-hip --enable-rocsparse --enable-rocrand --enable-shared --with-MPI-lib-dirs=/opt/cray/pe/mpich/8.1.16/ofi/crayclang/10.0/lib /opt/cray/pe/mpich/8.1.16/gtl/lib --with-MPI-libs=mpi mpi_gtl_hsa --with-MPI-include=/opt/cray/pe/mpich/8.1.16/ofi/crayclang/10.0/include
   make -j24
   make install
fi
cd -

# Build LMeX KPP2 Challenge Problem case
export PELE_HOME="${MYPWD}/PeleLMeX"
export AMREX_HOME="${MYPWD}/PeleLMeX/Submodules/amrex"
export AMREX_HYDRO_HOME="${MYPWD}/PeleLMeX/Submodules/AMReX-Hydro"
export PELE_PHYSICS_HOME="${MYPWD}/PeleLMeX/Submodules/PelePhysics"
export PELEMP_HOME="${MYPWD}/PeleLMeX/Submodules/PeleMP"
export HYPRE_HOME=${MYPWD}/hypre/install/
cd ${MYPWD}/PeleLMeX/Exec/Cases/ChallengeProblem/
if [ "$testMachine" = "summit" ]; then
   # Currently this branch default to HIP=TRUE, so switch it off for Summit
   #Build SUNDIALS (requires internet connection because a clone happens)
   make -j 12 COMP=gcc USE_HIP=FALSE USE_CUDA=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=summit TPLrealclean
   make -j 12 COMP=gcc USE_HIP=FALSE USE_CUDA=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=summit TPL

   #Build PeleLMeX
   make -j 12 COMP=gnu USE_HIP=FALSE USE_CUDA=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=summit realclean
   make -j 12 COMP=gnu USE_HIP=FALSE USE_CUDA=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=summit

elif [ "$testMachine" = "spock" ]; then
   #Build SUNDIALS (requires internet connection because a clone happens)
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=FALSE Chemistry_Model=dodecane_lu_qss HOSTNAME=spock TPLrealclean
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=FALSE Chemistry_Model=dodecane_lu_qss HOSTNAME=spock TPL

   #Build PeleLMeX
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=FALSE Chemistry_Model=dodecane_lu_qss HOSTNAME=spock realclean
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=FALSE Chemistry_Model=dodecane_lu_qss HOSTNAME=spock

elif [ "$testMachine" = "crusher" ]; then
   #Build SUNDIALS (requires internet connection because a clone happens)
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=crusher TPLrealclean
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=crusher TPL

   #Build PeleLMeX
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=crusher realclean
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=crusher
elif [ "$testMachine" = "frontier" ]; then
   #Build SUNDIALS (requires internet connection because a clone happens)
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=frontier TPLrealclean
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=frontier TPL

   #Build PeleLMeX
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=frontier realclean
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE Chemistry_Model=dodecane_lu_qss HOSTNAME=frontier
fi
