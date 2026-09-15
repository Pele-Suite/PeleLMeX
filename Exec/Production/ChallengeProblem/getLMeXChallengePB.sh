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

# Patch amrex
cd ${MYPWD}/PeleLMeX/Submodules/amrex
patch -p1 <<'EOF'
diff --git a/Src/LinearSolvers/MLMG/AMReX_MLNodeLap_3D_K.H b/Src/LinearSolvers/MLMG/AMReX_MLNodeLap_3D_K.H
index 5886ce2a3..0b7d3f62b 100644
--- a/Src/LinearSolvers/MLMG/AMReX_MLNodeLap_3D_K.H
+++ b/Src/LinearSolvers/MLMG/AMReX_MLNodeLap_3D_K.H
@@ -6666,6 +6666,13 @@ void mlndlap_set_stencil_eb (int i, int j, int k, Array4<Real> const& sten,

     // i+1,j+1,k+1
     sten(i,j,k,ist_ppp) = sig(i,j,k) * (facx*conn(i,j,k,i_c_ybzb) + facy*conn(i,j,k,i_c_xbzb) + facz*conn(i,j,k,i_c_xbyb) );
+
+    for (int n =ist_p00 ; n<=ist_ppp;n++) {
+       if ( sten(i,j,k,n) < 0.0 ) {
+          sten(i,j,k,n) *= 1e-2;
+       }
+    }
+
 }

 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EOF
cd -

# Build LMeX KPP2 Challenge Problem case
export PELE_HOME="${MYPWD}/PeleLMeX"
export AMREX_HOME="${MYPWD}/PeleLMeX/Submodules/amrex"
export AMREX_HYDRO_HOME="${MYPWD}/PeleLMeX/Submodules/AMReX-Hydro"
export PELE_PHYSICS_HOME="${MYPWD}/PeleLMeX/Submodules/PelePhysics"
export PELEMP_HOME="${MYPWD}/PeleLMeX/Submodules/PeleMP"
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
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE USE_HYPRE=FALSE Chemistry_Model=dodecane_lu_qss HOSTNAME=crusher TPLrealclean
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE USE_HYPRE=FALSE Chemistry_Model=dodecane_lu_qss HOSTNAME=crusher TPL

   #Build PeleLMeX
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE USE_HYPRE=FALSE Chemistry_Model=dodecane_lu_qss HOSTNAME=crusher realclean
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE USE_HYPRE=FALSE Chemistry_Model=dodecane_lu_qss HOSTNAME=crusher

elif [ "$testMachine" = "frontier" ]; then
   #Build SUNDIALS (requires internet connection because a clone happens)
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE USE_HYPRE=FALSE Chemistry_Model=dodecane_lu_qss HOSTNAME=frontier TPLrealclean
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE USE_HYPRE=FALSE Chemistry_Model=dodecane_lu_qss HOSTNAME=frontier TPL

   #Build PeleLMeX
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE USE_HYPRE=FALSE Chemistry_Model=dodecane_lu_qss HOSTNAME=frontier realclean
   make -j 8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE USE_HYPRE=FALSE Chemistry_Model=dodecane_lu_qss HOSTNAME=frontier
fi
