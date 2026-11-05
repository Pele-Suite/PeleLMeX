cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(PeleLMeX LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")
include(CMakePackageConfigHelpers)
include(PeleUtils)
enable_testing()
include(CTest)

set(PELE_DIM "3" CACHE STRING "Number of physical dimensions")
option(PELE_ENABLE_CLANG_TIDY "Enable clang-tidy analysis" OFF)
option(PELE_ENABLE_CPPCHECK "Enable cppcheck analysis" OFF)
option(PELE_ENABLE_FCOMPARE "Enable building fcompare when not testing" OFF)
option(PELE_ENABLE_FCOMPARE_FOR_TESTS "Check test plots against gold files" OFF)
option(PELE_SAVE_GOLDS "Enable saving of gold files to a specified directory" OFF)
option(PELE_ENABLE_SANITIZE_FOR_TESTS "Currently only disables certain long running MMS tests if set" OFF)
option(PELE_ENABLE_FPE_TRAP_FOR_TESTS "Enable FPE trapping in tests" ON)
option(PELE_ENABLE_TINY_PROFILE "Enable tiny profiler in AMReX" OFF)
option(PELE_ENABLE_HDF5 "Enable plot file output using HDF5" OFF)
option(PELE_ENABLE_HDF5_ZFP "Enable ZFP compression in HDF5" OFF)
option(PELE_ENABLE_ASCENT "Enable Ascent in-situ visualization" OFF)
option(PELE_ENABLE_HYPRE "Enable Offload to Hypre" OFF)
set(PELE_PRECISION "DOUBLE" CACHE STRING "Floating point precision SINGLE or DOUBLE")

# Physics options
option(PELE_ENABLE_EB "Enable Embedded Boundary" OFF)
option(PELE_ENABLE_PARTICLES "Enable particles and spray" ON)
option(PELE_CVODE_FORCE_YCORDER "Use YCOrder in CVODE on GPUs (enables sparse_direct, precGMRES, fixed_point)" OFF)

# HPC options
option(PELE_ENABLE_MPI "Enable MPI" OFF)
option(PELE_ENABLE_OPENMP "Enable OpenMP" OFF)
option(PELE_ENABLE_CUDA "Enable CUDA" OFF)
option(PELE_ENABLE_HIP "Enable HIP" OFF)
option(PELE_ENABLE_SYCL "Enable SyCL" OFF)

# C++ Options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(PELE_ENABLE_FCOMPARE_FOR_TESTS OR PELE_ENABLE_PARTICLES)
  set(PELE_ENABLE_FCOMPARE ON)
endif()

if(PELE_ENABLE_CUDA)
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.0")
    message(FATAL_ERROR "Your nvcc version is ${CMAKE_CUDA_COMPILER_VERSION} which is unsupported."
      "Please use CUDA toolkit version 12.0 or newer.")
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
  endif()
endif()

if(PELE_ENABLE_HIP)
   enable_language(HIP)
endif ()

if(PELE_DIM EQUAL 1)
    message(FATAL_ERROR "${PROJECT_NAME} does not support 1D.")
endif()

########################### AMReX #####################################
message(STATUS "AMReX Configure Section")
set(AMREX_SUBMOD_LOCATION "${CMAKE_SOURCE_DIR}/Submodules/PelePhysics/Submodules/amrex")
include(SetAmrexOptions)
list(APPEND CMAKE_MODULE_PATH "${AMREX_SUBMOD_LOCATION}/Tools/CMake")
add_subdirectory(${AMREX_SUBMOD_LOCATION})
include(SetAmrexCompileFlags)

########################### AMReX-Hydro ###############################
message(STATUS "AMReX-Hydro Configure Section")
set(AMREX_HYDRO_SUBMOD_LOCATION "${CMAKE_SOURCE_DIR}/Submodules/AMReX-Hydro")
include(SetAmrexHydroOptions)
add_subdirectory(${AMREX_HYDRO_SUBMOD_LOCATION})

########################### SUNDIALS ##################################
message(STATUS "Sundials Configure Section")
set(SUNDIALS_SUBMOD_LOCATION "${CMAKE_SOURCE_DIR}/Submodules/PelePhysics/Submodules/sundials")
include(SetSundialsOptions)
#BUILD_TESTING is an old CMake keyword so don't clear it in sundials configure
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
add_subdirectory(${SUNDIALS_SUBMOD_LOCATION})

########################### PeleLMeX ##################################
if(PELE_ENABLE_MPI)
  message(STATUS "MPI Configure Section")
  find_package(MPI REQUIRED CXX)
endif()

# General information about machine, compiler, and build type
message(STATUS "${PROJECT_NAME} Information:")
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "PELE_PRECISION = ${PELE_PRECISION}")

init_code_checks()

include(SetRpath)

########################### Git Hash Info ##################################
include(SetGitHashInfo)

add_subdirectory(Exec)
add_subdirectory(Tests)
add_subdirectory(${CMAKE_SOURCE_DIR}/Submodules/PelePhysics/Testing/Exec/Radiation)
