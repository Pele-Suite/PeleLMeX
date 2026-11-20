function(build_pele_exe pele_exe_name pele_physics_lib_name)

  set(SRC_DIR ${CMAKE_SOURCE_DIR}/Source)

  add_executable(${pele_exe_name} "")

  if(CLANG_TIDY_EXE)
    set_target_properties(${pele_exe_name} PROPERTIES CXX_CLANG_TIDY
                          "${CLANG_TIDY_EXE};--config-file=${CMAKE_SOURCE_DIR}/.clang-tidy")
  endif()

  target_sources(${pele_exe_name}
     PRIVATE
       pelelmex_prob.H
       pelelmex_prob.cpp
  )

  target_include_directories(${pele_exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(${pele_exe_name} PRIVATE ${SRC_DIR})
  target_include_directories(${pele_exe_name} PRIVATE ${SRC_DIR}/Mesh)
  target_include_directories(${pele_exe_name} PRIVATE ${CMAKE_BINARY_DIR})

  target_sources(${pele_exe_name}
     PRIVATE
       ${SRC_DIR}/PeleLMeX_DeriveUserDefined.cpp
       ${SRC_DIR}/PeleLMeX_DiffusionOp.H
       ${SRC_DIR}/PeleLMeX_DiffusionOp.cpp
       ${SRC_DIR}/PeleLMeX_FlowControllerData.H
       ${SRC_DIR}/PeleLMeX.H
       ${SRC_DIR}/PeleLMeX.cpp
       ${SRC_DIR}/PeleLMeX_Advance.cpp
       ${SRC_DIR}/PeleLMeX_Advection.cpp
       ${SRC_DIR}/PeleLMeX_BC.cpp
       ${SRC_DIR}/PeleLMeX_BCfill.H
       ${SRC_DIR}/PeleLMeX_Data.cpp
       ${SRC_DIR}/PeleLMeX_Derive.H
       ${SRC_DIR}/PeleLMeX_Derive.cpp
       ${SRC_DIR}/PeleLMeX_DeriveFunc.H
       ${SRC_DIR}/PeleLMeX_DeriveFunc.cpp
       ${SRC_DIR}/PeleLMeX_Diagnostics.cpp
       ${SRC_DIR}/PeleLMeX_Diffusion.cpp
       ${SRC_DIR}/PeleLMeX_EB.cpp
       ${SRC_DIR}/PeleLMeX_Eos.cpp
       ${SRC_DIR}/PeleLMeX_Evaluate.cpp
       ${SRC_DIR}/PeleLMeX_Evolve.cpp
       ${SRC_DIR}/PeleLMeX_FlowController.cpp
       ${SRC_DIR}/PeleLMeX_Forces.cpp
       ${SRC_DIR}/PeleLMeX_Init.cpp
       ${SRC_DIR}/PeleLMeX_ODEQty.cpp
       ${SRC_DIR}/PeleLMeX_Plot.cpp
       ${SRC_DIR}/PeleLMeX_Projection.cpp
       ${SRC_DIR}/PeleLMeX_Reactions.cpp
       ${SRC_DIR}/PeleLMeX_Regrid.cpp
       ${SRC_DIR}/PeleLMeX_Setup.cpp
       ${SRC_DIR}/PeleLMeX_Tagging.cpp
       ${SRC_DIR}/PeleLMeX_BPatch.H
       ${SRC_DIR}/PeleLMeX_BPatch.cpp
       ${SRC_DIR}/Mesh/PeleLMeX_MeshMap.H
       ${SRC_DIR}/Mesh/PeleLMeX_MeshMap.cpp
       ${SRC_DIR}/Mesh/PeleLMeX_ConstantMap.H
       ${SRC_DIR}/Mesh/PeleLMeX_ConstantMap.cpp
       ${SRC_DIR}/Mesh/PeleLMeX_ExpStretchMap.H
       ${SRC_DIR}/Mesh/PeleLMeX_ExpStretchMap.cpp
       ${SRC_DIR}/Mesh/PeleLMeX_TanhStretchMap.H
       ${SRC_DIR}/Mesh/PeleLMeX_TanhStretchMap.cpp
       ${SRC_DIR}/Mesh/PeleLMeX_InteriorStretchMap.H
       ${SRC_DIR}/Mesh/PeleLMeX_InteriorStretchMap.cpp
       ${SRC_DIR}/Mesh/PeleLMeX_MeshMappedCellConsInterp.H
       ${SRC_DIR}/Mesh/PeleLMeX_MeshMappedCellConsInterp.cpp
       ${SRC_DIR}/PeleLMeX_Temporals.cpp
       ${SRC_DIR}/PeleLMeX_Timestep.cpp
       ${SRC_DIR}/PeleLMeX_TransportProp.cpp
       ${SRC_DIR}/PeleLMeX_UMac.cpp
       ${SRC_DIR}/PeleLMeX_UserKeys.H
       ${SRC_DIR}/PeleLMeX_Utils.H
       ${SRC_DIR}/PeleLMeX_Utils.cpp
       ${SRC_DIR}/PeleLMeX_Index.H
       ${SRC_DIR}/PeleLMeX_ProblemSpecificFunctions.H
       ${SRC_DIR}/PeleLMeX_K.H
       ${SRC_DIR}/main.cpp
  )

  if(PELE_PHYSICS_ENABLE_SOOT)
    target_sources(${pele_exe_name}
      PRIVATE
        ${SRC_DIR}/PeleLMeX_Soot.cpp)
  endif()

  if(PELE_PHYSICS_ENABLE_SPRAY)
    target_sources(${pele_exe_name}
      PRIVATE
        ${SRC_DIR}/PeleLMeX_SprayParticles.cpp
        SprayParticlesInitInsert.cpp)
  endif()

  if(PELE_PHYSICS_ENABLE_RADIATION)
    target_sources(${pele_exe_name}
      PRIVATE
        ${SRC_DIR}/PeleLMeX_Radiation.cpp)
  endif()

  if(PELELM_ENABLE_PLASMA)
    set(PLASMA_DIR ${SRC_DIR}/Plasma)
    target_compile_definitions(${pele_exe_name} PRIVATE PELE_USE_PLASMA)
    target_include_directories(${pele_exe_name} PRIVATE ${PLASMA_DIR})
    target_include_directories(${pele_exe_name} PRIVATE ${PLASMA_DIR}/GMRES)
    target_include_directories(${pele_exe_name} PRIVATE ${PLASMA_DIR}/LinOps)
    target_sources(${pele_exe_name}
      PRIVATE
        ${PLASMA_DIR}/PeleLMeX_EF.H
        ${PLASMA_DIR}/PeleLMeX_EF_K.H
        ${PLASMA_DIR}/PeleLMeX_EF_Constants.H
        ${PLASMA_DIR}/PeleLMeX_EFDeriveFunc.H
        ${PLASMA_DIR}/PeleLMeX_EOS_Extension.H
        ${PLASMA_DIR}/PeleLMeX_EFDeriveFunc.cpp
        ${PLASMA_DIR}/PeleLMeX_EFIonDrift.cpp
        ${PLASMA_DIR}/PeleLMeX_EFNLSolve.cpp
        ${PLASMA_DIR}/PeleLMeX_EFPoisson.cpp
        ${PLASMA_DIR}/PeleLMeX_EFReactions.cpp
        ${PLASMA_DIR}/PeleLMeX_EFTimeStep.cpp
        ${PLASMA_DIR}/PeleLMeX_EFTransport.cpp
        ${PLASMA_DIR}/PeleLMeX_EFUtils.cpp
        ${PLASMA_DIR}/GMRES/MLGMRES.H
        ${PLASMA_DIR}/GMRES/MLGMRES.cpp
        ${PLASMA_DIR}/LinOps/PrecondOp.H
        ${PLASMA_DIR}/LinOps/PrecondOp.cpp
        ${PLASMA_DIR}/LinOps/AMReX_MLCellABecCecLap.H
        ${PLASMA_DIR}/LinOps/AMReX_MLCellABecCecLap.cpp
        ${PLASMA_DIR}/LinOps/AMReX_MLCellABecCecLap_K.H
        ${PLASMA_DIR}/LinOps/AMReX_MLCellABecCecLap_${PELE_DIM}D_K.H
        ${PLASMA_DIR}/LinOps/AMReX_MLABecCecLaplacian.H
        ${PLASMA_DIR}/LinOps/AMReX_MLABecCecLaplacian.cpp
        ${PLASMA_DIR}/LinOps/AMReX_MLABecCecLap_K.H
        ${PLASMA_DIR}/LinOps/AMReX_MLABecCecLap_${PELE_DIM}D_K.H)
  endif()

  if (PELELM_NUM_ODE GREATER 0)
    target_compile_definitions(${pele_exe_name} PRIVATE NUM_ODE=${PELELM_NUM_ODE})
  endif()

  if(NOT "${pele_exe_name}" STREQUAL "${PROJECT_NAME}-UnitTests")
    target_sources(${pele_exe_name}
       PRIVATE
         ${CMAKE_SOURCE_DIR}/Source/main.cpp
    )
  endif()

  if(PELE_ENABLE_CUDA)
    set(pctargets "${pele_exe_name};${pele_physics_lib_name}")
    foreach(tgt IN LISTS pctargets)
      get_target_property(PELE_SOURCES ${tgt} SOURCES)
      list(FILTER PELE_SOURCES INCLUDE REGEX "\\.cpp")
      set_source_files_properties(${PELE_SOURCES} PROPERTIES LANGUAGE CUDA)
    endforeach()
    set_target_properties(${pele_exe_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${pele_exe_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas --disable-optimizer-constants>)
  endif()

  target_link_libraries(${pele_exe_name} PRIVATE ${pele_physics_lib_name} AMReX-Hydro::amrex_hydro_api AMReX::amrex)

  install(TARGETS ${pele_exe_name}
          RUNTIME DESTINATION bin
          ARCHIVE DESTINATION lib
          LIBRARY DESTINATION lib)

endfunction()
