#=============================================================================
# PeleLMeX Testing
#=============================================================================

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_BINARY_DIR}/CTestCustom.cmake)

if(PELE_ENABLE_FCOMPARE_FOR_TESTS)
  if("${PELE_REFERENCE_GOLDS_DIRECTORY}" STREQUAL "")
    message(FATAL_ERROR "To reference gold files, PELE_REFERENCE_GOLDS_DIRECTORY must be set and exist")
  else()
    if(EXISTS ${PELE_REFERENCE_GOLDS_DIRECTORY})
      set(GOLD_FILES_DIRECTORY ${PELE_REFERENCE_GOLDS_DIRECTORY}/${CMAKE_SYSTEM_NAME}/${CMAKE_CXX_COMPILER_ID}/${CMAKE_CXX_COMPILER_VERSION})
      message(STATUS "Test golds directory for fcompare: ${GOLD_FILES_DIRECTORY}")
    else()
      message(FATAL_ERROR "Specified directory for reference gold files does not exist: ${PELE_REFERENCE_GOLDS_DIRECTORY}")
    endif()
  endif()
endif()

if(PELE_SAVE_GOLDS)
  if("${PELE_SAVED_GOLDS_DIRECTORY}" STREQUAL "")
    message(FATAL_ERROR "To save gold files, PELE_SAVED_GOLDS_DIRECTORY must be set and exist")
  else()
    if(EXISTS ${PELE_SAVED_GOLDS_DIRECTORY})
      set(SAVED_GOLDS_DIRECTORY ${PELE_SAVED_GOLDS_DIRECTORY}/${CMAKE_SYSTEM_NAME}/${CMAKE_CXX_COMPILER_ID}/${CMAKE_CXX_COMPILER_VERSION})
      message(STATUS "Gold files will be saved to: ${SAVED_GOLDS_DIRECTORY}")
    else()
      message(FATAL_ERROR "Specified directory for saving gold files does not exist: ${PELE_SAVED_GOLDS_DIRECTORY}")
    endif()
  endif()
endif()

#=============================================================================
# Functions for adding tests / Categories of tests
#=============================================================================

macro(setup_test)
    # Set variables for respective binary and source directories for the test
    set(CURRENT_TEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/Exec/RegTests/${TEST_EXE_DIR})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_BINARY_DIR}/Exec/RegTests/${TEST_EXE_DIR}/${TEST_NAME})
    set(CURRENT_TEST_EXE ${CMAKE_BINARY_DIR}/Exec/RegTests/${TEST_EXE_DIR}/${PROJECT_NAME}-${TEST_EXE_DIR})
    # Gold files should be submodule organized by machine and compiler (these are output during configure)
    set(PLOT_GOLD ${GOLD_FILES_DIRECTORY}/${TEST_EXE_DIR}/${TEST_NAME}/plt00010)
    # Test plot is currently expected to be after 10 steps
    set(PLOT_TEST ${CURRENT_TEST_BINARY_DIR}/plt00010)
    # Find fcompare
    if(PELE_ENABLE_FCOMPARE_FOR_TESTS)
      set(FCOMPARE ${CMAKE_BINARY_DIR}/Submodules/PelePhysics/Submodules/amrex/Tools/Plotfile/amrex_fcompare)
    endif()
    # Make working directory for test
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    # Stage case data not covered by the .dat/.py glob below — typically
    # subdirectories (ref_files), files with other extensions, etc.
    # Symlink from the case source folder into the per-test binary dir,
    # copying as a fallback if the platform refuses the symlink.
    foreach(_extra IN LISTS ATR_EXTRA_FILES)
      set(_src "${CURRENT_TEST_SOURCE_DIR}/${_extra}")
      set(_dst "${CURRENT_TEST_BINARY_DIR}/${_extra}")
      if(NOT EXISTS "${_src}")
        message(FATAL_ERROR
          "add_test_r(${TEST_NAME}): EXTRA_FILES entry '${_extra}' "
          "not found at '${_src}'.")
      endif()
      file(REMOVE_RECURSE "${_dst}")
      get_filename_component(_dst_dir "${_dst}" DIRECTORY)
      file(MAKE_DIRECTORY "${_dst_dir}")
      file(CREATE_LINK "${_src}" "${_dst}" SYMBOLIC RESULT _link_rc)
      if(NOT IS_SYMLINK "${_dst}")
        file(COPY "${_src}" DESTINATION "${_dst_dir}")
      endif()
    endforeach()
    # Gather all files in source directory for test
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*.dat" "${CURRENT_TEST_SOURCE_DIR}/*.py")
    # Copy files to test working directory
    file(COPY ${CURRENT_TEST_SOURCE_DIR}/${TEST_NAME}.inp DESTINATION "${CURRENT_TEST_BINARY_DIR}/")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")
    # Set some default runtime options for all tests
    set(RUNTIME_OPTIONS "amrex.abort_on_unused_inputs=1 amrex.the_arena_is_managed=0 amrex.the_arena_init_size=0 amr.plot_per=-1 amr.check_per=-1 amr.plot_int=-1 amr.check_int=-1")
    if(PELE_ENABLE_FCOMPARE_FOR_TESTS OR PELE_SAVE_GOLDS)
      set(RUNTIME_OPTIONS "${RUNTIME_OPTIONS} amr.plot_file=plt amr.plot_int=10")
    endif()
    if(PELE_ENABLE_FPE_TRAP_FOR_TESTS AND (NOT APPLE))
      set(RUNTIME_OPTIONS "${RUNTIME_OPTIONS} amrex.signal_handling=1 amrex.fpe_trap_invalid=1 amrex.fpe_trap_zero=1 amrex.fpe_trap_overflow=1")
    else()
      set(RUNTIME_OPTIONS "${RUNTIME_OPTIONS} amrex.signal_handling=0")
    endif()
    if(PELE_ENABLE_MPI)
      if(PELE_ENABLE_CUDA)
        set(PELE_NP 2) # 1 rank per GPU on Eagle
      else()
        set(PELE_NP 4)
      endif()
      set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${PELE_NP} ${MPIEXEC_PREFLAGS}")
    else()
      set(PELE_NP 1)
      unset(MPI_COMMANDS)
    endif()
    # Use fcompare to test diffs in plots against gold files
    if(PELE_ENABLE_FCOMPARE_FOR_TESTS AND (NOT "${TEST_NAME}" MATCHES "hdf5$"))
      if(PELE_ENABLE_CUDA)
        set(FCOMPARE_TOLERANCE "-r 1e-12 --abs_tol 1.0e-12")
      endif()
      set(FCOMPARE_COMMAND "&& ${MPI_COMMANDS} ${FCOMPARE} ${FCOMPARE_TOLERANCE} ${PLOT_TEST} ${PLOT_GOLD}")
    endif()
    if(PELE_SAVE_GOLDS)
      file(MAKE_DIRECTORY ${SAVED_GOLDS_DIRECTORY}/${TEST_EXE_DIR}/${TEST_NAME})
      set(SAVE_GOLDS_COMMAND "&& cp -R ${PLOT_TEST} ${SAVED_GOLDS_DIRECTORY}/${TEST_EXE_DIR}/${TEST_NAME}/")
    endif()
endmacro(setup_test)

# Standard regression test
function(add_test_r TEST_NAME TEST_EXE_DIR)
    cmake_parse_arguments(ATR "" "RESTART_STEP" "EXTRA_FILES" ${ARGN})

    setup_test()
    set(RUNTIME_OPTIONS "amr.max_step=10 ${RUNTIME_OPTIONS}")

    if(DEFINED ATR_RESTART_STEP)
        # Two-phase restart variant:
        #   phase 1 runs RESTART_STEP steps and writes a checkpoint;
        #   phase 2 restarts from that checkpoint and runs through step 10,
        #   with fcompare applied to plt00010 from the restarted run.
        # Zero-pad RESTART_STEP to AMReX's 5-digit chk format (chkNNNNN).
        set(_padded "0000${ATR_RESTART_STEP}")
        string(LENGTH "${_padded}" _len)
        math(EXPR _start "${_len} - 5")
        string(SUBSTRING "${_padded}" ${_start} 5 _step_padded)

        set(P1_OPTS "amr.max_step=${ATR_RESTART_STEP} amr.check_int=${ATR_RESTART_STEP} amr.plot_int=-1 amr.check_file=chk")
        set(P2_OPTS "amr.restart=chk${_step_padded} amr.max_step=10 amr.check_int=-1")

        add_test(${TEST_NAME} bash -c "set -o pipefail \
&& ${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.inp ${RUNTIME_OPTIONS} ${P1_OPTS} 2>&1 | tee ${TEST_NAME}.phase1.log \
&& ${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.inp ${RUNTIME_OPTIONS} ${P2_OPTS} 2>&1 | tee ${TEST_NAME}.phase2.log ${SAVE_GOLDS_COMMAND} ${FCOMPARE_COMMAND}")

        set_tests_properties(${TEST_NAME} PROPERTIES
            TIMEOUT 18000
            PROCESSORS ${PELE_NP}
            WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
            LABELS "regression"
            ATTACHED_FILES "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.phase1.log;${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.phase2.log")
    else()
        add_test(${TEST_NAME} bash -c "set -o pipefail && ${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.inp ${RUNTIME_OPTIONS} 2>&1 | tee ${TEST_NAME}.log ${SAVE_GOLDS_COMMAND} ${FCOMPARE_COMMAND}")

        set_tests_properties(${TEST_NAME} PROPERTIES
            TIMEOUT 18000
            PROCESSORS ${PELE_NP}
            WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
            LABELS "regression"
            ATTACHED_FILES "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
    endif()
endfunction(add_test_r)

# Regression test with mass conservation verification
# Mesh-mapping round-off equivalence: run TEST_NAME.inp (a ConstantMap run
# with factors FAC, comma separated) and TEST_NAME_ref.inp (the unmapped run
# of the same physical problem on the same index grid) and require their
# final plotfiles to agree field by field to TOL
# (Exec/RegTests/MeshMappingConvergence/compare_plotfiles.py).  1e-10 for a
# non-reacting case; a reacting case needs ~1e-8 because CVODE's adaptive
# stepping amplifies round-off differences in the incoming state.
function(add_test_mapped_pair TEST_NAME TEST_EXE_DIR FAC TOL)
    setup_test()
    file(COPY ${CURRENT_TEST_SOURCE_DIR}/${TEST_NAME}_ref.inp DESTINATION "${CURRENT_TEST_BINARY_DIR}/")
    set(COMPARE_PLOTFILES ${CMAKE_SOURCE_DIR}/Exec/RegTests/MeshMappingConvergence/compare_plotfiles.py)
    # The inputs fix max_step and plot_int; put them after RUNTIME_OPTIONS so
    # they win over the harness defaults (ParmParse keeps the last value).
    set(PAIR_OPTS "amr.max_step=5 amr.plot_int=5 amr.check_int=-1")
    add_test(${TEST_NAME} bash -c "set -o pipefail \
&& ${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}_ref.inp ${RUNTIME_OPTIONS} ${PAIR_OPTS} amr.plot_file=pltref 2>&1 | tee ${TEST_NAME}.ref.log \
&& ${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.inp ${RUNTIME_OPTIONS} ${PAIR_OPTS} amr.plot_file=pltmap 2>&1 | tee ${TEST_NAME}.log \
&& python3 ${COMPARE_PLOTFILES} pltref00005 pltmap00005 --fac ${FAC} --tol ${TOL}")
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 18000
        PROCESSORS ${PELE_NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression"
        ATTACHED_FILES "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.ref.log;${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
endfunction(add_test_mapped_pair)

function(add_test_rv TEST_NAME TEST_EXE_DIR)
    setup_test()
    set(RUNTIME_OPTIONS "amr.max_step=10 peleLM.do_temporals=1 peleLM.do_mass_balance=1 peleLM.temporal_int=1 ${RUNTIME_OPTIONS}")
    add_test(${TEST_NAME} bash -c "set -o pipefail && rm -f datlog && ${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.inp ${RUNTIME_OPTIONS} 2>&1 | tee ${TEST_NAME}.log ${SAVE_GOLDS_COMMAND} ${FCOMPARE_COMMAND} && nosetests ${CMAKE_CURRENT_SOURCE_DIR}/test_masscons.py")
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 18000 PROCESSORS ${PELE_NP} WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/" LABELS "regression;verification" ATTACHED_FILES "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
endfunction(add_test_rv)

# Regression tests with mass conservation excluded from CI
function(add_test_rev TEST_NAME TEST_EXE_DIR)
    add_test_rv(${TEST_NAME} ${TEST_EXE_DIR})
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS "regression;verification;no-ci")
endfunction(add_test_rev)

# Regression test with case-specific verification test.py
function(add_test_rt TEST_NAME TEST_EXE_DIR)
    setup_test()
    set(RUNTIME_OPTIONS "amr.max_step=10 peleLM.do_temporals=1 peleLM.do_extremas=1 peleLM.temporal_int=1 ${RUNTIME_OPTIONS}")
    add_test(${TEST_NAME} bash -c "set -o pipefail && ${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.inp ${RUNTIME_OPTIONS} 2>&1 | tee ${TEST_NAME}.log ${SAVE_GOLDS_COMMAND} ${FCOMPARE_COMMAND} && nosetests ${CURRENT_TEST_BINARY_DIR}/test.py")
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 18000 PROCESSORS ${PELE_NP} WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/" LABELS "regression;verification" ATTACHED_FILES "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
endfunction(add_test_rt)

# Regression tests excluded from CI
function(add_test_re TEST_NAME TEST_EXE_DIR)
    add_test_r(${TEST_NAME} ${TEST_EXE_DIR})
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS "regression;no-ci")
endfunction(add_test_re)

# Regression tests expected to fail
function(add_test_rf TEST_NAME TEST_EXE_DIR)
    add_test_r(${TEST_NAME} ${TEST_EXE_DIR})
    set_tests_properties(${TEST_NAME} PROPERTIES WILL_FAIL TRUE)
endfunction(add_test_rf)

# Verification test with 1 resolution
function(add_test_v1 TEST_NAME TEST_EXE_DIR)
    setup_test()
    set(RUN_COMMAND "rm -f mmslog datlog && ${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.inp")
    add_test(${TEST_NAME} bash -c "set -o pipefail && ${RUN_COMMAND} ${RUNTIME_OPTIONS} 2>&1 | tee ${TEST_NAME}.log && nosetests ${TEST_NAME}.py")
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 18000 PROCESSORS ${PELE_NP} WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/" LABELS "verification" ATTACHED_FILES "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log")
endfunction(add_test_v1)

# Verification test with multiple resolutions (each test runs on maximum number of processes on node)
function(add_test_v2 TEST_NAME TEST_EXE_DIR LIST_OF_GRID_SIZES)
    setup_test()
    unset(MASTER_RUN_COMMAND)
    # Get last item in resolution list so we can find out when we are on the last item in our loop
    list(GET LIST_OF_GRID_SIZES -1 LAST_GRID_SIZE_IN_LIST)
    # Create the commands to run for each resolution
    foreach(GRID_SIZE IN LISTS LIST_OF_GRID_SIZES)
      file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR}/${GRID_SIZE})
      file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*.dat" "${CURRENT_TEST_SOURCE_DIR}/*.py")
      file(COPY ${CURRENT_TEST_SOURCE_DIR}/${TEST_NAME}.inp DESTINATION "${CURRENT_TEST_BINARY_DIR}/${GRID_SIZE}/")
      file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/${GRID_SIZE}/")
      if(${PELE_DIM} EQUAL 3)
        set(NCELLS "${GRID_SIZE} ${GRID_SIZE} ${GRID_SIZE}")
      elseif(${PELE_DIM} EQUAL 2)
        set(NCELLS "${GRID_SIZE} ${GRID_SIZE}")
      elseif(${PELE_DIM} EQUAL 1)
        set(NCELLS "${GRID_SIZE}")
      endif()
      set(DELETE_PREVIOUS_FILES_COMMAND "rm -f mmslog datlog")
      set(RUN_COMMAND_${GRID_SIZE} "${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS} ${CURRENT_TEST_BINARY_DIR}/${GRID_SIZE}/${TEST_NAME}.inp")
      set(RUNTIME_OPTIONS_${GRID_SIZE} "${RUNTIME_OPTIONS} amr.n_cell=${NCELLS}")
      string(APPEND MASTER_RUN_COMMAND "cd ${CURRENT_TEST_BINARY_DIR}/${GRID_SIZE}")
      string(APPEND MASTER_RUN_COMMAND " && ")
      string(APPEND MASTER_RUN_COMMAND "${DELETE_PREVIOUS_FILES_COMMAND}")
      string(APPEND MASTER_RUN_COMMAND " && ")
      string(APPEND MASTER_RUN_COMMAND "set -o pipefail && ${RUN_COMMAND_${GRID_SIZE}} ${RUNTIME_OPTIONS_${GRID_SIZE}} 2>&1 | tee ${TEST_NAME}-${GRID_SIZE}.log")
      # Add another " && " unless we are on the last resolution in the list
      if(NOT ${GRID_SIZE} EQUAL ${LAST_GRID_SIZE_IN_LIST})
        string(APPEND MASTER_RUN_COMMAND " && ")
      endif()
    endforeach()
    set(IMAGES_TO_UPLOAD ${CURRENT_TEST_BINARY_DIR}/p_error.png ${CURRENT_TEST_BINARY_DIR}/rho_error.png ${CURRENT_TEST_BINARY_DIR}/u_error.png)
    if(${PELE_DIM} EQUAL 3)
      list(APPEND IMAGES_TO_UPLOAD ${CURRENT_TEST_BINARY_DIR}/v_error.png ${CURRENT_TEST_BINARY_DIR}/w_error.png)
    elseif(${PELE_DIM} EQUAL 2)
      list(APPEND IMAGES_TO_UPLOAD ${CURRENT_TEST_BINARY_DIR}/v_error.png)
    endif()
    add_test(${TEST_NAME} bash -c "${MASTER_RUN_COMMAND} && cd ${CURRENT_TEST_BINARY_DIR} && nosetests ${CMAKE_CURRENT_SOURCE_DIR}/test_second_order.py")
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 18000 PROCESSORS ${PELE_NP} WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}" LABELS "verification;no-ci" ATTACHED_FILES "${IMAGES_TO_UPLOAD}")
endfunction(add_test_v2)

# Standard unit test
function(add_test_u TEST_NAME)
    set(CURRENT_TEST_EXE ${CMAKE_BINARY_DIR}/Exec/UnitTests/${PROJECT_NAME}-UnitTests)
    set(CURRENT_TEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/Exec/UnitTests/${TEST_EXE_DIR})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_BINARY_DIR}/Exec/UnitTests/${TEST_EXE_DIR})
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    set(PELE_NP 1)
    if(PELE_ENABLE_MPI)
      set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${PELE_NP} ${MPIEXEC_PREFLAGS}")
    else()
      unset(MPI_COMMANDS)
    endif()
    add_test(${TEST_NAME} bash -c "${MPI_COMMANDS} ${CURRENT_TEST_EXE} ${MPIEXEC_POSTFLAGS}")
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 1800 PROCESSORS ${PELE_NP} WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/" LABELS "unit")
endfunction(add_test_u)

function(add_test_spray TEST_EXE_DIR)
    set(TEST_NAME ${TEST_EXE_DIR})
    # Set variables for respective binary and source directories for the test
    set(CURRENT_TEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/Exec/RegTests/${TEST_EXE_DIR})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_BINARY_DIR}/Exec/RegTests/${TEST_EXE_DIR})
    set(CURRENT_TEST_EXE ${CMAKE_BINARY_DIR}/Exec/RegTests/${TEST_EXE_DIR}/${PROJECT_NAME}-${TEST_EXE_DIR})
    if(PELE_ENABLE_MPI)
      if(PELE_ENABLE_CUDA)
        set(PELE_NP 2) # 1 rank per GPU on Eagle
      else()
        set(PELE_NP 4)
      endif()
      set(MPI_COMMANDS "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${PELE_NP} ${MPIEXEC_PREFLAGS}")
    else()
      set(PELE_NP 1)
      unset(MPI_COMMANDS)
    endif()
    set(FCOMPARE ${CMAKE_BINARY_DIR}/Submodules/PelePhysics/Submodules/amrex/Tools/Plotfile/amrex_fcompare)
    set(MULTIRUN_FLAGS "--test_dir ${CURRENT_TEST_BINARY_DIR} --run_cmd '${MPI_COMMANDS}' --pele_exec ${PROJECT_NAME}-${TEST_EXE_DIR} --fcomp_exec ${FCOMPARE} --input_file ${TEST_NAME}.inp")
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*.dat" "${CURRENT_TEST_SOURCE_DIR}/*.py")
    # Copy files to test working directory
    file(COPY ${CURRENT_TEST_SOURCE_DIR}/${TEST_NAME}.inp DESTINATION "${CURRENT_TEST_BINARY_DIR}/")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")
    add_test(${TEST_NAME} bash -c "python3 ${CURRENT_TEST_BINARY_DIR}/multiRuns.py ${MULTIRUN_FLAGS}")
endfunction(add_test_spray)

#=============================================================================
# Regression tests
#=============================================================================

if(NOT PELE_ENABLE_EB)
  add_test_r(flamesheet-drm19-${PELE_DIM}d FlameSheet)
  add_test_r(hotbubble-${PELE_DIM}d HotBubble)
  add_test_r(passive-scalar-${PELE_DIM}d PassiveScalar)
  add_test_rv(covo-${PELE_DIM}d PeriodicCases)
  add_test_rv(enclosedflame-${PELE_DIM}d EnclosedFlame)
  if(PELE_ENABLE_RADIATION)
    add_test_r(soot-rad-${PELE_DIM}d SootRadTest)
  endif()
  if(PELE_ENABLE_PARTICLES)
    add_test_r(single-drop-evap-gcm-test SingleDropEvap EXTRA_FILES ref_files)
  endif()
  if(PELE_DIM EQUAL 2)
    add_test_r(tripleflame-${PELE_DIM}d TripleFlame)
    add_test_rt(isothermal-soret IsothermalSoretChannel)
    add_test_mapped_pair(flamesheet-mapped-${PELE_DIM}d FlameSheet "2,1" 1e-8)
  endif()
  if(PELE_DIM EQUAL 3)
    add_test_r(hit-decay-${PELE_DIM}d HIT)
    add_test_r(hit-decay-les-${PELE_DIM}d HIT)
    add_test_r(hit-forced-${PELE_DIM}d HIT)
  endif()
else()
  if(PELE_DIM EQUAL 2)
    add_test_rt(composition-test-${PELE_DIM}d EB_ODEQty)
    add_test_r(eb_bfs EB_BackwardStepFlame)
    add_test_r(eb_bfs_pp EB_BackwardStepFlame)
    # Recycling-plane inflow capability (PeleLM::fillFromRecyclingPlane)
    add_test_r(eb_bfs_recycle_off     EB_BackwardStepFlame)
    add_test_r(eb_bfs_recycle_basic   EB_BackwardStepFlame)
    add_test_r(eb_bfs_recycle_warmup  EB_BackwardStepFlame)
    add_test_r(eb_bfs_recycle_window  EB_BackwardStepFlame)
    add_test_r(eb_bfs_recycle_amr     EB_BackwardStepFlame)
    add_test_r(eb_bfs_recycle_restart EB_BackwardStepFlame RESTART_STEP 5)
  endif()
  add_test_r(eb-odeqty-${PELE_DIM}d EB_ODEQty)
endif()
