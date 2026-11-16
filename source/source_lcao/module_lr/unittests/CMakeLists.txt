if(ENABLE_LIBXC)
  addtest(
    TARGET MODULE_LR_pot_hxc_lrtd
    LIBS base parameter device container psi Libxc::xc
    SOURCES test_pot_hxc_lrtd.cpp
            ../potentials/pot_hxc_lrtd.cpp
            ../kernel_pots.cpp
            ../potentials/pot_grad_xc.cpp
            ../potentials/xc_kernel.cpp
            ../potentials/xc_kxc.cpp
            ../utils/lr_util.cpp
            ../../../source_hamilt/module_xc/xc_functional.cpp
            ../../../source_hamilt/module_xc/libxc_setup.cpp)

  addtest(
    TARGET MODULE_LR_xc_kernel
    LIBS base parameter device container psi Libxc::xc
    SOURCES test_xc_kernel.cpp
            ../potentials/xc_kxc.cpp
            ../potentials/xc_kernel.cpp
            ../utils/lr_util.cpp
            ../../../source_hamilt/module_xc/xc_functional.cpp
            ../../../source_hamilt/module_xc/libxc_setup.cpp)

  addtest(
    TARGET MODULE_LR_pot_grad_xc
    LIBS base parameter device container psi Libxc::xc
    SOURCES test_pot_grad_xc.cpp
            ../potentials/pot_grad_xc.cpp
            ../potentials/xc_kernel.cpp
            ../potentials/xc_kxc.cpp
            ../utils/lr_util.cpp
            ../../../source_hamilt/module_xc/xc_functional.cpp
            ../../../source_hamilt/module_xc/libxc_setup.cpp)

endif()

set(root_overlap_test_sources
    ../state_ovlp.cpp
    ../../module_operator_lcao/ovlp_block.cpp
    ../../module_operator_lcao/test/tmp_mocks.cpp
    ../../../source_basis/module_ao/parallel_orbitals.cpp
    ../../../source_basis/module_ao/orb_atomic_lm.cpp
    ../../../source_hamilt/operator.cpp
    ../../../source_hamilt/module_hcontainer/base_matrix.cpp
    ../../../source_hamilt/module_hcontainer/hcontainer.cpp
    ../../../source_hamilt/module_hcontainer/atom_pair.cpp
    ../../../source_hamilt/module_hcontainer/func_folding.cpp)

AddTest(
  TARGET MODULE_LR_preconditioner
  LIBS base parameter ${math_libs} container device psi
  SOURCES test_lr_preconditioner.cpp ../lr_preconditioner.cpp ../../../source_hamilt/operator.cpp
)

AddTest(
  TARGET MODULE_LR_operator_lr_diag
  LIBS base parameter ${math_libs} container device psi
  SOURCES test_operator_lr_diag.cpp ../../../source_hamilt/operator.cpp
)

AddTest(
  TARGET MODULE_LR_state_projection
  LIBS base parameter ${math_libs} container device psi
  SOURCES test_state_projection.cpp ../state_projection.cpp ${root_overlap_test_sources}
)
if(ENABLE_MPI)
  target_sources(MODULE_LR_state_projection PRIVATE ../../../source_base/test/mpi_test_main.cpp)
  add_test(NAME MODULE_LR_state_projection_parallel
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:MODULE_LR_state_projection> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(MODULE_LR_state_projection_parallel PROPERTIES
    PROCESSORS 4 TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

AddTest(
  TARGET MODULE_LR_state_guess
  LIBS base parameter ${math_libs} container device psi
  SOURCES test_state_guess.cpp ../state_guess.cpp ../state_projection.cpp ${root_overlap_test_sources}
)
if(ENABLE_MPI)
  target_sources(MODULE_LR_state_guess PRIVATE ../../../source_base/test/mpi_test_main.cpp)
  add_test(NAME MODULE_LR_state_guess_parallel
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:MODULE_LR_state_guess> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(MODULE_LR_state_guess_parallel PROPERTIES
    PROCESSORS 4 TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()
