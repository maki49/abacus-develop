if(ENABLE_MPI)
  set(test_zeqlin_solv_main ../../../source_base/test/mpi_test_main.cpp)
endif()
if(ENABLE_LIBXC)
  AddTest(
    TARGET MODULE_LR_pot_hxc_lrtd
    LIBS base parameter device container psi Libxc::xc
    SOURCES test_pot_hxc_lrtd.cpp ../potentials/pot_hxc_lrtd.cpp ../kernel_pots.cpp ../potentials/pot_grad_xc.cpp
      ../potentials/xc_kernel.cpp ../potentials/xc_kxc.cpp ../utils/lr_util.cpp
      ../../../source_hamilt/module_xc/xc_functional.cpp
      ../../../source_hamilt/module_xc/libxc_setup.cpp
  )

  AddTest(
    TARGET MODULE_LR_xc_kxc
    LIBS Libxc::xc
    SOURCES test_xc_kxc.cpp ../potentials/xc_kxc.cpp
  )
endif()

AddTest(
  TARGET MODULE_LR_preconditioner
  LIBS base parameter ${math_libs} container device psi
  SOURCES test_lr_preconditioner.cpp ../lr_preconditioner.cpp ../../../source_hamilt/operator.cpp
)

AddTest(
  TARGET MODULE_LR_davidson
  LIBS base parameter ${math_libs} container device psi
  SOURCES test_lr_davidson.cpp ../lr_davidson.cpp
    ../../../source_hsolver/diago_dav_subspace.cpp
    ../../../source_hsolver/diago_dav_subspace_residual.cpp
    ../../../source_hsolver/diag_const_nums.cpp
    ../../../source_hsolver/diag_hs_para.cpp
    ../../../source_hsolver/diago_pxxxgvx.cpp
)
if(TARGET ELPA::ELPA)
  target_link_libraries(MODULE_LR_davidson PRIVATE ELPA::ELPA genelpa)
endif()
if(ENABLE_MPI)
  add_test(NAME MODULE_LR_davidson_parallel
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:MODULE_LR_davidson> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(MODULE_LR_davidson_parallel PROPERTIES
    PROCESSORS 4 TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

AddTest(
  TARGET MODULE_LR_operator_lr_diag
  LIBS base parameter ${math_libs} container device psi
  SOURCES test_operator_lr_diag.cpp ../../../source_hamilt/operator.cpp
)

AddTest(
  TARGET MODULE_LR_zeq_cg_solver
  LIBS base parameter ${math_libs} container device
  SOURCES test_zeq_cg_solver.cpp ../zeq_cg_solver.cpp ../../../source_hsolver/linear_pcg.cpp
    ../../../source_hsolver/linear_gmres.cpp ../../../source_hsolver/linear_algebra.cpp
    ../../../source_hsolver/linear_workspace.cpp
)
if(ENABLE_MPI)
  target_sources(MODULE_LR_zeq_cg_solver PRIVATE ../../../source_base/test/mpi_test_main.cpp)
  add_test(NAME MODULE_LR_zeq_cg_solver_parallel
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:MODULE_LR_zeq_cg_solver> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(MODULE_LR_zeq_cg_solver_parallel PROPERTIES
    PROCESSORS 4 TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

AddTest(
  TARGET MODULE_LR_exx_projection
  LIBS base parameter ${math_libs} container device
  SOURCES test_exx_proj.cpp ../exx_proj.cpp
)

AddTest(
  TARGET MODULE_LR_grad_degen
  LIBS base parameter ${math_libs} container device
  SOURCES test_grad_degen.cpp test_grad_jt.cpp ../grad_degen.cpp ../grad_jt.cpp
)

if(ENABLE_MPI)
  AddTest(
    TARGET MODULE_LR_zeqlin_solv
    LIBS base parameter ${math_libs} container device psi MPI::MPI_CXX
    SOURCES test_zeqlin_solv.cpp ${test_zeqlin_solv_main} ../zeq_cg.cpp ../zeqlin_solv.cpp ../utils/lr_util.cpp
  )
  add_test(
    NAME MODULE_LR_zeqlin_failure_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:MODULE_LR_zeqlin_solv> ${MPIEXEC_POSTFLAGS}
      --gtest_filter=ZeqLinearSolverTest.*FailsCollectively*:ZeqCGReview.*
  )
  set_tests_properties(MODULE_LR_zeqlin_failure_mpi PROPERTIES
    PROCESSORS 4 TIMEOUT 60
    ENVIRONMENT "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()

AddTest(
  TARGET MODULE_LR_gradient_amplitudes
  LIBS base parameter ${math_libs} container device
  SOURCES test_lr_amp.cpp
)

# AO construction shares the existing two-center mock fixture with operator tests.
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
  ../../../source_hamilt/module_hcontainer/func_folding.cpp
)

AddTest(
  TARGET MODULE_LR_state_ovlp
  LIBS base parameter ${math_libs} container device psi
  SOURCES test_state_ovlp.cpp state_ovlp_test.h ${root_overlap_test_sources}
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
  TARGET MODULE_LR_state_group
  LIBS base parameter ${math_libs} container device
  SOURCES test_state_group.cpp ../state_group.cpp
)

set(state_track_test_sources test_state_track.cpp ../state_track.cpp ../state_group.cpp ../grad_degen.cpp
                             ${root_overlap_test_sources})
if(ENABLE_MPI)
  list(APPEND state_track_test_sources ../../../source_base/test/mpi_test_main.cpp)
endif()
AddTest(
  TARGET MODULE_LR_state_track
  LIBS base parameter ${math_libs} container device psi
  SOURCES ${state_track_test_sources}
)

set(cal_edm_test_sources test_cal_edm.cpp ../cal_edm.cpp ../utils/lr_util.cpp)
if(ENABLE_MPI)
  list(APPEND cal_edm_test_sources ../../../source_base/test/mpi_test_main.cpp)
endif()
AddTest(
  TARGET MODULE_LR_cal_edm
  LIBS base parameter ${math_libs} container device psi
  SOURCES ${cal_edm_test_sources}
)

# LR/DM adapter regressions belong to LR; estate must not depend on this module.
set(lr_dm_test_sources
  test_lr_util.cpp ../utils/lr_util.cpp ../utils/lr_sparse.cpp
  ../../spar_hsr.cpp
  ../../../source_io/module_hs/hs_sparse_io.cpp
  ../../../source_io/module_hs/lat_r_csr.cpp
  ../../../source_estate/module_dm/density_matrix.cpp
  ../../../source_estate/module_dm/dmr_gamma.cpp
  ../../../source_estate/module_dm/dmr_init.cpp
  ../../../source_estate/module_dm/dm_setter.cpp
  ../../../source_estate/module_dm/dm_getter.cpp
  ../../../source_estate/module_dm/dm_tools.cpp
  ../../../source_estate/module_dm/dmr_k.cpp
  ../../../source_estate/module_dm/dmr_td.cpp
  ../../../source_estate/module_dm/dmr_full.cpp
  ../../../source_estate/module_dm/unittests/tmp_mocks.cpp
  ../../../source_hamilt/module_hcontainer/base_matrix.cpp
  ../../../source_hamilt/module_hcontainer/hcontainer.cpp
  ../../../source_hamilt/module_hcontainer/atom_pair.cpp
  ../../../source_basis/module_ao/parallel_orbitals.cpp
  ../../../source_cell/klist.cpp
  ../../../source_cell/klist_io.cpp
  ../../../source_cell/parallel_kpoints.cpp
  ../../../source_cell/reciprocal_grid.cpp
)
if(ENABLE_MPI)
  list(APPEND lr_dm_test_sources ../../../source_base/test/mpi_test_main.cpp)
endif()
AddTest(
  TARGET MODULE_LR_dm_regressions
  LIBS parameter base device symmetry container
  SOURCES ${lr_dm_test_sources}
)

if(ENABLE_MPI)
  add_test(NAME MODULE_LR_sparse_rank_keys
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
      $<TARGET_FILE:MODULE_LR_dm_regressions>
      --gtest_filter=LRDMTest.SparseWriterKeepsSpinFilesAndDisjointRankKeys)
  set_tests_properties(MODULE_LR_sparse_rank_keys PROPERTIES PROCESSORS 4 TIMEOUT 60)
endif()

if(ENABLE_MPI)
  add_test(NAME MODULE_LR_empty_z_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:MODULE_LR_zeqlin_solv>
      --gtest_filter=ZeqCGReview.EmptyBlocksJoinOperatorCollectives)
  set_tests_properties(MODULE_LR_empty_z_mpi PROPERTIES PROCESSORS 4 TIMEOUT 60
    ENVIRONMENT "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()

if(ENABLE_MPI AND ENABLE_LIBRI)
  AddTest(
    TARGET MODULE_LR_ri_response
    LIBS base parameter ${math_libs} container device psi MPI::MPI_CXX
    SOURCES test_lr_ri_response.cpp ../lr_ri_response.cpp ../exx_proj.cpp
      ../dm_trans/dm_trans_serial.cpp ../../module_ri/ri_hartree_fock.cpp
      ../../../source_hamilt/operator.cpp ../../../source_base/test/mpi_test_main.cpp
  )
  add_test(NAME MODULE_LR_ri_response_parallel
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:MODULE_LR_ri_response> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(MODULE_LR_ri_response_parallel PROPERTIES
    PROCESSORS 4 TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
  if(ENABLE_LIBRI_GPU)
    target_link_libraries(MODULE_LR_ri_response PRIVATE DDLA::DDLA)
  endif()
  AddTest(
    TARGET MODULE_LR_exx_force_dm
    LIBS base parameter ${math_libs} container device MPI::MPI_CXX
    SOURCES test_exx_force_dm.cpp ../../../source_base/test/mpi_test_main.cpp
  )
  if(ENABLE_LIBRI_GPU)
    target_link_libraries(MODULE_LR_exx_force_dm PRIVATE DDLA::DDLA)
  endif()
endif()

AddTest(
  TARGET MODULE_LR_block_preconditioner
  LIBS base parameter ${math_libs} container device psi
  SOURCES test_lr_block_preconditioner.cpp ../lr_block_preconditioner.cpp
    ../lr_preconditioner.cpp ../../../source_hamilt/operator.cpp
)

AddTest(
  TARGET MODULE_LR_shifted_inverse
  LIBS base parameter ${math_libs} container device psi
  SOURCES test_lr_shifted_inverse.cpp ../lr_shifted_inverse.cpp
    ../lr_preconditioner.cpp ../../../source_hamilt/operator.cpp
)
if(ENABLE_MPI)
  add_test(
    NAME MODULE_LR_block_preconditioner_parallel
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:MODULE_LR_block_preconditioner>
      --gtest_filter=LRBlockPreconditioner.DistributedBlockIncludingEmptyRanks ${MPIEXEC_POSTFLAGS}
  )
  set_tests_properties(MODULE_LR_block_preconditioner_parallel PROPERTIES
    PROCESSORS 4 TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()
