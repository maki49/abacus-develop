abacus_disable_feature_definitions(__MLALGO)
abacus_disable_feature_definitions(__CUDA)
abacus_disable_feature_definitions(__ROCM)
if(ENABLE_LCAO)
  AddTest(
    TARGET MODULE_RI_minimal_gaussian_ri
    LIBS base parameter device orb ${math_libs}
    SOURCES test_minimal_gaussian_ri.cpp ../minimal_gaussian_ri.cpp ../minimal_gaussian_basis.cpp
      ../ri_hartree_fock.cpp ../conv_coulomb_pot_k.cpp ../exx_abfs_ctor_orbs.cpp
      ../abfs_construct_pca.cpp ../matrix_orbs11.cpp ../matrix_orbs21.cpp
      ../../center2orb.cpp ../../center2orb_orb11.cpp ../../center2orb_orb21.cpp
      ../../../source_cell/unitcell.cpp ../../../source_cell/atom_spec.cpp
      ../../../source_cell/atom_pseudo.cpp ../../../source_cell/pseudo.cpp
      ../../../source_cell/sep.cpp ../../../source_cell/sep_cell.cpp ../../../source_cell/magnetism.cpp
      ../../../source_base/test/mpi_test_main.cpp
  )
  target_compile_definitions(MODULE_RI_minimal_gaussian_ri PRIVATE
    RI_TEST_ORBITAL_DIR="${PROJECT_SOURCE_DIR}/source/source_basis/module_ao/test/lcao_H2O")
  add_test(NAME MODULE_RI_minimal_gaussian_ri_parallel
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:MODULE_RI_minimal_gaussian_ri> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(MODULE_RI_minimal_gaussian_ri MODULE_RI_minimal_gaussian_ri_parallel PROPERTIES
    TIMEOUT 240 ENVIRONMENT "OMP_NUM_THREADS=1")
  set_tests_properties(MODULE_RI_minimal_gaussian_ri_parallel PROPERTIES PROCESSORS 4)
  AddTest(
    TARGET MODULE_RI_ri_hartree_fock
    LIBS base parameter device ${math_libs}
    SOURCES test_ri_hartree_fock.cpp ../ri_hartree_fock.cpp ../../../source_base/test/mpi_test_main.cpp
  )
  add_test(NAME MODULE_RI_ri_hartree_fock_parallel
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:MODULE_RI_ri_hartree_fock> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(MODULE_RI_ri_hartree_fock_parallel PROPERTIES
    PROCESSORS 4 TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
  if(ENABLE_LIBRI_GPU)
    target_link_libraries(MODULE_RI_minimal_gaussian_ri PRIVATE DDLA::DDLA)
    target_link_libraries(MODULE_RI_ri_hartree_fock PRIVATE DDLA::DDLA)
  endif()
  AddTest(
    TARGET MODULE_RI_minimal_gaussian_basis
    LIBS base parameter device ${math_libs}
    SOURCES test_minimal_gaussian_basis.cpp ../minimal_gaussian_basis.cpp
            ../../../source_basis/module_ao/orb_atomic_lm.cpp
  )
  AddTest(
    TARGET MODULE_RI_exx_workspace
    LIBS base parameter device ${math_libs}
    SOURCES test_exx_workspace.cpp ../exx_workspace.cpp ../../../source_basis/module_ao/orb_atomic_lm.cpp
  )
endif()
AddTest(
  TARGET MODULE_RI_dm_mixing_test
  LIBS parameter base device
  SOURCES test_mix_dmk_2d.cpp ../mix_dmk_2d.cpp
)
AddTest(
  TARGET MODULE_RI_ri_cv_io_test
  LIBS base device parameter
  SOURCES test_write_ri_cv.cpp
)
AddTest(
  TARGET MODULE_RI_abfs_vector3_order
  LIBS parameter
  SOURCES test_abfs_vector3_order.cpp
)
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/support"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
