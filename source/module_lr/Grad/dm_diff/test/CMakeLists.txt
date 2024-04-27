remove_definitions(-DUSE_LIBXC)
AddTest(
  TARGET dm_diff_test
  LIBS psi base ${math_libs} device container
  SOURCES dm_diff_test.cpp ../../../utils/lr_util.cpp
  ../../../../module_basis/module_ao/parallel_2d.cpp
)