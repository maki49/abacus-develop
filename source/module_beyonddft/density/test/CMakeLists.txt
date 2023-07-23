AddTest(
  TARGET dm_trans_test
  LIBS base ${math_libs} device container psi 
  SOURCES dm_trans_test.cpp ../../../module_basis/module_ao/parallel_2d.cpp
)