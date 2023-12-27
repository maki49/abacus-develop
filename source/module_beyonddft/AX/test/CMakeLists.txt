AddTest(
  TARGET AX_test
  LIBS base ${math_libs} container device psi 
  SOURCES AX_test.cpp ../../../module_basis/module_ao/parallel_2d.cpp
)