AddTest(
  TARGET lr_util_phys_test
  LIBS base ${math_libs} device 
  SOURCES lr_util_physics_test.cpp ../../../module_basis/module_ao/parallel_2d.cpp
)