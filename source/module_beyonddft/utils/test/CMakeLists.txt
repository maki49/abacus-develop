AddTest(
  TARGET lr_util_phys_test
  LIBS base ${math_libs} device orb
  SOURCES lr_util_physics_test.cpp
)