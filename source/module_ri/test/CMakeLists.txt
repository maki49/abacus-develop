remove_definitions(-D__DEEPKS)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)
AddTest(
  TARGET dm_mixing_test
  LIBS base ${math_libs} device
  SOURCES dm_mixing_test.cpp ../Mix_DMk_2D.cpp
)
AddTest(
  TARGET exx_symmetry_test
  LIBS base ${math_libs}  device orb psi
  SOURCES exx_symmetry_test.cpp ../exx_symmetry.cpp
)