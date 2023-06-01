remove_definitions(-D__DEEPKS)
remove_definitions(-D__CUDA)
remove_definitions(-D__ROCM)
AddTest(
  TARGET lr_util
  LIBS base ${math_libs} device
  SOURCES lr_util_physics_test.cpp
)
