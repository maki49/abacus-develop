remove_definitions(-DUSE_LIBXC)
AddTest(
  TARGET CVCX_test
  LIBS base ${math_libs} container device psi 
  SOURCES CVCX_test.cpp ../../../utils/lr_util.cpp ../CVCX_parallel.cpp ../CVCX_serial.cpp
  ../../../../module_basis/module_ao/parallel_2d.cpp
)