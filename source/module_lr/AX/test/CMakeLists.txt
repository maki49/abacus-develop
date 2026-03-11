remove_definitions(-DUSE_LIBXC)
AddTest(
  TARGET AX_test
  LIBS parameter base ${math_libs} container device psi 
  SOURCES AX_test.cpp ../../utils/lr_util.cpp ../AX_parallel.cpp ../AX_serial.cpp
)

if(USE_ELPA)
  target_link_libraries(AX_test ELPA::ELPA genelpa)
endif()
