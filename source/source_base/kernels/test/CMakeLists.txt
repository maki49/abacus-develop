abacus_disable_feature_definitions(__MPI)

addtest(
  TARGET MODULE_BASE_KERNELS_Unittests
  LIBS parameter base device
  SOURCES math_ylm_op_test.cpp ../unittests/test_math_kernel_op.cpp)
