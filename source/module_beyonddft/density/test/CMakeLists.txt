AddTest(
  TARGET dm_trans_test
  LIBS base ${math_libs} device psi orb
  SOURCES dm_trans_test.cpp
)