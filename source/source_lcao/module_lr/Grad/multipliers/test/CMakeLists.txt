if(ENABLE_MPI)
  if(TARGET ELPA::ELPA)
    AddTest(
      TARGET MODULE_LR_GRAD_zeq_linear_solver
      LIBS base parameter ${math_libs} container device psi ELPA::ELPA MPI::MPI_CXX
      SOURCES test_zeq_linear_solver.cpp ../zeq_linear_solver.cpp ../../../utils/lr_util.cpp
    )
  else()
    AddTest(
      TARGET MODULE_LR_GRAD_zeq_linear_solver
      LIBS base parameter ${math_libs} container device psi MPI::MPI_CXX
      SOURCES test_zeq_linear_solver.cpp ../zeq_linear_solver.cpp ../../../utils/lr_util.cpp
    )
  endif()
endif()
