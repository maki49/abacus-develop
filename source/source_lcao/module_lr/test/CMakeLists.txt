AddTest(
  TARGET MODULE_LR_zeq_cg_solver
  LIBS base parameter ${math_libs} container device
  SOURCES test_zeq_cg_solver.cpp ../zeq_cg_solver.cpp ../../../source_hsolver/linear_pcg.cpp
)

AddTest(
  TARGET MODULE_LR_exx_projection
  LIBS base parameter ${math_libs} container device
  SOURCES test_exx_projection.cpp ../exx_projection.cpp
)

AddTest(
  TARGET MODULE_LR_grad_degen
  LIBS base parameter ${math_libs} container device
  SOURCES test_grad_degen.cpp ../grad_degen.cpp
)

if(ENABLE_MPI)
  if(TARGET ELPA::ELPA)
    AddTest(
      TARGET MODULE_LR_zeqlin_solv
      LIBS base parameter ${math_libs} container device psi ELPA::ELPA MPI::MPI_CXX
      SOURCES test_zeqlin.cpp ../zeqlin_solv.cpp ../utils/lr_util.cpp
    )
  else()
    AddTest(
      TARGET MODULE_LR_zeqlin_solv
      LIBS base parameter ${math_libs} container device psi MPI::MPI_CXX
      SOURCES test_zeqlin.cpp ../zeqlin_solv.cpp ../utils/lr_util.cpp
    )
  endif()
endif()
