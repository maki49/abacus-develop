AddTest(
  TARGET MODULE_LR_exx_projection
  LIBS base parameter ${math_libs} container device
  SOURCES test_exx_proj.cpp ../exx_proj.cpp
)

AddTest(
  TARGET MODULE_LR_grad_degen
  LIBS base parameter ${math_libs} container device
  SOURCES test_grad_degen.cpp test_grad_jt.cpp ../grad_degen.cpp ../grad_jt.cpp
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

AddTest(
  TARGET MODULE_LR_gradient_amplitudes
  LIBS base parameter ${math_libs} container device
  SOURCES test_lr_amp.cpp
)
