if(ENABLE_LCAO)
  add_subdirectory(utils)
  add_subdirectory(ao_to_mo_transformer)
  add_subdirectory(dm_trans)
  add_subdirectory(ri_benchmark)
  if(BUILD_TESTING)
    add_subdirectory(test)
  endif()

  list(APPEND objects
    utils/lr_util.cpp
    utils/lr_io.cpp
    utils/exciton_plotter.cpp
    ao_to_mo_transformer/ao_to_mo_parallel.cpp
    ao_to_mo_transformer/ao_to_mo_serial.cpp
    ao_to_mo_transformer/CVCX_par.cpp
    ao_to_mo_transformer/CVCX_serial.cpp
    dm_trans/dm_trans_parallel.cpp
    dm_trans/dm_trans_serial.cpp
    operator_casida/operator_lr_hxc.cpp
    operator_casida/operator_lr_exx.cpp
    potentials/pot_hxc_lrtd.cpp
    potentials/pot_grad_xc.cpp
    lr_spectrum.cpp
    lr_spectrum_velocity.cpp
    hamilt_casida.cpp
    potentials/xc_kernel.cpp
    lr_force.cpp
    exx_projection.cpp
    lr_force_test.cpp
    grad_degen.cpp
    cal_edm.cpp
    zeqlin_solv.cpp
    zeq_cg_solver.cpp)

  # The RI k-point list reader requires LibRI headers.
  if(ENABLE_LIBRI)
    list(APPEND objects utils/lr_io_krlist.cpp)
  endif()

add_library(
    lr
    OBJECT
    ${objects}
)

endif()
