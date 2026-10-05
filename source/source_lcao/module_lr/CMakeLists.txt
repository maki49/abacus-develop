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
    lr_force_test.cpp
    grad_degen.cpp
    cal_edm.cpp
    zeqlin_solv.cpp)

  # BSE-related code and DMBand: only compiled with LibRI (__EXX). Unlike
  # operator_lr_exx.cpp, dm_band.cpp's own header includes <RI/global/Tensor.h>
  # unconditionally (not behind an __EXX guard), so it cannot even be
  # translated without LibRI; its only consumer is operator_lr_exx.cpp, which
  # is itself __EXX-only.
  if(ENABLE_LIBRI)
    list(APPEND objects
      utils/lr_io_krlist.cpp
      dm_band.cpp
    )
  endif()

add_library(
    lr
    OBJECT
    ${objects}
)

endif()
