add_subdirectory(dm_diff)
add_subdirectory(CVCX)

add_library(
lr_grad
OBJECT
CVCX/CVCX_parallel.cpp
CVCX/CVCX_serial.cpp
xc/kernel_grad_xc.cpp
)