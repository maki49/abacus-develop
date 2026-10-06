# LibRI's GPU header backend needs an explicitly installed DDLA library.
# DDLA public headers use C++17; CPU-only builds retain their existing standard.
set_if_higher(CMAKE_CXX_STANDARD 17)
set_if_higher(CMAKE_CUDA_STANDARD 17)
# Do not fetch an unpinned backend or silently fall back to CPU contractions.
find_path(DDLA_INCLUDE_DIR ddla/ddla_connector.h
  HINTS ${DDLA_ROOT} ENV DDLA_ROOT PATH_SUFFIXES include)
find_library(DDLA_LIBRARY NAMES ddla
  HINTS ${DDLA_ROOT} ENV DDLA_ROOT PATH_SUFFIXES lib lib64)
if(NOT DDLA_INCLUDE_DIR OR NOT DDLA_LIBRARY)
  message(FATAL_ERROR
    "DDLA is required for ENABLE_LIBRI_GPU. Set DDLA_ROOT, or DDLA_INCLUDE_DIR and DDLA_LIBRARY.")
endif()
if(NOT EXISTS "${LIBRI_DIR}/include/RI/global/gpu/GPU_Backend.h")
  message(FATAL_ERROR "The selected LibRI does not provide the GPU backend adapter")
endif()

add_library(DDLA::DDLA UNKNOWN IMPORTED)
set_target_properties(DDLA::DDLA PROPERTIES
  IMPORTED_LOCATION "${DDLA_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${DDLA_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES "CUDA::cudart;CUDA::cublas;CUDA::cusolver;CUDA::curand;MPI::MPI_CXX")

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_INCLUDES "${LIBRI_DIR}/include")
set(CMAKE_REQUIRED_LIBRARIES DDLA::DDLA)
set(CMAKE_REQUIRED_DEFINITIONS -D__GPU_RI -D__DDLA_RI -DDDLA_USE_CUDA)
unset(ABACUS_LIBRI_DDLA_COMPILES CACHE)
check_cxx_source_compiles("
  #include <RI/global/gpu/GPU_Backend.h>
  int main() {
    RI::GPU_Backend::Context context(MPI_COMM_WORLD);
    return 0;
  }
" ABACUS_LIBRI_DDLA_COMPILES)
cmake_pop_check_state()
if(NOT ABACUS_LIBRI_DDLA_COMPILES)
  message(FATAL_ERROR
    "LibRI/DDLA CUDA headers or linkage are incompatible; see CMakeFiles/CMakeConfigureLog.yaml or CMakeError.log. "
    "The tested LibRI 8a2e936 adapter fixes are supplied in cmake/patches/LibRI-8a2e936-ddla-gpu.patch; "
    "apply them to a separate dependency checkout. Use API-compatible DDLA (tested: 9c404ac).")
endif()
message(STATUS "LibRI GPU backend: DDLA/CUDA (${DDLA_LIBRARY})")
mark_as_advanced(DDLA_INCLUDE_DIR DDLA_LIBRARY)
