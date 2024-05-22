#pragma once
#include <ATen/core/tensor.h>
#include "module_psi/psi.h"
#include <vector>
#include "module_lr/utils/lr_util.h"
namespace hamilt
{
    // use templates in the future.
#ifdef __MPI
/// @brief calculate the 2d-block difference density matrix in AO basis using p?gemm
/// \f[ T=(C_v*X) * (C_v * X)^\dagger + (C_o*X^T) * (C_o*X^T)^\dagger \f]
    template<typename T>
    std::vector<container::Tensor> cal_dm_diff_pblas(
        const psi::Psi<T, base_device::DEVICE_CPU>& X_istate,
        const Parallel_2D& px,
        const psi::Psi<T, base_device::DEVICE_CPU>& c,
        const Parallel_2D& pc,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        const Parallel_2D& pmat,
        const bool renorm_k = true,
        const int nspin = 1);
#endif

    /// @brief calculate the 2d-block transition density matrix in AO basis using ?gemm
    template<typename T>
    std::vector<container::Tensor> cal_dm_diff_blas(
        const psi::Psi<T, base_device::DEVICE_CPU>& X_istate,
        const psi::Psi<T, base_device::DEVICE_CPU>& c,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        const bool renorm_k = true,
        const int nspin = 1);

    // for test
    /// @brief calculate the 2d-block transition density matrix in AO basis using for loop (for test)
    template<typename T>
    std::vector<container::Tensor> cal_dm_diff_forloop(
        const psi::Psi<T, base_device::DEVICE_CPU>& X_istate,
        const psi::Psi<T, base_device::DEVICE_CPU>& c,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        const bool renorm_k = true,
        const int nspin = 1);
}

#include "dm_diff_serial.hpp"
#include "dm_diff_parallel.hpp"