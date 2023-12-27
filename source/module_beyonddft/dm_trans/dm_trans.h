#pragma once
// use tensor or basematrix in the future
#include <ATen/core/tensor.h>
#include "module_psi/psi.h"
#include <vector>
#ifdef __MPI
#include "module_basis/module_ao/parallel_2d.h"
#endif
namespace hamilt
{
    // use templates in the future.
#ifdef __MPI
/// @brief calculate the 2d-block transition density matrix in AO basis using pdgemm
/// \f[ \tilde{\rho}_{\mu_j\mu_b}=\sum_{jb}c_{j,\mu_j}X_{jb}c^*_{b,\mu_b} \f]
    std::vector<container::Tensor> cal_dm_trans_pblas(
        const psi::Psi<double, psi::DEVICE_CPU>& X_istate,
        const Parallel_2D& px,
        const psi::Psi<double, psi::DEVICE_CPU>& c,
        const Parallel_2D& pc,
        const int naos,
        const int nocc,
        const int nvirt,
        Parallel_2D& pmat);
    /// @brief calculate the 2d-block transition density matrix in AO basis using pdgemm
    /// \f[ \tilde{\rho}_{\mu_j\mu_b}=\sum_{jb}c_{j,\mu_j}X_{jb}c^*_{b,\mu_b} \f]
    std::vector<container::Tensor> cal_dm_trans_pblas(
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate,
        const Parallel_2D& px,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c,
        const Parallel_2D& pc,
        const int naos,
        const int nocc,
        const int nvirt,
        Parallel_2D& pmat);
#endif

    /// @brief calculate the 2d-block transition density matrix in AO basis using dgemm
    std::vector<container::Tensor> cal_dm_trans_blas(
        const psi::Psi<double, psi::DEVICE_CPU>& X_istate,
        const psi::Psi<double, psi::DEVICE_CPU>& c);
    /// @brief calculate the 2d-block transition density matrix in AO basis using dgemm
    std::vector<container::Tensor> cal_dm_trans_blas(
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c,
        const int& nocc, const int& nvirt);

    // for test
    /// @brief calculate the 2d-block transition density matrix in AO basis using for loop (for test)
    std::vector<container::Tensor> cal_dm_trans_forloop_serial(
        const psi::Psi<double, psi::DEVICE_CPU>& X_istate,
        const psi::Psi<double, psi::DEVICE_CPU>& c,
        const int& nocc, const int& nvirt);
    /// @brief calculate the 2d-block transition density matrix in AO basis using for loop (for test)
    std::vector<container::Tensor> cal_dm_trans_forloop_serial(
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c,
        const int& nocc, const int& nvirt);
}
#include "dm_trans_parallel.hpp"
#include "dm_trans_serial.hpp"
