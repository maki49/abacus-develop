#pragma once
// use tensor or basematrix in the future
// #include "module_base/module_container/tensor.h"
#include "module_base/matrix.h"
#include "module_base/complexmatrix.h"
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
    std::vector<ModuleBase::matrix> cal_dm_trans_pblas(
        const psi::Psi<double, psi::DEVICE_CPU>& X_istate,
        const Parallel_2D& px,
        const psi::Psi<double, psi::DEVICE_CPU>& c,
        const Parallel_2D& pc,
        int naos,
        int nocc,
        int nvirt,
        const Parallel_2D& pmat);
    /// @brief calculate the 2d-block transition density matrix in AO basis using pdgemm
    /// \f[ \tilde{\rho}_{\mu_j\mu_b}=\sum_{jb}c_{j,\mu_j}X_{jb}c^*_{b,\mu_b} \f]
    std::vector<ModuleBase::ComplexMatrix> cal_dm_trans_pblas(
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate,
        const Parallel_2D& px,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c,
        const Parallel_2D& pc);
#endif

    /// @brief calculate the 2d-block transition density matrix in AO basis using dgemm
    std::vector<ModuleBase::matrix> cal_dm_trans_blas(
        const psi::Psi<double, psi::DEVICE_CPU>& X_istate,
        const psi::Psi<double, psi::DEVICE_CPU>& c);
    /// @brief calculate the 2d-block transition density matrix in AO basis using for loop (for test)
    std::vector<ModuleBase::matrix> cal_dm_trans_forloop_serial(
        const psi::Psi<double, psi::DEVICE_CPU>& X_istate,
        const psi::Psi<double, psi::DEVICE_CPU>& c);
    // #endif

    /// @brief calculate the 2d-block transition density matrix in AO basis using dgemm
    std::vector<ModuleBase::ComplexMatrix> cal_dm_trans_blas(
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c);
    /// @brief calculate the 2d-block transition density matrix in AO basis using for loop (for test)
    std::vector<ModuleBase::ComplexMatrix> cal_dm_trans_forloop_serial(
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c);

    // void cxc_out_test(
    //     const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate,
    //     const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c);
}
#include "dm_trans_parallel.hpp"
#include "dm_trans_serial.hpp"
