#pragma once
#include <ATen/core/tensor.h>
#include "module_psi/psi.h"
#include <vector>
#ifdef __MPI
#include "module_basis/module_ao/parallel_2d.h"
#endif
namespace hamilt
{
    // occ
    ///  $\sum_{k\mu\nu}C^*_{\mu i}K_{\mu\nu}C_{\nu k}X_{ak}^*$
    template <typename T>
    void CVCX_occ_forloop_serial(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<T, psi::DEVICE_CPU>& c,
        const psi::Psi<T, psi::DEVICE_CPU>& X_istate,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<T, psi::DEVICE_CPU>& AX_istate);
    template <typename T>
    void CVCX_occ_blas(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<T, psi::DEVICE_CPU>& c,
        const psi::Psi<T, psi::DEVICE_CPU>& X_istate,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<T, psi::DEVICE_CPU>& AX_istate,
        const bool add_on = true);
#ifdef __MPI
    template <typename T>
    void CVCX_occ_pblas(
        const std::vector<container::Tensor>& V_istate,
        const Parallel_2D& pmat,
        const psi::Psi<T, psi::DEVICE_CPU>& c,
        const Parallel_2D& pc,
        const psi::Psi<T, psi::DEVICE_CPU>& X_istate,
        const Parallel_2D& px,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<T, psi::DEVICE_CPU>& AX_istate,
        const bool add_on = true);
#endif
    // virt
    ///  $\sum_{b\mu\nu}X^*_{bi}C^*_{\mu b}K_{\mu\nu}C_{\nu a}$
    template <typename T>
    void CVCX_virt_forloop_serial(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<T, psi::DEVICE_CPU>& c,
        const psi::Psi<T, psi::DEVICE_CPU>& X_istate,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<T, psi::DEVICE_CPU>& AX_istate);
    template <typename T>
    void CVCX_virt_blas(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<T, psi::DEVICE_CPU>& c,
        const psi::Psi<T, psi::DEVICE_CPU>& X_istate,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<T, psi::DEVICE_CPU>& AX_istate,
        const bool add_on = true);
#ifdef __MPI
    template <typename T>
    void CVCX_virt_pblas(
        const std::vector<container::Tensor>& V_istate,
        const Parallel_2D& pmat,
        const psi::Psi<T, psi::DEVICE_CPU>& c,
        const Parallel_2D& pc,
        const psi::Psi<T, psi::DEVICE_CPU>& X_istate,
        const Parallel_2D& px,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<T, psi::DEVICE_CPU>& AX_istate,
        const bool add_on = true);
#endif
}