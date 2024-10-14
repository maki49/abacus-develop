#pragma once
#include "module_hsolver/hsolver.h"
#include "module_hsolver/diago_iter_assist.h"
#include "module_psi/psi.h"
namespace LR
{
    template<typename T, typename Device = base_device::DEVICE_CPU>
    class HSolverLR
    {
        using Real = typename GetTypeReal<T>::type;
        const int& nk;
        const int& npairs;
        const int& ispin_solve;
        const bool out_wfc_lr = false;
    public:
        HSolverLR(const int& nk_in, const int& npairs_in, const int& ispin_solve_in = 0, const Real& ethr = 1e-13, const bool& out_wfc_lr_in = false)
            :nk(nk_in), diag_ethr(ethr), npairs(npairs_in), out_wfc_lr(out_wfc_lr_in), ispin_solve(ispin_solve_in) {};

        /// eigensolver for common Hamilt
        void solve(hamilt::Hamilt<T, Device>* pHamilt,
            psi::Psi<T, Device>& psi,
            elecstate::ElecState* pes,
            const std::string method_in,
            const bool hermitian = true);

        Real diag_ethr = 0.0; // threshold for diagonalization
    };
};