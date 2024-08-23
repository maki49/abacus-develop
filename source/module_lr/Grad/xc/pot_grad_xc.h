#pragma once
#include "module_elecstate/potentials/H_Hartree_pw.h"
#include "module_lr/potentials/xc_kernel.h"

namespace LR
{
    /// the "potential" contributing to RHS of Z-vector equation
    /// from the derivative of xc kernel
    class PotGradXCLR
    {
    public:
        // constructor for exchange-correlation kernel
        PotGradXCLR(const KernelXC& xc_kernel_in, const ModulePW::PW_Basis& rho_basis, const UnitCell& ucell, const Charge& chg_gs/*ground state*/);
        ~PotGradXCLR() {}
        void cal_v_eff(double** rho, const UnitCell* ucell, ModuleBase::matrix& v_eff);
        const int nrxx_ = 1;
        const int nspin_ = 1;
        /// kernel components from PotHxcLR
        const KernelXC& xc_kernel_components_;
    };

}