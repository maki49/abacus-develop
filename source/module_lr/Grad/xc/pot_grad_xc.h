#pragma once
#include "module_lr/potentials/xc_kernel.h"
#include "module_lr/potentials/pot_lr_base.h"

namespace LR
{
    /// the "potential" contributing to RHS of Z-vector equation
    /// from the derivative of xc kernel
    class PotGradXCLR : public PotLRBase
    {
    public:
        // constructor for exchange-correlation kernel
        PotGradXCLR(const KernelXC& xc_kernel_in, const ModulePW::PW_Basis& rho_basis, const UnitCell& ucell, const int& nrxx);
        ~PotGradXCLR() {}
        virtual void cal_v_eff(double** rho, const UnitCell& ucell, ModuleBase::matrix& v_eff, const std::vector<int>& ispin_op = { 0,0 }) override;
        const int nrxx_ = 1;
        const int nspin_ = 1;
        /// kernel components from PotHxcLR
        const KernelXC& xc_kernel_components_;
    };

}