#pragma once
#include "module_lr/potentials/pot_hxc_lrtd.h"
#include "module_lr/potentials/kernel.h"

namespace LR
{
    /// the "potential" contributing to RHS of Z-vector equation
    /// from the derivative of xc kernel
    /// question: is there any difference with singlet or triplet?
    class PotGradXCLR : public PotHxcLR
    {
    public:
        // constructor for exchange-correlation kernel
        PotGradXCLR(const KernelXC& xc_kernel_in, const ModulePW::PW_Basis* rho_basis_in, const UnitCell* ucell_in, const int& nrxx);
        ~PotGradXCLR() {}
        void cal_v_eff(const Charge* chg/*excited state*/, const UnitCell* ucell, ModuleBase::matrix& v_eff) override {};
        void cal_v_eff(double** rho, const UnitCell* ucell, ModuleBase::matrix& v_eff);
        int nrxx;
        int nspin;
        /// kernel components from PotHxcLR
        const KernelXC& xc_kernel_components_;
    };

}