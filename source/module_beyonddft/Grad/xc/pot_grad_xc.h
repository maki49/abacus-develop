#pragma once
#include "module_beyonddft/potentials/kernel.h"
#include "module_beyonddft/potentials/pot_hxc_lrtd.h"

namespace elecstate
{
    /// the "potential" contributing to RHS of Z-vector equation
    /// from the derivative of xc kernel
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

} // namespace elecstate
