#pragma once
#include "module_basis/module_pw/pw_basis.h"
#include "module_elecstate/module_charge/charge.h"
#include "module_cell/unitcell.h"
// #include <ATen/tensor.h>

namespace elecstate
{
    class KernelBase
    {
    public:
        virtual ~KernelBase() = default;
        virtual void cal_kernel(const Charge* chg_gs, const UnitCell* ucell, int& nspin) = 0;
        virtual ModuleBase::matrix& get_kernel(const std::string& name) { return kernel_set_[name]; }
    protected:
        const ModulePW::PW_Basis* rho_basis_ = nullptr;
        std::map<std::string, ModuleBase::matrix> kernel_set_; // [kernel_type][nspin][nrxx]
    };

    class KernelXC : public KernelBase
    {
    public:
        KernelXC() {};
        ~KernelXC() {};

        void cal_kernel(const Charge* chg_gs/* ground state*/, const UnitCell* ucell, int& nspin) override;

        // xc kernel for LR-TDDFT
        void f_xc_libxc(const int& nspin, const double& omega, const double& tpiba, const Charge* chg_gs);
    };
}

