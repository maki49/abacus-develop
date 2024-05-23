#pragma once
#include "module_basis/module_pw/pw_basis.h"
#include "module_elecstate/module_charge/charge.h"
#include "module_cell/unitcell.h"
// #include <ATen/tensor.h>

namespace elecstate
{
    class KernelXC
    {
    public:
        KernelXC() {};
        ~KernelXC() {};

        void cal_kernel(const Charge* chg_gs/* ground state*/, const UnitCell* ucell, int& nspin);

        const std::vector<double>& get_kernel(const std::string& name) { return kernel_set_[name]; }
        const std::vector<double>& get_factor_rho() { return to_mul_rho_; }
        const double& get_factor_rho(const int& index) { return to_mul_rho_.at(index); }
        const std::vector<ModuleBase::Vector3<double>>& get_factor_drho() { return to_mul_drho_; }
        const ModuleBase::Vector3<double>& get_factor_drho(const int& index) { return to_mul_drho_.at(index); }
        const std::vector<double>& get_factor_d2rho() { return to_mul_d2rho_; }
        const double& get_factor_d2rho(const int& index) { return to_mul_d2rho_.at(index); }


    protected:
        // xc kernel for LR-TDDFT
        void f_xc_libxc(const int& nspin, const double& omega, const double& tpiba, const Charge* chg_gs);

        // derivative of xc kernel for analytical gradient of excitation energy
        void g_xc_libxc(const int& nspin, const double& omega, const double& tpiba, const Charge* chg_gs);

        void get_rho_drho_sigma(const int& nspin,
            const double& tpiba,
            const Charge* chg_gs,
            const bool& is_gga,
            std::vector<double>& rho,
            std::vector<std::vector<ModuleBase::Vector3<double>>>& drho,
            std::vector<double>& sigma);

        const ModulePW::PW_Basis* rho_basis_ = nullptr;
        std::map<std::string, std::vector<double>> kernel_set_; // [kernel_type][nrxx][nspin]
        std::vector<double> to_mul_rho_;
        std::vector<ModuleBase::Vector3<double>> to_mul_drho_;
        std::vector<double> to_mul_d2rho_;
    };
}

