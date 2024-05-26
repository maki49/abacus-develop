#pragma once
#include "module_elecstate/potentials/pot_base.h"
#include "module_elecstate/potentials/H_Hartree_pw.h"
#include "kernel.h"

namespace elecstate
{
    class PotHxcLR : public PotBase
    {
    public:
        PotHxcLR() = default;
        PotHxcLR(const std::string& xc_kernel_in,
            const ModulePW::PW_Basis* rho_basis_in,
            const UnitCell* ucell_in,
            const Charge* chg_gs/*ground state*/,
            const bool& grad = false);
        ~PotHxcLR() {}
        void cal_v_eff(const Charge* chg/*excited state*/, const UnitCell* ucell, ModuleBase::matrix& v_eff) override {};
        void cal_v_eff(double** rho, const UnitCell* ucell, ModuleBase::matrix& v_eff);
        const KernelXC& get_kernel_componets() const { return this->xc_kernel_components_; }
        const ModulePW::PW_Basis* get_rho_basis() const { return this->rho_basis_; }
        const int get_nrxx() const { return this->nrxx; }

    protected:
        int nrxx = 0;
        int nspin = 1;
        PotHartree* pot_hartree = nullptr;
        /// different components of local and semi-local xc kernels:
        /// LDA: v2rho2
        /// GGA: v2rho2, v2rhosigma, v2sigma2
        /// meta-GGA: v2rho2, v2rhosigma, v2sigma2, v2rholap, v2rhotau, v2sigmalap, v2sigmatau, v2laptau, v2lap2, v2tau2
        KernelXC xc_kernel_components_;
        const std::string xc_kernel = "";
    };

} // namespace elecstate
