#pragma once
#include "module_elecstate/potentials/pot_base.h"
#include "module_elecstate/potentials/H_Hartree_pw.h"
#include "kernel.h"

namespace elecstate
{
    class PotHxcLR : public PotBase
    {
    public:
        // constructor for exchange-correlation kernel
        PotHxcLR(const std::string& xc_kernel_in, const ModulePW::PW_Basis* rho_basis_in, const UnitCell* ucell_in, const Charge* chg_gs/*ground state*/);
        ~PotHxcLR()
        {
            if (this->xc_kernel == "lda" /*|| pbe... */)
                delete this->xc_kernel_components_;
        }
        void cal_v_eff(const Charge* chg/*excited state*/, const UnitCell* ucell, ModuleBase::matrix& v_eff) override {};
        void cal_v_eff(double** rho, const UnitCell* ucell, ModuleBase::matrix& v_eff);
        int nrxx;
        int nspin;
        PotHartree* pot_hartree;
        /// different components of local and semi-local xc kernels:
        /// LDA: v2rho2
        /// GGA: v2rho2, v2rhosigma, v2sigma2
        /// meta-GGA: v2rho2, v2rhosigma, v2sigma2, v2rholap, v2rhotau, v2sigmalap, v2sigmatau, v2laptau, v2lap2, v2tau2
        KernelBase* xc_kernel_components_;
        const std::string xc_kernel;
    };

} // namespace elecstate
