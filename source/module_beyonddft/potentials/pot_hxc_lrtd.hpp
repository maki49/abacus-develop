#pragma once
#include "module_elecstate/potentials/pot_base.h"
#include "module_elecstate/potentials/H_Hartree_pw.h"
#include "kernel_xc.hpp"

namespace elecstate
{
    class PotHxcLR : public PotBase
    {
    public:
        // constructor for exchange-correlation kernel
        PotHxcLR(const std::string& xc_kernel_in, const ModulePW::PW_Basis* rho_basis_in, const UnitCell* ucell_in, const Charge* chg_gs/*ground state*/)
            :xc_kernel(xc_kernel_in)
        {
            std::cout << "xc_kernel_in: " << xc_kernel_in << std::endl;
            this->rho_basis_ = rho_basis_in;
            // this->dynamic_mode = true;
            // this->fixed_mode = false;
            this->nrxx = chg_gs->nrxx;
            this->nspin = (GlobalV::NSPIN == 1 || (GlobalV::NSPIN == 4 && !GlobalV::DOMAG && !GlobalV::DOMAG_Z))
                ? 1 : 2;

            //one-time init and cal kernels
            // this->xc_kernel_components_.resize(1, nullptr);
            // this->xc_kernel_components_[0] = new KernelHartree(rho_basis_in);
            this->pot_hartree = new PotHartree(this->rho_basis_);
            if (this->xc_kernel == "lda" /*|| pbe... */)
            {
                XC_Functional::set_xc_type(this->xc_kernel);
                this->xc_kernel_components_ = new KernelXC();
                this->xc_kernel_components_->cal_kernel(chg_gs, ucell_in, this->nspin);
            }
        }
        ~PotHxcLR()
        {
            if (this->xc_kernel == "lda" /*|| pbe... */)
            delete this->xc_kernel_components_;
        }
        void cal_v_eff(const Charge* chg/*excited state*/, const UnitCell* ucell, ModuleBase::matrix& v_eff) override {};
        void cal_v_eff(double** rho, const UnitCell* ucell, ModuleBase::matrix& v_eff)
        {
            ModuleBase::TITLE("PotHxcLR", "cal_v_eff");
            ModuleBase::timer::tick("PotHxcLR", "cal_v_eff");
            const int nspin = v_eff.nr;
            v_eff += H_Hartree_pw::v_hartree(*ucell, const_cast<ModulePW::PW_Basis*>(this->rho_basis_), v_eff.nr, rho);
            if (xc_kernel == "rpa" || xc_kernel == "hf")  return;
            else if (XC_Functional::get_func_type() == 1)//LDA
                if (1 == nspin)// for LDA-spin0, just f*rho
                    for (int ir = 0;ir < nrxx;++ir)
                        v_eff(0, ir) += this->xc_kernel_components_->get_kernel("v2rho2")(0, ir) * rho[0][ir];
                else  //remain for spin2, 4
                    throw std::domain_error("nspin =" + std::to_string(nspin)
                        + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));
            // else if (XC_Functional::get_func_type() == 2) ...
            else
                throw std::domain_error("GlobalV::XC_Functional::get_func_type() =" + std::to_string(XC_Functional::get_func_type())
                    + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));

            ModuleBase::timer::tick("PotHxcLR", "cal_v_eff");
        }
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
