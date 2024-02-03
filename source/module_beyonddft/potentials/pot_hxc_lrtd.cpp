#include "pot_hxc_lrtd.h"
#include "module_elecstate/potentials/pot_base.h"
#include "module_elecstate/potentials/H_Hartree_pw.h"
#include "module_base/timer.h"
#include "module_hamilt_general/module_xc/xc_functional.h"

namespace elecstate
{
    // constructor for exchange-correlation kernel
    PotHxcLR::PotHxcLR(const std::string& xc_kernel_in, const ModulePW::PW_Basis* rho_basis_in, const UnitCell* ucell_in, const Charge* chg_gs/*ground state*/)
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
        if (this->xc_kernel == "lda" || this->xc_kernel == "pbe")
        {
            XC_Functional::set_xc_type(this->xc_kernel);
            this->xc_kernel_components_ = new KernelXC();
            this->xc_kernel_components_->cal_kernel(chg_gs, ucell_in, this->nspin);
        }
    }

    void PotHxcLR::cal_v_eff(double** rho, const UnitCell* ucell, ModuleBase::matrix& v_eff)
    {
        ModuleBase::TITLE("PotHxcLR", "cal_v_eff");
        ModuleBase::timer::tick("PotHxcLR", "cal_v_eff");
        const int nspin = v_eff.nr;
        v_eff += H_Hartree_pw::v_hartree(*ucell, const_cast<ModulePW::PW_Basis*>(this->rho_basis_), v_eff.nr, rho);
        if (xc_kernel == "rpa" || xc_kernel == "hf")  return;
        else if (XC_Functional::get_func_type() == 1 || XC_Functional::get_func_type() == 2)//LDA or GGA
            if (1 == nspin)// for LDA-spin0, just fxc*rho where fxc=v2rho2; for GGA, v2rho2 has been replaced by the true fxc
                for (int ir = 0;ir < nrxx;++ir)
                    v_eff(0, ir) += this->xc_kernel_components_->get_kernel("v2rho2")(0, ir) * rho[0][ir];
            else if (2 == nspin)
                for (int ir = 0;ir < nrxx;++ir)
                {
                    v_eff(0, ir) += this->xc_kernel_components_->get_kernel("v2rho2")(0, ir) * rho[0][ir]
                        + this->xc_kernel_components_->get_kernel("v2rho2")(1, ir) * rho[1][ir];
                    v_eff(1, ir) += this->xc_kernel_components_->get_kernel("v2rho2")(1, ir) * rho[0][ir]
                        + this->xc_kernel_components_->get_kernel("v2rho2")(2, ir) * rho[1][ir];
                }
            else  //remain for spin 4
                throw std::domain_error("nspin =" + std::to_string(nspin)
                    + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));
        else
            throw std::domain_error("GlobalV::XC_Functional::get_func_type() =" + std::to_string(XC_Functional::get_func_type())
                + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));

        ModuleBase::timer::tick("PotHxcLR", "cal_v_eff");
    }

} // namespace elecstate
