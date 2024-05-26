#include "pot_hxc_lrtd.h"
#include "module_elecstate/potentials/pot_base.h"
#include "module_elecstate/potentials/H_Hartree_pw.h"
#include "module_base/timer.h"
#include "module_hamilt_general/module_xc/xc_functional.h"
#include <set>
#include "module_beyonddft/utils/lr_util.h"
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
        std::set<std::string> local_xc = { "lda", "pbe", "hse" };
        if (local_xc.find(this->xc_kernel) != local_xc.end())
        {
            XC_Functional::set_xc_type(this->xc_kernel);
            this->xc_kernel_components_.cal_kernel(chg_gs, ucell_in, this->nspin);
        }
    }

    void PotHxcLR::cal_v_eff(double** rho, const UnitCell* ucell, ModuleBase::matrix& v_eff)
    {
        ModuleBase::TITLE("PotHxcLR", "cal_v_eff");
        ModuleBase::timer::tick("PotHxcLR", "cal_v_eff");
#ifdef USE_LIBXC
        const int nspin = v_eff.nr;
        v_eff += H_Hartree_pw::v_hartree(*ucell, const_cast<ModulePW::PW_Basis*>(this->rho_basis_), v_eff.nr, rho);
        if (xc_kernel == "rpa" || xc_kernel == "hf")  return;
        else if (XC_Functional::get_func_type() == 1)//LDA
            if (1 == nspin)// for LDA-spin0, just fxc*rho where fxc=v2rho2; for GGA, v2rho2 has been replaced by the true fxc
                for (int ir = 0;ir < nrxx;++ir)
                    v_eff(0, ir) += ModuleBase::e2 * this->xc_kernel_components_.get_kernel("v2rho2").at(ir) * rho[0][ir];
            else if (2 == nspin)
                for (int ir = 0;ir < nrxx;++ir)
                {
                    const int irs0 = 2 * ir;
                    const int irs1 = irs0 + 1;
                    const int irs2 = irs0 + 2;
                    v_eff(0, ir) += ModuleBase::e2 * this->xc_kernel_components_.get_kernel("v2rho2").at(irs0) * rho[0][ir]
                        + this->xc_kernel_components_.get_kernel("v2rho2").at(irs1) * rho[1][ir];
                    v_eff(1, ir) += ModuleBase::e2 * this->xc_kernel_components_.get_kernel("v2rho2").at(irs1) * rho[0][ir]
                        + this->xc_kernel_components_.get_kernel("v2rho2").at(irs2) * rho[1][ir];
                }
            else  //remain for spin 4
                throw std::domain_error("nspin =" + std::to_string(nspin)
                    + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));
        else if (XC_Functional::get_func_type() == 2 || XC_Functional::get_func_type() == 4)    // GGA or HYB_GGA
        {
            if (1 == nspin)
            {
                std::vector<ModuleBase::Vector3<double>> drho(nrxx);
                LR_Util::grad(rho[0], drho.data(), *(this->rho_basis_), ucell->tpiba);
                std::vector<double> d2rho(nrxx);
                LR_Util::laplace(rho[0], d2rho.data(), *(this->rho_basis_), ucell->tpiba);
                for (int ir = 0;ir < nrxx;++ir)
                    v_eff(0, ir) += ModuleBase::e2 *
                    (this->xc_kernel_components_.get_factor_rho(ir) * rho[0][ir]);
                // + this->xc_kernel_components_.get_factor_drho(ir) * drho.at(ir));
        // + this->xc_kernel_components_.get_factor_d2rho(ir) * d2rho.at(ir));
            }
            else if (2 == nspin)    // wrong, to be fixed
                for (int ir = 0;ir < nrxx;++ir)
                {
                    const int irs0 = 2 * ir;
                    const int irs1 = irs0 + 1;
                    const int irs2 = irs0 + 2;
                    v_eff(0, ir) += ModuleBase::e2 * this->xc_kernel_components_.get_kernel("v2rho2").at(irs0) * rho[0][ir]
                        + this->xc_kernel_components_.get_kernel("v2rho2").at(irs1) * rho[1][ir];
                    v_eff(1, ir) += ModuleBase::e2 * this->xc_kernel_components_.get_kernel("v2rho2").at(irs1) * rho[0][ir]
                        + this->xc_kernel_components_.get_kernel("v2rho2").at(irs2) * rho[1][ir];
                }
            else  //remain for spin 4
                throw std::domain_error("nspin =" + std::to_string(nspin)
                    + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));
        }
        else
#endif
            throw std::domain_error("GlobalV::XC_Functional::get_func_type() =" + std::to_string(XC_Functional::get_func_type())
                + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));

        ModuleBase::timer::tick("PotHxcLR", "cal_v_eff");
    }

} // namespace elecstate
