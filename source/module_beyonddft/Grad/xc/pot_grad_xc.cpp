#include "pot_grad_xc.h"
#include "module_elecstate/potentials/pot_base.h"
#include "module_elecstate/potentials/H_Hartree_pw.h"
#include "module_base/timer.h"
#include "module_hamilt_general/module_xc/xc_functional.h"
#include <set>
namespace elecstate
{
    // constructor for exchange-correlation kernel
    PotGradXCLR::PotGradXCLR(const KernelXC& xc_kernel_in, const ModulePW::PW_Basis* rho_basis_in, const UnitCell* ucell_in, const int& nrxx)
        :xc_kernel_components_(xc_kernel_in), nrxx(nrxx),
        nspin((GlobalV::NSPIN == 1 || (GlobalV::NSPIN == 4 && !GlobalV::DOMAG && !GlobalV::DOMAG_Z)) ? 1 : 2)
    {}

    void PotGradXCLR::cal_v_eff(double** rho, const UnitCell* ucell, ModuleBase::matrix& v_eff)
    {
        ModuleBase::TITLE("PotGradXCLR", "cal_v_eff");
        ModuleBase::timer::tick("PotGradXCLR", "cal_v_eff");
        const int nspin = v_eff.nr;
        if (XC_Functional::get_func_type() == 1 || XC_Functional::get_func_type() == 2 || XC_Functional::get_func_type() == 4)//LDA or GGA or HYBGGA
            if (1 == nspin)// for LDA-spin0, just fxc*rho where fxc=v2rho2; for GGA, v2rho2 has been replaced by the true fxc
                for (int ir = 0;ir < nrxx;++ir)
                    v_eff(0, ir) += this->xc_kernel_components_.get_kernel("v3rho3").at(ir) * rho[0][ir] * rho[0][ir];
            else  //remain for spin 4
                throw std::domain_error("nspin =" + std::to_string(nspin)
                    + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));
        else
            throw std::domain_error("GlobalV::XC_Functional::get_func_type() =" + std::to_string(XC_Functional::get_func_type())
                + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));

        ModuleBase::timer::tick("PotGradXCLR", "cal_v_eff");
    }

} // namespace elecstate
