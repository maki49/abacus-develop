#pragma once
#include "module_elecstate/potentials/pot_base.h"
#include "kernel_hartree.hpp"
#include "kernel_xc.hpp"

namespace elecstate
{
    class PotHxcLR : public PotBase
    {
    public:
        // constructor for exchange-correlation kernel
        PotHxcLR(const ModulePW::PW_Basis* rho_basis_in, const UnitCell* ucell_in, const Charge* chg_gs/*ground state*/)
        {
            this->rho_basis_ = rho_basis_in;
            this->dynamic_mode = true;
            this->fixed_mode = false;
            this->nrxx = chg_gs->nrxx;
            this->nspin = (GlobalV::NSPIN == 1 || (GlobalV::NSPIN == 4 && !GlobalV::DOMAG && !GlobalV::DOMAG_Z))
                ? 1 : 2;

            //one-time init and cal kernels
            this->kernel_hxc.resize(2);
            this->kernel_hxc[0] = new KernelHartree(rho_basis_in);
            this->kernel_hxc[1] = new KernelXC();
            for (auto k : this->kernel_hxc) k->cal_kernel(chg_gs, ucell_in, this->nspin);
        }
        ~PotHxcLR()
        {
            for (auto k : this->kernel_hxc) delete k;
        }
        void cal_v_eff(const Charge* chg/*excited state*/, const UnitCell* ucell, ModuleBase::matrix& v_eff) override {};
        void cal_v_eff(double** rho, const UnitCell* ucell, ModuleBase::matrix& v_eff)
        {
            ModuleBase::TITLE("PotHxcLR", "cal_v_eff");
            ModuleBase::timer::tick("PotHxcLR", "cal_v_eff");
            const int nspin = v_eff.nr;

            if (XC_Functional::get_func_type() == 1)
                if (1 == nspin)// for LDA-spin0, just f*rho
                    for (int ir = 0;ir < nrxx;++ir)
                        v_eff(0, ir) += (this->kernel_hxc[0]->get_kernel("Hartree")(0, ir) + this->kernel_hxc[1]->get_kernel("v2rho2")(0, ir)) * rho[0][ir];
                else  //remain for spin2, 4
                    throw std::domain_error("GlobalV::NSPIN =" + std::to_string(GlobalV::NSPIN)
                        + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));
            // else if (XC_Functional::get_func_type() == 2) ...
            else
                throw std::domain_error("GlobalV::XC_Functional::get_func_type() =" + std::to_string(XC_Functional::get_func_type())
                    + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));

            ModuleBase::timer::tick("PotHxcLR", "cal_v_eff");
        }
        int nrxx;
        int nspin;
        std::vector<KernelBase*> kernel_hxc;

    };

} // namespace elecstate
