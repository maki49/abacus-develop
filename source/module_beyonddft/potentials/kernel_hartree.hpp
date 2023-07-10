#pragma once
#include "module_elecstate/potentials/pot_base.h"
#include "module_base/constants.h"
#include "module_base/timer.h"
namespace elecstate
{
class KernelHartree : public PotBase
{
public:
    KernelHartree(const ModulePW::PW_Basis* rho_basis_in)
    {
        this->rho_basis_ = rho_basis_in;
        this->dynamic_mode = false;
        this->fixed_mode = true;
    }

    void cal_v_eff(const Charge* chg, const UnitCell* ucell, ModuleBase::matrix& v_eff)
    {
        ModuleBase::TITLE("KernelHartree", "cal_v_eff");
        ModuleBase::timer::tick("KernelHartree", "cal_v_eff");
        const int nspin = v_eff.nr;

        //1. Coulomb kernel in reciprocal space
        std::vector<std::complex<double>> Porter(this->rho_basis_->nmaxgr, std::complex<double>(0.0, 0.0));
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int ig = 0;ig < this->rho_basis_->npw;ig++) //what's the relationship between npw and nmaxgr?
            if (this->rho_basis_->gg[ig] >= 1.0e-8)
                Porter[ig] = ModuleBase::e2 * ModuleBase::FOUR_PI / (ucell->tpiba2 * this->rho_basis_->gg[ig]);

        //2. FFT to real space
        rho_basis_->recip2real(Porter.data(), Porter.data());

        //3. Add to v_eff
        if (nspin == 4)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
            for (int ir = 0;ir < this->rho_basis_->nrxx;ir++)
                v_eff(0, ir) += Porter[ir].real();
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 512)
#endif
            for (int is = 0;is < nspin;is++)
                for (int ir = 0;ir < this->rho_basis_->nrxx;ir++)
                    v_eff(is, ir) += Porter[ir].real();
        }
        ModuleBase::timer::tick("KernelXC", "cal_v_eff");
        return;
    }
};   //class
}   //namespace