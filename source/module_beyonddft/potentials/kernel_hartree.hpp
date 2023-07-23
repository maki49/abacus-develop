#pragma once
#include "kernel_base.h"
#include "module_basis/module_pw/pw_basis.h"
#include "module_base/constants.h"
#include "module_base/timer.h"
namespace elecstate
{
    class KernelHartree : public KernelBase
    {
    public:
        KernelHartree(const ModulePW::PW_Basis* rho_basis_in)
        {
            this->rho_basis_ = rho_basis_in;
        }

        void cal_kernel(const Charge* chg_gs, const UnitCell* ucell, int& nspin) override
        {
            ModuleBase::TITLE("KernelHartree", "cal_v_eff");
            ModuleBase::timer::tick("KernelHartree", "cal_v_eff");
            this->kernel_set_.emplace("Hartree", ModuleBase::matrix(nspin, chg_gs->nrxx));
            //1. Hartree kernel in reciprocal space
            std::vector<std::complex<double>> Porter(this->rho_basis_->nmaxgr, std::complex<double>(0.0, 0.0));
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int ig = 0;ig < this->rho_basis_->npw;ig++) //what's the relationship between npw and nmaxgr?
                if (this->rho_basis_->gg[ig] >= 1.0e-8)
                    Porter[ig] = ModuleBase::e2 * ModuleBase::FOUR_PI / (ucell->tpiba2 * this->rho_basis_->gg[ig]);

            //2. FFT to real space
            rho_basis_->recip2real(Porter.data(), Porter.data());

            //3. Add to kernel_set_
            if (nspin == 4)
            {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
                for (int ir = 0;ir < this->rho_basis_->nrxx;ir++)
                    kernel_set_["Hartree"](0, ir) += Porter[ir].real();
            }
            else
            {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 512)
#endif
                for (int is = 0;is < nspin;is++)
                    for (int ir = 0;ir < this->rho_basis_->nrxx;ir++)
                        kernel_set_["Hartree"](is, ir) += Porter[ir].real();
            }

            ModuleBase::timer::tick("KernelHartree", "cal_v_eff");
            return;
        }
    };   //class


}   //namespace