#pragma once
#include "kernel_base.h"
#include "module_hamilt_general/module_xc/xc_functional.h"
#include "module_base/timer.h"

namespace elecstate
{
    class KernelXC : public KernelBase
    {
    public:
        KernelXC() {};
        ~KernelXC() {};

        void cal_kernel(const Charge* chg_gs/* ground state*/, const UnitCell* ucell, int& nspin) override;

        // xc kernel for LR-TDDFT
        void f_xc_libxc(const int& nspin, const double& omega, const double& tpiba, const Charge* chg_gs);
    };//class


#ifdef USE_LIBXC
#include <xc.h>
    void KernelXC::f_xc_libxc(const int& nspin, const double& omega, const double& tpiba, const Charge* chg_gs)
    {
        ModuleBase::TITLE("XC_Functional", "f_xc_libxc");
        ModuleBase::timer::tick("XC_Functional", "f_xc_libxc");
        // https://www.tddft.org/programs/libxc/manual/libxc-5.1.x/

        std::vector<xc_func_type> funcs = XC_Functional::init_func((1 == nspin) ? XC_UNPOLARIZED : XC_POLARIZED);
        int nrxx = chg_gs->nrxx;

        // converting rho (extract it as a subfuntion in the future)
        // -----------------------------------------------------------------------------------
        std::vector<double> rho(nspin * nrxx);    //spin-major
        ModuleBase::matrix sigma;
        std::vector<double> amag;
        if (1 == nspin || 2 == nspin)
        {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
            for (int is = 0; is < nspin; ++is)
                for (int ir = 0; ir < nrxx; ++ir)
                    rho[ir * nspin + is] = chg_gs->rho[is][ir] + 1.0 / nspin * chg_gs->rho_core[ir];
        }
        else
        {
            amag.resize(nrxx);
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int ir = 0; ir < nrxx; ++ir)
            {
                const double arhox = std::abs(chg_gs->rho[0][ir] + chg_gs->rho_core[ir]);
                amag[ir] = std::sqrt(std::pow(chg_gs->rho[1][ir], 2) + std::pow(chg_gs->rho[2][ir], 2) + std::pow(chg_gs->rho[3][ir], 2));
                const double amag_clip = (amag[ir] < arhox) ? amag[ir] : arhox;
                rho[ir * nspin + 0] = (arhox + amag_clip) / 2.0;
                rho[ir * nspin + 1] = (arhox - amag_clip) / 2.0;
            }
        }

        // -----------------------------------------------------------------------------------

        // is_gga (currently false)

        // -----------------------------------------------------------------------------------
        //==================== XC Kernels (f_xc)=============================
        //LDA
        this->kernel_set_.emplace("v2rho2", ModuleBase::matrix(((1 == nspin) ? 1 : 3), nrxx));//(nrxx* ((1 == nspin) ? 1 : 3)): 00, 01, 11
        //GGA
        const bool is_gga = [&funcs]()
            {
                for (xc_func_type& func : funcs)
                {
                    switch (func.info->family)
                    {
                    case XC_FAMILY_GGA:
                    case XC_FAMILY_HYB_GGA:
                        return true;
                    }
                }
                return false;
            }();

            if (is_gga)
            {
                this->kernel_set_.emplace("v2rhosigma", ModuleBase::matrix(((1 == nspin) ? 1 : 6), nrxx)); //(nrxx*): 2 for rho * 3 for sigma: 00, 01, 02, 10, 11, 12
                this->kernel_set_.emplace("v2sigma2", ModuleBase::matrix(((1 == nspin) ? 1 : 6), nrxx));   //(nrxx* ((1 == nspin) ? 1 : 6)): 00, 01, 02, 11, 12, 22
            }

        //MetaGGA
        // ModuleBase::matrix v2rholapl;
        // ModuleBase::matrix v2rhotau;
        // ModuleBase::matrix v2sigmalapl;
        // ModuleBase::matrix v2sigmatau;
        // ModuleBase::matrix v2lapl2;
        // ModuleBase::matrix v2tau2;

        for (xc_func_type& func : funcs)
        {
            constexpr double rho_threshold = 1E-6;
            constexpr double grho_threshold = 1E-10;

            xc_func_set_dens_threshold(&func, rho_threshold);

            //cut off grho if not LDA (future subfunc)

            switch (func.info->family)
            {
            case XC_FAMILY_LDA:
                // call Libxc function: xc_lda_exc_vxc
                xc_lda_fxc(&func, nrxx, rho.data(),
                    this->kernel_set_["v2rho2"].c);
                break;
            case XC_FAMILY_GGA:
            case XC_FAMILY_HYB_GGA:
                // call Libxc function: xc_gga_exc_vxc
                xc_gga_fxc(&func, nrxx, rho.data(), sigma.c,
                    this->kernel_set_["v2rho2"].c, this->kernel_set_["v2rhosigma"].c, this->kernel_set_["v2sigma2"].c);
                break;
            default:
                throw std::domain_error("func.info->family =" + std::to_string(func.info->family)
                    + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));
                break;
            }
            // some formulas for GGA
            // if (func.info->family == XC_FAMILY_GGA || func.info->family == XC_FAMILY_HYB_GGA)
            // {
            // }
        } // end for( xc_func_type &func : funcs )
        XC_Functional::finish_func(funcs);

        if (1 == GlobalV::NSPIN || 2 == GlobalV::NSPIN) return;
        // else if (4 == GlobalV::NSPIN)
        else//NSPIN != 1,2,4 is not supported
        {
            throw std::domain_error("GlobalV::NSPIN =" + std::to_string(GlobalV::NSPIN)
                + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));
        }
    }
#endif

    void KernelXC::cal_kernel(const Charge* chg_gs/* ground state*/, const UnitCell* ucell, int& nspin)
    {
#ifdef USE_LIBXC
        this->f_xc_libxc(nspin, ucell->omega, ucell->tpiba, chg_gs);
#else
        ModuleBase::WARNING_QUIT("KernelXC", "to calculate xc-kernel in LR-TDDFT, compile with LIBXC");
#endif
    }
}