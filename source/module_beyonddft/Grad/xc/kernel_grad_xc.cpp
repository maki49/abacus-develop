#include "module_beyonddft/potentials/kernel.h"
#include "module_hamilt_general/module_xc/xc_functional.h"
#include "module_base/timer.h"
#include "module_beyonddft/utils/lr_util.h"

#ifdef USE_LIBXC
#include <xc.h>
void elecstate::KernelXC::g_xc_libxc(const int& nspin, const double& omega, const double& tpiba, const Charge* chg_gs)
{
    ModuleBase::TITLE("XC_Functional", "g_xc_libxc");
    ModuleBase::timer::tick("XC_Functional", "g_xc_libxc");
    // https://www.tddft.org/programs/libxc/manual/libxc-5.1.x/
    if (nspin == 2 || nspin == 4) throw std::domain_error("nspin =" + std::to_string(nspin)
        + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));

    std::vector<xc_func_type> funcs = XC_Functional::init_func((1 == nspin) ? XC_UNPOLARIZED : XC_POLARIZED);
    int nrxx = chg_gs->nrxx;

    assert(this->kernel_set_.count("v2rho2"));   //f_xc is calculated

    // -----------------------------------------------------------------------------------
    // getting rho, grad rho, and sigma
    std::vector<double> rho(nspin * nrxx);    // r major / spin contigous
    std::vector<std::vector<ModuleBase::Vector3<double>>> gdr;  // \nabla \rho
    std::vector<double> sigma;  // |\nabla\rho|^2
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
    this->get_rho_drho_sigma(nspin, tpiba, chg_gs, is_gga, rho, gdr, sigma);
    // -----------------------------------------------------------------------------------

    //=================Gradient of XC Kernels (g_xc)========================
    //LDA
    this->kernel_set_.emplace("v3rho3", std::vector<double>(((1 == nspin) ? 1 : 3) * nrxx));//(nrxx* ((1 == nspin) ? 1 : 3)): 00, 01, 11
    //GGA
    if (is_gga)
    {
        this->kernel_set_.emplace("v3rho2sigma", std::vector<double>(((1 == nspin) ? 1 : 9) * nrxx)); //(nrxx*): 2 for rho * 3 for sigma: 00, 01, 02, 10, 11, 12
        this->kernel_set_.emplace("v3rhosigma2", std::vector<double>(((1 == nspin) ? 1 : 12) * nrxx));   //(nrxx* ((1 == nspin) ? 1 : 6)): 00, 01, 02, 11, 12, 22
        this->kernel_set_.emplace("v3sigma3", std::vector<double>(((1 == nspin) ? 1 : 10) * nrxx));
    }
    //MetaGGA ...

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
            xc_lda_kxc(&func, nrxx, rho.data(),
                this->kernel_set_["v3rho3"].data());
            break;
        case XC_FAMILY_GGA:
        case XC_FAMILY_HYB_GGA:
            // call Libxc function: xc_gga_exc_vxc
            xc_gga_kxc(&func, nrxx, rho.data(), sigma.data(),
                this->kernel_set_["v3rho3"].data(),
                this->kernel_set_["v3rho2sigma"].data(),
                this->kernel_set_["v3rhosigma2"].data(),
                this->kernel_set_["v3sigma3"].data());
            break;
        default:
            throw std::domain_error("func.info->family =" + std::to_string(func.info->family)
                + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));
            break;
        }
        /*
        // some formulas for GGA
        if (func.info->family == XC_FAMILY_GGA || func.info->family == XC_FAMILY_HYB_GGA)
        {
            throw std::domain_error("GGA grad kernel is not implemented yet");

            const std::vector<double>& v2rs = this->kernel_set_["v2rhosigma"];
            const std::vector<double>& v2s2 = this->kernel_set_["v2sigma2"];
            std::vector<double>& v3r3 = this->kernel_set_["v3rho3"];
            const std::vector<double>& v3r2s = this->kernel_set_["v3rho2sigma"];
            const std::vector<double>& v3rs2 = this->kernel_set_["v3rhosigma2"];
            const std::vector<double>& v3s3 = this->kernel_set_["v3sigma3"];
            const double tpiba2 = tpiba * tpiba;

            if (1 == nspin)
            {
                // 1. $\nabla f^{\rho\sigma}$
                std::vector<ModuleBase::Vector3<double>> grad_v2rs(nrxx);
                LR_Util::grad(v2rs.data(), grad_v2rs.data(), *(chg_gs->rhopw), tpiba);

                // 2. $\nabla^2 f^{\rho\sigma}$
                std::vector<double> lap_v2rs(nrxx);
                XC_Functional::grad_dot(grad_v2rs.data(), lap_v2rs.data(), chg_gs->rhopw, tpiba);

                // 3. $\nabla\cdot(g^{\rho\rho\sigma}*\nabla\rho)$
                std::vector<ModuleBase::Vector3<double>>v3r2s_drho(nrxx);
                for (int ir = 0; ir < nrxx; ++ir)v3r2s_drho[ir] = gdr[0][ir] * v3r2s.at(ir);
                std::vector<double> div_v3r2s_drho(nrxx);
                XC_Functional::grad_dot(v3r2s_drho.data(), div_v3r2s_drho.data(), chg_gs->rhopw, tpiba);

                // 4. $\nabla^2(g^{\rho\sigma\sigma}*sigma)$
                std::vector<double> v3rs2_sigma(nrxx);
                for (int ir = 0; ir < nrxx; ++ir)v3rs2_sigma[ir] = v3rs2.at(ir) * sigma[ir];
                std::vector<double> lap_v3rs2_sigma(nrxx);
                LR_Util::laplace(v3rs2_sigma.data(), lap_v3rs2_sigma.data(), *(chg_gs->rhopw), tpiba2);

                // 5. $\nabla^3(f^{\sigma\sigma}*\nabla\rho)$
                std::vector<ModuleBase::Vector3<double>> v2s2_drho(nrxx);
                for (int ir = 0; ir < nrxx; ++ir)v2s2_drho[ir] = gdr[0][ir] * v2s2.at(ir);
                std::vector<double> div_v2s2_drho(nrxx);
                XC_Functional::grad_dot(v2s2_drho.data(), div_v2s2_drho.data(), chg_gs->rhopw, tpiba);
                std::vector<double> lap_div_v2s2_drho(nrxx);
                LR_Util::laplace(div_v2s2_drho.data(), lap_div_v2s2_drho.data(), *(chg_gs->rhopw), tpiba2);

                // 6. $\nabla^3(g^{\sigma\sigma\sigma}*\sigma*\nabla\rho)$
                std::vector<ModuleBase::Vector3<double>> v3s3_sigma_drho(nrxx);
                for (int ir = 0; ir < nrxx; ++ir)v3s3_sigma_drho[ir] = gdr[0][ir] * v3s3.at(ir) * sigma[ir];
                std::vector<double> div_v3s3_sigma_drho(nrxx);
                XC_Functional::grad_dot(v3s3_sigma_drho.data(), div_v3s3_sigma_drho.data(), chg_gs->rhopw, tpiba);
                std::vector<double> lap_div_v3s3_sigma_drho(nrxx);
                LR_Util::laplace(div_v3s3_sigma_drho.data(), lap_div_v3s3_sigma_drho.data(), *(chg_gs->rhopw), tpiba2);

                // add to v3rho3
                BlasConnector::axpy(nrxx, 8.0, lap_v2rs.data(), 1, v3r3.data(), 1);
                BlasConnector::axpy(nrxx, -6.0, div_v3r2s_drho.data(), 1, v3r3.data(), 1);
                BlasConnector::axpy(nrxx, 12.0, lap_v3rs2_sigma.data(), 1, v3r3.data(), 1);
                BlasConnector::axpy(nrxx, -16.0, lap_div_v2s2_drho.data(), 1, v3r3.data(), 1);
                BlasConnector::axpy(nrxx, -8.0, lap_div_v3s3_sigma_drho.data(), 1, v3r3.data(), 1);
            }
            else
            {
                throw std::domain_error("GGA grad kernel of nspin=2 or 4 is not implemented yet");
            }
        }
        */
    } // end for( xc_func_type &func : funcs )
    XC_Functional::finish_func(funcs);
    ModuleBase::timer::tick("XC_Functional", "g_xc_libxc");
}
#endif