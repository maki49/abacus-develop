#include "kernel.h"
#include "module_hamilt_general/module_xc/xc_functional.h"
#include "module_base/timer.h"
#include "module_beyonddft/utils/lr_util.h"

#ifdef USE_LIBXC
#include <xc.h>
void elecstate::KernelXC::f_xc_libxc(const int& nspin, const double& omega, const double& tpiba, const Charge* chg_gs)
{
    ModuleBase::TITLE("XC_Functional", "f_xc_libxc");
    ModuleBase::timer::tick("XC_Functional", "f_xc_libxc");
    // https://www.tddft.org/programs/libxc/manual/libxc-5.1.x/

    std::vector<xc_func_type> funcs = XC_Functional::init_func((1 == nspin) ? XC_UNPOLARIZED : XC_POLARIZED);
    int nrxx = chg_gs->nrxx;

    // converting rho (extract it as a subfuntion in the future)
    // -----------------------------------------------------------------------------------
    std::vector<double> rho(nspin * nrxx);    // r major / spin contigous
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
    // for GGA
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
        std::vector<std::vector<ModuleBase::Vector3<double>>> gdr;  // \nabla \rho
        std::vector<double> sigma;  // |\nabla\rho|^2
        if (is_gga)
        {
            // 1. \nabla \rho
            gdr.resize(nspin);
            for (int is = 0; is < nspin; ++is)
            {
                std::vector<double> rhor(nrxx);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
                for (int ir = 0; ir < nrxx; ++ir) rhor[ir] = rho[ir * nspin + is];
                gdr[is].resize(nrxx);
                LR_Util::grad(rhor.data(), gdr[is].data(), *(chg_gs->rhopw), tpiba);
            }
            // 2. |\nabla\rho|^2
            sigma.resize(nrxx * ((1 == nspin) ? 1 : 3));
            if (1 == nspin)
            {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
                for (int ir = 0; ir < nrxx; ++ir)
                    sigma[ir] = gdr[0][ir] * gdr[0][ir];
            }
            else
            {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
                for (int ir = 0; ir < nrxx; ++ir)
                {
                    sigma[ir * 3] = gdr[0][ir] * gdr[0][ir];
                    sigma[ir * 3 + 1] = gdr[0][ir] * gdr[1][ir];
                    sigma[ir * 3 + 2] = gdr[1][ir] * gdr[1][ir];
                }
            }
        }
        // -----------------------------------------------------------------------------------
        //==================== XC Kernels (f_xc)=============================
        //LDA
        this->kernel_set_.emplace("v2rho2", std::vector<double>(((1 == nspin) ? 1 : 3) * nrxx));//(nrxx* ((1 == nspin) ? 1 : 3)): 00, 01, 11
        //GGA
        if (is_gga)
        {
            this->kernel_set_.emplace("v2rhosigma", std::vector<double>(((1 == nspin) ? 1 : 6) * nrxx)); //(nrxx*): 2 for rho * 3 for sigma: 00, 01, 02, 10, 11, 12
            this->kernel_set_.emplace("v2sigma2", std::vector<double>(((1 == nspin) ? 1 : 6) * nrxx));   //(nrxx* ((1 == nspin) ? 1 : 6)): 00, 01, 02, 11, 12, 22
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
                xc_lda_fxc(&func, nrxx, rho.data(),
                    this->kernel_set_["v2rho2"].data());
                break;
            case XC_FAMILY_GGA:
            case XC_FAMILY_HYB_GGA:
                // call Libxc function: xc_gga_exc_vxc
                xc_gga_fxc(&func, nrxx, rho.data(), sigma.data(),
                    this->kernel_set_["v2rho2"].data(), this->kernel_set_["v2rhosigma"].data(), this->kernel_set_["v2sigma2"].data());
                break;
            default:
                throw std::domain_error("func.info->family =" + std::to_string(func.info->family)
                    + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));
                break;
            }
            // some formulas for GGA
            if (func.info->family == XC_FAMILY_GGA || func.info->family == XC_FAMILY_HYB_GGA)
            {
                std::vector<double>& v2r2 = this->kernel_set_["v2rho2"];
                const std::vector<double>& v2rs = this->kernel_set_["v2rhosigma"];
                const std::vector<double>& v2s2 = this->kernel_set_["v2sigma2"];
                const double tpiba2 = tpiba * tpiba;
                if (1 == nspin)
                {
                    // \div(v2rhosigma*gdrho) 
                    std::vector<double> div_v2rhosigma_gdrho_r(nrxx);
                    std::vector<ModuleBase::Vector3<double>> v2rhosigma_gdrho_r(nrxx);
                    for (int ir = 0; ir < nrxx; ++ir)v2rhosigma_gdrho_r[ir] = gdr[0][ir] * v2rs.at(ir);
                    XC_Functional::grad_dot(v2rhosigma_gdrho_r.data(), div_v2rhosigma_gdrho_r.data(), chg_gs->rhopw, tpiba);
                    // \lap(v2sigma2*sigma)
                    std::vector<double> v2sigma2_sigma_r(nrxx);
                    for (int ir = 0; ir < nrxx; ++ir) v2sigma2_sigma_r[ir] = v2s2.at(ir) * sigma[ir];
                    gdr[0].resize(nrxx);
                    LR_Util::laplace(v2sigma2_sigma_r.data(), v2sigma2_sigma_r.data(), *(chg_gs->rhopw), tpiba2);
                    // add to v2rho2
                    BlasConnector::axpy(nrxx, -4.0, div_v2rhosigma_gdrho_r.data(), 1, v2r2.data(), 1);
                    BlasConnector::axpy(nrxx, 4.0, v2sigma2_sigma_r.data(), 1, v2r2.data(), 1);
                }
                else if (2 == nspin)
                {
                    //\div(v2rhosigma*gdrho)
                    std::vector<double> div_v2rhosigma_gdrho_r(3 * nrxx);
                    std::vector<ModuleBase::Vector3<double>> v2rhosigma_gdrho_r(3 * nrxx);
                    for (int ir = 0; ir < nrxx; ++ir)
                    {
                        v2rhosigma_gdrho_r[ir] = gdr[0][ir] * v2rs.at(ir * 6) * 4.0
                            + gdr[1][ir] * v2rs.at(ir * 6 + 1) * 2.0;   //up-up
                        v2rhosigma_gdrho_r[nrxx + ir] = gdr[0][ir] * (v2rs.at(ir * 6 + 3) * 2.0 + v2rs.at(ir * 6 + 1))
                            + gdr[1][ir] * (v2rs.at(ir * 6 + 2) * 2.0 + v2rs.at(ir * 6 + 4));   //up-down
                        v2rhosigma_gdrho_r[2 * nrxx + ir] = gdr[1][ir] * v2rs.at(ir * 6 + 5) * 4.0
                            + gdr[0][ir] * v2rs.at(ir * 6 + 4) * 2.0;   //down-down
                    }
                    for (int isig = 0;isig < 3;++isig)
                        XC_Functional::grad_dot(v2rhosigma_gdrho_r.data() + isig * nrxx, div_v2rhosigma_gdrho_r.data() + isig * nrxx, chg_gs->rhopw, tpiba);
                    // \lap(v2sigma2*sigma)
                    std::vector<double> v2sigma2_sigma_r(3 * nrxx);
                    for (int ir = 0; ir < nrxx; ++ir)
                    {
                        v2sigma2_sigma_r[ir] = v2s2.at(ir * 6) * sigma[ir * 3] * 4.0
                            + v2s2.at(ir * 6 + 1) * sigma[ir * 3 + 1] * 4.0
                            + v2s2.at(ir * 6 + 3) * sigma[ir * 3 + 2];   //up-up
                        v2sigma2_sigma_r[nrxx + ir] = v2s2.at(ir * 6 + 1) * sigma[ir * 3] * 2.0
                            + v2s2.at(ir * 6 + 4) * sigma[ir * 3 + 2] * 2.0
                            + (v2s2.at(ir * 6 + 2) * 4.0 + v2s2.at(ir * 6 + 3)) * sigma[ir * 3 + 1];   //up-down
                        v2sigma2_sigma_r[2 * nrxx + ir] = v2s2.at(ir * 6 + 5) * sigma[ir * 3 + 2] * 4.0
                            + v2s2.at(ir * 6 + 4) * sigma[ir * 3 + 1] * 4.0
                            + v2s2.at(ir * 6 + 3) * sigma[ir * 3];   //down-down
                    }
                    for (int isig = 0;isig < 3;++isig)
                        LR_Util::laplace(v2sigma2_sigma_r.data() + isig * nrxx, v2sigma2_sigma_r.data() + isig * nrxx, *(chg_gs->rhopw), tpiba2);
                    // add to v2rho2
                    BlasConnector::axpy(3 * nrxx, -1.0, div_v2rhosigma_gdrho_r.data(), 1, v2r2.data(), 1);
                    BlasConnector::axpy(3 * nrxx, 1.0, v2sigma2_sigma_r.data(), 1, v2r2.data(), 1);

                }
            }
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

void elecstate::KernelXC::cal_kernel(const Charge* chg_gs/* ground state*/, const UnitCell* ucell, int& nspin)
{
#ifdef USE_LIBXC
    this->f_xc_libxc(nspin, ucell->omega, ucell->tpiba, chg_gs);
#else
    ModuleBase::WARNING_QUIT("KernelXC", "to calculate xc-kernel in LR-TDDFT, compile with LIBXC");
#endif
}