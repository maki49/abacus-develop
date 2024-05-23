#include "kernel.h"
#include "module_hamilt_general/module_xc/xc_functional.h"
#include "module_base/timer.h"
#include "module_beyonddft/utils/lr_util.h"

#ifdef USE_LIBXC
#include <xc.h>

void elecstate::KernelXC::get_rho_drho_sigma(const int& nspin,
    const double& tpiba,
    const Charge* chg_gs,
    const bool& is_gga,
    std::vector<double>& rho,
    std::vector<std::vector<ModuleBase::Vector3<double>>>& drho,
    std::vector<double>& sigma)
{
    const int& nrxx = chg_gs->nrxx;
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
        throw std::domain_error("nspin =" + std::to_string(nspin)
            + " unfinished in " + std::string(__FILE__) + " line " + std::to_string(__LINE__));
    }
    if (is_gga)
    {
        // 1. \nabla \rho
        drho.resize(nspin);
        for (int is = 0; is < nspin; ++is)
        {
            std::vector<double> rhor(nrxx);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for (int ir = 0; ir < nrxx; ++ir) rhor[ir] = rho[ir * nspin + is];
            drho[is].resize(nrxx);
            LR_Util::grad(rhor.data(), drho[is].data(), *(chg_gs->rhopw), tpiba);
        }
        // 2. |\nabla\rho|^2
        sigma.resize(nrxx * ((1 == nspin) ? 1 : 3));
        if (1 == nspin)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for (int ir = 0; ir < nrxx; ++ir)
                sigma[ir] = drho[0][ir] * drho[0][ir];
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
            for (int ir = 0; ir < nrxx; ++ir)
            {
                sigma[ir * 3] = drho[0][ir] * drho[0][ir];
                sigma[ir * 3 + 1] = drho[0][ir] * drho[1][ir];
                sigma[ir * 3 + 2] = drho[1][ir] * drho[1][ir];
            }
        }
    }
}

void elecstate::KernelXC::f_xc_libxc(const int& nspin, const double& omega, const double& tpiba, const Charge* chg_gs)
{
    ModuleBase::TITLE("XC_Functional", "f_xc_libxc");
    ModuleBase::timer::tick("XC_Functional", "f_xc_libxc");
    // https://www.tddft.org/programs/libxc/manual/libxc-5.1.x/

    std::vector<xc_func_type> funcs = XC_Functional::init_func((1 == nspin) ? XC_UNPOLARIZED : XC_POLARIZED);
    const int& nrxx = chg_gs->nrxx;

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

    //==================== XC Kernels (f_xc)=============================
    this->kernel_set_.emplace("vrho", std::vector<double>(nspin * nrxx));
    this->kernel_set_.emplace("v2rho2", std::vector<double>(((1 == nspin) ? 1 : 3) * nrxx));//(nrxx* ((1 == nspin) ? 1 : 3)): 00, 01, 11
        if (is_gga)
        {
            this->kernel_set_.emplace("vsigma", std::vector<double>(((1 == nspin) ? 1 : 3) * nrxx)); //(nrxx*): 2 for rho * 3 for sigma: 00, 01, 02, 10, 11, 12
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
                xc_lda_vxc(&func, nrxx, rho.data(), this->kernel_set_["vrho"].data());
                xc_lda_fxc(&func, nrxx, rho.data(), this->kernel_set_["v2rho2"].data());
                break;
            case XC_FAMILY_GGA:
            case XC_FAMILY_HYB_GGA:
                xc_gga_vxc(&func, nrxx, rho.data(), sigma.data(),
                    this->kernel_set_["vrho"].data(), this->kernel_set_["vsigma"].data());
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
                this->to_mul_rho_.resize(nrxx);
                this->to_mul_drho_.resize(nrxx);
                this->to_mul_d2rho_.resize(nrxx);
                const std::vector<double>& v2r2 = this->kernel_set_["v2rho2"];
                const std::vector<double>& v2rs = this->kernel_set_["v2rhosigma"];
                const std::vector<double>& v2s2 = this->kernel_set_["v2sigma2"];
                const std::vector<double>& vs = this->kernel_set_["vsigma"];
                const double tpiba2 = tpiba * tpiba;
                // method 1
                // if (1 == nspin)
                // {
                //     // ============= to be multiplied by transition density ===========
                //     // 1. $\nabla\cdot(f^{\rho\sigma}*\nabla\rho)$
                //     std::vector<double> div_v2rhosigma_gdrho_r(nrxx);
                //     std::vector<ModuleBase::Vector3<double>> v2rhosigma_gdrho_r(nrxx);
                //     for (int ir = 0; ir < nrxx; ++ir)v2rhosigma_gdrho_r[ir] = gdr[0][ir] * v2rs.at(ir);
                //     XC_Functional::grad_dot(v2rhosigma_gdrho_r.data(), div_v2rhosigma_gdrho_r.data(), chg_gs->rhopw, tpiba);
                //     // 2. $\nabla^2(f^{\sigma\sigma}*\sigma)$
                //     std::vector<double> v2sigma2_sigma_r(nrxx);
                //     for (int ir = 0; ir < nrxx; ++ir) v2sigma2_sigma_r[ir] = v2s2.at(ir) * sigma[ir];
                //     gdr[0].resize(nrxx);
                //     LR_Util::laplace(v2sigma2_sigma_r.data(), v2sigma2_sigma_r.data(), *(chg_gs->rhopw), tpiba2);
                //     // 3. $\nabla^2(v^\sigma)$
                //     std::vector<double> lap_vsigma(nrxx);
                //     LR_Util::laplace(vs.data(), lap_vsigma.data(), *(chg_gs->rhopw), tpiba2);
                //     // add up
                //     BlasConnector::axpy(nrxx, 1.0, v2r2.data(), 1, to_mul_rho_.data(), 1);
                //     BlasConnector::axpy(nrxx, -4.0, div_v2rhosigma_gdrho_r.data(), 1, to_mul_rho_.data(), 1);
                //     BlasConnector::axpy(nrxx, 4.0, v2sigma2_sigma_r.data(), 1, to_mul_rho_.data(), 1);
                //     BlasConnector::axpy(nrxx, 2.0, lap_vsigma.data(), 1, to_mul_rho_.data(), 1);

                //     // ============= to be multiplied by transition density gradient===========
                //     // 1. $-2 f^{\rho\sigma}*\nabla\rho$
                //     std::vector<ModuleBase::Vector3<double>> v2rhosigma_gdrho(nrxx);
                //     for (int ir = 0; ir < nrxx; ++ir)v2rhosigma_gdrho[ir] = -2.0 * gdr[0][ir] * v2rs.at(ir);
                //     // 2. $\nabla(4f^{\sigma\sigma}*sigma+2v^{\sigma})$
                //     std::vector<double> v2sigma2_sigma_vsigma(nrxx);
                //     for (int ir = 0; ir < nrxx; ++ir)v2sigma2_sigma_vsigma[ir] = 4.0 * sigma.at(ir) * v2s2.at(ir) + 2.0 * vs.at(ir);
                //     std::vector<ModuleBase::Vector3<double>> grad_v2sigma2_sigma_vsigma(nrxx);
                //     LR_Util::grad(v2sigma2_sigma_vsigma.data(), grad_v2sigma2_sigma_vsigma.data(), *(chg_gs->rhopw), tpiba);
                //     // add up
                //     for (int ir = 0; ir < nrxx; ++ir)
                //         this->to_mul_drho_[ir] = v2rhosigma_gdrho[ir] + grad_v2sigma2_sigma_vsigma[ir];
                // }
                if (1 == nspin)
                {
                    // ============= to be multiplied by transition rho ===========
                    // 1. $\nabla\cdot(f^{\rho\sigma}*\nabla\rho)$
                    std::vector<double> div_v2rhosigma_gdrho_r(nrxx);
                    std::vector<ModuleBase::Vector3<double>> v2rhosigma_gdrho_r(nrxx);
                    for (int ir = 0; ir < nrxx; ++ir)v2rhosigma_gdrho_r[ir] = gdr[0][ir] * v2rs.at(ir);
                    XC_Functional::grad_dot(v2rhosigma_gdrho_r.data(), div_v2rhosigma_gdrho_r.data(), chg_gs->rhopw, tpiba);
                    BlasConnector::axpy(nrxx, 1.0, v2r2.data(), 1, to_mul_rho_.data(), 1);
                    // BlasConnector::axpy(nrxx, -2.0, div_v2rhosigma_gdrho_r.data(), 1, to_mul_rho_.data(), 1);

                    // ============= to be multiplied by transition drho and d2rho===========
                    // 1. $\nabla(4f^{\sigma\sigma}*\sigma+2v^{\sigma})$ and its gradient
                    for (int ir = 0; ir < nrxx; ++ir) to_mul_d2rho_[ir] = -(4.0 * sigma.at(ir) * v2s2.at(ir) + 2.0 * vs.at(ir));
                    LR_Util::grad(to_mul_d2rho_.data(), to_mul_drho_.data(), *(chg_gs->rhopw), tpiba);

                }
                else if (2 == nspin)    // wrong, to be fixed
                {
                    // 1. $\nabla\cdot(f^{\rho\sigma}*\nabla\rho)$
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
                    // 2. $\nabla^2(f^{\sigma\sigma}*\sigma)$
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
                    // 3. $\nabla^2(v^\sigma)$
                    std::vector<double> lap_vsigma(3 * nrxx);
                    for (int ir = 0;ir < nrxx;++ir)
                    {
                        lap_vsigma[ir] = vs.at(ir * 3) * 2.0;
                        lap_vsigma[nrxx + ir] = vs.at(ir * 3 + 1);
                        lap_vsigma[2 * nrxx + ir] = vs.at(ir * 3 + 2) * 2.0;
                    }
                    for (int isig = 0;isig < 3;++isig)
                        LR_Util::laplace(lap_vsigma.data() + isig * nrxx, lap_vsigma.data() + isig * nrxx, *(chg_gs->rhopw), tpiba2);
                    // add to v2rho2
                    BlasConnector::axpy(3 * nrxx, 1.0, v2r2.data(), 1, to_mul_rho_.data(), 1);
                    BlasConnector::axpy(3 * nrxx, -1.0, div_v2rhosigma_gdrho_r.data(), 1, to_mul_rho_.data(), 1);
                    BlasConnector::axpy(3 * nrxx, 1.0, v2sigma2_sigma_r.data(), 1, to_mul_rho_.data(), 1);
                    BlasConnector::axpy(3 * nrxx, 1.0, lap_vsigma.data(), 1, to_mul_rho_.data(), 1);
                }
            }
        } // end for( xc_func_type &func : funcs )
        XC_Functional::finish_func(funcs);

        if (1 == nspin || 2 == nspin) return;
        // else if (4 == nspin)
        else//NSPIN != 1,2,4 is not supported
        {
            throw std::domain_error("nspin =" + std::to_string(nspin)
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