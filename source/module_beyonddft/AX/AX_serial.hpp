#include "AX.h"
#include "module_base/blas_connector.h"
#include "module_base/tool_title.h"
namespace hamilt
{
    void cal_AX_forloop_serial(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<double, psi::DEVICE_CPU>& c,
        const int& nocc,
        const int& nvirt,
        psi::Psi<double, psi::DEVICE_CPU>& AX_istate)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_AX_forloop");
        int nsk = c.get_nk();
        assert(V_istate.size() == nsk);
        int naos = c.get_nbasis();
        AX_istate.fix_k(0);
        ModuleBase::GlobalFunc::ZEROS(AX_istate.get_pointer(), nsk * nocc * nvirt);

        for (int isk = 0;isk < nsk;++isk)
        {
            c.fix_k(isk);
            AX_istate.fix_k(isk);
            for (int i = 0;i < nocc;++i)
            {
                for (int a = 0;a < nvirt;++a)
                {
                    for (int nu = 0;nu < naos;++nu)
                    {
                        for (int mu = 0;mu < naos;++mu)
                        {
                            AX_istate(i * nvirt + a) += c(nocc + a, mu) * V_istate[isk].data<double>()[mu * naos + nu] * c(i, nu);
                        }
                    }
                }
            }
        }
    }
    void cal_AX_forloop_serial(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c,
        const int& nocc,
        const int& nvirt,
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>& AX_istate)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_AX_forloop");
        int nsk = c.get_nk();
        assert(V_istate.size() == nsk);
        int naos = c.get_nbasis();
        AX_istate.fix_k(0);
        ModuleBase::GlobalFunc::ZEROS(AX_istate.get_pointer(), nsk * nocc * nvirt);

        for (int isk = 0;isk < nsk;++isk)
        {
            c.fix_k(isk);
            AX_istate.fix_k(isk);
            for (int i = 0;i < nocc;++i)
            {
                for (int a = 0;a < nvirt;++a)
                {
                    for (int nu = 0;nu < naos;++nu)
                    {
                        for (int mu = 0;mu < naos;++mu)
                        {
                            AX_istate(i * nvirt + a) += std::conj(c(nocc + a, mu)) * V_istate[isk].data<std::complex<double>>()[mu * naos + nu] * c(i, nu);
                        }
                    }
                }
            }
        }
    }

    void cal_AX_blas(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<double, psi::DEVICE_CPU>& c,
        const int& nocc,
        const int& nvirt,
        psi::Psi<double, psi::DEVICE_CPU>& AX_istate)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_AX_blas");
        int nsk = c.get_nk();
        assert(V_istate.size() == nsk);
        int naos = c.get_nbasis();

        for (int isk = 0;isk < nsk;++isk)
        {
            c.fix_k(isk);
            AX_istate.fix_k(isk);

            // Vc[naos*nocc]
            container::Tensor Vc(DAT::DT_DOUBLE, DEV::CpuDevice, { nocc, naos });// (Vc)^T
            char transa = 'T';
            char transb = 'N';  //c is col major
            const double alpha = 1.0;
            const double beta = 0.0;
            dgemm_(&transa, &transb, &naos, &nocc, &naos, &alpha,
                V_istate[isk].data<double>(), &naos, c.get_pointer(), &naos, &beta,
                Vc.data<double>(), &naos);

            //AX_istate=c^TVc (nvirt major)
            dgemm_(&transa, &transb, &nvirt, &nocc, &naos, &alpha,
                c.get_pointer(nocc), &naos, Vc.data<double>(), &naos, &beta,
                AX_istate.get_pointer(), &nvirt);
        }
    }
    void cal_AX_blas(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c,
        const int& nocc,
        const int& nvirt,
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>& AX_istate)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_AX_blas");
        int nsk = c.get_nk();
        assert(V_istate.size() == nsk);
        int naos = c.get_nbasis();

        for (int isk = 0;isk < nsk;++isk)
        {
            c.fix_k(isk);
            AX_istate.fix_k(isk);

            // Vc[naos*nocc]
            container::Tensor Vc(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { nocc, naos });// (Vc)^T
            char transa = 'T';
            char transb = 'N';  //c is col major
            const std::complex<double> alpha(1.0, 0.0);
            const std::complex<double> beta(0.0, 0.0);
            zgemm_(&transa, &transb, &naos, &nocc, &naos, &alpha,
                V_istate[isk].data<std::complex<double>>(), &naos, c.get_pointer(), &naos, &beta,
                Vc.data<std::complex<double>>(), &naos);

            //AX_istate=c^TVc (nvirt major)
            transa = 'C';
            transb = 'N';
            zgemm_(&transa, &transb, &nvirt, &nocc, &naos, &alpha,
                c.get_pointer(nocc), &naos, Vc.data<std::complex<double>>(), &naos, &beta,
                AX_istate.get_pointer(), &nvirt);
        }
    }
}