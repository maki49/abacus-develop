#include "dm_trans.h"
#include "module_base/blas_connector.h"
#include "module_base/tool_title.h"
#include "module_base/global_function.h"
namespace hamilt
{
    std::vector<container::Tensor> cal_dm_trans_forloop_serial(const psi::Psi<double, psi::DEVICE_CPU>& X_istate, const psi::Psi<double, psi::DEVICE_CPU>& c, const int& nocc, const int& nvirt)
    {
        // cxc_out_test(X_istate, c);
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_forloop");
        int nsk = c.get_nk();
        assert(nsk == X_istate.get_nk());
        assert(nocc * nvirt == X_istate.get_nbasis());
        int naos = c.get_nbasis();
        std::vector<container::Tensor> dm_trans(nsk, container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, { naos, naos }));
        for (auto& dm : dm_trans)ModuleBase::GlobalFunc::ZEROS(dm.data<double>(), naos * naos);
        // loop for AOs
        for (size_t isk = 0;isk < nsk;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            for (size_t mu = 0;mu < naos;++mu)
            {
                for (size_t nu = 0;nu < naos;++nu)
                {
                    // loop for ks states
                    for (size_t j = 0;j < nocc;++j)
                    {
                        for (size_t b = 0; b < nvirt;++b)
                            dm_trans[isk].data<double>()[mu * naos + nu] += c(j, mu) * X_istate(j * nvirt + b) * c(nocc + b, nu);
                    }
                }
            }
        }
        return dm_trans;
    }

    std::vector<container::Tensor> cal_dm_trans_forloop_serial(const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c, const int& nocc, const int& nvirt)
    {
        // cxc_out_test(X_istate, c);
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_forloop");
        int nsk = c.get_nk();
        assert(nsk == X_istate.get_nk());
        assert(nocc * nvirt == X_istate.get_nbasis());
        int naos = c.get_nbasis();
        std::vector<container::Tensor> dm_trans(nsk, container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { naos, naos }));
        for (auto& dm : dm_trans)ModuleBase::GlobalFunc::ZEROS(dm.data<std::complex<double>>(), naos * naos);
        // loop for AOs
        for (size_t isk = 0;isk < nsk;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            for (size_t mu = 0;mu < naos;++mu)
            {
                for (size_t nu = 0;nu < naos;++nu)
                {
                    // loop for ks states
                    for (size_t j = 0;j < nocc;++j)
                    {
                        for (size_t b = 0; b < nvirt;++b)
                            dm_trans[isk].data<std::complex<double>>()[mu * naos + nu] += c(j, mu) * X_istate(j * nvirt + b) * std::conj(c(nocc + b, nu));
                    }
                }
            }
        }
        return dm_trans;
    }


    std::vector<container::Tensor> cal_dm_trans_blas(const psi::Psi<double, psi::DEVICE_CPU>& X_istate, const psi::Psi<double, psi::DEVICE_CPU>& c, const int& nocc, const int& nvirt)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_blas");
        int nsk = c.get_nk();
        assert(nsk == X_istate.get_nk());
        assert(nocc * nvirt == X_istate.get_nbasis());
        int naos = c.get_nbasis();
        std::vector<container::Tensor> dm_trans(nsk, container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, { naos, naos }));
        for (size_t isk = 0;isk < nsk;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            // Xc^*, c^*=c [nocc, naos] for gamma_only
            char transa = 'T';
            char transb = 'T';
            const double alpha = 1.0;
            const double beta = 0.0;
            container::Tensor Xc(DAT::DT_DOUBLE, DEV::CpuDevice, { nocc, naos });
            dgemm_(&transa, &transb, &nocc, &naos, &nvirt, &alpha,
                X_istate.get_pointer(), &nvirt, c.get_pointer(nocc), &naos,
                &beta, Xc.data<double>(), &nocc);
            // cXc^*
            dgemm_(&transa, &transb, &naos, &naos, &nocc, &alpha,
                Xc.data<double>(), &nocc, c.get_pointer(), &naos, &beta,
                dm_trans[isk].data<double>(), &naos);
        }
        return dm_trans;
    }


    std::vector<container::Tensor> cal_dm_trans_blas(const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate, const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c, const int& nocc, const int& nvirt)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_blas");
        int nsk = c.get_nk();
        assert(nsk == X_istate.get_nk());
        assert(nocc * nvirt == X_istate.get_nbasis());
        int naos = c.get_nbasis();
        std::vector<container::Tensor> dm_trans(nsk, container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { naos, naos }));
        for (size_t isk = 0;isk < nsk;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            // Xc^*
            char transa = 'C';
            char transb = 'T';
            const std::complex<double> alpha(1.0, 0.0);
            const std::complex<double> beta(0.0, 0.0);
            container::Tensor Xc(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { nocc, naos });
            zgemm_(&transa, &transb, &nocc, &naos, &nvirt, &alpha,
                X_istate.get_pointer(), &nvirt, c.get_pointer(nocc), &naos,
                &beta, Xc.data<std::complex<double>>(), &nocc);
            // cXc^*
            zgemm_(&transa, &transb, &naos, &naos, &nocc, &alpha,
                Xc.data<std::complex<double>>(), &nocc, c.get_pointer(), &naos, &beta,
                dm_trans[isk].data<std::complex<double>>(), &naos);
        }
        return dm_trans;
    }

}
