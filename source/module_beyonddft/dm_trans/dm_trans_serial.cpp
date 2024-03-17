#include "dm_trans.h"
#include "module_base/blas_connector.h"
#include "module_base/tool_title.h"
#include "module_base/global_function.h"
#include "module_beyonddft/utils/lr_util.h"
namespace hamilt
{
    std::vector<container::Tensor> cal_dm_trans_forloop_serial(const psi::Psi<double, psi::DEVICE_CPU>& X_istate, const psi::Psi<double, psi::DEVICE_CPU>& c, const int& nocc, const int& nvirt)
    {
        // cxc_out_test(X_istate, c);
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_forloop");
        int nks = c.get_nk();
        assert(nks == X_istate.get_nk());
        assert(nocc * nvirt == X_istate.get_nbasis());
        int naos = c.get_nbasis();
        std::vector<container::Tensor> dm_trans(nks, container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, { naos, naos }));
        for (auto& dm : dm_trans)ModuleBase::GlobalFunc::ZEROS(dm.data<double>(), naos * naos);
        // loop for AOs
        for (size_t isk = 0;isk < nks;++isk)
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
        int nks = c.get_nk();
        assert(nks == X_istate.get_nk());
        assert(nocc * nvirt == X_istate.get_nbasis());
        int naos = c.get_nbasis();
        std::vector<container::Tensor> dm_trans(nks, container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { naos, naos }));
        for (auto& dm : dm_trans)ModuleBase::GlobalFunc::ZEROS(dm.data<std::complex<double>>(), naos * naos);
        // loop for AOs
        for (size_t isk = 0;isk < nks;++isk)
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
                            dm_trans[isk].data<std::complex<double>>()[mu * naos + nu] += std::conj(c(j, mu)) * X_istate(j * nvirt + b) * c(nocc + b, nu);
                    }
                }
            }
        }
        return dm_trans;
    }


    std::vector<container::Tensor> cal_dm_trans_blas(const psi::Psi<double, psi::DEVICE_CPU>& X_istate, const psi::Psi<double, psi::DEVICE_CPU>& c, const int& nocc, const int& nvirt)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_blas");
        int nks = c.get_nk();
        assert(nks == X_istate.get_nk());
        assert(nocc * nvirt == X_istate.get_nbasis());
        int naos = c.get_nbasis();
        std::vector<container::Tensor> dm_trans(nks, container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, { naos, naos }));
        for (size_t isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            // 1. [X*C_occ^T]^T=C_occ*X^T
            char transa = 'N';
            char transb = 'T';
            const double alpha = 1.0;
            const double beta = 0.0;
            container::Tensor Xc(DAT::DT_DOUBLE, DEV::CpuDevice, { nvirt, naos });
            dgemm_(&transa, &transb, &naos, &nvirt, &nocc, &alpha,
                c.get_pointer(), &naos, X_istate.get_pointer(), &nvirt,
                &beta, Xc.data<double>(), &naos);
            // 2. C_virt*[X*C_occ^T]
            dgemm_(&transa, &transb, &naos, &naos, &nvirt, &alpha,
                c.get_pointer(nocc), &naos, Xc.data<double>(), &naos, &beta,
                dm_trans[isk].data<double>(), &naos);
        }
        return dm_trans;
    }


    std::vector<container::Tensor> cal_dm_trans_blas(const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate, const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c, const int& nocc, const int& nvirt)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_blas");
        int nks = c.get_nk();
        assert(nks == X_istate.get_nk());
        assert(nocc * nvirt == X_istate.get_nbasis());
        int naos = c.get_nbasis();
        std::vector<container::Tensor> dm_trans(nks, container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { naos, naos }));
        for (size_t isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            // 1. [X*C_occ^\dagger]^\dagger=C_occ*X^\dagger
            char transa = 'N';
            char transb = 'C';
            const std::complex<double> alpha(1.0, 0.0);
            const std::complex<double> beta(0.0, 0.0);
            container::Tensor Xc(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { nvirt, naos });
            zgemm_(&transa, &transb, &naos, &nvirt, &nocc, &alpha,
                c.get_pointer(), &naos, X_istate.get_pointer(), &nvirt,
                &beta, Xc.data<std::complex<double>>(), &naos);
            // 2. C_virt*[X*C_occ^\dagger]
            zgemm_(&transa, &transb, &naos, &naos, &nvirt, &alpha,
                c.get_pointer(nocc), &naos, Xc.data<std::complex<double>>(), &naos, &beta,
                dm_trans[isk].data<std::complex<double>>(), &naos);
        }
        return dm_trans;
    }

}
