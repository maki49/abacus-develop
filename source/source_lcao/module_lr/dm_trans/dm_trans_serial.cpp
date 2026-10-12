#include "dm_trans.h"
#include "source_base/module_external/blas_connector.h"
#include "source_base/tool_title.h"
#include "source_base/timer.h"
#include "source_base/global_function.h"
#include "source_lcao/module_lr/utils/lr_util.h"
namespace LR
{
    template<> std::vector<container::Tensor> cal_dm_trans_forloop_serial(
        const double* const X_istate,
        const psi::Psi<double>& c,
        const int& nocc,
        const int& nvirt,
        const double factor,
        const LR_Util::MO_TYPE type)
    {
        // cxc_out_test(X_istate, c);
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_forloop");
        const int nks = c.get_nk();
        int nmo1, nmo2, imo1, imo2;
        LR_Util::set_dim(type, nocc, nvirt, nmo1, nmo2, imo1, imo2);

        const int naos = c.get_nbasis();
        std::vector<container::Tensor> dm_trans(nks, container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, { naos, naos }));
        for (auto& dm : dm_trans)ModuleBase::GlobalFunc::ZEROS(dm.data<double>(), naos * naos);
        // loop for AOs
        for (size_t isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            const int x_start = isk * nmo1 * nmo2;
            for (size_t mu = 0;mu < naos;++mu)
            {
                for (size_t nu = 0;nu < naos;++nu)
                {
                    // loop for ks states
                    for (size_t p = 0;p < nmo1;++p)
                    {
                        for (size_t q = 0; q < nmo2;++q)
                            dm_trans[isk].data<double>()[mu * naos + nu] += c(imo1 + p, mu) * X_istate[x_start + p * nmo2 + q] * c(imo2 + q, nu) * factor;
                    }
                }
            }
        }
        return dm_trans;
    }

    template<> std::vector<container::Tensor> cal_dm_trans_forloop_serial(
        const std::complex<double>* const X_istate,
        const psi::Psi<std::complex<double>>& c,
        const int& nocc,
        const int& nvirt,
        const std::complex<double> factor,
        const LR_Util::MO_TYPE type)
    {
        // cxc_out_test(X_istate, c);
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_forloop");
        const int nks = c.get_nk();
        const int naos = c.get_nbasis();
        int nmo1, nmo2, imo1, imo2;
        LR_Util::set_dim(type, nocc, nvirt, nmo1, nmo2, imo1, imo2);
        std::vector<container::Tensor> dm_trans(nks, container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { naos, naos }));
        for (auto& dm : dm_trans)ModuleBase::GlobalFunc::ZEROS(dm.data<std::complex<double>>(), naos * naos);
        // loop for AOs
        for (size_t isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            const int x_start = isk * nmo1 * nmo2;
            for (size_t mu = 0;mu < naos;++mu)
            {
                for (size_t nu = 0;nu < naos;++nu)
                {
                    // loop for ks states
                    for (size_t p = 0;p < nmo1;++p)
                    {
                        for (size_t q = 0; q < nmo2;++q)
                            dm_trans[isk].data<std::complex<double>>()[nu * naos + mu] +=
                            std::conj(c(imo1 + p, mu)) * X_istate[x_start + p * nmo2 + q] * c(imo2 + q, nu) * factor;
                    }
                }
            }
        }
        return dm_trans;
    }


    template<> std::vector<container::Tensor> cal_dm_trans_blas(
        const double* const X_istate, const psi::Psi<double>& c,
        const int& nocc, const int& nvirt, const double factor,
        const LR_Util::MO_TYPE type)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_blas");
        ModuleBase::timer::start("hamilt_lrtd", "cal_dm_trans_blas");
        const int nks = c.get_nk();
        const int naos = c.get_nbasis();
        int nmo1;
        int nmo2;
        int imo1;
        int imo2;
        LR_Util::set_dim(type, nocc, nvirt, nmo1, nmo2, imo1, imo2);
        const bool right_first = nmo1 < nmo2;
        const int inner = right_first ? nmo1 : nmo2;
        container::Tensor work(DAT::DT_DOUBLE, DEV::CpuDevice, {inner, naos});
        double* const tmp = work.data<double>();
        std::vector<container::Tensor> dm_trans(nks,
            container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, {naos, naos}));
        for (int ik = 0; ik < nks; ++ik)
        {
            c.fix_k(ik);
            const double* const c1 = c.get_pointer(imo1);
            const double* const c2 = c.get_pointer(imo2);
            const double* const x = X_istate + ik * nmo1 * nmo2;
            double* const dm = dm_trans[ik].data<double>();
            if (right_first)
            {
                // (C2 X) C1^T, keeping the intermediate at nao x nmo1.
                BlasConnector::gemm_cm('N', 'N', naos, nmo1, nmo2, 1.0,
                    c2, naos, x, nmo2, 0.0, tmp, naos);
                BlasConnector::gemm_cm('N', 'T', naos, naos, nmo1, factor,
                    tmp, naos, c1, naos, 0.0, dm, naos);
            }
            else
            {
                BlasConnector::gemm_cm('N', 'T', naos, nmo2, nmo1, 1.0,
                    c1, naos, x, nmo2, 0.0, tmp, naos);
                BlasConnector::gemm_cm('N', 'T', naos, naos, nmo2, factor,
                    c2, naos, tmp, naos, 0.0, dm, naos);
            }
        }
        ModuleBase::timer::end("hamilt_lrtd", "cal_dm_trans_blas");
        return dm_trans;
    }

    template<> std::vector<container::Tensor> cal_dm_trans_blas(
        const std::complex<double>* const X_istate, const psi::Psi<std::complex<double>>& c,
        const int& nocc, const int& nvirt, const std::complex<double> factor,
        const LR_Util::MO_TYPE type)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_blas");
        ModuleBase::timer::start("hamilt_lrtd", "cal_dm_trans_blas");
        const int nks = c.get_nk();
        const int naos = c.get_nbasis();
        int nmo1;
        int nmo2;
        int imo1;
        int imo2;
        LR_Util::set_dim(type, nocc, nvirt, nmo1, nmo2, imo1, imo2);
        const bool right_first = nmo1 < nmo2;
        const int inner = right_first ? nmo1 : nmo2;
        container::Tensor work(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, {naos, inner});
        auto* const tmp = work.data<std::complex<double>>();
        const std::complex<double> one(1.0, 0.0);
        const std::complex<double> zero(0.0, 0.0);
        const std::complex<double> conjugate_factor = std::conj(factor);
        const int matrix_size = naos * naos;
        std::vector<container::Tensor> dm_trans(nks,
            container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, {naos, naos}));
        for (int ik = 0; ik < nks; ++ik)
        {
            c.fix_k(ik);
            const auto* const c1 = c.get_pointer(imo1);
            const auto* const c2 = c.get_pointer(imo2);
            const auto* const x = X_istate + ik * nmo1 * nmo2;
            auto* const dm = dm_trans[ik].data<std::complex<double>>();
            if (right_first)
            {
                // Preserve the complex convention (C2 X C1^dagger)^T.
                BlasConnector::gemm_cm('C', 'C', nmo1, naos, nmo2, one,
                    x, nmo2, c2, naos, zero, tmp, nmo1);
                BlasConnector::gemm_cm('N', 'N', naos, naos, nmo1, conjugate_factor,
                    c1, naos, tmp, nmo1, zero, dm, naos);
                for (int i = 0; i < matrix_size; ++i) { dm[i] = std::conj(dm[i]); }
            }
            else
            {
                BlasConnector::gemm_cm('N', 'C', nmo2, naos, nmo1, one,
                    x, nmo2, c1, naos, zero, tmp, nmo2);
                BlasConnector::gemm_cm('T', 'T', naos, naos, nmo2, factor,
                    tmp, nmo2, c2, naos, zero, dm, naos);
            }
        }
        ModuleBase::timer::end("hamilt_lrtd", "cal_dm_trans_blas");
        return dm_trans;
    }
}
