#ifdef __MPI
#include "dm_trans.h"
#include "source_base/module_external/scalapack_connector.h"
#include "source_base/tool_title.h"
#include "source_base/timer.h"
#include "source_lcao/module_lr/utils/lr_util.h"

namespace LR
{
// Coefficients and X use column-major distributed matrices. The real output
// is C2 X C1^T; the complex output retains the historical (C2 X C1^dagger)^T.
template <>
std::vector<container::Tensor> cal_dm_trans_pblas(const double* const X_istate,
    const Parallel_2D& px, const psi::Psi<double>& c, const Parallel_2D& pc,
    const int naos, const int nocc, const int nvirt, const Parallel_2D& pmat,
    const double factor, const LR_Util::MO_TYPE type)
{
    ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_pblas");
    ModuleBase::timer::start("hamilt_lrtd", "cal_dm_trans_pblas");
    assert(px.comm() == pc.comm() && px.comm() == pmat.comm());
    assert(px.blacs_ctxt == pc.blacs_ctxt && px.blacs_ctxt == pmat.blacs_ctxt);
    assert(pmat.get_local_size() > 0);
    int nmo1;
    int nmo2;
    int imo1;
    int imo2;
    LR_Util::set_dim(type, nocc, nvirt, nmo1, nmo2, imo1, imo2);
    ++imo1;
    ++imo2;
    const bool right_first = nmo1 < nmo2;
    const int inner = right_first ? nmo1 : nmo2;
    const int block_size = px.get_block_size();
    Parallel_2D pwork;
    LR_Util::setup_2d_division(pwork, block_size, naos, inner, px.blacs_ctxt);
    container::Tensor work(DAT::DT_DOUBLE, DEV::CpuDevice,
                           {pwork.get_col_size(), pwork.get_row_size()});
    double* const tmp = work.data<double>();
    const int nks = c.get_nk();
    std::vector<container::Tensor> dm_trans(nks,
        container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice,
                          {pmat.get_col_size(), pmat.get_row_size()}));
    for (int ik = 0; ik < nks; ++ik)
    {
        c.fix_k(ik);
        const double* const coeff = c.get_pointer();
        const double* const x = X_istate + ik * px.get_local_size();
        double* const dm = dm_trans[ik].data<double>();
        if (right_first)
        {
            // (C2 X) C1^T: for VO with many virtuals, work is nao x nocc.
            ScalapackConnector::gemm('N', 'N', naos, nmo1, nmo2, 1.0,
                coeff, 1, imo2, pc.desc, x, 1, 1, px.desc,
                0.0, tmp, 1, 1, pwork.desc);
            ScalapackConnector::gemm('N', 'T', naos, naos, nmo1, factor,
                tmp, 1, 1, pwork.desc, coeff, 1, imo1, pc.desc,
                0.0, dm, 1, 1, pmat.desc);
        }
        else
        {
            // C2 (C1 X^T)^T: cheaper when nmo2 <= nmo1 (including OO/VV).
            ScalapackConnector::gemm('N', 'T', naos, nmo2, nmo1, 1.0,
                coeff, 1, imo1, pc.desc, x, 1, 1, px.desc,
                0.0, tmp, 1, 1, pwork.desc);
            ScalapackConnector::gemm('N', 'T', naos, naos, nmo2, factor,
                coeff, 1, imo2, pc.desc, tmp, 1, 1, pwork.desc,
                0.0, dm, 1, 1, pmat.desc);
        }
    }
    ModuleBase::timer::end("hamilt_lrtd", "cal_dm_trans_pblas");
    return dm_trans;
}

template <>
std::vector<container::Tensor> cal_dm_trans_pblas(const std::complex<double>* const X_istate,
    const Parallel_2D& px, const psi::Psi<std::complex<double>>& c, const Parallel_2D& pc,
    const int naos, const int nocc, const int nvirt, const Parallel_2D& pmat,
    const std::complex<double> factor, const LR_Util::MO_TYPE type)
{
    ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_pblas");
    ModuleBase::timer::start("hamilt_lrtd", "cal_dm_trans_pblas");
    assert(px.comm() == pc.comm() && px.comm() == pmat.comm());
    assert(px.blacs_ctxt == pc.blacs_ctxt && px.blacs_ctxt == pmat.blacs_ctxt);
    assert(pmat.get_local_size() > 0);
    int nmo1;
    int nmo2;
    int imo1;
    int imo2;
    LR_Util::set_dim(type, nocc, nvirt, nmo1, nmo2, imo1, imo2);
    ++imo1;
    ++imo2;
    const bool right_first = nmo1 < nmo2;
    const int inner = right_first ? nmo1 : nmo2;
    const int block_size = px.get_block_size();
    Parallel_2D pwork;
    LR_Util::setup_2d_division(pwork, block_size, inner, naos, px.blacs_ctxt);
    container::Tensor work(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice,
                           {pwork.get_col_size(), pwork.get_row_size()});
    auto* const tmp = work.data<std::complex<double>>();
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    const std::complex<double> conjugate_factor = std::conj(factor);
    const int nks = c.get_nk();
    const int local_size = pmat.get_local_size();
    std::vector<container::Tensor> dm_trans(nks,
        container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice,
                          {pmat.get_col_size(), pmat.get_row_size()}));
    for (int ik = 0; ik < nks; ++ik)
    {
        c.fix_k(ik);
        const auto* const coeff = c.get_pointer();
        const auto* const x = X_istate + ik * px.get_local_size();
        auto* const dm = dm_trans[ik].data<std::complex<double>>();
        if (right_first)
        {
            // U^dagger = X^dagger C2^dagger, then conjugate C1 U^dagger.
            // This gives (C2 X C1^dagger)^T without a distributed transpose.
            ScalapackConnector::gemm('C', 'C', nmo1, naos, nmo2, one,
                x, 1, 1, px.desc, coeff, 1, imo2, pc.desc,
                zero, tmp, 1, 1, pwork.desc);
            ScalapackConnector::gemm('N', 'N', naos, naos, nmo1, conjugate_factor,
                coeff, 1, imo1, pc.desc, tmp, 1, 1, pwork.desc,
                zero, dm, 1, 1, pmat.desc);
            for (int i = 0; i < local_size; ++i) { dm[i] = std::conj(dm[i]); }
        }
        else
        {
            // (X C1^dagger)^T C2^T, with the smaller nmo2 intermediate.
            ScalapackConnector::gemm('N', 'C', nmo2, naos, nmo1, one,
                x, 1, 1, px.desc, coeff, 1, imo1, pc.desc,
                zero, tmp, 1, 1, pwork.desc);
            ScalapackConnector::gemm('T', 'T', naos, naos, nmo2, factor,
                tmp, 1, 1, pwork.desc, coeff, 1, imo2, pc.desc,
                zero, dm, 1, 1, pmat.desc);
        }
    }
    ModuleBase::timer::end("hamilt_lrtd", "cal_dm_trans_pblas");
    return dm_trans;
}
}
#endif
