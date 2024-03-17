#ifdef __MPI
#include "dm_trans.h"
#include "module_base/scalapack_connector.h"
#include "module_base/tool_title.h"
#include "module_beyonddft/utils/lr_util.h"
namespace hamilt
{

    //output: col first, consistent with blas
    // c: nao*nbands in para2d, nbands*nao in psi  (row-para and constructed: nao)
    // X: nvirt*nocc in para2d, nocc*nvirt in psi (row-para and constructed: nvirt)
    std::vector<container::Tensor> cal_dm_trans_pblas(
        const psi::Psi<double, psi::DEVICE_CPU>& X_istate,
        const Parallel_2D& px,
        const psi::Psi<double, psi::DEVICE_CPU>& c,
        const Parallel_2D& pc,
        int naos,
        int nocc,
        int nvirt,
        Parallel_2D& pmat)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_pblas");
        assert(px.comm_2D == pc.comm_2D);
        assert(px.blacs_ctxt == pc.blacs_ctxt);

        if (pmat.comm_2D != px.comm_2D || pmat.blacs_ctxt != px.blacs_ctxt)
            LR_Util::setup_2d_division(pmat, px.get_block_size(), naos, naos, px.comm_2D, px.blacs_ctxt);
        else assert(pmat.get_local_size() > 0);

        int nks = c.get_nk();
        assert(nks == X_istate.get_nk());

        std::vector<container::Tensor> dm_trans(nks, container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, { pmat.get_col_size(), pmat.get_row_size() }));
        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            int i1 = 1;
            int ivirt = nocc + 1;
            char transa = 'N';
            char transb = 'T';
            const double alpha = 1;
            const double beta = 0;

            // 1. [X*C_occ^T]^T=C_occ*X^T
            Parallel_2D pXc;    //nvirt*naos
            LR_Util::setup_2d_division(pXc, px.get_block_size(), naos, nvirt, px.comm_2D, px.blacs_ctxt);
            container::Tensor Xc(DAT::DT_DOUBLE, DEV::CpuDevice, { pXc.get_col_size(), pXc.get_row_size() });//row is "inside"(memory contiguity) for pblas
            Xc.zero();
            pdgemm_(&transa, &transb, &naos, &nvirt, &nocc,
                &alpha, c.get_pointer(), &i1, &i1, pc.desc,
                X_istate.get_pointer(), &i1, &i1, px.desc,
                &beta, Xc.data<double>(), &i1, &i1, pXc.desc);

            // 2. C_virt*[X*C_occ^T]
            pdgemm_(&transa, &transb, &naos, &naos, &nvirt,
                &alpha, c.get_pointer(), &i1, &ivirt, pc.desc,
                Xc.data<double>(), &i1, &i1, pXc.desc,
                &beta, dm_trans[isk].data<double>(), &i1, &i1, pmat.desc);
        }
        return dm_trans;
    }
    std::vector<container::Tensor> cal_dm_trans_pblas(
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate,
        const Parallel_2D& px,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c,
        const Parallel_2D& pc,
        int naos,
        int nocc,
        int nvirt,
        Parallel_2D& pmat)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_pblas");
        assert(px.comm_2D == pc.comm_2D);
        assert(px.blacs_ctxt == pc.blacs_ctxt);

        if (pmat.comm_2D != px.comm_2D || pmat.blacs_ctxt != px.blacs_ctxt)
            LR_Util::setup_2d_division(pmat, px.get_block_size(), naos, naos, px.comm_2D, px.blacs_ctxt);
        else assert(pmat.get_local_size() > 0);

        int nks = c.get_nk();
        assert(nks == X_istate.get_nk());

        std::vector<container::Tensor> dm_trans(nks, container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { pmat.get_col_size(), pmat.get_row_size() }));
        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            int i1 = 1;
            int ivirt = nocc + 1;
            char transa = 'N';
            char transb = 'C';

            // 1. [X*C_occ^\dagger]^\dagger=C_occ*X^\dagger
            Parallel_2D pXc;
            LR_Util::setup_2d_division(pXc, px.get_block_size(), naos, nvirt, px.comm_2D, px.blacs_ctxt);
            container::Tensor Xc(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { pXc.get_col_size(), pXc.get_row_size() });//row is "inside"(memory contiguity) for pblas
            Xc.zero();
            const std::complex<double> alpha(1.0, 0.0);
            const std::complex<double> beta(0.0, 0.0);
            pzgemm_(&transa, &transb, &naos, &nvirt, &nocc,
                &alpha, c.get_pointer(), &i1, &i1, pc.desc,
                X_istate.get_pointer(), &i1, &i1, px.desc,
                &beta, Xc.data<std::complex<double>>(), &i1, &i1, pXc.desc);

            // 2. C_virt*[X*C_occ^\dagger]
            pzgemm_(&transa, &transb, &naos, &naos, &nvirt,
                &alpha, c.get_pointer(), &i1, &ivirt, pc.desc,
                Xc.data<std::complex<double>>(), &i1, &i1, pXc.desc,
                &beta, dm_trans[isk].data<std::complex<double>>(), &i1, &i1, pmat.desc);
        }
        return dm_trans;
    }

}
#endif
