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

        int nsk = c.get_nk();
        assert(nsk == X_istate.get_nk());

        std::vector<container::Tensor> dm_trans(nsk, container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, { pmat.get_col_size(), pmat.get_row_size() }));
        for (size_t isk = 0;isk < nsk;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            // Xc^*
            //c^ *= c[nocc, naos] for gamma_only. 
            Parallel_2D pXc;    //nvirt*naos
            LR_Util::setup_2d_division(pXc, px.get_block_size(), nocc, naos, px.comm_2D, px.blacs_ctxt);
            container::Tensor Xc(DAT::DT_DOUBLE, DEV::CpuDevice, { pXc.get_col_size(), pXc.get_row_size() });//row is "inside"(memory contiguity) for pblas
            int i1 = 1;
            int ivirt = nocc + 1;

            char transa = 'T';
            char transb = 'T';
            const double alpha = 1;
            const double beta = 0;
            pdgemm_(&transa, &transb, &nocc, &naos, &nvirt,
                &alpha, X_istate.get_pointer(), &i1, &i1, px.desc,
                c.get_pointer(), &i1, &ivirt, pc.desc,
                &beta, Xc.data<double>(), &i1, &i1, pXc.desc);

            // cXc^*
            pdgemm_(&transa, &transb, &naos, &naos, &nocc,
                &alpha, Xc.data<double>(), &i1, &i1, pXc.desc,
                c.get_pointer(), &i1, &i1, pc.desc,
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

        int nsk = c.get_nk();
        assert(nsk == X_istate.get_nk());

        std::vector<container::Tensor> dm_trans(nsk, container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { pmat.get_col_size(), pmat.get_row_size() }));
        for (size_t isk = 0;isk < nsk;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);

            // (X^T)^T c_{virt}^\dagger
            Parallel_2D pXc;
            LR_Util::setup_2d_division(pXc, px.get_block_size(), nocc, naos, px.comm_2D, px.blacs_ctxt);
            container::Tensor Xc(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { pXc.get_col_size(), pXc.get_row_size() });//row is "inside"(memory contiguity) for pblas
            int i1 = 1;
            int ivirt = nocc + 1;

            char transa = 'C';
            char transb = 'T';
            const std::complex<double> alpha(1.0, 0.0);
            const std::complex<double> beta(0.0, 0.0);
            pzgemm_(&transa, &transb, &nocc, &naos, &nvirt,
                &alpha, X_istate.get_pointer(), &i1, &i1, px.desc,
                c.get_pointer(), &i1, &ivirt, pc.desc,
                &beta, Xc.data<std::complex<double>>(), &i1, &i1, pXc.desc);

            // cXc^*
            pzgemm_(&transa, &transb, &naos, &naos, &nocc,
                &alpha, Xc.data<std::complex<double>>(), &i1, &i1, pXc.desc,
                c.get_pointer(), &i1, &i1, pc.desc,
                &beta, dm_trans[isk].data<std::complex<double>>(), &i1, &i1, pmat.desc);
        }
        return dm_trans;
    }

}
#endif
