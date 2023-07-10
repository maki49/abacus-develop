#ifdef __MPI
#include "dm_trans.h"
#include "module_base/scalapack_connector.h"
#include "module_base/tool_title.h"
#include "module_beyonddft/utils/lr_util.h"
#include "module_beyonddft/utils/lr_util_algorithms.hpp"
namespace hamilt
{
    //output: col first, consistent with blas
    // c: nao*nbands in para2d, nbands*nao in psi  (row-para and constructed: nao)
    // X: nvirt*nocc in para2d, nocc*nvirt in psi (row-para and constructed: nvirt)
    std::vector<ModuleBase::matrix> cal_dm_trans_pblas(
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

        int nb = px.get_block_size();
        assert(nb == pc.get_block_size());
        LR_Util::setup_2d_division(pmat, nb, naos, naos, px.comm_2D, px.blacs_ctxt);

        int nsk = c.get_nk();
        assert(nsk == X_istate.get_nk());
        int naos_local = c.get_nbasis();
        int nocc_local = X_istate.get_nbands();
        int nvirt_local = X_istate.get_nbasis();
        std::vector<ModuleBase::matrix> dm_trans(nsk);
        for (size_t isk = 0;isk < nsk;++isk)
        {
            dm_trans[isk].create(pmat.get_row_size(), pmat.get_col_size());
            c.fix_k(isk);
            X_istate.fix_k(isk);
            // Xc^*
            //c^ *= c[nocc, naos] for gamma_only. 
            char transa = 'N';
            char transb = 'T'; // for multi-k, use 'C' here in the first gemm and 'T' for the second
            const double alpha = 1.0;
            const double beta = 0.0;

            Parallel_2D pXc;    //nvirt*naos
            LR_Util::setup_2d_division(pXc, nb, nvirt, naos, px.comm_2D, px.blacs_ctxt);
            ModuleBase::matrix Xc(pXc.get_col_size(), pXc.get_row_size(), true);//row is "inside"(memory contiguity) for pblas
            int i1 = 1;
            int ivirt = nocc + 1;

            pdgemm_(&transa, &transb, &nvirt, &naos, &nocc,
                &alpha, X_istate.get_pointer(), &i1, &i1, px.desc,
                c.get_pointer(), &i1, &i1, pc.desc,  // 'T' or 'C': B_{ib:ib+n-1}, B_{jb:jb+k-1}
                &beta, Xc.c, &i1, &i1, pXc.desc);

            // row-first result: 
            // transa = 'T';
            // transb = 'T';
            // // cXc^*
            // pdgemm_(&transa, &transb, &naos, &naos, &nvirt,
            //     &alpha, Xc.c, &i1, &i1, pXc.desc,
            //     c.get_pointer(), &i1, &ivirt, pc.desc,
            //     &beta, dm_trans[isk].c, &i1, &i1, pmat.desc);

            //col-first result:
            transa = 'N';
            transb = 'N';
            // cXc^*
            pdgemm_(&transa, &transb, &naos, &naos, &nvirt,
                &alpha, c.get_pointer(), &i1, &ivirt, pc.desc,
                Xc.c, &i1, &i1, pXc.desc,
                &beta, dm_trans[isk].c, &i1, &i1, pmat.desc);
        }
        return dm_trans;
    }

}
#endif
