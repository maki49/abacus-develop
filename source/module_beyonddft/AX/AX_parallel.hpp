#ifdef __MPI
#include "AX.h"
#include "module_base/scalapack_connector.h"
#include "module_base/tool_title.h"
#include "module_beyonddft/utils/lr_util.h"
namespace hamilt
{
    //output: col first, consistent with blas
    // c: nao*nbands in para2d, nbands*nao in psi  (row-para and constructed: nao)
    // X: nvirt*nocc in para2d, nocc*nvirt in psi (row-para and constructed: nvirt)
    void cal_AX_pblas(
        const std::vector<container::Tensor>& V_istate,
        const Parallel_2D& pmat,
        const psi::Psi<double, psi::DEVICE_CPU>& c,
        const Parallel_2D& pc,
        int naos,
        int nocc,
        int nvirt,
        Parallel_2D& pX,
        psi::Psi<double, psi::DEVICE_CPU>& AX_istate)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_AX_pblas");
        assert(pmat.comm_2D == pc.comm_2D);
        assert(pmat.blacs_ctxt == pc.blacs_ctxt);

        if (pX.comm_2D != pmat.comm_2D || pX.blacs_ctxt != pmat.blacs_ctxt)
            LR_Util::setup_2d_division(pX, pmat.get_block_size(), nvirt, nocc, pmat.comm_2D, pmat.blacs_ctxt);
        else assert(pX.get_local_size() > 0 && AX_istate.get_nbasis() == pX.get_local_size());

        int nsk = c.get_nk();
        assert(V_istate.size() == nsk);

        Parallel_2D pVc;        // for intermediate Vc
        LR_Util::setup_2d_division(pVc, pmat.get_block_size(), naos, nocc, pmat.comm_2D, pmat.blacs_ctxt);
        for (int isk = 0;isk < nsk;++isk)
        {
            AX_istate.fix_k(isk);
            c.fix_k(isk);

            //Vc
            container::Tensor Vc(DAT::DT_DOUBLE, DEV::CpuDevice, { pVc.get_col_size(), pVc.get_row_size() });//row is "inside"(memory contiguity) for pblas

            int i1 = 1;
            int ivirt = nocc + 1;

            char transa = 'T';
            char transb = 'N';
            const double alpha = 1.0;
            const double beta = 0.0;
            pdgemm_(&transa, &transb, &naos, &nocc, &naos,
                &alpha, V_istate[isk].data<double>(), &i1, &i1, pmat.desc,
                c.get_pointer(), &i1, &i1, pc.desc,
                &beta, Vc.data<double>(), &i1, &i1, pVc.desc);

            // AX_istate = c ^ TVc
            // descC puts M(nvirt) to row
            transa = 'T';
            transb = 'N';
            // output c
            std::cout << "c.get_pointer() =" << std::endl;
            for (int i = 0;i < c.get_nbasis();++i)
            {
                std::cout << c.get_pointer()[i] << " ";
            }
            std::cout << std::endl;
            // output Vc
            std::cout << "Vc:" << std::endl;
            for (int i = 0;i < naos * nocc;++i)
            {
                std::cout << Vc.data<double>()[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "AX:" << std::endl;
            for (int i = 0;i < nvirt * nocc;++i)
            {
                std::cout << AX_istate.get_pointer()[i] << " ";
            }

            pdgemm_(&transa, &transb, &nvirt, &nocc, &naos,
                &alpha, c.get_pointer(), &i1, &ivirt, pc.desc,
                Vc.data<double>(), &i1, &i1, pVc.desc,
                &beta, AX_istate.get_pointer(), &i1, &i1, pX.desc);

        }
    }

    void cal_AX_pblas(
        const std::vector<container::Tensor>& V_istate,
        const Parallel_2D& pmat,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c,
        const Parallel_2D& pc,
        int naos,
        int nocc,
        int nvirt,
        Parallel_2D& pX,
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>& AX_istate)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_AX_plas");
        assert(pmat.comm_2D == pc.comm_2D);
        assert(pmat.blacs_ctxt == pc.blacs_ctxt);

        if (pX.comm_2D != pmat.comm_2D || pX.blacs_ctxt != pmat.blacs_ctxt)
            LR_Util::setup_2d_division(pX, pmat.get_block_size(), nvirt, nocc, pmat.comm_2D, pmat.blacs_ctxt);
        else assert(pX.get_local_size() > 0 && AX_istate.get_nbasis() == pX.get_local_size());

        int nsk = c.get_nk();
        assert(V_istate.size() == nsk);

        Parallel_2D pVc;        // for intermediate Vc
        LR_Util::setup_2d_division(pVc, pmat.get_block_size(), naos, nocc, pmat.comm_2D, pmat.blacs_ctxt);
        for (size_t isk = 0;isk < nsk;++isk)
        {
            AX_istate.fix_k(isk);
            c.fix_k(isk);

            //Vc
            container::Tensor Vc(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { pVc.get_col_size(), pVc.get_row_size() });

            int i1 = 1;
            int ivirt = nocc + 1;

            char transa = 'T';
            char transb = 'N';
            const std::complex<double> alpha(1.0, 0.0);
            const std::complex<double> beta(0.0, 0.0);
            pzgemm_(&transa, &transb, &naos, &nocc, &naos,
                &alpha, V_istate[isk].data<std::complex<double>>(), &i1, &i1, pmat.desc,
                c.get_pointer(), &i1, &i1, pc.desc,
                &beta, Vc.data<std::complex<double>>(), &i1, &i1, pVc.desc);

            // AX_istate = c ^ TVc
            // descC puts M(nvirt) to row
            transa = 'C';
            transb = 'N';
            pzgemm_(&transa, &transb, &nvirt, &nocc, &naos,
                &alpha, c.get_pointer(), &i1, &ivirt, pc.desc,
                Vc.data<std::complex<double>>(), &i1, &i1, pVc.desc,
                &beta, AX_istate.get_pointer(), &i1, &i1, pX.desc);
        }
    }
}
#endif