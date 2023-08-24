#include "exx_symmetry.h"
#include <utility>
#include "module_psi/psi.h"
#ifdef __MPI
#include <mpi.h>
#include "module_base/scalapack_connector.h"
#endif
#include "module_base/lapack_connector.h"

namespace ExxSym
{
    std::vector<std::complex<double>> rearrange_col(
        const int& nbasis,
        const Parallel_2D& p2d,
        const UnitCell& ucell,
        const bool col_inside,
        const std::vector<int>& iat_rotiat, //g, g(iat0)=iat1
        const std::vector<std::complex<double>>& sloc_in)
    {
        // set rearanged sloc_ik
        std::vector<std::complex<double>> sloc_out(p2d.get_local_size());    // result
        // note: the globalfunc-major means outside, while scalapack's major means inside
        // bool col_inside = !ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER();
        //sfull is symmetric but sloc not, so row/col major should be considered
        std::vector<std::complex<double>> sfull_in = get_full_smat(sloc_in, nbasis, p2d, col_inside);
        //get g^{-1} : iat1 to iat0
        std::vector<int> rotiat_iat = ModuleSymmetry::Symmetry::invmap(iat_rotiat.data(), ucell.nat);
        for (int mu = 0;mu < p2d.get_row_size();++mu)
        {
            int gr = p2d.local2global_row(mu);
            for (int nu = 0;nu < p2d.get_col_size();++nu)
            {
                int gc = p2d.local2global_col(nu);
                int iat1 = ucell.iwt2iat[gc];
                int iw = ucell.iwt2iw[gc];//orb-index of iat1, the same as iat0
                int iat0 = rotiat_iat[iat1];   //g(iat0)=iat1
                assert(ucell.iat2it[iat1] == ucell.iat2it[iat0]);//check for the same it
                int gc0 = ucell.iat2iwt[iat0] + iw;
                if (col_inside)
                    sloc_out[mu * p2d.get_col_size() + nu] = sfull_in[gr * nbasis + gc0];
                else
                    sloc_out[nu * p2d.get_row_size() + mu] = sfull_in[gc0 * nbasis + gr];
            }
        }
        return sloc_out;
    }

    std::vector<std::complex<double>> get_full_smat(
        const std::vector<std::complex<double>>& locmat,
        const int& nbasis,
        const Parallel_2D& p2d,
        const bool col_inside)
    {
        ModuleBase::TITLE("ExxSym", "get_full_mat");
#ifdef __MPI
        std::vector<std::complex<double>> fullmat(nbasis * nbasis, 0);
        if (col_inside)
            for (int r = 0; r < p2d.get_row_size(); ++r)
                for (int c = 0; c < p2d.get_col_size(); ++c)
                    fullmat[p2d.local2global_row(r) * nbasis + p2d.local2global_col(c)] = locmat[r * p2d.get_col_size() + c];
        else
            for (int c = 0; c < p2d.get_col_size(); ++c)
                for (int r = 0; r < p2d.get_row_size(); ++r)
                    fullmat[p2d.local2global_col(c) * nbasis + p2d.local2global_row(r)] = locmat[c * p2d.get_row_size() + r];
        MPI_Allreduce(MPI_IN_PLACE, fullmat.data(), nbasis * nbasis, MPI_DOUBLE_COMPLEX, MPI_SUM, p2d.comm_2D);
#else
        std::vector<std::complex<double>> fullmat = locmat;
#endif
        return fullmat;
    }

    psi::Psi<std::complex<double>, psi::DEVICE_CPU> restore_psik(
        const int& nkstot_full,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& psi_ibz,
        const std::vector<std::vector<std::complex<double>>>& sloc_ibz,
        const std::vector<std::vector<std::vector<std::complex<double>>>>& sloc_full,
        const int& nbasis,
        const int& nbands,
        const Parallel_Orbitals& pv)
    {
        ModuleBase::TITLE("ExxSym", "restore_psik");
        psi::Psi<std::complex<double>, psi::DEVICE_CPU> psi_full(nkstot_full, pv.ncol_bands, pv.get_row_size());
        int ikfull_start = 0;
        for (int ik_ibz = 0;ik_ibz < psi_ibz.get_nk();++ik_ibz)
        {
#ifdef __MPI
            restore_psik_scalapack(ik_ibz, ikfull_start, psi_ibz, sloc_ibz[ik_ibz], sloc_full[ik_ibz], nbasis, nbands, pv, &psi_full);
#else
            restore_psik_lapack(ik_ibz, ikfull_start, psi_ibz, sloc_ibz[ik_ibz], sloc_full[ik_ibz], nbasis, nbands, &psi_full);
#endif
            ikfull_start += sloc_full[ik_ibz].size();
        }
        return psi_full;
    }

    psi::Psi<std::complex<double>, psi::DEVICE_CPU> restore_psik(
        const int& nkstot_full,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& psi_ibz,
        const std::vector<std::vector<std::vector<std::complex<double>>>>& invSkrot_Sgk,
        const int& nbasis,
        const int& nbands,
        const Parallel_Orbitals& pv)
    {
        ModuleBase::TITLE("ExxSym", "restore_psik");
        psi::Psi<std::complex<double>, psi::DEVICE_CPU> psi_full(nkstot_full, pv.ncol_bands, pv.get_row_size());
        int ikfull_start = 0;
        for (int ik_ibz = 0;ik_ibz < psi_ibz.get_nk();++ik_ibz)
        {
#ifdef __MPI
            restore_psik_scalapack(ik_ibz, ikfull_start, psi_ibz, invSkrot_Sgk[ik_ibz], nbasis, nbands, pv, &psi_full);
#else
            restore_psik_lapack(ik_ibz, ikfull_start, psi_ibz, invSkrot_Sgk[ik_ibz], nbasis, nbands, &psi_full);
#endif
            ikfull_start += invSkrot_Sgk[ik_ibz].size();
        }
        return psi_full;
    }

    void restore_psik_lapack(
        const int& ikibz,
        const int& ikfull_start,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& psi_ikibz,
        const std::vector<std::complex<double>>& sloc_ikibz,
        const std::vector<std::vector<std::complex<double>>>& sloc_ik,
        const int& nbasis,
        const int& nbands,
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>* psi_full)
    {
        ModuleBase::TITLE("ExxSym", "restore_psik_lapack");
        psi_ikibz.fix_k(ikibz);
        psi::Psi<std::complex<double>, psi::DEVICE_CPU> tmpSc(1, nbands, nbasis);
        // only col_maj is considered now 
        // 1. S(gk)c_gk
        char transa = 'N';
        char transb = 'N';
        std::complex<double> alpha = 1.0;
        std::complex<double> beta = 0.0;

        zgemm_(&transa, &transb, &nbasis, &nbands, &nbasis,
            &alpha, sloc_ikibz.data(), &nbasis, psi_ikibz.get_pointer(), &nbasis,
            &beta, tmpSc.get_pointer(), &nbasis);

        //2. c_k = S^{-1}(k)S(gk)c_{gk} for each k
        int ik = 0;
        for (auto sk : sloc_ik)// copy, not reference: sk will be replaced by S^{-1}(k) after 2.1 (sk is const)
        {
            psi_full->fix_k(ikfull_start + ik);

            // 2.1 S^{-1}(k)
            char uplo = 'U';
            int info = -1;
            zpotrf_(&uplo, &nbasis, sk.data(), &nbasis, &info);
            if (info != 0) ModuleBase::WARNING_QUIT("restore_psik", "Error when factorizing S(k).(info=" + std::to_string(info) + ").");
            zpotri_(&uplo, &nbasis, sk.data(), &nbasis, &info);
            if (info != 0) ModuleBase::WARNING_QUIT("restore_psik", "Error when calculating inv(S(k)).(info=" + std::to_string(info) + ").");
            //transpose and copy the upper triangle
            std::vector<std::complex<double>> invsk(sk.size());
            std::vector<std::complex<double>> ones(sk.size(), 0);
            for (int i = 0;i < nbasis;i++) ones[i * nbasis + i] = std::complex<double>(1, 0);
            char t = 'T';
            zgemm_(&t, &t, &nbasis, &nbasis, &nbasis, &alpha, sk.data(), &nbasis, ones.data(), &nbasis, &beta, invsk.data(), &nbasis);
            zlacpy_(&uplo, &nbasis, &nbasis, sk.data(), &nbasis, invsk.data(), &nbasis);

            //2.2 S^{-1}(k) * S(gk)c_{gk}
            zgemm_(&transa, &transb, &nbasis, &nbands, &nbasis,
                &alpha, invsk.data(), &nbasis, tmpSc.get_pointer(), &nbasis,
                &beta, psi_full->get_pointer(), &nbasis);
            ++ik;
        }
    }

    std::vector<std::vector<std::complex<double>>> cal_invSkrot_Sgk_lapack(
        const int& ikibz,
        const std::vector<std::complex<double>>& sloc_ikibz,
        const std::vector<std::vector<std::complex<double>>>& sloc_ik,
        const int& nbasis)
    {
        ModuleBase::TITLE("ExxSym", "cal_invSkrot_Sgk_lapack");
        std::vector<std::vector<std::complex<double>>> invSkrot_Sgk;    //result
        for (auto sk : sloc_ik)
        {
            std::vector<std::complex<double>> invSkrot_Sgk_ik(sloc_ikibz.size());
            // 2.1 S^{-1}(k)
            int info = -1;
            std::vector<int> ipiv(nbasis);
            zgetrf_(&nbasis, &nbasis, sk.data(), &nbasis, ipiv.data(), &info);
            int ispec = 1;
            int N234 = -1;
            int NB = ilaenv_(&ispec, "ZGETRI", "N", &nbasis, &N234, &N234, &N234);
            int lwork = NB * nbasis;
            std::vector<std::complex<double>> work(lwork);
            zgetri_(&nbasis, sk.data(), &nbasis, ipiv.data(), work.data(), &lwork, &info);

            //2.2 S^{-1}(k) * S(gk)c_{gk}
            char transa = 'N';
            char transb = 'N';
            std::complex<double> alpha(1.0, 0.0);
            std::complex<double> beta(0.0, 0.0);
            zgemm_(&transa, &transb, &nbasis, &nbasis, &nbasis,
                &alpha, sk.data(), &nbasis, sloc_ikibz.data(), &nbasis,
                &beta, invSkrot_Sgk_ik.data(), &nbasis);

            invSkrot_Sgk.push_back(invSkrot_Sgk_ik);
        }
        return invSkrot_Sgk;
    }
    void restore_psik_lapack(
        const int& ikibz,
        const int& ikfull_start,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& psi_ikibz,
        const std::vector<std::vector<std::complex<double>>>& invSkrot_Sgk,
        const int& nbasis,
        const int& nbands,
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>* psi_full)
    {
        ModuleBase::TITLE("ExxSym", "restore_psik_lapack");
        psi_ikibz.fix_k(ikibz);
        for (int ik = 0;ik < invSkrot_Sgk.size();++ik)
        {
            psi_full->fix_k(ikfull_start + ik);

            //[S^{-1}(k) * S(gk)]c_{gk}
            char transa = 'N';
            char transb = 'N';
            std::complex<double> alpha = 1.0;
            std::complex<double> beta = 0.0;
            zgemm_(&transa, &transb, &nbasis, &nbands, &nbasis,
                &alpha, invSkrot_Sgk[ik].data(), &nbasis, psi_ikibz.get_pointer(), &nbasis,
                &beta, psi_full->get_pointer(), &nbasis);
        }
    }

#ifdef __MPI
    void restore_psik_scalapack(
        const int& ikibz,
        const int& ikfull_start,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& psi_ikibz,
        const std::vector<std::complex<double>>& sloc_ikibz,
        const std::vector<std::vector<std::complex<double>>>& sloc_ik,
        const int& nbasis,
        const int& nbands,
        const Parallel_Orbitals& pv,
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>* psi_full)
    {
        ModuleBase::TITLE("ExxSym", "restore_psik_scalapack");
        psi_ikibz.fix_k(ikibz);
        psi::Psi<std::complex<double>, psi::DEVICE_CPU> tmpSc(1, pv.ncol_bands, pv.get_row_size());
        // only col_maj is considered now 
        // 1. S(gk)c_gk
        char transa = 'N';
        char transb = 'N';
        std::complex<double> alpha = 1.0;
        std::complex<double> beta = 0.0;
        int i1 = 1;
        pzgemm_(&transa, &transb, &nbasis, &nbands, &nbasis,
            &alpha, sloc_ikibz.data(), &i1, &i1, pv.desc,
            psi_ikibz.get_pointer(), &i1, &i1, pv.desc_wfc, &beta,
            tmpSc.get_pointer(), &i1, &i1, pv.desc_wfc);

        //2. c_k = S^{-1}(k)S(gk)c_{gk} for each k
        int ik = 0;
        for (auto sk : sloc_ik)// copy, not reference: sk will be replaced by S^{-1}(k) after 2.1 (sk is const)
        {
            psi_full->fix_k(ikfull_start + ik);

            // 2.1 S^{-1}(k)
            char uplo = 'U';
            int info = -1;
            pzpotrf_(&uplo, &nbasis, sk.data(), &i1, &i1, pv.desc, &info);
            if (info != 0) ModuleBase::WARNING_QUIT("restore_psik", "Error when factorizing S(k).(info=" + std::to_string(info) + ").");
            pzpotri_(&uplo, &nbasis, sk.data(), &i1, &i1, pv.desc, &info);
            if (info != 0) ModuleBase::WARNING_QUIT("restore_psik", "Error when calculating inv(S(k)).(info=" + std::to_string(info) + ").");
            //transpose and copy the upper triangle
            std::vector<std::complex<double>> invsk(sk.size());
            std::vector<std::complex<double>> ones(sk.size(), 0);   //row-major
            for (int i = 0;i < nbasis;++i)
                if (pv.in_this_processor(i, i))
                    ones[pv.global2local_col(i) * pv.get_row_size() + pv.global2local_row(i)] = std::complex<double>(1, 0);
            char t = 'T';
            pzgemm_(&t, &t, &nbasis, &nbasis, &nbasis,
                &alpha, sk.data(), &i1, &i1, pv.desc,
                ones.data(), &i1, &i1, pv.desc, &beta,
                invsk.data(), &i1, &i1, pv.desc);
            pzlacpy_(&uplo, &nbasis, &nbasis, sk.data(), &i1, &i1, pv.desc, invsk.data(), &i1, &i1, pv.desc);

            //2.2 S^{-1}(k) * S(gk)c_{gk}
            pzgemm_(&transa, &transb, &nbasis, &nbands, &nbasis,
                &alpha, invsk.data(), &i1, &i1, pv.desc,
                tmpSc.get_pointer(), &i1, &i1, pv.desc_wfc, &beta,
                psi_full->get_pointer(), &i1, &i1, pv.desc_wfc);
            ++ik;
        }
    }
    std::vector<std::vector<std::complex<double>>> cal_invSkrot_Sgk_scalapack(
        const int& ikibz,
        const std::vector<std::complex<double>>& sloc_ikibz,
        const std::vector<std::vector<std::complex<double>>>& sloc_ik,
        const int& nbasis,
        const Parallel_2D& pv)
    {
        ModuleBase::TITLE("ExxSym", "cal_invSkrot_Sgk_scalapack");
        std::vector<std::vector<std::complex<double>>> invSkrot_Sgk;    //result
        for (auto sk : sloc_ik)
        {
            std::vector<std::complex<double>> invSkrot_Sgk_ik(sloc_ikibz.size());
            // 2.1 S^{-1}(k)
            int info = -1;
            int i1 = 1;
            std::vector<int> ipiv(pv.get_local_size());
            pzgetrf_(&nbasis, &nbasis, sk.data(), &i1, &i1, pv.desc, ipiv.data(), &info);
            int lwork = -1;
            int liwork = -1;
            std::vector<std::complex<double>> work(1, 0);
            std::vector<int> iwork(1, 0);
            pzgetri_(&nbasis, sk.data(), &i1, &i1, pv.desc, ipiv.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
            lwork = work[0].real();
            liwork = iwork[0];
            work.resize(lwork);
            iwork.resize(liwork);
            pzgetri_(&nbasis, sk.data(), &i1, &i1, pv.desc, ipiv.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

            //2.2 S^{-1}(k) * S(gk)
            char transa = 'N';
            char transb = 'N';
            std::complex<double> alpha(1.0, 0.0);
            std::complex<double> beta(0.0, 0.0);
            pzgemm_(&transa, &transb, &nbasis, &nbasis, &nbasis,
                &alpha, sk.data(), &i1, &i1, pv.desc,
                sloc_ikibz.data(), &i1, &i1, pv.desc, &beta,
                invSkrot_Sgk_ik.data(), &i1, &i1, pv.desc);

            invSkrot_Sgk.push_back(invSkrot_Sgk_ik);
        }
        return invSkrot_Sgk;
    }
    void restore_psik_scalapack(
        const int& ikibz,
        const int& ikfull_start,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& psi_ikibz,
        const std::vector<std::vector<std::complex<double>>>& invSkrot_Sgk,
        const int& nbasis,
        const int& nbands,
        const Parallel_Orbitals& pv,
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>* psi_full)
    {
        ModuleBase::TITLE("ExxSym", "restore_psik_scalapack");
        psi_ikibz.fix_k(ikibz);
        for (int ik = 0;ik < invSkrot_Sgk.size();++ik)
        {
            psi_full->fix_k(ikfull_start + ik);

            //[S^{-1}(k) * S(gk)]c_{gk}
            char transa = 'N';
            char transb = 'N';
            std::complex<double> alpha = 1.0;
            std::complex<double> beta = 0.0;
            int i1 = 1;
            pzgemm_(&transa, &transb, &nbasis, &nbands, &nbasis,
                &alpha, invSkrot_Sgk[ik].data(), &i1, &i1, pv.desc,
                psi_ikibz.get_pointer(), &i1, &i1, pv.desc_wfc, &beta,
                psi_full->get_pointer(), &i1, &i1, pv.desc_wfc);
        }
    }
#endif
}