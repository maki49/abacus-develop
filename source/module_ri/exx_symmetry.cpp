#ifdef __EXX
#include "exx_symmetry.h"
#include <utility>
#include "module_psi/psi.h"
#include "module_cell/module_symmetry/symmetry.h"
#ifdef __MPI
#include <mpi.h>
#include "module_base/scalapack_connector.h"
#else
#include "module_base/lapack_connector.h"
#endif

namespace ExxSym
{

    std::vector<std::vector<std::complex<double>>> rearange_smat(
        const int ikibz,
        const std::vector<std::complex<double>> sloc_ikibz,
        const int nbasis,
        const Parallel_2D& p2d,
        std::vector<std::vector<int>>& isym_iat_rotiat,
        std::vector<std::map<int, ModuleBase::Vector3<double>>> kstars,
        const UnitCell& ucell)
    {
        // add title
        std::vector<std::vector<std::complex<double>>> kvd_sloc; //result

        //1. get the full smat
        // note: the globalfunc-major means outside, while scalapack's major means inside
        bool col_inside = !ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER();
        //sfull is symmetric but sloc not, so row/col major should be considered
        std::vector<std::complex<double>> sfull_ikbz = get_full_smat(sloc_ikibz, nbasis, p2d,
            col_inside);

        //2. get invmap: iat1 to iat0
        std::vector<int> isym_rotiat_iat = ModuleSymmetry::Symmetry::invmap(isym_iat_rotiat[0].data(), ucell.nat);

        for (auto& isym_kvecd : kstars[ikibz])
        {
            int isym = isym_kvecd.first;
            // 1 symmetry operation - may more than one kvec_d ?? check it !!!
            // ModuleBase::Vector3<double> kvds = isym_kvecd.second;
            // set rearanged sloc_ik
            std::vector<std::complex<double>> sloc_ik(p2d.get_local_size(), 0);
            for (int mu = 0;mu < p2d.get_row_size();++mu)
            {
                int gr = p2d.local2global_row(mu);
                for (int nu = 0;nu < p2d.get_col_size();++nu)
                {
                    int gc = p2d.local2global_col(nu);
                    int iat1 = ucell.iwt2iat[gc];
                    int iw = ucell.iwt2iw[gc];//orb-index of iat1, the same as iat0
                    int iat0 = isym_rotiat_iat[iat1];   //g(iat0)=iat1
                    assert(ucell.iat2it[iat1] == ucell.iat2it[iat0]);//check for the same it
                    int gc0 = ucell.iat2iwt[iat0] + iw;
                    if (col_inside)
                        sloc_ik[mu * p2d.get_col_size() + nu] = sfull_ikbz[gr * nbasis + gc0];
                    else
                        sloc_ik[nu * p2d.get_row_size() + mu] = sfull_ikbz[gc0 * nbasis + gr];
                }
            }
            kvd_sloc.push_back(sloc_ik);
        }
        return kvd_sloc;
    }

    std::vector<std::complex<double>> get_full_smat(
        const std::vector<std::complex<double>>& locmat,
        const int& nbasis,
        const Parallel_2D& p2d,
        const bool col_inside)
    {
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
        const int& ikibz,
        const std::vector<std::complex<double>>& psi_ikibz,
        const std::vector<std::complex<double>>& sloc_ikibz,
        const std::vector<std::vector<std::complex<double>>>& sloc_ik,
        const int& nbasis,
        const int& nbands,
        const Parallel_Orbitals& pv,
        const bool col_inside)
    {
        int kstar_size = sloc_ik.size();
        psi::Psi<std::complex<double>, psi::DEVICE_CPU> c_k(kstar_size, pv.ncol_bands, pv.get_row_size());
        psi::Psi<std::complex<double>, psi::DEVICE_CPU> tmpSc(1, pv.ncol_bands, pv.get_row_size());
        // only col_maj is considered now 
            // 1. S(gk)c_gk
        char transa = 'N';
        char transb = 'N';
        std::complex<double> alpha = 1.0;
        std::complex<double> beta = 0.0;
#ifdef __MPI
        int i1 = 1;
        pzgemm_(&transa, &transb, &nbasis, &nbands, &nbasis,
            &alpha, sloc_ikibz.data(), &i1, &i1, pv.desc,
            psi_ikibz.data(), &i1, &i1, pv.desc_wfc, &beta,
            tmpSc.get_pointer(), &i1, &i1, pv.desc_wfc);
#else
        zgemm_(&transa, &transb, &nbasis, &nbands, &nbasis,
            &alpha, sloc_ikibz.data(), &nbasis, psi_ikibz.data(), &nbasis,
            &beta, tmpSc.get_pointer(), &nbasis);
#endif

        //2. c_k = S^{-1}(k)S(gk)c_{gk} for each k
        int ik = 0;
        for (auto sk : sloc_ik)// copy, not reference: sk will be replaced by S^{-1}(k) after 2.1 (sk is const)
        {
            c_k.fix_k(ik);

            // 
            // std::vector<std::complex<double>> invsk = sk;

            // 2.1 S^{-1}(k)
            char uplo = 'U';
            int info = -1;
#ifdef __MPI
            pzpotrf_(&uplo, &nbasis, sk.data(), &i1, &i1, pv.desc, &info);
            if (info != 0) ModuleBase::WARNING_QUIT("restore_psik", "Error when factorizing S(k).");
            pzpotri_(&uplo, &nbasis, sk.data(), &i1, &i1, pv.desc, &info);
#else
            zpotrf_(&uplo, &nbasis, sk.data(), &nbasis, &info);
            if (info != 0) ModuleBase::WARNING_QUIT("restore_psik", "Error when factorizing S(k).");
            zpotri_(&uplo, &nbasis, sk.data(), &nbasis, &info);
#endif

            //2.2 S^{-1}(k) * S(gk)c_{gk}
#ifdef __MPI
            pzgemm_(&transa, &transb, &nbasis, &nbands, &nbasis,
                &alpha, sk.data(), &i1, &i1, pv.desc,
                tmpSc.get_pointer(), &i1, &i1, pv.desc_wfc, &beta,
                c_k.get_pointer(), &i1, &i1, pv.desc_wfc);
#else
            zgemm_(&transa, &transb, &nbasis, &nbands, &nbasis,
                &alpha, sk.data(), &nbasis, tmpSc.get_pointer(), &nbasis,
                &beta, c_k.get_pointer(), &nbasis);
#endif
        }
        return c_k;
    }
}
#endif