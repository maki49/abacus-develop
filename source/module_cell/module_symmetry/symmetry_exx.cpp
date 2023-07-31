#include "symmetry_exx.h"
#include <utility>
#ifdef __MPI
#include <mpi.h>
#endif
std::vector<int> SymExx::invmap(const int* map, const size_t size)
{
    std::vector<int> invf(size);
    for (size_t i = 0; i < size; ++i) invf[map[i]] = i;
    return invf;
}
std::vector<int> SymExx::mapmul(const int* map1, const int* map2, const size_t size)
{
    std::vector<int> f2f1(size);    // f1 first
    for (size_t i = 0; i < size; ++i) f2f1[i] = map2[map1[i]];
    return f2f1;
}
std::vector<std::vector<std::complex<double>>> SymExx::rearange_smat(
    const int ikibz,
    const std::vector<std::complex<double>> sloc_ikibz,
    const int nbasis,
    const Parallel_2D& p2d,
    std::vector<std::map<int, ModuleBase::Vector3<double>>> kstars,
    const UnitCell& ucell)
{
    // add title
    std::vector<std::vector<std::complex<double>>> kvd_sloc; //result

    //1. get the full smat
    // note: the globalfunc-major means outside, while scalapack's major means inside
    bool col_inside = !ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER();
    //sfull is symmetric but sloc not, so row/col major should be considered
    std::vector<std::complex<double>> sfull_ikbz = this->get_full_smat(sloc_ikibz, nbasis, p2d,
        col_inside);

    //2. get invmap: iat1 to iat0
    std::vector<int> isym_rotiat_iat = this->invmap(this->isym_iat_rotiat[0].data(), ucell.nat);

    for (auto isym_kvecd : kstars[ikibz])
    {
        int isym = isym_kvecd.first;
        // 1 symmetry operation - may more than one kvec_d ?? check it !!!
        ModuleBase::Vector3<double> kvds = isym_kvecd.second;
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

std::vector<std::complex<double>> SymExx::get_full_smat(
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
