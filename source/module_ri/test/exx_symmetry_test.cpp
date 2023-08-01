#include "gtest/gtest.h"
#include "module_cell/module_symmetry/symmetry.h"
#include "../exx_symmetry.h"
#include <map>
#include <tuple>


class SymExxTest : public testing::Test
{
protected:
    int dsize = 1;
    int my_rank = 0;
    std::ofstream ofs_running;
    Parallel_Orbitals pv;

    //cases
    std::map<std::vector<int>, std::vector<int>> invmap_cases = {
        {{3, 2, 1, 0}, {3, 2, 1, 0}},
     { {4, 1, 3, 0, 2}, {3, 1, 4, 2, 0} } };
    std::vector<std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>> mapmul_cases = {
        {{4, 1, 3, 0, 2}, {3, 1, 4, 2, 0}, {0, 1, 2, 3, 4}},
        {{3, 1, 4, 2, 0}, {2, 3, 0, 1, 4}, {1, 3, 4, 0, 2}} };
    std::vector<std::vector<int>> nkstar_nbands_nbasis = {
            {1, 3, 4}, {2, 8, 11} };

    void set2d(int nbasis, int nbands)
    {
        pv.set_block_size(1);
        pv.set_proc_dim(dsize);
#ifdef __MPI
        pv.mpi_create_cart(MPI_COMM_WORLD);
        pv.set_local2global(nbasis, nbasis, ofs_running, ofs_running);
        pv.set_desc(nbasis, nbasis, pv.get_row_size());
        pv.set_global2local(nbasis, nbasis, true, ofs_running);
        pv.set_nloc_wfc_Eij(nbands, ofs_running, ofs_running);
        pv.set_desc_wfc_Eij(nbasis, nbands, pv.get_row_size());
#else
        pv.set_serial(nbasis, nbands);
        pv.set_global2local(nbasis, nbasis, false, ofs_running);
        pv.ncol_bands = nbands;
        pv.nloc_wfc = nbasis * nbands;
#endif
    }
    void set_int(std::complex<double>* s, int size)
    {
        for (int i = 0; i < size; i++)  s[i] = std::complex<double>(i, -i);
    }
    void set_rand(std::complex<double>* s, int size)
    {
        for (int i = 0; i < size; i++)
            s[i] = std::complex<double>(rand(), rand()) / double(RAND_MAX) * 10.0 - 5.0;
    }
    void set_int_posisym(std::complex<double>* s, int nbasis, int seed)
    {
        for (int i = 0; i < nbasis; i++)
            for (int j = i; j < nbasis; j++)
            {
                int diff = j - i;
                int sgn = diff % 2 ? -1 : 1;
                s[j * nbasis + i] = s[i * nbasis + j] = std::complex<double>(static_cast<double>((nbasis - diff) * sgn + seed), 0);
            }
    }
    void copy_from_global(const std::complex<double>* sg, std::complex<double>* sl, const int gr, const int gc, const int lr, const int lc, bool gcol_inside, bool lcol_inside)
    {
        // global is column major
        for (int i = 0;i < lr;++i)
            for (int j = 0;j < lc;++j)
                if (gcol_inside)
                    if (lcol_inside)
                        sl[i * lc + j] = sg[pv.local2global_row(i) * gc + pv.local2global_col(j)];
                    else
                        sl[j * lr + i] = sg[pv.local2global_row(i) * gc + pv.local2global_col(j)];
                else
                    if (lcol_inside)
                        sl[i * lc + j] = sg[pv.local2global_col(j) * gr + pv.local2global_row(i)];
                    else
                        sl[j * lr + i] = sg[pv.local2global_col(j) * gr + pv.local2global_row(i)];
    }
#ifdef __MPI
    void SetUp() override
    {
        MPI_Comm_size(MPI_COMM_WORLD, &dsize);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        this->ofs_running.open("log" + std::to_string(my_rank) + ".txt");
        ofs_running << "dsize(nproc) = " << dsize << std::endl;
        ofs_running << "my_rank = " << my_rank << std::endl;
    }
    void TearDown() override
    {
        ofs_running.close();
    }
#endif
};

TEST_F(SymExxTest, invmap)
{
    for (auto c : invmap_cases)
    {
        std::vector<int> invf = ModuleSymmetry::Symmetry::invmap(c.first.data(), c.first.size());
        EXPECT_EQ(invf, c.second);
    }
}

TEST_F(SymExxTest, mapmul)
{
    for (auto c : mapmul_cases)
    {
        std::vector<int> f2f1 = ModuleSymmetry::Symmetry::mapmul(std::get<0>(c).data(), std::get<1>(c).data(), std::get<0>(c).size());
        EXPECT_EQ(f2f1, std::get<2>(c));
    }
}

#ifdef __MPI
TEST_F(SymExxTest, restore_psik)
{
    //get from case
    for (auto& sizes : nkstar_nbands_nbasis)
    {
        int nbasis = sizes[2];
        int nbands = sizes[1];
        int nkstar = sizes[0];

        // setup 2d division for this size
        this->set2d(nbasis, nbands);

        // generate global symmitric matrix and copy to local
        std::vector<std::complex<double>> sfull_gk(nbasis * nbasis);
        this->set_int_posisym(sfull_gk.data(), nbasis, nkstar);

        std::vector<std::complex<double>> sloc_gk(pv.get_local_size());
        this->copy_from_global(sfull_gk.data(), sloc_gk.data(), nbasis, nbasis, pv.get_row_size(), pv.get_col_size(), false, false);

        std::vector<std::vector<std::complex<double>>> sfull_ks(nkstar, std::vector<std::complex<double>>(nbasis * nbasis));
        for (int ik = 0;ik < nkstar;++ik)this->set_int_posisym(sfull_ks[ik].data(), nbasis, ik);
        std::vector<std::vector<std::complex<double>>> sloc_ks(nkstar, std::vector<std::complex<double>>(pv.get_local_size()));

        for (int ik = 0;ik < nkstar;++ik)
            this->copy_from_global(sfull_ks[ik].data(), sloc_ks[ik].data(), nbasis, nbasis, pv.get_row_size(), pv.get_col_size(), false, false);

        // generate global psi and copy to local (both are row-major)
        int ikibz = 0;
        psi::Psi<std::complex<double>, psi::DEVICE_CPU> psi_full_gk(1, nbands, nbasis);
        this->set_int(psi_full_gk.get_pointer(), nbasis * nbands);
        psi::Psi<std::complex<double>, psi::DEVICE_CPU> psi_loc_gk(1, pv.ncol_bands, pv.get_row_size());
        this->copy_from_global(psi_full_gk.get_pointer(), psi_loc_gk.get_pointer(), nbasis, nbands, pv.get_row_size(), pv.ncol_bands, false, false);

        // run
        psi::Psi<std::complex<double>, psi::DEVICE_CPU> psi_loc_ks = ExxSym::restore_psik_scalapack(ikibz, psi_loc_gk, sloc_gk, sloc_ks, nbasis, nbands, pv);
        psi::Psi<std::complex<double>, psi::DEVICE_CPU> psi_full_ks = ExxSym::restore_psik_lapack(ikibz, psi_full_gk, sfull_gk, sfull_ks, nbasis, nbands);

        // check
        for (int ik = 0;ik < nkstar;ik++)
        {
            psi_loc_ks.fix_k(ik);
            psi_full_ks.fix_k(ik);
            for (int i = 0;i < pv.ncol_bands;i++)
                for (int j = 0;j < pv.get_row_size();j++)
                    EXPECT_NEAR(psi_loc_ks(i, j).real(), psi_full_ks(pv.local2global_col(i), pv.local2global_row(j)).real(), 1e-10);
        }
    }
}
#endif
int main(int argc, char** argv)
{
    srand(time(NULL));  // for random number generator
#ifdef __MPI
    MPI_Init(&argc, &argv);
#endif
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
#ifdef __MPI
    MPI_Finalize();
#endif
}