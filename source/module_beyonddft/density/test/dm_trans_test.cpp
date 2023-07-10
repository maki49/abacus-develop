#include <gtest/gtest.h>
#include "mpi.h"
#include "../dm_trans.h"
#ifdef __MPI
#include "module_beyonddft/utils/lr_util.h"
#endif
struct matsize
{
    int nsk = 1;
    int naos;
    int nocc;
    int nvirt;
    int nb = 1;
    matsize(int nsk, int naos, int nocc, int nvirt, int nb = 1)
        :nsk(nsk), naos(naos), nocc(nocc), nvirt(nvirt), nb(nb) {
        assert(nocc + nvirt <= naos);
    };
};

class DMTransTest : public testing::Test
{
public:
    std::vector<matsize> sizes{
        {2, 14, 9, 4},
        {2, 20, 10, 7}
    };  //why failed for 3, 9 cores? (-nan)
    std::ofstream ofs_running;
    int my_rank;
#ifdef __MPI
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        this->ofs_running.open("log" + std::to_string(my_rank) + ".txt");
        ofs_running << "my_rank = " << my_rank << std::endl;
    }
    void TearDown() override
    {
        ofs_running.close();
    }
#endif

    void set_ones(double* data, int size) { for (int i = 0;i < size;++i) data[i] = 1.0; };
    void set_int(double* data, int size) { for (int i = 0;i < size;++i) data[i] = static_cast<double>(i + 1); };
    void set_rand(double* data, int size) { for (int i = 0;i < size;++i) data[i] = double(rand()) / double(RAND_MAX) * 10.0 - 5.0; };
    void check_eq(double* data1, double* data2, int size) { for (int i = 0;i < size;++i) EXPECT_DOUBLE_EQ(data1[i], data2[i]); };
};

TEST_F(DMTransTest, DoubleSerial)
{

    for (auto s : this->sizes)
    {
        int size_c = s.nsk * (s.nocc + s.nvirt) * s.naos;
        int size_X = s.nsk * s.nocc * s.nvirt;
        psi::Psi<double, psi::DEVICE_CPU> c(s.nsk, s.nocc + s.nvirt, s.naos);
        psi::Psi<double, psi::DEVICE_CPU> X(s.nsk, s.nocc, s.nvirt);
        // set_rand(c.get_pointer(), size_c);
        // set_rand(X.get_pointer(), size_X);
        set_int(c.get_pointer(), size_c);
        set_int(X.get_pointer(), size_X);
        std::vector<ModuleBase::matrix> dm_for = hamilt::cal_dm_trans_forloop_serial(X, c);
        std::vector<ModuleBase::matrix> dm_blas = hamilt::cal_dm_trans_blas(X, c);
        for (int isk = 0;isk < s.nsk;++isk) check_eq(dm_for[isk].c, dm_blas[isk].c, s.naos * s.naos);
    }
}

#ifdef __MPI
TEST_F(DMTransTest, DoubleParallel)
{
    for (auto s : this->sizes)
    {
        // c: nao*nbands in para2d, nbands*nao in psi  (row-para and constructed: nao)
        // X: nvirt*nocc in para2d, nocc*nvirt in psi (row-para and constructed: nvirt)
        Parallel_2D px;
        LR_Util::setup_2d_division(px, s.nb, s.nvirt, s.nocc);
        psi::Psi<double, psi::DEVICE_CPU> X(s.nsk, px.get_col_size(), px.get_row_size());   //for pblas, col first??
        Parallel_2D pc;
        LR_Util::setup_2d_division(pc, s.nb, s.naos, s.nocc + s.nvirt, px.comm_2D, px.blacs_ctxt);
        psi::Psi<double, psi::DEVICE_CPU> c(s.nsk, pc.get_col_size(), pc.get_row_size());

        EXPECT_EQ(px.dim0, pc.dim0);
        EXPECT_EQ(px.dim1, pc.dim1);
        EXPECT_GE(s.nvirt, px.dim0);
        EXPECT_GE(s.nocc, px.dim1);
        EXPECT_GE(s.naos, pc.dim0);

        //set values
        for (int isk = 0;isk < s.nsk;++isk)
        {
            X.fix_k(isk);
            set_rand(X.get_pointer(), px.get_local_size());
            c.fix_k(isk);
            set_rand(c.get_pointer(), pc.get_local_size());
        }


        Parallel_2D pmat;
        std::vector<ModuleBase::matrix> dm_pblas_loc = hamilt::cal_dm_trans_pblas(X, px, c, pc, s.naos, s.nocc, s.nvirt, pmat);

        // gather dm and output
        std::vector<ModuleBase::matrix> dm_gather(s.nsk);
        for (int isk = 0;isk < s.nsk;++isk)
        {
            dm_gather[isk].create(s.naos, s.naos);
            LR_Util::gather_2d_to_full(pmat, dm_pblas_loc[isk].c, dm_gather[isk].c, false, s.naos, s.naos);
        }

        // compare to global matrix
        psi::Psi<double, psi::DEVICE_CPU> X_full(s.nsk, s.nocc, s.nvirt);
        psi::Psi<double, psi::DEVICE_CPU> c_full(s.nsk, s.nocc + s.nvirt, s.naos);
        for (int isk = 0;isk < s.nsk;++isk)
        {
            X.fix_k(isk);
            X_full.fix_k(isk);
            LR_Util::gather_2d_to_full(px, X.get_pointer(), X_full.get_pointer(), false, s.nvirt, s.nocc);
            c.fix_k(isk);
            c_full.fix_k(isk);
            LR_Util::gather_2d_to_full(pc, c.get_pointer(), c_full.get_pointer(), false, s.naos, s.nocc + s.nvirt);
        }

        if (my_rank == 0)
        {
            std::vector<ModuleBase::matrix> dm_full = hamilt::cal_dm_trans_blas(X_full, c_full);
            for (int isk = 0;isk < s.nsk;++isk)
                for (int i = 0;i < s.naos;++i)
                    for (int j = 0;j < s.naos;++j)
                        EXPECT_NEAR(dm_full[isk](i, j), dm_gather[isk](i, j), 1e-10);
        }
    }
}
#endif


int main(int argc, char** argv)
{
    srand(time(NULL));  // for random number generator
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}