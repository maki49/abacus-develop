#include <gtest/gtest.h>
#include "mpi.h"
#include "../CVCX.h"

#include "module_lr/utils/lr_util.h"

struct matsize
{
    int nks = 1;
    int naos;
    int nocc;
    int nvirt;
    int nb = 1;
    matsize(int nks, int naos, int nocc, int nvirt, int nb = 1)
        :nks(nks), naos(naos), nocc(nocc), nvirt(nvirt), nb(nb) {
        assert(nocc + nvirt <= naos);
    };
};

class AXTest : public testing::Test
{
public:
    std::vector<matsize> sizes{
        // {2, 3, 2, 1},
        {2, 13, 7, 4},
        {2, 14, 8, 5}
    };
    int nstate = 2;
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
    void set_int(std::complex<double>* data, int size) { for (int i = 0;i < size;++i) data[i] = std::complex<double>(i + 1, -i - 1); };
    void set_rand(double* data, int size) { for (int i = 0;i < size;++i) data[i] = double(rand()) / double(RAND_MAX) * 10.0 - 5.0; };
    void set_rand(std::complex<double>* data, int size) { for (int i = 0;i < size;++i) data[i] = std::complex<double>(rand(), rand()) / double(RAND_MAX) * 10.0 - 5.0; };
    void check_eq(double* data1, double* data2, int size) { for (int i = 0;i < size;++i) EXPECT_NEAR(data1[i], data2[i], 1e-8); };
    void check_eq(std::complex<double>* data1, std::complex<double>* data2, int size)
    {
        for (int i = 0;i < size;++i)
        {
            EXPECT_NEAR(data1[i].real(), data2[i].real(), 1e-8);
            EXPECT_NEAR(data1[i].imag(), data2[i].imag(), 1e-8);
        }
    };
};

TEST_F(AXTest, DoubleSerial)
{
    for (auto s : this->sizes)
    {
        psi::Psi<double, base_device::DEVICE_CPU> X(s.nks, nstate, s.nocc * s.nvirt, nullptr, false);
        psi::Psi<double, base_device::DEVICE_CPU> AX_for(s.nks, nstate, s.nocc * s.nvirt, nullptr, false);
        psi::Psi<double, base_device::DEVICE_CPU> AX_blas(s.nks, nstate, s.nocc * s.nvirt, nullptr, false);
        const int size_x = nstate * s.nks * s.nocc * s.nvirt;
        set_rand(X.get_pointer(), size_x);

        const int size_c = s.nks * (s.nocc + s.nvirt) * s.naos;
        const int size_v = s.naos * s.naos;
        for (int istate = 0;istate < nstate;++istate)
        {
            psi::Psi<double, base_device::DEVICE_CPU> c(s.nks, s.nocc + s.nvirt, s.naos);
            std::vector<container::Tensor> V(s.nks, container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, { s.naos, s.naos }));
            set_rand(c.get_pointer(), size_c);
            for (auto& v : V)set_rand(v.data<double>(), size_v);
            X.fix_b(istate);
            AX_for.fix_b(istate);
            AX_blas.fix_b(istate);
            // occ
            CVCX_occ_forloop_serial(V, c, X, s.naos, s.nocc, s.nvirt, AX_for);
            CVCX_occ_blas(V, c, X, s.naos, s.nocc, s.nvirt, AX_blas, false);
            AX_for.fix_k(0);
            AX_blas.fix_k(0);
            check_eq(AX_for.get_pointer(), AX_blas.get_pointer(), s.nks * s.nocc * s.nvirt);
            // virt
            CVCX_virt_forloop_serial(V, c, X, s.naos, s.nocc, s.nvirt, AX_for);
            CVCX_virt_blas(V, c, X, s.naos, s.nocc, s.nvirt, AX_blas, false);
            AX_for.fix_k(0);
            AX_blas.fix_k(0);
            check_eq(AX_for.get_pointer(), AX_blas.get_pointer(), s.nks * s.nocc * s.nvirt);
        }
    }
}

TEST_F(AXTest, ComplexSerial)
{
    for (auto s : this->sizes)
    {
        psi::Psi<std::complex<double>, base_device::DEVICE_CPU> X(s.nks, nstate, s.nocc * s.nvirt, nullptr, false);
        psi::Psi<std::complex<double>, base_device::DEVICE_CPU> AX_for(s.nks, nstate, s.nocc * s.nvirt, nullptr, false);
        psi::Psi<std::complex<double>, base_device::DEVICE_CPU> AX_blas(s.nks, nstate, s.nocc * s.nvirt, nullptr, false);
        const int size_x = nstate * s.nks * s.nocc * s.nvirt;
        set_rand(X.get_pointer(), size_x);

        int size_c = s.nks * (s.nocc + s.nvirt) * s.naos;
        int size_v = s.naos * s.naos;
        for (int istate = 0;istate < nstate;++istate)
        {
            psi::Psi<std::complex<double>, base_device::DEVICE_CPU> c(s.nks, s.nocc + s.nvirt, s.naos);
            std::vector<container::Tensor> V(s.nks, container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { s.naos, s.naos }));
            set_rand(c.get_pointer(), size_c);
            for (auto& v : V)set_rand(v.data<std::complex<double>>(), size_v);
            X.fix_b(istate);
            AX_for.fix_b(istate);
            AX_blas.fix_b(istate);
            // occ
            CVCX_occ_forloop_serial(V, c, X, s.naos, s.nocc, s.nvirt, AX_for);
            CVCX_occ_blas(V, c, X, s.naos, s.nocc, s.nvirt, AX_blas, false);
            AX_for.fix_k(0);
            AX_blas.fix_k(0);
            check_eq(AX_for.get_pointer(), AX_blas.get_pointer(), s.nks * s.nocc * s.nvirt);
            // virt
            CVCX_virt_forloop_serial(V, c, X, s.naos, s.nocc, s.nvirt, AX_for);
            CVCX_virt_blas(V, c, X, s.naos, s.nocc, s.nvirt, AX_blas, false);
            AX_for.fix_k(0);
            AX_blas.fix_k(0);
            check_eq(AX_for.get_pointer(), AX_blas.get_pointer(), s.nks * s.nocc * s.nvirt);
        }
    }
}
#ifdef __MPI
TEST_F(AXTest, DoubleParallel)
{
    for (auto s : this->sizes)
    {
        // c: nao*nbands in para2d, nbands*nao in psi  (row-para and constructed: nao)
        // X: nvirt*nocc in para2d, nocc*nvirt in psi (row-para and constructed: nvirt)
        Parallel_2D pV;
        LR_Util::setup_2d_division(pV, s.nb, s.naos, s.naos);
        std::vector<container::Tensor> V(s.nks, container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, { pV.get_col_size(), pV.get_row_size() }));
        Parallel_2D pc;
        LR_Util::setup_2d_division(pc, s.nb, s.naos, s.nocc + s.nvirt, pV.blacs_ctxt);
        psi::Psi<double, base_device::DEVICE_CPU> c(s.nks, pc.get_col_size(), pc.get_row_size());
        Parallel_2D px;
        LR_Util::setup_2d_division(px, s.nb, s.nvirt, s.nocc, pV.blacs_ctxt);

        EXPECT_EQ(pV.dim0, pc.dim0);
        EXPECT_EQ(pV.dim1, pc.dim1);
        EXPECT_GE(s.nvirt, px.dim0);
        EXPECT_GE(s.nocc, px.dim1);
        EXPECT_GE(s.naos, pc.dim0);

        psi::Psi<double, base_device::DEVICE_CPU> AX_pblas_loc(s.nks, nstate, px.get_local_size());
        psi::Psi<double, base_device::DEVICE_CPU> AX_gather(s.nks, nstate, s.nocc * s.nvirt, nullptr, false);

        //set X and X_full
        psi::Psi<double, base_device::DEVICE_CPU> X(s.nks, nstate, px.get_local_size(), nullptr, false);
        set_rand(X.get_pointer(), nstate * s.nks * px.get_local_size());
        psi::Psi<double, base_device::DEVICE_CPU> X_full(s.nks, nstate, s.nocc * s.nvirt, nullptr, false);        // allocate X_full
        for (int istate = 0;istate < nstate;++istate)
        {
            X.fix_b(istate);
            X_full.fix_b(istate);
            for (int isk = 0;isk < s.nks;++isk)
            {
                X.fix_k(isk);
                X_full.fix_k(isk);
                LR_Util::gather_2d_to_full(px, X.get_pointer(), X_full.get_pointer(), false, s.nvirt, s.nocc);
            }
        }

        for (int istate = 0;istate < nstate;++istate)
        {
            for (int isk = 0;isk < s.nks;++isk)
            {
                set_rand(V.at(isk).data<double>(), pV.get_local_size());
                c.fix_k(isk);
                set_rand(c.get_pointer(), pc.get_local_size());
            }
            X.fix_b(istate);
            X_full.fix_b(istate);
            AX_pblas_loc.fix_b(istate);
            AX_gather.fix_b(istate);
            CVCX_occ_pblas(V, pV, c, pc, X, px, s.naos, s.nocc, s.nvirt, AX_pblas_loc, false);
            // gather AX and output
            for (int isk = 0;isk < s.nks;++isk)
            {
                AX_pblas_loc.fix_k(isk);
                AX_gather.fix_k(isk);
                LR_Util::gather_2d_to_full(px, AX_pblas_loc.get_pointer(), AX_gather.get_pointer(), false/*pblas: row first*/, s.nvirt, s.nocc);
            }
            // compare to global AX
            std::vector<container::Tensor> V_full(s.nks, container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, { s.naos, s.naos }));
            psi::Psi<double, base_device::DEVICE_CPU> c_full(s.nks, s.nocc + s.nvirt, s.naos);
            for (int isk = 0;isk < s.nks;++isk)
            {
                LR_Util::gather_2d_to_full(pV, V.at(isk).data<double>(), V_full.at(isk).data<double>(), false, s.naos, s.naos);
                c.fix_k(isk);
                c_full.fix_k(isk);
                LR_Util::gather_2d_to_full(pc, c.get_pointer(), c_full.get_pointer(), false, s.naos, s.nocc + s.nvirt);
            }
            if (my_rank == 0)
            {
                psi::Psi<double, base_device::DEVICE_CPU>  AX_full_istate(s.nks, 1, s.nocc * s.nvirt, nullptr, false);
                CVCX_occ_blas(V_full, c_full, X_full, s.naos, s.nocc, s.nvirt, AX_full_istate, false);
                AX_full_istate.fix_b(0);
                AX_gather.fix_b(istate);
                check_eq(AX_full_istate.get_pointer(), AX_gather.get_pointer(), s.nks * s.nocc * s.nvirt);
            }

            // //============ the same for virtual  ==========
            CVCX_virt_pblas(V, pV, c, pc, X, px, s.naos, s.nocc, s.nvirt, AX_pblas_loc, false);
            for (int isk = 0;isk < s.nks;++isk)
            {
                AX_pblas_loc.fix_k(isk);
                AX_gather.fix_k(isk);
                LR_Util::gather_2d_to_full(px, AX_pblas_loc.get_pointer(), AX_gather.get_pointer(), false/*pblas: row first*/, s.nvirt, s.nocc);
            }
            if (my_rank == 0)
            {
                psi::Psi<double, base_device::DEVICE_CPU>  AX_full_istate(s.nks, 1, s.nocc * s.nvirt, nullptr, false);
                CVCX_virt_blas(V_full, c_full, X_full, s.naos, s.nocc, s.nvirt, AX_full_istate, false);
                AX_full_istate.fix_b(0);
                AX_gather.fix_b(istate);
                check_eq(AX_full_istate.get_pointer(), AX_gather.get_pointer(), s.nks * s.nocc * s.nvirt);
            }
        }
    }
}
TEST_F(AXTest, ComplexParallel)
{
    for (auto s : this->sizes)
    {
        // c: nao*nbands in para2d, nbands*nao in psi  (row-para and constructed: nao)
        // X: nvirt*nocc in para2d, nocc*nvirt in psi (row-para and constructed: nvirt)
        Parallel_2D pV;
        LR_Util::setup_2d_division(pV, s.nb, s.naos, s.naos);
        std::vector<container::Tensor> V(s.nks, container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { pV.get_col_size(), pV.get_row_size() }));
        Parallel_2D pc;
        LR_Util::setup_2d_division(pc, s.nb, s.naos, s.nocc + s.nvirt, pV.blacs_ctxt);
        psi::Psi<std::complex<double>, base_device::DEVICE_CPU> c(s.nks, pc.get_col_size(), pc.get_row_size());
        Parallel_2D px;
        LR_Util::setup_2d_division(px, s.nb, s.nvirt, s.nocc, pV.blacs_ctxt);

        psi::Psi<std::complex<double>, base_device::DEVICE_CPU> AX_pblas_loc(s.nks, nstate, px.get_local_size());
        psi::Psi<std::complex<double>, base_device::DEVICE_CPU> AX_gather(s.nks, nstate, s.nocc * s.nvirt, nullptr, false);

        //set X and X_full
        psi::Psi<std::complex<double>, base_device::DEVICE_CPU> X(s.nks, nstate, px.get_local_size(), nullptr, false);
        set_rand(X.get_pointer(), nstate * s.nks * px.get_local_size());
        psi::Psi<std::complex<double>, base_device::DEVICE_CPU> X_full(s.nks, nstate, s.nocc * s.nvirt, nullptr, false);        // allocate X_full
        for (int istate = 0;istate < nstate;++istate)
        {
            X.fix_b(istate);
            X_full.fix_b(istate);
            for (int isk = 0;isk < s.nks;++isk)
            {
                X.fix_k(isk);
                X_full.fix_k(isk);
                LR_Util::gather_2d_to_full(px, X.get_pointer(), X_full.get_pointer(), false, s.nvirt, s.nocc);
            }
        }

        for (int istate = 0;istate < nstate;++istate)
        {
            for (int isk = 0;isk < s.nks;++isk)
            {
                set_rand(V.at(isk).data<std::complex<double>>(), pV.get_local_size());
                c.fix_k(isk);
                set_rand(c.get_pointer(), pc.get_local_size());
            }
            X.fix_b(istate);
            X_full.fix_b(istate);
            AX_pblas_loc.fix_b(istate);
            AX_gather.fix_b(istate);
            CVCX_occ_pblas(V, pV, c, pc, X, px, s.naos, s.nocc, s.nvirt, AX_pblas_loc, false);

            // gather AX and output
            for (int isk = 0;isk < s.nks;++isk)
            {
                AX_pblas_loc.fix_k(isk);
                AX_gather.fix_k(isk);
                LR_Util::gather_2d_to_full(px, AX_pblas_loc.get_pointer(), AX_gather.get_pointer(), false/*pblas: row first*/, s.nvirt, s.nocc);
            }
            // compare to global AX
            std::vector<container::Tensor> V_full(s.nks, container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { s.naos, s.naos }));
            psi::Psi<std::complex<double>, base_device::DEVICE_CPU> c_full(s.nks, s.nocc + s.nvirt, s.naos);
            for (int isk = 0;isk < s.nks;++isk)
            {
                LR_Util::gather_2d_to_full(pV, V.at(isk).data<std::complex<double>>(), V_full.at(isk).data<std::complex<double>>(), false, s.naos, s.naos);
                c.fix_k(isk);
                c_full.fix_k(isk);
                LR_Util::gather_2d_to_full(pc, c.get_pointer(), c_full.get_pointer(), false, s.naos, s.nocc + s.nvirt);
            }
            if (my_rank == 0)
            {
                psi::Psi<std::complex<double>, base_device::DEVICE_CPU>  AX_full_istate(s.nks, 1, s.nocc * s.nvirt, nullptr, false);
                CVCX_occ_blas(V_full, c_full, X_full, s.naos, s.nocc, s.nvirt, AX_full_istate, false);
                AX_full_istate.fix_b(0);
                AX_gather.fix_b(istate);
                check_eq(AX_full_istate.get_pointer(), AX_gather.get_pointer(), s.nks * s.nocc * s.nvirt);
            }
            // //============ the same for virtual  ==========
            CVCX_virt_pblas(V, pV, c, pc, X, px, s.naos, s.nocc, s.nvirt, AX_pblas_loc, false);
            for (int isk = 0;isk < s.nks;++isk)
            {
                AX_pblas_loc.fix_k(isk);
                AX_gather.fix_k(isk);
                LR_Util::gather_2d_to_full(px, AX_pblas_loc.get_pointer(), AX_gather.get_pointer(), false/*pblas: row first*/, s.nvirt, s.nocc);
            }
            if (my_rank == 0)
            {
                psi::Psi<std::complex<double>, base_device::DEVICE_CPU>  AX_full_istate(s.nks, 1, s.nocc * s.nvirt, nullptr, false);
                CVCX_virt_blas(V_full, c_full, X_full, s.naos, s.nocc, s.nvirt, AX_full_istate, false);
                AX_full_istate.fix_b(0);
                AX_gather.fix_b(istate);
                check_eq(AX_full_istate.get_pointer(), AX_gather.get_pointer(), s.nks * s.nocc * s.nvirt);
            }
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