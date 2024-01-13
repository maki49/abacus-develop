#include "mpi.h"
#include "../symmetry_rotation.h"
#include  "gtest/gtest.h"
#define DOUBLETHRESHOLD 1e-8

/*

tested functions:
- wigner_d
- wigner_D
- ovlp_Ylm_Slm
- get_euler_angle
- cal_rotmat_Slm
- get_return_lattice
- cal_phase_factor_from_return_lattice

untested functions:
- cal_Ms (depending on UnitCell,K_Vectors, Parallel_2D)
- restore_dm  (depending on Ms)
- rot_matrix_ao (depending on Ms)

*/

// mock the useless functions
pseudo::pseudo() {}
pseudo::~pseudo() {}
Atom::Atom() {}
Atom::~Atom() {}
Atom_pseudo::Atom_pseudo() {}
Atom_pseudo::~Atom_pseudo() {}
UnitCell::UnitCell() {}
UnitCell::~UnitCell() {}
InfoNonlocal::InfoNonlocal() {}
InfoNonlocal::~InfoNonlocal() {}
Magnetism::Magnetism() {}
Magnetism::~Magnetism() {}

class SymmetryRotationTest : public testing::Test
{
protected:
    void SetUp() override
    {
        //init pv
        int myrank, dsize;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &dsize);
        pv.set_block_size(1);
        pv.set_proc_dim(dsize);
        pv.mpi_create_cart(MPI_COMM_WORLD);
        std::ofstream ofs;
        pv.set_local2global(matsize, matsize, ofs, ofs);
        pv.set_desc(matsize, matsize, pv.get_row_size());
        pv.set_global2local(matsize, matsize, true, ofs);
    }
    ModuleBase::Matrix3 C41 = ModuleBase::Matrix3(0, 1, 0, -1, 0, 0, 0, 0, 1);
    std::vector<std::complex<double>> wigerD_p_C41_ref = { ModuleBase::IMAG_UNIT, 0, 0, 0, 1, 0, 0, 0, -ModuleBase::IMAG_UNIT };
    std::vector<std::complex<double>> cmm_p_C41_ref = { -ModuleBase::IMAG_UNIT / sqrt(2), 0, -1 / sqrt(2), 0, 1, 0, -ModuleBase::IMAG_UNIT / sqrt(2), 0, 1 / sqrt(2) };
    std::vector<std::complex<double>> c_dagger_D_c_C41_ref = { 1, 0, 0, 0, 0, -1, 0, 1, 0 };
    ModuleSymmetry::Symmetry_rotation symrot;
    Parallel_2D pv;
    const int matsize = 5;  //2s1p
};

// inline void outmat(ModuleBase::ComplexMatrix& mat, int size, std::string name)
// {
//     std::cout << name << std::endl;
//     for (int i = 0;i < size;++i)
//     {
//         for (int j = 0;j < size;++j)std::cout << mat(i, j) << " ";
//         std::cout << std::endl;
//     }
// }
TEST_F(SymmetryRotationTest, Wignerd)
{
    EXPECT_NEAR(symrot.wigner_d(0, 1, 0, 0), 1, DOUBLETHRESHOLD);
    EXPECT_NEAR(symrot.wigner_d(0, 1, 1, 1), 1, DOUBLETHRESHOLD);
    EXPECT_NEAR(symrot.wigner_d(0, 1, -1, -1), 1, DOUBLETHRESHOLD);
}
TEST_F(SymmetryRotationTest, WignerD)
{
    ModuleBase::ComplexMatrix wignerD_p_C41(3, 3);
    int l = 1;
    for (int m1 = -l;m1 <= l;++m1)
        for (int m2 = -l;m2 <= l;++m2)
        {
            int i = m1 + l, j = m2 + l;
            wignerD_p_C41(i, j) = symrot.wigner_D(ModuleBase::Vector3<double>(0, 0, ModuleBase::PI / 2), 1, m1, m2, false);
            EXPECT_NEAR(wignerD_p_C41(i, j).real(), wigerD_p_C41_ref[i * 3 + j].real(), DOUBLETHRESHOLD);
            EXPECT_NEAR(wignerD_p_C41(i, j).imag(), wigerD_p_C41_ref[i * 3 + j].imag(), DOUBLETHRESHOLD);
            // alpha and gamma are the same when beta = 0
            wignerD_p_C41(i, j) = symrot.wigner_D(ModuleBase::Vector3<double>(ModuleBase::PI / 2, 0, 0), 1, m1, m2, false);
            EXPECT_NEAR(wignerD_p_C41(i, j).real(), wigerD_p_C41_ref[i * 3 + j].real(), DOUBLETHRESHOLD);
            EXPECT_NEAR(wignerD_p_C41(i, j).imag(), wigerD_p_C41_ref[i * 3 + j].imag(), DOUBLETHRESHOLD);
        }
    // outmat(wignerD_p_C41, 3, "wignerD_p_C41_cal");

}

TEST_F(SymmetryRotationTest, EulerAngle)
{
    ModuleBase::Vector3<double> euler_angle = symrot.get_euler_angle(C41);
    EXPECT_NEAR(euler_angle.x + euler_angle.z, ModuleBase::PI / 2, DOUBLETHRESHOLD);
    EXPECT_NEAR(euler_angle.y, 0, DOUBLETHRESHOLD);
}

TEST_F(SymmetryRotationTest, OvlpYS)
{
    ModuleBase::ComplexMatrix c_mm_p(3, 3);
    int l = 1;
    for (int m1 = -l;m1 <= l;++m1)
        for (int m2 = -l;m2 <= l;++m2)
        {
            int i = m1 + l, j = m2 + l;
            c_mm_p(i, j) = symrot.ovlp_Ylm_Slm(l, m1, m2);
            EXPECT_NEAR(c_mm_p(i, j).real(), cmm_p_C41_ref[i * 3 + j].real(), DOUBLETHRESHOLD);
            EXPECT_NEAR(c_mm_p(i, j).imag(), cmm_p_C41_ref[i * 3 + j].imag(), DOUBLETHRESHOLD);
        }
}

TEST_F(SymmetryRotationTest, RotMat)
{
    symrot.cal_rotmat_Slm(&C41, 1);
    ModuleBase::ComplexMatrix& rotmat = symrot.get_rotmat_Slm()[0][1];
    int l = 1;
    for (int m1 = -l;m1 <= l;++m1)
        for (int m2 = -l;m2 <= l;++m2)
        {
            int i = m1 + l, j = m2 + l;
            EXPECT_NEAR(rotmat(i, j).real(), c_dagger_D_c_C41_ref[i * 3 + j].real(), DOUBLETHRESHOLD);
            EXPECT_NEAR(rotmat(i, j).imag(), c_dagger_D_c_C41_ref[i * 3 + j].imag(), DOUBLETHRESHOLD);
        }
}

TEST_F(SymmetryRotationTest, GetReturnLattice)
{
    ModuleBase::Vector3<double> posd_a1(1. / 3., 1. / 3., 0.2);
    ModuleBase::Vector3<double> posd_a2(1. / 3., 1. / 3., -0.2);
    ModuleBase::Vector3<double> gtransd(0, 0, 0);
    ModuleBase::Matrix3 gmatd(-1, 1, 0, -1, 0, 0, 0, 0, -1);
    ModuleBase::Vector3<double> return_lattice = symrot.get_return_lattice(ModuleSymmetry::Symmetry(), gmatd, gtransd, posd_a1, posd_a2);
    EXPECT_NEAR(return_lattice.x, -1, DOUBLETHRESHOLD);
    EXPECT_NEAR(return_lattice.y, 0, DOUBLETHRESHOLD);
    EXPECT_NEAR(return_lattice.z, 0, DOUBLETHRESHOLD);
}

TEST_F(SymmetryRotationTest, PhaseFactor)
{
    ModuleBase::Vector3<double> posd_a1(1. / 3., 1. / 3., 0.2);
    ModuleBase::Vector3<double> posd_a2(1. / 3., 1. / 3., -0.2);
    ModuleBase::Vector3<double> kvec_d_ibz(0.5, -1. / 3. - 0.25);
    ModuleBase::Vector3<double> gtransd(0, 0, 0);
    ModuleSymmetry::Symmetry symm;
    symm.gmatrix[0] = ModuleBase::Matrix3(-1, 1, 0, -1, 0, 0, 0, 0, -1);
    symm.gtrans[0] = ModuleBase::Vector3<double>(0, 0, 0);
    std::complex<double> phase_factor = symrot.cal_phase_factor_from_return_attice(symm, posd_a1, posd_a2, 0, kvec_d_ibz);
    EXPECT_NEAR(phase_factor.real(), -1, DOUBLETHRESHOLD);
    EXPECT_NEAR(phase_factor.imag(), 0, DOUBLETHRESHOLD);
}

TEST_F(SymmetryRotationTest, SetBlockToMat2d)
{
    std::vector<std::complex<double>> obj_mat(pv.get_local_size());
    for (int j = 0;j < pv.get_col_size();++j)
        for (int i = 0;i < pv.get_row_size();++i)
            obj_mat[j * pv.get_row_size() + i] = std::complex<double>(static_cast<double>(pv.local2global_row(i)), static_cast<double>(pv.local2global_col(j)));
    ModuleBase::ComplexMatrix block(2, 2);
    block(0, 0) = 0; block(0, 1) = -1; block(1, 0) = -2; block(1, 1) = -3;
    symrot.set_block_to_mat2d(2, 3, block, obj_mat, pv);
    for (int i = 2;i < 4;++i)
        for (int j = 3;j < 5;++j)
        {
            int local_index = pv.global2local_col(j) * pv.get_row_size() + pv.global2local_row(i);
            if (pv.in_this_processor(i, j))
            {
                EXPECT_NEAR(obj_mat[local_index].real(), block(j - 3, i - 2).real(), DOUBLETHRESHOLD);
                EXPECT_NEAR(obj_mat[local_index].imag(), block(j - 3, i - 2).imag(), DOUBLETHRESHOLD);
            }
        }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
