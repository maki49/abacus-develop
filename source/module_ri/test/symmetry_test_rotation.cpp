#include "mpi.h"
#include "../symmetry_rotation.h"
#include <gtest/gtest.h>
/************************************************
 *  unit test of class Symmetry
 * 4. function: `symmetrize_vec3_nat`
 * 5. function `symmetrize_mat3`
 *
***********************************************/
#define DOUBLETHRESHOLD 1e-8
class SymmetryRotationTest : public testing::Test
{
protected:
    ModuleBase::Matrix3 C41 = ModuleBase::Matrix3(0, 1, 0, -1, 0, 0, 0, 0, 1);
    std::vector<std::complex<double>> wigerD_p_C41_ref = { ModuleBase::IMAG_UNIT, 0, 0, 0, 1, 0, 0, 0, -ModuleBase::IMAG_UNIT };
    std::vector<std::complex<double>> cmm_p_C41_ref = { -ModuleBase::IMAG_UNIT / sqrt(2), 0, -1 / sqrt(2), 0, 1, 0, -ModuleBase::IMAG_UNIT / sqrt(2), 0, 1 / sqrt(2) };
    std::vector<std::complex<double>> c_dagger_D_c_C41_ref = { 0, 0, 1, 0, 1, 0, -1, 0, 0 };
    ModuleSymmetry::Symmetry_rotation symrot;
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
    symrot.cal_rotmat_Slm(&C41, 1, 1);
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
int main(int argc, char** argv)
{
    srand(time(NULL));  // for random number generator
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
