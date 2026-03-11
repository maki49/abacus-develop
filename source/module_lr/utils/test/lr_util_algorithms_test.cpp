#include <gtest/gtest.h>

#include "../lr_util.h"
#include "../lr_util_print.h"

TEST(LR_Util, PsiWrapper)
{
    int nk = 2;
    int nbands = 5;
    int nbasis = 6;

    psi::Psi<float> k1(1, nbands, nk * nbasis);
    for (int i = 0;i < nbands * nk * nbasis;++i)k1.get_pointer()[i] = i;

    k1.fix_b(2);
    psi::Psi<float> bf = LR_Util::k1_to_bfirst_wrapper(k1, nk, nbasis);
    EXPECT_EQ(k1.get_current_k(), 0);
    EXPECT_EQ(k1.get_current_b(), 2); // invariance after wrapper
    EXPECT_EQ(bf.get_current_k(), 0);
    EXPECT_EQ(bf.get_current_b(), 0);

    bf.fix_kb(1, 3);
    psi::Psi<float> kb = LR_Util::bfirst_to_k1_wrapper(bf);
    EXPECT_EQ(bf.get_current_k(), 1);
    EXPECT_EQ(bf.get_current_b(), 3);
    EXPECT_EQ(kb.get_current_k(), 0);
    EXPECT_EQ(kb.get_current_b(), 0);


    EXPECT_EQ(bf.get_k_first(), false);
    EXPECT_EQ(bf.get_nk(), nk);
    EXPECT_EQ(bf.get_nbands(), nbands);
    EXPECT_EQ(bf.get_nbasis(), nbasis);

    EXPECT_EQ(kb.get_k_first(), true);
    EXPECT_EQ(kb.get_nk(), 1);
    EXPECT_EQ(kb.get_nbands(), nbands);
    EXPECT_EQ(kb.get_nbasis(), nk * nbasis);

    k1.fix_b(0);
    bf.fix_kb(0, 0);
    EXPECT_EQ(bf.get_pointer(), k1.get_pointer());
    EXPECT_EQ(bf.get_pointer(), kb.get_pointer());
    for (int ik = 0; ik < nk; ik++)
    {
        for (int ib = 0; ib < nbands; ib++)
        {
            bf.fix_kb(ik, ib);
            kb.fix_b(ib);
            k1.fix_b(ib);
            for (int ibasis = 0; ibasis < nbasis; ibasis++)
            {
                int ikb = ik * nbasis + ibasis;
                EXPECT_EQ(kb(ikb), bf(ibasis));
                EXPECT_EQ(k1(ikb), kb(ikb));
            }
        }
    }
}
#ifdef __MPI
void set_rand(double* data, int size) { for (int i = 0;i < size;++i) data[i] = double(rand()) / double(RAND_MAX) * 10.0 - 5.0; };
TEST(LR_Util, MatSymDouble)
{
    int n = 7;
    std::vector<double> din(n * n);
    set_rand(din.data(), n * n);
    std::vector<double> dref(n * n, 0.0);
    LR_Util::matsym(din.data(), n, dref.data());

    Parallel_2D pmat;
    LR_Util::setup_2d_division(pmat, 1, n, n);
    std::vector<double> din_local(pmat.get_local_size(), 0.0);
    for (int i = 0;i < pmat.get_row_size();++i)
        for (int j = 0;j < pmat.get_col_size();++j)
            din_local[j * pmat.get_row_size() + i] = din[pmat.local2global_col(j) * n + pmat.local2global_row(i)];

    std::vector<double> dout_local(pmat.get_local_size(), 0.0);
    LR_Util::matsym(din_local.data(), n, pmat, dout_local.data());
    for (int i = 0;i < pmat.get_row_size();++i)
        for (int j = 0;j < pmat.get_col_size();++j)
            EXPECT_DOUBLE_EQ(dout_local[j * pmat.get_row_size() + i], dref[pmat.local2global_col(j) * n + pmat.local2global_row(i)]);

    //in-place version
    LR_Util::matsym(din.data(), n);
    for (int i = 0;i < n * n;++i)
        EXPECT_DOUBLE_EQ(din[i], dref[i]);

    LR_Util::matsym(din_local.data(), n, pmat);
    for (int i = 0;i < pmat.get_local_size();++i)
        EXPECT_DOUBLE_EQ(din_local[i], dout_local[i]);
}

void set_rand(std::complex<double>* data, int size) { for (int i = 0;i < size;++i) data[i] = std::complex<double>(rand(), rand()) / double(RAND_MAX) * 10.0 - 5.0; };
TEST(LR_Util, MatSymComplex)
{
    int n = 5;
    std::vector<std::complex<double>> din(n * n);
    set_rand(din.data(), n * n);
    std::vector<std::complex<double>> dref(n * n, std::complex<double>(0.0, 0.0));
    LR_Util::matsym(din.data(), n, dref.data());

    Parallel_2D pmat;
    LR_Util::setup_2d_division(pmat, 1, n, n);
    std::vector<std::complex<double>> din_local(pmat.get_local_size(), std::complex<double>(0.0, 0.0));
    for (int i = 0;i < pmat.get_row_size();++i)
        for (int j = 0;j < pmat.get_col_size();++j)
            din_local[j * pmat.get_row_size() + i] = din[pmat.local2global_col(j) * n + pmat.local2global_row(i)];

    std::vector<std::complex<double>> dout_local(pmat.get_local_size(), std::complex<double>(0.0, 0.0));
    LR_Util::matsym(din_local.data(), n, pmat, dout_local.data());
    for (int i = 0;i < pmat.get_row_size();++i)
        for (int j = 0;j < pmat.get_col_size();++j)
        {
            EXPECT_DOUBLE_EQ(dout_local[j * pmat.get_row_size() + i].real(), dref[pmat.local2global_col(j) * n + pmat.local2global_row(i)].real());
            EXPECT_DOUBLE_EQ(dout_local[j * pmat.get_row_size() + i].imag(), dref[pmat.local2global_col(j) * n + pmat.local2global_row(i)].imag());
        }

    //in-place version
    LR_Util::matsym(din.data(), n);
    for (int i = 0;i < n * n;++i)
    {
        EXPECT_DOUBLE_EQ(din[i].real(), dref[i].real());
        EXPECT_DOUBLE_EQ(din[i].imag(), dref[i].imag());
    }

    LR_Util::matsym(din_local.data(), n, pmat);
    for (int i = 0;i < pmat.get_local_size();++i)
    {
        EXPECT_DOUBLE_EQ(din_local[i].real(), dout_local[i].real());
        EXPECT_DOUBLE_EQ(din_local[i].imag(), dout_local[i].imag());
    }
}

TEST(LR_Util, RWValue)
{
    const std::string file = "RWValue.txt";
    std::ofstream ofs(file);
    std::vector<int> vec(2 * 3 * 4 * 5, 0);
    for (int i = 0;i < vec.size();++i) vec[i] = i;
    LR_Util::write_value(ofs, vec.data(), 2, 3, 4, 5);
    ofs.close();

    std::vector<int> vec1(2 * 3 * 4 * 5, 0);
    std::ifstream ifs1(file);
    EXPECT_EQ(LR_Util::read_value(ifs1, vec1.data(), 2, 3, 4, 5), 120);
    ifs1.close();
    for (int i = 0;i < vec1.size();++i) { EXPECT_EQ(vec1[i], vec[i]); };
    std::vector<int> vec2(2 * 3 * 4 * 5, 0);
    std::ifstream ifs2(file);
    EXPECT_EQ(LR_Util::read_value(ifs2, vec2.data(), 2 * 3, 4 * 5), 120);
    ifs2.close();
    for (int i = 0;i < vec2.size();++i) { EXPECT_EQ(vec2[i], vec[i]); };
}

static void matmul2_real(const double* A, const double* V, double* AV) {
    AV[0] = A[0] * V[0] + A[2] * V[1];
    AV[1] = A[1] * V[0] + A[3] * V[1];
    AV[2] = A[0] * V[2] + A[2] * V[3];
    AV[3] = A[1] * V[2] + A[3] * V[3];
}
static void matmul2_complex(const std::complex<double>* A, const std::complex<double>* V, std::complex<double>* AV) {
    AV[0] = A[0] * V[0] + A[2] * V[1];
    AV[1] = A[1] * V[0] + A[3] * V[1];
    AV[2] = A[0] * V[2] + A[2] * V[3];
    AV[3] = A[1] * V[2] + A[3] * V[3];
}
static double resid2_real(const double* A, const double* V, const double* w) {
    double AV[4]; matmul2_real(A, V, AV);
    double VD[4] = { V[0] * w[0], V[1] * w[0], V[2] * w[1], V[3] * w[1] };
    double r_col = 0.0; for (int i = 0; i < 4; ++i) r_col += std::abs(AV[i] - VD[i]);

    double Vt[4] = { V[0], V[2], V[1], V[3] };
    matmul2_real(A, Vt, AV);
    VD[0] = Vt[0] * w[0]; VD[1] = Vt[1] * w[0]; VD[2] = Vt[2] * w[1]; VD[3] = Vt[3] * w[1];
    double r_row = 0.0; for (int i = 0; i < 4; ++i) r_row += std::abs(AV[i] - VD[i]);
    return std::min(r_col, r_row);
}
static double resid2_complex(const std::complex<double>* A, const std::complex<double>* V, const double* w) {
    std::complex<double> AV[4]; matmul2_complex(A, V, AV);
    std::complex<double> VD[4] = { V[0] * w[0], V[1] * w[0], V[2] * w[1], V[3] * w[1] };
    double r_col = 0.0; for (int i = 0; i < 4; ++i) r_col += std::abs(AV[i] - VD[i]);

    std::complex<double> Vt[4] = { V[0], V[2], V[1], V[3] };
    matmul2_complex(A, Vt, AV);
    VD[0] = Vt[0] * w[0]; VD[1] = Vt[1] * w[0]; VD[2] = Vt[2] * w[1]; VD[3] = Vt[3] * w[1];
    double r_row = 0.0; for (int i = 0; i < 4; ++i) r_row += std::abs(AV[i] - VD[i]);
    return std::min(r_col, r_row);
}
static void make_sym2(double* A) {
    A[0] = 2.0; A[1] = 1.0; A[2] = 1.0; A[3] = 3.0;
}
static void make_herm2(std::complex<double>* A) {
    A[0] = std::complex<double>(2.0, 0.0);
    A[1] = std::complex<double>(1.0, -1.0);
    A[2] = std::conj(A[1]);
    A[3] = std::complex<double>(3.0, 0.0);
}
TEST(LR_Util, DiagLapackZheevReal)
{
    const int n = 2;
    double A[4]; make_sym2(A);
    double A0[4]; std::copy(A, A + 4, A0);
    double w[2];
    LR_Util::diag_lapack_zheev(n, A, w);
    double t = 1e-10;
    double a = A0[0], c = A0[3], b = A0[1];
    double tr = a + c;
    double disc = std::sqrt((a - c) * (a - c) + 4.0 * b * b);
    double wexp0 = 0.5 * (tr - disc);
    double wexp1 = 0.5 * (tr + disc);
    EXPECT_NEAR(w[0], wexp0, t);
    EXPECT_NEAR(w[1], wexp1, t);
    const double r = resid2_real(A0, A, w);
    if (r > 1e-8)
    {
        std::cerr << "DiagLapackZheevReal residual=" << r << "\n";
        std::cerr << "A0: " << A0[0] << " " << A0[2] << " ; " << A0[1] << " " << A0[3] << "\n";
        std::cerr << "w: " << w[0] << " " << w[1] << "\n";
        std::cerr << "V(col-major): " << A[0] << " " << A[2] << " ; " << A[1] << " " << A[3] << "\n";
    }
    EXPECT_LE(r, 1e-8);
}
TEST(LR_Util, DiagLapackZheevComplex)
{
    const int n = 2;
    std::complex<double> A[4]; make_herm2(A);
    std::complex<double> A0[4]; std::copy(A, A + 4, A0);
    double w[2];
    LR_Util::diag_lapack_zheev(n, A, w);
    double t = 1e-10;
    double a = A0[0].real(), c = A0[3].real();
    double b2 = std::norm(A0[1]);
    double tr = a + c;
    double disc = std::sqrt((a - c) * (a - c) + 4.0 * b2);
    double wexp0 = 0.5 * (tr - disc);
    double wexp1 = 0.5 * (tr + disc);
    EXPECT_NEAR(w[0], wexp0, t);
    EXPECT_NEAR(w[1], wexp1, t);
    const double r = resid2_complex(A0, A, w);
    if (r > 1e-8)
    {
        std::cerr << "DiagLapackZheevComplex residual=" << r << "\n";
        std::cerr << "A0: (" << A0[0] << ") (" << A0[2] << ") ; (" << A0[1] << ") (" << A0[3] << ")\n";
        std::cerr << "w: " << w[0] << " " << w[1] << "\n";
        std::cerr << "V(col-major): (" << A[0] << ") (" << A[2] << ") ; (" << A[1] << ") (" << A[3] << ")\n";
    }
    EXPECT_LE(r, 1e-8);
}
TEST(LR_Util, DiagLapackZheevxReal)
{
    const int n = 2;
    double Aref[4]; make_sym2(Aref);
    double wref[2];
    double Aref_evec[4]; std::copy(Aref, Aref + 4, Aref_evec);
    LR_Util::diag_lapack_zheev(n, Aref_evec, wref);
    double A[4]; make_sym2(A);
    double w[2];
    LR_Util::diag_lapack_zheevx(n, A, w);
    EXPECT_NEAR(w[0], wref[0], 1e-10);
    EXPECT_NEAR(w[1], wref[1], 1e-10);
    EXPECT_LE(resid2_real(Aref, A, w), 1e-8);
}
TEST(LR_Util, DiagLapackZheevxComplex)
{
    const int n = 2;
    std::complex<double> Aref[4]; make_herm2(Aref);
    double wref[2];
    std::complex<double> Aref_evec[4]; std::copy(Aref, Aref + 4, Aref_evec);
    LR_Util::diag_lapack_zheev(n, Aref_evec, wref);
    std::complex<double> A[4]; make_herm2(A);
    double w[2];
    LR_Util::diag_lapack_zheevx(n, A, w);
    EXPECT_NEAR(w[0], wref[0], 1e-10);
    EXPECT_NEAR(w[1], wref[1], 1e-10);
    EXPECT_LE(resid2_complex(Aref, A, w), 1e-8);
}
TEST(LR_Util, DiagLapackZheevrReal)
{
    const int n = 2;
    double Aref[4]; make_sym2(Aref);
    double wref[2];
    double Aref_evec[4]; std::copy(Aref, Aref + 4, Aref_evec);
    LR_Util::diag_lapack_zheev(n, Aref_evec, wref);
    double A[4]; make_sym2(A);
    double w[2];
    LR_Util::diag_lapack_zheevr(n, A, w);
    EXPECT_NEAR(w[0], wref[0], 1e-10);
    EXPECT_NEAR(w[1], wref[1], 1e-10);
    EXPECT_LE(resid2_real(Aref, A, w), 1e-8);
}
TEST(LR_Util, DiagLapackZheevrComplex)
{
    const int n = 2;
    std::complex<double> Aref[4]; make_herm2(Aref);
    double wref[2];
    std::complex<double> Aref_evec[4]; std::copy(Aref, Aref + 4, Aref_evec);
    LR_Util::diag_lapack_zheev(n, Aref_evec, wref);
    std::complex<double> A[4]; make_herm2(A);
    double w[2];
    LR_Util::diag_lapack_zheevr(n, A, w);
    EXPECT_NEAR(w[0], wref[0], 1e-10);
    EXPECT_NEAR(w[1], wref[1], 1e-10);
    EXPECT_LE(resid2_complex(Aref, A, w), 1e-8);
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
#endif
