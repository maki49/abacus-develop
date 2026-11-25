#include "../utils/lr_util.h"
#include "source_cell/unitcell.h"
#include <gtest/gtest.h>
#include <vector>

TEST(LRForcePP, AllowsForcesAndRelaxationWithoutNLCC)
{
    UnitCell cell;
    cell.ntype = 1;
    cell.atoms = new Atom[1];
    cell.set_atom_flag = true;
    EXPECT_NO_THROW(LR_Util::check_force_pp(cell, true, "scf"));
    EXPECT_NO_THROW(LR_Util::check_force_pp(cell, true, "relax"));
}

TEST(LRForcePP, AllowsNLCCTDDFTSpectra)
{
    UnitCell cell;
    cell.ntype = 2;
    cell.atoms = new Atom[2];
    cell.set_atom_flag = true;
    cell.pseudo_fn = {"H.upf", "Si_NLCC.upf"};
    cell.atoms[1].ncpp.nlcc = true;
    EXPECT_NO_THROW(LR_Util::check_force_pp(cell, false, "scf"));
    EXPECT_NO_THROW(LR_Util::check_force_pp(cell, false, "nscf"));
}

TEST(LRForcePP, RejectsNLCCForcesAndIdentifiesThePseudopotential)
{
    UnitCell cell;
    cell.ntype = 2;
    cell.atoms = new Atom[2];
    cell.set_atom_flag = true;
    cell.pseudo_fn = {"H.upf", "Si_NLCC.upf"};
    cell.atoms[1].ncpp.nlcc = true;
    cell.atoms[1].label = "Si";
    try
    {
        LR_Util::check_force_pp(cell, true, "scf");
        FAIL() << "NLCC forces were accepted";
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("element Si"), std::string::npos);
        EXPECT_NE(message.find("Si_NLCC.upf"), std::string::npos);
        EXPECT_NE(message.find("not implemented"), std::string::npos);
    }
}

TEST(LRForcePP, RejectsNLCCRelaxationEvenWithoutExplicitForceFlag)
{
    UnitCell cell;
    cell.ntype = 2;
    cell.atoms = new Atom[2];
    cell.set_atom_flag = true;
    cell.pseudo_fn = {"H.upf", "Si_NLCC.upf"};
    cell.atoms[1].ncpp.nlcc = true;
    EXPECT_THROW(LR_Util::check_force_pp(cell, false, "relax"), std::invalid_argument);
    EXPECT_THROW(LR_Util::check_force_pp(cell, true, "relax"), std::invalid_argument);
}

TEST(LR_XCSigma, TargetsValidatedLibxcKernels)
{
    EXPECT_TRUE(LR_Util::uses_xc_sigma_floor("hse"));
    EXPECT_TRUE(LR_Util::uses_xc_sigma_floor("lc_pbe"));
    EXPECT_TRUE(LR_Util::uses_xc_sigma_floor("lc_wpbe"));
    EXPECT_FALSE(LR_Util::uses_xc_sigma_floor("lrc_wpbe"));
    EXPECT_FALSE(LR_Util::uses_xc_sigma_floor("lrc_wpbeh"));
    EXPECT_FALSE(LR_Util::uses_xc_sigma_floor("cam_pbeh"));
    EXPECT_FALSE(LR_Util::uses_xc_sigma_floor("pbe"));
}

TEST(LR_XCSigma, OnlyFloorsSameSpinAndRetainsSignedMixedDerivative)
{
    const std::vector<double> sigma{4., -2., 1., 0.5e-6, 0.2e-6, 2e-6};
    std::vector<double> buffer;
    std::vector<double> jacobian;
    const auto& input = LR_Util::prepare_xc_sigma(sigma, 2, true, buffer, &jacobian);
    EXPECT_EQ(input, (std::vector<double>{4., -2., 1., 1e-6, 0.2e-6, 2e-6}));
    EXPECT_EQ(jacobian, (std::vector<double>{1., 1., 1., 0., 1., 1.}));
    EXPECT_EQ(sigma[3], 0.5e-6);

    const auto& original = LR_Util::prepare_xc_sigma(sigma, 2, false, buffer, &jacobian);
    EXPECT_EQ(&original, &sigma);
    EXPECT_TRUE(jacobian.empty());
}

TEST(LR_XCSigma, FloorsScalarSigmaWithZeroDerivativeAtTheJunction)
{
    const std::vector<double> sigma{0.5e-6, 1e-6, 2e-6};
    std::vector<double> buffer;
    std::vector<double> jacobian;
    const auto& input = LR_Util::prepare_xc_sigma(sigma, 1, true, buffer, &jacobian);
    EXPECT_EQ(input, (std::vector<double>{1e-6, 1e-6, 2e-6}));
    EXPECT_EQ(jacobian, (std::vector<double>{0., 0., 1.}));

    const std::vector<double> empty;
    const auto& empty_result = LR_Util::prepare_xc_sigma(empty, 1, true, buffer, &jacobian);
    EXPECT_TRUE(empty_result.empty());
    EXPECT_TRUE(jacobian.empty());
}

TEST(LRForceWindow, CompleteSpinResolvedWindowsNeedNoWarning)
{
    const std::vector<int> virtuals{11, 13};
    EXPECT_TRUE(LR_Util::force_window_warning(17, 17, 0, virtuals, virtuals).empty());
}

TEST(LRForceWindow, ReportsVirtualTruncationAndForceLimitation)
{
    const std::vector<int> x{1, 3};
    const std::vector<int> z{14, 16};
    const std::string message = LR_Util::force_window_warning(17, 17, 0, x, z);
    EXPECT_NE(message.find("virtual X window is truncated"), std::string::npos);
    EXPECT_EQ(message.find("occupied X window is truncated"), std::string::npos);
    EXPECT_NE(message.find("finite differences"), std::string::npos);
    EXPECT_NE(message.find("nbands = NLOCAL (17)"), std::string::npos);
}

TEST(LRForceWindow, EqualXAndZDoNotHideOccupiedOrBandTruncation)
{
    const std::vector<int> virtuals{8};
    const std::string occupied = LR_Util::force_window_warning(17, 17, 2, virtuals, virtuals);
    EXPECT_NE(occupied.find("occupied X window is truncated"), std::string::npos);
    const std::string bands = LR_Util::force_window_warning(17, 11, 0, virtuals, virtuals);
    EXPECT_NE(bands.find("ground-state band space is incomplete"), std::string::npos);
}

TEST(LRSpinPopulation, RoundAfterSummingFractionalOccupations)
{
    ModuleBase::matrix occupations(2, 3);
    occupations(0, 0) = 1.0;
    occupations(0, 1) = 1.0;
    occupations(0, 2) = 1.0;
    occupations(1, 0) = 1.0;
    occupations(1, 1) = 0.5000000052;
    occupations(1, 2) = 0.4999999947;
    EXPECT_EQ(LR_Util::spin_population_difference(occupations, 1, 0), 1);
    std::swap(occupations(1, 1), occupations(1, 2));
    EXPECT_EQ(LR_Util::spin_population_difference(occupations, 1, 0), 1);
}

TEST(LRSpinPopulation, SumKPointsBeforeRounding)
{
    ModuleBase::matrix occupations(4, 2);
    occupations(0, 0) = 0.5;
    occupations(1, 0) = 0.5;
    occupations(0, 1) = 0.5;
    occupations(1, 1) = 0.5;
    occupations(2, 0) = 0.5;
    occupations(3, 0) = 0.5;
    EXPECT_EQ(LR_Util::spin_population_difference(occupations, 1, 0), 1);
}

namespace
{
template<typename T> void check_eigensystem(const std::vector<T>& original)
{
    const int n = 2;
    std::vector<T> eigenvectors = original;
    std::vector<double> eigenvalues(n);
    LR_Util::diag_lapack(n, eigenvectors.data(), eigenvalues.data());
    for (int band = 0; band < n; ++band)
    {
        for (int row = 0; row < n; ++row)
        {
            T product = T(0.0);
            for (int col = 0; col < n; ++col)
            {
                product += original[col * n + row] * eigenvectors[band * n + col];
            }
            const T expected = eigenvalues[band] * eigenvectors[band * n + row];
            EXPECT_NEAR(std::abs(product - expected), 0.0, 1e-12);
        }
    }
    const double lower = (5.0 - std::sqrt(5.0)) / 2.0;
    const double upper = (5.0 + std::sqrt(5.0)) / 2.0;
    EXPECT_NEAR(eigenvalues[0], lower, 1e-12);
    EXPECT_NEAR(eigenvalues[1], upper, 1e-12);
}
}

TEST(LREigensolver, RealWorkspaceRetainsEigenpairs)
{
    const std::vector<double> matrix{2.0, 1.0, 1.0, 3.0};
    check_eigensystem(matrix);
}

TEST(LREigensolver, ComplexWorkspaceRetainsEigenpairs)
{
    using Complex = std::complex<double>;
    const std::vector<Complex> matrix{Complex(2.0, 0.0), Complex(0.0, 1.0),
        Complex(0.0, -1.0), Complex(3.0, 0.0)};
    check_eigensystem(matrix);
}

TEST(LREigensolver, RealGeneralEigenproblemKeepsComplexPair)
{
    const int n = 2;
    std::vector<double> matrix{0.0, 1.0, -1.0, 0.0};
    std::vector<std::complex<double>> eigenvalues(n);
    LR_Util::diag_lapack_nh(n, matrix.data(), eigenvalues.data());
    for (const auto& eigenvalue : eigenvalues)
    {
        EXPECT_NEAR(eigenvalue.real(), 0.0, 1e-12);
        EXPECT_NEAR(std::abs(eigenvalue.imag()), 1.0, 1e-12);
    }
    EXPECT_NEAR(std::abs(eigenvalues[0] + eigenvalues[1]), 0.0, 1e-12);
}

TEST(LREigensolver, ComplexGeneralEigenproblemKeepsEigenvalues)
{
    using Complex = std::complex<double>;
    const int n = 2;
    std::vector<Complex> matrix{Complex(1.0, 2.0), Complex(0.0, 0.0),
        Complex(0.7, -0.4), Complex(3.0, -1.0)};
    std::vector<Complex> eigenvalues(n);
    LR_Util::diag_lapack_nh(n, matrix.data(), eigenvalues.data());
    EXPECT_NEAR(std::abs(eigenvalues[0] - Complex(1.0, 2.0)), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(eigenvalues[1] - Complex(3.0, -1.0)), 0.0, 1e-12);
}

TEST(LRLinearSolver, ReportsSingularRealFactorization)
{
    const double matrix[] = {1.0, 2.0, 2.0, 4.0};
    const double rhs[] = {1.0, 1.0};
    double result[] = {0.0, 0.0};
    const int info = LR_Util::lapack_linear_solver(matrix, result, rhs, 2, 1);
    EXPECT_GT(info, 0);
}

TEST(LRLinearSolver, ReportsSingularComplexFactorization)
{
    using Complex = std::complex<double>;
    const Complex matrix[] = {Complex(1.0), Complex(2.0), Complex(2.0), Complex(4.0)};
    const Complex rhs[] = {Complex(1.0), Complex(1.0)};
    Complex result[] = {Complex(0.0), Complex(0.0)};
    const int info = LR_Util::lapack_linear_solver(matrix, result, rhs, 2, 1);
    EXPECT_GT(info, 0);
}
