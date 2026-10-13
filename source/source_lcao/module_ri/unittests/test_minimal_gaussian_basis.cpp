#include "../minimal_gaussian_basis.h"
#include "source_base/math_integral.h"
#include <gtest/gtest.h>
#include <limits>

TEST(MinimalGaussianBasis, LiteratureExponents)
{
    EXPECT_NEAR(LR::minimal_gaussian_exponent(1, 0.2), 0.1999828038466018, 1e-15);
    EXPECT_NEAR(LR::minimal_gaussian_exponent(6, 0.2), 0.1320292535005648, 1e-15);
    EXPECT_NEAR(LR::minimal_gaussian_exponent(8, 0.2), 0.2587932305664396, 1e-15);
    for (int z = 1; z <= 103; ++z)
    {
        const double ris = LR::minimal_gaussian_exponent(z, 0.2);
        EXPECT_NEAR(LR::minimal_gaussian_exponent(z, 0.6), 3.0 * ris, 1e-14);
    }
    EXPECT_THROW(LR::minimal_gaussian_exponent(0, 0.6), std::invalid_argument);
    EXPECT_THROW(LR::minimal_gaussian_exponent(104, 0.6), std::invalid_argument);
    EXPECT_THROW(LR::minimal_gaussian_exponent(6, 0.0), std::invalid_argument);
}

TEST(MinimalGaussianBasis, CompleteShellsAndRadialNormalization)
{
    LR::MinimalGaussianBasisSettings settings;
    const std::vector<int> elements{1, 6, 7};
    const auto basis = LR::make_minimal_gaussian_basis(elements, settings);
    ASSERT_EQ(basis[0].size(), 1);
    ASSERT_EQ(basis[1].size(), 3);
    ASSERT_EQ(basis[2].size(), 3);
    for (const auto& species : basis)
    {
        for (const auto& shell : species)
        {
            ASSERT_EQ(shell.size(), 1);
            const auto& orbital = shell[0];
            const int nr = orbital.getNr();
            std::vector<double> norm_integrand(nr);
            for (int ir = 0; ir < nr; ++ir)
            {
                const double radial_value = orbital.getPsi_r(ir);
                norm_integrand[ir] = radial_value * radial_value;
            }
            double norm = 0.0;
            const double* integrand = norm_integrand.data();
            const double* rab = orbital.getRab();
            ModuleBase::Integral::Simpson_Integral(nr, integrand, rab, norm);
            EXPECT_NEAR(norm, 1.0, 2e-12);
            EXPECT_LT(std::abs(orbital.get_psi().back()), 1e-12);
            EXPECT_EQ(orbital.getNk() % 2, 1);
        }
    }
    settings.lmax = 0;
    const auto exchange_basis = LR::make_minimal_gaussian_basis(elements, settings);
    EXPECT_EQ(exchange_basis[1].size(), 1);
}

TEST(MinimalGaussianBasis, RejectInvalidMesh)
{
    LR::MinimalGaussianBasisSettings settings;
    const std::vector<int> carbon{6};
    settings.nk = 1000;
    EXPECT_THROW(LR::make_minimal_gaussian_basis(carbon, settings), std::invalid_argument);
    settings.nk = 1001;
    settings.radial_step = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(LR::make_minimal_gaussian_basis(carbon, settings), std::invalid_argument);
}

TEST(MinimalGaussianBasis, MatchSpeciesRadialGrids)
{
    LR::MinimalGaussianBasisSettings settings;
    settings.lmax = 0;
    settings.species_radial_steps = {0.01, 0.02};
    const std::vector<int> elements{1, 6};
    const auto basis = LR::make_minimal_gaussian_basis(elements, settings);
    EXPECT_DOUBLE_EQ(basis[0][0][0].getRadial(1), 0.01);
    EXPECT_DOUBLE_EQ(basis[1][0][0].getRadial(1), 0.02);
    settings.species_radial_steps = {0.01};
    EXPECT_THROW(LR::make_minimal_gaussian_basis(elements, settings), std::invalid_argument);
}
