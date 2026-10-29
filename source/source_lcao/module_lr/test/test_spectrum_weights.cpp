#include "../utils/spectrum_weights.h"
#include <gtest/gtest.h>
#include <cmath>
#include <complex>

TEST(SpectrumWeights, RemovesOnlyUnpolarizedSpinDegeneracy)
{
    EXPECT_DOUBLE_EQ(LR_Util::spatial_k_weight(0.002, 1), 0.001);
    EXPECT_DOUBLE_EQ(LR_Util::spatial_k_weight(0.001, 2), 0.001);
    EXPECT_DOUBLE_EQ(LR_Util::spatial_k_weight(0.001, 4), 0.001);
}

TEST(SpectrumWeights, PreservesNonuniformSpatialWeights)
{
    const double ks_weights[] = {0.2, 0.6, 1.2};
    const double expected[] = {0.1, 0.3, 0.6};
    double sum = 0.0;
    for (int k = 0; k < 3; ++k)
    {
        const double w = LR_Util::spatial_k_weight(ks_weights[k], 1);
        EXPECT_DOUBLE_EQ(w, expected[k]);
        sum += w;
    }
    EXPECT_DOUBLE_EQ(sum, 1.0);
}

TEST(SpectrumWeights, SingletIntensityMatchesExplicitSpinChannels)
{
    // A unit-normalized singlet has X_up=X_down=X/sqrt(2).
    // Compare that explicit contraction with the spatial-orbital sqrt(2) path.
    const int nk = 2;
    const std::complex<double> dipoles[] = {{1.0, 0.5}, {-0.2, 0.7}};
    const double amplitudes[] = {0.6, 0.8};
    std::complex<double> singlet = 0.0;
    std::complex<double> explicit_spins = 0.0;
    for (int k = 0; k < nk; ++k)
    {
        const double unpolarized_w = LR_Util::spatial_k_weight(1.0, 1);
        const double polarized_w = LR_Util::spatial_k_weight(0.5, 2);
        singlet += std::sqrt(2.0) * dipoles[k] * amplitudes[k] * unpolarized_w * double(nk);
        for (int spin = 0; spin < 2; ++spin)
        {
            explicit_spins += dipoles[k] * amplitudes[k] / std::sqrt(2.0) * polarized_w * double(nk);
        }
    }
    EXPECT_NEAR(std::abs(singlet - explicit_spins), 0.0, 1e-14);
    EXPECT_NEAR(std::norm(singlet), 1.8664, 1e-14);
}
