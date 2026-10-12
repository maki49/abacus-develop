#include "../lr_preconditioner.h"
#include "source_base/parallel_2d.h"

#include <gtest/gtest.h>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

TEST(LRPreconditioner, PacksUnequalSpinBlocksAcrossKPoints)
{
    const std::vector<double> eigenvalues = {-2.0, -1.0, 0.5, 2.0,
                                            -3.0, -0.5, 1.0, 4.0,
                                            -2.5, 0.2, 0.8, 3.0,
                                            -1.5, 0.1, 2.0, 5.0};
    std::vector<Parallel_2D> pairs(2);
    pairs[0].set_serial(2, 2);
    pairs[1].set_serial(3, 1);
    const std::vector<int> occupied = {2, 1};
    const std::vector<int> virtuals = {2, 3};
    const auto gaps = LR::orbital_energy_gaps(eigenvalues.data(), pairs, 2, occupied, virtuals, true);
    const std::vector<double> expected = {2.5, 4.0, 1.5, 3.0, 4.0, 7.0, 1.5, 4.5,
                                         2.7, 3.3, 5.5, 1.6, 3.5, 6.5};
    ASSERT_EQ(gaps.size(), expected.size());
    for (std::size_t j = 0; j < gaps.size(); ++j) { EXPECT_NEAR(gaps[j], expected[j], 1e-14); }
    const auto restricted = LR::orbital_energy_gaps(eigenvalues.data(), pairs, 2, occupied, virtuals, false);
    ASSERT_EQ(restricted.size(), 8);
    for (std::size_t j = 0; j < restricted.size(); ++j) { EXPECT_DOUBLE_EQ(restricted[j], gaps[j]); }
}

TEST(LRPreconditioner, GammaPacking)
{
    const std::vector<double> eigenvalues = {-2.0, -1.0, 0.5, 2.0, -2.5, 0.2, 0.8, 3.0};
    std::vector<Parallel_2D> pairs(2);
    pairs[0].set_serial(2, 2);
    pairs[1].set_serial(3, 1);
    const std::vector<int> occupied = {2, 1};
    const std::vector<int> virtuals = {2, 3};
    const auto gaps = LR::orbital_energy_gaps(eigenvalues.data(), pairs, 1, occupied, virtuals, true);
    const std::vector<double> expected = {2.5, 4.0, 1.5, 3.0, 2.7, 3.3, 5.5};
    ASSERT_EQ(gaps.size(), expected.size());
    for (std::size_t j = 0; j < gaps.size(); ++j) { EXPECT_NEAR(gaps[j], expected[j], 1e-14); }
}

TEST(LRPreconditioner, InitialVectorsSeparateDegenerateKAndSpinTransitions)
{
    const std::vector<double> eigenvalues = {-2.0, -1.0, 0.5, 2.0,
                                            -3.0, -0.5, 1.0, 4.0,
                                            -2.5, 0.2, 0.8, 3.0,
                                            -1.5, 0.1, 2.0, 5.0};
    std::vector<Parallel_2D> pairs(2);
    pairs[0].set_serial(2, 2);
    pairs[1].set_serial(3, 1);
    const std::vector<int> occupied = {2, 1};
    const std::vector<int> virtuals = {2, 3};
    const int dim = 14;
    const int roots = 6;
    std::vector<std::complex<double>> vectors(dim * roots, {2.0, 3.0});
    LR::initialize_orbital_gap_vectors(vectors.data(), roots, eigenvalues.data(),
                                       pairs, 2, occupied, virtuals, true);
    const int selected[] = {2, 6, 11, 0, 8, 3};
    for (int n = 0; n < roots; ++n)
    {
        for (int j = 0; j < dim; ++j)
        {
            const double expected = j == selected[n] ? 1.0 : 0.0;
            EXPECT_NEAR(vectors[n * dim + j].real(), expected, 1e-3);
            EXPECT_DOUBLE_EQ(vectors[n * dim + j].imag(), 0.0);
        }
    }
    const auto previous = vectors;
    LR::initialize_orbital_gap_vectors(vectors.data(), roots, eigenvalues.data(),
                                       pairs, 2, occupied, virtuals, true);
    EXPECT_EQ(vectors, previous);
    EXPECT_GT(std::abs(vectors[1]), 0.0);
    // A restricted spin channel selects only its own transitions.
    std::vector<double> restricted(16, 9.0);
    LR::initialize_orbital_gap_vectors(restricted.data(), 2, eigenvalues.data(),
                                       pairs, 2, occupied, virtuals, false);
    EXPECT_NEAR(restricted[2], 1.0, 1e-3);
    EXPECT_NEAR(restricted[14], 1.0, 1e-3);
    EXPECT_THROW(LR::initialize_orbital_gap_vectors(restricted.data(), 9, eigenvalues.data(),
                                                   pairs, 2, occupied, virtuals, false),
                 std::invalid_argument);
}

TEST(LRPreconditioner, SignedShiftedGapsAndZeroFloor)
{
    const std::vector<double> gaps = {0.3, 0.6, 0.9};
    const double shifts[] = {0.6, 0.8};
    std::vector<double> residuals = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const auto input = residuals;
    LR::apply_orbital_gap_preconditioner(residuals.data(), shifts, 3, 2, gaps, 1e-4);
    const double denominators[] = {-0.3, 1e-4, 0.3, -0.5, -0.2, 0.1};
    for (int j = 0; j < 6; ++j) { EXPECT_NEAR(residuals[j], input[j] / denominators[j], 1e-10); }
}

TEST(LRPreconditioner, CandidateBoundaryIncludesDegenerateKPartners)
{
    const std::vector<double> eigenvalues = {-1.0, 0.0, 2.0, -1.0, 0.0, 2.0};
    std::vector<Parallel_2D> pairs(1);
    pairs[0].set_serial(2, 1);
    const std::vector<int> occupied = {1};
    const std::vector<int> virtuals = {2};
    const LR::OrbitalTransitionSpace layout{eigenvalues.data(), pairs, 2, occupied, virtuals, false};
    std::vector<double> vectors(12, 0.0);
    const int selected = LR::initialize_orbital_gap_candidates(vectors.data(), 1, 3, layout);
    ASSERT_EQ(selected, 2);
    EXPECT_NEAR(vectors[0], 1.0, 1e-3);
    EXPECT_NEAR(vectors[6], 1.0, 1e-3);
    const int limited = LR::initialize_orbital_gap_candidates(vectors.data(), 1, 1, layout);
    EXPECT_EQ(limited, 1); // Capacity remains a hard memory bound.
}

TEST(LRPreconditioner, AugmentedGammaSpacePreservesSuppliedGuesses)
{
    const std::vector<double> eigenvalues = {-1.0, 0.0, 0.0, 2.0};
    std::vector<Parallel_2D> pairs(1);
    pairs[0].set_serial(3, 1);
    const std::vector<int> occupied = {1};
    const std::vector<int> virtuals = {3};
    const LR::OrbitalTransitionSpace layout{eigenvalues.data(), pairs, 1, occupied, virtuals, false};
    const std::vector<double> guesses = {0.5, 0.5, 0.0};
    const auto initial = LR::make_davidson_initial_space(guesses.data(), 3, 1, 3, layout);
    ASSERT_EQ(initial.count, 3); // The one extra nominal seed has a degenerate partner.
    for (int j = 0; j < 3; ++j) { EXPECT_DOUBLE_EQ(initial.vectors[j], guesses[j]); }
    EXPECT_NEAR(initial.vectors[3], 1.0, 1e-3);
    EXPECT_NEAR(initial.vectors[7], 1.0, 1e-3);
    EXPECT_THROW(LR::make_davidson_initial_space(guesses.data(), 2, 1, 3, layout), std::invalid_argument);
}

TEST(LRPreconditioner, ComplexResidualsAndNearResonantSigns)
{
    const std::vector<double> gaps = {1.0 - 1e-8, 1.0 + 1e-8, 1.0};
    const double shift = 1.0;
    const std::complex<double> value(0.3, -0.4);
    std::vector<std::complex<double>> residuals(3, value);
    LR::apply_orbital_gap_preconditioner(residuals.data(), &shift, 3, 1, gaps, 1e-4);
    EXPECT_NEAR(std::abs(residuals[0] + value * 1e4), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(residuals[1] - value * 1e4), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(residuals[2] - value * 1e4), 0.0, 1e-10);
}

TEST(LRPreconditioner, RejectsInvalidLayoutAndFloor)
{
    std::vector<double> residuals(1, 1.0);
    std::vector<double> gaps(1, 1.0);
    const double shift = 0.5;
    EXPECT_THROW(LR::apply_orbital_gap_preconditioner(residuals.data(), &shift, 1, 1, gaps, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(LR::apply_orbital_gap_preconditioner(residuals.data(), &shift, 2, 1, gaps, 1e-4),
                 std::invalid_argument);
    gaps[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(LR::apply_orbital_gap_preconditioner(residuals.data(), &shift, 1, 1, gaps, 1e-4),
                 std::invalid_argument);
    const std::vector<double> empty;
    LR::apply_orbital_gap_preconditioner<double>(nullptr, &shift, 0, 1, empty, 1e-4);
}

TEST(LRPreconditioner, LowEnergyBlockCompletesSpinAndKDegeneracy)
{
    const std::vector<double> eigenvalues = {-1.0, 0.0, 2.0, -1.0, 0.0, 2.0,
                                            -1.0, 0.0, 2.0, -1.0, 0.0, 2.0};
    std::vector<Parallel_2D> pairs(2);
    pairs[0].set_serial(2, 1);
    pairs[1].set_serial(2, 1);
    const std::vector<int> occupied = {1, 1};
    const std::vector<int> virtuals = {2, 2};
    const LR::OrbitalTransitionSpace layout{eigenvalues.data(), pairs, 2, occupied, virtuals, true};
    const std::vector<int> expected = {0, 2, 4, 6};
    EXPECT_EQ(LR::select_low_energy_transitions(layout, 1), expected);
    EXPECT_EQ(LR::select_low_energy_transitions(layout, 4), expected);
    const auto all = LR::select_low_energy_transitions(layout, 100);
    EXPECT_EQ(all.size(), 8);
    EXPECT_TRUE(LR::select_low_energy_transitions(layout, 0).empty());
    EXPECT_THROW(LR::select_low_energy_transitions(layout, -1), std::invalid_argument);
}
