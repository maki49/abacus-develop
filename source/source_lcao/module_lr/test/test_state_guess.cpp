#include "source_lcao/module_lr/state_guess.h"
#include "source_base/parallel_2d.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_basis/module_nao/two_center_integrator.h"
#include "source_cell/unitcell.h"
#include "source_psi/psi.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

namespace
{
class StateGuess : public testing::Test
{
  protected:
    Parallel_2D distribution;
    void SetUp() override
    {
        // A single distributed column deliberately leaves empty ranks in the 2x2 grid.
        const int rows = 4;
        const int columns = 1;
        const int block_size = 1;
#ifdef __MPI
        distribution.init(rows, columns, block_size, MPI_COMM_WORLD);
#else
        distribution.set_serial(rows, columns);
#endif
    }

    template <typename T>
    std::vector<T> local(const std::vector<T>& global) const
    {
        std::vector<T> result(distribution.get_local_size());
        for (int row = 0; row < distribution.get_row_size() && !result.empty(); ++row)
        {
            const int global_row = distribution.local2global_row(row);
            result[row] = global[global_row];
        }
        return result;
    }

    template <typename T>
    std::vector<T> pack(const std::vector<std::vector<T>>& roots) const
    {
        std::vector<T> result;
        for (const auto& root : roots)
        {
            const auto values = local(root);
            result.insert(result.end(), values.begin(), values.end());
        }
        return result;
    }
};

TEST_F(StateGuess, ReusesAllRotatedComplexRoots)
{
    using C = std::complex<double>;
    const double half = 1.0 / std::sqrt(2.0);
    const std::vector<std::vector<C>> roots{{C(half), C(0, half), C(0), C(0)},
        {C(0, half), C(half), C(0), C(0)}, {C(0), C(0), C(0, -1), C(0)}};
    std::vector<std::vector<C>> projected;
    for (const auto& root : roots) { projected.push_back(local(root)); }
    const std::vector<std::vector<C>> cold{{C(1), C(0), C(0), C(0)},
        {C(0), C(1), C(0), C(0)}, {C(0), C(0), C(1), C(0)}};
    auto guesses = pack(cold);
    C* data = guesses.data();
    const int dimension = distribution.get_local_size();
    const int count = 3;
    const int reused = LR::select_projected_guesses(projected, data, dimension, count);
    EXPECT_EQ(reused, count);
    const auto expected = pack(roots);
    for (std::size_t row = 0; row < guesses.size(); ++row)
    {
        const double error = std::abs(guesses[row] - expected[row]);
        EXPECT_NEAR(error, 0.0, 1e-14);
    }
}

TEST_F(StateGuess, FillsLostAndDependentDirectionsFromColdSeeds)
{
    const std::vector<std::vector<double>> roots{{0, 1, 0, 0}, {0, 1, 1e-8, 0}, {0, 0, 0.1, 0}};
    std::vector<std::vector<double>> projected;
    for (const auto& root : roots) { projected.push_back(local(root)); }
    const std::vector<std::vector<double>> cold{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
    auto guesses = pack(cold);
    double* data = guesses.data();
    const int dimension = distribution.get_local_size();
    const int count = 3;
    const int reused = LR::select_projected_guesses(projected, data, dimension, count);
    EXPECT_EQ(reused, 1);
    const std::vector<std::vector<double>> expected_roots{cold[1], cold[0], cold[2]};
    const auto expected = pack(expected_roots);
    EXPECT_EQ(guesses, expected);
}

TEST_F(StateGuess, LeavesColdBlockIntactWhenFullRankCannotBeRecovered)
{
    const std::vector<double> first{1, 0, 0, 0};
    const auto direction = local(first);
    const std::vector<std::vector<double>> projected{direction, direction};
    const std::vector<std::vector<double>> cold{first, first};
    auto guesses = pack(cold);
    const auto original = guesses;
    double* data = guesses.data();
    const int dimension = distribution.get_local_size();
    const int count = 2;
    const int reused = LR::select_projected_guesses(projected, data, dimension, count);
    EXPECT_EQ(reused, 0);
    EXPECT_EQ(guesses, original);
}

TEST_F(StateGuess, RejectsNonfiniteAndInconsistentDistributedData)
{
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> root{invalid, 0, 0, 0};
    std::vector<std::vector<double>> projected{local(root)};
    const std::vector<std::vector<double>> cold{{1, 0, 0, 0}};
    auto guesses = pack(cold);
    const auto original = guesses;
    double* data = guesses.data();
    const int dimension = distribution.get_local_size();
    const int count = 1;
    int reused = LR::select_projected_guesses(projected, data, dimension, count);
    EXPECT_EQ(reused, 0);
    EXPECT_EQ(guesses, original);
    // Only the owner of row zero supplies an invalid local vector length.
    if (distribution.in_this_processor(0, 0)) { projected[0].push_back(0.0); }
    reused = LR::select_projected_guesses(projected, data, dimension, count);
    EXPECT_EQ(reused, 0);
    EXPECT_EQ(guesses, original);
}

TEST(StateGuessSnapshot, OwnsBothChannelsAndAllRootsAfterScfBuffersChange)
{
    Parallel_Orbitals pc;
    const int bands = 4;
    const int block_size = 1;
#ifdef __MPI
    pc.init(bands, bands, block_size, MPI_COMM_WORLD);
#else
    pc.set_serial(bands, bands);
#endif
    UnitCell cell;
    cell.nat = 0;
    TwoCenterIntegrator integrator;
    const int spins = 2;
    const int columns = pc.get_col_size();
    const int rows = pc.get_row_size();
    const std::vector<int> grid_sizes;
    psi::Psi<double> orbitals(spins, columns, rows, grid_sizes, true);
    for (int spin = 0; spin < spins; ++spin)
    {
        for (int col = 0; col < columns; ++col)
        {
            for (int row = 0; row < rows; ++row) { orbitals(spin, col, row) = 1 + spin + col + row; }
        }
    }
    const std::vector<double> cutoff;
    const std::vector<Parallel_2D> px;
    const std::vector<int> occupied{1, 1};
    const std::vector<int> virtuals{3, 3};
    const LR::TransitionBasisView<double> basis{cell, cutoff, integrator, orbitals,
        pc, pc, px, occupied, virtuals, false, 0};
    std::vector<double> singlet{1, 2, 3, 4, 5, 6};
    std::vector<double> triplet{7, 8, 9, 10, 11, 12};
    const std::vector<const double*> amplitudes{singlet.data(), triplet.data()};
    const int dimension = 2;
    const int roots = 3;
    const auto snapshot = LR::capture_excitation_guess(basis, amplitudes, dimension, roots);
    singlet.assign(6, 0);
    triplet.assign(6, 0);
    orbitals.zero_out();
    EXPECT_EQ(snapshot.amplitudes.size(), 2u);
    for (int channel = 0; channel < 2; ++channel)
    {
        ASSERT_EQ(snapshot.amplitudes[channel].size(), 3u);
        for (int root = 0; root < roots; ++root)
        {
            const std::vector<double> expected{double(1 + 6 * channel + 2 * root), double(2 + 6 * channel + 2 * root)};
            EXPECT_EQ(snapshot.amplitudes[channel][root], expected);
        }
        for (int col = 0; col < columns; ++col)
        {
            for (int row = 0; row < rows; ++row)
            { EXPECT_DOUBLE_EQ(snapshot.orbitals.coefficients[channel][col * rows + row], 1 + channel + col + row); }
        }
    }
}
}
