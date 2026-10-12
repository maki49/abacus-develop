#include "../lr_block_preconditioner.h"
#include "source_base/parallel_device.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_global.h"
#include "source_hsolver/diag_comm_info.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <complex>
#include <stdexcept>

namespace
{
int world_rank = 0;
int world_size = 1;

hsolver::diag_comm_info serial_comm()
{
#ifdef __MPI
    return {MPI_COMM_SELF, 0, 1};
#else
    return {0, 1};
#endif
}

template<typename T>
void check_shifted_inverse(const T coupling)
{
    // P is deliberately not a contiguous slice. Nonzero P-Q entries must
    // affect neither the P solve nor the diagonal Q solve of this preconditioner.
    const int dim = 4;
    const std::vector<int> selected = {2, 0};
    const std::vector<double> gaps = {2.0, 5.0, 3.0, 6.0};
    int probed_vectors = 0;
    const std::function<void(const T*, T*, int)> action = [&](const T* x, T* ax, int columns)
    {
        probed_vectors += columns;
        for (int n = 0; n < columns; ++n)
        {
            const T* v = x + n * dim;
            T* av = ax + n * dim;
            av[0] = 2.0 * v[0] + coupling * v[2] + 0.7 * v[1];
            av[1] = 5.0 * v[1] + 0.7 * v[0];
            av[2] = 3.0 * v[2] + std::real(coupling) * v[0] * 2.0 - coupling * v[0];
            av[3] = 6.0 * v[3];
        }
    };
    const auto comm = serial_comm();
    const auto block = LR::build_low_energy_preconditioner(selected, dim, action, comm);
    EXPECT_EQ(probed_vectors, 2);
    EXPECT_NEAR(block.hermiticity_error, 0.0, 1e-14);
    const double shifts[] = {1.0, 2.5}; // The second shifted block is indefinite.
    const std::vector<T> original = {T(1), T(2), T(3), T(4), T(-1), T(3), T(2), T(-2)};
    auto result = original;
    LR::apply_low_energy_preconditioner(result.data(), shifts, dim, 2, gaps, 1e-4, block, comm);
    for (int n = 0; n < 2; ++n)
    {
        const T* z = result.data() + n * dim;
        const T* r = original.data() + n * dim;
        const T conjugate_coupling = 2.0 * std::real(coupling) - coupling;
        EXPECT_NEAR(std::abs((2.0 - shifts[n]) * z[0] + coupling * z[2] - r[0]), 0.0, 1e-12);
        EXPECT_NEAR(std::abs((3.0 - shifts[n]) * z[2] + conjugate_coupling * z[0] - r[2]), 0.0, 1e-12);
        EXPECT_NEAR(std::abs((5.0 - shifts[n]) * z[1] - r[1]), 0.0, 1e-12);
        EXPECT_NEAR(std::abs((6.0 - shifts[n]) * z[3] - r[3]), 0.0, 1e-12);
    }
}
} // namespace

TEST(LRBlockPreconditioner, RealShiftedInverse) { check_shifted_inverse(0.4); }
TEST(LRBlockPreconditioner, ComplexShiftedInverse)
{
    const std::complex<double> coupling(0.4, 0.3);
    check_shifted_inverse(coupling);
}

TEST(LRBlockPreconditioner, SpectralPoleRegularization)
{
    const std::vector<int> selected = {0, 1};
    const std::function<void(const double*, double*, int)> action = [](const double* x, double* ax, int columns)
    {
        for (int n = 0; n < columns; ++n)
        {
            ax[2 * n] = 2.0 * x[2 * n] + x[2 * n + 1];
            ax[2 * n + 1] = x[2 * n] + 2.0 * x[2 * n + 1];
        }
    };
    const auto comm = serial_comm();
    const auto block = LR::build_low_energy_preconditioner(selected, 2, action, comm);
    const std::vector<double> gaps(2, 2.0);
    const double shifts[] = {3.0 + 1e-8, 3.0 - 1e-8};
    std::vector<double> residuals(4, 1.0); // Exact eigenmode of A_PP with lambda = 3.
    LR::apply_low_energy_preconditioner(residuals.data(), shifts, 2, 2, gaps, 1e-4, block, comm);
    EXPECT_NEAR(residuals[0], -1e4, 1e-8);
    EXPECT_NEAR(residuals[1], -1e4, 1e-8);
    EXPECT_NEAR(residuals[2], 1e4, 1e-8);
    EXPECT_NEAR(residuals[3], 1e4, 1e-8);
}

TEST(LRBlockPreconditioner, EmptyBlockAndInvalidCoordinates)
{
    const auto comm = serial_comm();
    const std::function<void(const double*, double*, int)> unused = [](const double*, double*, int) { FAIL(); };
    const std::vector<int> none;
    const auto empty = LR::build_low_energy_preconditioner(none, 2, unused, comm);
    const std::vector<double> gaps = {1.0, 3.0};
    const double shift = 2.0;
    double residuals[] = {2.0, 4.0};
    LR::apply_low_energy_preconditioner(residuals, &shift, 2, 1, gaps, 1e-4, empty, comm);
    EXPECT_DOUBLE_EQ(residuals[0], -2.0);
    EXPECT_DOUBLE_EQ(residuals[1], 4.0);
    const std::vector<int> duplicate = {0, 0};
    const std::vector<int> missing = {-1};
    const std::vector<int> outside = {2};
    EXPECT_THROW(LR::build_low_energy_preconditioner(duplicate, 2, unused, comm), std::invalid_argument);
    EXPECT_THROW(LR::build_low_energy_preconditioner(missing, 2, unused, comm), std::invalid_argument);
    EXPECT_THROW(LR::build_low_energy_preconditioner(outside, 2, unused, comm), std::invalid_argument);
}

TEST(LRBlockPreconditioner, DistributedBlockIncludingEmptyRanks)
{
#ifdef __MPI
    const hsolver::diag_comm_info comm{MPI_COMM_WORLD, world_rank, world_size};
#else
    const hsolver::diag_comm_info comm{0, 1};
#endif
    // Three global rows on four ranks exercises a rank with no local rows.
    const int global_dim = 3;
    std::vector<int> global_rows;
    for (int row = 0; row < global_dim; ++row)
    {
        if (row % world_size == world_rank) { global_rows.push_back(row); }
    }
    const int dim = global_rows.size();
    std::vector<int> selected(2, -1);
    for (int j = 0; j < dim; ++j)
    {
        if (global_rows[j] < 2) { selected[global_rows[j]] = j; }
    }
    const std::function<void(const double*, double*, int)> action = [&](const double* x, double* ax, int columns)
    {
        std::vector<double> global(columns * global_dim, 0.0);
        for (int n = 0; n < columns; ++n)
        {
            for (int j = 0; j < dim; ++j) { global[n * global_dim + global_rows[j]] = x[n * dim + j]; }
        }
#ifdef __MPI
        const int size = global.size();
        Parallel_Common::reduce_data(global.data(), size, comm.comm);
#endif
        for (int n = 0; n < columns; ++n)
        {
            const double* v = global.data() + n * global_dim;
            const double image[] = {2.0 * v[0] + v[1], v[0] + 3.0 * v[1], 5.0 * v[2]};
            for (int j = 0; j < dim; ++j) { ax[n * dim + j] = image[global_rows[j]]; }
        }
    };
    const auto block = LR::build_low_energy_preconditioner(selected, dim, action, comm);
    const double shift = 1.0;
    const double global_gaps[] = {2.0, 3.0, 5.0};
    const double global_residual[] = {3.0, 4.0, 8.0};
    const double expected[] = {2.0, 1.0, 2.0};
    std::vector<double> gaps;
    std::vector<double> residual;
    for (int row : global_rows)
    {
        gaps.push_back(global_gaps[row]);
        residual.push_back(global_residual[row]);
    }
    LR::apply_low_energy_preconditioner(residual.data(), &shift, dim, 1, gaps, 1e-4, block, comm);
    for (int j = 0; j < dim; ++j) { EXPECT_NEAR(residual[j], expected[global_rows[j]], 1e-12); }
}

int main(int argc, char** argv)
{
#ifdef __MPI
    int threads = 1;
    Parallel_Global::read_pal_param(argc, argv, world_size, threads, world_rank);
    POOL_WORLD = MPI_COMM_NULL;
    KP_WORLD = MPI_COMM_NULL;
    INT_BGROUP = MPI_COMM_NULL;
    BP_WORLD = MPI_COMM_NULL;
    GRID_WORLD = MPI_COMM_NULL;
    DIAG_WORLD = MPI_COMM_NULL;
#endif
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
#ifdef __MPI
    Parallel_Global::finalize_mpi();
#endif
    return result;
}
