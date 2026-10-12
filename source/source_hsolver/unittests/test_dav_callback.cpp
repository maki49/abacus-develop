#include "../diago_dav_subspace.h"
#include "../diag_comm_info.h"
#include "source_base/parallel_global.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace
{
constexpr int dimension = 24;
constexpr int roots = 2;
int world_rank = 0;
int world_size = 1;

hsolver::diag_comm_info serial_comm()
{
#ifdef __MPI
    return hsolver::diag_comm_info(MPI_COMM_SELF, 0, 1);
#else
    return hsolver::diag_comm_info(0, 1);
#endif
}

template<typename T>
std::vector<T> initial_vectors()
{
    std::vector<T> vectors(dimension * roots);
    // Dense, orthonormal guesses avoid dependent residuals from adjacent unit
    // vectors in a nearest-neighbor Hamiltonian.
    for (int n = 0; n < roots; ++n)
    {
        for (int j = 0; j < dimension; ++j)
        {
            vectors[n * dimension + j] = std::sin((0.37 + 0.19 * n) * (j + 1));
        }
        if (n == 1)
        {
            T overlap = 0.0;
            for (int j = 0; j < dimension; ++j) { overlap += vectors[j] * vectors[dimension + j]; }
            for (int j = 0; j < dimension; ++j) { vectors[dimension + j] -= overlap * vectors[j]; }
        }
        double norm_squared = 0.0;
        for (int j = 0; j < dimension; ++j) { norm_squared += std::norm(vectors[n * dimension + j]); }
        const double norm = std::sqrt(norm_squared);
        for (int j = 0; j < dimension; ++j) { vectors[n * dimension + j] /= norm; }
    }
    return vectors;
}

// Hermitian tridiagonal matrix with known spectrum 2 - 0.3 cos((n+1) pi/(N+1)).
template<typename T>
class ChainHamiltonian : public hsolver::HSOperator<T>
{
  public:
    explicit ChainHamiltonian(T hopping) : hopping_(hopping) {}
    void update_k(int) override {}
    void hpsi(const T* x, T* hx, int ld, int nvec) const override
    {
        for (int n = 0; n < nvec; ++n)
        {
            for (int j = 0; j < dimension; ++j)
            {
                const int index = n * ld + j;
                hx[index] = 2.0 * x[index];
                if (j > 0) { hx[index] += hopping_ * x[index - 1]; }
                if (j + 1 < dimension)
                {
                    const T conjugate = T(0.0225) / hopping_;
                    hx[index] += conjugate * x[index + 1];
                }
            }
        }
    }
    void spsi(const T* x, T* sx, int ld, int nvec) const override
    {
        const int size = ld * nvec;
        std::copy(x, x + size, sx);
    }
  private:
    T hopping_;
};

template<typename T>
void check_solution(const ChainHamiltonian<T>& op, const std::vector<T>& vectors,
                    const std::vector<double>& eigenvalues)
{
    std::vector<T> image(vectors.size());
    op.hpsi(vectors.data(), image.data(), dimension, roots);
    const double pi = std::acos(-1.0);
    for (int n = 0; n < roots; ++n)
    {
        const double expected = 2.0 - 0.3 * std::cos((n + 1) * pi / (dimension + 1));
        EXPECT_NEAR(eigenvalues[n], expected, 1e-10);
        double residual_squared = 0.0;
        double norm_squared = 0.0;
        for (int j = 0; j < dimension; ++j)
        {
            const int index = n * dimension + j;
            const T residual = image[index] - eigenvalues[n] * vectors[index];
            residual_squared += std::norm(residual);
            norm_squared += std::norm(vectors[index]);
        }
        EXPECT_LT(std::sqrt(residual_squared), 2e-6);
        EXPECT_NEAR(norm_squared, 1.0, 1e-10);
    }
}

template<typename T>
void compare_legacy_callback(T hopping)
{
    const ChainHamiltonian<T> op(hopping);
    const auto comm = serial_comm();
    std::vector<double> diagonal(dimension);
    for (int j = 0; j < dimension; ++j) { diagonal[j] = 1.0 + 0.2 * j; }
    const std::vector<double> tolerances(roots, 1e-13);
    std::vector<T> legacy = initial_vectors<T>();
    auto custom = legacy;
    std::vector<double> legacy_eigenvalues(roots);
    std::vector<double> custom_eigenvalues(roots);
    hsolver::Diago_DavSubspace<T> old_solver(diagonal, roots, dimension, 6, 1e-13, 200, comm, 0, 0);
    hsolver::Diago_DavSubspace<T> new_solver(diagonal, roots, dimension, 6, 1e-13, 200, comm, 0, 0);
    int calls = 0;
    const typename hsolver::Diago_DavSubspace<T>::Preconditioner smooth =
        [&](T* residuals, const double* shifts, int dim, int nvec)
    {
        ++calls;
        EXPECT_EQ(dim, dimension);
        EXPECT_GE(nvec, 1);
        EXPECT_LE(nvec, roots);
        for (int n = 0; n < nvec; ++n)
        {
            for (int j = 0; j < dim; ++j)
            {
                const double x = std::abs(diagonal[j] - shifts[n]);
                const double denominator = 0.5 * (1.0 + x + std::sqrt(1.0 + (x - 1.0) * (x - 1.0)));
                residuals[n * dim + j] /= denominator;
            }
        }
    };
    const int legacy_steps = old_solver.diag(op, legacy.data(), dimension, legacy_eigenvalues.data(),
                                             tolerances, false);
    const int custom_steps = new_solver.diag(op, custom.data(), dimension, custom_eigenvalues.data(),
                                             tolerances, false, smooth);
    EXPECT_EQ(calls, custom_steps);
    EXPECT_EQ(legacy_steps, custom_steps);
    EXPECT_EQ(legacy_eigenvalues, custom_eigenvalues);
    EXPECT_EQ(legacy, custom);
    check_solution(op, custom, custom_eigenvalues);
}
}

TEST(DavSubspace, LegacyAndCallbackReal)
{
    compare_legacy_callback(0.15);
}

TEST(DavSubspace, LegacyAndCallbackComplex)
{
    const std::complex<double> hopping(0.12, 0.09);
    compare_legacy_callback(hopping);
}

TEST(DavSubspace, NonDiagonalCallbackWithCompactedShifts)
{
    const ChainHamiltonian<double> op(0.15);
    const auto comm = serial_comm();
    const std::vector<double> unused_diagonal(dimension, 999.0);
    const std::vector<double> tolerances = {1.0, 1e-13};
    std::vector<double> vectors = initial_vectors<double>();
    std::vector<double> eigenvalues(roots);
    bool saw_single_root = false;
    double last_single_shift = 0.0;
    const hsolver::Diago_DavSubspace<double>::Preconditioner block_inverse =
        [&](double* residuals, const double* shifts, int dim, int nvec)
    {
        if (nvec == 1)
        {
            saw_single_root = true;
            last_single_shift = shifts[0];
        }
        const int size = dim * nvec;
        const std::vector<double> input(residuals, residuals + size);
        for (int n = 0; n < nvec; ++n)
        {
            for (int j = 0; j < dim; ++j)
            {
                const int index = n * dim + j;
                residuals[index] = input[index];
                if (j > 0) { residuals[index] -= 0.1 * input[index - 1]; }
                if (j + 1 < dim) { residuals[index] -= 0.1 * input[index + 1]; }
            }
        }
    };
    hsolver::Diago_DavSubspace<double> solver(unused_diagonal, roots, dimension, 6, 1e-13, 200, comm, 0, 0);
    solver.diag(op, vectors.data(), dimension, eigenvalues.data(), tolerances, false, block_inverse);
    EXPECT_TRUE(saw_single_root);
    EXPECT_NEAR(last_single_shift, eigenvalues[1], 1e-10);
    // Only root 1 was given a tight stopping tolerance.
    std::vector<double> image(vectors.size());
    op.hpsi(vectors.data(), image.data(), dimension, roots);
    double residual_squared = 0.0;
    for (int j = 0; j < dimension; ++j)
    {
        const int index = dimension + j;
        const double residual = image[index] - eigenvalues[1] * vectors[index];
        residual_squared += residual * residual;
    }
    EXPECT_LT(std::sqrt(residual_squared), 2e-6);
    const hsolver::Diago_DavSubspace<double>::Preconditioner empty;
    EXPECT_THROW(solver.diag(op, vectors.data(), dimension, eigenvalues.data(), tolerances, false, empty),
                 std::invalid_argument);
}

namespace
{
// H_jj = (j+1) S_jj has a known, ordered spectrum even when S != I.
// An offset lets the same operator exercise a genuinely distributed vector.
template<typename T>
class DiagonalPencil : public hsolver::HSOperator<T>
{
  public:
    DiagonalPencil(int local_dim, int offset, bool generalized)
        : local_dim_(local_dim), offset_(offset), generalized_(generalized)
    {
        spectrum.resize(local_dim);
        for (int j = 0; j < local_dim; ++j) { spectrum[j] = offset + j + 1; }
    }
    void update_k(int) override {}
    void hpsi(const T* x, T* hx, int ld, int count) const override
    {
        applied_vectors += count;
        for (int n = 0; n < count; ++n)
        {
            for (int j = 0; j < local_dim_; ++j)
            {
                hx[n * ld + j] = spectrum[j] * overlap(j) * x[n * ld + j];
            }
        }
    }
    void spsi(const T* x, T* sx, int ld, int count) const override
    {
        for (int n = 0; n < count; ++n)
        {
            for (int j = 0; j < local_dim_; ++j) { sx[n * ld + j] = overlap(j) * x[n * ld + j]; }
        }
    }
    double overlap(int j) const { return generalized_ ? 1.0 + 0.1 * (offset_ + j) : 1.0; }
    mutable int applied_vectors = 0;
    std::vector<double> spectrum;
  private:
    int local_dim_;
    int offset_;
    bool generalized_;
};

template<typename T>
void exact_initial_vectors()
{
    const auto comm = serial_comm();
    const DiagonalPencil<T> op(dimension, 0, false);
    const std::vector<double> diagonal(dimension, 1.0);
    const std::vector<double> energy_tolerances(roots, 1e-12);
    std::vector<T> vectors(dimension * roots, T(0));
    vectors[0] = 1;
    vectors[dimension + 1] = 1;
    std::vector<double> eigenvalues(roots);
    int inverse_calls = 0;
    const typename hsolver::Diago_DavSubspace<T>::Preconditioner inverse =
        [&](T*, const double*, int, int) { ++inverse_calls; };
    hsolver::DavidsonOptions options;
    options.residual_tolerance = 1e-12;
    hsolver::Diago_DavSubspace<T> solver(diagonal, roots, dimension, 4, 1e-12, 20, comm, 0, 0);
    const auto result = solver.diag(op, vectors.data(), dimension, eigenvalues.data(), energy_tolerances, inverse, options);
    EXPECT_EQ(result.status, hsolver::DavidsonStatus::converged);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.converged_roots, roots);
    EXPECT_EQ(inverse_calls, 0);
    EXPECT_EQ(op.applied_vectors, roots);
    EXPECT_DOUBLE_EQ(eigenvalues[0], 1.0);
    EXPECT_DOUBLE_EQ(eigenvalues[1], 2.0);
}

template<typename T>
void partially_converged_pencil(const hsolver::diag_comm_info& comm, bool zero_inverse)
{
    const int local_dim = dimension / comm.nproc;
    const int offset = local_dim * comm.rank;
    const DiagonalPencil<T> op(local_dim, offset, true);
    const std::vector<double> diagonal(local_dim, 1.0);
    const std::vector<double> energy_tolerances(roots, 1e-12);
    std::vector<T> vectors(local_dim * roots, T(0));
    for (int j = 0; j < local_dim; ++j)
    {
        const int global = offset + j;
        if (global == 0) { vectors[j] = 1; }
        if (global == 1) { vectors[local_dim + j] = 1; }
        if (global == 2) { vectors[local_dim + j] = 0.2; }
    }
    std::vector<double> eigenvalues(roots);
    int calls = 0;
    const typename hsolver::Diago_DavSubspace<T>::Preconditioner inverse =
        [&](T* residuals, const double* shifts, int dim, int count)
    {
        ++calls;
        EXPECT_EQ(count, 1); // Root 0 was exact before the first expansion.
        EXPECT_GT(shifts[0], 2.0);
        EXPECT_LT(shifts[0], 3.0);
        if (zero_inverse) { std::fill(residuals, residuals + dim * count, T(0)); }
    };
    hsolver::DavidsonOptions options;
    options.residual_tolerance = 1e-10;
    hsolver::Diago_DavSubspace<T> solver(diagonal, roots, local_dim, 4, 1e-12, 20, comm, 0, 0);
    const auto result = solver.diag(op, vectors.data(), local_dim, eigenvalues.data(), energy_tolerances, inverse, options);
    ASSERT_EQ(result.status, hsolver::DavidsonStatus::converged);
    EXPECT_EQ(calls, 1);
    EXPECT_NEAR(eigenvalues[0], 1.0, 1e-12);
    EXPECT_NEAR(eigenvalues[1], 2.0, 1e-12);
    EXPECT_LE(result.residual_norms[0], options.residual_tolerance);
    EXPECT_LE(result.residual_norms[1], options.residual_tolerance);
    const int expected_fallbacks = zero_inverse ? 1 : 0;
    EXPECT_EQ(result.residual_fallbacks, expected_fallbacks);
}
}

TEST(DavSubspace, ExactInitialReal) { exact_initial_vectors<double>(); }
TEST(DavSubspace, ExactInitialComplex) { exact_initial_vectors<std::complex<double>>(); }
TEST(DavSubspace, ResidualCompactionGeneralized)
{
    const auto comm = serial_comm();
    partially_converged_pencil<std::complex<double>>(comm, false);
}
TEST(DavSubspace, ZeroInverseFallsBackToRawResidual)
{
    const auto comm = serial_comm();
    partially_converged_pencil<double>(comm, true);
}
TEST(DavSubspace, DistributedResidualCompaction)
{
#ifdef __MPI
    const hsolver::diag_comm_info comm(MPI_COMM_WORLD, world_rank, world_size);
#else
    const auto comm = serial_comm();
#endif
    const int remainder = dimension % comm.nproc;
    ASSERT_EQ(remainder, 0);
    partially_converged_pencil<std::complex<double>>(comm, true);
}

TEST(DavSubspace, DependentCorrectionsAndRestart)
{
    const auto comm = serial_comm();
    const ChainHamiltonian<double> op(0.15);
    const std::vector<double> diagonal(dimension, 2.0);
    const std::vector<double> energy_tolerances(roots, 1e-12);
    std::vector<double> vectors(dimension * roots, 0.0);
    vectors[0] = 1;
    vectors[dimension + 1] = 1;
    std::vector<double> eigenvalues(roots);
    const hsolver::Diago_DavSubspace<double>::Preconditioner identity = [](double*, const double*, int, int) {};
    hsolver::DavidsonOptions options;
    options.residual_tolerance = 1e-9;
    hsolver::Diago_DavSubspace<double> solver(diagonal, roots, dimension, 6, 1e-12, 300, comm, 0, 0);
    const auto result = solver.diag(op, vectors.data(), dimension, eigenvalues.data(), energy_tolerances, identity, options);
    ASSERT_EQ(result.status, hsolver::DavidsonStatus::converged);
    EXPECT_GT(result.rejected_directions, 0);
    check_solution(op, vectors, eigenvalues);
    for (double norm : result.residual_norms) { EXPECT_LE(norm, options.residual_tolerance); }
}

TEST(DavSubspace, IterationLimitIsNotConvergence)
{
    const auto comm = serial_comm();
    const ChainHamiltonian<double> op(0.15);
    const std::vector<double> diagonal(dimension, 2.0);
    const std::vector<double> energy_tolerances(roots, 1.0); // A loose energy test cannot override a large residual.
    std::vector<double> vectors = initial_vectors<double>();
    std::vector<double> eigenvalues(roots);
    const hsolver::Diago_DavSubspace<double>::Preconditioner identity = [](double*, const double*, int, int) {};
    hsolver::DavidsonOptions options;
    options.residual_tolerance = 1e-12;
    hsolver::Diago_DavSubspace<double> solver(diagonal, roots, dimension, 4, 1.0, 1, comm, 0, 0);
    const auto result = solver.diag(op, vectors.data(), dimension, eigenvalues.data(), energy_tolerances, identity, options);
    EXPECT_EQ(result.status, hsolver::DavidsonStatus::iteration_limit);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_LT(result.converged_roots, roots);
}

TEST(DavSubspace, DuplicateInitialVectorsReturnFailure)
{
    const auto comm = serial_comm();
    const DiagonalPencil<double> op(dimension, 0, false);
    const std::vector<double> diagonal(dimension, 1.0);
    const std::vector<double> energy_tolerances(roots, 1e-12);
    std::vector<double> vectors(dimension * roots, 0.0);
    vectors[0] = 1;
    vectors[dimension] = 1;
    std::vector<double> eigenvalues(roots);
    const hsolver::Diago_DavSubspace<double>::Preconditioner identity = [](double*, const double*, int, int) {};
    const hsolver::DavidsonOptions options;
    hsolver::Diago_DavSubspace<double> solver(diagonal, roots, dimension, 4, 1e-12, 20, comm, 0, 0);
    const auto result = solver.diag(op, vectors.data(), dimension, eigenvalues.data(), energy_tolerances, identity, options);
    EXPECT_EQ(result.status, hsolver::DavidsonStatus::dependent_initial_vectors);
    EXPECT_EQ(op.applied_vectors, 0);
}

TEST(DavSubspace, ExtraInitialCandidatesRecoverSkippedRoot)
{
    const auto comm = serial_comm();
    const DiagonalPencil<double> op(dimension, 0, false);
    const std::vector<double> diagonal(dimension, 1.0);
    const std::vector<double> energy_tolerances(roots, 1e-12);
    std::vector<double> vectors(dimension * 3, 0.0);
    vectors[0] = 1;
    vectors[dimension + 2] = 1;
    vectors[2 * dimension + 1] = 1; // Independent candidate for the missing second root.
    std::vector<double> eigenvalues(roots);
    const hsolver::Diago_DavSubspace<double>::Preconditioner identity = [](double*, const double*, int, int) {};
    hsolver::DavidsonOptions options;
    hsolver::Diago_DavSubspace<double> solver(diagonal, roots, dimension, 4, 1e-12, 20, comm, 0, 0);
    const auto restricted = solver.diag(op, vectors.data(), dimension, eigenvalues.data(), energy_tolerances, identity, options);
    ASSERT_EQ(restricted.status, hsolver::DavidsonStatus::converged);
    EXPECT_DOUBLE_EQ(eigenvalues[1], 3.0);
    EXPECT_DOUBLE_EQ(restricted.residual_norms[1], 0.0); // Residual alone cannot certify the root index.
    options.initial_vectors = 3;
    options.retained_vectors = 3;
    const auto augmented = solver.diag(op, vectors.data(), dimension, eigenvalues.data(), energy_tolerances, identity, options);
    ASSERT_EQ(augmented.status, hsolver::DavidsonStatus::converged);
    EXPECT_DOUBLE_EQ(eigenvalues[0], 1.0);
    EXPECT_DOUBLE_EQ(eigenvalues[1], 2.0);
}

TEST(DavSubspace, CandidateSpacePreservesDegenerateMultiplicity)
{
    const auto comm = serial_comm();
    DiagonalPencil<double> op(dimension, 0, false);
    op.spectrum[2] = 2.0;
    const int requested = 3;
    const std::vector<double> diagonal(dimension, 1.0);
    const std::vector<double> energy_tolerances(requested, 1e-12);
    std::vector<double> vectors(dimension * 4, 0.0);
    vectors[0] = 1;
    vectors[dimension + 3] = 1;
    vectors[2 * dimension + 1] = 1;
    vectors[3 * dimension + 2] = 1;
    std::vector<double> eigenvalues(requested);
    const hsolver::Diago_DavSubspace<double>::Preconditioner identity = [](double*, const double*, int, int) {};
    hsolver::DavidsonOptions options;
    options.initial_vectors = 4;
    options.retained_vectors = 4;
    hsolver::Diago_DavSubspace<double> solver(diagonal, requested, dimension, 4, 1e-12, 20, comm, 0, 0);
    const auto result = solver.diag(op, vectors.data(), dimension, eigenvalues.data(), energy_tolerances, identity, options);
    ASSERT_EQ(result.status, hsolver::DavidsonStatus::converged);
    EXPECT_DOUBLE_EQ(eigenvalues[0], 1.0);
    EXPECT_DOUBLE_EQ(eigenvalues[1], 2.0);
    EXPECT_DOUBLE_EQ(eigenvalues[2], 2.0);
    double overlap = 0.0;
    for (int j = 0; j < dimension; ++j) { overlap += vectors[dimension + j] * vectors[2 * dimension + j]; }
    EXPECT_NEAR(overlap, 0.0, 1e-14);
}

TEST(DavSubspace, ExtraRitzVectorsSurviveRestart)
{
    const auto comm = serial_comm();
    const std::complex<double> hopping(0.12, 0.09);
    const ChainHamiltonian<std::complex<double>> op(hopping);
    const std::vector<double> diagonal(dimension, 2.0);
    const std::vector<double> energy_tolerances(roots, 1e-12);
    auto vectors = initial_vectors<std::complex<double>>();
    std::vector<double> eigenvalues(roots);
    const hsolver::Diago_DavSubspace<std::complex<double>>::Preconditioner identity =
        [](std::complex<double>*, const double*, int, int) {};
    hsolver::DavidsonOptions options;
    options.retained_vectors = 4;
    options.residual_tolerance = 1e-9;
    hsolver::Diago_DavSubspace<std::complex<double>> solver(diagonal, roots, dimension, 6, 1e-12, 300, comm, 0, 0);
    const auto result = solver.diag(op, vectors.data(), dimension, eigenvalues.data(), energy_tolerances, identity, options);
    ASSERT_EQ(result.status, hsolver::DavidsonStatus::converged);
    EXPECT_GT(result.restarts, 0);
    check_solution(op, vectors, eigenvalues);
    for (double norm : result.residual_norms) { EXPECT_LE(norm, options.residual_tolerance); }
}

int main(int argc, char** argv)
{
#ifdef __MPI
    int process_count = 1;
    int process_rank = 0;
    int threads = 1;
    Parallel_Global::read_pal_param(argc, argv, process_count, threads, process_rank);
    world_rank = process_rank;
    world_size = process_count;
    int band_count = 0;
    int band_rank = 0;
    int band_group = 0;
    int pool_count = 0;
    int pool_rank = 0;
    int pool_group = 0;
    // The existing normalization kernel reduces over POOL_WORLD. Give each
    // independent dense test its own one-rank pool using the runtime wrappers.
    Parallel_Global::init_pools(process_count, process_rank, 1, process_count,
                                band_count, band_rank, band_group, pool_count, pool_rank, pool_group);
    Parallel_Global::split_diag_world(1, process_count, process_rank, band_rank, band_count, band_group);
    Parallel_Global::split_grid_world(1, process_count, process_rank, pool_rank, pool_count);
#endif
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
#ifdef __MPI
    Parallel_Global::finalize_mpi();
#endif
    return result;
}
