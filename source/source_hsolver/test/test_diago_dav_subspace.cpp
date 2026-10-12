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

int main(int argc, char** argv)
{
#ifdef __MPI
    int process_count = 1;
    int process_rank = 0;
    int threads = 1;
    Parallel_Global::read_pal_param(argc, argv, process_count, threads, process_rank);
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
