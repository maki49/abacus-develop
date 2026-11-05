#include "../lr_davidson.h"
#include "../lr_preconditioner.h"

#include "source_base/parallel_device.h"
#include "source_base/parallel_global.h"
#include "source_base/global_variable.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <vector>
#ifdef __CUDA
#include <cuda_runtime.h>
#endif

namespace
{
int world_rank = 0;
int world_size = 1;
const int dimension = 48;
const int roots = 3;

hsolver::diag_comm_info communicator()
{
#ifdef __MPI
    return hsolver::diag_comm_info(MPI_COMM_WORLD, world_rank, world_size);
#else
    return hsolver::diag_comm_info(0, 1);
#endif
}

template<typename T> T phase(int index) { return T(1); }
template<> std::complex<double> phase<std::complex<double>>(int index)
{
    const double angle = 0.17 * index;
    return std::polar(1.0, angle);
}

/// Unitary transform of a symmetric tridiagonal matrix with known eigenpairs:
/// lambda_j = 4 - 2 cos(j*pi/(N+1)). Complex phases exercise conjugation and transfers.
template<typename T>
class CoupledHamiltonian final : public hsolver::HSOperator<T>
{
public:
    void update_k(int) override {}
    void hpsi(const T* x, T* hx, int ld, int columns) const override
    {
        const auto comm = communicator();
        const int global_elements = dimension * columns;
        std::vector<T> global(global_elements, T(0));
        const int begin = world_rank * ld;
        for (int column = 0; column < columns; ++column)
        {
            for (int row = 0; row < ld; ++row)
            { global[column * dimension + begin + row] = x[column * ld + row]; }
        }
#ifdef __MPI
        if (world_size > 1)
        {
            const int count = global.size();
            Parallel_Common::reduce_data(global.data(), count, comm.comm);
        }
#endif
        for (int column = 0; column < columns; ++column)
        {
            const T* input = global.data() + column * dimension;
            for (int row = 0; row < ld; ++row)
            {
                const int global_row = begin + row;
                T value = T(4) * input[global_row];
                if (global_row > 0)
                {
                    const int previous = global_row - 1;
                    value -= phase<T>(global_row) / phase<T>(previous) * input[previous];
                }
                if (global_row + 1 < dimension)
                {
                    const int next = global_row + 1;
                    value -= phase<T>(global_row) / phase<T>(next) * input[next];
                }
                hx[column * ld + row] = value;
            }
        }
    }
    void spsi(const T* x, T* sx, int ld, int columns) const override
    {
        const int elements = ld * columns;
        std::copy_n(x, elements, sx);
    }
};

template<typename T>
void check_eigenpairs(bool use_gpu, bool exact_initial, bool erase_inverse)
{
    ASSERT_EQ(dimension % world_size, 0);
    const int dim = dimension / world_size;
    const int initial_columns = exact_initial ? roots : roots + 2;
    const int vector_elements = dim * initial_columns;
    std::vector<T> vectors(vector_elements, T(0));
    const double pi = std::acos(-1.0);
    const double squared_scale = 2.0 / (dimension + 1);
    const double scale = std::sqrt(squared_scale);
    for (int column = 0; column < initial_columns; ++column)
    {
        for (int row = 0; row < dim; ++row)
        {
            const int global_row = world_rank * dim + row;
            if (exact_initial)
            {
                const double angle = (column + 1) * (global_row + 1) * pi / (dimension + 1);
                const double amplitude = scale * std::sin(angle);
                vectors[column * dim + row] = amplitude * phase<T>(global_row);
            }
            else if (global_row == column) { vectors[column * dim + row] = phase<T>(global_row); }
        }
    }
    const CoupledHamiltonian<T> op;
    const auto comm = communicator();
    const std::vector<double> diagonal(dim, 4.0);
    std::vector<double> eigenvalues(roots);
    hsolver::DavidsonOptions options;
    options.residual_tolerance = 1e-10;
    options.initial_vectors = initial_columns;
    options.retained_vectors = initial_columns;
    const LR::DavidsonSolveSettings settings{roots, dim, 10, 1e-12, 300, 0, 0};
    int inverse_calls = 0;
    const typename hsolver::Diago_DavSubspace<T>::Preconditioner inverse =
        [&inverse_calls, erase_inverse](T* residuals, const double* shifts, int ld, int columns)
        {
            ++inverse_calls;
            for (int column = 0; column < columns; ++column)
            {
                ASSERT_TRUE(std::isfinite(shifts[column]));
                const double difference = 4.0 - shifts[column];
                const double absolute_difference = std::abs(difference);
                const double magnitude = std::max(1e-4, absolute_difference);
                const double denominator = std::copysign(magnitude, difference);
                for (int row = 0; row < ld; ++row)
                {
                    T& value = residuals[column * ld + row];
                    if (erase_inverse) { value = T(0); }
                    else { value /= denominator; }
                }
            }
        };
    const auto result = LR::solve_davidson<T>(op, vectors.data(), eigenvalues.data(), diagonal,
                                             inverse, options, settings, comm, use_gpu);
    ASSERT_EQ(result.status, hsolver::DavidsonStatus::converged);
    if (exact_initial)
    {
        EXPECT_EQ(result.iterations, 0);
        EXPECT_EQ(inverse_calls, 0);
    }
    else
    {
        EXPECT_GT(inverse_calls, 0);
        EXPECT_GT(result.restarts, 0);
        if (erase_inverse) { EXPECT_GT(result.residual_fallbacks, 0); }
    }
    const int returned_elements = dim * roots;
    std::vector<T> images(returned_elements);
    op.hpsi(vectors.data(), images.data(), dim, roots);
    const int norm_count = 2 * roots;
    std::vector<double> squared_norms(norm_count, 0.0);
    for (int root = 0; root < roots; ++root)
    {
        const double angle = (root + 1) * pi / (dimension + 1);
        const double expected = 4.0 - 2.0 * std::cos(angle);
        EXPECT_NEAR(eigenvalues[root], expected, 1e-10);
        for (int row = 0; row < dim; ++row)
        {
            const int index = root * dim + row;
            const T residual = images[index] - eigenvalues[root] * vectors[index];
            squared_norms[root] += std::norm(residual);
            squared_norms[roots + root] += std::norm(vectors[index]);
        }
    }
#ifdef __MPI
    if (world_size > 1)
    {
        const int count = squared_norms.size();
        Parallel_Common::reduce_data(squared_norms.data(), count, comm.comm);
    }
#endif
    for (int root = 0; root < roots; ++root)
    {
        const double residual_norm = std::sqrt(squared_norms[root]);
        EXPECT_LE(residual_norm, options.residual_tolerance);
        EXPECT_NEAR(squared_norms[roots + root], 1.0, 1e-11);
    }
}
} // namespace

TEST(LRDavidson, CpuRealCoupled) { check_eigenpairs<double>(false, false, false); }
TEST(LRDavidson, CpuComplexCoupled) { check_eigenpairs<std::complex<double>>(false, false, false); }
TEST(LRDavidson, CpuExactInitial) { check_eigenpairs<double>(false, true, false); }

#ifdef __CUDA
class LRDavidsonGpu : public testing::Test
{
protected:
    void SetUp() override
    {
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess || devices == 0) { GTEST_SKIP() << "CUDA device unavailable"; }
    }
};
TEST_F(LRDavidsonGpu, RealCoupled) { check_eigenpairs<double>(true, false, false); }
TEST_F(LRDavidsonGpu, ComplexCoupled) { check_eigenpairs<std::complex<double>>(true, false, false); }
TEST_F(LRDavidsonGpu, ExactInitial) { check_eigenpairs<std::complex<double>>(true, true, false); }
TEST_F(LRDavidsonGpu, ZeroInverseFallback) { check_eigenpairs<double>(true, false, true); }
#endif


namespace
{
// Frozen 50-transition N2/LDA operator, gaps and guesses from a failing run.
// Keeping the physical matrix fixed separates solver stagnation from SCF/OMP
// changes of the degenerate orbital basis. The last three values are LAPACK roots.
template<typename T>
class FrozenCasida final : public hsolver::HSOperator<T>
{
public:
    FrozenCasida(const std::vector<double>& matrix, int dim) : matrix_(matrix), dim_(dim) {}
    void update_k(int) override {}
    void hpsi(const T* x, T* hx, int ld, int columns) const override
    {
        for (int column = 0; column < columns; ++column)
        {
            for (int row = 0; row < dim_; ++row)
            {
                T value = T(0);
                for (int j = 0; j < dim_; ++j)
                {
                    value += matrix_[j * dim_ + row] * phase<T>(row)
                             / phase<T>(j) * x[column * ld + j];
                }
                hx[column * ld + row] = value;
            }
        }
    }
    void spsi(const T* x, T* sx, int ld, int columns) const override
    {
        const int count = ld * columns;
        std::copy_n(x, count, sx);
    }
private:
    const std::vector<double>& matrix_;
    int dim_;
};

template<typename T>
void check_frozen_degenerate_roots()
{
    std::ifstream input("support/n2_davidson_stagnation.dat");
    ASSERT_TRUE(input.is_open());
    int dim = 0;
    int count = 0;
    int initial = 0;
    input >> dim >> count >> initial;
    ASSERT_TRUE(input.good());
    ASSERT_EQ(dim, 50);
    ASSERT_EQ(count, 3);
    ASSERT_EQ(initial, 5);
    std::vector<double> gaps(dim);
    std::vector<T> vectors(dim * initial);
    std::vector<double> matrix(dim * dim);
    std::vector<double> expected(count);
    for (double& value : gaps) { input >> value; }
    for (int i = 0; i < dim * initial; ++i)
    {
        double value = 0.0;
        input >> value;
        vectors[i] = value * phase<T>(i % dim);
    }
    for (double& value : matrix) { input >> value; }
    for (double& value : expected) { input >> value; }
    ASSERT_FALSE(input.fail());
    const FrozenCasida<T> op(matrix, dim);
#ifdef __MPI
    const hsolver::diag_comm_info comm(MPI_COMM_SELF, 0, 1);
#else
    const hsolver::diag_comm_info comm(0, 1);
#endif
    std::vector<double> eigenvalues(count);
    hsolver::DavidsonOptions options;
    options.initial_vectors = initial;
    options.retained_vectors = initial;
    const LR::DavidsonSolveSettings settings{count, dim, 4, 1e-6, 300, 0, 0};
    const typename hsolver::Diago_DavSubspace<T>::Preconditioner inverse =
        [&gaps](T* residuals, const double* shifts, int ld, int columns)
        {
            LR::apply_orbital_gap_preconditioner(residuals, shifts, ld, columns, gaps, 1e-4, LR::GapInverseSign::Positive);
        };
    const auto result = LR::solve_davidson<T>(op, vectors.data(), eigenvalues.data(), gaps,
                                             inverse, options, settings, comm, false);
    ASSERT_EQ(result.status, hsolver::DavidsonStatus::converged);
    std::vector<T> images(dim * count);
    op.hpsi(vectors.data(), images.data(), dim, count);
    for (int i = 0; i < count; ++i)
    {
        // Residuals alone cannot detect a missed lower root.
        EXPECT_NEAR(eigenvalues[i], expected[i], 1e-10);
        double squared_residual = 0.0;
        for (int j = 0; j < dim; ++j)
        {
            const int index = i * dim + j;
            const T residual = images[index] - eigenvalues[i] * vectors[index];
            squared_residual += std::norm(residual);
        }
        EXPECT_LE(std::sqrt(squared_residual), options.residual_tolerance);
        for (int j = 0; j < count; ++j)
        {
            std::complex<double> overlap(0.0, 0.0);
            for (int row = 0; row < dim; ++row)
            {
                overlap += std::conj(vectors[i * dim + row]) * vectors[j * dim + row];
            }
            const double target = i == j ? 1.0 : 0.0;
            EXPECT_NEAR(std::abs(overlap - target), 0.0, 1e-10);
        }
    }
}
} // namespace

TEST(LRDavidsonCpu, DegenerateCasidaLowestRootsReal) { check_frozen_degenerate_roots<double>(); }
TEST(LRDavidsonCpu, DegenerateCasidaLowestRootsComplex)
{
    check_frozen_degenerate_roots<std::complex<double>>();
}

int main(int argc, char** argv)
{
#ifdef __MPI
    int threads = 1;
    Parallel_Global::read_pal_param(argc, argv, world_size, threads, world_rank);
    // These tests deliberately distribute vectors on WORLD, without a PW pool.
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
