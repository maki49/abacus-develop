#include "../linear_pcg.h"
#include "../diag_comm_info.h"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef __CUDA
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/module_container/ATen/core/tensor.h"
#endif

namespace
{
using Cpu = base_device::DEVICE_CPU;

hsolver::diag_comm_info serial_comm()
{
#ifdef __MPI
    return hsolver::diag_comm_info(MPI_COMM_SELF, 0, 1);
#else
    return hsolver::diag_comm_info(0, 1);
#endif
}

class Diagonal : public hsolver::LinearOperator<double, Cpu>
{
public:
    explicit Diagonal(const std::vector<double>& values) : values_(values) {}
    void apply(const double* x, double* y, int ld, int columns) const override
    {
        for (int column = 0; column < columns; ++column)
        {
            for (std::size_t row = 0; row < values_.size(); ++row)
            {
                y[column * ld + row] = values_[row] * x[column * ld + row];
            }
        }
    }
private:
    const std::vector<double> values_;
};

class FailingInverse : public hsolver::LinearOperator<double, Cpu>
{
public:
    void apply(const double*, double*, int, int) const override
    {
        throw hsolver::LinearPreconditionerError();
    }
};
}

TEST(LinearPCG, IndependentColumnsAndTrueResidual)
{
    const Diagonal op({1.0, 3.0, 7.0, 11.0});
    const Diagonal inverse({1.0, 1.0, 1.0, 1.0});
    const auto comm = serial_comm();
    const int ld = 6;
    const int dim = 4;
    const int columns = 3;
    std::vector<double> rhs(ld * columns, 0.0);
    std::vector<double> x(ld * columns, 0.0);
    rhs[0] = 2.0;  // first RHS converges in one iteration
    for (int row = 0; row < dim; ++row) { rhs[ld + row] = 1.0 + row; }
    const auto result = hsolver::linear_pcg<Cpu>(op, inverse, x.data(), rhs.data(),
                                                ld, dim, columns, 1e-12, 20, comm);
    EXPECT_EQ(result.status, hsolver::LinearSolveStatus::converged);
    EXPECT_EQ(result.iterations, 4);
    EXPECT_GE(result.true_checks, 2);
    EXPECT_LE(result.max_residual, 1e-12);
    const std::vector<double> diagonal = {1.0, 3.0, 7.0, 11.0};
    for (int column = 0; column < columns; ++column)
    {
        for (int row = 0; row < dim; ++row)
        {
            EXPECT_NEAR(x[column * ld + row], rhs[column * ld + row] / diagonal[row], 1e-12);
        }
        EXPECT_EQ(x[column * ld + dim], 0.0); // padding is untouched
    }
}

TEST(LinearPCG, PreconditionedSolveFromNonzeroGuess)
{
    const Diagonal op({1.0, 3.0, 7.0});
    const Diagonal inverse({1.0, 1.0 / 3.0, 1.0 / 7.0});
    const auto comm = serial_comm();
    std::vector<double> x(6, 0.2);
    const std::vector<double> rhs = {1.0, 2.0, 3.0, -4.0, 5.0, 6.0};
    const auto result = hsolver::linear_pcg<Cpu>(op, inverse, x.data(), rhs.data(),
                                                3, 3, 2, 1e-12, 20, comm);
    EXPECT_EQ(result.status, hsolver::LinearSolveStatus::converged);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_LE(result.max_residual, 1e-12);
}

TEST(LinearPCG, RejectsCurvatureAndPreconditionerFailure)
{
    const Diagonal op({-1.0, 2.0});
    const Diagonal inverse({1.0, 1.0});
    const FailingInverse failing;
    const auto comm = serial_comm();
    std::vector<double> x(2, 0.0);
    const std::vector<double> rhs = {1.0, 0.0};
    auto result = hsolver::linear_pcg<Cpu>(op, inverse, x.data(), rhs.data(), 2, 2, 1, 1e-12, 20, comm);
    EXPECT_EQ(result.status, hsolver::LinearSolveStatus::breakdown);
    EXPECT_EQ(result.failed_band, 0);
    result = hsolver::linear_pcg<Cpu>(op, failing, x.data(), rhs.data(), 2, 2, 1, 1e-12, 20, comm);
    EXPECT_EQ(result.status, hsolver::LinearSolveStatus::preconditioner_failure);
}

TEST(LinearPCG, IterationLimitReportsTrueResidual)
{
    const Diagonal op({1.0, 3.0});
    const Diagonal inverse({1.0, 1.0});
    const auto comm = serial_comm();
    std::vector<double> x(2, 0.0);
    const std::vector<double> rhs = {1.0, 1.0};
    const auto result = hsolver::linear_pcg<Cpu>(op, inverse, x.data(), rhs.data(), 2, 2, 1, 1e-12, 1, comm);
    EXPECT_EQ(result.status, hsolver::LinearSolveStatus::max_iterations);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_NEAR(result.max_residual, std::sqrt(0.5), 1e-12);
    EXPECT_EQ(result.true_checks, 2);
}

TEST(LinearPCG, ZeroRhsAndNonfiniteData)
{
    const Diagonal op({1.0, 2.0});
    const Diagonal inverse({1.0, 1.0});
    const auto comm = serial_comm();
    std::vector<double> x(2, 0.0);
    std::vector<double> rhs(2, 0.0);
    auto result = hsolver::linear_pcg<Cpu>(op, inverse, x.data(), rhs.data(), 2, 2, 1, 1e-12, 20, comm);
    EXPECT_EQ(result.status, hsolver::LinearSolveStatus::converged);
    EXPECT_EQ(result.iterations, 0);
    rhs[0] = std::numeric_limits<double>::quiet_NaN();
    result = hsolver::linear_pcg<Cpu>(op, inverse, x.data(), rhs.data(), 2, 2, 1, 1e-12, 20, comm);
    EXPECT_EQ(result.status, hsolver::LinearSolveStatus::breakdown);
}

TEST(LinearPCG, EmptyPartitionAndInvalidPolicy)
{
    const Diagonal empty({});
    const auto comm = serial_comm();
    double x = 0.0;
    const double rhs = 0.0;
    const auto result = hsolver::linear_pcg<Cpu>(empty, empty, &x, &rhs, 0, 0, 2, 1e-12, 20, comm);
    EXPECT_EQ(result.status, hsolver::LinearSolveStatus::converged);
    EXPECT_EQ(result.operator_calls, 1);
    EXPECT_THROW(hsolver::linear_pcg<Cpu>(empty, empty, &x, &rhs, 0, 0, 2, 0.0, 20, comm),
                 std::invalid_argument);
}

#ifdef __CUDA
namespace
{
class GpuDiagonal : public hsolver::LinearOperator<double, base_device::DEVICE_GPU>
{
public:
    explicit GpuDiagonal(const std::vector<double>& values)
    {
        rows_ = static_cast<int>(values.size());
        diagonal_ = ct::Tensor(ct::DataType::DT_DOUBLE, ct::DeviceType::GpuDevice, {rows_});
        base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            diagonal_.data<double>(), values.data(), rows_);
    }
    void apply(const double* x, double* y, int ld, int columns) const override
    {
        const bool add = false;
        const double* diagonal = diagonal_.data<double>();
        for (int column = 0; column < columns; ++column)
        {
            const int offset = column * ld;
            const double* input = x + offset;
            double* output = y + offset;
            ModuleBase::vector_mul_vector_op<double, base_device::DEVICE_GPU, double>()(
                rows_, output, input, diagonal, add);
        }
    }
private:
    int rows_ = 0;
    ct::Tensor diagonal_;
};
}

TEST(LinearPCG, GpuIndependentColumnsAndTrueResidual)
{
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) { GTEST_SKIP() << "CUDA device unavailable"; }
    ModuleBase::createGpuBlasHandle();
    {
        const GpuDiagonal op({1.0, 3.0, 7.0});
        const GpuDiagonal inverse({1.0, 1.0, 1.0});
        const auto comm = serial_comm();
        const int ld = 4;
        const int dim = 3;
        const int columns = 3;
        const int size = ld * columns;
        ct::Tensor x(ct::DataType::DT_DOUBLE, ct::DeviceType::GpuDevice, {size});
        ct::Tensor b(ct::DataType::DT_DOUBLE, ct::DeviceType::GpuDevice, {size});
        x.zero();
        const std::vector<double> rhs = {2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            b.data<double>(), rhs.data(), size);
        const auto result = hsolver::linear_pcg<base_device::DEVICE_GPU>(
            op, inverse, x.data<double>(), b.data<double>(), ld, dim, columns, 1e-12, 20, comm);
        EXPECT_EQ(result.status, hsolver::LinearSolveStatus::converged);
        EXPECT_LE(result.max_residual, 1e-12);
        EXPECT_GE(result.true_checks, 2);
        std::vector<double> solution(size);
        base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
            solution.data(), x.data<double>(), size);
        const std::vector<double> diagonal = {1.0, 3.0, 7.0};
        for (int column = 0; column < columns; ++column)
        {
            for (int row = 0; row < dim; ++row)
            {
                const int offset = column * ld + row;
                EXPECT_NEAR(solution[offset], rhs[offset] / diagonal[row], 1e-11);
            }
        }
    }
    ModuleBase::destoryBLAShandle();
}
#endif
