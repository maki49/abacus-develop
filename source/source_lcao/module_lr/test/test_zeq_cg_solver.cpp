#include "../zeq_cg_solver.h"
#include "source_hsolver/diag_comm_info.h"
#ifdef __MPI
#include "source_base/parallel_comm.h"
#include "source_base/parallel_device.h"
#endif

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef __CUDA
#include "source_base/kernels/math_kernel_op.h"
#endif

namespace
{
hsolver::diag_comm_info serial_comm()
{
#ifdef __MPI
    return hsolver::diag_comm_info(MPI_COMM_SELF, 0, 1);
#else
    return hsolver::diag_comm_info(0, 1);
#endif
}
}

namespace
{
void compare_orbital_preconditioner(bool use_gpu)
{
    const std::vector<double> diagonal = {1e-4, 0.5, 3.0, 100.0};
    const std::vector<double> expected = {1.0, -2.0, 3.0, -4.0, -3.0, 1.0, 2.0, 4.0};
    std::vector<double> rhs(expected.size());
    for (std::size_t i = 0; i < rhs.size(); ++i) { rhs[i] = diagonal[i % 4] * expected[i]; }
    const auto original_rhs = rhs;
    std::vector<double> unpreconditioned(rhs.size());
    std::vector<double> preconditioned(rhs.size());
    const LR::ZHessianAction action = [&diagonal](const double* x, double* y, int ld, int states)
    {
        for (int state = 0; state < states; ++state)
        {
            for (int row = 0; row < ld; ++row)
            {
                const int offset = state * ld + row;
                y[offset] = diagonal[row] * x[offset];
            }
        }
    };
    const auto comm = serial_comm();
    const auto baseline = LR::solve_Z_CG(unpreconditioned.data(), rhs.data(), 4, 2, action, comm, use_gpu, {});
    const auto result = LR::solve_Z_CG(preconditioned.data(), rhs.data(), 4, 2, action, comm, use_gpu, diagonal);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_LT(result.iterations, baseline.iterations);
    EXPECT_LE(result.max_residual, 1e-6);
    EXPECT_GE(result.true_checks, 2);
    for (std::size_t i = 0; i < rhs.size(); ++i)
    {
        EXPECT_NEAR(preconditioned[i], expected[i], 1e-10);
        EXPECT_NEAR(unpreconditioned[i], expected[i], 1e-5);
    }
    EXPECT_EQ(rhs, original_rhs);
}
}

TEST(ZeqCGSolver, OrbitalPreconditionerReducesIterations)
{
    compare_orbital_preconditioner(false);
}

TEST(ZeqCGSolver, RejectsInvalidOrbitalGaps)
{
    double z = 0.0;
    const double rhs = 1.0;
    const auto comm = serial_comm();
    const LR::ZHessianAction action = [](const double* x, double* y, int, int) { y[0] = x[0]; };
    const std::vector<double> invalid = {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::quiet_NaN()};
    for (const double gap : invalid)
    {
        const std::vector<double> diagonal = {gap};
        EXPECT_THROW(LR::solve_Z_CG(&z, &rhs, 1, 1, action, comm, false, diagonal), std::invalid_argument);
        EXPECT_THROW(LR::solve_Z_GMRES(&z, &rhs, 1, 1, action, comm, diagonal), std::invalid_argument);
    }
    const std::vector<double> wrong_size = {1.0, 2.0};
    EXPECT_THROW(LR::solve_Z_CG(&z, &rhs, 1, 1, action, comm, false, wrong_size), std::invalid_argument);
    EXPECT_THROW(LR::solve_Z_GMRES(&z, &rhs, 1, 1, action, comm, wrong_size), std::invalid_argument);
}

TEST(ZeqCGSolver, GapFloorDoesNotAlterHessian)
{
    double z = 0.0;
    const double rhs = 1.0;
    const std::vector<double> diagonal = {1e-12};
    const auto comm = serial_comm();
    const LR::ZHessianAction action = [](const double* x, double* y, int, int) { y[0] = 1e-12 * x[0]; };
    const auto result = LR::solve_Z_CG(&z, &rhs, 1, 1, action, comm, false, diagonal);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_NEAR(z, 1e12, 1e-3);
    EXPECT_LE(result.max_residual, 1e-6);
}

TEST(ZeqCGSolver, GmresSolvesIndefiniteHessianWithIndependentAndZeroRhs)
{
    // H = [[-2, 1], [1, 3]] has one negative eigenvalue. Positive orbital
    // gaps remain a valid inverse preconditioner; they do not make H positive.
    const std::vector<double> expected = {1.0, 2.0, 0.0, 0.0, -3.0, 1.0};
    const std::vector<double> rhs = {0.0, 7.0, 0.0, 0.0, 7.0, 0.0};
    const auto original_rhs = rhs;
    std::vector<double> z(rhs.size(), 99.0);
    int calls = 0;
    const LR::ZHessianAction action = [&calls](const double* x, double* y, int ld, int states)
    {
        ++calls;
        for (int state = 0; state < states; ++state)
        {
            const int offset = state * ld;
            y[offset] = -2 * x[offset] + x[offset + 1];
            y[offset + 1] = x[offset] + 3 * x[offset + 1];
        }
    };
    const auto comm = serial_comm();
    const std::vector<double> gaps = {2.0, 3.0};
    const auto result = LR::solve_Z_GMRES(z.data(), rhs.data(), 2, 3, action, comm, gaps);
    ASSERT_EQ(result.status, hsolver::LinearSolveStatus::converged);
    EXPECT_LE(result.max_residual, 1e-6);
    EXPECT_GE(result.true_checks, 1);
    EXPECT_EQ(result.operator_calls, calls);
    for (std::size_t i = 0; i < z.size(); ++i) { EXPECT_NEAR(z[i], expected[i], 1e-12); }
    EXPECT_EQ(rhs, original_rhs);
}

TEST(ZeqCGSolver, GmresUsesAbsoluteToleranceForLargeRhs)
{
    const LR::ZHessianAction action = [](const double* x, double* y, int ld, int states)
    {
        for (int state = 0; state < states; ++state)
        {
            for (int row = 0; row < ld; ++row)
            {
                const int offset = state * ld + row;
                const double diagonal = row == 0 ? -1.0 : 2.0;
                y[offset] = diagonal * x[offset];
            }
        }
    };
    // A relative tolerance of 1e-6 would accept missing the small second component.
    const std::vector<double> rhs = {1e6, 0.2};
    std::vector<double> z(2, 0.0);
    const auto comm = serial_comm();
    const auto result = LR::solve_Z_GMRES(z.data(), rhs.data(), 2, 1, action, comm, {});
    EXPECT_LE(result.max_residual, 1e-6);
    EXPECT_NEAR(z[0], -1e6, 1e-6);
    EXPECT_NEAR(z[1], 0.1, 1e-7);
}

TEST(ZeqCGSolver, GmresRejectsInconsistentSingularHessian)
{
    const LR::ZHessianAction action = [](const double* x, double* y, int, int)
    {
        y[0] = x[0];
        y[1] = 0.0;
    };
    const std::vector<double> rhs = {0.0, 1.0};
    std::vector<double> z(2);
    const auto comm = serial_comm();
    EXPECT_THROW(LR::solve_Z_GMRES(z.data(), rhs.data(), 2, 1, action, comm, {}), std::runtime_error);
}

#ifdef __MPI
TEST(ZeqCGSolver, GmresDistributedHessianIncludingEmptyRanks)
{
    MPICommGroup group(MPI_COMM_WORLD);
    if (group.gsize < 2) { GTEST_SKIP(); }
    const hsolver::diag_comm_info comm(MPI_COMM_WORLD, group.grank, group.gsize);
    const int ld = group.grank < 2 ? 1 : 0;
    const int states = 3;
    std::vector<double> rhs(states, 0.0);
    std::vector<double> z(states, 99.0);
    const std::vector<double> full_rhs = {0.0, 7.0, 0.0, 0.0, 7.0, 0.0};
    const std::vector<double> expected = {1.0, 2.0, 0.0, 0.0, -3.0, 1.0};
    if (ld > 0)
    {
        for (int state = 0; state < states; ++state) { rhs[state] = full_rhs[2 * state + group.grank]; }
    }
    const LR::ZHessianAction action = [&comm](const double* x, double* y, int local_ld, int columns)
    {
        const int count = 2 * columns;
        std::vector<double> full(count, 0.0);
        if (local_ld > 0)
        {
            for (int state = 0; state < columns; ++state) { full[2 * state + comm.rank] = x[state]; }
        }
        Parallel_Common::reduce_data(full.data(), count, comm.comm);
        for (int state = 0; local_ld > 0 && state < columns; ++state)
        {
            const double first = full[2 * state];
            const double second = full[2 * state + 1];
            y[state] = comm.rank == 0 ? -2 * first + second : first + 3 * second;
        }
    };
    const std::vector<double> gaps(ld, 2.0);
    const auto result = LR::solve_Z_GMRES(z.data(), rhs.data(), ld, states, action, comm, gaps);
    EXPECT_EQ(result.status, hsolver::LinearSolveStatus::converged);
    EXPECT_LE(result.max_residual, 1e-6);
    for (int state = 0; ld > 0 && state < states; ++state)
    {
        EXPECT_NEAR(z[state], expected[2 * state + group.grank], 1e-12);
    }
}
#endif

TEST(ZeqCGSolver, CoupledHessianIndependentStatesAndUnchangedRhs)
{
    // Symmetric coupled Hessian, with analytic solutions [1,2], [-3,1], [0,0].
    const std::vector<double> rhs = {6.0, 7.0, -11.0, 0.0, 0.0, 0.0};
    std::vector<double> z(6, 99.0);
    int calls = 0;
    const LR::ZHessianAction action = [&](const double* x, double* y, int ld, int states)
    {
        EXPECT_EQ(ld, 2);
        EXPECT_EQ(states, 3);
        ++calls;
        for (int state = 0; state < states; ++state)
        {
            const int offset = state * ld;
            y[offset] = 4.0 * x[offset] + x[offset + 1];
            y[offset + 1] = x[offset] + 3.0 * x[offset + 1];
        }
    };
    const auto comm = serial_comm();
    const auto result = LR::solve_Z_CG(z.data(), rhs.data(), 2, 3, action, comm, false, {});
    ASSERT_EQ(result.status, hsolver::LinearSolveStatus::converged);
    EXPECT_GE(result.true_checks, 2);
    EXPECT_EQ(result.operator_calls, calls);
    const std::vector<double> expected = {1.0, 2.0, -3.0, 1.0, 0.0, 0.0};
    for (std::size_t i = 0; i < z.size(); ++i) { EXPECT_NEAR(z[i], expected[i], 1e-12); }
    EXPECT_EQ(rhs, (std::vector<double>{6.0, 7.0, -11.0, 0.0, 0.0, 0.0}));
}

TEST(ZeqCGSolver, RejectsIndefiniteHessian)
{
    double z = 0.0;
    const double rhs = 1.0;
    const LR::ZHessianAction action = [](const double* x, double* y, int, int) { y[0] = -x[0]; };
    const auto comm = serial_comm();
    EXPECT_THROW(LR::solve_Z_CG(&z, &rhs, 1, 1, action, comm, false, {}), std::runtime_error);
}

TEST(ZeqCGSolver, RejectsComplexEquation)
{
    std::complex<double> z;
    const std::complex<double> rhs(1.0, 0.0);
    const std::function<void(const std::complex<double>*, std::complex<double>*, int, int)> action;
    const auto comm = serial_comm();
    EXPECT_THROW(LR::solve_Z_CG(&z, &rhs, 1, 1, action, comm, false, {}), std::runtime_error);
}

#ifdef __CUDA
TEST(ZeqCGSolver, GpuOrbitalPreconditionerReducesIterations)
{
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) { GTEST_SKIP() << "No CUDA device available"; }
    compare_orbital_preconditioner(true);
    ModuleBase::destoryBLAShandle();
}

TEST(ZeqCGSolver, GpuRecurrencesWithHostHessian)
{
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) { GTEST_SKIP() << "No CUDA device available"; }
    const std::vector<double> rhs = {6.0, 7.0, -11.0, 0.0, 0.0, 0.0};
    std::vector<double> z(6, 99.0);
    const LR::ZHessianAction action = [](const double* x, double* y, int ld, int states)
    {
        for (int state = 0; state < states; ++state)
        {
            const int offset = state * ld;
            y[offset] = 4.0 * x[offset] + x[offset + 1];
            y[offset + 1] = x[offset] + 3.0 * x[offset + 1];
        }
    };
    const auto comm = serial_comm();
    const auto result = LR::solve_Z_CG(z.data(), rhs.data(), 2, 3, action, comm, true, {});
    ASSERT_EQ(result.status, hsolver::LinearSolveStatus::converged);
    const std::vector<double> expected = {1.0, 2.0, -3.0, 1.0, 0.0, 0.0};
    for (std::size_t i = 0; i < z.size(); ++i) { EXPECT_NEAR(z[i], expected[i], 1e-12); }
    const LR::ZHessianAction indefinite = [](const double* x, double* y, int, int) { y[0] = -x[0]; };
    EXPECT_THROW(LR::solve_Z_CG(z.data(), rhs.data(), 1, 1, indefinite, comm, true, {}), std::runtime_error);
    ModuleBase::destoryBLAShandle();
}
#else
TEST(ZeqCGSolver, RejectsGpuInCpuBuild)
{
    double z = 0.0;
    const double rhs = 1.0;
    const LR::ZHessianAction action;
    const auto comm = serial_comm();
    EXPECT_THROW(LR::solve_Z_CG(&z, &rhs, 1, 1, action, comm, true, {}), std::runtime_error);
}
#endif
