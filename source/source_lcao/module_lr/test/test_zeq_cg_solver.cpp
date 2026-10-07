#include "../zeq_cg_solver.h"
#include "source_hsolver/diag_comm_info.h"

#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <vector>

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
    const auto result = LR::solve_Z_CG(z.data(), rhs.data(), 2, 3, action, comm);
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
    EXPECT_THROW(LR::solve_Z_CG(&z, &rhs, 1, 1, action, comm), std::runtime_error);
}

TEST(ZeqCGSolver, RejectsComplexEquation)
{
    std::complex<double> z;
    const std::complex<double> rhs(1.0, 0.0);
    const std::function<void(const std::complex<double>*, std::complex<double>*, int, int)> action;
    const auto comm = serial_comm();
    EXPECT_THROW(LR::solve_Z_CG(&z, &rhs, 1, 1, action, comm), std::runtime_error);
}
