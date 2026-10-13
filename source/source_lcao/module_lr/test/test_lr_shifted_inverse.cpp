#include "../lr_shifted_inverse.h"
#include "source_hsolver/diag_comm_info.h"
#include <gtest/gtest.h>
#include <complex>

namespace
{
hsolver::diag_comm_info serial_comm()
{
#ifdef __MPI
    return {MPI_COMM_SELF, 0, 1};
#else
    return {0, 1};
#endif
}

template<typename T> void check_inverse(T coupling)
{
    const T adjoint = 2.0 * std::real(coupling) - coupling;
    const std::function<void(const T*, T*)> action = [&](const T* x, T* y) {
        y[0] = 2.0 * x[0] + coupling * x[1];
        y[1] = adjoint * x[0] + 3.0 * x[1];
        y[2] = 6.0 * x[2];
    };
    const std::vector<double> gaps = {2, 3, 6};
    LR::ShiftedInverseSettings settings;
    settings.relative_tolerance = 1e-11;
    const auto comm = serial_comm();
    for (double shift : {0.0, 2.5, 3.0}) // SPD, indefinite, and a zero gap denominator.
    {
        std::vector<T> rhs = {T(1), T(-3), T(2)};
        const auto original = rhs;
        const auto result = LR::apply_shifted_inverse(rhs.data(), 3, shift, gaps, action, settings, comm);
        EXPECT_TRUE(result.converged);
        EXPECT_FALSE(result.used_gap_fallback);
        EXPECT_LE(result.operator_vectors, 4);
        std::vector<T> image(3);
        action(rhs.data(), image.data());
        for (int j = 0; j < 3; ++j) { EXPECT_NEAR(std::abs(image[j] - shift * rhs[j] - original[j]), 0, 1e-9); }
    }
}
}

TEST(LRShiftedInverse, RealIndefinite) { check_inverse(0.4); }
TEST(LRShiftedInverse, ComplexIndefinite)
{
    const std::complex<double> coupling(0.4, 0.3);
    check_inverse(coupling);
}

TEST(LRShiftedInverse, ZeroRhsAndSingularFallback)
{
    int calls = 0;
    const std::function<void(const double*, double*)> zero = [&](const double*, double* y) {
        ++calls;
        y[0] = 0;
        y[1] = 0;
    };
    const auto comm = serial_comm();
    const std::vector<double> gaps = {2, 4};
    const LR::ShiftedInverseSettings settings;
    double rhs[] = {0, 0};
    auto result = LR::apply_shifted_inverse(rhs, 2, 0, gaps, zero, settings, comm);
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(calls, 0);
    rhs[0] = 2;
    rhs[1] = 8;
    result = LR::apply_shifted_inverse(rhs, 2, 0, gaps, zero, settings, comm);
    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(result.used_gap_fallback);
    EXPECT_DOUBLE_EQ(result.relative_residual, 1);
    EXPECT_DOUBLE_EQ(rhs[0], 1);
    EXPECT_DOUBLE_EQ(rhs[1], 2);
}

TEST(LRShiftedInverse, InexactCorrectionRetainsMeasuredResidual)
{
    const auto comm = serial_comm();
    LR::ShiftedInverseSettings settings;
    settings.max_iterations = 1;
    const std::vector<double> gaps = {1, 1};
    const std::function<void(const double*, double*)> action = [](const double* x, double* y) {
        y[0] = 2 * x[0];
        y[1] = 5 * x[1];
    };
    double rhs[] = {1, 1};
    const auto result = LR::apply_shifted_inverse(rhs, 2, 0, gaps, action, settings, comm);
    EXPECT_FALSE(result.converged);
    EXPECT_FALSE(result.used_gap_fallback);
    const double error = std::hypot(2 * rhs[0] - 1, 5 * rhs[1] - 1) / std::sqrt(2.0);
    EXPECT_NEAR(result.relative_residual, error, 1e-14);
    EXPECT_EQ(result.operator_vectors, 2);
}
