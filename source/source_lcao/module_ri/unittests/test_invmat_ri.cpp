#include "../invmat_ri.h"
#include "source_base/parallel_global.h"
#include <gtest/gtest.h>
#include <complex>

namespace
{
template<typename T>
void check_derivative(const double cutoff, const T phase)
{
    const double angle = 0.31;
    const double step = 1e-5;
    auto metric = [phase](const double x) {
        RI::Tensor<T> q({3, 3});
        q(0, 0) = std::cos(x);
        q(0, 2) = -std::sin(x) * phase;
        q(1, 1) = 1;
        q(2, 0) = std::sin(x) * RI::Global_Func::get_conj(phase);
        q(2, 2) = std::cos(x);
        RI::Tensor<T> diagonal({3, 3});
        diagonal(0, 0) = 2;
        diagonal(1, 1) = 2;
        diagonal(2, 2) = 0.01;
        return q * diagonal * q.conjugate().transpose();
    };
    RI::Tensor<T> generator({3, 3});
    generator(0, 2) = -phase;
    generator(2, 0) = RI::Global_Func::get_conj(phase);
    const auto v = metric(angle);
    const auto dv = generator * v - v * generator;
    const std::array<RI::Tensor<T>, 3> directions = {dv, RI::Tensor<T>({3, 3}), RI::Tensor<T>({3, 3})};
    Inverse_Matrix<T> inverse;
    inverse.input(v);
    const auto derivatives = inverse.cal_inverse_with_derivatives(directions, cutoff);
    Inverse_Matrix<T> plus;
    plus.input(metric(angle + step));
    plus.cal_inverse(Inverse_Matrix<T>::Method::syev, cutoff);
    Inverse_Matrix<T> minus;
    minus.input(metric(angle - step));
    minus.cal_inverse(Inverse_Matrix<T>::Method::syev, cutoff);
    const auto fd = (plus.output() - minus.output()) * T(0.5 / step);
    const auto response = inverse.output() * v;
    for (std::size_t i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(std::abs(derivatives[0].ptr()[i] - fd.ptr()[i]), 0, 2e-6);
        EXPECT_NEAR(std::abs(derivatives[1].ptr()[i]), 0, 1e-12);
        EXPECT_NEAR(std::abs(derivatives[2].ptr()[i]), 0, 1e-12);
    }
    const double expected_trace = cutoff > 0 ? 2 : 3;
    const T trace = response(0, 0) + response(1, 1) + response(2, 2);
    EXPECT_NEAR(std::abs(trace - T(expected_trace)), 0, 1e-10);
}
}

TEST(SpectralInverse, RealRetainedDiscardedRotation) { check_derivative<double>(0.1, 1.0); }
TEST(SpectralInverse, ComplexRetainedDiscardedRotation)
{
    check_derivative<std::complex<double>>(0.1, std::polar(1.0, 0.7));
}
TEST(SpectralInverse, RealFullInverse) { check_derivative<double>(0.0, 1.0); }
TEST(SpectralInverse, ComplexFullInverse)
{
    check_derivative<std::complex<double>>(0.0, std::polar(1.0, 0.7));
}

int main(int argc, char** argv)
{
    int processes = 1;
    int threads = 1;
    int rank = 0;
    Parallel_Global::read_pal_param(argc, argv, processes, threads, rank);
    POOL_WORLD = MPI_COMM_NULL;
    KP_WORLD = MPI_COMM_NULL;
    INT_BGROUP = MPI_COMM_NULL;
    BP_WORLD = MPI_COMM_NULL;
    GRID_WORLD = MPI_COMM_NULL;
    DIAG_WORLD = MPI_COMM_NULL;
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    Parallel_Global::finalize_mpi();
    return result;
}
