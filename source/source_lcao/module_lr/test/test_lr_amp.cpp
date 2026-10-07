#include <gtest/gtest.h>
#include "../lr_amp.h"
#include "source_base/parallel_global.h"
#include <complex>
#include <sstream>

TEST(GradientAmplitudes, InitialRootSeedsTheReference)
{
    const double roots[] = {1.0, 0.0, 0.0, 1.0};
    int target = -1;
    std::vector<double> previous;
    std::ostringstream log;
    LR::follow_root(roots, 2, 2, 1, target, previous, log);
    EXPECT_EQ(target, 1);
    EXPECT_EQ(previous, (std::vector<double>{0.0, 1.0}));
}

TEST(GradientAmplitudes, FollowsACrossingIndependentOfSign)
{
    const double roots[] = {0.0, 1.0, -1.0, 0.0};
    int target = 0;
    std::vector<double> previous{1.0, 0.0};
    std::ostringstream log;
    LR::follow_root(roots, 2, 2, 0, target, previous, log);
    EXPECT_EQ(target, 1);
    EXPECT_EQ(previous, (std::vector<double>{-1.0, 0.0}));
    EXPECT_NE(log.str().find("root moved"), std::string::npos);
}

TEST(GradientAmplitudes, ComplexPhaseDoesNotChangeTheFollowedRoot)
{
    using Complex = std::complex<double>;
    const Complex roots[] = {Complex(0.0), Complex(1.0), Complex(0.0, 1.0), Complex(0.0)};
    int target = 0;
    std::vector<Complex> previous{Complex(1.0), Complex(0.0)};
    std::ostringstream log;
    LR::follow_root(roots, 2, 2, 0, target, previous, log);
    EXPECT_EQ(target, 1);
    EXPECT_EQ(previous[0], Complex(0.0, 1.0));
}

int main(int argc, char** argv)
{
    int processes = 1;
    int threads = 1;
    int rank = 0;
    Parallel_Global::read_pal_param(argc, argv, processes, threads, rank);
#ifdef __MPI
    // This focused test has no pool/grid setup; the cleanup wrapper needs null handles.
    POOL_WORLD = MPI_COMM_NULL;
    KP_WORLD = MPI_COMM_NULL;
    INT_BGROUP = MPI_COMM_NULL;
    BP_WORLD = MPI_COMM_NULL;
    GRID_WORLD = MPI_COMM_NULL;
    DIAG_WORLD = MPI_COMM_NULL;
#endif
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
#ifdef __MPI
    Parallel_Global::finalize_mpi();
#endif
    return result;
}
