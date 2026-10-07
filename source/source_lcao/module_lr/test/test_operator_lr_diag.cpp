#include "../operator_casida/operator_lr_diag.h"
#include "source_base/parallel_2d.h"

#include <gtest/gtest.h>
#include <vector>

TEST(OperatorLRDiag, ExposesLocalGapsForEveryKPoint)
{
    // Two occupied, three virtual bands per k; [k][occupied][virtual] local layout.
    const std::vector<double> eigenvalues = {-2.0, -1.0, 1.0, 3.0, 5.0,
                                            -3.0, -0.5, 0.5, 2.0, 4.0};
    Parallel_2D pairs;
    pairs.set_serial(3, 2);
    const LR::OperatorLRDiag<double> diagonal(eigenvalues.data(), pairs, 2, 2, 3, false);
    const auto& gaps = diagonal.energy_differences();
    const std::vector<double> expected = {3.0, 5.0, 7.0, 2.0, 4.0, 6.0,
                                         3.5, 5.0, 7.0, 1.0, 2.5, 4.5};
    for (std::size_t i = 0; i < expected.size(); ++i) { EXPECT_DOUBLE_EQ(gaps.c[i], expected[i]); }
}
