#include <gtest/gtest.h>
#include <xc.h>
#include "../potentials/xc_components.h"
#include <cmath>
#include <array>
#include <algorithm>
#include <vector>

class XCComponents : public testing::Test
{
protected:
    std::vector<xc_func_type> functions = std::vector<xc_func_type>(1);
    void TearDown() override
    {
        if (functions[0].info != nullptr) { xc_func_end(&functions[0]); }
    }
};

TEST_F(XCComponents, PBEAndPBE0RemainUnsplitAndUnclamped)
{
    for (const int id : {XC_GGA_X_PBE, XC_GGA_C_PBE, XC_HYB_GGA_XC_PBEH})
    {
        ASSERT_EQ(xc_func_init(&functions[0], id, XC_POLARIZED), 0);
        const auto components = LR::kernel_components(functions);
        ASSERT_EQ(components.size(), 1u);
        EXPECT_EQ(components[0].functional, &functions[0]);
        EXPECT_EQ(components[0].weight, 1.0);
        EXPECT_FALSE(components[0].stabilize_exchange);
        xc_func_end(&functions[0]);
        functions[0].info = nullptr;
    }
}

TEST_F(XCComponents, BorrowsHSEScreenedAndUnscreenedExchangeWithInitializedWeights)
{
    ASSERT_EQ(xc_func_init(&functions[0], XC_HYB_GGA_XC_HSE06, XC_POLARIZED), 0);
    const double parameters[]{0.37, 0.21, 0.21};
    xc_func_set_ext_params(&functions[0], parameters);
    const auto components = LR::kernel_components(functions);
    ASSERT_EQ(components.size(), 3u);
    EXPECT_DOUBLE_EQ(components[0].weight, 1.0);
    EXPECT_DOUBLE_EQ(components[1].weight, -0.37);
    EXPECT_DOUBLE_EQ(components[2].weight, 1.0);
    for (int index = 0; index < 3; ++index)
    {
        EXPECT_EQ(components[index].functional, functions[0].func_aux[index]);
        EXPECT_EQ(components[index].cutoff_functional, &functions[0]);
        EXPECT_EQ(components[index].stabilize_exchange, index < 2);
    }
}

TEST_F(XCComponents, RawComponentSumMatchesHSEAndMixedSpinKernelIsCorrelation)
{
    ASSERT_EQ(xc_func_init(&functions[0], XC_HYB_GGA_XC_HSE06, XC_POLARIZED), 0);
    const double rho[]{0.2, 0.15};
    const double sigma[]{0.01, -0.008, 0.0064};
    double combined[3]{};
    double mixed[6]{};
    double gradient[6]{};
    xc_gga_fxc(&functions[0], 1, rho, sigma, combined, mixed, gradient);
    std::vector<double> sum(3, 0.0);
    for (const auto& component : LR::kernel_components(functions))
    {
        double rr[3]{};
        double rs[6]{};
        double ss[6]{};
        xc_gga_fxc(component.functional, 1, rho, sigma, rr, rs, ss);
        if (component.stabilize_exchange) { EXPECT_NEAR(rr[1], 0.0, 1e-13); }
        else { EXPECT_NEAR(rr[1], combined[1], 1e-13); }
        for (int index = 0; index < 3; ++index) { sum[index] += component.weight * rr[index]; }
    }
    for (int index = 0; index < 3; ++index) { EXPECT_NEAR(sum[index], combined[index], 1e-13); }
}

namespace
{
std::array<double, 9> gga_derivatives(xc_func_type* functional)
{
    const double rho = 0.4;
    const double sigma = 0.1;
    std::array<double, 9> values{};
    xc_gga_vxc(functional, 1, &rho, &sigma, &values[0], &values[1]);
    xc_gga_fxc(functional, 1, &rho, &sigma, &values[2], &values[3], &values[4]);
    xc_gga_kxc(functional, 1, &rho, &sigma, &values[5], &values[6], &values[7], &values[8]);
    return values;
}
}

TEST_F(XCComponents, AllDerivativeOrdersUseInitializedMixingWeights)
{
    ASSERT_EQ(xc_func_init(&functions[0], XC_HYB_GGA_XC_HSE06, XC_UNPOLARIZED), 0);
    if ((xc_func_info_get_flags(functions[0].info) & XC_FLAGS_HAVE_KXC) == 0)
    {
        GTEST_SKIP() << "This LibXC runtime does not supply HSE KXC";
    }
    const double parameters[]{0.37, 0.21, 0.21};
    xc_func_set_ext_params(&functions[0], parameters);
    const auto expected = gga_derivatives(&functions[0]);
    std::array<double, 9> actual{};
    for (const auto& component : LR::kernel_components(functions))
    {
        const auto values = gga_derivatives(component.functional);
        for (size_t index = 0; index < actual.size(); ++index)
        {
            actual[index] += component.weight * values[index];
        }
    }
    for (size_t index = 0; index < actual.size(); ++index)
    {
        const double tolerance = 1e-11 * std::max(1.0, std::abs(expected[index]));
        EXPECT_NEAR(actual[index], expected[index], tolerance);
    }
}
