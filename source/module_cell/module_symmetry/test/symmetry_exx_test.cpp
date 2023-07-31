#include "gtest/gtest.h"
#include "../symmetry_exx.h"
#include <map>
#include <tuple>


class SymExxTest : public testing::Test
{
protected:
    std::map<std::vector<int>, std::vector<int>> invmap_cases = {
        {{3, 2, 1, 0}, {3, 2, 1, 0}},
     { {4, 1, 3, 0, 2}, {3, 1, 4, 2, 0} } };
    std::vector<std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>> mapmul_cases = {
        {{4, 1, 3, 0, 2}, {3, 1, 4, 2, 0}, {0, 1, 2, 3, 4}},
        {{3, 1, 4, 2, 0}, {2, 3, 0, 1, 4}, {1, 3, 4, 0, 2}}
    };
};

TEST_F(SymExxTest, invmap)
{
    for (auto c : invmap_cases)
    {
        std::vector<int> invf = SymExx::invmap(c.first.data(), c.first.size());
        EXPECT_EQ(invf, c.second);
    }
}

TEST_F(SymExxTest, mapmul)
{
    for (auto c : mapmul_cases)
    {
        std::vector<int> f2f1 = SymExx::mapmul(std::get<0>(c).data(), std::get<1>(c).data(), std::get<0>(c).size());
        EXPECT_EQ(f2f1, std::get<2>(c));
    }
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}