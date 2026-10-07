#include "../exx_force_dm.h"

#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>
#ifdef __GPU_RI
#include <cuda_runtime_api.h>
#endif

namespace
{
using Kernel = LR::ExxForceTwoDM<int, int, 3, double>;
using Blocks = std::map<int, std::map<Kernel::TAC, RI::Tensor<double>>>;

Blocks tensors(int rank, double scale)
{
    Blocks result;
    const Kernel::TC cell = {{0, 0, 0}};
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            const std::size_t ni = 2 + i;
            const std::size_t nj = 2 + j;
            const std::size_t qi = 2 - i;
            const std::size_t qj = 2 - j;
            RI::Tensor<double> tensor;
            if (rank == 3) { tensor = RI::Tensor<double>({qi, ni, nj}); }
            else if (rank == 2) { tensor = RI::Tensor<double>({qi, qj}); }
            else { tensor = RI::Tensor<double>({ni, nj}); }
            const std::size_t size = tensor.shape.get_shape_all();
            for (std::size_t p = 0; p < size; ++p)
            {
                tensor.ptr()[p] = scale * std::sin(0.7 + 0.4 * i + 0.3 * j + 0.2 * p);
            }
            const Kernel::TAC key(j, cell);
            result[i][key] = std::move(tensor);
        }
    }
    return result;
}

void prepare(Kernel& kernel, RI::LRI_Cal_Mode mode)
{
    const std::map<int, Kernel::Tatom_pos> atoms = {{0, {{0.0, 0.0, 0.0}}}, {1, {{0.3, 0.2, 0.1}}}};
    const std::array<Kernel::Tatom_pos, 3> lattice = {{{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}};
    const Kernel::TC period = {{1, 1, 1}};
    kernel.set_parallel(MPI_COMM_SELF, atoms, lattice, period);
    kernel.lri.cal_mode = mode;
    const Blocks cs = tensors(3, 0.7);
    const Blocks vs = tensors(2, 0.4);
    const Blocks ds = tensors(1, 0.3);
    std::array<Blocks, 3> dcs;
    std::array<Blocks, 3> dvs;
    for (int axis = 0; axis < 3; ++axis)
    {
        const double cscale = 0.2 + 0.1 * axis;
        const double vscale = -0.1 - 0.03 * axis;
        dcs[axis] = tensors(3, cscale);
        dvs[axis] = tensors(2, vscale);
    }
    const std::string suffix;
    kernel.set_Cs(cs, 0.0, suffix);
    kernel.set_Vs(vs, 0.0, suffix);
    kernel.set_Ds(ds, 0.0, suffix);
    kernel.set_dCs(dcs, 0.0, suffix);
    kernel.set_dVs(dvs, 0.0, suffix);
    const std::array<std::string, 3> names = {{suffix, suffix, suffix}};
    kernel.cal_Hs(names);
}

void compare_blocks(const Blocks& expected, const Blocks& actual)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (const auto& atom : expected)
    {
        ASSERT_EQ(actual.at(atom.first).size(), atom.second.size());
        for (const auto& block : atom.second)
        {
            const auto& output = actual.at(atom.first).at(block.first);
            ASSERT_EQ(output.shape.size(), block.second.shape.size());
            for (std::size_t axis = 0; axis < output.shape.size(); ++axis)
            {
                EXPECT_EQ(output.shape[axis], block.second.shape[axis]);
            }
            const std::size_t size = output.shape.get_shape_all();
            for (std::size_t i = 0; i < size; ++i) { EXPECT_NEAR(output.ptr()[i], block.second.ptr()[i], 1e-11); }
        }
    }
}
}

TEST(ExxForceTwoDM, SeparateLeftDensityScalesForce)
{
    Kernel kernel;
    prepare(kernel, RI::LRI_Cal_Mode::CPU);
    const std::array<std::string, 5> names = {{"", "", "", "", ""}};
    const Blocks left = tensors(1, 0.6);
    const Blocks negative_left = tensors(1, -0.3);
    kernel.cal_force(left, names);
    const auto reference = kernel.force;
    kernel.cal_force(negative_left, names);
    double magnitude = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        for (const auto& atom : reference[axis])
        {
            magnitude += std::abs(atom.second);
            const double expected = -0.5 * atom.second;
            EXPECT_NEAR(kernel.force[axis].at(atom.first), expected, 1e-11);
        }
    }
    EXPECT_GT(magnitude, 1e-5);
}

#ifdef __GPU_RI
TEST(ExxForceTwoDM, GpuSignedAccumulationAndDerivativeForce)
{
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) { GTEST_SKIP() << "No CUDA device available"; }
    Kernel cpu;
    Kernel gpu;
    prepare(cpu, RI::LRI_Cal_Mode::CPU);
    prepare(gpu, RI::LRI_Cal_Mode::GPU);
    compare_blocks(cpu.Hs, gpu.Hs);
    const std::vector<RI::Label::ab_ab> labels = {
        RI::Label::ab_ab::a0b0_a1b1, RI::Label::ab_ab::a0b0_a1b2,
        RI::Label::ab_ab::a0b0_a2b1, RI::Label::ab_ab::a0b0_a2b2};
    Blocks cpu_sum;
    Blocks gpu_sum;
    // Existing output must be preserved, and each signed factor applied exactly once.
    for (const double factor : {-0.75, 0.25, 0.0})
    {
        cpu.lri.cal_loop3(labels, cpu_sum, factor);
        gpu.lri.cal_loop3(labels, gpu_sum, factor);
        compare_blocks(cpu_sum, gpu_sum);
    }
    const Blocks left = tensors(1, 0.6);
    const std::array<std::string, 5> names = {{"", "", "", "", ""}};
    cpu.cal_force(left, names);
    gpu.cal_force(left, names);
    double magnitude = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        ASSERT_EQ(cpu.force[axis].size(), gpu.force[axis].size());
        for (const auto& atom : cpu.force[axis])
        {
            magnitude += std::abs(atom.second);
            EXPECT_NEAR(gpu.force[axis].at(atom.first), atom.second, 1e-10);
        }
    }
    EXPECT_GT(magnitude, 1e-5);
}
#endif
