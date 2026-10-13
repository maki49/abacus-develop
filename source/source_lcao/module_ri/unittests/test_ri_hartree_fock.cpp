#include "../ri_hartree_fock.h"
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#include <RI/global/Global_Func-1.h>
#include <RI/global/MPI_Wrapper.h>
#include <gtest/gtest.h>
#include <complex>
#include <set>

namespace
{
template<typename T> T amplitude(const int i, const int j);
template<> double amplitude<double>(const int i, const int j) { return 0.17 + 0.11 * i - 0.07 * j; }
template<> std::complex<double> amplitude<std::complex<double>>(const int i, const int j)
{
    return {0.17 + 0.11 * i - 0.07 * j, 0.13 - 0.05 * i + 0.09 * j};
}

LR::RIGeometry geometry(const int cells)
{
    LR::RIGeometry result;
    result.positions = {{0, {{0.0, 0.0, 0.0}}}, {1, {{0.3, 0.0, 0.0}}}};
    result.lattice = {{{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}};
    result.period = {{cells, 1, 1}};
    result.orbitals_per_atom = {2, 1};
    return result;
}

template<typename T> LR::RIFittingTensors<T> fitting(const int cells, const bool hartree)
{
    LR::RIFittingTensors<T> fit;
    fit.auxiliary_per_atom = hartree ? std::vector<int>{2, 1} : std::vector<int>{1, 1};
    const std::vector<int> nao{2, 1};
    for (int a = 0; a < 2; ++a)
    {
        for (int b = 0; b < 2; ++b)
        {
            for (int r = 0; r < cells; ++r)
            {
                const std::size_t naux_a = fit.auxiliary_per_atom[a];
                const std::size_t naux_b = fit.auxiliary_per_atom[b];
                const std::size_t nao_a = nao[a];
                const std::size_t nao_b = nao[b];
                const LR::RIAtomCell pair{b, {{r, 0, 0}}};
                auto& c = fit.Cs[a][pair];
                c = RI::Tensor<T>({naux_a, nao_a, nao_b});
                for (int p = 0; p < naux_a; ++p)
                {
                    for (int i = 0; i < nao_a; ++i)
                    {
                        for (int j = 0; j < nao_b; ++j)
                        {
                            c(p, i, j) = (0.2 + 0.03 * p + 0.02 * a + 0.04 * i - 0.015 * j) / (r + 1);
                        }
                    }
                }
                auto& v = fit.Vs[a][pair];
                v = RI::Tensor<T>({naux_a, naux_b});
                for (int p = 0; p < naux_a; ++p)
                {
                    for (int q = 0; q < naux_b; ++q)
                    {
                        v(p, q) = 0.05;
                        if (a == b && p == q && r == 0) { v(p, q) += 1.0; }
                    }
                }
            }
        }
    }
    return fit;
}

template<typename T> LR::RIAtomPairMap<T> distribute(const LR::RIAtomPairMap<T>& input)
{
    const int rank = RI::MPI_Wrapper::mpi_get_rank(MPI_COMM_WORLD);
    const int size = RI::MPI_Wrapper::mpi_get_size(MPI_COMM_WORLD);
    LR::RIAtomPairMap<T> result;
    int index = 0;
    for (const auto& row : input)
    {
        for (const auto& block : row.second)
        {
            if (index % size == rank) { result[row.first][block.first] = block.second; }
            ++index;
        }
    }
    return result;
}

// Independent reference: expand the two-center fit into the full Born-von Karman
// supercell, explicitly form (ij|kl), then contract its Hartree/exchange index order.
template<typename T> LR::RIAtomPairMap<T> dense_reference(
    const LR::RIFittingTensors<T>& fit, const LR::RIAtomPairMap<T>& density,
    const int cells, const bool hartree)
{
    const std::vector<int> nao{2, 1};
    const std::vector<int> ao_offset{0, 2};
    const int aux_per_cell = fit.auxiliary_per_atom[0] + fit.auxiliary_per_atom[1];
    const std::vector<int> aux_offset{0, fit.auxiliary_per_atom[0]};
    const std::size_t n = 3 * cells;
    const std::size_t m = aux_per_cell * cells;
    RI::Tensor<T> product({n, n, m});
    RI::Tensor<T> metric({m, m});
    RI::Tensor<T> dm({n, n});
    for (int origin = 0; origin < cells; ++origin)
    {
        for (int a = 0; a < 2; ++a)
        {
            for (int b = 0; b < 2; ++b)
            {
                for (int r = 0; r < cells; ++r)
                {
                    const LR::RIAtomCell pair{b, {{r, 0, 0}}};
                    const auto& c = fit.Cs.at(a).at(pair);
                    const auto& v = fit.Vs.at(a).at(pair);
                    const auto& d = density.at(a).at(pair);
                    const int neighbor = (origin + r) % cells;
                    for (int i = 0; i < nao[a]; ++i)
                    {
                        const int oi = 3 * origin + ao_offset[a] + i;
                        for (int j = 0; j < nao[b]; ++j)
                        {
                            const int oj = 3 * neighbor + ao_offset[b] + j;
                            dm(oi, oj) = d(i, j);
                            for (int p = 0; p < fit.auxiliary_per_atom[a]; ++p)
                            {
                                const int ap = aux_per_cell * origin + aux_offset[a] + p;
                                product(oi, oj, ap) += c(p, i, j);
                                product(oj, oi, ap) += c(p, i, j);
                            }
                        }
                    }
                    for (int p = 0; p < fit.auxiliary_per_atom[a]; ++p)
                    {
                        const int ap = aux_per_cell * origin + aux_offset[a] + p;
                        for (int q = 0; q < fit.auxiliary_per_atom[b]; ++q)
                        {
                            const int aq = aux_per_cell * neighbor + aux_offset[b] + q;
                            metric(ap, aq) = v(p, q);
                        }
                    }
                }
            }
        }
    }
    LR::RIAtomPairMap<T> output;
    for (int a = 0; a < 2; ++a)
    {
        for (int b = 0; b < 2; ++b)
        {
            for (int r = 0; r < cells; ++r)
            {
                const LR::RIAtomCell pair{b, {{r, 0, 0}}};
                const std::size_t rows = nao[a];
                const std::size_t cols = nao[b];
                auto& h = output[a][pair];
                h = RI::Tensor<T>({rows, cols});
                for (int i = 0; i < rows; ++i)
                {
                    const int oi = ao_offset[a] + i;
                    for (int j = 0; j < cols; ++j)
                    {
                        const int oj = 3 * r + ao_offset[b] + j;
                        for (int k = 0; k < n; ++k)
                        {
                            for (int l = 0; l < n; ++l)
                            {
                                for (int p = 0; p < m; ++p)
                                {
                                    for (int q = 0; q < m; ++q)
                                    {
                                        const T left = hartree ? product(oi, oj, p) : product(oi, k, p);
                                        const T right = hartree ? product(k, l, q) : product(oj, l, q);
                                        h(i, j) += left * metric(p, q) * right * dm(k, l);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return output;
}

template<typename T> void compare(const LR::RIAtomPairMap<T>& local, const LR::RIAtomPairMap<T>& reference,
                                  const int cells)
{
    const std::set<int> atoms{0, 1};
    const auto all = RI::Communicate_Tensors_Map_Judge::comm_map2_first(MPI_COMM_WORLD, local, atoms, atoms);
    for (const auto& row : reference)
    {
        for (const auto& block : row.second)
        {
            LR::RIAtomCell centered = block.first;
            if (centered.second[0] > (cells - 1) / 2) { centered.second[0] -= cells; }
            const auto& actual = RI::Global_Func::find(all, row.first, centered);
            for (std::size_t index = 0; index < block.second.get_shape_all(); ++index)
            {
                const T value = actual.empty() ? T(0.0) : actual.ptr()[index];
                EXPECT_NEAR(std::abs(value - block.second.ptr()[index]), 0.0, 2e-10)
                    << "atoms " << row.first << "," << block.first.first
                    << " cell " << block.first.second[0] << " entry " << index;
            }
        }
    }
}

template<typename T> void exercise(const int cells)
{
    const auto geom = geometry(cells);
    const auto j_fit = fitting<T>(cells, true);
    const auto k_fit = fitting<T>(cells, false);
    auto local_j = j_fit;
    auto local_k = k_fit;
    local_j.Cs = distribute(j_fit.Cs);
    local_j.Vs = distribute(j_fit.Vs);
    local_k.Cs = distribute(k_fit.Cs);
    local_k.Vs = distribute(k_fit.Vs);
    LR::RIHartreeFock<T> kernel(MPI_COMM_WORLD, geom, local_j, local_k);
    LR::RIAtomPairMap<T> density;
    for (const auto& row : j_fit.Cs)
    {
        for (const auto& block : row.second)
        {
            const std::size_t rows = block.second.shape[1];
            const std::size_t cols = block.second.shape[2];
            auto& d = density[row.first][block.first];
            d = RI::Tensor<T>({rows, cols});
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    const int first_index = i + row.first;
                    const int second_index = j + block.first.first + block.first.second[0];
                    d(i, j) = amplitude<T>(first_index, second_index);
                }
            }
        }
    }
    LR::RIAtomPairMap<T> centered_density;
    for (const auto& row : density)
    {
        for (const auto& block : row.second)
        {
            LR::RIAtomCell centered = block.first;
            if (centered.second[0] > (cells - 1) / 2) { centered.second[0] -= cells; }
            centered_density[row.first][centered] = block.second;
        }
    }
    const auto d = distribute(centered_density);
    const auto j_ref = dense_reference(j_fit, density, cells, true);
    const auto k_ref = dense_reference(k_fit, density, cells, false);
    const auto j_result = kernel.apply_hartree(d);
    const auto k_result = kernel.apply_exchange(d);
    compare(j_result, j_ref, cells);
    compare(k_result, k_ref, cells);
    // Zero and a new vector must not inherit the previous density/potential.
    const LR::RIAtomPairMap<T> zero;
    const auto j_zero = kernel.apply_hartree(zero);
    const auto k_zero = kernel.apply_exchange(zero);
    auto zero_ref = j_ref;
    for (auto& row : zero_ref)
    {
        for (auto& block : row.second) { block.second = block.second * T(0.0); }
    }
    compare(j_zero, zero_ref, cells);
    compare(k_zero, zero_ref, cells);
    const auto j_again = kernel.apply_hartree(d);
    const auto k_again = kernel.apply_exchange(d);
    compare(j_again, j_ref, cells);
    compare(k_again, k_ref, cells);
    // Complex scaling detects conjugation of a response amplitude; tiny scaling detects screening.
    const T scale = amplitude<T>(2, 5) * 1e-8;
    auto scaled_d = d;
    for (auto* tensors : {&scaled_d})
    {
        for (auto& row : *tensors)
        {
            for (auto& block : row.second) { block.second = block.second * scale; }
        }
    }
    auto actual_j = kernel.apply_hartree(scaled_d);
    auto actual_k = kernel.apply_exchange(scaled_d);
    // Rescale to keep the accuracy check sensitive at tiny density magnitude.
    const T inverse_scale = T(1.0) / scale;
    for (auto* tensors : {&actual_j, &actual_k})
    {
        for (auto& row : *tensors)
        {
            for (auto& block : row.second) { block.second = block.second * inverse_scale; }
        }
    }
    compare(actual_j, j_ref, cells);
    compare(actual_k, k_ref, cells);
    // Tensor copies share storage in LibRI. Check that the kernel did not retain writable
    // aliases into the fitting tensors passed by its caller.
    for (auto* tensors : {&local_j.Cs, &local_j.Vs, &local_k.Cs, &local_k.Vs})
    {
        for (auto& row : *tensors)
        {
            for (auto& block : row.second) { block.second.ptr()[0] += T(0.7); }
        }
    }
    const auto isolated_j = kernel.apply_hartree(d);
    const auto isolated_k = kernel.apply_exchange(d);
    compare(isolated_j, j_ref, cells);
    compare(isolated_k, k_ref, cells);

    // A malformed block on one rank must produce a collective exception, not a deadlock.
    LR::RIAtomPairMap<T> malformed;
    const int rank = RI::MPI_Wrapper::mpi_get_rank(MPI_COMM_WORLD);
    if (rank == 0)
    {
        const LR::RIAtomCell pair{0, {{0, 0, 0}}};
        malformed[0][pair] = RI::Tensor<T>({1, 1});
    }
    EXPECT_THROW(kernel.apply_hartree(malformed), std::invalid_argument);
    EXPECT_THROW(kernel.apply_exchange(malformed), std::invalid_argument);
}
}

TEST(RIHartreeFock, RealGamma) { exercise<double>(1); }
TEST(RIHartreeFock, ComplexGamma) { exercise<std::complex<double>>(1); }
TEST(RIHartreeFock, ComplexThreeCells) { exercise<std::complex<double>>(3); }
