#include "../lr_ri_response.h"
#include "source_lcao/module_ri/ri_hartree_fock.h"
#include "source_hsolver/diag_comm_info.h"
#include <RI/global/MPI_Wrapper.h>
#include <gtest/gtest.h>
#include <complex>

namespace
{
double coefficient(int index, double*) { return 0.13 * (index + 1); }
std::complex<double> coefficient(int index, std::complex<double>*)
{ return {0.13 * (index + 1), 0.07 * (index % 3 - 1)}; }
double conjugate(double x) { return x; }
std::complex<double> conjugate(std::complex<double> x) { return std::conj(x); }

template<typename T> void check_response(LR::RIResponseSpin channel, int nk)
{
    const int rank = RI::MPI_Wrapper::mpi_get_rank(MPI_COMM_WORLD);
    const int ranks = RI::MPI_Wrapper::mpi_get_size(MPI_COMM_WORLD);
    const hsolver::diag_comm_info comm(MPI_COMM_WORLD, rank, ranks);
    const int spins = channel == LR::RIResponseSpin::unrestricted ? 2 : 1;
    LR::RIGeometry geometry;
    geometry.positions[0] = {{0, 0, 0}};
    geometry.lattice = {{{{12, 0, 0}}, {{0, 12, 0}}, {{0, 0, 12}}}};
    geometry.period = {{nk, 1, 1}};
    geometry.orbitals_per_atom = {3};
    LR::RIFittingTensors<T> fit;
    fit.auxiliary_per_atom = {1};
    // With one center the fitted pair coefficient is 2 C_ij (local-fit half).
    const double c[] = {0.5, 0.2, 0.1, 0.2, 0.7, 0.3, 0.1, 0.3, 0.6};
    if (rank == 0)
    {
        RI::Tensor<T> cs({1, 3, 3});
        RI::Tensor<T> vs({1, 1});
        for (int i = 0; i < 9; ++i) { cs.ptr()[i] = c[i]; }
        vs(0, 0) = 0.8;
        fit.Cs[0][{0, {{0, 0, 0}}}] = cs;
        fit.Vs[0][{0, {{0, 0, 0}}}] = vs;
    }
    std::unique_ptr<LR::RIHartreeFock<T>> kernel(new LR::RIHartreeFock<T>(MPI_COMM_WORLD, geometry, fit, fit));
    LR::RITransitionBasis<T> basis;
    basis.naos = 3;
    basis.nk = nk;
    basis.nocc = {1};
    basis.nvirt = {2};
    if (spins == 2) { basis.nocc.push_back(2); basis.nvirt.push_back(1); }
    for (int k = 0; k < nk; ++k) { basis.kpoints.push_back({{double(k) / nk, 0, 0}}); }
    basis.coefficients.resize(spins);
    for (int spin = 0; spin < spins; ++spin)
    {
        auto& coeff = basis.coefficients[spin];
        for (int i = 0; i < nk * 9; ++i)
        {
            const int index = i % 9 + spin + i / 9;
            coeff.push_back(coefficient(index, static_cast<T*>(nullptr)));
        }
    }
    const int global_dim = spins * nk * 2;
    std::vector<T> global_x(global_dim);
    std::vector<T> x;
    std::vector<double> gaps;
    for (int i = 0; i < global_dim; ++i)
    {
        const int index = i + 2;
        global_x[i] = coefficient(index, static_cast<T*>(nullptr));
        if (i % ranks == rank)
        {
            basis.local_to_global.push_back(i);
            x.push_back(global_x[i]);
            gaps.push_back(1 + 0.1 * i);
        }
    }
    LR::RIResponseOperator<T> response(geometry, basis, std::move(kernel), gaps, 0.25, comm);
    std::vector<T> actual(x.size());
    response.apply(x.data(), actual.data(), channel);
    // Independent dense four-index ERIs and explicit MO contractions. Here all
    // fitted Cs/Vs are on-site, so only D(R=0) contributes, including at multiple k.
    std::vector<std::vector<T>> densities(spins, std::vector<T>(9, T(0)));
    const bool real = std::is_same<T, double>::value;
    for (int spin = 0; spin < spins; ++spin)
    {
        const int no = basis.nocc[spin];
        const int nv = basis.nvirt[spin];
        for (int k = 0; k < nk; ++k)
        {
            const T* coeff = basis.coefficients[spin].data() + k * 9;
            for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a)
            for (int mu = 0; mu < 3; ++mu)
            for (int nu = 0; nu < 3; ++nu)
            {
                const T transition = real ? coeff[i * 3 + mu] * coeff[(no + a) * 3 + nu]
                    : coeff[(no + a) * 3 + mu] * conjugate(coeff[i * 3 + nu]);
                densities[spin][mu * 3 + nu] += transition * global_x[(spin * nk + k) * 2 + i * nv + a] / double(nk);
            }
        }
    }
    std::vector<T> expected(global_dim, T(0));
    for (int spin = 0; spin < spins; ++spin)
    {
        std::vector<T> potential(9, T(0));
        for (int mu = 0; mu < 3; ++mu)
        for (int nu = 0; nu < 3; ++nu)
        for (int rho = 0; rho < 3; ++rho)
        for (int sigma = 0; sigma < 3; ++sigma)
        {
            const double exchange_eri = 3.2 * c[mu * 3 + rho] * c[nu * 3 + sigma];
            potential[mu * 3 + nu] -= 0.5 * exchange_eri * densities[spin][rho * 3 + sigma];
            if (channel == LR::RIResponseSpin::triplet) { continue; }
            const double hartree_eri = 3.2 * c[mu * 3 + nu] * c[rho * 3 + sigma];
            const double factor = channel == LR::RIResponseSpin::singlet ? 4 : 2;
            for (int other = 0; other < spins; ++other)
            { potential[mu * 3 + nu] += factor * hartree_eri * densities[other][rho * 3 + sigma]; }
        }
        const int no = basis.nocc[spin];
        const int nv = basis.nvirt[spin];
        for (int k = 0; k < nk; ++k)
        {
            const T* coeff = basis.coefficients[spin].data() + k * 9;
            for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a)
            for (int mu = 0; mu < 3; ++mu)
            for (int nu = 0; nu < 3; ++nu)
            {
                const T projection = real ? coeff[i * 3 + mu] * coeff[(no + a) * 3 + nu]
                    : conjugate(coeff[(no + a) * 3 + mu]) * coeff[i * 3 + nu];
                expected[(spin * nk + k) * 2 + i * nv + a] += projection * potential[mu * 3 + nu];
            }
        }
    }
    for (std::size_t j = 0; j < x.size(); ++j)
    {
        const T answer = expected[basis.local_to_global[j]] + gaps[j] * x[j];
        EXPECT_NEAR(std::abs(actual[j] - answer), 0, 1e-11);
    }
    std::fill(x.begin(), x.end(), T(0));
    response.apply(x.data(), actual.data(), channel);
    for (T value : actual) { EXPECT_NEAR(std::abs(value), 0, 1e-13); }
}
}

TEST(LRRIResponse, RealSpinFactors)
{
    for (auto spin : {LR::RIResponseSpin::singlet, LR::RIResponseSpin::triplet, LR::RIResponseSpin::unrestricted})
    { check_response<double>(spin, 1); }
}
TEST(LRRIResponse, ComplexSpinFactors)
{
    for (auto spin : {LR::RIResponseSpin::singlet, LR::RIResponseSpin::triplet, LR::RIResponseSpin::unrestricted})
    { check_response<std::complex<double>>(spin, 1); }
}
TEST(LRRIResponse, ComplexKPointNormalization)
{ check_response<std::complex<double>>(LR::RIResponseSpin::unrestricted, 2); }

TEST(LRRIResponse, NonzeroCellBlochPhases)
{
    using T = std::complex<double>;
    const int rank = RI::MPI_Wrapper::mpi_get_rank(MPI_COMM_WORLD);
    const int ranks = RI::MPI_Wrapper::mpi_get_size(MPI_COMM_WORLD);
    const hsolver::diag_comm_info comm(MPI_COMM_WORLD, rank, ranks);
    LR::RIGeometry geometry;
    geometry.positions[0] = {{0, 0, 0}};
    geometry.lattice = {{{{12, 0, 0}}, {{0, 12, 0}}, {{0, 0, 12}}}};
    geometry.period = {{3, 1, 1}};
    geometry.orbitals_per_atom = {2};
    LR::RIFittingTensors<T> fit;
    fit.auxiliary_per_atom = {1};
    if (rank == 0)
    {
        RI::Tensor<T> cs({1, 2, 2});
        cs(0, 0, 0) = 0.5;
        cs(0, 0, 1) = 0.2;
        cs(0, 1, 0) = 0.2;
        cs(0, 1, 1) = 0.7;
        fit.Cs[0][{0, {{0, 0, 0}}}] = cs;
        for (int r = -1; r <= 1; ++r)
        {
            RI::Tensor<T> vs({1, 1});
            vs(0, 0) = r == 0 ? 0.8 : 0.4;
            fit.Vs[0][{0, {{r, 0, 0}}}] = vs;
        }
    }
    std::unique_ptr<LR::RIHartreeFock<T>> kernel(new LR::RIHartreeFock<T>(MPI_COMM_WORLD, geometry, fit, fit));
    LR::RITransitionBasis<T> basis;
    basis.naos = 2;
    basis.nk = 3;
    basis.nocc = {1};
    basis.nvirt = {1};
    basis.coefficients = {{1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1}};
    const std::vector<T> full = {{1, 1}, {2, -1}, {-1, 0.3}};
    std::vector<T> input;
    std::vector<double> gaps;
    for (int k = 0; k < 3; ++k)
    {
        basis.kpoints.push_back({{double(k) / 3, 0, 0}});
        if (k % ranks == rank)
        {
            basis.local_to_global.push_back(k);
            input.push_back(full[k]);
            gaps.push_back(1);
        }
    }
    LR::RIResponseOperator<T> response(geometry, basis, std::move(kernel), gaps, 0.25, comm);
    std::vector<T> image(input.size());
    response.apply(input.data(), image.data(), LR::RIResponseSpin::singlet);
    const T average = (full[0] + full[1] + full[2]) / 3.0;
    for (std::size_t j = 0; j < input.size(); ++j)
    {
        const int k = basis.local_to_global[j];
        T exchange = 0;
        for (int q = 0; q < 3; ++q)
        {
            // Independent reciprocal-space ERI: V(k-q)=0.8+0.8 cos(2pi(k-q)/3).
            const double angle = 2 * std::acos(-1.0) * (k - q) / 3;
            exchange += (0.8 + 0.8 * std::cos(angle)) * full[q] / 3.0;
        }
        const T expected = input[j] + 16 * 0.2 * 0.2 * 1.6 * average - 2 * 0.5 * 0.7 * exchange;
        EXPECT_NEAR(std::abs(image[j] - expected), 0, 1e-12);
    }
}
