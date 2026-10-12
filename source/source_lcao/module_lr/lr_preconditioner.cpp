#include "lr_preconditioner.h"

#include "operator_casida/operator_lr_diag.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>

namespace LR
{
std::vector<double> orbital_energy_gaps(const double* eigenvalues,
                                      const std::vector<Parallel_2D>& pair_distribution,
                                      int nk, const std::vector<int>& nocc,
                                      const std::vector<int>& nvirt, bool unrestricted)
{
    const int spin_count = unrestricted ? 2 : 1;
    if (nk <= 0 || nocc.size() < spin_count || nvirt.size() < spin_count
        || pair_distribution.size() < spin_count)
    {
        throw std::invalid_argument("Invalid LR orbital-gap layout");
    }
    const int ks_bands = nocc[0] + nvirt[0];
    std::vector<double> result;
    for (int spin = 0; spin < spin_count; ++spin)
    {
        const double* spin_eigenvalues = eigenvalues + spin * nk * ks_bands;
        const OperatorLRDiag<double> diagonal(spin_eigenvalues, pair_distribution[spin], nk,
                                               nocc[spin], nvirt[spin], false);
        const auto& gaps = diagonal.energy_differences();
        const std::size_t count = static_cast<std::size_t>(gaps.nr) * gaps.nc;
        // Append a whole spin block, including every k point. No in-place
        // multiplication or manually maintained spin offset is needed.
        if (count > 0) { result.insert(result.end(), gaps.c, gaps.c + count); }
    }
    return result;
}

template<typename T>
void initialize_orbital_gap_vectors(T* vectors, int nvec, const double* eigenvalues,
                                   const std::vector<Parallel_2D>& pair_distribution,
                                   int nk, const std::vector<int>& nocc,
                                   const std::vector<int>& nvirt, bool unrestricted)
{
    struct Transition
    {
        double gap;
        int spin;
        int k;
        int occupied;
        int virtual_orbital;
    };
    const int spin_count = unrestricted ? 2 : 1;
    if (nk <= 0 || nocc.size() < spin_count || nvirt.size() < spin_count
        || pair_distribution.size() < spin_count)
    {
        throw std::invalid_argument("Invalid LR orbital-gap initial-vector layout");
    }
    const int ks_bands = nocc[0] + nvirt[0];
    int local_dim = 0;
    std::vector<Transition> transitions;
    for (int spin = 0; spin < spin_count; ++spin)
    {
        local_dim += nk * pair_distribution[spin].get_local_size();
        for (int k = 0; k < nk; ++k)
        {
            const double* energies = eigenvalues + (spin * nk + k) * ks_bands;
            for (int i = 0; i < nocc[spin]; ++i)
            {
                for (int a = 0; a < nvirt[spin]; ++a)
                {
                    transitions.push_back({energies[nocc[spin] + a] - energies[i], spin, k, i, a});
                }
            }
        }
    }
    if (nvec < 0 || static_cast<std::size_t>(nvec) > transitions.size())
    {
        throw std::invalid_argument("Too many LR orbital-gap initial vectors");
    }
    // Stable ordering gives degenerate transitions distinct, reproducible seeds.
    std::stable_sort(transitions.begin(), transitions.end(),
                     [](const Transition& left, const Transition& right) { return left.gap < right.gap; });
    const std::size_t size = static_cast<std::size_t>(nvec) * local_dim;
    if (size > 0) { std::fill(vectors, vectors + size, T(0)); }
    const auto local_index = [&](const Transition& transition) -> int
    {
        const auto& pairs = pair_distribution[transition.spin];
        if (!pairs.in_this_processor(transition.virtual_orbital, transition.occupied)) { return -1; }
        const int spin_offset = transition.spin == 0 ? 0 : nk * pair_distribution[0].get_local_size();
        const int pair_offset = pairs.global2local_col(transition.occupied) * pairs.get_row_size()
                                + pairs.global2local_row(transition.virtual_orbital);
        return spin_offset + transition.k * pairs.get_local_size() + pair_offset;
    };
    for (int n = 0; n < nvec; ++n)
    {
        // The current Davidson implementation expands every initial Ritz root,
        // even an exact eigenvector. A small dense component both avoids zero
        // initial residuals and gives other symmetry sectors nonzero overlap.
        // Draw for all global transitions so the seed is MPI-layout independent.
        std::mt19937 generator(n);
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        const double perturbation = 1e-3 / std::sqrt(static_cast<double>(transitions.size()));
        for (const Transition& candidate : transitions)
        {
            const double noise = perturbation * uniform(generator);
            const int index = local_index(candidate);
            if (index < 0) { continue; }
            const std::size_t offset = static_cast<std::size_t>(n) * local_dim + index;
            vectors[offset] = T(noise);
        }
        const Transition& transition = transitions[n];
        const int index = local_index(transition);
        if (index < 0) { continue; }
        const std::size_t offset = static_cast<std::size_t>(n) * local_dim + index;
        vectors[offset] += T(1);
    }
}

template<typename T>
void apply_orbital_gap_preconditioner(T* residuals, const double* ritz_values,
                                     int dim, int nvec, const std::vector<double>& gaps,
                                     double denominator_floor)
{
    if (dim < 0 || nvec < 0 || gaps.size() != static_cast<std::size_t>(dim)
        || !std::isfinite(denominator_floor) || denominator_floor <= 0.0)
    {
        throw std::invalid_argument("Invalid LR orbital-gap preconditioner arguments");
    }
    for (int n = 0; n < nvec; ++n)
    {
        for (int j = 0; j < dim; ++j)
        {
            const double difference = gaps[j] - ritz_values[n];
            if (!std::isfinite(difference))
            {
                throw std::invalid_argument("LR orbital gaps and Ritz values must be finite");
            }
            const double magnitude = std::abs(difference);
            const double bounded_magnitude = std::max(magnitude, denominator_floor);
            const double denominator = std::copysign(bounded_magnitude, difference);
            const std::size_t index = static_cast<std::size_t>(n) * dim + j;
            residuals[index] /= denominator;
        }
    }
}

template void apply_orbital_gap_preconditioner<double>(double*, const double*, int, int,
                                                       const std::vector<double>&, double);
template void apply_orbital_gap_preconditioner<std::complex<double>>(
    std::complex<double>*, const double*, int, int, const std::vector<double>&, double);
template void initialize_orbital_gap_vectors<double>(double*, int, const double*,
    const std::vector<Parallel_2D>&, int, const std::vector<int>&, const std::vector<int>&, bool);
template void initialize_orbital_gap_vectors<std::complex<double>>(std::complex<double>*, int, const double*,
    const std::vector<Parallel_2D>&, int, const std::vector<int>&, const std::vector<int>&, bool);
}
