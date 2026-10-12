#include "lr_preconditioner.h"

#include "operator_casida/operator_lr_diag.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>

namespace LR
{
namespace
{
struct Transition
{
    double gap;
    int local_index; // -1 when this global transition belongs to another rank.
};

std::vector<Transition> ordered_transitions(const OrbitalTransitionSpace& layout)
{
    const double* eigenvalues = layout.eigenvalues;
    const auto& pair_distribution = layout.pair_distribution;
    const int nk = layout.nk;
    const auto& nocc = layout.nocc;
    const auto& nvirt = layout.nvirt;
    const bool unrestricted = layout.unrestricted;
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
        const int spin_offset = local_dim;
        const auto& pairs = pair_distribution[spin];
        local_dim += nk * pairs.get_local_size();
        for (int k = 0; k < nk; ++k)
        {
            const double* energies = eigenvalues + (spin * nk + k) * ks_bands;
            for (int i = 0; i < nocc[spin]; ++i)
            {
                for (int a = 0; a < nvirt[spin]; ++a)
                {
                    int local_index = -1;
                    if (pairs.in_this_processor(a, i))
                    {
                        local_index = spin_offset + k * pairs.get_local_size()
                                      + pairs.global2local_col(i) * pairs.get_row_size() + pairs.global2local_row(a);
                    }
                    const double gap = energies[nocc[spin] + a] - energies[i];
                    if (!std::isfinite(gap)) { throw std::invalid_argument("Nonfinite LR orbital gap"); }
                    transitions.push_back({gap, local_index});
                }
            }
        }
    }
    // Stable ordering gives degenerate transitions distinct, reproducible seeds.
    std::stable_sort(transitions.begin(), transitions.end(),
                     [](const Transition& left, const Transition& right) { return left.gap < right.gap; });
    return transitions;
}

int complete_gap_degeneracy(const std::vector<Transition>& transitions, int minimum, int capacity)
{
    int count = minimum;
    const int limit = std::min<std::size_t>(capacity, transitions.size());
    if (count > 0)
    {
        const double boundary_gap = transitions[count - 1].gap;
        const double absolute_gap = std::abs(boundary_gap);
        const double gap_scale = std::max(1.0, absolute_gap);
        const double tolerance = 1e-10 * gap_scale;
        while (count < limit)
        {
            const double difference = transitions[count].gap - boundary_gap;
            if (std::abs(difference) > tolerance) { break; }
            ++count;
        }
    }
    return count;
}
} // namespace

std::vector<int> select_low_energy_transitions(const OrbitalTransitionSpace& layout, int requested)
{
    if (requested < 0) { throw std::invalid_argument("Negative LR low-energy block size"); }
    const auto transitions = ordered_transitions(layout);
    const int total = transitions.size();
    const int minimum = std::min(requested, total);
    const int count = complete_gap_degeneracy(transitions, minimum, total);
    std::vector<int> indices(count);
    for (int j = 0; j < count; ++j) { indices[j] = transitions[j].local_index; }
    return indices;
}


template<typename T>
DavidsonInitialSpace<T> make_davidson_initial_space(const T* guesses, int dim, int roots,
                                                   int capacity, const OrbitalTransitionSpace& layout)
{
    const int spin_count = layout.unrestricted ? 2 : 1;
    if (dim < 0 || roots <= 0 || capacity < roots || layout.nk <= 0
        || layout.pair_distribution.size() < spin_count)
    {
        throw std::invalid_argument("Invalid LR Davidson initial-space dimensions");
    }
    int expected_dim = 0;
    for (int spin = 0; spin < spin_count; ++spin)
    {
        expected_dim += layout.nk * layout.pair_distribution[spin].get_local_size();
    }
    if (dim != expected_dim) { throw std::invalid_argument("LR initial-space dimension does not match its transition layout"); }
    DavidsonInitialSpace<T> result;
    const std::size_t size = static_cast<std::size_t>(dim) * capacity;
    const std::size_t allocated = std::max<std::size_t>(1, size);
    result.vectors.resize(allocated);
    const int extra_roots = std::min(2, roots);
    const int requested_columns = roots + extra_roots;
    const int initial_target = std::min(requested_columns, capacity);
    if (layout.nk > 1)
    {
        // Independent spin/k transitions cover low-gap partners; extend the
        // boundary degeneracy while preserving room for subsequent corrections.
        result.count = initialize_orbital_gap_candidates(result.vectors.data(), initial_target, capacity, layout);
    }
    else
    {
        // Retain efficient Gamma guesses, but add independent low-gap partners
        // rather than tying both spins together in every starting direction.
        const int requested_size = dim * roots;
        if (requested_size > 0) { std::copy_n(guesses, requested_size, result.vectors.data()); }
        T* extra_vectors = result.vectors.data() + requested_size;
        const int extra_target = initial_target - roots;
        const int extra_capacity = capacity - roots;
        const int added = initialize_orbital_gap_candidates(extra_vectors, extra_target, extra_capacity, layout);
        result.count = roots + added;
    }
    return result;
}

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
    const OrbitalTransitionSpace layout{eigenvalues, pair_distribution, nk, nocc, nvirt, unrestricted};
    initialize_orbital_gap_candidates(vectors, nvec, nvec, layout);
}

template<typename T>
int initialize_orbital_gap_candidates(T* vectors, int minimum_vectors, int capacity,
                                      const OrbitalTransitionSpace& layout)
{
    const auto transitions = ordered_transitions(layout);
    if (minimum_vectors < 0 || capacity < minimum_vectors
        || static_cast<std::size_t>(minimum_vectors) > transitions.size())
    {
        throw std::invalid_argument("Too many LR orbital-gap initial vectors");
    }
    const int nvec = complete_gap_degeneracy(transitions, minimum_vectors, capacity);
    const int spin_count = layout.unrestricted ? 2 : 1;
    int local_dim = 0;
    for (int spin = 0; spin < spin_count; ++spin)
    {
        local_dim += layout.nk * layout.pair_distribution[spin].get_local_size();
    }
    const std::size_t size = static_cast<std::size_t>(nvec) * local_dim;
    if (size > 0) { std::fill(vectors, vectors + size, T(0)); }
    for (int n = 0; n < nvec; ++n)
    {
        // The residual-controlled solver accepts exact initial eigenpairs.
        // This small dense component is only for overlap with other symmetry
        // sectors; it is no longer needed to avoid zero-residual normalization.
        // Draw for all global transitions so the seed is MPI-layout independent.
        std::mt19937 generator(n);
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        const double perturbation = 1e-3 / std::sqrt(static_cast<double>(transitions.size()));
        for (const Transition& candidate : transitions)
        {
            const double noise = perturbation * uniform(generator);
            const int index = candidate.local_index;
            if (index < 0) { continue; }
            const std::size_t offset = static_cast<std::size_t>(n) * local_dim + index;
            vectors[offset] = T(noise);
        }
        const Transition& transition = transitions[n];
        const int index = transition.local_index;
        if (index < 0) { continue; }
        const std::size_t offset = static_cast<std::size_t>(n) * local_dim + index;
        vectors[offset] += T(1);
    }
    return nvec;
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
template int initialize_orbital_gap_candidates<double>(double*, int, int, const OrbitalTransitionSpace&);
template int initialize_orbital_gap_candidates<std::complex<double>>(
    std::complex<double>*, int, int, const OrbitalTransitionSpace&);
template DavidsonInitialSpace<double> make_davidson_initial_space(
    const double*, int, int, int, const OrbitalTransitionSpace&);
template DavidsonInitialSpace<std::complex<double>> make_davidson_initial_space(
    const std::complex<double>*, int, int, int, const OrbitalTransitionSpace&);
template void apply_orbital_gap_preconditioner<std::complex<double>>(
    std::complex<double>*, const double*, int, int, const std::vector<double>&, double);
template void initialize_orbital_gap_vectors<double>(double*, int, const double*,
    const std::vector<Parallel_2D>&, int, const std::vector<int>&, const std::vector<int>&, bool);
template void initialize_orbital_gap_vectors<std::complex<double>>(std::complex<double>*, int, const double*,
    const std::vector<Parallel_2D>&, int, const std::vector<int>&, const std::vector<int>&, bool);
}
