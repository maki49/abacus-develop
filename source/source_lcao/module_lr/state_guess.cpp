#include "state_guess.h"
#include "state_ovlp.h"
#include "source_base/parallel_reduce.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <utility>

namespace LR
{
namespace
{
template <typename T> T conjugate(T value) { return value; }
template <> std::complex<double> conjugate(std::complex<double> value) { return std::conj(value); }

template <typename T>
T global_inner_product(const std::vector<T>& left, const std::vector<T>& right)
{
    T dot = T(0);
    for (std::size_t row = 0; row < left.size(); ++row) { dot += conjugate(left[row]) * right[row]; }
    Parallel_Reduce::reduce_all(&dot, 1);
    return dot;
}

template <typename T>
bool append_independent(std::vector<T> direction, std::vector<std::vector<T>>& accepted,
                        const double minimum_norm2)
{
    const T original_dot = global_inner_product(direction, direction);
    const double original_norm2 = std::real(original_dot);
    if (!std::isfinite(original_norm2) || original_norm2 <= minimum_norm2) { return false; }
    for (int pass = 0; pass < 2; ++pass)
    {
        for (const auto& basis : accepted)
        {
            const T overlap = global_inner_product(basis, direction);
            for (std::size_t row = 0; row < direction.size(); ++row)
            { direction[row] -= overlap * basis[row]; }
        }
    }
    const T remaining_dot = global_inner_product(direction, direction);
    const double remaining_norm2 = std::real(remaining_dot);
    if (!std::isfinite(remaining_norm2) || remaining_norm2 <= 1e-12 * original_norm2) { return false; }
    const double inverse_norm = 1.0 / std::sqrt(remaining_norm2);
    for (auto& value : direction) { value *= inverse_norm; }
    accepted.push_back(std::move(direction));
    return true;
}
}

template <typename T>
ExcitationGuess<T> capture_excitation_guess(const TransitionBasisView<T>& basis,
    const std::vector<const T*>& amplitudes, int local_dimension, int roots)
{
    ExcitationGuess<T> snapshot;
    snapshot.orbitals = capture_orbital_basis(basis);
    snapshot.amplitudes.resize(amplitudes.size());
    for (std::size_t channel = 0; channel < amplitudes.size(); ++channel)
    {
        for (int root = 0; root < roots; ++root)
        {
            std::vector<T> vector(local_dimension);
            if (local_dimension > 0)
            {
                const T* source = amplitudes[channel] + root * local_dimension;
                std::copy_n(source, local_dimension, vector.begin());
            }
            snapshot.amplitudes[channel].push_back(std::move(vector));
        }
    }
    return snapshot;
}

template <typename T>
int select_projected_guesses(const std::vector<std::vector<T>>& projected,
    T* guesses, int local_dimension, int roots)
{
    int invalid = local_dimension < 0 || roots <= 0 || projected.size() != static_cast<std::size_t>(roots);
    for (const auto& vector : projected)
    { invalid += vector.size() != static_cast<std::size_t>(local_dimension); }
    Parallel_Reduce::reduce_all(invalid);
    if (invalid) { return 0; }
    std::vector<std::vector<T>> selected;
    const double minimum_projection_weight = 0.5;
    for (const auto& vector : projected) { append_independent(vector, selected, minimum_projection_weight); }
    const int reused = selected.size();
    if (reused == 0) { return 0; }
    const double minimum_cold_norm2 = 1e-24;
    for (int root = 0; root < roots && selected.size() < static_cast<std::size_t>(roots); ++root)
    {
        std::vector<T> cold(local_dimension);
        if (local_dimension > 0)
        {
            const T* source = guesses + root * local_dimension;
            std::copy_n(source, local_dimension, cold.begin());
        }
        append_independent(std::move(cold), selected, minimum_cold_norm2);
    }
    if (selected.size() != static_cast<std::size_t>(roots)) { return 0; }
    for (int root = 0; root < roots; ++root)
    {
        if (local_dimension > 0)
        {
            T* destination = guesses + root * local_dimension;
            std::copy(selected[root].begin(), selected[root].end(), destination);
        }
    }
    return reused;
}

template <typename T>
std::vector<int> project_excitation_guesses(const TransitionBasisView<T>& basis,
    const ExcitationGuess<T>& previous, const std::vector<T*>& guesses,
    int local_dimension, int roots)
{
    std::vector<int> reused(guesses.size(), 0);
    int invalid = previous.orbitals.positions.empty() || previous.amplitudes.size() != guesses.size();
    for (const auto& channel : previous.amplitudes)
    {
        invalid += channel.size() != static_cast<std::size_t>(roots);
        for (const auto& vector : channel)
        { invalid += vector.size() != static_cast<std::size_t>(local_dimension); }
    }
    Parallel_Reduce::reduce_all(invalid);
    if (invalid) { return reused; }
    const auto current = capture_orbital_basis(basis);
    invalid = !compatible_orbital_bases(previous.orbitals, current);
    Parallel_Reduce::reduce_all(invalid);
    if (invalid) { return reused; }
    const auto ao = cross_ao_overlap(basis.cell, previous.orbitals.positions,
        basis.cutoff, basis.integrator, basis.pmat);
    for (std::size_t channel = 0; channel < guesses.size(); ++channel)
    {
        auto projected = previous.amplitudes[channel];
        std::vector<std::vector<T>*> vectors;
        for (auto& root : projected) { vectors.push_back(&root); }
        auto channel_basis = basis;
        channel_basis.channel = channel;
        project_transition_vectors(channel_basis, previous.orbitals, current, ao, vectors);
        reused[channel] = select_projected_guesses(projected, guesses[channel], local_dimension, roots);
    }
    return reused;
}

template ExcitationGuess<double> capture_excitation_guess(const TransitionBasisView<double>&,
    const std::vector<const double*>&, int, int);
template ExcitationGuess<std::complex<double>> capture_excitation_guess(const TransitionBasisView<std::complex<double>>&,
    const std::vector<const std::complex<double>*>&, int, int);
template int select_projected_guesses(const std::vector<std::vector<double>>&, double*, int, int);
template int select_projected_guesses(const std::vector<std::vector<std::complex<double>>>&, std::complex<double>*, int, int);
template std::vector<int> project_excitation_guesses(const TransitionBasisView<double>&,
    const ExcitationGuess<double>&, const std::vector<double*>&, int, int);
template std::vector<int> project_excitation_guesses(const TransitionBasisView<std::complex<double>>&,
    const ExcitationGuess<std::complex<double>>&, const std::vector<std::complex<double>*>&, int, int);
}
