#ifndef ABACUS_LR_STATE_GUESS_H
#define ABACUS_LR_STATE_GUESS_H

#include "state_projection.h"

namespace LR
{
// Own every converged root, grouped by response channel, and the MO basis in which
// its amplitudes were calculated. This cache is independent of target-state tracking.
template <typename T>
struct ExcitationGuess
{
    OrbitalBasisSnapshot<T> orbitals;
    std::vector<std::vector<std::vector<T>>> amplitudes;
};

template <typename T>
ExcitationGuess<T> capture_excitation_guess(const TransitionBasisView<T>& basis,
    const std::vector<const T*>& amplitudes, int local_dimension, int roots);

// Replace cold guesses by independent projected roots, then fill missing directions
// from the cold guesses. Reject projection weights below 1/2; use global two-pass
// Gram-Schmidt. If a full-rank starting block cannot be formed, leave guesses intact.
// Return the number of reused roots. All ranks in the Gamma calculation participate.
template <typename T>
int select_projected_guesses(const std::vector<std::vector<T>>& projected,
    T* guesses, int local_dimension, int roots);

// Transport X with O_vv^dagger X O_oo into the current geometry, then select guesses.
// On a changed cell/window/layout or absent snapshot, keep the cold guesses.
// Return reused root counts per channel; the full LR operator is never approximated.
template <typename T>
std::vector<int> project_excitation_guesses(const TransitionBasisView<T>& basis,
    const ExcitationGuess<T>& previous, const std::vector<T*>& guesses,
    int local_dimension, int roots);
}
#endif
