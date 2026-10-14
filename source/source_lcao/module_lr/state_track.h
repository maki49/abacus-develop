#ifndef ABACUS_LR_STATE_TRACK_H
#define ABACUS_LR_STATE_TRACK_H
#include "state_projection.h"
#include <ostream>
namespace LR
{
template <typename T>
struct RootBasis
{
    OrbitalBasisSnapshot<T> orbitals;
    std::vector<std::vector<T>> group_amplitudes;
};

template <typename T>
struct RootInputs
{
    TransitionBasisView<T> basis;
    bool reference_first;
};

// Form old-new AO/MO overlaps and project the old reference into the current basis,
// then match energy groups by total overlap (average) or actual mixed-state projection (JT).
// Store every selected-group root in basis; previous remains the single/JT reference. Update target;
// current amplitudes stay unchanged. Requires Gamma, fixed cell/window and local layout.
template <typename T>
void follow_cross_root(const RootInputs<T>& inputs, const T* amplitudes,
    int nlocal, int nstates, int initial_state, const std::vector<double>& energies,
    double group_threshold, int& target,
    std::vector<T>& previous, RootBasis<T>& basis, std::ostream& log);
}
#endif
