#ifndef ABACUS_SOURCE_LCAO_MODULE_LR_PRECONDITIONER_H
#define ABACUS_SOURCE_LCAO_MODULE_LR_PRECONDITIONER_H

#include <vector>

class Parallel_2D;

namespace LR
{
/// Borrow the KS spectrum and distributed (spin, k, occupied, virtual) basis.
/// Used for independent seed selection and global/local amplitude mapping;
/// no amplitudes or operator objects are owned.
struct OrbitalTransitionSpace
{
    const double* eigenvalues;
    const std::vector<Parallel_2D>& pair_distribution;
    int nk;
    const std::vector<int>& nocc;
    const std::vector<int>& nvirt;
    bool unrestricted;
};

/// Select the globally lowest orbital-gap transitions over every spin and k.
/// Complete the boundary degeneracy (1e-10 relative gap tolerance), so the
/// returned count can exceed requested. Each rank gets the same ordered list,
/// with local amplitude indices or -1 for transitions owned by another rank.
std::vector<int> select_low_energy_transitions(const OrbitalTransitionSpace& layout, int requested);

template<typename T>
struct DavidsonInitialSpace
{
    std::vector<T> vectors; ///< Local column-major amplitudes, including spare capacity.
    int count = 0;
};

/// Preserve the supplied Gamma guesses and append low-gap candidates; for
/// multiple k points choose independent spin/k seeds throughout. Add up to two
/// nominal guard vectors and complete the boundary gap degeneracy if capacity allows.
template<typename T>
DavidsonInitialSpace<T> make_davidson_initial_space(const T* guesses, int dim, int roots,
                                                   int capacity, const OrbitalTransitionSpace& layout);

/// Fill at least minimum_vectors lowest-gap seeds, extending a boundary gap
/// degeneracy up to capacity. Returns the actual column count; allocate capacity
/// columns. A reproducible dense component covers other symmetry sectors.
template<typename T>
int initialize_orbital_gap_candidates(T* vectors, int minimum_vectors, int capacity,
                                      const OrbitalTransitionSpace& layout);

/// Construct epsilon_a - epsilon_i in local [spin][k][occupied][virtual] order.
/// KS eigenvalues contain contiguous spin blocks, each with nk * (nocc[0]+nvirt[0]) entries.
std::vector<double> orbital_energy_gaps(const double* eigenvalues,
                                      const std::vector<Parallel_2D>& pair_distribution,
                                      int nk, const std::vector<int>& nocc,
                                      const std::vector<int>& nvirt, bool unrestricted);

/// Seed Davidson with distinct transitions having the lowest orbital gaps,
/// selected over all spins and k points. This avoids tying different uncoupled
/// transitions to one amplitude. A reproducible small dense perturbation avoids
/// exact invariant starting subspaces. Output is local contiguous [nvec][dim].
template<typename T>
void initialize_orbital_gap_vectors(T* vectors, int nvec, const double* eigenvalues,
                                   const std::vector<Parallel_2D>& pair_distribution,
                                   int nk, const std::vector<int>& nocc,
                                   const std::vector<int>& nvirt, bool unrestricted);

/// Apply the signed inverse gap to a CPU residual block, in place:
/// r_{j,n} /= sign(D_j-omega_n) * max(|D_j-omega_n|, denominator_floor).
/// The floor is in Ry; signed zero is regularized too. dim is both row count and stride.
template<typename T>
void apply_orbital_gap_preconditioner(T* residuals, const double* ritz_values,
                                     int dim, int nvec, const std::vector<double>& gaps,
                                     double denominator_floor);
}

#endif // ABACUS_SOURCE_LCAO_MODULE_LR_PRECONDITIONER_H
