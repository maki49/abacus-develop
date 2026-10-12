#ifndef ABACUS_SOURCE_LCAO_MODULE_LR_PRECONDITIONER_H
#define ABACUS_SOURCE_LCAO_MODULE_LR_PRECONDITIONER_H

#include <vector>

class Parallel_2D;

namespace LR
{
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
