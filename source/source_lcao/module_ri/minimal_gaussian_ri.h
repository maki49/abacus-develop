#ifndef MINIMAL_GAUSSIAN_RI_H
#define MINIMAL_GAUSSIAN_RI_H

#include "ri_hartree_fock.h"

class UnitCell;
class LCAO_Orbitals;

namespace LR
{
struct MinimalGaussianRISettings
{
    double theta = 0.6; // rid exponent alpha=theta/R_A^2; ris can be requested with theta=0.2.
    int hartree_lmax = 2;
    int exchange_lmax = 0;
    // This prototype uses Theta(rc-r)/r for J and for unscreened K.
    // Required positive radius in bohr: this is a finite-range periodic proxy, not Ewald Hartree.
    double coulomb_cutoff = 0.0;
    // Positive omega uses the standard erfc(omega*r)/r exchange kernel, without a sharp cutoff.
    double exchange_omega = 0.0; // bohr^-1; zero gives the same finite-range kernel as J.
    double kmesh_multiplier = 4.0;
    double metric_relative_cutoff = 1e-8; // Remove near-dependent local metric eigenvectors.
};

template<typename T> struct MinimalGaussianRIFits
{
    RIGeometry geometry;
    RIFittingTensors<T> hartree;
    RIFittingTensors<T> exchange;
};

// Refit AO products in the two-center union of minimal Gaussian auxiliary shells,
// using the existing numerical Coulomb and three-center integrals. Cs and Vs are new
// distributed tensors; neither fitted KS auxiliary orbitals nor KS Exx_LRI are modified.
// atomic_numbers is per species, independent of valence charges or decorated atom labels.
// AO radial grids must be uniform and start at zero; Gaussian grids are matched to them.
template<typename T> MinimalGaussianRIFits<T> build_minimal_gaussian_ri(
    const MPI_Comm& comm, const UnitCell& cell, const LCAO_Orbitals& orbitals,
    const std::vector<int>& atomic_numbers, const RICell& period,
    const MinimalGaussianRISettings& settings);

// Convenience factory for an independently owned D -> J/K operator. Building it is
// geometry work (once per ionic step), whereas apply_hartree/exchange are electronic work.
template<typename T> std::unique_ptr<RIHartreeFock<T>> make_minimal_gaussian_ri(
    const MPI_Comm& comm, const UnitCell& cell, const LCAO_Orbitals& orbitals,
    const std::vector<int>& atomic_numbers, const RICell& period,
    const MinimalGaussianRISettings& settings);
}
#endif
