#ifndef MINIMAL_GAUSSIAN_BASIS_H
#define MINIMAL_GAUSSIAN_BASIS_H

#include "source_basis/module_ao/orb_atomic_lm.h"
#include <vector>

namespace LR
{
using NumericalAuxiliaryBasis = std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>;

struct MinimalGaussianBasisSettings
{
    double theta = 0.6;
    int lmax = 2; // Complete s/p/d shells on non-H atoms; H always has one s function.
    double radial_step = 0.01; // bohr
    std::vector<double> species_radial_steps; // Optional overrides, matching the AO grids used in products.
    double tail_exponent = 36.0; // alpha * rcut^2; truncation is independent of the AO cutoff.
    int nk = 1001;
    double dk = 0.01; // bohr^-1
    double dr_uniform = 0.001;
};

// alpha_A = theta / R_A^2 in bohr^-2, using the Ghosh radii tabulated by
// Zhou et al. for TDDFT-ris/rid. atomic_number is the nuclear charge, not valence.
double minimal_gaussian_exponent(int atomic_number, double theta);

// One normalized radial function N_l r^l exp(-alpha_A r^2) per angular shell.
// The returned [species][l][radial index] basis uses ABACUS normalized real Y_lm.
NumericalAuxiliaryBasis make_minimal_gaussian_basis(
    const std::vector<int>& atomic_numbers, const MinimalGaussianBasisSettings& settings);
}
#endif
