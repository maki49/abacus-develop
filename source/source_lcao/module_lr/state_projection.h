#ifndef ABACUS_LR_STATE_PROJECTION_H
#define ABACUS_LR_STATE_PROJECTION_H

#include "source_base/matrix3.h"
#include <vector>

class UnitCell;
class Parallel_2D;
class Parallel_Orbitals;
class TwoCenterIntegrator;
namespace base_device { struct DEVICE_CPU; }
namespace psi { template <typename T, typename Device> class Psi; }
namespace LR
{
// Geometry and local MO coefficients needed to compare transition vectors at two
// ionic steps. Positions use lat0 units; coefficients are column-major AO x MO.
template <typename T>
struct OrbitalBasisSnapshot
{
    ModuleBase::Matrix3 lattice;
    double lat0 = 0.0;
    int nbasis = 0;
    int nbands = 0;
    std::vector<ModuleBase::Vector3<double>> positions;
    std::vector<int> nocc;
    std::vector<int> nvirt;
    // Local column-major AO-by-band blocks; layout must stay fixed between steps.
    std::vector<int> layout;
    std::vector<std::vector<T>> coefficients;
};

// Borrow the current geometry and distributions for a transition-space projection.
// openshell selects [up | down]; otherwise channel selects singlet/triplet orbitals.
template <typename T>
struct TransitionBasisView
{
    const UnitCell& cell;
    const std::vector<double>& cutoff;
    const TwoCenterIntegrator& integrator;
    const psi::Psi<T, base_device::DEVICE_CPU>& orbitals;
    const Parallel_2D& pc;
    const Parallel_Orbitals& pmat;
    const std::vector<Parallel_2D>& px;
    const std::vector<int>& nocc;
    const std::vector<int>& nvirt;
    bool openshell;
    int channel;
};

template <typename T>
OrbitalBasisSnapshot<T> capture_orbital_basis(const TransitionBasisView<T>& inputs);

// Projection requires a fixed cell, orbital window and MPI distribution.
template <typename T>
bool compatible_orbital_bases(const OrbitalBasisSnapshot<T>& old,
                             const OrbitalBasisSnapshot<T>& current);

// Project each borrowed vector with X_new = O_vv^dagger X_old O_oo, separately
// for each spin. ao is the local ordered old-bra/new-ket AO overlap; no normalization.
template <typename T>
void project_transition_vectors(const TransitionBasisView<T>& inputs,
    const OrbitalBasisSnapshot<T>& old, const OrbitalBasisSnapshot<T>& current,
    const std::vector<double>& ao, const std::vector<std::vector<T>*>& vectors);
}
#endif
