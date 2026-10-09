#ifndef ABACUS_LR_ROOT_TRACK_H
#define ABACUS_LR_ROOT_TRACK_H
#include "source_base/matrix3.h"
#include <vector>
#include <ostream>
class UnitCell;
class Parallel_2D;
class Parallel_Orbitals;
class TwoCenterIntegrator;
namespace base_device { struct DEVICE_CPU; }
namespace psi { template <typename T, typename Device> class Psi; }
namespace LR
{
// Snapshot of the previous geometry and MO basis; reference amplitudes are stored separately.
// Lattice and Cartesian positions use lat0 units; coefficients are local per-spin blocks.
template <typename T>
struct RootBasis
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

// Borrow current geometry, orbitals and AO/MO/electron-hole distributions for one tracking step.
// Cutoffs are in Bohr; pc/pmat/px describe coefficient, AO and amplitude matrices respectively.
template <typename T>
struct RootInputs
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

// Form old-new AO/MO overlaps and project the old reference into the current basis,
// then select the current root with largest overlap. Update target, previous and basis;
// current amplitudes stay unchanged. Requires Gamma, fixed cell/window and local layout.
template <typename T>
void follow_cross_root(const RootInputs<T>& inputs, const T* amplitudes,
    int nlocal, int nstates, int initial_state, int& target,
    std::vector<T>& previous, RootBasis<T>& basis, std::ostream& log);
}
#endif
