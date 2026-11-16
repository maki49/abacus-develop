#include "state_projection.h"
#include "state_ovlp.h"
#include "source_psi/psi.h"
#include "source_base/parallel_blkcyc.h"
#include "source_cell/unitcell.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include <algorithm>
#include <complex>
#include <utility>

namespace LR
{
template <typename T>
OrbitalBasisSnapshot<T> capture_orbital_basis(const TransitionBasisView<T>& inputs)
{
    OrbitalBasisSnapshot<T> basis;
    basis.lattice = inputs.cell.latvec;
    basis.lat0 = inputs.cell.lat0;
    basis.nbasis = inputs.pc.nrglb();
    basis.nbands = inputs.nocc[0] + inputs.nvirt[0];
    basis.nocc = inputs.nocc;
    basis.nvirt = inputs.nvirt;
    basis.layout = {inputs.pc.nb(), inputs.pc.nrproc(), inputs.pc.ncproc(),
        inputs.pc.irproc(), inputs.pc.icproc(), inputs.pc.nrloc(), inputs.pc.ncloc()};
    for (const auto& px : inputs.px)
    {
        basis.layout.push_back(px.nb());
        basis.layout.push_back(px.nrloc());
        basis.layout.push_back(px.ncloc());
    }

    for (int atom = 0; atom < inputs.cell.nat; ++atom)
    {
        const ModuleBase::Vector3<double> position = inputs.cell.get_tau(atom);
        basis.positions.push_back(position);
    }
    // Snapshot only this rank's column-major coefficient block; no AO-by-band gather.
    const int coefficient_count = inputs.pc.size();
    const T zero = T(0);
    for (int spin = 0; spin < inputs.orbitals.get_nk(); ++spin)
    {
        std::vector<T> coefficients(coefficient_count, zero);
        for (int band = 0; band < inputs.pc.ncloc(); ++band)
        {
            for (int ao = 0; ao < inputs.pc.nrloc(); ++ao)
            {
                coefficients[band * inputs.pc.nrloc() + ao] = inputs.orbitals(spin, band, ao);
            }
        }
        basis.coefficients.push_back(std::move(coefficients));
    }
    return basis;
}


template <typename T>
void project_transition_vectors(const TransitionBasisView<T>& inputs, const OrbitalBasisSnapshot<T>& old,
    const OrbitalBasisSnapshot<T>& current, const std::vector<double>& ao, const std::vector<std::vector<T>*>& references)
{
    const std::vector<T> typed_ao(ao.begin(), ao.end());
    int offset = 0;
    const std::vector<int> spins = inputs.openshell ? std::vector<int>{0, 1} : std::vector<int>{inputs.channel};
    for (const int spin : spins)
    {
        const Parallel_Blkcyclic& px = inputs.px[spin];
        const int no = inputs.nocc[spin];
        const int nv = inputs.nvirt[spin];
        const int local_count = px.size();
        Parallel_Blkcyclic pi;
        Parallel_Blkcyclic pv;
#ifdef __MPI
        const int block_size = px.nb();
        pi.set(no, no, block_size, px.ictxt());
        pv.set(nv, nv, block_size, px.ictxt());
#else
        pi.set_serial(no, no);
        pv.set_serial(nv, nv);
#endif
        // Only oo/vv are needed; construct each once and reuse across references.
        const int occupied_start = 0;
        const std::vector<T> occupied = mo_overlap_dist(old.coefficients[spin], typed_ao,
            current.coefficients[spin], inputs.pc, inputs.pmat.para_2d(), pi, occupied_start);
        const std::vector<T> virtuals = mo_overlap_dist(old.coefficients[spin], typed_ao,
            current.coefficients[spin], inputs.pc, inputs.pmat.para_2d(), pv, no);
        for (std::vector<T>* reference : references)
        {
            std::vector<T> local_reference(local_count);
            std::copy_n(reference->begin() + offset, local_count, local_reference.begin());
            const std::vector<T> projected = project_reference_dist(local_reference, occupied, virtuals, pi, pv, px);
            std::copy(projected.begin(), projected.end(), reference->begin() + offset);
        }
        offset += px.size();
    }
}

template <typename T>
bool compatible_orbital_bases(const OrbitalBasisSnapshot<T>& old,
                             const OrbitalBasisSnapshot<T>& current)
{
    const auto difference = current.lattice - old.lattice;
    const double cell_change = std::abs(difference.e11) + std::abs(difference.e12) + std::abs(difference.e13)
            + std::abs(difference.e21) + std::abs(difference.e22) + std::abs(difference.e23)
            + std::abs(difference.e31) + std::abs(difference.e32) + std::abs(difference.e33);
    return !(cell_change > 1e-12 || std::abs(current.lat0 - old.lat0) > 1e-12
            || current.nbasis != old.nbasis || current.nbands != old.nbands
            || current.nocc != old.nocc || current.nvirt != old.nvirt
            || current.layout != old.layout
            || current.positions.size() != old.positions.size()
            || current.coefficients.size() != old.coefficients.size());
}

template OrbitalBasisSnapshot<double> capture_orbital_basis(const TransitionBasisView<double>&);
template bool compatible_orbital_bases(const OrbitalBasisSnapshot<double>&, const OrbitalBasisSnapshot<double>&);
template void project_transition_vectors(const TransitionBasisView<double>&,
    const OrbitalBasisSnapshot<double>&, const OrbitalBasisSnapshot<double>&,
    const std::vector<double>&, const std::vector<std::vector<double>*>&);
template OrbitalBasisSnapshot<std::complex<double>> capture_orbital_basis(const TransitionBasisView<std::complex<double>>&);
template bool compatible_orbital_bases(const OrbitalBasisSnapshot<std::complex<double>>&, const OrbitalBasisSnapshot<std::complex<double>>&);
template void project_transition_vectors(const TransitionBasisView<std::complex<double>>&,
    const OrbitalBasisSnapshot<std::complex<double>>&, const OrbitalBasisSnapshot<std::complex<double>>&,
    const std::vector<double>&, const std::vector<std::vector<std::complex<double>>*>&);
}
