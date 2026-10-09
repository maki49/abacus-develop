#include "root_track.h"
#include "root_ovlp.h"
#include "source_psi/psi.h"
#include "lr_amp.h"
#include "source_base/parallel_2d.h"
#include "source_cell/unitcell.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include <algorithm>
#include <utility>

namespace LR
{
namespace
{
template <typename T>
RootBasis<T> capture_basis(const RootInputs<T>& inputs)
{
    RootBasis<T> basis;
    basis.lattice = inputs.cell.latvec;
    basis.lat0 = inputs.cell.lat0;
    basis.nbasis = inputs.pc.get_global_row_size();
    basis.nbands = inputs.nocc[0] + inputs.nvirt[0];
    basis.nocc = inputs.nocc;
    basis.nvirt = inputs.nvirt;
    basis.layout = {inputs.pc.get_block_size(), inputs.pc.get_dim0(), inputs.pc.get_dim1(),
        inputs.pc.get_coord_row(), inputs.pc.get_coord_col(), inputs.pc.get_row_size(), inputs.pc.get_col_size()};
    for (const auto& px : inputs.px)
    {
        basis.layout.push_back(px.get_block_size());
        basis.layout.push_back(px.get_row_size());
        basis.layout.push_back(px.get_col_size());
    }

    for (int atom = 0; atom < inputs.cell.nat; ++atom)
    {
        const auto position = inputs.cell.get_tau(atom);
        basis.positions.push_back(position);
    }
    // Snapshot only this rank's column-major coefficient block; no AO-by-band gather.
    const int coefficient_count = inputs.pc.get_local_size();
    const T zero = T(0);
    for (int spin = 0; spin < inputs.orbitals.get_nk(); ++spin)
    {
        std::vector<T> coefficients(coefficient_count, zero);
        for (int band = 0; band < inputs.pc.get_col_size(); ++band)
        {
            for (int ao = 0; ao < inputs.pc.get_row_size(); ++ao)
            {
                coefficients[band * inputs.pc.get_row_size() + ao] = inputs.orbitals(spin, band, ao);
            }
        }
        basis.coefficients.push_back(std::move(coefficients));
    }
    return basis;
}


template <typename T>
void project_local_reference(const RootInputs<T>& inputs, const RootBasis<T>& old,
    const RootBasis<T>& current, const std::vector<double>& ao, std::vector<T>& previous)
{
    Parallel_2D po;
#ifdef __MPI
    po.set(current.nbands, current.nbands, inputs.pc.get_block_size(), inputs.pc.blacs_ctxt);
#else
    po.set_serial(current.nbands, current.nbands);
#endif
    const std::vector<T> typed_ao(ao.begin(), ao.end());
    int offset = 0;
    const std::vector<int> spins = inputs.openshell ? std::vector<int>{0, 1} : std::vector<int>{inputs.channel};
    for (const int spin : spins)
    {
        const Parallel_2D& px = inputs.px[spin];
        const int no = inputs.nocc[spin];
        const int nv = inputs.nvirt[spin];
        const int local_count = px.get_local_size();
        std::vector<T> local_reference(local_count);
        std::copy_n(previous.begin() + offset, local_count, local_reference.begin());
        const auto local_mo = mo_overlap_dist(old.coefficients[spin], typed_ao,
            current.coefficients[spin], inputs.pc, inputs.pmat, po);
        const auto projected = project_reference_dist(local_reference, local_mo, po, px, no, nv);
        std::copy(projected.begin(), projected.end(), previous.begin() + offset);
        offset += px.get_local_size();
    }
}
}

template <typename T>
void follow_cross_root(const RootInputs<T>& inputs, const T* amplitudes,
    const int nlocal, const int nstates, const int initial_state, int& target,
    std::vector<T>& previous, RootBasis<T>& basis, std::ostream& log)
{
    RootBasis<T> current = capture_basis(inputs);
    if (!basis.positions.empty())
    {
        const auto difference = current.lattice - basis.lattice;
        const double cell_change = std::abs(difference.e11) + std::abs(difference.e12) + std::abs(difference.e13)
            + std::abs(difference.e21) + std::abs(difference.e22) + std::abs(difference.e23)
            + std::abs(difference.e31) + std::abs(difference.e32) + std::abs(difference.e33);
        if (cell_change > 1e-12 || std::abs(current.lat0 - basis.lat0) > 1e-12
            || current.nbasis != basis.nbasis || current.nbands != basis.nbands
            || current.nocc != basis.nocc || current.nvirt != basis.nvirt
            || current.layout != basis.layout
            || current.positions.size() != basis.positions.size()
            || current.coefficients.size() != basis.coefficients.size())
        {
            throw std::invalid_argument("LR root overlap: cell, orbital window or MPI layout changed between steps");
        }
        const auto ao = cross_ao_overlap(inputs.cell, basis.positions, inputs.cutoff, inputs.integrator, inputs.pmat);
        project_local_reference(inputs, basis, current, ao, previous);
        log << " EXCITED-STATE RELAX: root overlap uses old-new AO and MO bases." << std::endl;
    }
    follow_root(amplitudes, nlocal, nstates, initial_state, target, previous, log);
    basis = std::move(current);
}

template void follow_cross_root<double>(const RootInputs<double>&, const double*, int, int, int,
    int&, std::vector<double>&, RootBasis<double>&, std::ostream&);
template void follow_cross_root<std::complex<double>>(const RootInputs<std::complex<double>>&,
    const std::complex<double>*, int, int, int, int&, std::vector<std::complex<double>>&,
    RootBasis<std::complex<double>>&, std::ostream&);
}
