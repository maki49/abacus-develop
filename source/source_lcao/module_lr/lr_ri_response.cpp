#include "lr_ri_response.h"
#include "source_lcao/module_ri/ri_hartree_fock.h"

#include "dm_trans/dm_trans.h"
#include "exx_proj.h"
#include "source_base/parallel_device.h"
#include "source_hsolver/diag_comm_info.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace LR
{
namespace
{
double bloch_phase(double angle, double*)
{
    if (std::abs(std::sin(angle)) > 1e-12) { throw std::invalid_argument("Real RI response requires Gamma"); }
    return std::cos(angle);
}
std::complex<double> bloch_phase(double angle, std::complex<double>*)
{
    const std::complex<double> exponent(0, angle);
    return std::exp(exponent);
}

template<typename T> void sum(std::vector<T>& values, const hsolver::diag_comm_info& comm)
{
    const int size = values.size();
    if (size > 0 && comm.nproc > 1) { Parallel_Common::reduce_data(values.data(), size, comm.comm); }
}
}

template<typename T> struct RIResponseOperator<T>::Impl
{
    RIGeometry geometry;
    RITransitionBasis<T> basis;
    std::unique_ptr<RIHartreeFock<T>> kernel;
    std::vector<double> gaps;
    double exchange_fraction;
    hsolver::diag_comm_info comm;
    std::vector<int> atom_offsets;
    std::vector<int> spin_offsets;
    std::vector<RICell> cells;

    Impl(const RIGeometry& geometry_in, const RITransitionBasis<T>& basis_in,
         const std::vector<double>& gaps_in, double alpha, const hsolver::diag_comm_info& comm_in)
        : geometry(geometry_in), basis(basis_in), gaps(gaps_in), exchange_fraction(alpha), comm(comm_in)
    {}

    // Reuse the exact LR MO->AO contraction, then convert its storage to the
    // RI convention. Atom rows partition D across ranks; no hidden spin half.
    RIAtomPairMap<T> density(const T* amplitudes, int spin)
    {
        const int bands = basis.nocc[spin] + basis.nvirt[spin];
        psi::Psi<T> coefficients(basis.coefficients[spin].data(), basis.nk, bands, basis.naos, basis.naos, true);
        const T weight = T(1.0 / basis.nk);
        const auto matrices = cal_dm_trans_blas(amplitudes, coefficients, basis.nocc[spin], basis.nvirt[spin], weight);
        RIAtomPairMap<T> result;
        const int atoms = geometry.orbitals_per_atom.size();
        for (int a = comm.rank; a < atoms; a += comm.nproc)
        {
            const int ni = geometry.orbitals_per_atom[a];
            for (int b = 0; b < atoms; ++b)
            {
                const int nj = geometry.orbitals_per_atom[b];
                for (std::size_t r = 0; r < cells.size(); ++r)
                {
                    RI::Tensor<T> block({static_cast<std::size_t>(ni), static_cast<std::size_t>(nj)});
                    for (int k = 0; k < basis.nk; ++k)
                    {
                        const double angle = -phase_angle(k, cells[r]);
                        const T phase = bloch_phase(angle, static_cast<T*>(nullptr));
                        const T* matrix = matrices[k].template data<T>();
                        for (int i = 0; i < ni; ++i)
                        {
                            const int row = atom_offsets[a] + i;
                            for (int j = 0; j < nj; ++j)
                            {
                                const int col = atom_offsets[b] + j;
                                block(i, j) += phase * matrix[row * basis.naos + col];
                            }
                        }
                    }
                    result[a][{b, cells[r]}] = std::move(block);
                }
            }
        }
        return result;
    }

    double phase_angle(int k, const RICell& cell) const
    {
        double angle = 0;
        for (int d = 0; d < 3; ++d) { angle += basis.kpoints[k][d] * cell[d]; }
        return 2 * std::acos(-1.0) * angle;
    }

    // Every rank projects its additive AO contribution before the MO reduction.
    // Filtering by AO ownership here would discard contributions from the RI kernel.
    void project(const RIAtomPairMap<T>& potential, int spin, double factor, T* result)
    {
        const int nao = basis.naos;
        const int no = basis.nocc[spin];
        const int nv = basis.nvirt[spin];
        const int bands = no + nv;
        std::vector<T> matrix(static_cast<std::size_t>(nao) * nao);
        std::vector<T> scratch(static_cast<std::size_t>(nao) * no);
        for (int k = 0; k < basis.nk; ++k)
        {
            std::fill(matrix.begin(), matrix.end(), T(0));
            for (const auto& atom : potential)
            {
                const int row_offset = atom_offsets[atom.first];
                for (const auto& neighbor : atom.second)
                {
                    const int col_offset = atom_offsets[neighbor.first.first];
                    const double angle = phase_angle(k, neighbor.first.second);
                    const T phase = bloch_phase(angle, static_cast<T*>(nullptr));
                    const auto& block = neighbor.second;
                    for (std::size_t i = 0; i < block.shape[0]; ++i)
                    {
                        for (std::size_t j = 0; j < block.shape[1]; ++j)
                        {
                            matrix[(col_offset + j) * nao + row_offset + i] += phase * block(i, j);
                        }
                    }
                }
            }
            const T* occupied = basis.coefficients[spin].data() + k * bands * nao;
            const T* virtuals = occupied + no * nao;
            T* output = result + k * no * nv;
            project_exx(matrix.data(), occupied, virtuals, nao, no, nv, factor, scratch.data(), output);
        }
    }
};

template<typename T>
RIResponseOperator<T>::RIResponseOperator(const RIGeometry& geometry, const RITransitionBasis<T>& basis,
    std::unique_ptr<RIHartreeFock<T>> kernel, const std::vector<double>& local_gaps,
    double exchange_fraction, const hsolver::diag_comm_info& comm)
{
    impl_.reset(new Impl(geometry, basis, local_gaps, exchange_fraction, comm));
    auto& s = *impl_;
    s.kernel = std::move(kernel);
    const int spins = basis.nocc.size();
    if (!s.kernel || basis.naos < 1 || basis.nk < 1 || (spins != 1 && spins != 2)
        || basis.nvirt.size() != spins || basis.coefficients.size() != spins
        || basis.kpoints.size() != basis.nk || local_gaps.size() != basis.local_to_global.size())
    {
        throw std::invalid_argument("Invalid RI transition basis");
    }
    s.spin_offsets.push_back(0);
    for (int spin = 0; spin < spins; ++spin)
    {
        const int bands = basis.nocc[spin] + basis.nvirt[spin];
        const std::size_t size = static_cast<std::size_t>(basis.nk) * bands * basis.naos;
        if (basis.coefficients[spin].size() != size) { throw std::invalid_argument("Invalid RI MO coefficients"); }
        const int pairs = basis.nk * basis.nocc[spin] * basis.nvirt[spin];
        s.spin_offsets.push_back(s.spin_offsets.back() + pairs);
    }
    std::vector<double> owners(s.spin_offsets.back(), 0);
    for (int index : basis.local_to_global) { owners.at(index) += 1; }
    sum(owners, comm);
    for (double count : owners)
    {
        if (count != 1) { throw std::invalid_argument("RI amplitudes require exactly one MPI owner"); }
    }
    s.atom_offsets.push_back(0);
    for (int count : geometry.orbitals_per_atom) { s.atom_offsets.push_back(s.atom_offsets.back() + count); }
    if (s.atom_offsets.back() != basis.naos) { throw std::invalid_argument("RI atom/AO counts disagree"); }
    for (int x = -geometry.period[0] / 2; x < (geometry.period[0] + 1) / 2; ++x)
    {
        for (int y = -geometry.period[1] / 2; y < (geometry.period[1] + 1) / 2; ++y)
        {
            for (int z = -geometry.period[2] / 2; z < (geometry.period[2] + 1) / 2; ++z)
            { s.cells.push_back({{x, y, z}}); }
        }
    }
}

template<typename T> RIResponseOperator<T>::~RIResponseOperator() = default;

template<typename T> void RIResponseOperator<T>::apply(const T* amplitudes, T* image, RIResponseSpin spin)
{
    auto& s = *impl_;
    const int spins = s.basis.nocc.size();
    if ((spin == RIResponseSpin::unrestricted) != (spins == 2))
    { throw std::invalid_argument("RI response spin channel disagrees with MO basis"); }
    const int local_size = s.basis.local_to_global.size();
    std::vector<T> full(s.spin_offsets.back(), T(0));
    std::vector<T> response(full.size(), T(0));
    for (int i = 0; i < local_size; ++i) { full[s.basis.local_to_global[i]] = amplitudes[i]; }
    sum(full, s.comm);
    RIAtomPairMap<T> total_density;
    for (int input_spin = 0; input_spin < spins; ++input_spin)
    {
        const T* input = full.data() + s.spin_offsets[input_spin];
        const auto density = s.density(input, input_spin);
        if (s.exchange_fraction != 0)
        {
            const auto exchange = s.kernel->apply_exchange(density);
            const double factor = -2 * s.exchange_fraction; // Hartree -> Ry; same-spin only.
            T* output = response.data() + s.spin_offsets[input_spin];
            s.project(exchange, input_spin, factor, output);
        }
        if (spin == RIResponseSpin::triplet) { continue; }
        for (const auto& atom : density)
        {
            for (const auto& block : atom.second)
            {
                auto& target = total_density[atom.first][block.first];
                if (target.empty()) { target = block.second.copy(); }
                else { target = target + block.second; }
            }
        }
    }
    if (spin != RIResponseSpin::triplet)
    {
        const auto hartree = s.kernel->apply_hartree(total_density);
        const double factor = spin == RIResponseSpin::singlet ? 4.0 : 2.0;
        for (int output_spin = 0; output_spin < spins; ++output_spin)
        {
            T* output = response.data() + s.spin_offsets[output_spin];
            s.project(hartree, output_spin, factor, output);
        }
    }
    sum(response, s.comm);
    for (int i = 0; i < local_size; ++i)
    { image[i] = s.gaps[i] * amplitudes[i] + response[s.basis.local_to_global[i]]; }
}

template class RIResponseOperator<double>;
template class RIResponseOperator<std::complex<double>>;
}
