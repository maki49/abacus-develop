#include "ri_hartree_fock.h"

#include "source_base/parallel_device.h"
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#include <RI/global/Tensor_Multiply.h>
#include <RI/physics/Exx.h>
#include <set>
#include <stdexcept>
#include <tuple>

namespace LR
{
namespace
{
RICell fold_cell(const RICell& cell, const RICell& period)
{
    RICell folded;
    // Match LibRI and RI_2D_Comm: centered representatives, including negative cells.
    for (int d = 0; d < 3; ++d)
    {
        folded[d] = (cell[d] % period[d] + 3 * period[d] / 2) % period[d] - period[d] / 2;
    }
    return folded;
}

RIAtomCell reverse_pair(const int atom, const RICell& cell, const RICell& period)
{
    const RICell negative{{-cell[0], -cell[1], -cell[2]}};
    const RICell folded = fold_cell(negative, period);
    return RIAtomCell(atom, folded);
}

template<typename T> void add_tensor(const RI::Tensor<T>& input, RI::Tensor<T>& sum)
{
    if (sum.empty()) { sum = input.copy(); }
    else { sum += input; }
}

// Requests exactly the density blocks contracted by the locally owned C rows,
// including the transposed partner. The judge is serialized by LibComm.
struct DensityBlocks
{
    std::set<std::tuple<int, RIAtomCell>> keys;
    bool judge(const std::tuple<int, RIAtomCell>& key) const { return keys.count(key) != 0; }
    template<class Archive> void serialize(Archive& archive) { archive(keys); }
};

template<typename T> RIAtomPairMap<T> fold_geometry(const RIAtomPairMap<T>& input, const RICell& period)
{
    RIAtomPairMap<T> folded;
    for (const auto& row : input)
    {
        for (const auto& block : row.second)
        {
            const RICell cell = fold_cell(block.first.second, period);
            const RIAtomCell pair(block.first.first, cell);
            add_tensor(block.second, folded[row.first][pair]);
        }
    }
    return folded;
}

template<typename T> void validate_density(const MPI_Comm& comm, const RIGeometry& geometry,
                                           const RIAtomPairMap<T>& density)
{
    double invalid = 0.0;
    const int nat = geometry.orbitals_per_atom.size();
    for (const auto& row : density)
    {
        for (const auto& block : row.second)
        {
            const int a = row.first;
            const int b = block.first.first;
            if (a < 0 || a >= nat || b < 0 || b >= nat || block.second.shape.size() != 2)
            {
                invalid = 1.0;
                continue;
            }
            if (block.second.shape[0] != geometry.orbitals_per_atom[a]
                || block.second.shape[1] != geometry.orbitals_per_atom[b]) { invalid = 1.0; }
            const RICell folded = fold_cell(block.first.second, geometry.period);
            if (folded != block.first.second) { invalid = 1.0; }
        }
    }
    // Fail on every rank before entering the tensor exchange, avoiding a rank-local throw/deadlock.
    Parallel_Common::reduce_data(&invalid, 1, comm);
    if (invalid != 0.0) { throw std::invalid_argument("RI density has incompatible AO dimensions or cells"); }
}

template<typename T> void validate_fit(const MPI_Comm& comm, const RIGeometry& geometry,
                                      const RIFittingTensors<T>& fit)
{
    const int nat = geometry.orbitals_per_atom.size();
    double invalid = fit.auxiliary_per_atom.size() != nat;
    for (const int count : fit.auxiliary_per_atom) { if (count <= 0) { invalid = 1.0; } }
    if (invalid == 0.0)
    {
        for (const auto* tensors : {&fit.Cs, &fit.Vs})
        {
            const bool coefficients = tensors == &fit.Cs;
            for (const auto& row : *tensors)
            {
                for (const auto& block : row.second)
                {
                    const int a = row.first;
                    const int b = block.first.first;
                    if (a < 0 || a >= nat || b < 0 || b >= nat) { invalid = 1.0; continue; }
                    std::vector<std::size_t> shape;
                    shape.push_back(fit.auxiliary_per_atom[a]);
                    if (coefficients)
                    {
                        shape.push_back(geometry.orbitals_per_atom[a]);
                        shape.push_back(geometry.orbitals_per_atom[b]);
                    }
                    else { shape.push_back(fit.auxiliary_per_atom[b]); }
                    if (block.second.shape.size() != shape.size()) { invalid = 1.0; }
                    else
                    {
                        for (std::size_t axis = 0; axis < shape.size(); ++axis)
                        {
                            if (block.second.shape[axis] != shape[axis]) { invalid = 1.0; }
                        }
                    }
                }
            }
        }
    }
    Parallel_Common::reduce_data(&invalid, 1, comm);
    if (invalid != 0.0) { throw std::invalid_argument("Incompatible RI fitting tensor dimensions"); }
}
}

template<typename T> struct RIHartreeFock<T>::Impl
{
    MPI_Comm comm;
    RIGeometry geometry;
    std::set<int> atoms;
    std::set<int> owned_atoms;
    std::vector<int> hartree_auxiliary_sizes;
    RIAtomPairMap<T> hartree_coefficients;
    std::map<int, std::map<int, RI::Tensor<T>>> hartree_metric_q0;
    DensityBlocks density_blocks;
    RI::Exx<int, int, 3, T> exchange;
};

template<typename T>
RIHartreeFock<T>::RIHartreeFock(const MPI_Comm& comm, const RIGeometry& geometry,
                              const RIFittingTensors<T>& hartree_fit, const RIFittingTensors<T>& exchange_fit)
{
    impl_.reset(new Impl);
    auto& data = *impl_;
    data.comm = comm;
    data.geometry = geometry;
    const int nat = geometry.orbitals_per_atom.size();
    double invalid = nat == 0 || geometry.positions.size() != nat;
    for (int d = 0; d < 3; ++d) { if (geometry.period[d] <= 0) { invalid = 1.0; } }
    for (int a = 0; a < nat; ++a)
    {
        if (geometry.positions.count(a) == 0 || geometry.orbitals_per_atom[a] <= 0) { invalid = 1.0; }
    }
    Parallel_Common::reduce_data(&invalid, 1, comm);
    if (invalid != 0.0) { throw std::invalid_argument("Invalid RI geometry or atom dimensions"); }
    validate_fit(comm, geometry, hartree_fit);
    validate_fit(comm, geometry, exchange_fit);
    data.hartree_auxiliary_sizes = hartree_fit.auxiliary_per_atom;
    const int rank = RI::MPI_Wrapper::mpi_get_rank(comm);
    const int size = RI::MPI_Wrapper::mpi_get_size(comm);
    for (int a = 0; a < nat; ++a)
    {
        data.atoms.insert(a);
        if (a % size == rank) { data.owned_atoms.insert(a); }
    }
    // Only the auxiliary-center owner stores these C rows. Geometry images are summed
    // once into the Born-von Karman cells; the density must not be summed over images again.
    const auto local_cs = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
        comm, hartree_fit.Cs, data.owned_atoms, data.atoms);
    data.hartree_coefficients = fold_geometry(local_cs, geometry.period);
    const auto local_vs = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
        comm, hartree_fit.Vs, data.owned_atoms, data.atoms);
    for (const auto& row : local_vs)
    {
        for (const auto& block : row.second)
        {
            add_tensor(block.second, data.hartree_metric_q0[row.first][block.first.first]);
        }
    }
    for (const auto& row : data.hartree_coefficients)
    {
        for (const auto& block : row.second)
        {
            data.density_blocks.keys.emplace(row.first, block.first);
            const RIAtomCell reverse = reverse_pair(row.first, block.first.second, geometry.period);
            data.density_blocks.keys.emplace(block.first.first, reverse);
        }
    }
    // Independent copies also protect the kernel against later mutation of caller tensors.
    const auto exchange_cs = fold_geometry(exchange_fit.Cs, geometry.period);
    const auto exchange_vs = fold_geometry(exchange_fit.Vs, geometry.period);
    data.exchange.set_parallel(comm, geometry.positions, geometry.lattice, geometry.period);
    data.exchange.set_Cs(exchange_cs, 0.0);
    data.exchange.set_Vs(exchange_vs, 0.0);
}

template<typename T> RIHartreeFock<T>::~RIHartreeFock() = default;

template<typename T>
RIAtomPairMap<T> RIHartreeFock<T>::apply_hartree(const RIAtomPairMap<T>& density) const
{
    const auto& data = *impl_;
    validate_density(data.comm, data.geometry, density);
    const auto needed_density = RI::Communicate_Tensors_Map::comm_map2(
        data.comm, density, data.density_blocks);
    std::map<int, RI::Tensor<T>> charge;
    for (const int a : data.owned_atoms)
    {
        const std::size_t naux = data.hartree_auxiliary_sizes[a];
        charge[a] = RI::Tensor<T>({naux});
    }
    // q_A = sum_BR,ij C^A_ij(R) [D_Ai,Bj(R) + D_Bj,Ai(-R)].
    // Transposition here exchanges real AO labels; it never conjugates the LR amplitude.
    for (const auto& row : data.hartree_coefficients)
    {
        for (const auto& block : row.second)
        {
            const auto& forward = RI::Global_Func::find(needed_density, row.first, block.first);
            if (!forward.empty())
            {
                const auto contribution = RI::Tensor_Multiply::gemv(block.second, forward);
                add_tensor(contribution, charge.at(row.first));
            }
            const RIAtomCell reverse = reverse_pair(row.first, block.first.second, data.geometry.period);
            const auto& backward = RI::Global_Func::find(needed_density, block.first.first, reverse);
            if (!backward.empty())
            {
                const auto transposed = backward.transpose();
                const auto contribution = RI::Tensor_Multiply::gemv(block.second, transposed);
                add_tensor(contribution, charge.at(row.first));
            }
        }
    }
    charge = RI::Communicate_Tensors_Map_Judge::comm_map(data.comm, charge, data.atoms);
    std::map<int, RI::Tensor<T>> potential;
    for (const int a : data.owned_atoms)
    {
        const std::size_t naux = data.hartree_auxiliary_sizes[a];
        potential[a] = RI::Tensor<T>({naux});
    }
    // w_A = sum_B V_AB(q=0) q_B. Only auxiliary vectors are replicated across ranks.
    for (const auto& row : data.hartree_metric_q0)
    {
        for (const auto& block : row.second)
        {
            const auto contribution = RI::Tensor_Multiply::gemv(block.second, charge.at(block.first));
            add_tensor(contribution, potential.at(row.first));
        }
    }
    RIAtomPairMap<T> result;
    for (const auto& row : data.hartree_coefficients)
    {
        for (const auto& block : row.second)
        {
            const auto contribution = RI::Tensor_Multiply::gemv_trans(block.second, potential.at(row.first));
            add_tensor(contribution, result[row.first][block.first]);
            const RIAtomCell reverse = reverse_pair(row.first, block.first.second, data.geometry.period);
            const auto transposed = contribution.transpose();
            add_tensor(transposed, result[block.first.first][reverse]);
        }
    }
    return result;
}

template<typename T>
RIAtomPairMap<T> RIHartreeFock<T>::apply_exchange(const RIAtomPairMap<T>& density)
{
    auto& data = *impl_;
    validate_density(data.comm, data.geometry, density);
    // No density screening or delta update: inner iterations require a fixed linear operator.
    data.exchange.set_Ds(density, 0.0);
    data.exchange.cal_Hs();
    return std::move(data.exchange.Hs);
}

template class RIHartreeFock<double>;
template class RIHartreeFock<std::complex<double>>;
}
