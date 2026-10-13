#ifndef RI_HARTREE_FOCK_H
#define RI_HARTREE_FOCK_H

#include <RI/global/Tensor.h>
#include <array>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

namespace LR
{
using RICell = std::array<int, 3>;
using RIAtomCell = std::pair<int, RICell>;
template<typename T> using RIAtomPairMap = std::map<int, std::map<RIAtomCell, RI::Tensor<T>>>;

struct RIGeometry
{
    std::map<int, std::array<double, 3>> positions; // Cartesian; same units as lattice.
    std::array<std::array<double, 3>, 3> lattice;
    RICell period; // Born-von Karman period; positive entries.
    std::vector<int> orbitals_per_atom; // Atom indices must be contiguous, starting at zero.
};

template<typename T> struct RIFittingTensors
{
    // phi_Ai phi_Bj = sum_mu C^{A,mu}_{Ai,Bj} P_A,mu + the B-centered term.
    // Cs[A][{B,R}] has shape [naux_A, nao_A, nao_B]; Vs has [naux_A, naux_B].
    // The on-site C contains the usual LRI half factor. Entries are additive across ranks.
    RIAtomPairMap<T> Cs;
    RIAtomPairMap<T> Vs;
    std::vector<int> auxiliary_per_atom;
};

// Fixed real-AO RI kernel for q=0 response, with independent Hartree and exchange fitting spaces.
// Owns its tensors and electronic workspace; no link to a KS Exx_LRI instance.
// All methods are collective on comm, which must outlive this object.
template<typename T> class RIHartreeFock
{
  public:
    RIHartreeFock(const MPI_Comm& comm, const RIGeometry& geometry,
                 const RIFittingTensors<T>& hartree_fit, const RIFittingTensors<T>& exchange_fit);
    ~RIHartreeFock();
    RIHartreeFock(const RIHartreeFock&) = delete;
    RIHartreeFock& operator=(const RIHartreeFock&) = delete;

    // J_ij = sum_kl (ij|kl)_J D_kl and K_ij = sum_kl (ik|jl)_K D_kl, in Hartree.
    // Positive J/K: caller supplies spin factors, -hybrid_fraction for Fock, and 2 for Ry.
    // D[A][{B,R}](i,j) multiplies phi_Ai(0) phi_Bj(R); it need not be Hermitian.
    // D(R) includes the k weights (1/Nk on a full uniform mesh), but no implicit spin factor.
    // Inputs/outputs are additive distributed maps, never replicated contributions.
    // Density cells use LibRI's centered representatives [-N/2,(N-1)/2] in each direction;
    // missing blocks mean zero. Unlike geometry images, density images must not be duplicated.
    RIAtomPairMap<T> apply_hartree(const RIAtomPairMap<T>& density) const;
    RIAtomPairMap<T> apply_exchange(const RIAtomPairMap<T>& density);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
#endif
