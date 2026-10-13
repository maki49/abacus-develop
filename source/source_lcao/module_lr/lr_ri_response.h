#ifndef LR_RI_RESPONSE_H
#define LR_RI_RESPONSE_H

#include <array>
#include <memory>
#include <vector>

namespace hsolver { struct diag_comm_info; }

namespace LR
{
struct RIGeometry;
template<typename T> class RIHartreeFock;
enum class RIResponseSpin { singlet, triplet, unrestricted };

/// Owned MO data for the approximate kernel. Coefficients[spin][k][MO][AO]
/// are replicated; local_to_global maps distributed [spin][k][i][a] amplitudes
/// to that same global order. kpoints use reciprocal fractional coordinates.
template<typename T> struct RITransitionBasis
{
    int naos = 0;
    int nk = 0;
    std::vector<int> nocc;
    std::vector<int> nvirt;
    std::vector<std::vector<T>> coefficients;
    std::vector<std::array<double, 3>> kpoints;
    std::vector<int> local_to_global;
};

/// A_proxy = gaps + J - alpha K in Ry, without the semilocal XC kernel.
/// Singlet: 2 J[D]; triplet: no J; unrestricted: J[D_up+D_down] for both
/// output spins. K acts only within each spin. J/K return Hartree internally.
/// Owns fixed RI tensors and MO coefficients for one geometry/KS spectrum;
/// no link to the full LR Hamiltonian or the KS EXX electronic workspace.
template<typename T> class RIResponseOperator
{
  public:
    RIResponseOperator(const RIGeometry& geometry, const RITransitionBasis<T>& basis,
                       std::unique_ptr<RIHartreeFock<T>> kernel,
                       const std::vector<double>& local_gaps, double exchange_fraction,
                       const hsolver::diag_comm_info& comm);
    ~RIResponseOperator();
    /// Overwrite one distributed MO vector. Collective on the construction communicator.
    void apply(const T* amplitudes, T* image, RIResponseSpin spin);
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
#endif
