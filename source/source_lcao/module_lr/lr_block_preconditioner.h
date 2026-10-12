#ifndef ABACUS_LR_BLOCK_PRECONDITIONER_H
#define ABACUS_LR_BLOCK_PRECONDITIONER_H

#include <functional>
#include <vector>

namespace hsolver { struct diag_comm_info; }

namespace LR
{
/// Own the small replicated eigendecomposition A_PP = U Lambda U^dagger.
/// P contains selected transition coordinates; no full operator or AO data is
/// retained. local_indices maps P into this rank's amplitude rows (-1 off rank).
template<typename T>
struct LowEnergyPreconditioner
{
    std::vector<int> local_indices;
    std::vector<double> eigenvalues;
    std::vector<T> eigenvectors; // Column-major U in the replicated P space.
    double hermiticity_error = 0.0; // max |A_PP - A_PP^dagger| before symmetrizing, Ry.
};

/// Probe the full LR operator on the selected unit transitions, gather A_PP,
/// and diagonalize its Hermitian part once. Probes are batched to limit scratch
/// storage to O(local_dim); exactly |P| operator vectors are applied.
template<typename T>
LowEnergyPreconditioner<T> build_low_energy_preconditioner(
    const std::vector<int>& local_indices, int local_dim,
    const std::function<void(const T*, T*, int)>& apply_operator,
    const hsolver::diag_comm_info& comm);

/// Apply z_P = U (Lambda - omega)^(-1) U^dagger r_P and
/// z_Q = (D_Q - omega)^(-1) r_Q to a CPU residual block in place.
/// P-Q coupling is omitted only here; the outer operator/window is unchanged.
/// Both inverses use the signed denominator floor in Ry. P residuals are
/// gathered on comm; dim is the local row count and column stride.
template<typename T>
void apply_low_energy_preconditioner(T* residuals, const double* shifts, int dim, int nvec,
                                    const std::vector<double>& gaps, double denominator_floor,
                                    const LowEnergyPreconditioner<T>& block,
                                    const hsolver::diag_comm_info& comm);
} // namespace LR

#endif
