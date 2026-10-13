#ifndef ABACUS_LR_DAVIDSON_H
#define ABACUS_LR_DAVIDSON_H

#include "source_hsolver/diago_dav_subspace.h"

namespace LR
{

/// Dimensions and numerical settings for the LR excitation-space Davidson solve.
struct DavidsonSolveSettings
{
    int roots;
    int local_dimension;
    int subspace_multiplier;
    double energy_tolerance;
    int max_iterations;
    int projected_solver;
    int block_size;
};

/// Solve H X = X Omega (S = I) with host LR actions and host input/output amplitudes.
/// On GPU, Davidson's basis, Ritz rotations and reduced solve reside on the device;
/// only H applications and the caller's inverse action cross the explicit host boundary.
/// host_vectors contains options.initial_vectors columns on entry and roots on return.
template<typename T>
hsolver::DavidsonResult solve_davidson(
    const hsolver::HSOperator<T>& host_operator,
    T* host_vectors,
    double* eigenvalues,
    const std::vector<double>& diagonal,
    const typename hsolver::Diago_DavSubspace<T>::Preconditioner& host_inverse,
    const hsolver::DavidsonOptions& options,
    const DavidsonSolveSettings& settings,
    const hsolver::diag_comm_info& comm,
    bool use_gpu);

} // namespace LR
#endif
