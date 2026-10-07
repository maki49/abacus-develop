#ifndef HSOLVER_LINEAR_PCG_H
#define HSOLVER_LINEAR_PCG_H

#include "linear_operator.h"
#include "linear_solver_types.h"

namespace hsolver
{
struct diag_comm_info;

/// Real SPD linear PCG with independent recurrences and batched operator calls.
/// x and rhs are Device buffers, with dim valid rows and ld rows per column.
/// The preconditioner applies an SPD approximation to the inverse operator.
/// Convergence uses the absolute, globally reduced Euclidean residual norm.
template<typename Device>
LinearSolveResult linear_pcg(const LinearOperator<double, Device>& op,
                             const LinearOperator<double, Device>& inverse,
                             double* x,
                             const double* rhs,
                             int ld,
                             int dim,
                             int columns,
                             double tolerance,
                             int max_iterations,
                             const diag_comm_info& comm);
}

#endif
