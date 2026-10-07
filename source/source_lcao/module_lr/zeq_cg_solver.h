#ifndef ABACUS_LR_ZEQ_CG_SOLVER_H
#define ABACUS_LR_ZEQ_CG_SOLVER_H

#include "source_hsolver/linear_solver_types.h"

#include <complex>
#include <functional>

namespace hsolver
{
struct diag_comm_info;
}

namespace LR
{
/// Host Hessian application; columns are independent states in the local pair layout.
using ZHessianAction = std::function<void(const double*, double*, int, int)>;

/// Solve the real Gamma-only Z equation and reject unconverged forces.
hsolver::LinearSolveResult solve_Z_CG(double* z, const double* rhs, int ld, int states,
                                     const ZHessianAction& action,
                                     const hsolver::diag_comm_info& comm, bool use_gpu);

void solve_Z_CG(std::complex<double>* z, const std::complex<double>* rhs, int ld, int states,
               const std::function<void(const std::complex<double>*, std::complex<double>*, int, int)>& action,
               const hsolver::diag_comm_info& comm, bool use_gpu);
}

#endif
