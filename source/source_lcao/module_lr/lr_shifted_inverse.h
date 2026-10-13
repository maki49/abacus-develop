#ifndef LR_SHIFTED_INVERSE_H
#define LR_SHIFTED_INVERSE_H

#include <functional>
#include <vector>

namespace hsolver { struct diag_comm_info; }

namespace LR
{
struct ShiftedInverseSettings
{
    double relative_tolerance = 1e-2;
    int max_iterations = 20;
    double denominator_floor = 1e-4; // Ry, for the signed gap inverse inside GMRES.
};

struct ShiftedInverseResult
{
    int operator_vectors = 0; // Includes the independent final residual check.
    double relative_residual = 0;
    bool converged = false;
    bool used_gap_fallback = false;
};

/// Approximately solve (A_proxy - shift I) z = rhs using right-preconditioned
/// GMRES; A_proxy and gaps are in Ry. Distributed local rows may be empty.
/// Overwrite rhs with z. Check the true inner residual; if it did not decrease,
/// return the signed gap correction. An indefinite shifted matrix is allowed.
template<typename T>
ShiftedInverseResult apply_shifted_inverse(
    T* rhs, int dim, double shift, const std::vector<double>& gaps,
    const std::function<void(const T*, T*)>& proxy,
    const ShiftedInverseSettings& settings, const hsolver::diag_comm_info& comm);
}
#endif
