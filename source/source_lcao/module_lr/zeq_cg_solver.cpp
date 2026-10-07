#include "zeq_cg_solver.h"

#include "source_base/timer.h"
#include "source_hsolver/linear_pcg.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace LR
{
namespace
{
class HostZHessian final : public hsolver::LinearOperator<double, base_device::DEVICE_CPU>
{
public:
    explicit HostZHessian(const ZHessianAction& action) : action_(action) {}
    void apply(const double* x, double* y, int ld, int columns) const override
    {
        action_(x, y, ld, columns);
    }
private:
    const ZHessianAction& action_;
};

class Identity final : public hsolver::LinearOperator<double, base_device::DEVICE_CPU>
{
public:
    void apply(const double* x, double* y, int ld, int columns) const override
    {
        const std::size_t size = static_cast<std::size_t>(ld) * columns;
        if (size > 0) { std::copy(x, x + size, y); }
    }
    bool is_identity() const override { return true; }
};
}

hsolver::LinearSolveResult solve_Z_CG(double* z, const double* rhs, int ld, int states,
                                     const ZHessianAction& action,
                                     const hsolver::diag_comm_info& comm)
{
    ModuleBase::timer::start("Z_vector", "solve_Z_CG");
    if (ld < 0 || states < 0) { throw std::invalid_argument("Invalid Z-vector dimensions"); }
    const std::size_t size = static_cast<std::size_t>(ld) * states;
    if (size > 0) { std::fill(z, z + size, 0.0); }
    const HostZHessian hessian(action);
    const Identity inverse;
    const double tolerance = 1e-6;
    const int max_iterations = 100;
    const auto result = hsolver::linear_pcg<base_device::DEVICE_CPU>(
        hessian, inverse, z, rhs, ld, ld, states, tolerance, max_iterations, comm);
    std::cout << "Z-vector PCG: iterations=" << result.iterations
              << " true residual=" << result.max_residual
              << " operator calls=" << result.operator_calls << std::endl;
    ModuleBase::timer::end("Z_vector", "solve_Z_CG");
    if (result.status != hsolver::LinearSolveStatus::converged)
    {
        std::ostringstream message;
        message << "Z-vector PCG failed for RHS " << result.failed_band
                << ": residual=" << result.max_residual
                << ", iterations=" << result.iterations
                << ". Check convergence and positive definiteness of the orbital Hessian.";
        throw std::runtime_error(message.str());
    }
    return result;
}

void solve_Z_CG(std::complex<double>*, const std::complex<double>*, int, int,
               const std::function<void(const std::complex<double>*, std::complex<double>*, int, int)>&,
               const hsolver::diag_comm_info&)
{
    throw std::runtime_error("complex Z-vector solver is not implemented yet");
}
}
