#include "zeq_cg_solver.h"

#include "source_base/timer.h"
#include "source_base/module_device/memory_op.h"
#include "source_hsolver/linear_pcg.h"
#if defined(__CUDA) || defined(__ROCM)
#include "source_base/kernels/math_kernel_op.h"
#include "source_hsolver/linear_algebra.h"
#endif

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

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

template<typename Device>
class Identity final : public hsolver::LinearOperator<double, Device>
{
public:
    void apply(const double* x, double* y, int ld, int columns) const override
    {
        const std::size_t size = static_cast<std::size_t>(ld) * columns;
        if (size > 0)
        {
            base_device::memory::synchronize_memory_op<double, Device, Device>()(y, x, size);
        }
    }
    bool is_identity() const override { return true; }
};

#if defined(__CUDA) || defined(__ROCM)
/// Transitional boundary: Krylov vectors stay on device; the AO Hessian still accepts host data.
/// Reuse staging buffers for the whole solve, including its true-residual applications.
class DeviceZHessian final : public hsolver::LinearOperator<double, base_device::DEVICE_GPU>
{
public:
    DeviceZHessian(const ZHessianAction& action, std::size_t elements) : action_(action)
    {
        const std::size_t capacity = std::max<std::size_t>(1, elements);
        input_.resize(capacity);
        output_.resize(capacity);
    }
    void apply(const double* x, double* y, int ld, int columns) const override
    {
        const std::size_t size = static_cast<std::size_t>(ld) * columns;
        if (size > input_.size()) { throw std::invalid_argument("Z Hessian staging buffer is too small"); }
        if (size > 0)
        {
            base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
                input_.data(), x, size);
        }
        action_(input_.data(), output_.data(), ld, columns);
        if (size > 0)
        {
            base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
                y, output_.data(), size);
        }
    }
private:
    const ZHessianAction& action_;
    mutable std::vector<double> input_;
    mutable std::vector<double> output_;
};

hsolver::LinearSolveResult solve_on_gpu(double* z, const double* rhs, int ld, int states,
                                      const ZHessianAction& action,
                                      const hsolver::diag_comm_info& comm,
                                      double tolerance, int max_iterations)
{
    ModuleBase::createGpuBlasHandle();
    const std::int64_t size = static_cast<std::int64_t>(ld) * states;
    ct::Tensor x;
    ct::Tensor b;
    hsolver::linear_buffer<double, base_device::DEVICE_GPU>(&x, size);
    hsolver::linear_buffer<double, base_device::DEVICE_GPU>(&b, size);
    x.zero();
    if (size > 0)
    {
        base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            b.data<double>(), rhs, size);
    }
    const DeviceZHessian hessian(action, size);
    const Identity<base_device::DEVICE_GPU> inverse;
    const auto result = hsolver::linear_pcg<base_device::DEVICE_GPU>(
        hessian, inverse, x.data<double>(), b.data<double>(), ld, ld, states, tolerance, max_iterations, comm);
    if (result.status == hsolver::LinearSolveStatus::converged && size > 0)
    {
        base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
            z, x.data<double>(), size);
    }
    // Excludes the small scalar dot products / MPI coefficients handled by PCG.
    const std::int64_t bridge_bytes = 2 * size * sizeof(double) * result.operator_calls;
    std::cout << "Z-vector GPU host-Hessian bridge: local bytes=" << bridge_bytes << std::endl;
    return result;
}
#endif
}

hsolver::LinearSolveResult solve_Z_CG(double* z, const double* rhs, int ld, int states,
                                     const ZHessianAction& action,
                                     const hsolver::diag_comm_info& comm, bool use_gpu)
{
    ModuleBase::timer::start("Z_vector", "solve_Z_CG");
    if (ld < 0 || states < 0)
    {
        ModuleBase::timer::end("Z_vector", "solve_Z_CG");
        throw std::invalid_argument("Invalid Z-vector dimensions");
    }
    const std::size_t size = static_cast<std::size_t>(ld) * states;
    if (size > 0) { std::fill(z, z + size, 0.0); }
    const double tolerance = 1e-6;
    const int max_iterations = 100;
    hsolver::LinearSolveResult result;
    if (use_gpu)
    {
#if defined(__CUDA) || defined(__ROCM)
        result = solve_on_gpu(z, rhs, ld, states, action, comm, tolerance, max_iterations);
#else
        ModuleBase::timer::end("Z_vector", "solve_Z_CG");
        throw std::runtime_error("GPU Z-vector PCG requires a GPU-enabled build");
#endif
    }
    else
    {
        const HostZHessian hessian(action);
        const Identity<base_device::DEVICE_CPU> inverse;
        result = hsolver::linear_pcg<base_device::DEVICE_CPU>(
            hessian, inverse, z, rhs, ld, ld, states, tolerance, max_iterations, comm);
    }
    const char* backend = use_gpu ? "GPU" : "CPU";
    std::cout << "Z-vector PCG backend=" << backend << ": iterations=" << result.iterations
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
               const hsolver::diag_comm_info&, bool)
{
    throw std::runtime_error("complex Z-vector solver is not implemented yet");
}
}
