#include "zeq_cg_solver.h"

#include "source_base/timer.h"
#include "source_base/module_device/memory_op.h"
#include "source_hsolver/linear_pcg.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/parallel_device.h"
#include "source_hsolver/linear_algebra.h"

#include <algorithm>
#include <cmath>
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

// A small positive gap floor limits the inverse without changing the Hessian.
const double orbital_gap_floor = 1e-8; // Ry

std::vector<double> inverse_gaps(const std::vector<double>& diagonal, int ld,
                                 const hsolver::diag_comm_info& comm)
{
    double invalid = 0.0;
    const bool identity = diagonal.empty();
    if (!identity && diagonal.size() != static_cast<std::size_t>(ld)) { invalid = 1.0; }
    std::vector<double> inverse(ld, 1.0);
    if (!identity && invalid == 0.0)
    {
        for (int row = 0; row < ld; ++row)
        {
            const double gap = diagonal[row];
            if (!std::isfinite(gap) || gap <= 0.0) { invalid = 1.0; }
            else { inverse[row] = 1.0 / std::max(gap, orbital_gap_floor); }
        }
    }
#ifdef __MPI
    // Every rank must reject invalid gaps before entering Krylov collectives.
    if (comm.nproc > 1) { Parallel_Common::reduce_data(&invalid, 1, comm.comm); }
#endif
    if (invalid > 0.0) { throw std::invalid_argument("Z-vector orbital gaps must be finite, positive and match local ld"); }
    return inverse;
}

template<typename Device>
class OrbitalInverse final : public hsolver::LinearOperator<double, Device>
{
public:
    explicit OrbitalInverse(const std::vector<double>& inverse)
    {
        const std::int64_t size = inverse.size();
        hsolver::linear_buffer<double, Device>(&inverse_, size);
        if (size > 0)
        {
            base_device::memory::synchronize_memory_op<double, Device, base_device::DEVICE_CPU>()(
                inverse_.data<double>(), inverse.data(), size);
        }
    }
    void apply(const double* x, double* y, int ld, int columns) const override
    {
        const bool add = false;
        for (int column = 0; column < columns; ++column)
        {
            const int offset = column * ld;
            const double* input = x + offset;
            double* output = y + offset;
            ModuleBase::vector_mul_vector_op<double, Device, double>()(
                ld, output, input, inverse_.data<double>(), add);
        }
    }
private:
    ct::Tensor inverse_;
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
                                      double tolerance, int max_iterations,
                                      const std::vector<double>& inverse_gaps)
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
    const OrbitalInverse<base_device::DEVICE_GPU> inverse(inverse_gaps);
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
                                     const hsolver::diag_comm_info& comm, bool use_gpu,
                                     const std::vector<double>& orbital_diagonal)
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
    std::vector<double> gaps;
    try
    {
        gaps = inverse_gaps(orbital_diagonal, ld, comm);
    }
    catch (...)
    {
        ModuleBase::timer::end("Z_vector", "solve_Z_CG");
        throw;
    }
    hsolver::LinearSolveResult result;
    if (use_gpu)
    {
#if defined(__CUDA) || defined(__ROCM)
        result = solve_on_gpu(z, rhs, ld, states, action, comm, tolerance, max_iterations, gaps);
#else
        ModuleBase::timer::end("Z_vector", "solve_Z_CG");
        throw std::runtime_error("GPU Z-vector PCG requires a GPU-enabled build");
#endif
    }
    else
    {
        const HostZHessian hessian(action);
        const OrbitalInverse<base_device::DEVICE_CPU> inverse(gaps);
        result = hsolver::linear_pcg<base_device::DEVICE_CPU>(
            hessian, inverse, z, rhs, ld, ld, states, tolerance, max_iterations, comm);
    }
    const char* preconditioner = orbital_diagonal.empty() ? "identity" : "orbital-energy-gap";
    std::cout << "Z-vector PCG preconditioner=" << preconditioner
              << " gap floor (Ry)=" << orbital_gap_floor << std::endl;
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
               const hsolver::diag_comm_info&, bool, const std::vector<double>&)
{
    throw std::runtime_error("complex Z-vector solver is not implemented yet");
}
}
