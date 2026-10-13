#include "lr_davidson.h"

#include "source_base/kernels/math_kernel_op.h"
#include "source_base/module_container/ATen/core/tensor.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/timer.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace LR
{
namespace
{
template<typename T, typename Device>
hsolver::DavidsonResult run_davidson(
    const hsolver::HSOperator<T, Device>& op,
    T* vectors,
    double* eigenvalues,
    const std::vector<double>& diagonal,
    const typename hsolver::Diago_DavSubspace<T, Device>::Preconditioner& inverse,
    const hsolver::DavidsonOptions& options,
    const DavidsonSolveSettings& settings,
    const hsolver::diag_comm_info& comm)
{
    hsolver::Diago_DavSubspace<T, Device> solver(diagonal, settings.roots, settings.local_dimension,
        settings.subspace_multiplier, settings.energy_tolerance, settings.max_iterations, comm,
        settings.projected_solver, settings.block_size);
    const std::vector<double> energy_tolerances(settings.roots, settings.energy_tolerance);
    return solver.diag(op, vectors, settings.local_dimension, eigenvalues, energy_tolerances, inverse, options);
}

#if defined(__CUDA) || defined(__ROCM)
/// LR's AO/RI operator and inverse callback still accept host pointers. Reuse one
/// staging workspace for their sequential applications; never pass them device addresses.
template<typename T>
class DeviceDavidsonBridge final : public hsolver::HSOperator<T, base_device::DEVICE_GPU>
{
public:
    using HostInverse = typename hsolver::Diago_DavSubspace<T>::Preconditioner;
    DeviceDavidsonBridge(const hsolver::HSOperator<T>& op, const HostInverse& inverse,
                         std::size_t elements, int roots) : host_operator_(op), host_inverse_(inverse)
    {
        const std::size_t capacity = std::max<std::size_t>(1, elements);
        input_.resize(capacity);
        output_.resize(capacity);
        shifts_.resize(roots);
    }

    void update_k(int) override {}

    void hpsi(const T* x, T* hx, int ld, int columns) const override
    {
        ModuleBase::timer::start("DeviceDavidsonBridge", "hpsi");
        const std::size_t elements = static_cast<std::size_t>(ld) * columns;
        if (elements > input_.size()) { throw std::invalid_argument("LR Davidson H staging capacity exceeded"); }
        if (elements > 0)
        {
            base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
                input_.data(), x, elements);
        }
        host_operator_.hpsi(input_.data(), output_.data(), ld, columns);
        if (elements > 0)
        {
            base_device::memory::synchronize_memory_op<T, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
                hx, output_.data(), elements);
        }
        h_bytes_ += 2 * elements * sizeof(T);
        ModuleBase::timer::end("DeviceDavidsonBridge", "hpsi");
    }

    void spsi(const T* x, T* sx, int ld, int columns) const override
    {
        // The excitation-space metric is the identity, even though AO orbitals overlap.
        const std::size_t elements = static_cast<std::size_t>(ld) * columns;
        if (elements > 0)
        {
            base_device::memory::synchronize_memory_op<T, base_device::DEVICE_GPU, base_device::DEVICE_GPU>()(
                sx, x, elements);
        }
    }

    void precondition(T* residuals, const double* ritz_values, int ld, int columns) const
    {
        ModuleBase::timer::start("DeviceDavidsonBridge", "precondition");
        const std::size_t elements = static_cast<std::size_t>(ld) * columns;
        const int shift_capacity = shifts_.size();
        if (elements > input_.size() || columns > shift_capacity)
        { throw std::invalid_argument("LR Davidson inverse staging capacity exceeded"); }
        if (elements > 0)
        {
            base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
                input_.data(), residuals, elements);
        }
        base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
            shifts_.data(), ritz_values, columns);
        // Preserve the same compacted root order for residuals and Ritz shifts.
        // This boundary also supports the non-diagonal LR inverse callbacks.
        host_inverse_(input_.data(), shifts_.data(), ld, columns);
        if (elements > 0)
        {
            base_device::memory::synchronize_memory_op<T, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
                residuals, input_.data(), elements);
        }
        inverse_bytes_ += 2 * elements * sizeof(T) + columns * sizeof(double);
        ModuleBase::timer::end("DeviceDavidsonBridge", "precondition");
    }

    std::int64_t h_bytes() const { return h_bytes_; }
    std::int64_t inverse_bytes() const { return inverse_bytes_; }

private:
    const hsolver::HSOperator<T>& host_operator_;
    const HostInverse& host_inverse_;
    mutable std::vector<T> input_;
    mutable std::vector<T> output_;
    mutable std::vector<double> shifts_;
    mutable std::int64_t h_bytes_ = 0;
    mutable std::int64_t inverse_bytes_ = 0;
};
#endif
} // namespace

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
    bool use_gpu)
{
    ModuleBase::timer::start("LR", "solve_davidson");
    if (settings.roots <= 0 || settings.local_dimension < 0
        || diagonal.size() != static_cast<std::size_t>(settings.local_dimension) || !host_inverse)
    { throw std::invalid_argument("Invalid LR Davidson dimensions or inverse action"); }
    const int initial_columns = options.initial_vectors == 0 ? settings.roots : options.initial_vectors;
    const int basis_capacity = settings.roots * settings.subspace_multiplier;
    if (initial_columns < settings.roots || initial_columns > basis_capacity)
    { throw std::invalid_argument("Invalid LR Davidson initial column count"); }
    hsolver::DavidsonResult result;
    const char* backend = use_gpu ? "GPU" : "CPU";
    std::cout << "LR Davidson backend=" << backend << '\n';
    if (use_gpu)
    {
#if defined(__CUDA) || defined(__ROCM)
        ModuleBase::createGpuBlasHandle();
        ct::kernels::createGpuSolverHandle();
        const std::int64_t elements = static_cast<std::int64_t>(settings.local_dimension) * initial_columns;
        const std::int64_t capacity = std::max<std::int64_t>(1, elements);
        const auto data_type = ct::DataTypeToEnum<T>::value;
        ct::Tensor vectors(data_type, ct::DeviceType::GpuDevice, {capacity});
        if (elements > 0)
        {
            base_device::memory::synchronize_memory_op<T, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
                vectors.data<T>(), host_vectors, elements);
        }
        const DeviceDavidsonBridge<T> bridge(host_operator, host_inverse, elements, settings.roots);
        const typename hsolver::Diago_DavSubspace<T, base_device::DEVICE_GPU>::Preconditioner inverse =
            [&bridge](T* residuals, const double* ritz_values, int ld, int active_columns)
            { bridge.precondition(residuals, ritz_values, ld, active_columns); };
        result = run_davidson<T, base_device::DEVICE_GPU>(bridge, vectors.data<T>(), eigenvalues,
                                                         diagonal, inverse, options, settings, comm);
        const std::int64_t returned_elements = static_cast<std::int64_t>(settings.local_dimension) * settings.roots;
        if (returned_elements > 0)
        {
            base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
                host_vectors, vectors.data<T>(), returned_elements);
        }
        std::cout << "LR Davidson GPU host bridge: H bytes=" << bridge.h_bytes()
                  << "; inverse bytes=" << bridge.inverse_bytes() << '\n';
#else
        throw std::runtime_error("GPU LR Davidson requires a GPU-enabled build");
#endif
    }
    else
    {
        result = run_davidson<T, base_device::DEVICE_CPU>(host_operator, host_vectors, eigenvalues,
                                                         diagonal, host_inverse, options, settings, comm);
    }
    ModuleBase::timer::end("LR", "solve_davidson");
    return result;
}

template hsolver::DavidsonResult solve_davidson<double>(
    const hsolver::HSOperator<double>&, double*, double*, const std::vector<double>&,
    const hsolver::Diago_DavSubspace<double>::Preconditioner&, const hsolver::DavidsonOptions&,
    const DavidsonSolveSettings&, const hsolver::diag_comm_info&, bool);
template hsolver::DavidsonResult solve_davidson<std::complex<double>>(
    const hsolver::HSOperator<std::complex<double>>&, std::complex<double>*, double*, const std::vector<double>&,
    const hsolver::Diago_DavSubspace<std::complex<double>>::Preconditioner&, const hsolver::DavidsonOptions&,
    const DavidsonSolveSettings&, const hsolver::diag_comm_info&, bool);
} // namespace LR
