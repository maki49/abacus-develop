#include "linear_pcg.h"

#include "linear_algebra.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/parallel_device.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hsolver
{
namespace
{
void reduce_products(std::vector<double>& products, const diag_comm_info& comm)
{
#ifdef __MPI
    if (comm.nproc > 1)
    {
        const int count = static_cast<int>(products.size());
        Parallel_Common::reduce_data(products.data(), count, comm.comm);
    }
#endif
}

template<typename Device>
std::vector<double> products(const double* a, const double* b, int ld, int dim,
                             int columns, const diag_comm_info& comm)
{
    std::vector<double> values(columns, 0.0);
    const bool reduce_in_kernel = false;
    if (dim > 0)
    {
        for (int column = 0; column < columns; ++column)
        {
            const int offset = column * ld;
            const double* a_column = a + offset;
            const double* b_column = b + offset;
            values[column] = ModuleBase::dot_real_op<double, Device>()(
                dim, a_column, b_column, reduce_in_kernel);
        }
    }
    reduce_products(values, comm);
    return values;
}
}

template<typename Device>
LinearSolveResult linear_pcg(const LinearOperator<double, Device>& op,
                             const LinearOperator<double, Device>& inverse,
                             double* x, const double* rhs, int ld, int dim,
                             int columns, double tolerance, int max_iterations,
                             const diag_comm_info& comm)
{
    if (dim < 0 || ld < dim || columns < 0 || max_iterations < 0
        || !std::isfinite(tolerance) || tolerance <= 0.0)
    {
        throw std::invalid_argument("Invalid linear PCG dimensions or convergence policy");
    }
    LinearSolveResult result;
    result.status = LinearSolveStatus::converged;
    if (columns == 0) { return result; }
    ct::Tensor residual;
    ct::Tensor direction;
    ct::Tensor image;
    ct::Tensor preconditioned;
    const int64_t size = static_cast<int64_t>(ld) * columns;
    linear_buffer<double, Device>(&residual, size);
    linear_buffer<double, Device>(&direction, size);
    linear_buffer<double, Device>(&image, size);
    linear_buffer<double, Device>(&preconditioned, size);
    residual.zero();
    direction.zero();
    image.zero();
    preconditioned.zero();
    double* r = residual.data<double>();
    double* p = direction.data<double>();
    double* q = image.data<double>();
    double* z = preconditioned.data<double>();
    std::vector<double> rho(columns, 0.0);
    std::vector<bool> active(columns, true);
    bool restart = true;
    const auto apply = [&](const double* input, double* output)
    {
        op.apply(input, output, ld, columns);
        ++result.operator_calls;
        result.operator_columns += columns;
    };
    const auto true_residual = [&]()
    {
        apply(x, q);
        if (dim > 0)
        {
            for (int column = 0; column < columns; ++column)
            {
                const int offset = column * ld;
                double* r_column = r + offset;
                const double* rhs_column = rhs + offset;
                const double* q_column = q + offset;
                ModuleBase::vector_add_vector_op<double, Device, double>()(
                    dim, r_column, rhs_column, 1.0, q_column, -1.0);
            }
        }
        ++result.true_checks;
    };
    const auto check = [&]()
    {
        const auto norms = products<Device>(r, r, ld, dim, columns, comm);
        result.max_residual = 0.0;
        result.failed_band = -1;
        for (int column = 0; column < columns; ++column)
        {
            if (!std::isfinite(norms[column]) || norms[column] < 0.0)
            {
                result.status = LinearSolveStatus::breakdown;
                result.max_residual = std::numeric_limits<double>::infinity();
                result.failed_band = column;
                return false;
            }
            const double norm = std::sqrt(norms[column]);
            result.max_residual = std::max(result.max_residual, norm);
            active[column] = norm > tolerance;
            if (active[column] && result.failed_band < 0) { result.failed_band = column; }
        }
        return result.failed_band < 0;
    };
    true_residual();
    if (check() || result.status == LinearSolveStatus::breakdown) { return result; }
    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
        std::vector<double> failure(1, 0.0);
        try
        {
            inverse.apply(r, z, ld, columns);
        }
        catch (const LinearPreconditionerError&)
        {
            failure[0] = 1.0;
        }
        reduce_products(failure, comm);
        if (failure[0] > 0.0)
        {
            result.status = LinearSolveStatus::preconditioner_failure;
            return result;
        }
        const auto next_rho = products<Device>(r, z, ld, dim, columns, comm);
        for (int column = 0; column < columns; ++column)
        {
            const int offset = column * ld;
            double* p_column = p + offset;
            const double* z_column = z + offset;
            if (!active[column])
            {
                if (dim > 0) { base_device::memory::set_memory_op<double, Device>()(p_column, 0, dim); }
                continue;
            }
            if (!std::isfinite(next_rho[column]) || next_rho[column] <= 0.0)
            {
                result.status = LinearSolveStatus::preconditioner_failure;
                result.failed_band = column;
                return result;
            }
            const double beta = restart ? 0.0 : next_rho[column] / rho[column];
            if (dim > 0)
            {
                ModuleBase::vector_add_vector_op<double, Device, double>()(
                    dim, p_column, z_column, 1.0, p_column, beta);
            }
        }
        rho = next_rho;
        restart = false;
        apply(p, q);
        const auto curvature = products<Device>(p, q, ld, dim, columns, comm);
        for (int column = 0; column < columns; ++column)
        {
            if (!active[column]) { continue; }
            if (!std::isfinite(curvature[column]) || curvature[column] <= 0.0)
            {
                result.status = LinearSolveStatus::breakdown;
                result.failed_band = column;
                return result;
            }
            const double alpha = rho[column] / curvature[column];
            const double negative_alpha = -alpha;
            const int offset = column * ld;
            if (dim > 0)
            {
                double* x_column = x + offset;
                double* r_column = r + offset;
                const double* p_column = p + offset;
                const double* q_column = q + offset;
                ModuleBase::vector_add_vector_op<double, Device, double>()(
                    dim, x_column, x_column, 1.0, p_column, alpha);
                ModuleBase::vector_add_vector_op<double, Device, double>()(
                    dim, r_column, r_column, 1.0, q_column, negative_alpha);
            }
        }
        result.iterations = iteration + 1;
        const bool converged = check();
        if (result.status == LinearSolveStatus::breakdown) { return result; }
        const bool last = result.iterations == max_iterations;
        if (converged || last)
        {
            true_residual();
            if (check())
            {
                result.status = LinearSolveStatus::converged;
                return result;
            }
            if (result.status == LinearSolveStatus::breakdown) { return result; }
            restart = true;
            ++result.restarts;
        }
    }
    result.status = LinearSolveStatus::max_iterations;
    return result;
}

template LinearSolveResult linear_pcg<base_device::DEVICE_CPU>(
    const LinearOperator<double, base_device::DEVICE_CPU>&,
    const LinearOperator<double, base_device::DEVICE_CPU>&,
    double*, const double*, int, int, int, double, int, const diag_comm_info&);
#if defined(__CUDA) || defined(__ROCM)
template LinearSolveResult linear_pcg<base_device::DEVICE_GPU>(
    const LinearOperator<double, base_device::DEVICE_GPU>&,
    const LinearOperator<double, base_device::DEVICE_GPU>&,
    double*, const double*, int, int, int, double, int, const diag_comm_info&);
#endif
}
