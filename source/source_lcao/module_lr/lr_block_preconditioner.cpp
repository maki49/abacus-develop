#include "lr_block_preconditioner.h"

#include "lr_preconditioner.h"
#include "source_base/module_external/lapack_connector.h"
#include "source_base/parallel_device.h"
#include "source_hsolver/diag_comm_info.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace LR
{
namespace
{
double conjugate(double value) { return value; }
std::complex<double> conjugate(std::complex<double> value) { return std::conj(value); }

void diagonalize_block(int size, double* matrix, double* eigenvalues)
{
    const char vectors = 'V';
    const char upper = 'U';
    const int minimum_workspace = 3 * size;
    const int workspace_size = std::max(1, minimum_workspace);
    std::vector<double> workspace(workspace_size);
    int info = 0;
    dsyev_(&vectors, &upper, &size, matrix, &size, eigenvalues, workspace.data(), &workspace_size, &info);
    if (info != 0) { throw std::runtime_error("LR preconditioner: dsyev failed"); }
}

void diagonalize_block(int size, std::complex<double>* matrix, double* eigenvalues)
{
    const char vectors = 'V';
    const char upper = 'U';
    const int minimum_workspace = 2 * size;
    const int minimum_real_workspace = 3 * size;
    const int workspace_size = std::max(1, minimum_workspace);
    const int real_workspace_size = std::max(1, minimum_real_workspace);
    std::vector<std::complex<double>> workspace(workspace_size);
    std::vector<double> real_workspace(real_workspace_size);
    int info = 0;
    zheev_(&vectors, &upper, &size, matrix, &size, eigenvalues, workspace.data(), &workspace_size,
           real_workspace.data(), &info);
    if (info != 0) { throw std::runtime_error("LR preconditioner: zheev failed"); }
}

template<typename T>
void sum_selected_entries(std::vector<T>& entries, const hsolver::diag_comm_info& comm)
{
#ifdef __MPI
    const int size = entries.size();
    if (comm.nproc > 1 && size > 0) { Parallel_Common::reduce_data(entries.data(), size, comm.comm); }
#endif
}
} // namespace

template<typename T>
LowEnergyPreconditioner<T> build_low_energy_preconditioner(
    const std::vector<int>& local_indices, int local_dim,
    const std::function<void(const T*, T*, int)>& apply_operator,
    const hsolver::diag_comm_info& comm)
{
    if (local_dim < 0) { throw std::invalid_argument("Negative LR amplitude dimension"); }
    // P is a set of distinct coordinate vectors, each owned by exactly one rank.
    // Check this before probing, to avoid silently multiplying replicated rows.
    const int count = local_indices.size();
    std::vector<double> owners(count, 0.0);
    std::vector<bool> seen(local_dim, false);
    for (int j = 0; j < count; ++j)
    {
        const int index = local_indices[j];
        if (index < -1 || index >= local_dim) { throw std::invalid_argument("Invalid LR block index"); }
        if (index < 0) { continue; }
        if (seen[index]) { throw std::invalid_argument("Duplicate LR block index"); }
        seen[index] = true;
        owners[j] = 1;
    }
    sum_selected_entries(owners, comm);
    for (double owners_count : owners)
    {
        if (owners_count != 1) { throw std::invalid_argument("LR block coordinate must have exactly one owner"); }
    }
    LowEnergyPreconditioner<T> block;
    block.local_indices = local_indices;
    block.eigenvalues.resize(count);
    const std::size_t matrix_size = static_cast<std::size_t>(count) * count;
    block.eigenvectors.resize(matrix_size, T(0));
    const int batch_size = 4;
    const std::size_t scratch_size = static_cast<std::size_t>(local_dim) * batch_size;
    const std::size_t allocated_size = std::max<std::size_t>(1, scratch_size);
    std::vector<T> probes(allocated_size);
    std::vector<T> images(allocated_size);
    for (int first = 0; first < count; first += batch_size)
    {
        const int remaining = count - first;
        const int columns = std::min(batch_size, remaining);
        std::fill(probes.begin(), probes.end(), T(0));
        std::fill(images.begin(), images.end(), T(0));
        for (int col = 0; col < columns; ++col)
        {
            const int index = local_indices[first + col];
            if (index >= 0) { probes[col * local_dim + index] = T(1); }
        }
        apply_operator(probes.data(), images.data(), columns);
        for (int col = 0; col < columns; ++col)
        {
            for (int row = 0; row < count; ++row)
            {
                const int index = local_indices[row];
                if (index >= 0) { block.eigenvectors[(first + col) * count + row] = images[col * local_dim + index]; }
            }
        }
    }
    sum_selected_entries(block.eigenvectors, comm);
    for (int col = 0; col < count; ++col)
    {
        for (int row = 0; row <= col; ++row)
        {
            const T upper = block.eigenvectors[col * count + row];
            const T lower = conjugate(block.eigenvectors[row * count + col]);
            const T difference = upper - lower;
            const double error = std::abs(difference);
            const double upper_magnitude = std::abs(upper);
            const double lower_magnitude = std::abs(lower);
            if (!std::isfinite(upper_magnitude) || !std::isfinite(lower_magnitude))
            {
                throw std::runtime_error("Nonfinite LR low-energy kernel block");
            }
            block.hermiticity_error = std::max(block.hermiticity_error, error);
            // Grid/RI screening may leave a small numerical anti-Hermitian part.
            // Symmetrize only the approximate inverse, never the physical A.
            const T value = (upper + lower) * 0.5;
            block.eigenvectors[col * count + row] = value;
            block.eigenvectors[row * count + col] = conjugate(value);
        }
    }
    if (count > 0) { diagonalize_block(count, block.eigenvectors.data(), block.eigenvalues.data()); }
    return block;
}

template<typename T>
void apply_low_energy_preconditioner(T* residuals, const double* shifts, int dim, int nvec,
                                    const std::vector<double>& gaps, double denominator_floor,
                                    const LowEnergyPreconditioner<T>& block,
                                    const hsolver::diag_comm_info& comm)
{
    const int count = block.local_indices.size();
    const std::size_t matrix_size = static_cast<std::size_t>(count) * count;
    if (dim < 0 || nvec < 0 || gaps.size() != static_cast<std::size_t>(dim)
        || block.eigenvalues.size() != count || block.eigenvectors.size() != matrix_size)
    {
        throw std::invalid_argument("Invalid LR block preconditioner dimensions");
    }
    // Gather the ORIGINAL r_P before overwriting the residual with the gap inverse.
    const std::size_t selected_size = static_cast<std::size_t>(count) * nvec;
    std::vector<T> selected_residuals(selected_size, T(0));
    for (int j = 0; j < count; ++j)
    {
        const int index = block.local_indices[j];
        if (index < -1 || index >= dim) { throw std::invalid_argument("Invalid LR block amplitude index"); }
        if (index < 0) { continue; }
        for (int n = 0; n < nvec; ++n) { selected_residuals[n * count + j] = residuals[n * dim + index]; }
    }
    sum_selected_entries(selected_residuals, comm);
    apply_orbital_gap_preconditioner(residuals, shifts, dim, nvec, gaps, denominator_floor);
    std::vector<T> spectral_residuals(selected_size, T(0));
    for (int n = 0; n < nvec; ++n)
    {
        for (int mode = 0; mode < count; ++mode)
        {
            for (int j = 0; j < count; ++j)
            {
                const T coefficient = conjugate(block.eigenvectors[mode * count + j]);
                spectral_residuals[n * count + mode] += coefficient * selected_residuals[n * count + j];
            }
        }
    }
    // Reuse precisely the same signed pole regularization in the block's
    // eigenbasis; diagonal and block limits therefore agree, including poles.
    apply_orbital_gap_preconditioner(spectral_residuals.data(), shifts, count, nvec,
                                     block.eigenvalues, denominator_floor);
    for (int j = 0; j < count; ++j)
    {
        const int index = block.local_indices[j];
        if (index < 0) { continue; }
        for (int n = 0; n < nvec; ++n)
        {
            T correction = T(0);
            for (int mode = 0; mode < count; ++mode)
            {
                correction += block.eigenvectors[mode * count + j] * spectral_residuals[n * count + mode];
            }
            residuals[n * dim + index] = correction;
        }
    }
}

template LowEnergyPreconditioner<double> build_low_energy_preconditioner(
    const std::vector<int>&, int, const std::function<void(const double*, double*, int)>&,
    const hsolver::diag_comm_info&);
template LowEnergyPreconditioner<std::complex<double>> build_low_energy_preconditioner(
    const std::vector<int>&, int,
    const std::function<void(const std::complex<double>*, std::complex<double>*, int)>&,
    const hsolver::diag_comm_info&);
template void apply_low_energy_preconditioner(double*, const double*, int, int, const std::vector<double>&,
                                             double, const LowEnergyPreconditioner<double>&,
                                             const hsolver::diag_comm_info&);
template void apply_low_energy_preconditioner(std::complex<double>*, const double*, int, int,
                                             const std::vector<double>&, double,
                                             const LowEnergyPreconditioner<std::complex<double>>&,
                                             const hsolver::diag_comm_info&);
} // namespace LR
