#include "diago_dav_subspace.h"

#include "source_base/kernels/math_kernel_op.h"
#include "source_base/module_container/ATen/core/tensor.h"
#include "source_base/parallel_device.h"
#include "source_base/timer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hsolver
{
const char* davidson_status_name(DavidsonStatus status)
{
    switch (status)
    {
    case DavidsonStatus::converged: return "converged";
    case DavidsonStatus::iteration_limit: return "iteration limit";
    case DavidsonStatus::dependent_initial_vectors: return "dependent initial vectors";
    case DavidsonStatus::no_independent_direction: return "no independent expansion direction";
    case DavidsonStatus::nonfinite_residual: return "nonfinite residual";
    }
    return "unknown";
}

namespace
{
template<typename T, typename Device>
ct::Tensor device_buffer(int count)
{
    using ContainerDevice = typename ct::PsiToContainer<Device>::type;
    const auto device = ct::DeviceTypeToEnum<ContainerDevice>::value;
    const int allocated = std::max(1, count);
    return ct::Tensor(ct::DataTypeToEnum<T>::value, device, {allocated});
}

// Borrow the S-orthonormal basis V and SV. The caller owns both
// storage and the active column count; algebra helpers do not change iteration state.
template<typename T>
struct DavidsonBasis
{
    int dim;
    T* vectors;
    T* s_images;
    const diag_comm_info& comm;
};

template<typename T, typename Device>
double global_inner_real(const T* left, const T* right, int dim, const diag_comm_info& comm)
{
    double value = 0.0;
    if (dim > 0) { value = ModuleBase::dot_real_op<T, Device>()(dim, left, right, false); }
#ifdef __MPI
    if (comm.nproc > 1) { Parallel_Common::reduce_data(&value, 1, comm.comm); }
#endif
    return value;
}

// Form X = V C (also used for HX = (HV)C and SX = (SV)C). C has leading
// dimension coefficient_stride; all large buffers stay on Device.
template<typename T, typename Device>
void rotate_block(const T* vectors, int dim, int basis_count, const T* coefficients,
                  int coefficient_stride, int roots, T* output)
{
    if (dim == 0) { return; }
    const T one = 1;
    const T zero = 0;
    ModuleBase::gemm_op<T, Device>()('N', 'N', dim, roots, basis_count, &one,
                                    vectors, dim, coefficients, coefficient_stride, &zero, output, dim);
}

// Two-pass S-Gram-Schmidt: t <- t - V(V† S t), with the same update applied
// to St. Check the remaining norm BEFORE dividing; a dependent t is not a
// statement about convergence of the original eigenpair.
template<typename T, typename Device>
bool orthonormalize_direction(const DavidsonBasis<T>& basis, int basis_count,
                             T* direction, T* s_direction, T* device_overlaps)
{
    using Real = typename GetTypeReal<T>::type;
    const double original_squared = global_inner_real<T, Device>(direction, s_direction, basis.dim, basis.comm);
    if (!std::isfinite(original_squared) || original_squared <= 0.0) { return false; }
    std::vector<T> overlaps(basis_count);
    const T one = 1;
    const T zero = 0;
    const T minus_one = -1;
    for (int pass = 0; pass < 2 && basis_count > 0; ++pass)
    {
        if (basis.dim > 0)
        {
            ModuleBase::gemv_op<T, Device>()('C', basis.dim, basis_count, &one, basis.vectors,
                                            basis.dim, s_direction, 1, &zero, device_overlaps, 1);
            base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, Device>()(
                overlaps.data(), device_overlaps, basis_count);
        }
#ifdef __MPI
        if (basis.comm.nproc > 1) { Parallel_Common::reduce_data(overlaps.data(), basis_count, basis.comm.comm); }
#endif
        if (basis.dim > 0)
        {
            base_device::memory::synchronize_memory_op<T, Device, base_device::DEVICE_CPU>()(
                device_overlaps, overlaps.data(), basis_count);
            ModuleBase::gemv_op<T, Device>()('N', basis.dim, basis_count, &minus_one, basis.vectors,
                                            basis.dim, device_overlaps, 1, &one, direction, 1);
            ModuleBase::gemv_op<T, Device>()('N', basis.dim, basis_count, &minus_one, basis.s_images,
                                            basis.dim, device_overlaps, 1, &one, s_direction, 1);
        }
    }
    const double remaining_squared = global_inner_real<T, Device>(direction, s_direction, basis.dim, basis.comm);
    const double precision_cutoff = 64.0 * std::numeric_limits<Real>::epsilon();
    const double relative_cutoff = std::max(1e-12, precision_cutoff);
    const double minimum_squared = original_squared * relative_cutoff * relative_cutoff;
    if (!std::isfinite(remaining_squared) || remaining_squared <= minimum_squared) { return false; }
    const Real norm = std::sqrt(remaining_squared);
    ModuleBase::vector_div_constant_op<T, Device>()(basis.dim, direction, direction, norm);
    ModuleBase::vector_div_constant_op<T, Device>()(basis.dim, s_direction, s_direction, norm);
    return true;
}
} // namespace

template<typename T, typename Device>
DavidsonResult Diago_DavSubspace<T, Device>::diag(const HSOperator<T, Device>& op,
                                                T* psi_in,
                                                int psi_in_dmax,
                                                Real* eigenvalues,
                                                const std::vector<double>& energy_tolerances,
                                                const Preconditioner& preconditioner,
                                                const DavidsonOptions& options)
{
    ModuleBase::timer::start("Diago_DavSubspace", "diag");
    const int initial_count = options.initial_vectors == 0 ? n_band : options.initial_vectors;
    const int retained_count = options.retained_vectors == 0 ? n_band : options.retained_vectors;
    if (!preconditioner || !std::isfinite(options.residual_tolerance) || options.residual_tolerance <= 0.0
        || psi_in_dmax < dim || energy_tolerances.size() != static_cast<std::size_t>(n_band)
        || initial_count < n_band || initial_count > nbase_x
        || retained_count < n_band || retained_count + n_band > nbase_x)
    {
        throw std::invalid_argument("Invalid residual-controlled Davidson arguments");
    }
    for (double tolerance : energy_tolerances)
    {
        if (!std::isfinite(tolerance) || tolerance <= 0.0)
        {
            throw std::invalid_argument("Davidson energy tolerances must be finite and positive");
        }
    }

    DavidsonResult result;
    result.residual_norms.assign(n_band, std::numeric_limits<double>::infinity());
    const DavidsonBasis<T> basis{dim, psi_in_iter, spsi, diag_comm};
    const int work_columns = std::max(initial_count, retained_count);
    const int block_size = dim * work_columns;
    const int workspace_size = 6 * block_size + nbase_x;
    ct::Tensor workspace = device_buffer<T, Device>(workspace_size);
    T* ritz_vectors = workspace.template data<T>();
    T* ritz_h = ritz_vectors + block_size;
    T* ritz_s = ritz_h + block_size;
    T* raw_residuals = ritz_s + block_size;
    T* corrections = raw_residuals + block_size;
    T* s_corrections = corrections + block_size;
    T* overlaps = s_corrections + block_size;
    ct::Tensor shifts_buffer = device_buffer<Real, Device>(n_band);
    std::vector<Real> ritz_values(nbase_x);
    std::vector<Real> previous_values(n_band);

    // Accept exact initial eigenvectors, and reject duplicate guesses before a
    // generalized reduced solve could see a singular V† S V.
    syncmem_complex_2d_op()(corrections, dim, psi_in, psi_in_dmax, dim, initial_count);
    op.spsi(corrections, s_corrections, dim, initial_count);
    int initial_rank = 0;
    for (int root = 0; root < initial_count; ++root)
    {
        T* direction = corrections + root * dim;
        T* s_direction = s_corrections + root * dim;
        const bool independent = orthonormalize_direction<T, Device>(basis, initial_rank, direction, s_direction, overlaps);
        if (!independent) { continue; }
        T* basis_column = psi_in_iter + initial_rank * dim;
        T* s_column = spsi + initial_rank * dim;
        syncmem_complex_op()(basis_column, direction, dim);
        syncmem_complex_op()(s_column, s_direction, dim);
        ++initial_rank;
    }
    if (initial_rank < n_band)
    {
        result.status = DavidsonStatus::dependent_initial_vectors;
        ModuleBase::timer::end("Diago_DavSubspace", "diag");
        return result;
    }
    op.hpsi(psi_in_iter, hpsi, dim, initial_rank);
    int basis_count = 0;
    cal_elem(dim, basis_count, initial_rank, psi_in_iter, spsi, hpsi, hcc, scc);
    bool first_projection = true;
    bool solve_projected = true;

    while (true)
    {
        const int ritz_count = std::min(retained_count, basis_count);
        if (solve_projected)
        {
            diag_zhegvx(basis_count, ritz_count, hcc, scc, nbase_x, &ritz_values, vcc);
            rotate_block<T, Device>(psi_in_iter, dim, basis_count, vcc, nbase_x, ritz_count, ritz_vectors);
            rotate_block<T, Device>(hpsi, dim, basis_count, vcc, nbase_x, ritz_count, ritz_h);
            rotate_block<T, Device>(spsi, dim, basis_count, vcc, nbase_x, ritz_count, ritz_s);
        }
        std::vector<int> unconverged;
        std::vector<Real> shifts;
        result.converged_roots = 0;
        bool finite = true;
        for (int root = 0; root < n_band; ++root)
        {
            T* residual = raw_residuals + root * dim;
            const T* h_image = ritz_h + root * dim;
            const T* s_image = ritz_s + root * dim;
            const Real one = 1;
            const Real minus_shift = -ritz_values[root];
            ModuleBase::vector_add_vector_op<T, Device, Real>()(dim, residual, h_image, one, s_image, minus_shift);
            const double squared_norm = global_inner_real<T, Device>(residual, residual, dim, diag_comm);
            const double residual_norm = std::sqrt(squared_norm);
            result.residual_norms[root] = residual_norm;
            finite = finite && std::isfinite(residual_norm) && std::isfinite(ritz_values[root]);
            const bool residual_converged = residual_norm <= options.residual_tolerance;
            const bool energy_converged = first_projection
                || std::abs(ritz_values[root] - previous_values[root]) < energy_tolerances[root];
            if (residual_converged && energy_converged) { ++result.converged_roots; }
            if (!residual_converged)
            {
                // Keep the original root index: compaction must apply the same
                // permutation to residual columns and their Ritz shifts.
                T* correction = corrections + unconverged.size() * dim;
                syncmem_complex_op()(correction, residual, dim);
                unconverged.push_back(root);
                shifts.push_back(ritz_values[root]);
            }
            previous_values[root] = ritz_values[root];
            eigenvalues[root] = ritz_values[root];
        }
        syncmem_complex_2d_op()(psi_in, psi_in_dmax, ritz_vectors, dim, dim, n_band);
        first_projection = false;
        if (!finite) { result.status = DavidsonStatus::nonfinite_residual; break; }
        if (result.converged_roots == n_band) { result.status = DavidsonStatus::converged; break; }
        if (unconverged.empty())
        {
            // Every residual already passes. Recheck the unchanged Ritz values
            // once to settle the energy-change test, without normalizing zeros
            // or spending another H application.
            solve_projected = false;
            continue;
        }
        if (result.iterations >= iter_nmax) { result.status = DavidsonStatus::iteration_limit; break; }
        solve_projected = true;
        const int active_roots = unconverged.size();
        if (basis_count + active_roots > nbase_x)
        {
            // Soft locking: keep the requested roots AND extra low Ritz vectors.
            // A candidate just above the output cutoff can move below it later;
            // dropping it at every restart makes clustered/low roots easier to miss.
            // Cached HX/SX rotate with X, so restart needs no new H action.
            const int retained_size = ritz_count * dim;
            syncmem_complex_op()(psi_in_iter, ritz_vectors, retained_size);
            syncmem_complex_op()(hpsi, ritz_h, retained_size);
            syncmem_complex_op()(spsi, ritz_s, retained_size);
            basis_count = 0;
            cal_elem(dim, basis_count, ritz_count, psi_in_iter, spsi, hpsi, hcc, scc);
            ++result.restarts;
        }
        Real* device_shifts = shifts_buffer.template data<Real>();
        syncmem_var_h2d_op()(device_shifts, shifts.data(), active_roots);
        preconditioner(corrections, device_shifts, dim, active_roots);
        op.spsi(corrections, s_corrections, dim, active_roots);
        int added_directions = 0;
        for (int column = 0; column < active_roots; ++column)
        {
            T* direction = corrections + column * dim;
            T* s_direction = s_corrections + column * dim;
            const int existing_columns = basis_count + added_directions;
            bool independent = orthonormalize_direction<T, Device>(basis, existing_columns, direction, s_direction, overlaps);
            if (!independent)
            {
                ++result.rejected_directions;
                // A poor/singular inverse can erase r or send it back into V.
                // Try the unpreconditioned residual, whose convergence was
                // checked above; never infer convergence from ||M^-1 r||.
                const int root = unconverged[column];
                const T* residual = raw_residuals + root * dim;
                syncmem_complex_op()(direction, residual, dim);
                op.spsi(direction, s_direction, dim, 1);
                independent = orthonormalize_direction<T, Device>(basis, existing_columns, direction, s_direction, overlaps);
                if (independent) { ++result.residual_fallbacks; }
            }
            if (!independent) { continue; }
            T* basis_column = psi_in_iter + existing_columns * dim;
            T* s_column = spsi + existing_columns * dim;
            syncmem_complex_op()(basis_column, direction, dim);
            syncmem_complex_op()(s_column, s_direction, dim);
            ++added_directions;
        }
        if (added_directions == 0) { result.status = DavidsonStatus::no_independent_direction; break; }
        // Only independent columns consume H applications or enlarge Hcc/Scc.
        // Their count can be smaller than the number of unconverged roots.
        const T* new_vectors = psi_in_iter + basis_count * dim;
        T* new_h_images = hpsi + basis_count * dim;
        op.hpsi(new_vectors, new_h_images, dim, added_directions);
        cal_elem(dim, basis_count, added_directions, psi_in_iter, spsi, hpsi, hcc, scc);
        ++result.iterations;
    }
    ModuleBase::timer::end("Diago_DavSubspace", "diag");
    return result;
}

// Keep the new policy available on the same scalar/device combinations as the
// existing solver. LR selects CPU/GPU for double and complex-double amplitudes.
template DavidsonResult Diago_DavSubspace<std::complex<double>, base_device::DEVICE_CPU>::diag(
    const HSOperator<std::complex<double>, base_device::DEVICE_CPU>&, std::complex<double>*, int, double*,
    const std::vector<double>&, const Preconditioner&, const DavidsonOptions&);
template DavidsonResult Diago_DavSubspace<std::complex<float>, base_device::DEVICE_CPU>::diag(
    const HSOperator<std::complex<float>, base_device::DEVICE_CPU>&, std::complex<float>*, int, float*,
    const std::vector<double>&, const Preconditioner&, const DavidsonOptions&);
#ifdef __LCAO
template DavidsonResult Diago_DavSubspace<double, base_device::DEVICE_CPU>::diag(
    const HSOperator<double, base_device::DEVICE_CPU>&, double*, int, double*,
    const std::vector<double>&, const Preconditioner&, const DavidsonOptions&);
#endif
#if defined(__CUDA) || defined(__ROCM)
template DavidsonResult Diago_DavSubspace<std::complex<double>, base_device::DEVICE_GPU>::diag(
    const HSOperator<std::complex<double>, base_device::DEVICE_GPU>&, std::complex<double>*, int, double*,
    const std::vector<double>&, const Preconditioner&, const DavidsonOptions&);
template DavidsonResult Diago_DavSubspace<std::complex<float>, base_device::DEVICE_GPU>::diag(
    const HSOperator<std::complex<float>, base_device::DEVICE_GPU>&, std::complex<float>*, int, float*,
    const std::vector<double>&, const Preconditioner&, const DavidsonOptions&);
#ifdef __LCAO
template DavidsonResult Diago_DavSubspace<double, base_device::DEVICE_GPU>::diag(
    const HSOperator<double, base_device::DEVICE_GPU>&, double*, int, double*,
    const std::vector<double>&, const Preconditioner&, const DavidsonOptions&);
#endif
#endif
} // namespace hsolver
