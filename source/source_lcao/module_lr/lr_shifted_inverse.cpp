#include "lr_shifted_inverse.h"

#include "lr_preconditioner.h"
#include "source_base/parallel_device.h"
#include "source_hsolver/diag_comm_info.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

namespace LR
{
namespace
{
double conjugate(double x) { return x; }
std::complex<double> conjugate(std::complex<double> x) { return std::conj(x); }

template<typename T>
T dot(const T* x, const T* y, int dim, const hsolver::diag_comm_info& comm)
{
    T sum = 0;
    for (int i = 0; i < dim; ++i) { sum += conjugate(x[i]) * y[i]; }
#ifdef __MPI
    if (comm.nproc > 1) { Parallel_Common::reduce_data(&sum, 1, comm.comm); }
#endif
    return sum;
}

template<typename T>
double norm(const T* x, int dim, const hsolver::diag_comm_info& comm)
{
    const double squared = std::real(dot(x, x, dim, comm));
    return std::sqrt(squared);
}

template<typename T>
void rotate(double cosine, T sine, T& upper, T& lower)
{
    const T first = cosine * upper + sine * lower;
    lower = -conjugate(sine) * upper + cosine * lower;
    upper = first;
}
}

template<typename T>
ShiftedInverseResult apply_shifted_inverse(
    T* rhs, int dim, double shift, const std::vector<double>& gaps,
    const std::function<void(const T*, T*)>& proxy,
    const ShiftedInverseSettings& settings, const hsolver::diag_comm_info& comm)
{
    if (dim < 0 || gaps.size() != static_cast<std::size_t>(dim) || !proxy
        || settings.max_iterations < 1 || !(settings.relative_tolerance > 0)
        || !(settings.relative_tolerance < 1) || !(settings.denominator_floor > 0))
    {
        throw std::invalid_argument("Invalid LR shifted inverse settings");
    }
    ShiftedInverseResult result;
    const double rhs_norm = norm(rhs, dim, comm);
    if (rhs_norm == 0) { result.converged = true; return result; }
    const int steps = settings.max_iterations;
    const int stride = std::max(1, dim);
    const int rows = steps + 1;
    std::vector<T> basis(static_cast<std::size_t>(stride) * rows, T(0));
    std::vector<T> corrections(static_cast<std::size_t>(stride) * steps, T(0));
    std::vector<T> triangular(static_cast<std::size_t>(rows) * steps, T(0));
    std::vector<T> image(stride);
    std::vector<T> solution(stride, T(0));
    std::vector<T> projected_rhs(rows, T(0));
    std::vector<double> cosines(steps);
    std::vector<T> sines(steps);
    projected_rhs[0] = rhs_norm;
    for (int i = 0; i < dim; ++i) { basis[i] = rhs[i] / rhs_norm; }
    int count = 0;
    for (int j = 0; j < steps; ++j)
    {
        T* correction = corrections.data() + j * stride;
        std::copy_n(basis.data() + j * stride, dim, correction);
        apply_orbital_gap_preconditioner(correction, &shift, dim, 1, gaps, settings.denominator_floor);
        proxy(correction, image.data());
        ++result.operator_vectors;
        for (int i = 0; i < dim; ++i) { image[i] -= shift * correction[i]; }
        const double before = norm(image.data(), dim, comm);
        T* column = triangular.data() + j * rows;
        // Twice-modified Gram-Schmidt protects near-pole/dependent Krylov directions.
        for (int pass = 0; pass < 2; ++pass)
        {
            for (int k = 0; k <= j; ++k)
            {
                const T* previous = basis.data() + k * stride;
                const T overlap = dot(previous, image.data(), dim, comm);
                column[k] += overlap;
                for (int i = 0; i < dim; ++i) { image[i] -= overlap * previous[i]; }
            }
        }
        const double remainder = norm(image.data(), dim, comm);
        column[j + 1] = remainder;
        const double breakdown_tolerance = 64 * std::numeric_limits<double>::epsilon() * before;
        const bool dependent = remainder <= breakdown_tolerance;
        if (!dependent)
        {
            T* next = basis.data() + (j + 1) * stride;
            for (int i = 0; i < dim; ++i) { next[i] = image[i] / remainder; }
        }
        for (int k = 0; k < j; ++k) { rotate(cosines[k], sines[k], column[k], column[k + 1]); }
        const double upper_norm = std::abs(column[j]);
        const double lower_norm = std::abs(column[j + 1]);
        const double rotation_norm = std::hypot(upper_norm, lower_norm);
        if (rotation_norm == 0 || !std::isfinite(rotation_norm)) { break; }
        const T phase = upper_norm == 0 ? T(1) : column[j] / upper_norm;
        cosines[j] = upper_norm / rotation_norm;
        sines[j] = phase * conjugate(column[j + 1]) / rotation_norm;
        rotate(cosines[j], sines[j], column[j], column[j + 1]);
        rotate(cosines[j], sines[j], projected_rhs[j], projected_rhs[j + 1]);
        count = j + 1;
        const double estimated_residual = std::abs(projected_rhs[j + 1]) / rhs_norm;
        if (dependent || estimated_residual <= settings.relative_tolerance) { break; }
    }
    // The small least-squares solution gives z = sum_j M^{-1} v_j y_j.
    for (int j = count - 1; j >= 0; --j)
    {
        for (int k = j + 1; k < count; ++k) { projected_rhs[j] -= triangular[k * rows + j] * projected_rhs[k]; }
        projected_rhs[j] /= triangular[j * rows + j];
        const T* correction = corrections.data() + j * stride;
        for (int i = 0; i < dim; ++i) { solution[i] += projected_rhs[j] * correction[i]; }
    }
    proxy(solution.data(), image.data());
    ++result.operator_vectors;
    for (int i = 0; i < dim; ++i) { image[i] -= shift * solution[i] + rhs[i]; }
    result.relative_residual = norm(image.data(), dim, comm) / rhs_norm;
    result.converged = result.relative_residual <= settings.relative_tolerance;
    result.used_gap_fallback = !std::isfinite(result.relative_residual) || result.relative_residual >= 1.0;
    if (result.used_gap_fallback)
    {
        apply_orbital_gap_preconditioner(rhs, &shift, dim, 1, gaps, settings.denominator_floor);
    }
    else { std::copy_n(solution.data(), dim, rhs); }
    return result;
}

template ShiftedInverseResult apply_shifted_inverse(double*, int, double, const std::vector<double>&,
    const std::function<void(const double*, double*)>&, const ShiftedInverseSettings&, const hsolver::diag_comm_info&);
template ShiftedInverseResult apply_shifted_inverse(std::complex<double>*, int, double, const std::vector<double>&,
    const std::function<void(const std::complex<double>*, std::complex<double>*)>&,
    const ShiftedInverseSettings&, const hsolver::diag_comm_info&);
}
