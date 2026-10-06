#include "exx_projection.h"

#include "source_base/module_external/blas_connector.h"
#include "source_base/module_container/ATen/core/tensor.h"
#include "source_base/module_device/memory_op.h"
#include <stdexcept>

namespace LR
{
namespace
{
void project_on_device(const double* h,
                 const double* left,
                 const double* right,
                 const int naos,
                 const int nleft,
                 const int nright,
                 const double factor,
                 double* scratch,
                 double* result,
                 const base_device::AbacusDevice_t device)
{
    if (nleft == 0 || nright == 0) { return; }
    const double one = 1.0;
    const double zero = 0.0;
    // Transpose, rather than symmetrize: K[D^X] need not be symmetric.
    BlasConnector::gemm_cm('T', 'N', naos, nleft, naos, one,
                           h, naos, left, naos, zero, scratch, naos, device);
    BlasConnector::gemm_cm('T', 'N', nright, nleft, naos, factor,
                           right, naos, scratch, naos, one, result, nright, device);
}

void project_on_device(const std::complex<double>* h,
                 const std::complex<double>* left,
                 const std::complex<double>* right,
                 const int naos,
                 const int nleft,
                 const int nright,
                 const double factor,
                 std::complex<double>* scratch,
                 std::complex<double>* result,
                 const base_device::AbacusDevice_t device)
{
    if (nleft == 0 || nright == 0) { return; }
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    const std::complex<double> scale(factor, 0.0);
    BlasConnector::gemm_cm('N', 'N', naos, nleft, naos, one,
                           h, naos, left, naos, zero, scratch, naos, device);
    BlasConnector::gemm_cm('C', 'N', nright, nleft, naos, scale,
                           right, naos, scratch, naos, one, result, nright, device);
}

}

void project_exx(const double* h, const double* left, const double* right,
                 int naos, int nleft, int nright, double factor,
                 double* scratch, double* result)
{
    project_on_device(h, left, right, naos, nleft, nright, factor, scratch, result,
                      base_device::AbacusDevice_t::CpuDevice);
}

void project_exx(const std::complex<double>* h, const std::complex<double>* left,
                 const std::complex<double>* right, int naos, int nleft,
                 int nright, double factor, std::complex<double>* scratch,
                 std::complex<double>* result)
{
    project_on_device(h, left, right, naos, nleft, nright, factor, scratch, result,
                      base_device::AbacusDevice_t::CpuDevice);
}

template<typename T>
class ExxProjectionWorkspace<T>::Storage
{
public:
    const int naos;
    const int nleft;
    const int nright;
    const bool use_gpu;
    ct::Tensor h;
    ct::Tensor left;
    ct::Tensor right;
    ct::Tensor scratch;
    ct::Tensor result;

    Storage(bool gpu, int ao, int nl, int nr)
        : naos(ao), nleft(nl), nright(nr), use_gpu(gpu)
    {
#ifndef __CUDA
        if (gpu) { throw std::runtime_error("EXX GPU projection requires a CUDA build"); }
#endif
        if (ao <= 0 || nl < 0 || nr < 0)
        {
            throw std::invalid_argument("Invalid EXX projection dimensions");
        }
#ifdef __CUDA
        // LR's host solver does not initialize the BlasConnector CUDA handle.
        // Creation is idempotent; the process handle is shared by sequential projections.
        if (gpu) { BlasUtils::createGpuBlasHandle(); }
#endif
        const ct::DeviceType device = gpu ? ct::DeviceType::GpuDevice : ct::DeviceType::CpuDevice;
        const ct::DataType type = ct::DataTypeToEnum<T>::value;
        h = ct::Tensor(type, device, {ao, ao});
        left = ct::Tensor(type, device, {nl, ao});
        right = ct::Tensor(type, device, {nr, ao});
        scratch = ct::Tensor(type, device, {nl, ao});
        result = ct::Tensor(type, device, {nl, nr});
    }
};

template<typename T>
ExxProjectionWorkspace<T>::ExxProjectionWorkspace(bool use_gpu, int naos, int nleft, int nright)
{
    storage_.reset(new Storage(use_gpu, naos, nleft, nright));
}

template<typename T>
ExxProjectionWorkspace<T>::~ExxProjectionWorkspace() = default;

template<typename T>
void ExxProjectionWorkspace<T>::prepare(const T* h)
{
    Storage& s = *storage_;
    const auto size = s.h.NumElements();
#ifdef __CUDA
    if (s.use_gpu)
    {
        base_device::memory::synchronize_memory_op<T, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            s.h.template data<T>(), h, size);
    }
    else
#endif
    {
        std::copy(h, h + size, s.h.template data<T>());
    }
    s.result.zero();
}

template<typename T>
void ExxProjectionWorkspace<T>::accumulate(const T* left, const T* right, double factor)
{
    Storage& s = *storage_;
    if (s.nleft == 0 || s.nright == 0) { return; }
    const T* l = left;
    const T* r = right;
    base_device::AbacusDevice_t device = base_device::AbacusDevice_t::CpuDevice;
    if (s.use_gpu)
    {
#ifdef __CUDA
        const auto left_size = s.left.NumElements();
        const auto right_size = s.right.NumElements();
        base_device::memory::synchronize_memory_op<T, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            s.left.template data<T>(), left, left_size);
        base_device::memory::synchronize_memory_op<T, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            s.right.template data<T>(), right, right_size);
#endif
        l = s.left.template data<T>();
        r = s.right.template data<T>();
        device = base_device::AbacusDevice_t::GpuDevice;
    }
    project_on_device(s.h.template data<T>(), l, r, s.naos, s.nleft, s.nright,
                      factor, s.scratch.template data<T>(), s.result.template data<T>(), device);
}

template<typename T>
void ExxProjectionWorkspace<T>::download(T* result) const
{
    const Storage& s = *storage_;
    const auto size = s.result.NumElements();
    if (size == 0) { return; }
#ifdef __CUDA
    if (s.use_gpu)
    {
        base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
            result, s.result.template data<T>(), size);
        return;
    }
#endif
    const T* begin = s.result.template data<T>();
    std::copy(begin, begin + size, result);
}

template class ExxProjectionWorkspace<double>;
template class ExxProjectionWorkspace<std::complex<double>>;
}
