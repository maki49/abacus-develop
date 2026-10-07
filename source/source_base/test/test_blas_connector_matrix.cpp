#include "../module_external/blas_connector.h"
#include "../module_container/ATen/core/tensor.h"
#include "../module_device/memory_op.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <complex>
#include <vector>

namespace
{
template<typename Real>
void check_complex_gemm(bool gpu, double tolerance)
{
    using T = std::complex<Real>;
    const int m = 2;
    const int n = 2;
    const int k = 3;
    const T alpha(0.7, -0.3);
    const T beta(-0.2, 0.4);
    const ct::DeviceType storage = gpu ? ct::DeviceType::GpuDevice : ct::DeviceType::CpuDevice;
    const base_device::AbacusDevice_t device = gpu
        ? base_device::AbacusDevice_t::GpuDevice : base_device::AbacusDevice_t::CpuDevice;
    const ct::DataType type = ct::DataTypeToEnum<T>::value;
    ct::Tensor a_buffer(type, storage, {6});
    ct::Tensor b_buffer(type, storage, {6});
    ct::Tensor c_buffer(type, storage, {4});
    for (const char ta : {'N', 'T', 'C'})
    {
        for (const char tb : {'N', 'T', 'C'})
        {
            const int lda = ta == 'N' ? m : k;
            const int ldb = tb == 'N' ? k : n;
            std::vector<T> a(6);
            std::vector<T> b(6);
            std::vector<T> c(4);
            for (int i = 0; i < 6; ++i)
            {
                a[i] = T(0.3 + 0.1 * i, 0.2 - 0.07 * i);
                b[i] = T(-0.2 + 0.04 * i, 0.1 + 0.08 * i);
            }
            for (int i = 0; i < 4; ++i) { c[i] = T(0.05 * i, -0.02 * i); }
            std::vector<T> expected(4);
            for (int j = 0; j < n; ++j)
            {
                for (int i = 0; i < m; ++i)
                {
                    T sum(0.0, 0.0);
                    for (int p = 0; p < k; ++p)
                    {
                        T av = ta == 'N' ? a[p * lda + i] : a[i * lda + p];
                        T bv = tb == 'N' ? b[j * ldb + p] : b[p * ldb + j];
                        if (ta == 'C') { av = std::conj(av); }
                        if (tb == 'C') { bv = std::conj(bv); }
                        sum += av * bv;
                    }
                    expected[j * m + i] = alpha * sum + beta * c[j * m + i];
                }
            }
#ifdef __CUDA
            if (gpu)
            {
                base_device::memory::synchronize_memory_op<T, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
                    a_buffer.data<T>(), a.data(), 6);
                base_device::memory::synchronize_memory_op<T, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
                    b_buffer.data<T>(), b.data(), 6);
                base_device::memory::synchronize_memory_op<T, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
                    c_buffer.data<T>(), c.data(), 4);
            }
            else
#endif
            {
                std::copy(a.begin(), a.end(), a_buffer.data<T>());
                std::copy(b.begin(), b.end(), b_buffer.data<T>());
                std::copy(c.begin(), c.end(), c_buffer.data<T>());
            }
            BlasConnector::gemm_cm(ta, tb, m, n, k, alpha, a_buffer.data<T>(), lda,
                                   b_buffer.data<T>(), ldb, beta, c_buffer.data<T>(), m, device);
#ifdef __CUDA
            if (gpu)
            {
                base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
                    c.data(), c_buffer.data<T>(), 4);
            }
            else
#endif
            {
                const T* begin = c_buffer.data<T>();
                std::copy(begin, begin + 4, c.data());
            }
            for (int i = 0; i < 4; ++i)
            {
                EXPECT_NEAR(std::abs(c[i] - expected[i]), 0.0, tolerance) << ta << tb << " index=" << i;
            }
        }
    }
}
}

TEST(BlasConnectorMatrix, ComplexFloatTransposeModesCpu) { check_complex_gemm<float>(false, 2e-6); }
TEST(BlasConnectorMatrix, ComplexDoubleTransposeModesCpu) { check_complex_gemm<double>(false, 1e-12); }

#ifdef __CUDA
TEST(BlasConnectorMatrix, ComplexTransposeModesGpu)
{
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) { GTEST_SKIP() << "No CUDA device available"; }
    BlasUtils::createGpuBlasHandle();
    check_complex_gemm<float>(true, 2e-6);
    check_complex_gemm<double>(true, 1e-12);
    BlasUtils::destoryBLAShandle();
}
#endif
