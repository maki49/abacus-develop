#ifndef ABACUS_LR_EXX_PROJECTION_H
#define ABACUS_LR_EXX_PROJECTION_H

#include <complex>
#include <memory>

namespace LR
{
/// Project a real, generally nonsymmetric AO matrix against two sets of columns.
/// All matrices are column major. result(right,left) += factor * left^T H right.
/// The caller owns the naos*nleft scratch buffer; inputs and outputs cannot alias.
void project_exx(const double* h,
                 const double* left,
                 const double* right,
                 int naos,
                 int nleft,
                 int nright,
                 double factor,
                 double* scratch,
                 double* result);

/// Complex Bloch projection: result(right,left) += factor * right^H H(k) left.
/// H(k) must contain the positive Bloch phase, conjugate to the probe density.
/// All matrices are column major; scratch has naos*nleft elements.
void project_exx(const std::complex<double>* h,
                 const std::complex<double>* left,
                 const std::complex<double>* right,
                 int naos,
                 int nleft,
                 int nright,
                 double factor,
                 std::complex<double>* scratch,
                 std::complex<double>* result);
/// Reusable host/device boundary for a folded AO response. Buffers belong to
/// one operator instance; prepare() resets accumulation for the next k point.
/// GPU mode uses CUDA GEMM; host maps and MPI reduction remain on the CPU.
template<typename T>
class ExxProjectionWorkspace
{
public:
    ExxProjectionWorkspace(bool use_gpu, int naos, int nleft, int nright);
    ~ExxProjectionWorkspace();
    void prepare(const T* h);
    void accumulate(const T* left, const T* right, double factor);
    void download(T* result) const;

private:
    class Storage;
    std::unique_ptr<Storage> storage_;
};

}

#endif
