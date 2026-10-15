#ifndef ABACUS_LR_HSOLVER_LRTD_H
#define ABACUS_LR_HSOLVER_LRTD_H
#include "source_base/parallel_2d.h"
#include "source_base/macros.h"
#include <complex>
#include <functional>
#include <string>
#include <vector>
namespace LR
{
template<typename T> class HamiltLR;
template<typename T> class HamiltULR;
struct OrbitalTransitionSpace;
template<typename T> using Real = typename GetTypeReal<T>::type;
namespace HSolver
{
/// Solve the distributed LR eigenproblem using the selected dense or iterative backend.
        template<typename T, typename THamilt>
        void solve(const THamilt& hm,
            T* psi,
            const int& dim, ///< local leading dimension (or nbasis)
            const int& nband,   ///< nstates in LR-TDDFT, not (nocc+nvirt)
            const OrbitalTransitionSpace& transitions,
            double* eig,
            const std::string method,
            const Real<T>& diag_ethr, ///< threshold for diagonalization
            const std::vector<Real<T>>& precondition,
            const int preconditioner_block_size,
            const std::function<void(const T*, T*)>& preconditioner_proxy,
            const bool use_gpu,
            const bool hermitian);
}
}
#endif
