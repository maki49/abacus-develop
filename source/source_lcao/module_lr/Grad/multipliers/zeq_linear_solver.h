#pragma once
#include "source_base/parallel_2d.h"

namespace LR
{
#ifdef __MPI
    /// @brief Solve A Z = B with ScaLAPACK LU (p?gesv), the distributed counterpart of
    /// `LR_Util::lapack_linear_solver`.
    /// @param A [in/out] local block of the n x n matrix on `pA`; overwritten by its LU factors
    /// @param B [in/out] local block of the n x nrhs right-hand side on `pB`; overwritten by Z
    /// @attention `pA` and `pB` must share the BLACS context and the block size.
    template <typename T>
    void scalapack_linear_solver(T* A, T* B, const Parallel_2D& pA, const Parallel_2D& pB);

    /// @brief Solve A Z = B for a Hermitian positive-definite A: ELPA Cholesky A = U^H U,
    /// then the two triangular solves with ScaLAPACK p?potrs.
    /// A is Hermitized as (A + A^H)/2 first, since the Cholesky reads only the upper triangle.
    /// Arguments as in `scalapack_linear_solver`; A is overwritten by U.
    /// @attention requires an ELPA build (`__ELPA`); throws otherwise. A non-positive-definite A
    /// throws only on a single rank: ELPA's Cholesky returns early on the rank owning the failing
    /// diagonal block and leaves the others in a broadcast, so with several ranks it HANGS.
    /// (Its error code is no help either: ELPA 2022.11 reports ELPA_OK even then, so the failure
    /// is detected from the pivots of U.)
    template <typename T>
    void elpa_linear_solver(T* A, T* B, const Parallel_2D& pA, const Parallel_2D& pB);
#endif
}
