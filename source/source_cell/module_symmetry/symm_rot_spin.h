#ifndef SYMM_ROT_SPIN_H
#define SYMM_ROT_SPIN_H

#include "source_base/matrix3.h"

#include <array>
#include <complex>
#include <vector>

namespace ModuleSymmetry
{
/// @brief  SU(2) spin-1/2 representation of a real-space symmetry operation.
///
/// These utilities provide the spin (SU(2)) part of the symmetry operation needed for
/// nspin=4 (non-collinear / SOC) symmetrization. The orbital (real-spherical-harmonics)
/// part is handled separately by the existing Wigner-D / T(V) machinery; the full
/// representation of a symmetry operation on a spinor orbital is the tensor product
/// T(V) (x) U(V).
///
/// Convention notes:
///  - ABACUS uses the ROW-vector convention, i.e. a point transforms as r' = r * V,
///    so the cartesian rotation matrix `gmatc` stored by ABACUS equals the transpose of
///    the textbook (column-vector) rotation:  V_row = V_col^T.
///  - Spin is a pseudovector: it only sees the PROPER part of the operation. For an
///    improper operation (det(gmatc) = -1) we factor out the inversion and build U from
///    the proper rotation  R_proper = -gmatc  (which then has det = +1).
///  - With U built from the axis-angle of R_proper via
///        U = cos(theta/2) I - i sin(theta/2) (n . sigma),
///    the induced action on the Pauli (spin-density) vector is
///        rho'^i = sum_j W_ij rho^j,   with   W = R_proper^T  ( = R_col ),
///    where W_ij = (1/2) Tr(sigma_i U sigma_j U^dagger). This relation is the basis of
///    the unit tests and is convention-self-consistent regardless of the textbook
///    index ordering quoted in the formula document.
namespace SpinRotation
{
/// A 2x2 complex matrix stored row-major: {m00, m01, m10, m11}.
using Su2 = std::array<std::complex<double>, 4>;

/// @brief Build the SU(2) spin-1/2 matrix U corresponding to a cartesian symmetry
///        operation `gmatc` (row-vector convention, det = +-1).
///
/// Handles the special cases theta = 0 (U = I) and theta = pi (axis from the symmetric
/// part of the rotation, where the standard axis-angle formula is singular). For an
/// improper operation the proper part R_proper = -gmatc is used.
///
/// The returned U is defined up to the double-group sign (+-U); both signs give the same
/// similarity transform U D U^dagger, so this ambiguity is harmless for symmetrization.
Su2 so3_to_su2(const ModuleBase::Matrix3& gmatc, const double eps = 1e-6);

/// @brief The proper rotation acting on the spin-density 3-vector (Pauli x,y,z
///        components), i.e. the W matrix with rho'^i = sum_j W_ij rho^j.
///        Computed directly from gmatc as W = R_proper^T (R_proper = proper part).
ModuleBase::Matrix3 spin_so3(const ModuleBase::Matrix3& gmatc);

/// @brief (nspin=4 SSG) Fit the INDEPENDENT proper spin rotation R (det=+1) that maps one
///        set of moment vectors onto another: R * from[i] == to[i] for all i, using the
///        same column-vector action `R * m` as spin_so3() (so it drops in for W).
///
/// Unlike spin_so3(gmatc), R is NOT derived from any spatial operation -- it is found from
/// the moment configuration alone, which is exactly the spin/space decoupling of the spin
/// space group (valid with SOC off). Built by an orthonormal-frame (Gram-Schmidt) construction
/// from two linearly independent moment pairs, so no SVD is needed and det(R)=+1 is guaranteed:
///   e1=norm(from[a]), e2=norm(from[b] perp e1), e3=e1 x e2;  f_k likewise from to[];  R = F E^T.
/// Then R is VERIFIED to map every moment within `tol`; `ok` is set false if the verification
/// fails or the moments are rank-deficient (collinear -> no independent second axis; that is the
/// nspin=2 collinear regime, out of Phase-2 scope). A false `ok` means "reject this operation".
///
/// @param from  source moments m_i (all atoms)
/// @param to    target moments (m_{g(i)} for the unitary test, -m_{g(i)} for the antiunitary test)
/// @param[out] ok  true iff a proper rotation exactly mapping all moments (within tol) was found
/// @param tol   absolute tolerance for the per-component moment match (default matches Symmetry::epsilon usage)
/// @return the fitted rotation R (identity when ok is false; caller must check ok)
ModuleBase::Matrix3 fit_spin_rotation(const std::vector<ModuleBase::Vector3<double>>& from,
                                      const std::vector<ModuleBase::Vector3<double>>& to,
                                      bool& ok,
                                      const double tol = 1e-5);

/// @brief The same W matrix computed independently from a given SU(2) matrix U via
///        W_ij = (1/2) Tr(sigma_i U sigma_j U^dagger). Used for verification.
ModuleBase::Matrix3 pauli_rotation_matrix(const Su2& U);

/// @brief Hermitian conjugate of a 2x2 SU(2) matrix.
Su2 dagger(const Su2& U);

/// @brief 2x2 complex matrix product A * B (both row-major).
Su2 mat2_mul(const Su2& A, const Su2& B);

/// @brief Rotate the 2x2 spin block of a spinor matrix element in place:
///        m_block' = U * m_block * U^dagger.
///        `block` is the 2x2 spin sub-matrix {uu, ud, du, dd} of a fixed orbital pair.
Su2 rotate_spin_block(const Su2& block, const Su2& U);

/// @brief Rotate the four Pauli components (rho^0, rho^x, rho^y, rho^z) of a single
///        real-space density point: rho^0 is unchanged, (rho^x, rho^y, rho^z) are mixed
///        by W = spin_so3(gmatc). The input/output are component values at one grid point.
void rotate_pauli_components(const ModuleBase::Matrix3& gmatc,
                             const double in[4],
                             double out[4]);
} // namespace SpinRotation
} // namespace ModuleSymmetry

#endif // SYMM_ROT_SPIN_H
