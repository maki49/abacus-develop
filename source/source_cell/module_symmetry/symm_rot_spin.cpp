#include "symm_rot_spin.h"

#include "source_base/constants.h"

#include <cmath>

namespace ModuleSymmetry
{
namespace SpinRotation
{
using cd = std::complex<double>;

namespace
{
// Proper part of an orthogonal cartesian operation: spin only sees the proper rotation.
// For det(gmatc) = +1 it is gmatc itself; for det = -1 (improper) it is -gmatc.
ModuleBase::Matrix3 proper_part(const ModuleBase::Matrix3& gmatc)
{
    return gmatc.Det() < 0.0 ? gmatc * (-1.0) : gmatc;
}

double clamp_cos(double c)
{
    if (c > 1.0)
    {
        return 1.0;
    }
    if (c < -1.0)
    {
        return -1.0;
    }
    return c;
}

// U(n, theta) = cos(theta/2) I - i sin(theta/2) (n . sigma)
Su2 su2_from_axis_angle(double nx, double ny, double nz, double theta)
{
    const double c = std::cos(0.5 * theta);
    const double s = std::sin(0.5 * theta);
    // -i s (n.sigma) =
    //   [ -i s nz          -i s nx - s ny ]
    //   [ -i s nx + s ny    i s nz        ]
    Su2 U;
    U[0] = cd(c, -s * nz);            // (uu)
    U[1] = cd(-s * ny, -s * nx);      // (ud)
    U[2] = cd(s * ny, -s * nx);       // (du)
    U[3] = cd(c, s * nz);             // (dd)
    return U;
}
} // namespace

Su2 so3_to_su2(const ModuleBase::Matrix3& gmatc, const double eps)
{
    const ModuleBase::Matrix3 R = proper_part(gmatc);

    const double trace = R.e11 + R.e22 + R.e33;
    const double cos_theta = clamp_cos(0.5 * (trace - 1.0));
    const double theta = std::acos(cos_theta);
    const double sin_theta = std::sin(theta);

    // theta ~ 0 : identity rotation, U = I
    if (theta < eps)
    {
        return Su2{cd(1.0, 0.0), cd(0.0, 0.0), cd(0.0, 0.0), cd(1.0, 0.0)};
    }

    // theta ~ pi : axis-angle formula is singular (sin theta -> 0). 
    // Extract the axis from the symmetric part: 
    // for theta = pi,  R = 2 n n^T - I, so n_i n_j = (R_ij + delta_ij)/2.
    if (std::abs(theta - ModuleBase::PI) < eps || std::abs(sin_theta) < eps)
    {
        double nn[3] = {0.5 * (R.e11 + 1.0), 0.5 * (R.e22 + 1.0), 0.5 * (R.e33 + 1.0)};
        for (int i = 0; i < 3; ++i)
        {
            nn[i] = nn[i] > 0.0 ? nn[i] : 0.0; // guard tiny negatives
        }
        // pick the largest diagonal as the reference component (sign fixed to +)
        int imax = 0;
        if (nn[1] > nn[imax])
        {
            imax = 1;
        }
        if (nn[2] > nn[imax])
        {
            imax = 2;
        }
        double n[3] = {0.0, 0.0, 0.0};
        n[imax] = std::sqrt(nn[imax]);
        // For theta = pi, R = 2 n n^T - I, so the off-diagonal R_ij = 2 n_i n_j and the symmetric
        // combination R_ij + R_ji = 4 n_i n_j. Hence n_i n_j = 0.25*(R_ij + R_ji).
        const double off[3][3] = {{0.0, 0.25 * (R.e12 + R.e21), 0.25 * (R.e13 + R.e31)},
                                  {0.25 * (R.e21 + R.e12), 0.0, 0.25 * (R.e23 + R.e32)},
                                  {0.25 * (R.e31 + R.e13), 0.25 * (R.e32 + R.e23), 0.0}};
        for (int j = 0; j < 3; ++j)
        {
            if (j == imax)
            {
                continue;
            }
            n[j] = off[imax][j] / n[imax];
        }
        // normalize for safety
        const double norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (norm > 0.0)
        {
            n[0] /= norm;
            n[1] /= norm;
            n[2] /= norm;
        }
        return su2_from_axis_angle(n[0], n[1], n[2], ModuleBase::PI);
    }

    // general case: axis from the antisymmetric part, expressed in ROW-vector elements.
    // n_x = (R_yz - R_zy)/(2 sin),  n_y = (R_zx - R_xz)/(2 sin),  n_z = (R_xy - R_yx)/(2 sin)
    const double inv = 1.0 / (2.0 * sin_theta);
    const double nx = (R.e23 - R.e32) * inv;
    const double ny = (R.e31 - R.e13) * inv;
    const double nz = (R.e12 - R.e21) * inv;
    return su2_from_axis_angle(nx, ny, nz, theta);
}

ModuleBase::Matrix3 spin_so3(const ModuleBase::Matrix3& gmatc)
{
    // rho'^i = sum_j W_ij rho^j with W = R_proper^T (= column-vector rotation R_col).
    return proper_part(gmatc).Transpose();
}

ModuleBase::Matrix3 fit_spin_rotation(const std::vector<ModuleBase::Vector3<double>>& from,
                                      const std::vector<ModuleBase::Vector3<double>>& to,
                                      bool& ok,
                                      const double tol)
{
    ok = false;
    const ModuleBase::Matrix3 identity(1, 0, 0, 0, 1, 0, 0, 0, 1);
    const int n = static_cast<int>(from.size());
    if (n == 0 || static_cast<int>(to.size()) != n) { return identity; }

    // A norm below this is treated as a zero (non-magnetic) moment: it constrains nothing.
    const double nrm_eps = 1e-6;

    // 1st frame axis: the first non-zero moment.
    int a = -1;
    for (int i = 0; i < n; ++i) { if (from[i].norm() > nrm_eps) { a = i; break; } }
    if (a < 0) { return identity; }   // all moments zero: no constraint (handled by has_moment gate upstream)
    const ModuleBase::Vector3<double> e1 = from[a] * (1.0 / from[a].norm());
    const ModuleBase::Vector3<double> f1 = to[a] * (1.0 / std::max(to[a].norm(), nrm_eps));

    // 2nd frame axis: the first moment with a component perpendicular to e1 (rank >= 2).
    int b = -1;
    ModuleBase::Vector3<double> e2, f2;
    for (int i = 0; i < n; ++i)
    {
        const ModuleBase::Vector3<double> perp = from[i] - e1 * (e1 * from[i]);
        if (perp.norm() > nrm_eps) { b = i; e2 = perp * (1.0 / perp.norm()); break; }
    }
    if (b < 0) { return identity; }   // rank-1 (collinear moments): degenerate, out of nspin=4 SSG scope
    const ModuleBase::Vector3<double> fperp = to[b] - f1 * (f1 * to[b]);
    if (fperp.norm() <= nrm_eps) { return identity; }   // target lost its perpendicular part -> not a rotation
    f2 = fperp * (1.0 / fperp.norm());

    // 3rd axis by right-handed cross product -> both frames right-handed -> det(R)=+1.
    const ModuleBase::Vector3<double> e3 = e1 ^ e2;
    const ModuleBase::Vector3<double> f3 = f1 ^ f2;

    // R maps e_k -> f_k :  R = F E^T,  R_ij = sum_k f_k[i] e_k[j].  Acts as column vectors: R * m.
    const ModuleBase::Matrix3 R(
        f1.x * e1.x + f2.x * e2.x + f3.x * e3.x, f1.x * e1.y + f2.x * e2.y + f3.x * e3.y, f1.x * e1.z + f2.x * e2.z + f3.x * e3.z,
        f1.y * e1.x + f2.y * e2.x + f3.y * e3.x, f1.y * e1.y + f2.y * e2.y + f3.y * e3.y, f1.y * e1.z + f2.y * e2.z + f3.y * e3.z,
        f1.z * e1.x + f2.z * e2.x + f3.z * e3.x, f1.z * e1.y + f2.z * e2.y + f3.z * e3.y, f1.z * e1.z + f2.z * e2.z + f3.z * e3.z);

    // Verify R exactly maps EVERY moment (this is what makes the fit correct AND, for rank>=2
    // moments, guarantees group closure: R is the unique proper map of the moment set).
    for (int i = 0; i < n; ++i)
    {
        const ModuleBase::Vector3<double> d = R * from[i] - to[i];
        if (std::fabs(d.x) > tol || std::fabs(d.y) > tol || std::fabs(d.z) > tol) { return identity; }
    }
    ok = true;
    return R;
}

ModuleBase::Matrix3 pauli_rotation_matrix(const Su2& U)
{
    // sigma matrices (row-major 2x2)
    static const Su2 sx = {cd(0, 0), cd(1, 0), cd(1, 0), cd(0, 0)};
    static const Su2 sy = {cd(0, 0), cd(0, -1), cd(0, 1), cd(0, 0)};
    static const Su2 sz = {cd(1, 0), cd(0, 0), cd(0, 0), cd(-1, 0)};
    const Su2 sig[3] = {sx, sy, sz};
    const Su2 Ud = dagger(U);

    double w[3][3];
    for (int j = 0; j < 3; ++j)
    {
        const Su2 rot = mat2_mul(mat2_mul(U, sig[j]), Ud); // U sigma_j U^dagger
        for (int i = 0; i < 3; ++i)
        {
            // W_ij = (1/2) Tr(sigma_i * rot)
            const Su2& si = sig[i];
            const cd tr = si[0] * rot[0] + si[1] * rot[2] + si[2] * rot[1] + si[3] * rot[3];
            w[i][j] = 0.5 * tr.real();
        }
    }
    return ModuleBase::Matrix3(w[0][0], w[0][1], w[0][2],
                               w[1][0], w[1][1], w[1][2],
                               w[2][0], w[2][1], w[2][2]);
}

Su2 dagger(const Su2& U)
{
    return Su2{std::conj(U[0]), std::conj(U[2]), std::conj(U[1]), std::conj(U[3])};
}

Su2 mat2_mul(const Su2& A, const Su2& B)
{
    return Su2{A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3],
               A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3]};
}

Su2 rotate_spin_block(const Su2& block, const Su2& U)
{
    return mat2_mul(mat2_mul(U, block), dagger(U));
}

void rotate_pauli_components(const ModuleBase::Matrix3& gmatc, const double in[4], double out[4])
{
    const ModuleBase::Matrix3 W = spin_so3(gmatc);
    out[0] = in[0]; // charge component is a scalar, untouched by spin rotation
    out[1] = W.e11 * in[1] + W.e12 * in[2] + W.e13 * in[3];
    out[2] = W.e21 * in[1] + W.e22 * in[2] + W.e23 * in[3];
    out[3] = W.e31 * in[1] + W.e32 * in[2] + W.e33 * in[3];
}
} // namespace SpinRotation
} // namespace ModuleSymmetry
