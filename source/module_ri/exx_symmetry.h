#pragma once
#include <vector>
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_cell/unitcell.h"
#include "module_psi/psi.h"

namespace ExxSym
{
    /// @brief Rearrange the $\nu$ index of overlap matrices of ibz-kpoints $S_{\mu,\nu}(gk)$ according to the symmetry operations
    /// @param ikibz [in] ibz-kpoint index
    /// @param sloc_ikibz [in] local overlap matrices of current ibz-kpoint: S(gk)
    /// @param nbasis [in] global number of basis
    /// @param p2d [in]2d parallel info
    /// @param isym_iat_rotiat [in] atom index map corresponding to each symmetry operation
    /// @param kstars [in] symmetry-equal k points to each ibz-kpont: [iksibz][isym][kvec_d]
    /// @param ucell [in] unitcell
    /// @return local S(k) for each k in kstars[ikibz]
    std::vector<std::vector<std::complex<double>>> rearange_smat(
        const int ikibz,
        const std::vector<std::complex<double>> sloc_ikibz,
        const int nbasis,
        const Parallel_2D& p2d,
        std::vector<std::vector<int>>& isym_iat_rotiat,
        std::vector<std::map<int, ModuleBase::Vector3<double>>> kstars,
        const UnitCell& ucell);

    /// @brief restore c_k from c_gk: $c_k=\tilde{S}^{-1}(k)S(gk)c_{gk}$
    /// @param ikibz  [in] ibz-kpoint index
    /// @param psi_ikibz [in] c_gk: wavefunction of current ibz-kpoint
    /// @param sloc_ikibz [in] local overlap matrices of current ibz-kpoint: S(gk)
    /// @param sloc_ik [in] $\nu$-rearranged local overlap matrices of each k in current kstar : S(k)
    /// @param nbasis [in] global number of basis
    /// @param nbands [in] global number of bands
    /// @param pv [in] parallel orbitals (for both matrix and wavefunction)
    /// @param col_inside [in] whether the matrix is column-major (major means memory continuity)
    /// @return c_k: wavefunction of each k in kstars[ikibz]
#ifdef __MPI
    psi::Psi<std::complex<double>, psi::DEVICE_CPU> restore_psik_scalapack(
        const int& ikibz,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& psi_ikibz,
        const std::vector<std::complex<double>>& sloc_ikibz,
        const std::vector<std::vector<std::complex<double>>>& sloc_ik,
        const int& nbasis,
        const int& nbands,
        const Parallel_Orbitals& pv);
#endif
    psi::Psi<std::complex<double>, psi::DEVICE_CPU> restore_psik_lapack(
        const int& ikibz,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& psi_ikibz,
        const std::vector<std::complex<double>>& sloc_ikibz,
        const std::vector<std::vector<std::complex<double>>>& sloc_ik,
        const int& nbasis,
        const int& nbands);

    std::vector<std::complex<double>> get_full_smat(
        const std::vector<std::complex<double>>& locmat,
        const int& nbasis,
        const Parallel_2D& p2d,
        const bool col_inside);
}