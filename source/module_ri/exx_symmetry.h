#pragma once
#include <vector>
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_base/vector3.h"
#include "module_psi/psi.h"
#include "module_cell/unitcell.h"
namespace ExxSym
{
    /// @brief Rearrange the $\nu$ index of overlap matrices of ibz-kpoints $S_{\mu,\nu}(gk)$ according to the symmetry operations
    /// @param ikibz [in] ibz-kpoint index
    /// @param sloc_ikibz [in] local overlap matrices of current ibz-kpoint: S(gk)
    /// @param nbasis [in] global number of basis
    /// @param p2d [in]2d parallel info
    /// @param isym_iat_rotiat [in] atom index map corresponding to each symmetry operation
    /// @param kstar_ibz [in] symmetry-equal k points to current ibz-kpont: [isym][kvec_d]
    /// @param ucell [in] unitcell
    /// @param col_inside [in] whether the matrix is column-major (major means memory continuity)
    /// @return local S(k) for each k in kstars[ikibz]
    std::vector<std::vector<std::complex<double>>>cal_Sk_rot(
        const std::vector<std::complex<double>> sloc_ikibz,
        const int nbasis,
        const Parallel_2D& p2d,
        const std::vector<std::vector<int>>& isym_iat_rotiat,
        const std::map<int, ModuleBase::Vector3<double>>& kstar_ibz,
        const UnitCell& ucell,
        const bool col_inside);


    /// @brief restore c_k from c_gk: $c_k=\tilde{S}^{-1}(k)S(gk)c_{gk}$ for all the ibz-kpoints gk
    /// @param nkstot_full  [in] number of kpoints in all the kstars
    /// @param psi_ikibz [in] c_gk: wavefunction of current ibz-kpoint
    /// @param sloc_ikibz [in] local overlap matrices of current ibz-kpoint: S(gk)
    /// @param sloc_ik [in] $\nu$-rearranged local overlap matrices of each k in current kstar : S(k)
    /// @param nbasis [in] global number of basis
    /// @param nbands [in] global number of bands
    /// @param pv [in] parallel orbitals (for both matrix and wavefunction)
    psi::Psi<std::complex<double>, psi::DEVICE_CPU> restore_psik(
        const int& nkstot_full,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& psi_ibz,
        const std::vector<std::vector<std::complex<double>>>& sloc_ibz,
        const std::vector<std::vector<std::vector<std::complex<double>>>>& sloc_full,
        const int& nbasis,
        const int& nbands,
        const Parallel_Orbitals& pv);

    /// @brief restore c_k from c_gk: $c_k=\tilde{S}^{-1}(k)S(gk)c_{gk}$ for one ibz-kpoint
    /// @param ikibz  [in] ibz-kpoint index
    /// @param ikfull_start [in] start index of k in all the kstars
    /// @param psi_ikibz [in] c_gk: wavefunction of current ibz-kpoint
    /// @param sloc_ikibz [in] local overlap matrices of current ibz-kpoint: S(gk)
    /// @param sloc_ik [in] $\nu$-rearranged local overlap matrices of each k in current kstar : S(k)
    /// @param nbasis [in] global number of basis
    /// @param nbands [in] global number of bands
    /// @param pv [in] parallel orbitals (for both matrix and wavefunction)
    /// @param col_inside [in] whether the matrix is column-major (major means memory continuity)
    /// @param psi_full: [out]wavefunction of each k in kstars[ikibz]
#ifdef __MPI
    void restore_psik_scalapack(
        const int& ikibz,
        const int& ikfull_start,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& psi_ikibz,
        const std::vector<std::complex<double>>& sloc_ikibz,
        const std::vector<std::vector<std::complex<double>>>& sloc_ik,
        const int& nbasis,
        const int& nbands,
        const Parallel_Orbitals& pv,
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>* psi_full);
#endif
    void restore_psik_lapack(
        const int& ikibz,
        const int& ikfull_start,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& psi_ikibz,
        const std::vector<std::complex<double>>& sloc_ikibz,
        const std::vector<std::vector<std::complex<double>>>& sloc_ik,
        const int& nbasis,
        const int& nbands,
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>* psi_full);

    std::vector<std::complex<double>> get_full_smat(
        const std::vector<std::complex<double>>& locmat,
        const int& nbasis,
        const Parallel_2D& p2d,
        const bool col_inside);
}