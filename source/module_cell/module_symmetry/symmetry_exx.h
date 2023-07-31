#pragma once
#include <vector>
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_cell/unitcell.h"
#include "module_psi/psi.h"

class SymExx
{
public:
    static std::vector<int> invmap(const int* map, const size_t size);
    static std::vector<int> mapmul(const int* map1, const int* map2, const size_t size);


    /// @brief size: atom index map corresponding to each symmetry operation. size: [nrotk][nat]
    std::vector<std::vector<int>> isym_iat_rotiat;

    /// @brief Rearrange the $\nu$ index of overlap matrices of ibz-kpoints $S_{\mu,\nu}(gk)$ according to the symmetry operations
    /// @param ikibz [in] ibz-kpoint index
    /// @param sloc_ikibz [in] local overlap matrices of current ibz-kpoint: S(gk)
    /// @param nbasis [in] global size of basis
    /// @param p2d [in]2d parallel info
    /// @param kstars [in] symmetry-equal k points to each ibz-kpont: [iksibz][isym][kvec_d]
    /// @param ucell [in] unitcell
    /// @return local S(k) for each k in kstars[ikibz]
    std::vector<std::vector<std::complex<double>>> rearange_smat(
        const int ikibz,
        const std::vector<std::complex<double>> sloc_ikibz,
        const int nbasis,
        const Parallel_2D& p2d,
        std::vector<std::map<int, ModuleBase::Vector3<double>>> kstars,
        const UnitCell& ucell);

    /// @brief restore c_k from c_gk: $c_k=\tilde{S}^{-1}(k)S(gk)c_{gk}$
    /// @param ikibz  [in] ibz-kpoint index
    /// @param psi_ikibz [in] c_gk: wavefunction of current ibz-kpoint
    /// @param sloc_ikibz [in] local overlap matrices of current ibz-kpoint: S(gk)
    /// @param sloc_ik [in] $\nu$-rearranged local overlap matrices of each k in current kstar : S(k)
    /// @param nbasis [in] global size of basis
    /// @param pv [in] parallel orbitals (for both matrix and wavefunction)
    /// @param col_inside [in] whether the matrix is column-major (major means memory continuity)
    /// @return 
    psi::Psi<std::complex<double>, psi::DEVICE_CPU> restore_psik(
        const int& ikibz,
        const std::vector<std::complex<double>>& psi_ikibz,
        const std::vector<std::complex<double>>& sloc_ikibz,
        const std::vector<std::vector<std::complex<double>>>& sloc_ik,
        const int& nbasis,
        const Parallel_Orbitals& pv,
        const bool col_inside);

private:
    std::vector<std::complex<double>> get_full_smat(
        const std::vector<std::complex<double>>& locmat,
        const int& nbasis,
        const Parallel_2D& p2d,
        const bool col_inside);
};