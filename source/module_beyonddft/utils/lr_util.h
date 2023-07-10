#pragma once
#include <cstddef>
#include <vector>
#include <utility>
#include "module_base/matrix.h"
#include "module_basis/module_ao/parallel_2d.h"

namespace LR_Util
{

    /// =====================PHYSICS====================

    /// @brief calculate the number of electrons
    /// @tparam TCell 
    /// @param ucell 
    template <typename TCell>
    const size_t cal_nelec(const TCell& ucell);
    
    /// @brief calculate the number of occupied orbitals
    /// @param nelec 
    const size_t cal_nocc(size_t nelec);
    
    /// @brief  set the index map: ix to (ic, iv) and vice versa
    /// by diagonal traverse the c-v pairs
    /// leftdown -> rightup for mode 0, rightup -> leftdown for mode 1
    /// @param mode  0: homo-1 -> lumo first; 1: homo -> lumo+1 first
    /// @param nc number of occupied bands
    /// @param nv number of virtual bands
    /// @return [iciv2ix, ix2iciv]
    std::pair<ModuleBase::matrix, std::vector<std::pair<int, int>>>
        set_ix_map_diagonal(bool mode, int nc, int nv);

    /// =================ALGORITHM====================

    //====== newers and deleters========
    //(arbitrary dimention will be supported in the future)
    /// @brief  delete 2d pointer 
    /// @tparam T 
    /// @param p2 
    /// @param size 
    template <typename T>
    void delete_p2(T** p2, size_t size);
    
    /// @brief  delete 3d pointer 
    /// @tparam T 
    /// @param p2 
    /// @param size1
    /// @param size2
    template <typename T>
    void delete_p3(T*** p3, size_t size1, size_t size2);


    void setup_2d_division(Parallel_2D& pv, int nb, int gr, int gc);

#ifdef __MPI
    template <typename T>
    void gather_2d_block(const Parallel_2D& pv, const T* submat, T* fullmat);
#endif
}