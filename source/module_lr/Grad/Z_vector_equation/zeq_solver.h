#pragma once
#include "hamilt_zeq_left.h"
#include "hamilt_zeq_right.h"

namespace LR
{
    template<typename T, typename TGint>
    psi::Psi<T> Z_vector(const psi::Psi<T>* X,
        const int& nstates,
        const int& nspin,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        const UnitCell& ucell,
        Grid_Driver& gd,
        const psi::Psi<T>* psi_ks,
        const ModuleBase::matrix& eig_ks,
#ifdef __EXX
        Exx_LRI<T>* exx_lri,
#endif 
        TGint* gint,
        std::weak_ptr<PotHxcLR> pot,
        const K_Vectors& kv,
        Parallel_2D& px,
        Parallel_2D& pc,
        Parallel_Orbitals& pmat);
}

#include "zeq_solver.hpp"