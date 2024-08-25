#pragma once
#include "module_hamilt_general/hamilt.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_lr/Grad/xc/pot_grad_xc.h"
#include "module_lr/Grad/operator_lr_hxc_common/operator_lr_hxc_common.h"
// #include "module_lr/Grad/operator_lr_hxc_common/operator_lr_exx_common.h"
#include "module_basis/module_ao/parallel_orbitals.h"
namespace LR
{
    template<typename T>
    class Z_vector_L : public hamilt::Hamilt<T, base_device::DEVICE_CPU>
    {
        using DTYPE = typename OperatorLRHxcCommon<T>::DM_TYPE;
        using ATYPE = typename OperatorLRHxcCommon<T>::AX_TYPE;
    public:
        template<typename TGint>
        Z_vector_L(const int& nspin,
            const int& naos,
            const int& nocc,
            const int& nvirt,
            const UnitCell& ucell_in,
            const std::vector<double>& orb_cutoff,
            Grid_Driver& gd_in,
            const psi::Psi<T>* psi_ks_in,
            const ModuleBase::matrix& eig_ks,
#ifdef __EXX
            Exx_LRI<T>* exx_lri_in,
#endif 
            TGint* gint_in,
            std::weak_ptr<PotHxcLR> pot_in,
            const K_Vectors& kv_in,
            Parallel_2D* pX_in,
            Parallel_2D* pc_in,
            Parallel_Orbitals* pmat_in) : nocc(nocc), nvirt(nvirt), pX(pX_in), nks(kv_in.get_nks())
        {
            ModuleBase::TITLE("Z_vector_L", "Z_vector_L");
            this->classname = "Z_vector_L";
            this->DM_trans.resize(1);
            this->DM_trans[0] = std::unique_ptr<elecstate::DensityMatrix<T, T>>(new elecstate::DensityMatrix<T, T>(&kv_in, pmat_in, nspin));
            // 1. $2\sum_bX_{ib}K_{ab}[D^X]-2\sum_jX_{ja}K_{ij}[D^X]$
            this->ops = new OperatorLRDiag<T>(eig_ks, pX_in, kv_in.get_nks(), nocc, nvirt);
            // 2. $H_{ia}[T]$, equals to $2K_{ab}[T]$ when $T$ is symmetrized
            OperatorLRHxcCommon<T>* op_hz = new OperatorLRHxcCommon<T>(nspin, naos, nocc, nvirt, psi_ks_in,
                this->DM_trans, gint_in, pot_in, ucell_in, orb_cutoff, gd_in, kv_in, pX_in, pc_in, pmat_in,
                DTYPE::X, ATYPE::CC, 2.0);
            this->ops->add(op_hz);
#ifdef __EXX
            // add EXX operators here
#endif
        }
        ~Z_vector_L()
        {
            delete this->ops;
        };

        hamilt::HContainer<T>* getHR() { return this->hR; }

    private:
        int nocc;
        int nvirt;
        int nks;
        Parallel_2D* pX = nullptr;
        T one();
        hamilt::HContainer<T>* hR = nullptr;
        /// transition density matrix in AO representation
        /// Hxc only: size=1, calculate on the same address for each bands
        /// Hxc+Exx: size=nbands, store the result of each bands for common use
        std::vector<std::unique_ptr<elecstate::DensityMatrix<T, T>>> DM_trans;
    };
}