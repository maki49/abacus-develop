#pragma once
#include "module_hamilt_general/hamilt.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_beyonddft/Grad/xc/pot_grad_xc.h"
#include "module_beyonddft/Grad/operator_lr_hxc_common/operator_lr_hxc_common.h"
// #include "module_beyonddft/Grad/operator_lr_hxc_common/operator_lr_exx_common.h"
#include "module_basis/module_ao/parallel_orbitals.h"
namespace hamilt
{
    template<typename T>
    class Z_vector_R : public Hamilt<T, psi::DEVICE_CPU>
    {
        using DTYPE = typename hamilt::OperatorLRHxcCommon<T>::DM_TYPE;
        using ATYPE = typename hamilt::OperatorLRHxcCommon<T>::AX_TYPE;
    public:
        template<typename TGint>
        Z_vector_R(const int& nspin,
            const int& naos,
            const int& nocc,
            const int& nvirt,
            const UnitCell& ucell_in,
            Grid_Driver& gd_in,
            const psi::Psi<T>* psi_ks_in,
            const ModuleBase::matrix& eig_ks,
#ifdef __EXX
            Exx_LRI<T>* exx_lri_in,
#endif 
            TGint* gint_in,
            elecstate::PotHxcLR* pot_in,
            const K_Vectors& kv_in,
            Parallel_2D* pX_in,
            Parallel_2D* pc_in,
            Parallel_Orbitals* pmat_in) : nocc(nocc), nvirt(nvirt), pX(pX_in), nks(kv_in.nks)
        {
            ModuleBase::TITLE("Z_vector_R", "Z_vector_R");
            this->classname = "Z_vector_R";
            this->DM_trans.resize(1);
            this->DM_trans[0] = new elecstate::DensityMatrix<T, T>(&kv_in, pmat_in, nspin);
            // 1. $2\sum_bX_{ib}K_{ab}[D^X]-2\sum_jX_{ja}K_{ij}[D^X]$
            this->ops = new OperatorLRHxcCommon<T>(nspin, naos, nocc, nvirt, psi_ks_in,
                this->DM_trans, gint_in, pot_in, ucell_in, gd_in, kv_in, pX_in, pc_in, pmat_in,
                DTYPE::X, ATYPE::CXC, -2.0);
            // 2. $H_{ia}[T]$, equals to $2K_{ab}[T]$ when $T$ is symmetrized
            OperatorLRHxcCommon<T>* op_ht = new OperatorLRHxcCommon<T>(nspin, naos, nocc, nvirt, psi_ks_in,
                this->DM_trans, gint_in, pot_in, ucell_in, gd_in, kv_in, pX_in, pc_in, pmat_in,
                DTYPE::Diff, ATYPE::CC, -2.0);
            this->ops->add(op_ht);
            // 3. $2\sum_{jb,kc} g^{xc}_{ia, jb, kc}X_{jb}X_{kc}$
            this->pot_grad = new elecstate::PotGradXCLR(pot_in->get_kernel_componets(), pot_in->get_rho_basis(), &ucell_in, pot_in->nrxx);
            OperatorLRHxcCommon<T>* op_gxc = new OperatorLRHxcCommon<T>(nspin, naos, nocc, nvirt, psi_ks_in,
                this->DM_trans, gint_in, this->pot_grad, ucell_in, gd_in, kv_in, pX_in, pc_in, pmat_in,
                DTYPE::X, ATYPE::CC, -2.0);
#ifdef __EXX
            // add EXX operators here
#endif
        }
        ~Z_vector_R()
        {
            delete this->pot_grad;
            delete this->ops;
            for (auto& d : this->DM_trans)delete d;
        };

    private:
        int nocc;
        int nvirt;
        int nks;
        Parallel_2D* pX = nullptr;
        T one();
        HContainer<T>* hR = nullptr;
        /// transition density matrix in AO representation
        /// Hxc only: size=1, calculate on the same address for each bands
        /// Hxc+Exx: size=nbands, store the result of each bands for common use
        std::vector<elecstate::DensityMatrix<T, T>*> DM_trans;

        elecstate::PotGradXCLR* pot_grad = nullptr;
    };
}