#pragma once
#include "module_lr/hamilt_casida.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_lr/Grad/xc/pot_grad_xc.h"
#include "module_lr/operator_casida/operator_lr_hxc.h"
#include "module_lr/Grad/dm_diff/dm_diff.h"
#include "module_basis/module_ao/parallel_orbitals.h"
namespace LR
{
    template<typename T>
    class Z_vector_L : public HamiltLR<T>
    {
        using ATYPE = typename OperatorLRHxc<T>::MO_TO_AO_TYPE;
    public:
        template<typename TGint>
        Z_vector_L(const std::string& xc_kernel,
            const int& nspin,
            const int& naos,
            const std::vector<int>& nocc,
            const std::vector<int>& nvirt,
            const UnitCell& ucell_in,
            const std::vector<double>& orb_cutoff,
            const Grid_Driver& gd_in,
            const psi::Psi<T>& psi_ks_in,
            const ModuleBase::matrix& eig_ks,
#ifdef __EXX
            std::weak_ptr<Exx_LRI<T>> exx_lri_in,
            const double& exx_alpha,
#endif 
            TGint* gint_in,
            std::weak_ptr<PotLRBase> pot_in,
            const K_Vectors& kv_in,
            const std::vector<Parallel_2D>& pX_in,
            const Parallel_2D& pc_in,
            const Parallel_Orbitals& pmat_in,
            const std::string& spin_type)
            : HamiltLR<T>(xc_kernel, nspin, naos, nocc, nvirt, ucell_in, orb_cutoff, gd_in, psi_ks_in, eig_ks,
#ifdef __EXX
                exx_lri_in, exx_alpha,
#endif
                gint_in, pot_in, kv_in, pX_in, pc_in, pmat_in, spin_type)
        {
            ModuleBase::TITLE("Z_vector_L", "Z_vector_L");
            this->DM_trans = LR_Util::make_unique<elecstate::DensityMatrix<T, T>>(&pmat_in, 1, kv_in.kvec_d, this->nk);
            // 1. $2\sum_bX_{ib}K_{ab}[D^X]-2\sum_jX_{ja}K_{ij}[D^X]$
            this->ops = new OperatorLRDiag<T>(eig_ks.c, pX_in[0], kv_in.get_nks(), nocc[0], nvirt[0]);
            // 2. $H_{ia}[T]$, equals to $2K_{ab}[T]$ when $T$ is symmetrized
            OperatorLRHxc<T>* op_hz = new OperatorLRHxc<T>(nspin, naos, nocc, nvirt, psi_ks_in,
                this->DM_trans, gint_in, pot_in, ucell_in, orb_cutoff, gd_in, kv_in, pX_in, pc_in, pmat_in,
                { 0 }, 2.0, ATYPE::CC_vo);
            this->ops->add(op_hz);
#ifdef __EXX
            // add EXX operators here
#endif
            this->cal_dm_trans = [&, this](const int& is, const T* X)->void
                {
                    const auto psi_ks_is = LR_Util::get_psi_spin(psi_ks_in, is, this->nk);
#ifdef __MPI
                    std::vector<ct::Tensor>  dm_trans_2d = cal_dm_trans_pblas(X, this->pX[is], psi_ks_is, pc_in, naos, nocc[is], nvirt[is], pmat_in);
                    for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos, pmat_in);
#else
                    std::vector<ct::Tensor>  dm_trans_2d = cal_dm_trans_blas(X, psi_ks_is, nocc[is], nvirt[is]);
                    for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos);
#endif
                    // LR_Util::print_tensor<T>(dm_trans_2d[0], "dm_trans_2d[0]", &pmat_in);
                    // tensor to vector, then set DMK
                    for (int ik = 0;ik < this->nk;++ik) { this->DM_trans->set_DMK_pointer(ik, dm_trans_2d[ik].data<T>()); }
                };
        }
    };
}