#pragma once
#include "module_hamilt_general/hamilt.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_lr/Grad/xc/pot_grad_xc.h"
#include "module_lr/potentials/pot_hxc_lrtd.h"
#include "module_lr/operator_casida/operator_lr_hxc.h"
#include "module_basis/module_ao/parallel_orbitals.h"
namespace LR
{
    template<typename T>
    class Z_vector_R : public HamiltLR<T>
    {
        using ATYPE = typename OperatorLRHxc<T>::MO_TO_AO_TYPE;
    public:
        template<typename TGint>
        Z_vector_R(const std::string& xc_kernel,
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
            std::weak_ptr<PotHxcLR> pot_in,
            const K_Vectors& kv_in,
            const std::vector<Parallel_2D>& pX_in,
            const Parallel_2D& pc_in,
            const Parallel_Orbitals& pmat_in,
            const std::string& spin_type = "singlet")
            : HamiltLR<T>(xc_kernel, nspin, naos, nocc, nvirt, ucell_in, orb_cutoff, gd_in, psi_ks_in, eig_ks,
#ifdef __EXX
                exx_lri_in, exx_alpha,
#endif
                gint_in, pot_in, kv_in, pX_in, pc_in, pmat_in, spin_type)
        {
            ModuleBase::TITLE("Z_vector_R", "Z_vector_R");
            this->DM_trans = LR_Util::make_unique<elecstate::DensityMatrix<T, T>>(&pmat_in, 1, kv_in.kvec_d, this->nk);
            this->DM_diff = LR_Util::make_unique<elecstate::DensityMatrix<T, T>>(&pmat_in, 1, kv_in.kvec_d, this->nk);
            // 1. $2\sum_bX_{ib}K_{ab}[D^X]-2\sum_jX_{ja}K_{ij}[D^X]$
            this->ops = new OperatorLRHxc<T>(nspin, naos, nocc, nvirt, psi_ks_in,
                this->DM_trans, gint_in, pot_in, ucell_in, orb_cutoff, gd_in, kv_in, pX_in, pc_in, pmat_in,
                { 0 }, -2.0, ATYPE::CXC);
            // 2. $H_{ia}[T]$, equals to $2K_{ab}[T]$ when $T$ is symmetrized
            OperatorLRHxc<T>* op_ht = new OperatorLRHxc<T>(nspin, naos, nocc, nvirt, psi_ks_in,
                this->DM_diff, gint_in, pot_in, ucell_in, orb_cutoff, gd_in, kv_in, pX_in, pc_in, pmat_in,
                { 0 }, T(-2.0), ATYPE::CC_vo);
            this->ops->add(op_ht);
            // 3. $2\sum_{jb,kc} g^{xc}_{ia, jb, kc}X_{jb}X_{kc}$
            this->pot_grad = std::make_shared<PotGradXCLR>(pot_in.lock()->xc_kernel_components, pot_in.lock()->rho_basis, ucell_in, pot_in.lock()->nrxx);
            OperatorLRHxc<T>* op_gxc = new OperatorLRHxc<T>(nspin, naos, nocc, nvirt, psi_ks_in,
                this->DM_trans, gint_in, this->pot_grad, ucell_in, orb_cutoff, gd_in, kv_in, pX_in, pc_in, pmat_in,
                { 0 }, T(-2.0), ATYPE::CC_vo);
            this->ops->add(op_gxc);
#ifdef __EXX
            // add EXX operators here
#endif
            this->cal_dm_trans = [&, this](const int& is, const T* X)->void
                {
                    const auto psi_ks_is = LR_Util::get_psi_spin(psi_ks_in, is, this->nk);
#ifdef __MPI
                    std::vector<ct::Tensor> dm_trans_2d = cal_dm_trans_pblas(X, this->pX[is], psi_ks_is, pc_in, naos, nocc[is], nvirt[is], pmat_in);
                    for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos, pmat_in);
#else
                    std::vector<ct::Tensor> dm_trans_2d = cal_dm_trans_blas(X, psi_ks_is, nocc[is], nvirt[is]);
                    for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos);
#endif
                    for (int ik = 0;ik < this->nk;++ik) { this->DM_trans->set_DMK_pointer(ik, dm_trans_2d[ik].data<T>()); }
                };

            this->cal_dm_diff = [&, this](const int& is, const T* const X)->void
                {
                    const auto psi_ks_is = LR_Util::get_psi_spin(psi_ks_in, is, this->nk);
#ifdef __MPI
                    std::vector<ct::Tensor> dm_diff_2d = cal_dm_diff_pblas(X, this->pX[is], psi_ks_is, pc_in, naos, nocc[is], nvirt[is], pmat_in);
#else
                    std::vector<ct::Tensor> dm_diff_2d = cal_dm_diff_blas(X, psi_ks_is, naos, nocc[is], nvirt[is]);
#endif
                    for (int ik = 0;ik < this->nk;++ik) { this->DM_diff->set_DMK_pointer(ik, dm_diff_2d[ik].data<T>()); }
                };
        }
        virtual void hPsi(const T* const psi_in, T* const hpsi, const int ld_psi, const int nband) const override
        {
            assert(ld_psi == this->nk * this->pX[0].get_local_size());
            for (int ib = 0;ib < nband;++ib)
            {
                const int offset = ib * ld_psi;
                this->cal_dm_trans(0, psi_in + offset);  // transition density matrix
                this->cal_dm_diff(0, psi_in + offset);  // difference density matrix
                hamilt::Operator<T>* node(this->ops);
                while (node != nullptr)
                {
                    node->act(/*nband=*/1, ld_psi, /*npol=*/1, psi_in + offset, hpsi + offset);
                    node = (hamilt::Operator<T>*)(node->next_op);
                }
            }
        }

    private:
        std::unique_ptr<elecstate::DensityMatrix<T, T>> DM_diff;
        std::function<void(const int&, const T* const)> cal_dm_diff;
        std::shared_ptr<PotLRBase> pot_grad;
    };
}