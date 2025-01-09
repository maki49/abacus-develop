#pragma once
#include "module_hamilt_general/hamilt.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_lr/Grad/xc/pot_grad_xc.h"
#include "module_lr/potentials/pot_hxc_lrtd.h"
#include "module_lr/operator_casida/operator_lr_hxc.h"
#include "module_basis/module_ao/parallel_orbitals.h"
namespace LR
{
    // $\sum_a (\Omega - \epsilon_a)\delta_{ij}$
    // !!! Not yet support multi-k
    template<typename T>
    void add_sum_eig_minus_evirt(T* const inout,
        const double eig_ext_istate,
        const double* const eig_ks,
        const int nocc,
        const int nvirt,
        const Parallel_2D& p_occ_occ)
    {
        for (int i = 0;i < nocc;++i)
        {
            if (p_occ_occ.in_this_processor(i, i))    // diagonal elements in this processor
            {
                const int lrow = p_occ_occ.global2local_row(i);
                const int lcol = p_occ_occ.global2local_col(i);
                for (int a = 0;a < nvirt;++i)
                {
                    inout[lcol * p_occ_occ.get_row_size() + lrow] += (eig_ext_istate - eig_ks[/**ik, */nocc + i]);
                }
            }
        }
    }

    template<typename T, typename TGint>
    void cal_W_from_Z(T* const W,
        const T* const Z,
        const T* const X,
        const double* const eig,
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
        std::weak_ptr<Exx_LRI<T>> exx_lri,
        const double& exx_alpha,
#endif 
        TGint* gint_in,
        std::weak_ptr<PotHxcLR> pot,
        const K_Vectors& kv,
        const std::vector<Parallel_2D>& px,
        const Parallel_2D& pc,
        const Parallel_2D& p_occ_occ,   // < for W
        const Parallel_Orbitals& pmat,
        const std::string& spin_type = "singlet")
    {
        ModuleBase::TITLE("Z_vector_R", "Z_vector_R");
        using ATYPE = typename OperatorLRHxc<T>::MO_TO_AO_TYPE;
        const int nk = kv.get_nks() / nspin;
        // allocate memory for DMs
        elecstate::DensityMatrix<T, T> DM_trans(&pmat, 1, kv.kvec_d, nk);   //DX
        elecstate::DensityMatrix<T, T> DM_diff_relaxed(&pmat, 1, kv.kvec_d, nk);    //T+DZ

        /// operators
        // 1. 0.5$H_{ia}[T]$, equals to $K_{ab}[T]$ when $T$ is symmetrized
        OperatorLRHxc<T> op_ht(nspin, naos, nocc, nvirt, psi_ks_in,
            DM_diff_relaxed, gint_in, pot, ucell_in, orb_cutoff, gd_in, kv, px, pc, pmat,
            { 0 }, T(1.0), ATYPE::CC_oo);
        // 2. $2\sum_{jb,kc} g^{xc}_{ia, jb, kc}X_{jb}X_{kc}$
        PotGradXCLR pot_grad(pot.lock()->xc_kernel_components, pot.lock()->rho_basis, ucell_in, pot.lock()->nrxx);
        OperatorLRHxc<T> op_gxc(nspin, naos, nocc, nvirt, psi_ks_in,
            DM_trans, gint_in, pot_grad, ucell_in, orb_cutoff, gd_in, kv, px, pc, pmat,
            { 0 }, T(-2.0), ATYPE::CC_oo);
#ifdef __EXX
        // add EXX operators here
#endif
        auto cal_dm_trans = [&](const int is, const T* const x_ptr)->void //DX
            {
                const auto psi_ks_is = LR_Util::get_psi_spin(psi_ks_in, is, nk);
#ifdef __MPI
                std::vector<ct::Tensor> dm_trans_2d = cal_dm_trans_pblas(x_ptr, px[is], psi_ks_is, pc, naos, nocc[is], nvirt[is], pmat);
                for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos, pmat);
#else
                std::vector<ct::Tensor> dm_trans_2d = cal_dm_trans_blas(x_ptr, psi_ks_is, nocc[is], nvirt[is]);
                for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos);
#endif
                for (int ik = 0;ik < nk;++ik) { DM_trans->set_DMK_pointer(ik, dm_trans_2d[ik].data<T>()); }
            };

        auto cal_dm_diff_relaxed = [&](const int& is, const T* const x_ptr, const T* const z_ptr)->void  // T+DZ
            {
                const auto psi_ks_is = LR_Util::get_psi_spin(psi_ks_in, is, nk);
#ifdef __MPI
                std::vector<ct::Tensor> z_2d = cal_dm_trans_pblas(z_ptr, px[is], psi_ks_is, pc, naos, nocc[is], nvirt[is], pmat);
                for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos, pmat);
#else
                std::vector<ct::Tensor> z_2d = cal_dm_trans_blas(z_ptr, psi_ks_is, nocc[is], nvirt[is]);
                for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos);
#endif

#ifdef __MPI
                std::vector<ct::Tensor> dm_diff_2d = cal_dm_diff_pblas(x_ptr, px[is], psi_ks_is, pc, naos, nocc[is], nvirt[is], pmat);
#else
                std::vector<ct::Tensor> dm_diff_2d = cal_dm_diff_blas(x_ptr, psi_ks_is, naos, nocc[is], nvirt[is]);
#endif
                for (int ik = 0;ik < nk;++ik)
                {
                    dm_diff_2d[i] += z_2d[i];
                    DM_diff->set_DMK_pointer(ik, dm_diff_2d[ik].data<T>());
                }
            };

        // act the operators onto each state
        const int ld_vo = nk * px[0].get_local_size();
        const int ld_oo = nk * p_occ_occ.get_local_size();
        for (int ib = 0;ib < nstates;++ib)
        {
            const int offset_vo = ib * ld_vo;
            const int offset_oo = ib * ld_oo;
            cal_dm_trans(0, X + offset_vo, Z + offset_vo);  // transition density matrix DX
            cal_dm_diff_relaxed(0, X + offset_vo, Z + offset_vo);  // relaxed difference density matrix T+DZ
            // the 3 terms
            op_ht.act(/*nband=*/1, ld_vo, /*npol=*/1, X + offset_vo, W + offset_oo);
            op_gxc.act(/*nband=*/1, ld_vo, /*npol=*/1, X + offset_vo, W + offset_oo);
            add_sum_eig_minus_evirt(W + offset_oo, eig[ib], eig_ks.data(), nocc[0], nvirt[0], p_occ_occ);
        }
    }
}