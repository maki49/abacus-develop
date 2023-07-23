#pragma once
#include "operatorA_hxc.h"
#include <vector>
#include "module_base/blas_connector.h"
#include "module_beyonddft/utils/lr_util.h"
#include "module_hamilt_lcao/hamilt_lcaodft/DM_gamma_2d_to_grid.h"
#include "module_beyonddft/density/dm_trans.h"
#include "module_beyonddft/AX/AX.h"
namespace hamilt
{
    // for double
    template<typename T, typename Device>
    // psi::Psi<T> OperatorA_Hxc<T, Device>::act(const psi::Psi<T>& psi_in) const
    void OperatorA_Hxc<T, Device>::act(const psi::Psi<T>& psi_in, psi::Psi<T>& psi_out) const
    {
        ModuleBase::TITLE("OperatorA_Hxc", "act");
        const int& lgd = gg->gridt->lgd;
        // gamma-only now
        // 1. transition density matrix
        // multi-k needs k-to-R FT
#ifdef __MPI
        std::vector<container::Tensor>  dm_trans_2d = cal_dm_trans_pblas(psi_in, *pX, *psi_ks, *pc, naos, nocc, nvirt, *pmat);
#else
        std::vector<container::Tensor>  dm_trans_2d = cal_dm_trans_blas(*pX, *pc, nocc, nvirt);
#endif
        double*** dm_trans_grid;
        LR_Util::new_p3(dm_trans_grid, nsk, lgd, lgd);
        //2d block to grid
        DMgamma_2dtoGrid dm2g;
#ifdef __MPI
        dm2g.setAlltoallvParameter(pmat->comm_2D, naos, pmat->blacs_ctxt, pmat->nb, lgd, gg->gridt->trace_lo);
#endif
        dm2g.cal_dk_gamma_from_2D(LR_Util::ten2mat_double(dm_trans_2d), dm_trans_grid, nsk, naos, lgd, GlobalV::ofs_running);

        // 2. transition electron density
        double** rho_trans;
        LR_Util::new_p2(rho_trans, nsk, this->pot->nrxx);
        Gint_inout inout_rho(dm_trans_grid, rho_trans, Gint_Tools::job_type::rho);
        this->gg->cal_gint(&inout_rho);

        // 3. v_hxc = f_hxc * rho_trans
        ModuleBase::matrix vr_hxc(nsk, this->pot->nrxx);   //grid
        this->pot->cal_v_eff(rho_trans, &GlobalC::ucell, vr_hxc);

        // 4. V^{Hxc}_{\mu,\nu}=\int{dr} \phi_\mu(r) v_{Hxc}(r) \phi_\mu(r)
        // loop for nsk, or use current spin (how?)
        // results are stored in gg->pvpR_grid(gamma_only)
        // or gint_k->pvpR_reduced(multi_k)
        std::vector<ModuleBase::matrix> v_hxc_2d(nsk);
        auto setter = [this](const int& iw1_all, const int& iw2_all, const double& v, double* out) {
            const int ir = this->pmat->global2local_row(iw1_all);
            const int ic = this->pmat->global2local_col(iw2_all);
            out[ic * this->pmat->nrow + ir] += v;
            };
        for (int is = 0;is < this->nsk;++is)
        {
            double* vr_hxc_is = &vr_hxc.c[is * this->pot->nrxx];   //current spin
            Gint_inout inout_vlocal(vr_hxc_is, Gint_Tools::job_type::vlocal);
            this->gg->cal_gint(&inout_vlocal);
            v_hxc_2d[is].create(pmat->get_col_size(), pmat->get_row_size());
            this->gg->transfer_pvpR(this->gg->get_hRGint());
            // this->gg->vl_grid_to_2D(this->gg->get_pvpR_grid(), *pmat, lgd, (is == 0), v_hxc_2d[is].c, setter);
        }
        // clear useless matrices
        LR_Util::delete_p3(dm_trans_grid, nsk, lgd);
        LR_Util::delete_p2(rho_trans, nsk);

        // 5. [AX]^{Hxc}_{ai}=\sum_{\mu,\nu}c^*_{a,\mu,}V^{Hxc}_{\mu,\nu}c_{\nu,i}
#ifdef __MPI
        cal_AX_pblas(LR_Util::mat2ten_double(v_hxc_2d), *this->pmat, *this->psi_ks, *this->pc, naos, nocc, nvirt, *this->pX, psi_out);
#else
        cal_AX_blas(LR_Util::mat2ten_double(v_hxc_2d), *this->psi_ks, nocc, nvirt, psi_out);
#endif
    }
}