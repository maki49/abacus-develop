#include "operatorA_hxc.h"
#include <vector>
#include "module_base/blas_connector.h"
#include "utils/lr_util.h"

namespace hamilt
{
    void OperatorA_Hxc_Double::act()
    {
        auto block2grid = [](std::vector<ModuleBase::matrix> block, double*** grid)->void {};
        // 1. transition density matrix
        std::vector<ModuleBase::matrix> dm_trans = cal_dm_trans_blas(*px, *pc);
        double*** dm_trans_grid = LR_Util::new_p3(nspin, naos_local_grid, naos_local_grid);
        block2grid(dm_trans, dm_trans_grid);

        // 2. transition electron density
        double** rho_trans = LR_Util::new_p2(nspin, this->gint_g->nbxx);  // is nbxx local grid num ? 
        Gint_inout inout_rho(dm_trans_grid, rho_trans, Gint_Tools::job_type::rho);
        this->gint_g->cal_gint(&inout_rho);

        // 3. v_hxc = f_hxc * rho_trans
        this->pot->update_from_charge(rho_trans, GlobalC::ucell);

        // 4. V^{Hxc}_{\mu,\nu}=\int{dr} \phi_\mu(r) v_{Hxc}(r) \phi_\mu(r)
        // loop for nspin, or use current spin (how?)
        // results are stored in gint_g->pvpR_grid(gamma_only)
        // or gint_k->pvpR_reduced(multi_k)

        std::vector<ModuleBase::matrix> v_hxc_local(nspin);   // 2D local matrix)
        for (int is = 0;is < this->nspin;++is)
        {
            const double* v1_hxc_grid = this->pot->get_effective_v(is);
            Gint_intout inout_vlocal(v1_hxc_grid, is, Gint_Tools::job_type::vlocal);
            // this->gint_g->cal_gint(&inout);
            bool new_e_iteration = false;   // what is this?
            this->gint_g->cal_vlocal(&inout_vlocal, new_e_iteration);

            // grid-to-2d needs refactor !
            v_hxc_2d[is].create(naos_local_row, naos_local_col);
            //LR_Util::grid2block(this->px, this->pc, this->gint_g->pvpR_grid, v_hxc_local.c);
        }


        // 5. [AX]^{Hxc}_{ai}=\sum_{\mu,\nu}c^*_{a,\mu,}V^{Hxc}_{\mu,\nu}c_{\nu,i}
        // use 2 pzgemms
        this->cal_AX_cVc(v_hxc_2d, this->px);
        // result is in which "psi" ? X？

        // final clear
        LR_Util::delete_p3(dm_trans_grid);
        LR_Util::delete_p2(rho_trans);
        return;
    }
}