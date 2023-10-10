#pragma once
#include "operatorA_hxc.h"
#include <vector>
#include "module_base/blas_connector.h"
#include "module_beyonddft/utils/lr_util.h"
// #include "module_hamilt_lcao/hamilt_lcaodft/DM_gamma_2d_to_grid.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer_funcs.h"
#include "module_beyonddft/dm_trans/dm_trans.h"
#include "module_beyonddft/AX/AX.h"
namespace hamilt
{
    // for double
    template<typename T, typename Device>
    // psi::Psi<T> OperatorA_Hxc<T, Device>::act(const psi::Psi<T>& psi_in) const
    void OperatorA_Hxc<T, Device>::act(const psi::Psi<T>& psi_in, psi::Psi<T>& psi_out, const int nbands) const
    {
        ModuleBase::TITLE("OperatorA_Hxc", "act");

        assert(nbands <= psi_in.get_nbands());
        psi::Psi<T> psi_in_bfirst = LR_Util::k1_to_bfirst_wrapper(psi_in, this->nsk, this->pX->get_local_size());
        psi::Psi<T> psi_out_bfirst = LR_Util::k1_to_bfirst_wrapper(psi_out, this->nsk, this->pX->get_local_size());

        const int& lgd = gint->gridt->lgd;
        for (int ib = 0;ib < nbands;++ib)
        {
            GlobalV::ofs_running << "ib=" << ib << std::endl;
            psi_in_bfirst.fix_b(ib);
            psi_out_bfirst.fix_b(ib);

            // 1. transition density matrix (nsk)
            GlobalV::ofs_running << "1. transition density matrix" << std::endl;
            // multi-k needs k-to-R FT
#ifdef __MPI
            std::vector<container::Tensor>  dm_trans_2d = cal_dm_trans_pblas(psi_in_bfirst, *pX, *psi_ks, *pc, naos, nocc, nvirt, *pmat);
#else
            std::vector<container::Tensor>  dm_trans_2d = cal_dm_trans_blas(psi_in_bfirst, psi_ks, nocc, nvirt);
#endif
            // tensor to vector, then set DMK
            for (int isk = 0;isk < this->nsk;++isk)this->DM_trans->set_DMK_pointer(isk, dm_trans_2d[isk].data<T>());

            // use cal_DMR to get DMR form DMK by FT
            this->DM_trans->cal_DMR();
            GlobalV::ofs_running << "return cal_DMR (outside)" << std::endl;
            // 2d block to grid
            // new interface: transfer_DM2DtoGrid, set DMRGint for the next step Gint
            this->gint->transfer_DM2DtoGrid(this->DM_trans->get_DMR_vector());
            GlobalV::ofs_running << "return transfer_DMR (outside)" << std::endl;
            // double*** dm_trans_grid;
            // LR_Util::new_p3(dm_trans_grid, nsk, lgd, lgd);
            //         DMgamma_2dtoGrid dm2g;
            // #ifdef __MPI
            //         dm2g.setAlltoallvParameter(pmat->comm_2D, naos, pmat->blacs_ctxt, pmat->nb, lgd, gint->gridt->trace_lo);
            // #endif
            //         dm2g.cal_dk_gamma_from_2D(LR_Util::ten2mat_double(dm_trans_2d), dm_trans_grid, nsk, naos, lgd, GlobalV::ofs_running);

            // 2. transition electron density
            GlobalV::ofs_running << "2. transition electron density" << std::endl;
            double** rho_trans;
            LR_Util::new_p2(rho_trans, nspin, this->pot->nrxx);
            for (int is = 0;is < nspin;++is)ModuleBase::GlobalFunc::ZEROS(rho_trans[is], this->pot->nrxx);
            Gint_inout inout_rho((double**)nullptr, rho_trans, Gint_Tools::job_type::rho);
            this->gint->cal_gint(&inout_rho);

            // 3. v_hxc = f_hxc * rho_trans
            GlobalV::ofs_running << "3. v_hxc = f_hxc * rho_trans" << std::endl;
            ModuleBase::matrix vr_hxc(nspin, this->pot->nrxx);   //grid
            this->pot->cal_v_eff(rho_trans, &GlobalC::ucell, vr_hxc);

            // 4. V^{Hxc}_{\mu,\nu}=\int{dr} \phi_\mu(r) v_{Hxc}(r) \phi_\mu(r)
            // loop for nspin, or use current spin (how?)
            // results are stored in gint->pvpR_grid(gamma_only)
            // or gint_k->pvpR_reduced(multi_k)
            GlobalV::ofs_running << "4.Vxc" << std::endl;
            std::vector<ct::Tensor> v_hxc_2d(nsk, ct::Tensor(ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, { pmat->get_col_size(), pmat->get_row_size() }));
            // auto setter = [this](const int& iw1_all, const int& iw2_all, const double& v, double* out) {
            //     const int ir = this->pmat->global2local_row(iw1_all);
            //     const int ic = this->pmat->global2local_col(iw2_all);
            //     out[ic * this->pmat->nrow + ir] += v;
            //     };
            // V(R) for each spin
            for (int is = 0;is < nspin;++is)
            {
                double* vr_hxc_is = &vr_hxc.c[is * this->pot->nrxx];   //v(r) at current spin
                Gint_inout inout_vlocal(vr_hxc_is, is, Gint_Tools::job_type::vlocal);
                this->gint->get_hRGint()->set_zero();
                this->gint->cal_gint(&inout_vlocal);
                GlobalV::ofs_running << "return gint(vlocal, outside)" << std::endl;
            }
            GlobalV::ofs_running << "return gint(vlocal, outof spin loop)" << std::endl;
            this->gint->transfer_pvpR(this->hR);
            // V(R)->V(k)
            int nrow = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER() ? this->pmat->get_row_size() : this->pmat->get_col_size();
            for (int isk = 0;isk < this->nsk;++isk)
            {
                // this->gint->vl_grid_to_2D(this->gint->get_pvpR_grid(), *pmat, lgd, (is == 0), v_hxc_2d[is].c, setter);
                hamilt::folding_HR(*this->hR, v_hxc_2d[isk].data<T>(), this->kvec_d[isk], nrow, 1);            // V(R) -> V(k)
            }
            // clear useless matrices
            // LR_Util::delete_p3(dm_trans_grid, nsk, lgd);
            LR_Util::delete_p2(rho_trans, nspin);

            GlobalV::ofs_running << "5.AX" << std::endl;
            // 5. [AX]^{Hxc}_{ai}=\sum_{\mu,\nu}c^*_{a,\mu,}V^{Hxc}_{\mu,\nu}c_{\nu,i}
#ifdef __MPI
            cal_AX_pblas(v_hxc_2d, *this->pmat, *this->psi_ks, *this->pc, naos, nocc, nvirt, *this->pX, psi_out_bfirst);
#else
            cal_AX_blas(v_hxc_2d, *this->psi_ks, nocc, nvirt, psi_out_bfirst);
#endif
            GlobalV::ofs_running << "return b-loop" << std::endl;
        }
        GlobalV::ofs_running << "return act (inside)" << std::endl;
    }
}