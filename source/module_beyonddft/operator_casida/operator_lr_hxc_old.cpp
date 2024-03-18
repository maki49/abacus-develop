#include "operator_lr_hxc.h"
#include <vector>
#include "module_base/blas_connector.h"
#include "module_beyonddft/utils/lr_util.h"
// #include "module_hamilt_lcao/hamilt_lcaodft/DM_gamma_2d_to_grid.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer_funcs.h"
#include "module_beyonddft/dm_trans/dm_trans.h"
#include "module_beyonddft/AX/AX.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
inline double conj(double a) { return a; }
inline std::complex<double> conj(std::complex<double> a) { return std::conj(a); }
namespace hamilt
{
    // for double
    template<typename T, typename Device>
    // psi::Psi<T> OperatorLRHxc<T, Device>::act(const psi::Psi<T>& psi_in) const
    void OperatorLRHxc<T, Device>::act(const psi::Psi<T>& psi_in, psi::Psi<T>& psi_out, const int nbands) const
    {
        ModuleBase::TITLE("OperatorLRHxc", "act");
        assert(nbands <= psi_in.get_nbands());
        const int& nks = this->kv.nks;

        /// initialize transion density matrix
        if (this->next_op != nullptr)
        {
            int prev_size = this->DM_trans.size();
            std::cout << "prev_size= " << prev_size << std::endl;
            std::cout << "nbands=" << nbands << std::endl;
            if (prev_size > nbands)for (int ib = nbands;ib < prev_size;++ib)delete this->DM_trans[ib];
            this->DM_trans.resize(nbands);
            for (int ib = prev_size;ib < nbands;++ib)
            {
                this->DM_trans[ib] = new elecstate::DensityMatrix<T, T>(&this->kv, this->pmat, this->nspin);
                this->DM_trans[ib]->init_DMR(*this->hR);
            }
        }

        psi::Psi<T> psi_in_bfirst = LR_Util::k1_to_bfirst_wrapper(psi_in, nks, this->pX->get_local_size());
        psi::Psi<T> psi_out_bfirst = LR_Util::k1_to_bfirst_wrapper(psi_out, nks, this->pX->get_local_size());

        const int& lgd = gint->gridt->lgd;
        for (int ib = 0;ib < nbands;++ib)
        {
            int ib_dm = (this->next_op == nullptr) ? 0 : ib;
            GlobalV::ofs_running << "ib=" << ib << std::endl;
            psi_in_bfirst.fix_b(ib);
            psi_out_bfirst.fix_b(ib);

            // 1. transition density matrix (nks)
            // GlobalV::ofs_running << "1. transition density matrix" << std::endl;
            // // output c and X
            // GlobalV::ofs_running << "nvirt:" << nvirt << " nocc:" << nocc << std::endl;
            // GlobalV::ofs_running << "nvirt local:" << pX->get_row_size();
            // GlobalV::ofs_running << "nocc local:" << pX->get_col_size() << std::endl;

            // GlobalV::ofs_running << "X: " << std::endl;
            // for (int ik = 0;ik < nks;++ik)
            // {
            //     GlobalV::ofs_running << "ik=" << ik << ", kv=" << this->kv.kvec_d[ik][0] << ", " << this->kv.kvec_d[ik][1] << ", " << this->kv.kvec_d[ik][2] << std::endl;
            //     for (int j = 0;j < pX->get_col_size();++j)  //nbands
            //     {
            //         for (int i = 0;i < pX->get_row_size();++i)  //nlocal
            //             GlobalV::ofs_running << psi_in_bfirst.get_pointer()[ik * pX->get_local_size() + j * pX->get_row_size() + i] << " ";
            //         GlobalV::ofs_running << std::endl;
            //     }
            // }
            // GlobalV::ofs_running << std::endl;
            // GlobalV::ofs_running << "C: " << std::endl;
            // GlobalV::ofs_running << "nks:" << psi_ks->get_nk() << "naos:" << naos << " nbands:" << nocc + nvirt << std::endl;
            // GlobalV::ofs_running << "naos local:" << pc->get_row_size();
            // GlobalV::ofs_running << "nbands local:" << pc->get_col_size() << std::endl;
            // GlobalV::ofs_running.setf(std::ios::fixed);
            // for (int ik = 0;ik < nks;++ik)
            // {
            //     psi_ks->fix_k(ik);
            //     GlobalV::ofs_running << "ik=" << ik << ", kv=" << this->kv.kvec_d[ik][0] << ", " << this->kv.kvec_d[ik][1] << ", " << this->kv.kvec_d[ik][2] << std::endl;
            //     for (int j = 0;j < pc->get_col_size();++j)  //nbands
            //     {
            //         for (int i = 0;i < pc->get_row_size();++i)  //nlocal
            //             GlobalV::ofs_running << std::setprecision(6) << psi_ks->get_pointer()[j * pc->get_row_size() + i] << " ";
            //         GlobalV::ofs_running << std::endl;
            //     }
            // }

#ifdef __MPI
            std::vector<container::Tensor>  dm_trans_2d = cal_dm_trans_pblas(psi_in_bfirst, *pX, *psi_ks, *pc, naos, nocc, nvirt, *pmat);
            // GlobalV::ofs_running << "1. Dm_trans before symmetrization: " << std::endl;
            // GlobalV::ofs_running << "local row:" << pmat->get_row_size() << " local col:" << pmat->get_col_size() << std::endl;
            // for (int ik = 0;ik < nks;++ik)
            // {
            //     GlobalV::ofs_running << "ik=" << ik << ", kv=" << this->kv.kvec_d[ik][0] << ", " << this->kv.kvec_d[ik][1] << ", " << this->kv.kvec_d[ik][2] << std::endl;
            //     for (int j = 0;j < pmat->get_col_size();++j)
            //     {
            //         for (int i = 0;i < pmat->get_row_size();++i)
            //             GlobalV::ofs_running << std::setprecision(6) << dm_trans_2d[ik].data<T>()[j * pmat->get_row_size() + i] << " ";
            //         GlobalV::ofs_running << std::endl;
            //     }
            // }

            if (this->tdm_sym) for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos, *pmat);
            // test: force D(-k)=D(k)^* in k444
            // for (int i = 0; i < pmat->get_local_size();++i) dm_trans_2d[3].data<T>()[i] = conj(dm_trans_2d[1].data<T>()[i]);
#else
            std::vector<container::Tensor>  dm_trans_2d = cal_dm_trans_blas(psi_in_bfirst, psi_ks, nocc, nvirt);
            if (this->tdm_sym) for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos);
#endif
            // GlobalV::ofs_running << "1. Dm_trans after symmetrization: " << std::endl;
            // GlobalV::ofs_running << "local row:" << pmat->get_row_size() << " local col:" << pmat->get_col_size() << std::endl;
            // for (int ik = 0;ik < nks;++ik)
            // {
            //     GlobalV::ofs_running << "ik=" << ik << ", kv=" << this->kv.kvec_d[ik][0] << ", " << this->kv.kvec_d[ik][1] << ", " << this->kv.kvec_d[ik][2] << std::endl;
            //     for (int j = 0;j < pmat->get_col_size();++j)
            //     {
            //         for (int i = 0;i < pmat->get_row_size();++i)
            //             GlobalV::ofs_running << std::setprecision(6) << dm_trans_2d[ik].data<T>()[j * pmat->get_row_size() + i] << " ";
            //         GlobalV::ofs_running << std::endl;
            //     }
            // }

            // tensor to vector, then set DMK
            for (int isk = 0;isk < nks;++isk)this->DM_trans[ib_dm]->set_DMK_pointer(isk, dm_trans_2d[isk].data<T>());

            // use cal_DMR to get DMR form DMK by FT
            this->DM_trans[ib_dm]->cal_DMR();  //DM_trans->get_DMR_vector() is 2d-block parallized
            GlobalV::ofs_running << "return cal_DMR (outside)" << std::endl;
            // 2d block to grid
            // new interface: transfer_DM2DtoGrid, set DMRGint for the next step Gint
            // this->gint->transfer_DM2DtoGrid(this->gint->get_DMRGint());//err?
            this->gint->transfer_DM2DtoGrid(this->DM_trans[ib_dm]->get_DMR_vector());
            // GlobalV::ofs_running << "return transfer_DMR (outside)" << std::endl;
            // GlobalV::ofs_running << "2. DM(R) (2d): " << std::endl;
            // for (auto& dr : this->DM_trans[ib_dm]->get_DMR_vector())
            // {
            //     for (int ia = 0;ia < GlobalC::ucell.nat;ia++)
            //         for (int ja = 0;ja < GlobalC::ucell.nat;ja++)
            //         {
            //             auto ap = dr->find_pair(ia, ja);
            //             // GlobalV::ofs_running << "R-index size of atom pair(" << ia << ", " << ja << "): " << ap->get_R_size() << std::endl;
            //             // for (int iR = 0;iR < ap->get_R_size();++iR)
            //             int iR = 0;
            //             {
            //                 GlobalV::ofs_running << "R(" << ap->get_R_index(iR)[0] << ", " << ap->get_R_index(iR)[1] << ", " << ap->get_R_index(iR)[2] << "): ";
            //                 auto ptr = ap->get_HR_values(iR).get_pointer();
            //                 int size = ap->get_size();
            //                 for (int i = 0;i < size;++i)GlobalV::ofs_running << ptr[i] << " ";
            //                 GlobalV::ofs_running << std::endl;
            //             }
            //         }
            // }
            // GlobalV::ofs_running << "2. DM(R) (Grid): " << std::endl;
            // for (auto& dr : this->gint->get_DMRGint())
            // {
            //     for (int ia = 0;ia < GlobalC::ucell.nat;ia++)
            //         for (int ja = 0;ja < GlobalC::ucell.nat;ja++)
            //         {
            //             auto ap = dr->find_pair(ia, ja);
            //             // GlobalV::ofs_running << "R-index size of atom pair(" << ia << ", " << ja << "): " << ap->get_R_size() << std::endl;
            //             // for (int iR = 0;iR < ap->get_R_size();++iR)
            //             int iR = 0;
            //             {
            //                 GlobalV::ofs_running << "R(" << ap->get_R_index(iR)[0] << ", " << ap->get_R_index(iR)[1] << ", " << ap->get_R_index(iR)[2] << "): ";
            //                 auto ptr = ap->get_HR_values(iR).get_pointer();
            //                 int size = ap->get_size();
            //                 GlobalV::ofs_running << "size of atom pair(" << ia << ", " << ja << "): " << size << std::endl;
            //                 for (int i = 0;i < size;++i)GlobalV::ofs_running << ptr[i] << " ";
            //                 GlobalV::ofs_running << std::endl;
            //             }
            //         }
            // }
            // // double*** dm_trans_grid;
            // LR_Util::new_p3(dm_trans_grid, nks, lgd, lgd);
            //         DMgamma_2dtoGrid dm2g;
            // #ifdef __MPI
            //         dm2g.setAlltoallvParameter(pmat->comm_2D, naos, pmat->blacs_ctxt, pmat->nb, lgd, gint->gridt->trace_lo);
            // #endif
            //         dm2g.cal_dk_gamma_from_2D(LR_Util::ten2mat_double(dm_trans_2d), dm_trans_grid, nks, naos, lgd, GlobalV::ofs_running);

            // 2. transition electron density
            // \f[ \tilde{\rho}(r)=\sum_{\mu_j, \mu_b}\tilde{\rho}_{\mu_j,\mu_b}\phi_{\mu_b}(r)\phi_{\mu_j}(r) \f]
            // GlobalV::ofs_running << "2. transition electron density" << std::endl;
            double** rho_trans;
            LR_Util::new_p2(rho_trans, nspin, this->pot->nrxx);
            for (int is = 0;is < nspin;++is)ModuleBase::GlobalFunc::ZEROS(rho_trans[is], this->pot->nrxx);
            Gint_inout inout_rho((double**)nullptr, rho_trans, Gint_Tools::job_type::rho, false);
            this->gint->cal_gint(&inout_rho);

            // GlobalV::ofs_running << "first 10 non-zero elements of rho_trans ";
            // int n = 0;int i = 0;
            // while (n < 10 && i < this->pot->nrxx)
            // {
            //     if (rho_trans[0][++i] - 0.0 > 1e-4) { GlobalV::ofs_running << rho_trans[0][i] << " ";++n; };
            // }

            // 3. v_hxc = f_hxc * rho_trans
            // GlobalV::ofs_running << "3. v_hxc = f_hxc * rho_trans" << std::endl;
            ModuleBase::matrix vr_hxc(nspin, this->pot->nrxx);   //grid
            this->pot->cal_v_eff(rho_trans, &GlobalC::ucell, vr_hxc);
            // GlobalV::ofs_running << "first 10 non-zero elements of vr_hxc: ";
            // n = 0;i = 0;
            // while (n < 10 && i < this->pot->nrxx)
            // {
            //     if (vr_hxc.c[++i] - 0.0 > 1e-4) { GlobalV::ofs_running << vr_hxc.c[i] << " ";++n; };
            // }
            // GlobalV::ofs_running << std::endl;

            // 4. V^{Hxc}_{\mu,\nu}=\int{dr} \phi_\mu(r) v_{Hxc}(r) \phi_\mu(r)
            // loop for nspin, or use current spin (how?)
            // results are stored in gint->pvpR_grid(gamma_only)
            // or gint_k->pvpR_reduced(multi_k)
            // GlobalV::ofs_running << "4.Vxc" << std::endl;
            std::vector<ct::Tensor> v_hxc_2d(nks, ct::Tensor(ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<Device>::value, { pmat->get_col_size(), pmat->get_row_size() }));
            for (auto& v : v_hxc_2d) v.zero();
            // V(R) for each spin
            for (int is = 0;is < nspin;++is)
            {
                double* vr_hxc_is = &vr_hxc.c[is * this->pot->nrxx];   //v(r) at current spin
                Gint_inout inout_vlocal(vr_hxc_is, is, Gint_Tools::job_type::vlocal);
                this->gint->get_hRGint()->set_zero();
                this->gint->cal_gint(&inout_vlocal);
            }
            this->hR->set_zero();   // clear hR for each bands
            this->gint->transfer_pvpR(this->hR);
            // GlobalV::ofs_running << "4.V(R):" << std::endl;
            // for (int ia = 0;ia < GlobalC::ucell.nat;ia++)
            //     for (int ja = 0;ja < GlobalC::ucell.nat;ja++)
            //     {
            //         auto ap = this->hR->find_pair(ia, ja);
            //         GlobalV::ofs_running << "R-index size of atom pair(" << ia << ", " << ja << "): " << ap->get_R_size() << std::endl;
            //         for (int iR = 0;iR < ap->get_R_size();++iR)
            //         {
            //             GlobalV::ofs_running << "R(" << ap->get_R_index(iR)[0] << ", " << ap->get_R_index(iR)[1] << ", " << ap->get_R_index(iR)[2] << "): ";
            //             auto ptr = ap->get_HR_values(iR).get_pointer();
            //             int size = ap->get_size();
            //             for (int i = 0;i < size;++i)GlobalV::ofs_running << ptr[i] << " ";
            //             GlobalV::ofs_running << std::endl;
            //         }
            //     }
            // V(R)->V(k)
            int nrow = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER() ? this->pmat->get_row_size() : this->pmat->get_col_size();
            for (int isk = 0;isk < nks;++isk)
            {
                // this->gint->vl_grid_to_2D(this->gint->get_pvpR_grid(), *pmat, lgd, (is == 0), v_hxc_2d[is].c, setter);
                hamilt::folding_HR(*this->hR, v_hxc_2d[isk].data<T>(), this->kv.kvec_d[isk], nrow, 1);            // V(R) -> V(k)
            }
            // GlobalV::ofs_running << "4. V(k)" << std::endl;
            // for (int isk = 0;isk < nks;++isk)
            // {
            //     for (int j = 0; j < pmat->get_col_size();++j)
            //     {
            //         for (int i = 0;i < pmat->get_row_size();++i)
            //             GlobalV::ofs_running << v_hxc_2d[isk].data<T>()[j * pmat->get_row_size() + i] << " ";
            //         GlobalV::ofs_running << std::endl;
            //     }
            // }
            // clear useless matrices
            // LR_Util::delete_p3(dm_trans_grid, nks, lgd);
            LR_Util::delete_p2(rho_trans, nspin);

            // GlobalV::ofs_running << "5.AX" << std::endl;
            // 5. [AX]^{Hxc}_{ai}=\sum_{\mu,\nu}c^*_{a,\mu,}V^{Hxc}_{\mu,\nu}c_{\nu,i}
#ifdef __MPI
            cal_AX_pblas(v_hxc_2d, *this->pmat, *this->psi_ks, *this->pc, naos, nocc, nvirt, *this->pX, psi_out_bfirst);
#else
            cal_AX_blas(v_hxc_2d, *this->psi_ks, nocc, nvirt, psi_out_bfirst);
#endif
            /// output AX
            // psi_out_bfirst.fix_k(0);
            // for (int ik = 0;ik < psi_out_bfirst.get_nk();++ik)
            // {
            //     assert(psi_out_bfirst.get_nbasis() == pX->get_local_size());
            //     for (int i = 0;i < psi_out_bfirst.get_nbasis();++i)
            //         GlobalV::ofs_running << psi_out_bfirst(ik, i) << " ";
            //     GlobalV::ofs_running << std::endl;
            // }
        }
    }
    template class OperatorLRHxc<double>;
    template class OperatorLRHxc<std::complex<double>>;
}