#include "operator_lr_hxc_common.h"
#include "module_base/timer.h"
#include "module_beyonddft/utils/lr_util.h"
#include "module_beyonddft/utils/lr_util_hcontainer.h"
#include "module_beyonddft/utils/lr_util_print.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer_funcs.h"
#include "module_beyonddft/dm_trans/dm_trans.h"
#include "module_beyonddft/AX/AX.h"
#include "module_beyonddft/dm_diff/dm_diff.h"
inline double conj(double a) { return a; }
inline std::complex<double> conj(std::complex<double> a) { return std::conj(a); }

namespace hamilt
{
    template<typename T, typename Device>
    void OperatorLRHxcCommon<T, Device>::act(const psi::Psi<T>& psi_in, psi::Psi<T>& psi_out, const int nbands) const
    {
        ModuleBase::TITLE("OperatorLRHxcCommon", "act");
        assert(nbands <= psi_in.get_nbands());
        const int& nks = this->kv.nks;

        //print 
        // if (this->first_print) LR_Util::print_psi_kfirst(*psi_ks, "psi_ks");

        this->init_DM_trans(nbands, this->DM_trans);    // initialize transion density matrix

        psi::Psi<T> psi_in_bfirst = LR_Util::k1_to_bfirst_wrapper(psi_in, nks, this->pX->get_local_size());
        psi::Psi<T> psi_out_bfirst = LR_Util::k1_to_bfirst_wrapper(psi_out, nks, this->pX->get_local_size());

        const int& lgd = gint->gridt->lgd;
        for (int ib = 0;ib < nbands;++ib)
        {
            // if (this->first_print) LR_Util::print_psi_bandfirst(psi_in_bfirst, "psi_in_bfirst", ib);

            // if Hxc-only, the memory of single-band DM_trans is enough.
            // if followed by EXX, we need to allocate memory for all bands.
            int ib_dm = (this->next_op == nullptr) ? 0 : ib;
            psi_in_bfirst.fix_b(ib);
            psi_out_bfirst.fix_b(ib);

            // 1. transition density matrix
            switch (this->dm_rs)
            {
            case DM_TYPE::T:
#ifdef __MPI
                std::vector<container::Tensor>  dm_trans_2d = cal_dm_diff_pblas(psi_in_bfirst, *pX, *psi_ks, *pc, naos, nocc, nvirt, *pmat);
#else
                std::vector<container::Tensor>  dm_trans_2d = cal_dm_diff_blas(psi_in_bfirst, psi_ks, naos, nocc, nvirt);
#endif
                break;
            case DM_TYPE::X, DM_TYPE::Z:
#ifdef __MPI
                std::vector<container::Tensor>  dm_trans_2d = cal_dm_trans_pblas(psi_in_bfirst, *pX, *psi_ks, *pc, naos, nocc, nvirt, *pmat);
                if (this->tdm_sym) for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos, *pmat);
#else
                std::vector<container::Tensor>  dm_trans_2d = cal_dm_trans_blas(psi_in_bfirst, psi_ks, nocc, nvirt);
                if (this->tdm_sym) for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos);
#endif
                break;
            default:
                throw std::runtime_error("Unknown DM_TYPE");
                break;
            }

            // tensor to vector, then set DMK
            for (int isk = 0;isk < nks;++isk)this->DM_trans[ib_dm]->set_DMK_pointer(isk, dm_trans_2d[isk].data<T>());

            // if (this->first_print)
            //     for (int ik = 0;ik < nks;++ik)
            //         LR_Util::print_tensor<std::complex<double>>(dm_trans_2d[ik], "1.DMK[ik=" + std::to_string(ik) + "]", this->pmat);

            // use cal_DMR to get DMR form DMK by FT
            this->DM_trans[ib_dm]->cal_DMR();  //DM_trans->get_DMR_vector() is 2d-block parallized
            // LR_Util::print_DMR(*this->DM_trans[0], ucell.nat, "DM(R) (complex)");

            // ========================= begin grid calculation=========================
            this->grid_calculation(nbands, ib_dm);   //DM(R) to H(R)
            // ========================= end grid calculation =========================

            // V(R)->V(k)
            std::vector<ct::Tensor> v_hxc_2d(this->kv.nks,
                ct::Tensor(ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<psi::DEVICE_CPU>::value,
                    { pmat->get_col_size(), pmat->get_row_size() }));
            for (auto& v : v_hxc_2d) v.zero();
            int nrow = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER() ? this->pmat->get_row_size() : this->pmat->get_col_size();
            for (int isk = 0;isk < nks;++isk)
                hamilt::folding_HR(*this->hR, v_hxc_2d[isk].data<T>(), this->kv.kvec_d[isk], nrow, 1);            // V(R) -> V(k)
            // LR_Util::print_HR(*this->hR, this->ucell.nat, "4.VR");
            // if (this->first_print)
            //     for (int ik = 0;ik < nks;++ik)
            //         LR_Util::print_tensor<T>(v_hxc_2d[ik], "4.V(k)[ik=" + std::to_string(ik) + "]", this->pmat);

            // 5. [AX]^{Hxc}_{ai}=\sum_{\mu,\nu}c^*_{a,\mu,}V^{Hxc}_{\mu,\nu}c_{\nu,i}
            switch (this->dm_pq)
            {
            case DM_TYPE:
                /* code */
                break;
            default:    // C_onebase_ai
#ifdef __MPI
                cal_AX_pblas(v_hxc_2d, *this->pmat, *this->psi_ks, *this->pc, naos, nocc, nvirt, *this->pX, psi_out_bfirst);
#else
                cal_AX_blas(v_hxc_2d, *this->psi_ks, nocc, nvirt, psi_out_bfirst);
#endif
                break;
            }
            // if (this->first_print) LR_Util::print_psi_bandfirst(psi_out_bfirst, "5.AX", ib);
        }
    }

    template class OperatorLRHxcCommon<double>;
    template class OperatorLRHxcCommon<std::complex<double>>;
}