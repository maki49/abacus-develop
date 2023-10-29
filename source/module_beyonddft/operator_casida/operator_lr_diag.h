#pragma once
#include "module_beyonddft/utils/lr_util.h"
#include "module_hsolver/kernels/math_kernel_op.h"
#include "module_base/global_variable.h"
namespace hamilt
{
    /// @brief  Diag part of A operator: [AX]_iak = (e_ak - e_ik) X_iak
    template<typename T = double, typename Device = psi::DEVICE_CPU>
    class OperatorLRDiag : public Operator<T, Device>
    {
    public:
        OperatorLRDiag(const ModuleBase::matrix& eig_ks_in, const Parallel_2D* pX_in, const int& nks_in, const int& nspin_in, const int& nocc_in, const int& nvirt_in)
            : eig_ks(eig_ks_in), pX(pX_in), nks(nks_in), nspin(nspin_in), nocc(nocc_in), nvirt(nvirt_in),
            nsk(std::is_same<T, double>::value ? nspin_in : nks_in)
        {   // calculate the difference of eigenvalues
            ModuleBase::TITLE("OperatorLRDiag", "OperatorLRDiag");
            this->act_type = 2;
            this->eig_ks_diff.create(nks, pX->get_local_size(), false);
            for (int ik = 0;ik < nks;++ik)
                for (int io = 0;io < pX->get_col_size();++io)    //nocc_local
                    for (int iv = 0;iv < pX->get_row_size();++iv)    //nvirt_local
                    {
                        int io_g = pX->local2global_col(io);
                        int iv_g = pX->local2global_row(iv);
                        this->eig_ks_diff(ik, io * pX->get_row_size() + iv) = eig_ks(ik, nocc + iv_g) - eig_ks(ik, io_g);
                    }
            // // output eig_ks_diff
            // GlobalV::ofs_running << "eig_ks_diff" << std::endl;
            // for (int ik = 0;ik < nks;++ik)
            //     for (int io = 0;io < pX->get_col_size();++io)    //nocc_local
            //         for (int iv = 0;iv < pX->get_row_size();++iv)    //nvirt_local
            //         {
            //             int io_g = pX->local2global_col(io);
            //             int iv_g = pX->local2global_row(iv);
            //             GlobalV::ofs_running << "io_global" << io_g << "iv_global +occ" << iv_g + nocc << " e_a-e_i=" << eig_ks_diff(ik, io * pX->get_row_size() + iv) << std::endl;
            //         }
        };
        void init(const int ik_in) override {};

        /// caution: put this operator at the head of the operator list,
        /// because vector_mul_vector_op directly assign to (rather than add on) psi_out.
        virtual void act(const psi::Psi<T>& psi_in, psi::Psi<T>& psi_out, const int nbands) const override
        {
            ModuleBase::TITLE("OperatorLRDiag", "act");
            assert(nbands <= psi_in.get_nbands());
            assert(psi_in.get_nbasis() == this->nsk * pX->get_local_size());
            for (int ib = 0;ib < nbands;++ib)
            {
                psi_in.fix_b(ib);
                psi_out.fix_b(ib);
                for (int is = 0;is < nsk / nks;++is)    // 1 or 2 for gamma_only, 1 for k
                    hsolver::vector_mul_vector_op<T, Device>()(this->ctx,
                        psi_in.get_nbasis(),
                        psi_out.get_pointer() + is * this->nks * pX->get_local_size(),
                        psi_in.get_pointer() + is * this->nks * pX->get_local_size(),
                        this->eig_ks_diff.c);
                //output psi_out
                // GlobalV::ofs_running << "psi_out:" << ib << "] in OperatorLRDiag : " << std::endl;
                // for (int ik = 0;ik < nsk;++ik)
                //     for (int io = 0;io < pX->get_col_size();++io)    //nocc_local
                //     {
                //         for (int iv = 0;iv < pX->get_row_size();++iv)    //nvirt_local
                //             GlobalV::ofs_running << psi_out.get_pointer()[ik * pX->get_local_size() + io * pX->get_row_size() + iv] << " ";
                //         GlobalV::ofs_running << std::endl;
                //     }
            }
        }
    private:
        const ModuleBase::matrix& eig_ks;
        const Parallel_2D* pX;
        ModuleBase::matrix eig_ks_diff;
        const int nks;
        const int nspin;
        const int nsk;
        const int nocc;
        const int nvirt;
        Device* ctx = {};
    };
}