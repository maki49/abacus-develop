#pragma once
#include "module_hamilt_general/hamilt.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_beyonddft/operator_casida/operator_lr_diag.h"
#include "module_beyonddft/operator_casida/operator_lr_hxc.h"
#include "module_beyonddft/operator_casida/operator_lr_exx.h"
#include "module_basis/module_ao/parallel_orbitals.h"
namespace hamilt
{
    template<typename T>
    class HamiltCasidaLR : public Hamilt<T, psi::DEVICE_CPU>
    {
    public:
        template<typename TGint>
        HamiltCasidaLR(std::string& xc_kernel,
            const int& nspin,
            const int& naos,
            const int& nocc,
            const int& nvirt,
            const UnitCell& ucell_in,
            const psi::Psi<T>* psi_ks_in,
            const ModuleBase::matrix& eig_ks,
            // elecstate::DensityMatrix<T, double>* DM_trans_in,
            HContainer<double>*& hR_in,
#ifdef __EXX
            Exx_LRI<T>* exx_lri_in,
#endif 
            TGint* gint_in,
            elecstate::PotHxcLR* pot_in,
            const K_Vectors& kv_in,
            Parallel_2D* pX_in,
            Parallel_2D* pc_in,
            Parallel_Orbitals* pmat_in) : nocc(nocc), nvirt(nvirt), pX(pX_in), nsk(eig_ks.nr)
        {
            ModuleBase::TITLE("HamiltCasidaLR", "HamiltCasidaLR");
            this->classname = "HamiltCasidaLR";
            assert(hR_in != nullptr);
            this->hR = new HContainer<double>(std::move(*hR_in));
            this->DM_trans.resize(1);
            this->DM_trans[0] = new elecstate::DensityMatrix<T, double>(&kv_in, pmat_in, nspin);
            this->DM_trans[0]->init_DMR(*this->hR);
            // add the diag operator  (the first one)
            this->ops = new OperatorLRDiag<T>(eig_ks, pX_in, kv_in.nks, nspin, nocc, nvirt);
            //add Hxc operator
            OperatorLRHxc<T>* lr_hxc = new OperatorLRHxc<T>(nspin, naos, nocc, nvirt, psi_ks_in,
                this->DM_trans, this->hR, gint_in, pot_in, kv_in, pX_in, pc_in, pmat_in);
            this->ops->add(lr_hxc);
#ifdef __EXX
            if (xc_kernel == "hf")
            {   //add Exx operator
                Operator<T>* lr_exx = new OperatorLREXX<T>(nspin, naos, nocc, nvirt, ucell_in, psi_ks_in,
                    this->DM_trans, exx_lri_in, kv_in, pX_in, pc_in, pmat_in);
                this->ops->add(lr_exx);
            }
#endif
        }
        ~HamiltCasidaLR()
        {
            if (this->ops != nullptr)
            {
                delete this->ops;
            }
            delete this->hR;
            for (auto& d : this->DM_trans)delete d;
        };

        HContainer<double>* getHR() { return this->hR; }

        virtual std::vector<T> matrix() override
        {
            ModuleBase::TITLE("HamiltCasidaLR", "matrix");
            int npairs = this->nocc * this->nvirt;
            std::vector<T> Amat_full(this->nsk * npairs * this->nsk * npairs, 0.0);
            for (int isk = 0;isk < this->nsk;++isk)
                for (int j = 0;j < nocc;++j)
                    for (int b = 0;b < nvirt;++b)
                    {//calculate A^{ai} for each bj
                        int bj = j * nvirt + b;
                        int kbj = isk * npairs + bj;
                        psi::Psi<T> X_bj(1, 1, this->nsk * this->pX->get_local_size()); // k1-first, like in iterative solver
                        X_bj.zero_out();
                        // X_bj(0, 0, lj * this->pX->get_row_size() + lb) = this->one();
                        int lj = this->pX->global2local_col(j);
                        int lb = this->pX->global2local_row(b);
                        if (this->pX->in_this_processor(b, j)) X_bj(0, 0, isk * this->pX->get_local_size() + lj * this->pX->get_row_size() + lb) = this->one();
                        psi::Psi<T> A_aibj(1, 1, this->nsk * this->pX->get_local_size()); // k1-first
                        A_aibj.zero_out();

                        Operator<T>* node(this->ops);
                        while (node != nullptr)
                        {   // act() on and return the k1-first type of psi
                            node->act(X_bj, A_aibj, 1);
                            node = (Operator<T>*)(node->next_op);
                        }
                        // reduce ai for a fixed bj
                        A_aibj.fix_kb(0, 0);
                        for (int isk_ai = 0;isk_ai < this->nsk;++isk_ai)
                            LR_Util::gather_2d_to_full(*this->pX, &A_aibj.get_pointer()[isk_ai * this->pX->get_local_size()],
                                Amat_full.data() + kbj * this->nsk * npairs /*col, bj*/ + isk_ai * npairs/*row, ai*/,
                                false, this->nvirt, this->nocc);
                    }
            // output Amat
            std::cout << "Amat_full:" << std::endl;
            for (int i = 0;i < this->nsk * npairs;++i)
            {
                for (int j = 0;j < this->nsk * npairs;++j)
                {
                    std::cout << Amat_full[i * this->nsk * npairs + j] << " ";
                }
                std::cout << std::endl;
            }
            return Amat_full;
        }
    private:
        int nocc;
        int nvirt;
        int nsk;
        Parallel_2D* pX = nullptr;
        T one();
        HContainer<double>* hR = nullptr;
        /// transition density matrix in AO representation
        /// Hxc only: size=1, calculate on the same address for each bands
        /// Hxc+Exx: size=nbands, store the result of each bands for common use
        std::vector<elecstate::DensityMatrix<T, double>*> DM_trans;
    };

    template<> double HamiltCasidaLR<double>::one() { return 1.0; }
    template<> std::complex<double> HamiltCasidaLR<std::complex<double>>::one() { return std::complex<double>(1.0, 0.0); }

}