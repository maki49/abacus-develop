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
            elecstate::DensityMatrix<T, double>* DM_trans_in,
            HContainer<double>*& hR_in,
#ifdef __EXX
            Exx_LRI<T>* exx_lri_in,
#endif 
            TGint* gint_in,
            elecstate::PotHxcLR* pot_in,
            const K_Vectors& kv_in,
            const std::vector<Parallel_2D*> p2d_in)
        {
            ModuleBase::TITLE("HamiltCasidaLR", "HamiltCasidaLR");
            this->classname = "HamiltCasidaLR";
            this->hR = new HContainer<double>(std::move(*hR_in));
            // add the diag operator  (the first one)
            this->ops = new OperatorLRDiag<T>(eig_ks, p2d_in.at(0), kv_in.nks, nspin, nocc, nvirt);
            //add Hxc operator
            OperatorLRHxc<T>* lr_hxc = new OperatorLRHxc<T>(nspin, naos, nocc, nvirt, psi_ks_in, DM_trans_in, this->hR, gint_in, pot_in, kv_in.kvec_d, p2d_in);
            this->ops->add(lr_hxc);
#ifdef __EXX
            if (xc_kernel == "hf")
            {
                //add Exx operator
                Operator<T>* lr_exx = new OperatorLREXX<T>(nspin, naos, nocc, nvirt, ucell_in, psi_ks_in, DM_trans_in, exx_lri_in, kv_in, p2d_in);
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
        };

        HContainer<double>* getHR() { return this->hR; }
    private:
        HContainer<double>* hR = nullptr;
    };
}