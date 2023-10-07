#pragma once
#include "module_hamilt_general/hamilt.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_beyonddft/operator_casida/operatorA_hxc.h"
#include "module_basis/module_ao/parallel_orbitals.h"
namespace hamilt
{
    template<typename T, typename Device = psi::DEVICE_CPU>
    class HamiltCasidaLR : public Hamilt<T, Device>
    {
    public:
        template<typename TGint>
        HamiltCasidaLR(const int& nsk,
            const int& naos,
            const int& nocc,
            const int& nvirt,
            const psi::Psi<T, Device>* psi_ks_in,
            elecstate::DensityMatrix<T, double>* DM_trans_in,
            HContainer<double>*& hR_in,
            TGint* gint_in,
            elecstate::PotHxcLR* pot_in,
            const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
            const std::vector<Parallel_2D*> p2d_in)
        {
            ModuleBase::TITLE("HamiltCasidaLR", "HamiltCasidaLR");
            this->classname = "HamiltCasidaLR";
            this->hR = new HContainer<double>(std::move(*hR_in));
            //add Hxc operator (the first one)
            this->ops = new OperatorA_Hxc<T, Device>(nsk, naos, nocc, nvirt, psi_ks_in, DM_trans_in, this->hR, gint_in, pot_in, kvec_d_in, p2d_in);
            //add Exx operator (remaining)
            // Operator<double>* A_Exx = new OperatorA_Exx<T, TGint>;
            // this->opsd->add(A_Exx);
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