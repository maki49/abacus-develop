#pragma once
#include "module_hamilt_general/hamilt.h"
#include "module_beyonddft/operator_casida/operatorA_hxc.h"
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
            TGint* gint_in,
            elecstate::PotHxcLR* pot_in,
            const std::vector<Parallel_2D*> p2d_in)
        {
            ModuleBase::TITLE("HamiltCasidaLR", "HamiltCasidaLR");
            this->classname = "HamiltCasidaLR";

            // ops and opsd in base class may be unified in the future?
            //add Hxc operator (the first one)
            this->ops = new OperatorA_Hxc<T, Device>(nsk, naos, nocc, nvirt, psi_ks_in, gint_in, pot_in, p2d_in);
            //add Exx operator (remaining)
            // Operator<double>* A_Exx = new OperatorA_Exx<T, TGint>;
            // this->opsd->add(A_Exx);
        }
        // private:
        // do not separate Operator  into ops and opsd as base class
        // ops in base class will be shadowed
        OperatorA_Hxc<T, Device>* ops = nullptr;
    };
}