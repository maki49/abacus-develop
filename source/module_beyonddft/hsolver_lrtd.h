#pragma once
#include "module_hsolver/hsolver.h"
#include "module_hsolver/diago_iter_assist.h"
#include "module_psi/psi.h"
namespace hsolver
{
    template<typename T, typename Device = psi::DEVICE_CPU>
    class HSolverLR : public HSolver<T, Device>
    {
    private:
        using Real = typename GetTypeReal<T>::type;
        const int npairs = 0;
    public:
        HSolverLR(const int npairs_in) :npairs(npairs_in) {};
        virtual Real set_diagethr(const int istep, const int iter, const Real ethr) override
        {
            this->diag_ethr = ethr;
            return ethr;
        }
        virtual void solve(hamilt::Hamilt<T, Device>* pHamilt,
            psi::Psi<T, Device>& psi,
            elecstate::ElecState* pes,
            const std::string method_in,
            const bool skip_charge) override;
    };
};