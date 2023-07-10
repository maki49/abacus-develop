#include "module_hamilt_general/hamilt.h"
#include "module_beyonddft/operator_casida/kernel_hxc.hpp"
namespace hamilt
{
template<typename FPTYPE, typename Device = psi::DEVICE_CPU>
class HamiltLRCasida : public Hamilt
{
    HamiltLRCasida()
    {
        this->classname = "HamiltLRCasida";
        if (typeid(FPTYPE) == typeid(double))
            this->opsd = new OperatorKernelHxc<FPTYPE, Device>;
        else
            this->ops = new OperatorKernelHxc<FPTYPE, Device>;

    }
    void hPsi(const std::complex<FPTYPE>* psi_in, std::complex<FPTYPE>* hpsi, const size_t size) const
    {
        return;
    }

    
};
}