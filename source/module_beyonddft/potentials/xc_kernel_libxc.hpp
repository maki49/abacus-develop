#pragma once
#ifdef USE_LIBXC
#include <xc.h>

class XC_Kernel_Libxc
{
public:
    XC_Kernel_Libxc();
    ~XC_Kernel_Libxc();
    
    static std::tuple<double,double,ModuleBase::matrix> f_xc_libxc(
		const int &nrxx, // number of real-space grid
		const double &omega, // volume of cell
		const double tpiba,
		const Charge* const chr); // charge density
}

static std::tuple<double,double,ModuleBase::matrix> XC_Kernel_Libxc::f_xc_libxc(
    const int &nrxx,
    const double &omega,
    const double tpiba,
    const Charge* const chr)
{
    ModuleBase::TITLE("XC_Kernel_Libxc","f_xc_libxc");
    ModuleBase::timer::tick("XC_Kernel_Libxc","f_xc_libxc");
}
#endif