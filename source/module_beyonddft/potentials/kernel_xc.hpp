#pragma once
#include "module_elecstate/potentials/pot_base.h"
namespace elecstate
{

class KernelXC : public PotBase
{
  public:
    // constructor for exchange-correlation kernel
    KernelXC(const ModulePW::PW_Basis* rho_basis_in,
          double* etxc_in,
          double* vtxc_in,
          ModuleBase::matrix* vofk_in = nullptr)
        : etxc_(etxc_in), vtxc_(vtxc_in), vofk(vofk_in)
    {
        this->rho_basis_ = rho_basis_in;
        this->dynamic_mode = false;
        this->fixed_mode = true;
    }

    void cal_v_eff(const Charge* chg, const UnitCell* ucell, ModuleBase::matrix& v_eff) override;

    ModuleBase::matrix* vofk = nullptr;
    double* etxc_ = nullptr;
    double* vtxc_ = nullptr;
};

} // namespace elecstate