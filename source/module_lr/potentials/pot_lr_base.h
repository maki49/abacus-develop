#pragma once
#include "module_cell/unitcell.h"
#include "module_basis/module_pw/pw_basis.h"

namespace LR
{
    class PotLRBase
    {
    public:
        PotLRBase(const ModulePW::PW_Basis& rho_basis, const int& nspin, const int& nrxx, const double& tpiba) : rho_basis_(rho_basis), nspin_(nspin), nrxx_(nrxx), tpiba_(tpiba) {}
        virtual void cal_v_eff(double** rho, const UnitCell& ucell, ModuleBase::matrix& v_eff, const std::vector<int>& ispin_op = { 0,0 }) = 0;
        // const references
        const ModulePW::PW_Basis& rho_basis = rho_basis_;
        const int& nrxx = nrxx_;
        const int& nspin = nspin_;
    protected:
        const ModulePW::PW_Basis& rho_basis_;
        const int nspin_ = 1;
        const int nrxx_ = 1;
        const double& tpiba_;
    };
}