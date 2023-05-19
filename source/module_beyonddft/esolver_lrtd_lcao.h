#pragma once
#include "module_esolver/esolver_fp.h"
#include "module_io/input.h"
#include "module_cell/unitcell.h"
#include "module_hamilt_general/hamilt.h"
#include "module_hsolver/hsolver.h"
#include "module_elecstate/elecstate_lcao.h"

#include <vector>   //future tensor
#include <memory>

#include "module_esolver/esolver_ks_lcao.h" //for the move constructor

#include "lr_util.hpp"

// tmp 
#include "module_hamilt_lcao/hamilt_lcaodft/record_adj.h"
#include "module_hamilt_lcao/module_gint/gint_gamma.h"
#include "module_hamilt_lcao/module_gint/gint_k.h"
#include "module_hamilt_lcao/module_gint/grid_technique.h"

namespace ModuleESolver
{
    template<typename FPTYPE, typename Device = psi::DEVICE_CPU>
    class ESolver_LRTD : public ESolver_FP
    {
    public:
        /// @brief a constructor from ground state info
        ESolver_LRTD(psi::Psi<FPTYPE, Device>&& psi_in, ModuleBase::matrix&& eig_in)
            : psi_ks(&psi_in) {
            if (this->pelec == nullptr) this->pelec = new elecstate::ElecStateLCAO;
            this->pelec->ekb = eig_in;
        }
        /// @brief  a move constructor from ESolver_KS_LCAO
        ESolver_LRTD(ModuleESolver::ESolver_KS_LCAO&& ks_sol);
        ESolver_LRTD() {}
        ~ESolver_LRTD() {
            if (this->pelec != nullptr) delete this->pelec;
            // if (this->X != nullptr) delete this->X;
        }

        ///input: input, call, basis(LCAO), psi(ground state), elecstate
        virtual void Init(Input& inp, UnitCell& cell) override;

        virtual void init_after_vc(Input& inp, UnitCell& cell) override {};
        virtual void Run(int istep, UnitCell& ucell) override;
        virtual void postprocess() override {};

        virtual double cal_Energy()  override {};
        virtual void cal_Force(ModuleBase::matrix& force) override {};
        virtual void cal_Stress(ModuleBase::matrix& stress) override {};

    protected:

        UnitCell* p_ucell = nullptr;
        Input* p_input = nullptr;

        std::unique_ptr<hamilt::Hamilt<FPTYPE, Device>> phamilt = nullptr;
        std::unique_ptr<hsolver::HSolver<FPTYPE, Device>> phsol = nullptr;
        // in Operator: 
        //mutable psi::Psi<FPTYPE, Device>* hpsi = nullptr;

        // ground state info 
        //pelec in  ESolver_FP
        const psi::Psi<FPTYPE, Device>* psi_ks = nullptr;
        ModuleBase::matrix eig_ks;
        // energy of ground state is in pelec->ekb

        // excited state info
        std::shared_ptr<psi::Psi<FPTYPE, Device>> X = nullptr;
        //psi::Psi<FPTYPE, Device>* AX = nullptr;

        size_t n_occ;
        size_t n_unocc;
        size_t n_basis;
        /// n_occ*n_unocc, the basis size of electron-hole pair representation
        size_t n_pairs;
        /// how many 2-particle states to be solved
        size_t n_states = 1;
        size_t nks = 1; //gamma_only now

        // basis info (currently use GlobalC)
        // LCAO_Orbitals orb;
        // adj info 
        Record_adj ra;
        // grid parallel info (no need for 2d-block distribution)?
        Grid_Technique gridt;
        // grid integration method(will be moved to OperatorKernelHxc)
        ModulePW::PW_Basis_Big bigpw;
        Gint_Gamma gint_g;
        Gint_k gint_k;

        void init_X();

    };
}