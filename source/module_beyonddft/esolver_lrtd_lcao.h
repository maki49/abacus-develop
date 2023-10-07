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
#include "module_hamilt_lcao/hamilt_lcaodft/record_adj.h"
#include "module_hamilt_lcao/module_gint/gint_gamma.h"
#include "module_hamilt_lcao/module_gint/gint_k.h"
#include "module_hamilt_lcao/module_gint/grid_technique.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_beyonddft/potentials/pot_hxc_lrtd.hpp"
#include "module_beyonddft/hamilt_casida.hpp"
#ifdef __EXX
// #include <RI/physics/Exx.h>
#include "module_ri/Exx_LRI.h"
#endif
namespace ModuleESolver
{
    template <typename T> struct TGint;
    template <>
    struct TGint<double> {
        using type = Gint_Gamma;
    };
    template <>
    struct TGint<std::complex<double>> {
        using type = Gint_k;
    };
    template<typename T, typename TR = double, typename Device = psi::DEVICE_CPU>
    // template<typename T, typename Texx = T, typename Device = psi::DEVICE_CPU>
    class ESolver_LRTD : public ESolver_FP
    {
    public:
        /// @brief  a move constructor from ESolver_KS_LCAO
        ESolver_LRTD(ModuleESolver::ESolver_KS_LCAO<T, TR>&& ks_sol, Input& inp, UnitCell& ucell);
        ESolver_LRTD() {}
        ~ESolver_LRTD() {
            delete this->p_hamilt;
            delete this->phsol;
            delete this->pot;
            delete this->psi_ks;
            delete this->DM_trans;
            delete this->X;
            delete this->AX;
        }

        ///input: input, call, basis(LCAO), psi(ground state), elecstate
        // initialize sth. independent of the ground state
        virtual void Init(Input& inp, UnitCell& cell) override {};

        virtual void init_after_vc(Input& inp, UnitCell& cell) override {};
        virtual void Run(int istep, UnitCell& ucell) override;
        virtual void postprocess() override {};

        virtual double cal_Energy()  override { return 0.0; };
        virtual void cal_Force(ModuleBase::matrix& force) override {};
        virtual void cal_Stress(ModuleBase::matrix& stress) override {};

    protected:

        const UnitCell* p_ucell = nullptr;
        const Input* p_input = nullptr;

        hamilt::Hamilt<T, Device>* p_hamilt = nullptr;  //opsd problem first to use base calss
        hsolver::HSolver<T, Device>* phsol = nullptr;
        // not to use ElecState because 2-particle state is quite different from 1-particle state.
        // implement a independent one (ExcitedState) to pack physical properties if needed.
        // put the components of ElecState here: 
        elecstate::PotHxcLR* pot = nullptr;

        // ground state info 
        //pelec in  ESolver_FP
        const psi::Psi<T, Device>* psi_ks = nullptr;
        ModuleBase::matrix eig_ks;
        /// transition density matrix in AO representation
        elecstate::DensityMatrix<T, double>* DM_trans = nullptr;
        // energy of ground state is in pelec->ekb

        /// @brief Excited state info. size: nstates * nks * (nocc(local) * nvirt (local))
        psi::Psi<T, Device>* X;
        psi::Psi<T, Device>* AX;

        size_t nocc;
        size_t nvirt;
        size_t nbasis;
        /// n_occ*n_unocc, the basis size of electron-hole pair representation
        size_t npairs;
        /// how many 2-particle states to be solved
        size_t nstates = 1;
        size_t nks = 1; //gamma_only now
        size_t nspin = 1;

        // basis info (currently use GlobalC)
        // LCAO_Orbitals orb;
        // adj info 
        Record_adj ra;
        // grid parallel info (no need for 2d-block distribution)?
        // Grid_Technique gridt;
        // grid integration method(will be moved to OperatorKernelHxc)
        ModulePW::PW_Basis_Big bigpw;
        Grid_Technique gt;
        Gint_Gamma gint_g;
        Gint_k gint_k;
        typename TGint<T>::type* gint = nullptr;

        void set_gint();

        K_Vectors kv;

        /// @brief variables for parallel distribution of KS orbitals
        Parallel_2D paraC_;
        /// @brief variables for parallel distribution of excited states
        Parallel_2D paraX_;
        /// @brief variables for parallel distribution of matrix in AO representation
        Parallel_Orbitals paraMat_;

        /// move to hsolver::updatePsiK
        void init_X();

#ifdef __EXX
        // using TA = int;
        // using Tcell = int;
        // static constexpr std::size_t Ndim = 3;
        // RI::Exx<TA, Tcell, Ndim, Texx>* exx_lri;
        std::shared_ptr<Exx_LRI<double>> exx_lri_double = nullptr;
#endif

    };
    template<>void ESolver_LRTD<double>::set_gint() { this->gint = &this->gint_g;this->gint_g.gridt = &this->gt; }
    template<>void ESolver_LRTD<std::complex<double>>::set_gint() { this->gint = &this->gint_k; this->gint_g.gridt = &this->gt; }
}