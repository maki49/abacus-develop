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
    template<typename T, typename TR = double>
    // template<typename T, typename Texx = T, typename Device = psi::DEVICE_CPU>
    class ESolver_LRTD : public ESolver_FP
    {
    public:
        /// @brief  a move constructor from ESolver_KS_LCAO
        ESolver_LRTD(ModuleESolver::ESolver_KS_LCAO<T, TR>&& ks_sol, Input& inp, UnitCell& ucell);
        /// @brief a from-scratch constructor
        ESolver_LRTD(Input& inp, UnitCell& ucell);
        ~ESolver_LRTD() {
            delete this->p_hamilt;
            delete this->phsol;
            delete this->pot;
            delete this->psi_ks;
            delete this->X;
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

        const UnitCell& ucell;

        hamilt::Hamilt<T>* p_hamilt = nullptr;  //opsd problem first to use base calss
        hsolver::HSolver<T>* phsol = nullptr;
        // not to use ElecState because 2-particle state is quite different from 1-particle state.
        // implement a independent one (ExcitedState) to pack physical properties if needed.
        // put the components of ElecState here: 
        elecstate::PotHxcLR* pot = nullptr;

        // ground state info 
        //pelec in  ESolver_FP
        // const psi::Psi<T>* psi_ks = nullptr;
        psi::Psi<T>* psi_ks = nullptr;
        ModuleBase::matrix eig_ks;///< energy of ground state

        /// @brief Excited state info. size: nstates * nks * (nocc(local) * nvirt (local))
        psi::Psi<T>* X;

        int nocc;
        int nvirt;
        int nbasis;
        /// n_occ*n_unocc, the basis size of electron-hole pair representation
        int npairs;
        /// how many 2-particle states to be solved
        int nstates = 1;
        int nsk = 1; //nspin*nks
        int nspin = 1;

        Grid_Technique gt;
        Gint_Gamma gint_g;
        Gint_k gint_k;
        typename TGint<T>::type* gint = nullptr;

        std::string xc_kernel;
        std::string lr_solver;

        void set_gint();

        K_Vectors kv;

        /// @brief variables for parallel distribution of KS orbitals
        Parallel_2D paraC_;
        /// @brief variables for parallel distribution of excited states
        Parallel_2D paraX_;
        /// @brief variables for parallel distribution of matrix in AO representation
        Parallel_Orbitals paraMat_;

        /// @brief allocate and initialize X
        void init_X(const int& nvirt_input);
        /// @brief allocate and initialize A matrix, density matrix and eignensolver
        void init_A(hamilt::HContainer<double>* pHR_in, const double lr_thr);
        /// @brief read in the ground state wave function, band energy and occupation
        void read_ks_wfc();
        /// @brief  read in the ground state charge density
        void read_ks_chg(Charge& chg);

        void init_pot(const Charge& chg_gs);

#ifdef __EXX
        /// Tdata of Exx_LRI is same as T, for the reason, see operator_lr_exx.h
        std::shared_ptr<Exx_LRI<T>> exx_lri = nullptr;
        void move_exx_lri(std::shared_ptr<Exx_LRI<double>>&);
        void move_exx_lri(std::shared_ptr<Exx_LRI<std::complex<double>>>&);

        std::unique_ptr<TwoCenterBundle> two_center_bundle;
#endif

    };
    template<>void ESolver_LRTD<double>::set_gint() { this->gint = &this->gint_g;this->gint_g.gridt = &this->gt; }
    template<>void ESolver_LRTD<std::complex<double>>::set_gint() { this->gint = &this->gint_k; this->gint_k.gridt = &this->gt; }
}