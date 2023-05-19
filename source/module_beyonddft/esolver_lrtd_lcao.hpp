#include "esolver_lrtd_lcao.h"
#include "move_gint.hpp"
#include <memory>

template<typename FPTYPE, typename Device>
ModuleESolver::ESolver_LRTD<FPTYPE, Device>::ESolver_LRTD(ModuleESolver::ESolver_KS_LCAO&& ks_sol)
{
    // move the ground state info 
    if (typeid(FPTYPE) == typeid(double))
    {
        this->psi_ks = reinterpret_cast<psi::Psi<FPTYPE>*>(ks_sol.psid);
        ks_sol.psid = nullptr;
    }
    else if (typeid(FPTYPE) == typeid(std::complex<double>))
    {
        this->psi_ks = reinterpret_cast<psi::Psi<FPTYPE>*>(ks_sol.psi);
        ks_sol.psi = nullptr;
    }
    else
        throw std::runtime_error("ESolver_LRTD: FPTYPE not supported");

    //only need the eigenvalues. the 'elecstates' of excited states is different from ground state.
    this->eig_ks = std::move(ks_sol.pelec->ekb);

    // move the basis info (2-center integrals currently not needed)
    // std::cout<<"before move orb"<<std::endl;
    // this->orb = std::forward<LCAO_Orbitals>(GlobalC::ORB);
    // std::cout << "after move orb" << std::endl;

    // move something that will be the same for all excited states
    // except RA, which has been deleted after cal_Force 
    
    //grid integration
    if (typeid(FPTYPE) == typeid(double))
        this->gint_g = std::move(ks_sol.UHM.GG);
    else
        this->gint_k = std::move(ks_sol.UHM.GK);
    // can GlobalC be moved? try

    // grid parallel info
    this->gridt = std::move(ks_sol.GridT);

}

template<typename FPTYPE, typename Device>
void ModuleESolver::ESolver_LRTD<FPTYPE, Device>::Init(Input& inp, UnitCell& ucell)
{
    ModuleBase::TITLE("ESolver_KS_LCAO", "Init");

    this->p_input = &inp;
    this->p_ucell = &ucell;

    // calculate the number of occupied and unoccupied states
    // which determines the basis size of the excited states
    this->n_occ = LR_Util::cal_nocc(LR_Util::cal_nelec(ucell));
    this->n_basis = GlobalV::NLOCAL;    //use GlobalV temporarily about basis
    this->n_unocc = n_basis - n_occ;
    this->n_pairs = this->n_occ * this->n_unocc;

    this->n_states = inp.nstates;

    //init Hamiltonian

    return;
}

template<typename FPTYPE, typename Device>
void ModuleESolver::ESolver_LRTD<FPTYPE, Device>::Run(int istep, UnitCell& cell)
{
    std::cout << "running ESolver_LRTD" << std::endl;
    return;
}

template<typename FPTYPE, typename Device>
void ModuleESolver::ESolver_LRTD<FPTYPE, Device>::init_X()
{
    //the eigenstate in the electron-hole pair representation
    //Psi.nbasis = npairs, Psi.nbands = nstates.
    //need a parallel distribution in the future 
    this->X = std::make_shared<psi::Psi<FPTYPE, Device>>(this->nks, this->nstates, this->npairs);
    X->zero_out();
    // use unit vectors as the initial guess
    for (int i = 0; i < this->nstates; i++)
    {
        (*X)(i, i) = static_cast<FPTYPE>(1.0);
    }
}