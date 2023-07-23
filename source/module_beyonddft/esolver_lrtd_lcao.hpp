#pragma once
#include "esolver_lrtd_lcao.h"
#include "move_gint.hpp"
#include "utils/lr_util_algorithms.hpp"
#include "utils/lr_util_physics.hpp"
#include "hamilt_casida.hpp"
#include "module_beyonddft/potentials/pot_hxc_lrtd.hpp"
#include "module_beyonddft/hsolver_lrtd.h"
#include <memory>

template<typename T, typename TR, typename Device>
ModuleESolver::ESolver_LRTD<T, TR, Device>::ESolver_LRTD(ModuleESolver::ESolver_KS_LCAO<T, TR>&& ks_sol,
    Input& inp, UnitCell& ucell) : p_input(&inp), p_ucell(&ucell)
{
    // move the ground state info 
    this->psi_ks = ks_sol.psi;
    ks_sol.psi = nullptr;

    //only need the eigenvalues. the 'elecstates' of excited states is different from ground state.
    this->eig_ks = std::move(ks_sol.pelec->ekb);

    //kv
    this->kv = std::move(ks_sol.kv);
    std::cout << "kv.kvec_d[0].x" << kv.kvec_d[0].x << std::endl;

    // move the basis info (2-center integrals currently not needed)
    // std::cout<<"before move orb"<<std::endl;
    // this->orb = std::forward<LCAO_Orbitals>(GlobalC::ORB);
    // std::cout << "after move orb" << std::endl;

    // move something that will be the same for all excited states
    // except RA, which has been deleted after cal_Force 
    
    //grid integration
    if (typeid(T) == typeid(double))
        this->gint_g = std::move(ks_sol.UHM.GG);
    else
        this->gint_k = std::move(ks_sol.UHM.GK);

    // xc kernel
    XC_Functional::set_xc_type(inp.xc_kernel);
    //init potential and calculate kernels using ground state charge
    if (this->pot == nullptr)
    {
        this->pot = new elecstate::PotHxcLR(ks_sol.pw_rho,
            p_ucell,
            ks_sol.pelec->charge);
    };

    // calculate the number of occupied and unoccupied states
    // which determines the basis size of the excited states
    this->nocc = LR_Util::cal_nocc(LR_Util::cal_nelec(ucell));
    this->nbasis = GlobalV::NLOCAL;    //use GlobalV temporarily about basis
    this->nvirt = this->eig_ks.nc - nocc;   //nbands-nocc
    this->npairs = this->nocc * this->nvirt;
    this->nstates = inp.nstates;
    GlobalV::ofs_running << "Setting LR-TDDFT parameters: " << std::endl;
    GlobalV::ofs_running << "number of occupied bands: " << this->nocc << std::endl;
    GlobalV::ofs_running << "number of virtual bands: " << this->nvirt << std::endl;
    GlobalV::ofs_running << "number of Atom orbitals (LCAO-basis size): " << this->nbasis << std::endl;
    GlobalV::ofs_running << "number of KS bands: " << this->eig_ks.nc << std::endl;
    GlobalV::ofs_running << "number of electron-hole pairs (2-particle basis size): " << this->npairs << std::endl;
    GlobalV::ofs_running << "number of excited states to be solved: " << this->nstates << std::endl;

    //allocate 2-particle state and setup 2d division
    this->init_X();

    //init Hamiltonian
    if (typeid(T) == typeid(double))
        this->p_hamilt = new hamilt::HamiltCasidaLR<T, Device>(this->nks, this->nbasis, this->nocc, this->nvirt, this->psi_ks,
            &this->gint_g, this->pot, std::vector<Parallel_2D*>({ &this->paraX_, &this->paraC_, &this->paraMat_ }));
    else if (typeid(T) == typeid(std::complex<double>))
        this->p_hamilt = new hamilt::HamiltCasidaLR<T, Device>(this->nks, this->nbasis, this->nocc, this->nvirt, this->psi_ks,
            &this->gint_k, this->pot, std::vector<Parallel_2D*>({ &this->paraX_, &this->paraC_, &this->paraMat_ }));
#ifdef __EXX
    if (inp.xc_kernel == "hf")
    {
        //complex-double problem....waiting for refactor
        // this->exx_lri = std::make_shared<Exx_LRI<double>>(ks_sol.exx_lri_double);
        if (ks_sol.exx_lri_double)  // move from ks solver
        {
            this->exx_lri_double = ks_sol.exx_lri_double;
            ks_sol.exx_lri_double = nullptr;
        }
        else    // construct C, V from scratch
        {
            this->exx_lri_double = std::make_shared<Exx_LRI<double>>(GlobalC::exx_info.info_ri);
            this->exx_lri_double->init(MPI_COMM_WORLD, kv); // using GlobalC::ORB
            // this->exx_lri_double->cal_exx_ions();
        }
    }
#endif

    /// =========just for test==============
    // try act
    // for (int istate = 0;istate < nstates;++istate)
    // {
    //     this->X->fix_b(istate);
    //     this->AX->fix_b(istate);
    //     this->p_hamilt->ops->act(*this->X, *this->AX);
    // }
    /// =====================================

    // init HSolver
    this->phsol = new hsolver::HSolverLR<T, Device>();

}

template<typename T, typename TR, typename Device>
void ModuleESolver::ESolver_LRTD<T, TR, Device>::Run(int istep, UnitCell& cell)
{
    ModuleBase::TITLE("ESolver_LRTD", "Run");
    std::cout << "running ESolver_LRTD" << std::endl;
    this->phsol->solve(dynamic_cast<hamilt::Hamilt<T, Device>*>(this->p_hamilt), *this->X, this->pelec, "dav");
    return;
}

template<typename T, typename TR, typename Device>
void ModuleESolver::ESolver_LRTD<T, TR, Device>::init_X()
{
    ModuleBase::TITLE("ESolver_LRTD", "Init");
    //the eigenstate in the electron-hole pair representation
    //Psi.nbasis = npairs, Psi.nbands = nstates.

    // setup ParaX

    int nsk = (nspin == 4) ? this->nks : this->nks * this->nspin;
    LR_Util::setup_2d_division(this->paraX_, 1, this->nvirt, this->nocc);//nvirt - row, nocc - col 
    this->X = new psi::Psi<T, Device>(nsk, this->nstates, this->paraX_.get_local_size(), nullptr, false);  // band(state)-first
    this->X->zero_out();
    LR_Util::setup_2d_division(this->paraMat_, 1, this->nbasis, this->nbasis, this->paraX_.comm_2D, this->paraX_.blacs_ctxt);
    LR_Util::setup_2d_division(this->paraC_, 1, this->nbasis, this->nocc + this->nvirt, this->paraX_.comm_2D, this->paraX_.blacs_ctxt);
    // this->AX = new psi::Psi<T, TR, Device>(*this->X);
    this->AX = new psi::Psi<T, Device>(nsk, this->nstates, this->paraX_.get_local_size(), nullptr, false);
    // set the initial guess of X
    // if (E_{lumo}-E_{homo-1} < E_{lumo+1}-E{homo}), mode = 0, else 1(smaller first)
    bool ix_mode = 0;   //default
    if (this->eig_ks.nc > nocc + 1 &&
        eig_ks(0, nocc) - eig_ks(0, nocc - 2) > eig_ks(0, nocc + 1) - eig_ks(0, nocc - 1))
        ix_mode = 1;
    GlobalV::ofs_running << "setting the initial guess of X: " << std::endl;
    GlobalV::ofs_running << "E_{lumo}-E_{homo-1}=" << eig_ks(0, nocc) - eig_ks(0, nocc - 2) << std::endl;
    GlobalV::ofs_running << "E_{lumo+1}-E{homo}=" << eig_ks(0, nocc + 1) - eig_ks(0, nocc - 1) << std::endl;
    GlobalV::ofs_running << "mode of X-index: " << ix_mode << std::endl;

    /// global index map between (i,c) and ix
    ModuleBase::matrix iciv2ix;
    std::vector<std::pair<int, int>> ix2iciv;
    std::pair<ModuleBase::matrix, std::vector<std::pair<int, int>>> indexmap =
        LR_Util::set_ix_map_diagonal(ix_mode, nocc, nvirt);
    iciv2ix = std::move(std::get<0>(indexmap));
    ix2iciv = std::move(std::get<1>(indexmap));
    
    // use unit vectors as the initial guess
    for (int i = 0; i < this->nstates; i++)
    {
        this->X->fix_b(i);
        int row_global = std::get<0>(ix2iciv[i]);
        int col_global = std::get<1>(ix2iciv[i]);
        if (this->paraX_.in_this_processor(row_global, col_global))
            for (int isk = 0;isk < nsk;++isk)
                (*X)(isk, this->paraX_.global2local_row(row_global) * this->paraX_.get_col_size() + this->paraX_.global2local_col(col_global)) = static_cast<T>(1.0);
    }
}
