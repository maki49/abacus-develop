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

    // 2d-distribution
    Parallel_2D* tmp_paraC = static_cast<Parallel_2D*>(&ks_sol.orb_con.ParaV);
    this->paraC_ = std::move(*tmp_paraC);
    tmp_paraC = nullptr;

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
    ModuleBase::TITLE("ESolver_LRTD", "Init");

    this->p_input = &inp;
    this->p_ucell = &ucell;

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

    //init X
    this->init_X();
    //init Hamiltonian

    return;
}

template<typename FPTYPE, typename Device>
void ModuleESolver::ESolver_LRTD<FPTYPE, Device>::Run(int istep, UnitCell& cell)
{
    ModuleBase::TITLE("ESolver_LRTD", "Run");
    std::cout << "running ESolver_LRTD" << std::endl;
    return;
}

template<typename FPTYPE, typename Device>
void ModuleESolver::ESolver_LRTD<FPTYPE, Device>::init_X()
{
    ModuleBase::TITLE("ESolver_LRTD", "Init");
    //the eigenstate in the electron-hole pair representation
    //Psi.nbasis = npairs, Psi.nbands = nstates.

    // setup ParaX
    this->setup_2d_division(1, this->nocc, this->nvirt);
    for (int i = 0; i < this->nstates; i++)
    {
        this->X.emplace_back(this->nks, this->paraX_.get_row_size(), this->paraX_.get_col_size());
        X[i].zero_out();
    }
    
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
    std::pair<ModuleBase::matrix, std::vector<std::pair<int, int>>> indexmap =
        LR_Util::set_ix_map_diagonal(ix_mode, nocc, nvirt);
    this->iciv2ix = std::move(std::get<0>(indexmap));
    this->ix2iciv = std::move(std::get<1>(indexmap));
    
    // use unit vectors as the initial guess
    for (int i = 0; i < this->nstates; i++)
    {
        int row_global = std::get<0>(ix2iciv[i]);
        int col_global = std::get<1>(ix2iciv[i]);
        if (this->paraX_.in_this_processor(row_global, col_global))
            X[i](this->paraX_.trace_loc_row[row_global], this->paraX_.trace_loc_col[col_global]) = static_cast<FPTYPE>(1.0);
    }
}

template<typename FPTYPE, typename Device>
void ModuleESolver::ESolver_LRTD<FPTYPE, Device>::setup_2d_division(int nb, int gr, int gc)
{
    ModuleBase::TITLE("ESolver_LRTD", "setup_2d_division");
    this->paraX_.set_block_size(nb);
#ifdef __MPI
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    this->paraX_.set_proc_dim(nprocs);
    this->paraX_.mpi_create_cart(MPI_COMM_WORLD);
    this->paraX_.set_local2global(gr, gc, GlobalV::ofs_running, GlobalV::ofs_warning);
    this->paraX_.set_desc(gr, gc, this->paraX_.get_row_size());
    this->paraX_.set_global2local(gr, gc, true, GlobalV::ofs_running);
#else
    this->paraX_.set_proc_dim(1);
    this->paraX_.set_serial(gr, gc);
    this->paraX_.set_global2local(gr, gc, false, GlobalV::ofs_running);
#endif
}