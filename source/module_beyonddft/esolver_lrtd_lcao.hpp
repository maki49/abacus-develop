#pragma once
#include "esolver_lrtd_lcao.h"
#include "move_gint.hpp"
#include "utils/lr_util_algorithms.hpp"
#include "utils/lr_util_physics.hpp"
#include "hamilt_casida.hpp"
#include "module_beyonddft/potentials/pot_hxc_lrtd.hpp"
#include "module_beyonddft/hsolver_lrtd.h"
#include "module_beyonddft/lr_spectrum.hpp"
#include <memory>
#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h"
#include "module_io/read_wfc_nao.h"
#include "module_io/rho_io.h"

#ifdef __EXX
template<>
void ModuleESolver::ESolver_LRTD<double>::move_exx_lri(std::shared_ptr<Exx_LRI<double>>& exx_ks)
{
    this->exx_lri = exx_ks;
    exx_ks = nullptr;
}
template<>
void ModuleESolver::ESolver_LRTD<std::complex<double>>::move_exx_lri(std::shared_ptr<Exx_LRI<std::complex<double>>>& exx_ks)
{
    this->exx_lri = exx_ks;
    exx_ks = nullptr;
}
template<>
void ModuleESolver::ESolver_LRTD<std::complex<double>>::move_exx_lri(std::shared_ptr<Exx_LRI<double>>& exx_ks)
{
    throw std::runtime_error("ESolver_LRTD<std::complex<double>>::move_exx_lri: cannot move double to complex<double>");
}
template<>
void ModuleESolver::ESolver_LRTD<double>::move_exx_lri(std::shared_ptr<Exx_LRI<std::complex<double>>>& exx_ks)
{
    throw std::runtime_error("ESolver_LRTD<double>::move_exx_lri: cannot move complex<double> to double");
}
#endif

inline double getreal(std::complex<double> x) { return x.real(); }
inline double getreal(double x) { return x; }

inline void redirect_log(const bool& out_alllog)
{
    GlobalV::ofs_running.close();
    std::stringstream   ss;
    if (out_alllog)
    {
        ss << GlobalV::global_out_dir << "running_lr_" << GlobalV::MY_RANK + 1 << ".log";
        GlobalV::ofs_running.open(ss.str());
    }
    else
    {
        if (GlobalV::MY_RANK == 0)
        {
            ss << GlobalV::global_out_dir << "running_lr.log";
            GlobalV::ofs_running.open(ss.str());
        }
    }
}

template<typename T, typename TR>
void ModuleESolver::ESolver_LRTD<T, TR>::parameter_check()
{
    if (std::is_same<T, std::complex<double>>::value && this->nsk / this->nspin > 1 && this->input.lr_solver == "lapack")
        throw std::invalid_argument("ESolver_LRTD: explicitly contruct A matrix is not supported for multi-k due to the complex density matrix.");
}

template<typename T, typename TR>
ModuleESolver::ESolver_LRTD<T, TR>::ESolver_LRTD(ModuleESolver::ESolver_KS_LCAO<T, TR>&& ks_sol,
    Input& inp, UnitCell& ucell) : input(inp), ucell(ucell)
{
    redirect_log(inp.out_alllog);
    ModuleBase::TITLE("ESolver_LRTD", "ESolver_LRTD");

    // xc kernel
    this->xc_kernel = inp.xc_kernel;
    std::transform(xc_kernel.begin(), xc_kernel.end(), xc_kernel.begin(), tolower);
    //check the input first
    if (xc_kernel != "rpa" && xc_kernel != "lda" && xc_kernel != "hf")
        throw std::invalid_argument("ESolver_LRTD: unknown type of xc_kernel");

    // move the ground state info 
    this->psi_ks = ks_sol.psi;
    ks_sol.psi = nullptr;


    // test: 强制psi_ks第一个元素为正
    // psi_ks->fix_kb(0, 0);
    // for (int j = 0;j < psi_ks->get_nbands();++j)  //nbands
    // {
    //     if (getreal(psi_ks->get_pointer()[j * psi_ks->get_nbasis()]) < 0)
    //         for (int i = 0;i < psi_ks->get_nbasis();++i)  //nlocal
    //             psi_ks->get_pointer()[j * psi_ks->get_nbasis() + i] *= -1;
    // }

    //only need the eigenvalues. the 'elecstates' of excited states is different from ground state.
    this->eig_ks = std::move(ks_sol.pelec->ekb);

    //kv
    this->nspin = GlobalV::NSPIN;
    this->kv = std::move(ks_sol.kv);
    this->nsk = std::is_same<T, double>::value ? this->nspin : this->kv.nks;

    this->parameter_check();

    //allocate 2-particle state and setup 2d division
    this->nbasis = GlobalV::NLOCAL;
    this->nstates = inp.nstates;
    this->init_X(inp.nvirt);

    //grid integration
    this->gt = std::move(ks_sol.GridT);
    if (std::is_same<T, double>::value)
        this->gint_g = std::move(ks_sol.UHM.GG);
    else
        this->gint_k = std::move(ks_sol.UHM.GK);
    this->set_gint();

    // move pw basis
    this->pw_rho = ks_sol.pw_rho;
    ks_sol.pw_rho = nullptr;
    //init potential and calculate kernels using ground state charge
    init_pot(*ks_sol.pelec->charge);

    LR_Util::setup_2d_division(this->paraMat_, 1, this->nbasis, this->nbasis, this->paraX_.comm_2D, this->paraX_.blacs_ctxt);
    this->paraMat_.atom_begin_row = std::move(ks_sol.LM.ParaV->atom_begin_row);
    this->paraMat_.atom_begin_col = std::move(ks_sol.LM.ParaV->atom_begin_col);
    this->paraMat_.iat2iwt_ = ucell.get_iat2iwt();

#ifdef __EXX
    if (xc_kernel == "hf")
    {
        //complex-double problem....waiting for refactor
        // this->exx_lri = std::make_shared<Exx_LRI<double>>(ks_sol.exx_lri_double);
        // if calculated in the esolver_ks, move it
        if (ks_sol.exx_lri_double && std::is_same<T, double>::value) this->move_exx_lri(ks_sol.exx_lri_double);
        else if (ks_sol.exx_lri_complex && std::is_same<T, std::complex<double>>::value) this->move_exx_lri(ks_sol.exx_lri_complex);
        else    // construct C, V from scratch
        {
            this->exx_lri = std::make_shared<Exx_LRI<T>>(GlobalC::exx_info.info_ri);
            this->exx_lri->init(MPI_COMM_WORLD, this->kv); // using GlobalC::ORB
            this->exx_lri->cal_exx_ions();
        }
    }
#endif
    this->init_A(dynamic_cast<hamilt::HamiltLCAO<T, TR>*>(ks_sol.p_hamilt)->getHR(), inp.lr_thr);
    this->lr_solver = inp.lr_solver;
    this->pelec = new elecstate::ElecStateLCAO<T>();
}


template<typename T, typename TR>
ModuleESolver::ESolver_LRTD<T, TR>::ESolver_LRTD(Input& inp, UnitCell& ucell) : input(inp), ucell(ucell)
{
    redirect_log(inp.out_alllog);
    ModuleBase::TITLE("ESolver_LRTD", "ESolver_LRTD");
    // xc kernel
    this->xc_kernel = inp.xc_kernel;
    std::transform(xc_kernel.begin(), xc_kernel.end(), xc_kernel.begin(), tolower);
    //check the input first
    if (xc_kernel != "rpa" && xc_kernel != "lda" && xc_kernel != "hf")
        throw std::invalid_argument("ESolver_LRTD: unknown type of xc_kernel");

    // necessary steps in ESolver_FP
    ESolver_FP::Init(inp, ucell);
    this->pelec = new elecstate::ElecStateLCAO<T>();

    // necessary steps in ESolver_KS::Init : symmetry and k-points
    ucell.cal_nelec(GlobalV::nelec);
    if (ModuleSymmetry::Symmetry::symm_flag == 1)
    {
        this->symm.analy_sys(ucell, GlobalV::ofs_running);
        ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "SYMMETRY");
    }
    this->nspin = GlobalV::NSPIN;
    std::cout << "nspin: " << this->nspin << std::endl;
    this->kv.set(this->symm, GlobalV::global_kpoint_card, nspin, ucell.G, ucell.latvec);
    this->nsk = std::is_same<T, double>::value ? this->nspin : this->kv.nks;
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "INIT K-POINTS");
    Print_Info::setup_parameters(ucell, this->kv);

    this->parameter_check();

    // necessary steps in ESolver_KS_LCAO::Init_Basis_lcao
    // read orbitals
    ORB_control orb_con;
    orb_con.read_orb_first(GlobalV::ofs_running,
        GlobalC::ORB,
        ucell.ntype,
        GlobalV::global_orbital_dir,
        ucell.orbital_fn,
        ucell.descriptor_file,
        ucell.lmax,
        inp.lcao_ecut,
        inp.lcao_dk,
        inp.lcao_dr,
        inp.lcao_rmax,
        0, //deepks_setorb,
        inp.out_mat_r,
        0,//cal_force
        GlobalV::MY_RANK);

    //  setup 2d-block distribution for AO-matrix and KS wfc
    this->nbasis = GlobalV::NLOCAL;
    LR_Util::setup_2d_division(this->paraMat_, 1, this->nbasis, this->nbasis);
    int err = this->paraMat_.set_nloc_wfc_Eij(GlobalV::NBANDS, GlobalV::ofs_running, GlobalV::ofs_warning);
    this->paraMat_.set_atomic_trace(ucell.get_iat2iwt(), ucell.nat, this->nbasis);

    // read the ground state info
    this->read_ks_wfc();

    //allocate 2-particle state and setup 2d division
    this->nstates = inp.nstates;
    this->init_X(inp.nvirt);

    // read the ground state charge density and calculate xc kernel
    GlobalC::Pgrid.init(this->pw_rho->nx,
        this->pw_rho->ny,
        this->pw_rho->nz,
        this->pw_rho->nplane,
        this->pw_rho->nrxx,
        pw_big->nbz,
        pw_big->bz);
    Charge chg_gs;
    this->read_ks_chg(chg_gs);
    this->init_pot(chg_gs);

    // search adjacent atoms and init Gint
    std::cout << "ucell.infoNL.get_rcutmax_Beta(): " << GlobalC::ucell.infoNL.get_rcutmax_Beta() << std::endl;
    GlobalV::SEARCH_RADIUS = atom_arrange::set_sr_NL(GlobalV::ofs_running,
        GlobalV::OUT_LEVEL,
        GlobalC::ORB.get_rcutmax_Phi(),
        GlobalC::ucell.infoNL.get_rcutmax_Beta(),
        GlobalV::GAMMA_ONLY_LOCAL);
    atom_arrange::search(GlobalV::SEARCH_PBC,
        GlobalV::ofs_running,
        GlobalC::GridD,
        this->ucell,
        GlobalV::SEARCH_RADIUS,
        GlobalV::test_atom_input);
    this->set_gint();
    this->gint->gridt = &this->gt;
    this->gt.set_pbc_grid(this->pw_rho->nx,
        this->pw_rho->ny,
        this->pw_rho->nz,
        this->pw_big->bx,
        this->pw_big->by,
        this->pw_big->bz,
        this->pw_big->nbx,
        this->pw_big->nby,
        this->pw_big->nbz,
        this->pw_big->nbxx,
        this->pw_big->nbzp_start,
        this->pw_big->nbzp,
        this->pw_rho->ny,
        this->pw_rho->nplane,
        this->pw_rho->startz_current);
    if (std::is_same<T, std::complex<double>>::value)
    {
        this->gt.cal_nnrg(&this->paraMat_);
        this->gint_k.allocate_pvpR();   // uses gt.nnrg
    }
    this->gint->prep_grid(this->gt, this->pw_big->nbx, this->pw_big->nby, this->pw_big->nbzp, this->pw_big->nbzp_start,
        this->pw_rho->nxyz, this->pw_big->bx, this->pw_big->by, this->pw_big->bz, this->pw_big->bxyz, this->pw_big->nbxx,
        this->pw_rho->ny, this->pw_rho->nplane, this->pw_rho->startz_current);
    this->gint->initialize_pvpR(ucell, &GlobalC::GridD);

    // if EXX from scratch, init 2-center integral and calclate Cs, Vs 
#ifdef __EXX
    if (xc_kernel == "hf")
    {
#ifndef USE_NEW_TWO_CENTER
        int Lmax = GlobalC::exx_info.info_ri.abfs_Lmax;
        this->orb_con.set_orb_tables(GlobalV::ofs_running,
            GlobalC::UOT,
            GlobalC::ORB,
            ucell.lat0,
            GlobalV::deepks_setorb,
            Lmax,
            ucell.infoNL.nprojmax,
            ucell.infoNL.nproj,
            ucell.infoNL.Beta);
#else
        two_center_bundle.reset(new TwoCenterBundle);
        two_center_bundle->build(ucell.ntype, ucell.orbital_fn, nullptr/*ucell.infoNL.Beta*/,
            GlobalV::deepks_setorb, &ucell.descriptor_file);
        GlobalC::UOT.two_center_bundle = std::move(two_center_bundle);
#endif
        std::cout << GlobalC::exx_info.info_ri.dm_threshold << std::endl;
        this->exx_lri = std::make_shared<Exx_LRI<T>>(GlobalC::exx_info.info_ri);
        this->exx_lri->init(MPI_COMM_WORLD, this->kv); // using GlobalC::ORB
        this->exx_lri->cal_exx_ions();
    }
    else
#endif
        ModuleBase::Ylm::set_coefficients();    // set Ylm only for Gint 

    this->init_A(nullptr, inp.lr_thr);
    this->lr_solver = inp.lr_solver;
    this->pelec = new elecstate::ElecState();
}
template<typename T, typename TR>
void ModuleESolver::ESolver_LRTD<T, TR>::Run(int istep, UnitCell& cell)
{
    ModuleBase::TITLE("ESolver_LRTD", "Run");
    this->phsol->solve(this->p_hamilt, *this->X, this->pelec, this->lr_solver);
    return;
}

template<typename T, typename TR>
void ModuleESolver::ESolver_LRTD<T, TR>::postprocess()
{
    //cal spectrum
    LR_Spectrum<T> spectrum(this->pelec->ekb.c, *this->X, this->nspin, this->nbasis, this->nocc, this->nvirt, this->gint, *this->pw_rho, *this->psi_ks, this->ucell, this->kv, this->paraX_, this->paraC_, this->paraMat_);
    spectrum.oscillator_strength();
    spectrum.transition_analysis();
    std::vector<double> freq(100);
    std::vector<double> abs_wavelen_range({ 20, 200 });//default range
    if (input.abs_wavelen_range.size() == 2 && std::abs(input.abs_wavelen_range[1] - input.abs_wavelen_range[0]) > 0.02)
        abs_wavelen_range = input.abs_wavelen_range;
    double lambda_diff = std::abs(abs_wavelen_range[1] - abs_wavelen_range[0]);
    double lambda_min = std::min(abs_wavelen_range[1], abs_wavelen_range[0]);
    for (int i = 0;i < freq.size();++i)freq[i] = 91.126664 / (lambda_min + 0.01 * static_cast<double>(i + 1) * lambda_diff);
    double eta = 0.01;
    spectrum.optical_absorption(freq, eta);
}

template<typename T, typename TR>
void ModuleESolver::ESolver_LRTD<T, TR>::init_X(const int& nvirt_input)
{
    ModuleBase::TITLE("ESolver_LRTD", "Init");
    // calculate the number of occupied and unoccupied states
    // which determines the basis size of the excited states
    this->nocc = LR_Util::cal_nocc(LR_Util::cal_nelec(ucell));
    this->nvirt = this->eig_ks.nc - nocc;   //nbands-nocc
    if (nvirt_input > this->nvirt)
        GlobalV::ofs_warning << "ESolver_LRTD: input nvirt is too large to cover by nbands, set nvirt = nbands - nocc = " << this->nvirt << std::endl;
    else if (nvirt_input > 0) this->nvirt = nvirt_input;
    this->npairs = this->nocc * this->nvirt;
    if (this->nstates > this->nocc * this->nvirt)
        throw std::invalid_argument("ESolver_LRTD: nstates > nocc*nvirt");

    GlobalV::ofs_running << "Setting LR-TDDFT parameters: " << std::endl;
    GlobalV::ofs_running << "number of occupied bands: " << this->nocc << std::endl;
    GlobalV::ofs_running << "number of virtual bands: " << this->nvirt << std::endl;
    GlobalV::ofs_running << "number of Atom orbitals (LCAO-basis size): " << this->nbasis << std::endl;
    GlobalV::ofs_running << "number of KS bands: " << this->eig_ks.nc << std::endl;
    GlobalV::ofs_running << "number of electron-hole pairs (2-particle basis size): " << this->npairs << std::endl;
    GlobalV::ofs_running << "number of excited states to be solved: " << this->nstates << std::endl;

    // setup ParaX
    LR_Util::setup_2d_division(this->paraX_, 1, this->nvirt, this->nocc);//nvirt - row, nocc - col 
    this->X = new psi::Psi<T>(this->nsk, this->nstates, this->paraX_.get_local_size(), nullptr, false);  // band(state)-first
    this->X->zero_out();

    // set the initial guess of X
    // if (E_{lumo}-E_{homo-1} < E_{lumo+1}-E{homo}), mode = 0, else 1(smaller first)
    bool ix_mode = 0;   //default
    if (this->eig_ks.nc > nocc + 1 && nocc >= 2 &&
        eig_ks(0, nocc) - eig_ks(0, nocc - 2) - 1e-5 > eig_ks(0, nocc + 1) - eig_ks(0, nocc - 1))
        ix_mode = 1;
    GlobalV::ofs_running << "setting the initial guess of X: " << std::endl;
    if (nocc >= 2 && eig_ks.nc > nocc)GlobalV::ofs_running << "E_{lumo}-E_{homo-1}=" << eig_ks(0, nocc) - eig_ks(0, nocc - 2) << std::endl;
    if (nocc >= 1 && eig_ks.nc > nocc + 1) GlobalV::ofs_running << "E_{lumo+1}-E{homo}=" << eig_ks(0, nocc + 1) - eig_ks(0, nocc - 1) << std::endl;
    GlobalV::ofs_running << "mode of X-index: " << ix_mode << std::endl;

    /// global index map between (i,c) and ix
    ModuleBase::matrix ioiv2ix;
    std::vector<std::pair<int, int>> ix2ioiv;
    std::pair<ModuleBase::matrix, std::vector<std::pair<int, int>>> indexmap =
        LR_Util::set_ix_map_diagonal(ix_mode, nocc, nvirt);

    ioiv2ix = std::move(std::get<0>(indexmap));
    ix2ioiv = std::move(std::get<1>(indexmap));
    
    // use unit vectors as the initial guess
    // for (int i = 0; i < std::min(this->nstates * GlobalV::PW_DIAG_NDIM, nocc * nvirt); i++)
    for (int i = 0; i < nstates; i++)
    {
        this->X->fix_b(i);
        int occ_global = std::get<0>(ix2ioiv[i]);   // occ
        int virt_global = std::get<1>(ix2ioiv[i]);   // virt
        if (this->paraX_.in_this_processor(virt_global, occ_global))
            for (int isk = 0;isk < this->nsk;++isk)
                (*X)(isk, this->paraX_.global2local_col(occ_global) * this->paraX_.get_row_size() + this->paraX_.global2local_row(virt_global)) = static_cast<T>(1.0);
    }
    this->X->fix_b(0);  //recover the pointer

    LR_Util::setup_2d_division(this->paraC_, 1, this->nbasis, this->nocc + this->nvirt, this->paraX_.comm_2D, this->paraX_.blacs_ctxt);
}

template<typename T, typename TR>
void ModuleESolver::ESolver_LRTD<T, TR>::init_A(hamilt::HContainer<double>* pHR_in, const double lr_thr)
{
    //HContainer-based DensityMatrix
    hamilt::HContainer<double>*& pHR = pHR_in;
    if (!pHR)
    {
        pHR = new hamilt::HContainer<TR>(&this->paraMat_);
        for (int iat1 = 0; iat1 < ucell.nat; iat1++)
        {
            auto tau1 = this->ucell.get_tau(iat1);
            int T1, I1;
            this->ucell.iat2iait(iat1, &I1, &T1);
            AdjacentAtomInfo adjs;
            GlobalC::GridD.Find_atom(this->ucell, tau1, T1, I1, &adjs);
            for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
            {
                const int T2 = adjs.ntype[ad];
                const int I2 = adjs.natom[ad];
                int iat2 = this->ucell.itia2iat(T2, I2);
                if (this->paraMat_.get_row_size(iat1) <= 0 || this->paraMat_.get_col_size(iat2) <= 0) continue;
                const ModuleBase::Vector3<int>& R_index = adjs.box[ad];
                const LCAO_Orbitals& orb = LCAO_Orbitals::get_const_instance();
                if (this->ucell.cal_dtau(iat1, iat2, R_index).norm() * this->ucell.lat0
                    >= orb.Phi[T1].getRcut() + orb.Phi[T2].getRcut()) continue;
                hamilt::AtomPair<TR> tmp(iat1, iat2, R_index.x, R_index.y, R_index.z, &this->paraMat_);
                pHR->insert_pair(tmp);
            }
        }
        pHR->allocate(true);
        if (std::is_same<T, double>::value) pHR->fix_gamma();
    }
    pHR->set_paraV(&this->paraMat_);
    this->p_hamilt = new hamilt::HamiltCasidaLR<T>(xc_kernel, this->nspin, this->nbasis, this->nocc, this->nvirt, this->ucell, this->psi_ks, this->eig_ks, pHR,
#ifdef __EXX
        this->exx_lri.get(),
#endif
        this->gint, this->pot, this->kv, & this->paraX_, & this->paraC_, & this->paraMat_);

    // init HSolver
    this->phsol = new hsolver::HSolverLR<T>(this->nsk, this->npairs);
    this->phsol->set_diagethr(0, 0, std::max(1e-13, lr_thr));
}

template<typename T, typename TR>
void ModuleESolver::ESolver_LRTD<T, TR>::init_pot(const Charge& chg_gs)
{
    if (this->pot == nullptr)
    {
        this->pot = new elecstate::PotHxcLR(xc_kernel,
            this->pw_rho,
            &ucell,
            &chg_gs);
    };
}

inline void quit_readwfc_err(int error)
{
    if (error == 1)
    {
        ModuleBase::WARNING_QUIT("Local_Orbital_wfc", "Can't find the wave function file: LOWF.dat");
    }
    else if (error == 2)
    {
        ModuleBase::WARNING_QUIT("Local_Orbital_wfc", "In wave function file, band number doesn't match");
    }
    else if (error == 3)
    {
        ModuleBase::WARNING_QUIT("Local_Orbital_wfc", "In wave function file, nlocal doesn't match");
    }
    else if (error == 4)
    {
        ModuleBase::WARNING_QUIT("Local_Orbital_wfc", "In k-dependent wave function file, k point is not correct");
    }
}

template<>
void ModuleESolver::ESolver_LRTD<double, double>::read_ks_wfc()
{
    GlobalV::NB2D = 1;
    this->pelec->ekb.create(this->nspin, GlobalV::NBANDS);
    this->pelec->wg.create(this->nspin, GlobalV::NBANDS);
    this->psi_ks = new psi::Psi<double>(this->nspin, this->paraMat_.ncol_bands, this->paraMat_.get_row_size());
    for (int is = 0;is < this->nspin;++is)
    {
        double** ctot;
        // caution: pelec needs init; 
        // caution: a full paraO is needed
        int error = ModuleIO::read_wfc_nao(ctot, is, &this->paraMat_, this->psi_ks, pelec);
#ifdef _MPI
        Parallel_Common::bcast_int(error);
#endif
        quit_readwfc_err(error);
    }
    this->eig_ks = std::move(this->pelec->ekb);
}
template<>
void ModuleESolver::ESolver_LRTD<std::complex<double>, double>::read_ks_wfc()
{
    GlobalV::NB2D = 1;
    this->pelec->ekb.create(this->kv.nks, GlobalV::NBANDS);
    this->pelec->wg.create(this->kv.nks, GlobalV::NBANDS);
    this->psi_ks = new psi::Psi<std::complex<double>>(this->kv.nks, this->paraMat_.ncol_bands, this->paraMat_.get_row_size());
    for (int ik = 0;ik < this->kv.nks;++ik)
    {
        std::complex<double>** ctot;
        // caution: pelec needs init; 
        // caution: a full paraO is needed
        int error = ModuleIO::read_wfc_nao_complex(ctot, ik, this->kv.kvec_c[ik], &this->paraMat_, this->psi_ks, this->pelec);
#ifdef _MPI
        Parallel_Common::bcast_int(error);
#endif
        quit_readwfc_err(error);
    }
    this->eig_ks = std::move(this->pelec->ekb);
}

template<typename T, typename TR>
void ModuleESolver::ESolver_LRTD<T, TR>::read_ks_chg(Charge& chg_gs)
{
    chg_gs.set_rhopw(this->pw_rho);
    chg_gs.allocate(this->nspin);
    GlobalV::ofs_running << " try to read charge from file : ";
    for (int is = 0; is < this->nspin; ++is)
    {
        std::stringstream ssc;
        ssc << GlobalV::global_readin_dir << "SPIN" << is + 1 << "_CHG.cube";
        GlobalV::ofs_running << ssc.str() << std::endl;
        double ef;
        if (ModuleIO::read_rho(
#ifdef __MPI
            & (GlobalC::Pgrid),
#endif
            is,
            this->nspin,
            ssc.str(),
            chg_gs.rho[is],
            this->pw_rho->nx,
            this->pw_rho->ny,
            this->pw_rho->nz,
            ef,
            &(GlobalC::ucell),
            chg_gs.prenspin))
            GlobalV::ofs_running << " Read in the charge density: " << ssc.str() << std::endl;
        else    // prenspin for nspin=4 is not supported currently
            ModuleBase::WARNING_QUIT(
                "init_rho",
                "!!! Couldn't find the charge file !!! The default directory \n of SPIN1_CHG.cube is OUT.suffix, "
                "or you must set read_file_dir \n to a specific directory. ");
    }
}