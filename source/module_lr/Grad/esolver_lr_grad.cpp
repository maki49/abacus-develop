#include "../esolver_lrtd_lcao.h"
#include "module_lr/Grad/multipliers/zeq_solver.h"
#include "module_lr/Grad/multipliers/cal_edm_from_multipliers.h"
#include "module_lr/Grad/force/lr_force.h"
#include "module_elecstate/module_dm/cal_dm_psi.h"
#include "module_io/output_log.h"

template <typename Tstream>
inline void print_force(const std::vector<ModuleBase::matrix>& force, Tstream& ofs)
{
    const int nstate = force.size();
    ofs << "Gradients of each excited state: (eV/Angstrom)" << std::endl;
    ofs << std::setw(6) << "state" << std::setw(6) << "atom"
        << std::setw(15) << "x" << std::setw(15) << "y" << std::setw(15) << "z" << std::endl;
    const double fac = ModuleBase::Ry_to_eV / ModuleBase::BOHR_TO_A;
    for (int i = 0;i < nstate;++i)
    {
        for (int iat = 0;iat < force[i].nr;++iat)
        {
            std::string istate = iat == 0 ? std::to_string(i) : " ";
            ofs << std::setw(6) << istate << std::setw(6) << iat << std::setw(6) << "force";
            for (int ixyz = 0;ixyz < 3;++ixyz) { ofs << std::setw(15) << force[i](iat, ixyz) * fac; }
            ofs << std::endl;
        }
    }
}
template<typename T, typename TR>
void LR::ESolver_LR<T, TR>::init_pot_groundstate(const Charge& chg_gs)
{
    ModuleBase::TITLE("ESolver_LR", "init_pot_gs");
    std::vector<std::string> pot_register;
    if (PARAM.inp.vl_in_h)
    {
        //! 11) calculate the structure factor
        this->sf.setup_structure_factor(&ucell, Pgrid, this->pw_rhod);
        this->locpp.init_vloc(this->ucell, this->pw_rho);
        pot_register.push_back("local");
    }
    if(PARAM.inp.vh_in_h)
    {
        pot_register.push_back("hartree");
    }
    pot_register.push_back("xc");
    
    // initialize the ground state potential
    this->pot_gs = LR_Util::make_unique<elecstate::Potential>(this->pw_rhod, this->pw_rho,
        &this->ucell, &this->locpp.vloc, &this->sf, &this->solvent,
       &this->etxc_gs, &this->vtxc_gs);
    this->pot_gs.get()->pot_register(pot_register);
    XC_Functional::set_xc_type(ucell.atoms[0].ncpp.xc_func);    // set XC type of the ground state
    this->pot_gs->init_pot(0, &chg_gs);  // call update_from_charge inside
    if (has_local_xc(this->xc_kernel))
    {
        XC_Functional::set_xc_type(this->xc_kernel);    // recover the excited state xc kernel type
    }
    if (PARAM.inp.test_force)
    {
        this->pot_gs_hartree = LR_Util::make_unique<elecstate::Potential>(this->pw_rhod, this->pw_rho,
            &this->ucell, &this->locpp.vloc, &this->sf, &this->solvent,
            &this->etxc_gs, &this->vtxc_gs);
        this->pot_gs_hartree->pot_register({ "hartree" });
        this->pot_gs_hartree->init_pot(0, &chg_gs);  // call update_from_charge inside
    }
}

template<typename T, typename TR>
ct::Tensor LR::ESolver_LR<T, TR>::solve_zvector_eqation(const int ispin)
{
    ModuleBase::TITLE("ESolver_LR", "cal_force");
    ModuleBase::timer::tick("ESolver_LR", "solve_zvector_eqation");
    ct::Tensor Z = LR_Util::newTensor<T>({ this->nstates, this->nloc_per_band });
    // construct and solve the Z-vector equation
    Z_vector_equation(this->X[ispin].template data<T>(), Z.template data<T>(),
        this->xc_kernel, this->nstates, this->nspin, this->nbasis, this->nocc, this->nvirt,
        this->ucell, orb_cutoff_, this->gd, *this->psi_ks, this->eig_ks,
#ifdef __EXX    
        std::weak_ptr<Exx_LRI<T>>(this->exx_lri), this->exx_info.info_global.hybrid_alpha,
#endif
        this->gint_, std::weak_ptr<PotHxcLR>(this->pot[ispin]), this->kv,
        this->paraX_, this->paraC_, this->paraMat_, this->spin_types[ispin]);
    ModuleBase::timer::tick("ESolver_LR", "solve_zvector_eqation");
    return Z;
}

template<typename T, typename TR>
std::vector<ModuleBase::matrix> LR::ESolver_LR<T, TR>::cal_force(const int ispin)
{
    if (PARAM.inp.test_force) { this->test_force(ispin); }

    const ct::Tensor& Z = this->solve_zvector_eqation(ispin);

    ModuleBase::TITLE("ESolver_LR", "cal_force");
    ModuleBase::timer::tick("ESolver_LR", "cal_force");
    const auto& c = LR_Util::get_psi_spin(*this->psi_ks, ispin, this->nk);   // wavefunction coefficients of ground state

    // calculate the force (the partial gradient of Lagrangian)
    LR_Force<T> lr_force(this->ucell, this->kv.kvec_d, this->paraMat_, *this->pw_rho, this->locpp, this->sf, this->gd, this->gint_, this->two_center_bundle_);
    GlobalV::ofs_running << "Start to calculate excited-state force of  " << this->spin_types[ispin] << std::endl;
    // ground state dm for currrent spin (only for test the correctness of the force)
    // elecstate::DensityMatrix<T, T> dm_gs(this->paraMat_, 1, this->kv.kvec_d, this->nk);

    // for each state, calculate dm_trans, dm_relaxed_diff, edm and force
    std::vector<ModuleBase::matrix> forces(this->nstates);
    for (int istate = 0;istate < this->nstates;++istate)
    {
        const int offset = istate * this->nloc_per_band;
        // The imag part will be cancelled in the force calculation, so we use double DM(R) to calculate force. 
        // But complex transition DM(R) is still used in energy density matrix calculation.
        elecstate::DensityMatrix<T, double> dm_trans_real =   // D(X)
            LR_Util::build_dm_from_dmk<T, double>(
                cal_dm_trans_pblas(this->X[ispin].template data<T>() + offset, this->paraX_[ispin], c, this->paraC_, this->nbasis, this->nocc[ispin], this->nvirt[ispin], this->paraMat_),
                this->paraMat_, this->nk, this->kv.kvec_d, this->ucell, this->gd, this->orb_cutoff_);

        elecstate::DensityMatrix<T, T> relaxed_diff_dm =    // T+D(Z), (R) can be complex
            LR_Util::build_dm_from_dmk<T, T>(
                // LR_Util::operator+(
                    cal_dm_diff_pblas(this->X[ispin].template data<T>() + offset, this->paraX_[ispin], c, this->paraC_, this->nbasis, this->nocc[ispin], this->nvirt[ispin], this->paraMat_)
                    + cal_dm_trans_pblas(Z.template data<T>() + offset, this->paraX_[ispin], c, this->paraC_, this->nbasis, this->nocc[ispin], this->nvirt[ispin], this->paraMat_)
                    ,// ),
                this->paraMat_, this->nk, this->kv.kvec_d, this->ucell, this->gd, this->orb_cutoff_);
        elecstate::DensityMatrix<T, double> relaxed_diff_dm_real(&this->paraMat_, 1, this->kv.kvec_d, this->nk);
        LR_Util::initialize_DMR(relaxed_diff_dm_real, this->paraMat_, this->ucell, this->gd, this->orb_cutoff_);
        LR_Util::get_DMR_real_imag_part(relaxed_diff_dm, relaxed_diff_dm_real, this->ucell.nat, 'R');

        // get edm of type DensityMatrix
        // weak_ptr here is to avoid "could not match 'weak_ptr' against 'shared_ptr'"
        // but why there're no bug in the previous code (HamiltLR and HamiltULF)? 
        // seems because those two are classes having constructors 
        // but `cal_edm_from_XZ_istate` here is a functions
        std::weak_ptr<PotHxcLR> pot_weak = this->pot[ispin];
        std::weak_ptr<PotHxcLR> pot_hxc_gs_weak = this->pot_hxc_gs;
#ifdef __EXX
        std::weak_ptr<Exx_LRI<T>> exx_lri_weak = this->exx_lri;
#endif
        const std::vector<ct::Tensor>& edm_k =
            cal_edm_from_XZ_istate(this->X[ispin].template data<T>() + offset,
                Z.template data<T>() + offset,
                this->pelec->ekb.c[ ispin * nstates + istate],
                // pack the following as a struct or use parameter package
                this->eig_ks.c, relaxed_diff_dm,
                c, this->nspin, this->nbasis, this->nocc, this->nvirt, this->ucell, this->orb_cutoff_,
#ifdef __EXX
                exx_lri_weak, this->exx_info.info_global.hybrid_alpha,
#endif
                this->gint_, pot_weak, pot_hxc_gs_weak,
                this->kv, this->gd, this->paraX_, this->paraC_, this->paraMat_,
                has_local_xc(this->xc_kernel));
        elecstate::DensityMatrix<T, double> edm_real = LR_Util::build_dm_from_dmk<T, double>(edm_k,
            this->paraMat_, this->nk, this->kv.kvec_d, this->ucell, this->gd, this->orb_cutoff_);

        ModuleBase::matrix force_hxc_dmtrans = lr_force.cal_force_hxc_dmtrans(dm_trans_real, *this->pot[ispin]);
        std::cout << "Force (Hxc-DMTrans term) of state " << istate << ": " << std::endl;
        LR_Util::print_value(force_hxc_dmtrans.c, ucell.nat, 3);
        ModuleBase::matrix force_hamiltgs_relaxed_diff = lr_force.cal_force_hamilt_gs_dm_relaxed_diff(relaxed_diff_dm_real, *pot_gs);
        std::cout << "Force (GS-(T+Z) term) of state " << istate << ": " << std::endl;
        LR_Util::print_value(force_hamiltgs_relaxed_diff.c, ucell.nat, 3);
        ModuleBase::matrix force_overlap_edm = lr_force.cal_force_overlap_edm(edm_real);    // "-" sign has been included in the force factor
        std::cout << "Force (Overlap-EDM term) of state " << istate << ": " << std::endl;
        LR_Util::print_value(force_overlap_edm.c, ucell.nat, 3);
        // remaining for EXX
        forces[istate] = force_hxc_dmtrans + force_hamiltgs_relaxed_diff + force_overlap_edm;
    }
    ModuleBase::timer::tick("ESolver_LR", "cal_force");
    print_force(forces, std::cout);
    print_force(forces, GlobalV::ofs_running);
    return forces;
}

template<typename T, typename TR>
void LR::ESolver_LR<T, TR>::test_force(const int ispin)
{
    LR_Force<T> lr_force(this->ucell, this->kv.kvec_d, this->paraMat_, *this->pw_rho, this->locpp, this->sf, this->gd, this->gint_, this->two_center_bundle_);

    elecstate::DensityMatrix<T, double> dm_gs(&this->paraMat_, this->nspin, this->kv.kvec_d, this->nk);   //DX
    elecstate::cal_dm_psi(&this->paraMat_, this->wg_ks, *this->psi_ks, dm_gs);
    LR_Util::initialize_DMR(dm_gs, this->paraMat_, this->ucell, this->gd, this->orb_cutoff_);
    dm_gs.cal_DMR();

    ///========================== test 1: reproduce the force of ground state =========================
    // energy density matrix of the ground state
    elecstate::DensityMatrix<T, double> edm_gs(&this->paraMat_, this->nspin, this->kv.kvec_d, this->nk);   //DX
    ModuleBase::matrix wg_ekb_ks(nspin, nbands);
    std::transform(this->wg_ks.c, this->wg_ks.c + nspin * nbands, this->eig_ks.c, wg_ekb_ks.c, std::multiplies<double>());
    elecstate::cal_dm_psi(&this->paraMat_, wg_ekb_ks, *this->psi_ks, edm_gs);
    LR_Util::initialize_DMR(edm_gs, this->paraMat_, this->ucell, this->gd, this->orb_cutoff_);
    edm_gs.cal_DMR();
    // ground-state force
    ModuleBase::matrix force_gs = lr_force.reproduce_force_gs(dm_gs, edm_gs, *this->pot_gs);
    ModuleIO::print_force(GlobalV::ofs_running, this->ucell, "Ground State FORCE (eV/Angstrom)", force_gs, false);
    /// ======================================= END test 1 =========================================
    ///========================== test 2: reproduce the DX Hartree term =========================
    ModuleBase::matrix f_hxc_potgs = lr_force.reproduce_force_gs_loc(dm_gs, *this->pot_gs_hartree);
    ModuleIO::print_force(GlobalV::ofs_running, this->ucell, "GS Hartree force calculated by 'cal_pulay_fs' from potential (eV/Angstrom)", f_hxc_potgs, false);
    ModuleBase::matrix f_hxc_potlr = lr_force.cal_force_hxc_dmtrans(dm_gs, *this->pot[ispin]);
    ModuleIO::print_force(GlobalV::ofs_running, this->ucell, "2* GS Hxc force calculated by 'LR_Force' from kernel (eV/Angstrom)", f_hxc_potlr * 2, false);
    // 2 for spin in f->v. Spin in v->f is already multiplied in the singlet Hartree factor 2.
    /// ======================================= END test 2 =========================================

    // exit(0);
}

template class LR::ESolver_LR<double, double>;
template class LR::ESolver_LR<std::complex<double>, double>;
