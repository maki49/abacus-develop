// LR execution flow and final spectrum/density output.
#include "esolver_lr_lcao_tddft.h"
#include "source_lcao/module_lr/utils/lr_io.h"
#include "source_lcao/module_lr/utils/lr_util.h"
#include "source_lcao/module_lr/hamilt_ulr.h"
#include "source_lcao/module_lr/hsolver_lrtd.h"
#include "source_lcao/module_lr/lr_spectrum.h"
#include "source_lcao/module_lr/lr_density.h"
#include "source_lcao/module_lr/ri_benchmark/ri_benchmark.h"
#include "source_lcao/module_lr/lr_preconditioner.h"
#include "source_lcao/module_lr/zeq_solver.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include <algorithm>
#include <iomanip>
#ifdef __EXX
#include "source_lcao/module_lr/lr_ri_response.h"
#include "source_lcao/module_ri/minimal_gaussian_ri.h"
#include "source_lcao/module_ri/ri_util_bvk.h"
#include "source_base/element_name.h"
#include <RI/global/MPI_Wrapper.h>
#endif

using namespace LR;

#ifdef __EXX
namespace
{
// Gather fixed KS coefficients once, and describe the distributed transition rows.
// The approximate operator owns this snapshot; neither the physical Hamiltonian
// nor the ground-state EXX workspace participates in the inner iterations.
template<typename T>
RITransitionBasis<T> ri_transition_basis(const psi::Psi<T>& coefficients, const Parallel_2D& distribution,
    const OrbitalTransitionSpace& transitions, const K_Vectors& kv, int naos)
{
    RITransitionBasis<T> basis;
    basis.naos = naos;
    basis.nk = transitions.nk;
    const int spins = transitions.unrestricted ? 2 : 1;
    basis.nocc.assign(transitions.nocc.begin(), transitions.nocc.begin() + spins);
    basis.nvirt.assign(transitions.nvirt.begin(), transitions.nvirt.begin() + spins);
    basis.coefficients.resize(spins);
    for (int k = 0; k < basis.nk; ++k)
    {
        const auto& point = kv.kvec_d[k];
        basis.kpoints.push_back({{point.x, point.y, point.z}});
    }
    int spin_offset = 0;
    for (int spin = 0; spin < spins; ++spin)
    {
        const int no = basis.nocc[spin];
        const int nv = basis.nvirt[spin];
        const int bands = no + nv;
        auto& full = basis.coefficients[spin];
        full.resize(static_cast<std::size_t>(basis.nk) * bands * naos);
        const auto& pairs = transitions.pair_distribution[spin];
        for (int k = 0; k < basis.nk; ++k)
        {
            const int isk = spin * basis.nk + k;
            const T* source = &coefficients(isk, 0, 0);
            T* target = full.data() + k * bands * naos;
            LR_Util::gather_2d_to_full(distribution, source, target, false, naos, bands);
            for (int i = 0; i < pairs.get_col_size(); ++i)
            {
                const int occupied = pairs.local2global_col(i);
                for (int a = 0; a < pairs.get_row_size(); ++a)
                {
                    const int virtual_index = pairs.local2global_row(a);
                    const int index = spin_offset + k * no * nv + occupied * nv + virtual_index;
                    basis.local_to_global.push_back(index);
                }
            }
        }
        spin_offset += basis.nk * no * nv;
    }
    return basis;
}

// Read the physical exchange kernel's explicit coefficients. The prototype
// supports one radial kernel, so reject mixtures instead of silently replacing them.
double ri_exchange_settings(const std::string& xc_kernel, const Exx_Info& exx,
                            MinimalGaussianRISettings& settings)
{
    if (!exx_kernel_list().count(xc_kernel)) { return 0; }
    const auto& components = exx.info_ri.coulomb_param;
    if (components.size() != 1 || components.begin()->second.size() != 1)
    { throw std::invalid_argument("Gaussian RI proxy supports one Fock or erfc exchange component"); }
    const auto& component = *components.begin();
    const auto& parameters = component.second.front();
    if (component.first == Conv_Coulomb_Pot_K::Coulomb_Type::Erfc)
    { settings.exchange_omega = std::stod(parameters.at("omega")); }
    else if (component.first != Conv_Coulomb_Pot_K::Coulomb_Type::Fock)
    { throw std::invalid_argument("Unsupported Gaussian RI proxy exchange kernel"); }
    const double alpha = xc_kernel == "hf" ? 1.0 : exx.info_global.hybrid_alpha;
    return alpha * std::stod(parameters.at("alpha"));
}
}
#endif

template <typename T, typename TR>
ModuleESolver::ESolver_LR<T, TR>::ESolver_LR(const Input_para& inp,
                                            const std::string& in_dir,
                                            const std::string& out_dir)
    : in_dir(in_dir), out_dir(out_dir)
{
#ifdef __EXX
    init_exx_info(this->exx_info, inp);
#endif
}

template <typename T, typename TR>
void ModuleESolver::ESolver_LR<T, TR>::runner(BaseCell& basecell, const int istep)
{
    basecell.require_kind(BaseCell::Kind::unitcell, __FUNCTION__);
    UnitCell& ucell = static_cast<UnitCell&>(basecell);

    ModuleBase::TITLE("ESolver_LR", "runner");
    ModuleBase::timer::start("ESolver_LR", "runner");

    if (this->ks_)
    {
        // `init_pot_groundstate` leaves the global XC type on the LR kernel, so put it back before
        // the ground-state SCF. `ESolver_KS::runner` is re-entrant: its `before_scf` rebuilds the
        // neighbour lists, the grid tables and the Hamiltonian for the geometry of this step.
        XC_Functional::set_xc_type(ucell.atoms[0].ncpp.xc_func);
        this->ks_->runner(ucell, istep);
        this->conv_esolver = this->ks_->conv_esolver;
        if (this->inp_->calculation == "relax" && !this->conv_esolver)
        {
            ModuleBase::WARNING_QUIT("ESolver_LR",
                "Excited-state relaxation requires a converged ground-state SCF; ionic step rejected");
        }
        this->etot_gs_ = this->ks_->cal_energy();
        if (!this->ks_initialized_)
        {
            this->initialize_from_ks_(ucell, *this->inp_);
            this->setup_relax_target_();   // needs the final dimensions, i.e. `openshell`
        }
        else { this->refresh_from_ks_(ucell); }
    }

    //allocate 2-particle state and setup 2d division
    this->setup_eigenvectors_X();
    this->pelec->ekb.create(nspin, this->nstates);

    // set once here rather than inside the solver branches: the open-shell `spectrum` branch used
    // to leave it empty, and both `after_all_runners` and the gradient index it
    this->spin_types = this->openshell ? std::vector<std::string>({ "updown" })
                                       : std::vector<std::string>({ "singlet", "triplet" });

    // a relaxation writes these once per ionic step; without the suffix every step would overwrite
    // the last, and the trajectory would be impossible to inspect afterwards
    const std::string step_suffix = this->excited_relax_ ? "_step" + std::to_string(istep) : "";
    auto efile_out = [&](const std::string& label)->std::string {return this->out_dir + "Excitation_Energy_" + label + step_suffix + ".dat";};
    auto vfile_out = [&](const std::string& label)->std::string {return this->out_dir + "Excitation_Amplitude_" + label + step_suffix + "_" + std::to_string(this->my_rank_+1) + ".dat";};
    if (this->inp_->lr_solver == "elpa")
    {
        ModuleBase::WARNING_QUIT("ESolver_LR", "ESolver_LR doesn't support elpa now.");

    }
    else if (this->inp_->lr_solver != "spectrum")
    {
        auto write_states = [&](const std::string& label, const Real<T>* e, const T* v, const int& dim, const int& nst, const int& prec = 8)->void
            {
                if (this->my_rank_ == 0) { assert(nst == LR_Util::write_value(efile_out(label), prec, e, nst)); }
                assert(nst * dim == LR_Util::write_value(vfile_out(label), prec, v, nst, dim));
            };
        std::vector<double> precondition;
        // Both spin branches use the same explicit transition-space view for
        // candidate selection and full-matrix eigenvector distribution.
        const LR::OrbitalTransitionSpace transitions{this->eig_ks.c, this->paraX_, this->nk,
                                                     this->nocc, this->nvirt, this->openshell};
        const bool use_gpu = this->inp_->device == "gpu";
        if (this->inp_->lr_solver != "lapack")
        {
            precondition = orbital_energy_gaps(this->eig_ks.c, this->paraX_, this->nk,
                                               this->nocc, this->nvirt, openshell);
        }
        std::function<void(const T*, T*)> preconditioner_proxy;
#ifdef __EXX
        std::unique_ptr<RIResponseOperator<T>> ri_response;
        if (this->inp_->lr_preconditioner_ri_cutoff > 0)
        {
            const auto start = std::chrono::steady_clock::now();
            MinimalGaussianRISettings settings;
            settings.coulomb_cutoff = this->inp_->lr_preconditioner_ri_cutoff;
            const double alpha = ri_exchange_settings(xc_kernel, this->exx_info, settings);
            const auto period = RI_Util::get_Born_vonKarmen_period(this->kv);
            const int mesh_size = period[0] * period[1] * period[2];
            if (mesh_size != this->nk)
            { throw std::invalid_argument("Gaussian RI proxy requires a full uniform k mesh"); }
            std::vector<int> atomic_numbers;
            for (int species = 0; species < ucell.ntype; ++species)
            {
                const auto& symbol = ucell.atoms[species].ncpp.psd;
                const auto found = std::find(ModuleBase::element_name.begin(), ModuleBase::element_name.end(), symbol);
                if (found == ModuleBase::element_name.end())
                { throw std::invalid_argument("Gaussian RI proxy: unknown pseudopotential element " + symbol); }
                const int number = found - ModuleBase::element_name.begin() + 1;
                atomic_numbers.push_back(number);
            }
            const LCAO_Orbitals& orbitals = this->ks_ ? this->ks_->orb_ : this->orb_;
            const auto comm = this->paraC_.comm();
            const int rank = RI::MPI_Wrapper::mpi_get_rank(comm);
            const int ranks = RI::MPI_Wrapper::mpi_get_size(comm);
            const hsolver::diag_comm_info comm_info(comm, rank, ranks);
            const auto fits = build_minimal_gaussian_ri<T>(comm, ucell, orbitals, atomic_numbers, period, settings);
            std::unique_ptr<RIHartreeFock<T>> kernel(new RIHartreeFock<T>(comm, fits.geometry, fits.hartree, fits.exchange));
            const auto basis = ri_transition_basis(*this->psi_ks, this->paraC_, transitions, this->kv, this->nbasis);
            ri_response.reset(new RIResponseOperator<T>(fits.geometry, basis, std::move(kernel), precondition, alpha, comm_info));
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "LR RI proxy setup: seconds=" << seconds << " theta=" << settings.theta
                      << " J/K lmax=" << settings.hartree_lmax << '/' << settings.exchange_lmax
                      << " Coulomb cutoff=" << settings.coulomb_cutoff << " bohr exchange fraction=" << alpha
                      << " omega=" << settings.exchange_omega << " bohr^-1\n";
        }
#endif
        // allocate and initialize A matrix and density matrix
        if (openshell)
        {
            std::cout << "Solving spin-conserving excitation for open-shell system." << std::endl;
            HamiltULR<T> hulr(xc_kernel,
                              nspin,
                              this->nbasis,
                              this->nocc,
                              this->nvirt,
                              *this->ucell_,
                              orb_cutoff_,
                              this->gd(),
                              *this->psi_ks,
                              this->eig_ks,
#ifdef __EXX
                              this->exx_lri,
                              this->exx_info.info_global.hybrid_alpha,
#endif
                              this->pot,
                              this->kv,
                              this->paraX_,
                              this->paraC_,
                              this->paraMat_);
#ifdef __EXX
            if (ri_response)
            {
                preconditioner_proxy = [&](const T* x, T* ax) { ri_response->apply(x, ax, RIResponseSpin::unrestricted); };
            }
#endif
            LR::HSolver::solve(hulr, this->X[0].template data<T>(), nloc_per_state, nstates,
                               transitions,
                               this->pelec->ekb.c, this->inp_->lr_solver,
                               this->inp_->lr_thr, precondition, this->inp_->lr_preconditioner_block_size, preconditioner_proxy, use_gpu, true);
            if (this->inp_->out_wfc_lr) { write_states("openshell", this->pelec->ekb.c, this->X[0].template data<T>(), nloc_per_state, nstates); }
        }
        else
        {
            const std::vector<std::string>& spin_types = this->spin_types;
            for (int is = 0;is < nspin;++is)
            {
                std::cout << " Calculating " << spin_types[is] << " excitations" << std::endl;
                HamiltLR<T> hlr(xc_kernel,
                                nspin,
                                this->nbasis,
                                this->nocc,
                                this->nvirt,
                                *this->ucell_,
                                orb_cutoff_,
                                this->gd(),
                                *this->psi_ks,
                                this->eig_ks,
#ifdef __EXX
                                this->exx_lri,
                                this->exx_info.info_global.hybrid_alpha,
#endif
                                this->pot[is],
                                this->kv,
                                this->paraX_,
                                this->paraC_,
                                this->paraMat_,
                                spin_types[is],
                                this->in_dir,
                                this->out_dir,
                                this->inp_->ri_hartree_benchmark,
                                (this->inp_->ri_hartree_benchmark == "aims" ? this->inp_->aims_nbasis : std::vector<int>({})));
#ifdef __EXX
                if (ri_response)
                {
                    const RIResponseSpin spin = is == 0 ? RIResponseSpin::singlet : RIResponseSpin::triplet;
                    preconditioner_proxy = [&, spin](const T* x, T* ax) { ri_response->apply(x, ax, spin); };
                }
#endif
                LR::HSolver::solve(hlr, this->X[is].template data<T>(), nloc_per_state, nstates,
                                   transitions,
                                   this->pelec->ekb.c + is * nstates,
                                   this->inp_->lr_solver,
                                   this->inp_->lr_thr,
                                   precondition, this->inp_->lr_preconditioner_block_size, preconditioner_proxy, use_gpu, true);
                if (this->inp_->out_wfc_lr) { write_states(spin_types[is], this->pelec->ekb.c + is * nstates, this->X[is].template data<T>(), nloc_per_state, nstates); }
            }
        }
    }
    else    // lr_solver == "spectrum", read the eigenvalues
    {
        auto efile_in = [&](const std::string& label)->std::string {return this->in_dir + "Excitation_Energy_" + label + ".dat";};
        auto vfile_in = [&](const std::string& label)->std::string {return this->in_dir + "Excitation_Amplitude_" + label + "_" + std::to_string(this->my_rank_+1) + ".dat";};
    
        auto read_states = [&](const std::string& label, Real<T>* e, T* v, const int& dim, const int& nst)->void
            {
                if (this->my_rank_ == 0) {
                    assert(nst == LR_Util::read_value(efile_in(label), e, nst));
                    std::cout <<"Rank "<< this->my_rank_ << ": finish reading " << efile_in(label) << std::endl;
                }
#ifdef __MPI
// in velocity gauge, the eigenvalues may be used to calculate the transition dipole, so we'd better broadcast them
                MPI_Bcast(e, nst, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
                assert(nst * dim == LR_Util::read_value(vfile_in(label), v, nst, dim));
                std::cout <<"Rank "<< this->my_rank_ << ": finish reading " << vfile_in(label) << std::endl;
            };
        std::cout << "reading the excitation states from file: \n";
        if (openshell)
        {
            read_states("openshell", this->pelec->ekb.c, this->X[0].template data<T>(), nloc_per_state, nstates);
        }
        else
        {
            const std::vector<std::string>& spin_types = this->spin_types;
            for (int is = 0;is < nspin;++is) { read_states(spin_types[is], this->pelec->ekb.c + is * nstates, this->X[is].template data<T>(), nloc_per_state, nstates); }
        }
    }
    if (this->excited_relax_)
    {
        // Re-select which root to follow BEFORE taking the gradient, so the force belongs to
        // the same diabatic state as the previous step's.
        this->follow_target_state_(this->ofs_running_);

        // The LR terms only carry the Omega part of the force; the ground-state part is separate
        // and comes straight from the KS solver.
        this->ks_->cal_force(ucell, this->force_gs_);
        this->lr_force_ = this->cal_lr_force_relax_(this->ofs_running_);

        // One line per ionic step with the two halves of the energy and of the gradient. Without
        // it the relaxation only reports a force, and whether E_gs + Omega actually goes down --
        // the thing being minimised -- cannot be read off the log at all.
        const double omega = this->target_omega_();
        auto max_abs = [](const ModuleBase::matrix& m) -> double
            { double v = 0.0; for (int i = 0;i < m.nr * m.nc;++i) { v = std::max(v, std::abs(m.c[i])); } return v; };
        this->ofs_running_ << std::setprecision(8) << std::fixed
            << " EXCITED-STATE RELAX step " << istep
            << ": E_gs = " << this->etot_gs_ * ModuleBase::Ry_to_eV
            << " eV, Omega = " << omega * ModuleBase::Ry_to_eV
            << " eV, E_exc = " << (this->etot_gs_ + omega) * ModuleBase::Ry_to_eV << " eV"
            << std::setprecision(6)
            << " | |F_gs|max = " << max_abs(this->force_gs_) * ModuleBase::Ry_to_eV / ModuleBase::BOHR_TO_A
            << ", |F_Omega|max = " << max_abs(this->lr_force_) * ModuleBase::Ry_to_eV / ModuleBase::BOHR_TO_A
            << " eV/Angstrom" << std::defaultfloat << std::endl;
    }

    ModuleBase::timer::end("ESolver_LR", "runner");
    return;
}

template <typename T, typename TR>
void ModuleESolver::ESolver_LR<T, TR>::after_all_runners(BaseCell& basecell)
{
    basecell.require_kind(BaseCell::Kind::unitcell, __FUNCTION__);
    UnitCell& ucell = static_cast<UnitCell&>(basecell);

    ModuleBase::TITLE("ESolver_LR", "after_all_runners");
    if (this->inp_->ri_hartree_benchmark != "none") { return; } //no need to calculate the spectrum in the benchmark routine

    // cal electron-hole density
    if (this->inp_->out_chg[0])
    {
        LR_Density<T> lr_density(*this->ucell_, kv, gd(), *psi_ks, orb_cutoff_, pgrid(),
            nspin, nocc, nvirt, nbasis,
            paraX_, paraC_, paraMat_, openshell);

        if (openshell)
        {
            for (int is = 0;is < this->nspin;++is)
            {
                const T* const amplitudes = this->X[0].template data<T>();
                lr_density.output_eh_density_all_states(amplitudes, is, nstates, this->ofs_running_);
            }
        }
        else
        {
            for (int is = 0;is < this->X.size();++is)
            {
                const T* const amplitudes = this->X[is].template data<T>();
                lr_density.output_eh_density_all_states(amplitudes, is, nstates, this->ofs_running_);
            }
        }
    }

    //cal spectrum
    if (LR_Util::tolower(this->inp_->abs_gauge) == "velocity" )
    {
        const int nspin_tmp = this->inp_->nspin == 2 ? 2 : 1;
        this->velocity_mo = LR_Util::cal_velocity_mo(*this->ucell_, this->gd(), this->tcb(), 
                                                    this->paraMat_, this->paraC_, this->kv, *this->psi_ks, 
                                                    this->nk, nspin_tmp, this->nbasis, this->nocc, this->nvirt);
    }
    std::vector<double> freq(100);
    std::vector<double> abs_wavelen_range({ 20, 200 });//default range
    if (this->inp_->abs_wavelen_range.size() >= 2 && std::abs(this->inp_->abs_wavelen_range[1] - this->inp_->abs_wavelen_range[0]) > 0.02)
    {
        abs_wavelen_range = this->inp_->abs_wavelen_range;
    }
    double lambda_diff = std::abs(abs_wavelen_range[1] - abs_wavelen_range[0]);
    double lambda_min = std::min(abs_wavelen_range[1], abs_wavelen_range[0]);
    for (int i = 0;i < freq.size();++i) { freq[i] = 91.126664 / (lambda_min + 0.01 * static_cast<double>(i + 1) * lambda_diff); }
    // auto spin_types = (nspin == 2 && !openshell) ? std::vector<std::string>({ "singlet", "triplet" }) : std::vector<std::string>({ "updown" });
    // for (int is = 0;is < this->X.size() - 1;++is)
    for (int is = 0;is < this->X.size();++is)
    {
        LR_Spectrum<T> spectrum(nspin, this->nbasis, this->nocc, this->nvirt, *this->pw_rho, *this->psi_ks,
            *this->ucell_, this->kv, this->gd(), this->orb_cutoff_, this->tcb(),
            this->paraX_, this->paraC_, this->paraMat_,
            &this->pelec->ekb.c[is * nstates], this->eig_ks.c, this->X[is].template data<T>(), nstates, openshell,
            LR_Util::tolower(this->inp_->abs_gauge), this->my_rank_, this->out_dir);
        if (LR_Util::tolower(this->inp_->abs_gauge) == "velocity" ) {spectrum.set_vmo(this->velocity_mo.data());}
        spectrum.cal_spectrum();
        spectrum.transition_analysis(spin_types[is]+"_tda");
        if (spin_types[is] != "triplet")        // triplets has no transition dipole and no contribution to the spectrum
        {
            spectrum.optical_absorption_method1(freq, this->inp_->abs_broadening);
            spectrum.write_transition_dipole(this->out_dir +
                "trans_dipole_" + spin_types[is] + "_tda.dat");
            // =============================================== for test ====================================================
            // spectrum.optical_absorption_method2(freq, this->inp_->abs_broadening);
            // if (LR_Util::tolower(this->inp_->abs_gauge) == "velocity")
            // {   // TEST the formula v/omega rather than v/(e_a-e_i)
            //     spectrum.test_transition_dipoles_velocity_omega();
            //     spectrum.write_transition_dipole(this->out_dir +
            //         "trans_dipole_" + spin_types[is] + "_vomega_tda.dat");
            // }
            // =============================================== for test ====================================================
        }
        if (this->inp_->cal_force && !this->excited_relax_) { this->cal_force_and_grad_matrix_(is, this->ofs_running_); }
    }
}
template class ModuleESolver::ESolver_LR<double, double>;
template class ModuleESolver::ESolver_LR<std::complex<double>, double>;
