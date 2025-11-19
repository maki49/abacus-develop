#include "lr_force.h"
#include "cal_hs_grad.h"
#include "pulay_force_hcontainer.h"
#include "module_hamilt_lcao/hamilt_lcaodft/pulay_force_stress.h"   // only for gint terms
#include "module_lr/utils/lr_util.h"
// #include "module_lr/utils/lr_util_hcontainer.h"
namespace LR
{
    template<typename TK>
    Charge LR_Force<TK>::dm_to_charge(const elecstate::DensityMatrix<TK, double>& dm)
    {
        const int& nspin_dm = dm.get_DMR_vector().size();
        const int& nspin_global = PARAM.inp.nspin;
        Charge chr;
        chr.set_rhopw(const_cast<ModulePW::PW_Basis*>(&this->rhopw_));
        chr.allocate(nspin_global); //chr still needs global nspin, because Forces (PW) depends on it
        // So huge a Charge class...
        // 1. Using a (private) `allocate_rho` to control whether to delete will definately cause memory leak here. No need for such judgement. 
        // 2. Charge-dependent interfaces need refactor: only rhopw_ and rho dependence are enough.

        this->gint_->transfer_DM2DtoGrid(dm.get_DMR_vector());     // 2d block to grid
        Gint_inout inout_rho(chr.rho, Gint_Tools::job_type::rho, nspin_dm, false);
        this->gint_->cal_gint(&inout_rho);
        // if (nspin_dm == 1 && nspin_global == 2), chr.rho[1][irxx]=0 has been set in Charge::allocate()
        return chr;
    }

    template<typename TK>
    elecstate::Potential LR_Force<TK>::dm_to_hxc_potential(const elecstate::DensityMatrix<TK, double>& dm)
    {
        double etxc = 0.0, vtxc = 0.0;
        elecstate::Potential pot(&this->rhodpw_, &this->rhopw_, &this->ucell_,
            &this->locpp_.vloc, const_cast<Structure_Factor*>(&this->sf_),
            nullptr/*surchem*/, &etxc, &vtxc);
        PARAM.inp.vh_in_h ? pot.pot_register({ "hartree", "xc" }) : pot.pot_register({ "xc" });
        const Charge& charge = this->dm_to_charge(dm);
        pot.init_pot(0, &charge); // call update_from_charge inside
        return pot;
    }

    template<typename TK>
    elecstate::Potential LR_Force<TK>::local_potential()
    {
        elecstate::Potential pot(&this->rhodpw_, &this->rhopw_, &this->ucell_,
            &this->locpp_.vloc, const_cast<Structure_Factor*>(&this->sf_),
            nullptr/*surchem*/, nullptr/*etxc*/, nullptr/*vtxc*/);
        pot.pot_register({ "local" });
        pot.init_pot(0, nullptr);
        return pot;
    }

    template<typename TK>
    ModuleBase::matrix LR_Force<TK>::cal_force_hamilt_gs_dm_relaxed_diff(const elecstate::DensityMatrix<TK, double>& relax_diff_dm,
        const elecstate::DensityMatrix<TK, double>& dm_gs,
        // const elecstate::Potential& pot_gs,
        const bool with_ewald)
    {
        const Charge chr_diff_relaxed = dm_to_charge(relax_diff_dm);

        // 1. local pp (Hellmann-Feynman)(fvl_dvl) + ewald + core correction (+ self-consistent charge)
        ModuleBase::matrix f_pw = PARAM.inp.vl_in_h ?
            ForcePWTerms<double>()(this->ucell_, chr_diff_relaxed, this->rhopw_, this->locpp_, this->sf_, with_ewald) :
            ModuleBase::matrix(this->ucell_.nat, 3);

        // 2. nonlocal pp (Hellmann-Feynman + Pulay)
        ModuleBase::matrix fvnl = cal_force_nonlocal(this->ucell_, this->kvec_d_, this->gd_, this->two_center_bundle_, relax_diff_dm);

        // // 3. local pp (Pulay) + Hartree + xc (grid integration)
        // ModuleBase::matrix fvl_dphi(this->ucell_.nat, 3);
        // ModuleBase::matrix stress_tmp;  // no use now, only for passing into interfaces
        // PulayForceStress::cal_pulay_fs(relax_diff_dm.get_DMR_vector().size()/*nspin*/, fvl_dphi, stress_tmp,
        //     relax_diff_dm, this->ucell_, &pot_gs, *this->gint_, true, false);

        // 3.1. local pp (Pulay)
        ModuleBase::matrix fvl_dphi(this->ucell_.nat, 3);
        ModuleBase::matrix stress_tmp;  // no use now, only for passing into interfaces
        elecstate::Potential pot_loc = this->local_potential();
        PulayForceStress::cal_pulay_fs(relax_diff_dm.get_DMR_vector().size()/*nspin*/, fvl_dphi, stress_tmp,
            relax_diff_dm, this->ucell_, &pot_loc, *this->gint_, true, false);

        // 3.2. Hartree + xc (Pulay) 
        //  method 1
        // ModuleBase::matrix fgs_dphi(this->ucell_.nat, 3);
        // PulayForceStress::cal_pulay_fs(relax_diff_dm.get_DMR_vector().size()/*nspin*/, fgs_dphi, stress_tmp,
        //     relax_diff_dm, this->ucell_, &pot_gs, *this->gint_, true, false);
        // ModuleBase::matrix fhxc_dphi = (fgs_dphi - fvl_dphi) * 0.5; // avoid double count of hxc Pulay term
        // method 2 
        ModuleBase::matrix fhxc_dphi(this->ucell_.nat, 3);
        this->gint_->reset_DMRGint(dm_gs.get_DMR_vector().size());
        elecstate::Potential pot_hxc = this->dm_to_hxc_potential(dm_gs);
        this->gint_->reset_DMRGint(relax_diff_dm.get_DMR_vector().size());
        PulayForceStress::cal_pulay_fs(relax_diff_dm.get_DMR_vector().size()/*nspin*/, fhxc_dphi, stress_tmp,
            relax_diff_dm, this->ucell_, &pot_hxc, *this->gint_, true, false);
        fhxc_dphi *= 0.5; // avoid double count

        // 3.3 Hartree + xc (Hellmann-Feynman)
        ModuleBase::matrix fhxc_dvhxc(this->ucell_.nat, 3);
        elecstate::Potential pot_hxc_relaxed_diff = this->dm_to_hxc_potential(relax_diff_dm);
        PulayForceStress::cal_pulay_fs(1/*nspin*/, fhxc_dvhxc, stress_tmp,
            dm_gs, this->ucell_, &pot_hxc_relaxed_diff, *this->gint_, true, false);
        // fhxc_dvhxc *= 0.5; // avoid double count, but nspin=2 of ground-state dm cancels it here 

        // 4. kinetic (Pulay)
        std::vector<hamilt::HContainer<double>> dT = cal_hs_grad('T', this->ucell_, this->pv_, this->gd_, this->two_center_bundle_);
        ModuleBase::matrix ft_dphi = PulayForceStress::cal_pulay_fs(relax_diff_dm, this->ucell_, dT);

        if (PARAM.inp.test_force)
        {
            ModuleIO::print_force(GlobalV::ofs_running, this->ucell_, "PW      FORCE (eV/Angstrom)", f_pw, false);
            ModuleIO::print_force(GlobalV::ofs_running, this->ucell_, "NONLOCAL     FORCE (eV/Angstrom)", fvnl, false);
            ModuleIO::print_force(GlobalV::ofs_running, this->ucell_, "KINETIC     FORCE (eV/Angstrom)", ft_dphi, false);
            ModuleIO::print_force(GlobalV::ofs_running, this->ucell_, "LOCAL-PP Pulay FORCE (eV/Angstrom)", fvl_dphi, false);
            ModuleIO::print_force(GlobalV::ofs_running, this->ucell_, "HARTREE+XC Pulay FORCE (eV/Angstrom)", fhxc_dphi, false);
            ModuleIO::print_force(GlobalV::ofs_running, this->ucell_, "HARTREE+XC Hellmann-Feynman FORCE (eV/Angstrom)", fhxc_dvhxc, false);
        }

        // from the formula, we do not need the non-ortho term (overlap*edm) here.
        return f_pw + fvnl + ft_dphi + fvl_dphi + fhxc_dphi + fhxc_dvhxc;
    }
    template<typename TK>
    ModuleBase::matrix LR_Force<TK>::cal_force_hxc_dmtrans(const elecstate::DensityMatrix<TK, double>& dm_trans, const PotHxcLR& pot_hxc)
    {
        return PulayForceStress::cal_pulay_fs(dm_trans, this->ucell_, &pot_hxc, *this->gint_);
    }

#ifdef __EXX
    template<typename TK>
    ModuleBase::matrix LR_Force<TK>::cal_force_exx_dm_trans(
        const std::map<int, std::map<TAC, RI::Tensor<TK>>>& dm_trans,
        const double& alpha,
        const std::string& spin_suffix)
    {
        ModuleBase::matrix f_exx_dmtrans(this->ucell_.nat, 3);
        auto& exx_lri_kernel = this->exx_lri_.lock()->get();
        exx_lri_kernel.set_Ds(dm_trans, this->exx_lri_.lock()->get_info().dm_threshold, spin_suffix);
        exx_lri_kernel.cal_Hs({ "", "", spin_suffix });
        exx_lri_kernel.cal_force({ "", "", spin_suffix, "", "" });// using dm_trans
        for (std::size_t idim = 0; idim < 3; ++idim)
            for (const auto& force_item : exx_lri_kernel.force[idim])
                f_exx_dmtrans(force_item.first, idim) = std::real(force_item.second);
        const double fac = -2.0 * alpha; //-2 is the same as post_process_Hexx (which didn't act on Hs)
        return f_exx_dmtrans * fac;
    }

    template<typename TK>
    ModuleBase::matrix LR_Force<TK>::cal_force_exx_gs_dm_relaxed_diff(
        const std::map<int, std::map<TAC, RI::Tensor<TK>>>& dm_gs,
        const std::map<int, std::map<TAC, RI::Tensor<TK>>>& relaxed_diff_dm,
        const double& alpha,
        const std::string& spin_suffix)
    {
        ModuleBase::matrix f_exx_gs_diff(this->ucell_.nat, 3);
        auto& exx_lri_kernel = this->exx_lri_.lock()->get();
        exx_lri_kernel.set_Ds(dm_gs, this->exx_lri_.lock()->get_info().dm_threshold, spin_suffix);
        exx_lri_kernel.cal_Hs({ "", "", spin_suffix });  // using dm_gs
        // // test: print Hs here
        // LR_Util::print_CV(exx_lri_kernel.Hs, "EXX Hs from gs dm in cal_force_exx_gs_dm_relaxed_diff");
        // auto* lr_ptr = dynamic_cast<RI::LR<int, int, 3, TK>*>(&exx_lri_kernel);  // wrong: 
        // assert(lr_ptr != nullptr);
        RI::LR<int, int, 3, TK> lr_exx_kernel(std::move(exx_lri_kernel));
        lr_exx_kernel.cal_force(relaxed_diff_dm, { "", "", spin_suffix , "", "" }); // using relaxed_diff_dm
        exx_lri_kernel = std::move(lr_exx_kernel);        // move back 
        for (std::size_t idim = 0; idim < 3; ++idim)
            for (const auto& force_item : exx_lri_kernel.force[idim])
                f_exx_gs_diff(force_item.first, idim) = std::real(force_item.second);
        const double fac = -2.0 * alpha; //-2 is the same as post_process_Hexx (which didn't act on Hs)
        return f_exx_gs_diff * fac;
    }
#endif
}

template class LR::LR_Force<double>;
template class LR::LR_Force<std::complex<double>>;