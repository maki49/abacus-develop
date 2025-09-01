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
    ModuleBase::matrix LR_Force<TK>::cal_force_hamilt_gs_dm_relaxed_diff(const elecstate::DensityMatrix<TK, double>& relax_diff_dm,
        const elecstate::Potential& pot_gs, const bool with_ewald)
    {
        const Charge chr_diff_relaxed = dm_to_charge(relax_diff_dm);

        // 1. local pp (Hellmann-Feynman)(fvl_dvl) + ewald + core correction (+ self-consistent charge)
        ModuleBase::matrix f_pw = PARAM.inp.vl_in_h ?
            ForcePWTerms<double>()(this->ucell_, chr_diff_relaxed, this->rhopw_, this->locpp_, this->sf_, with_ewald) :
            ModuleBase::matrix(this->ucell_.nat, 3);

        // 2. nonlocal pp (Hellmann-Feynman + Pulay)
        ModuleBase::matrix fvnl = cal_force_nonlocal(this->ucell_, this->kvec_d_, this->gd_, this->two_center_bundle_, relax_diff_dm);

        // 3. local pp (Pulay) + Hartree + xc (grid integration)
        ModuleBase::matrix fvl_dphi(this->ucell_.nat, 3);
        ModuleBase::matrix stress_tmp;  // no use now, only for passing into interfaces
        PulayForceStress::cal_pulay_fs(relax_diff_dm.get_DMR_vector().size()/*nspin*/, fvl_dphi, stress_tmp,
            relax_diff_dm, this->ucell_, &pot_gs, *this->gint_, true, false);

        // 4. kinetic (Pulay)
        std::vector<hamilt::HContainer<double>> dT = cal_hs_grad('T', this->ucell_, this->pv_, this->gd_, this->two_center_bundle_);
        ModuleBase::matrix ft_dphi = PulayForceStress::cal_pulay_fs(relax_diff_dm, this->ucell_, dT);

        if (PARAM.inp.test_force)
        {
            ModuleIO::print_force(GlobalV::ofs_running, this->ucell_, "PW      FORCE (eV/Angstrom)", f_pw, false);
            ModuleIO::print_force(GlobalV::ofs_running, this->ucell_, "NONLOCAL     FORCE (eV/Angstrom)", fvnl, false);
            ModuleIO::print_force(GlobalV::ofs_running, this->ucell_, "KINETIC     FORCE (eV/Angstrom)", ft_dphi, false);
            ModuleIO::print_force(GlobalV::ofs_running, this->ucell_, "LOCAL Pulay FORCE (pp+Hartree+XC) (eV/Angstrom)", fvl_dphi, false);
        }

        // from the formula, we do not need the non-ortho term (overlap*edm) here.
        return f_pw + fvnl + ft_dphi + fvl_dphi;
    }
    template<typename TK>
    ModuleBase::matrix LR_Force<TK>::cal_force_hxc_dmtrans(const elecstate::DensityMatrix<TK, double>& dm_trans, const PotHxcLR& pot_hxc)
    {
        return PulayForceStress::cal_pulay_fs(dm_trans, this->ucell_, &pot_hxc, *this->gint_);
    }

    template<typename TK>
    ModuleBase::matrix LR_Force<TK>::cal_force_overlap_edm(const elecstate::DensityMatrix<TK, double>& edm)
    {
        // const double* dS[3] = { dSloc_x,  dSloc_y,  dSloc_z };
        std::vector<hamilt::HContainer<double>>  dS = cal_hs_grad('S', this->ucell_, this->pv_, this->gd_, this->two_center_bundle_);
        // test: output dS
        // std::cout << "dS in 3 directions:\n";
        // for (int i = 0;i < 3;++i) { LR_Util::print_HR(dS.at(i), this->ucell_.nat, "dS" + std::to_string(i)); }
        ModuleBase::matrix foverlap = PulayForceStress::cal_pulay_fs(edm, this->ucell_, dS, -1.);
        if (PARAM.inp.test_force)
        {
            ModuleIO::print_force(GlobalV::ofs_running, this->ucell_, "OVERLAP     FORCE (eV/Angstrom)", foverlap, false);
        }
        return foverlap;
    }

    template<typename TK>
    ModuleBase::matrix LR_Force<TK>::reproduce_force_gs(
        const elecstate::DensityMatrix<TK, double>& dm_gs,
        const elecstate::DensityMatrix<TK, double>& edm_gs,
        const elecstate::Potential& pot_gs)
    {
        this->gint_->reset_DMRGint(PARAM.inp.nspin);
        // Hartree+xc term
        ModuleBase::matrix f_gs_hf_pulay = cal_force_hamilt_gs_dm_relaxed_diff(dm_gs, pot_gs); // pw+vnl+t_dphi+vl_dphi
        // edm term
        ModuleBase::matrix f_nonortho = cal_force_overlap_edm(edm_gs); // overlap
        this->gint_->reset_DMRGint(1);
        return f_gs_hf_pulay + f_nonortho;
    }

    template<typename TK>
    ModuleBase::matrix LR_Force<TK>::reproduce_force_gs_loc(
        const elecstate::DensityMatrix<TK, double>& dm_gs,
        const elecstate::Potential& pot_gs)
    {
        this->gint_->reset_DMRGint(PARAM.inp.nspin);
        const Charge chr_gs = dm_to_charge(dm_gs);
        //  local pp (Pulay) + Hartree + xc (grid integration)
        ModuleBase::matrix fvl_dphi(this->ucell_.nat, 3);
        ModuleBase::matrix stress_tmp;  // no use now, only for passing into interfaces
        PulayForceStress::cal_pulay_fs(dm_gs.get_DMR_vector().size()/*nspin*/, fvl_dphi, stress_tmp,
            dm_gs, this->ucell_, &pot_gs, *this->gint_, true, false);
        this->gint_->reset_DMRGint(1);
        return fvl_dphi;
    }
}

template class LR::LR_Force<double>;
template class LR::LR_Force<std::complex<double>>;