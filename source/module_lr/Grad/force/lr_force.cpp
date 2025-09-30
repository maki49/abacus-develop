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

#ifdef __EXX
    template<typename TK>
    ModuleBase::matrix LR_Force<TK>::cal_force_exx_dm_trans(
        const std::map<int, std::map<TAC, RI::Tensor<TK>>>& dm_trans,
        const double& alpha)
    {
        ModuleBase::matrix f_exx_dmtrans(this->ucell_.nat, 3);
        auto& exx_lri_kernel = this->exx_lri_.lock()->get();
        exx_lri_kernel.set_Ds(dm_trans, this->exx_lri_.lock()->get_info().dm_threshold, "0");
        exx_lri_kernel.cal_Hs();
        exx_lri_kernel.cal_force();// using dm_trans
        for (std::size_t idim = 0; idim < 3; ++idim)
            for (const auto& force_item : exx_lri_kernel.force[idim])
                f_exx_dmtrans(force_item.first, idim) = std::real(force_item.second);
        return f_exx_dmtrans * alpha;
    }

    template<typename TK>
    ModuleBase::matrix LR_Force<TK>::cal_force_exx_gs_dm_relaxed_diff(
        const std::map<int, std::map<TAC, RI::Tensor<TK>>>& dm_gs,
        const std::map<int, std::map<TAC, RI::Tensor<TK>>>& relaxed_diff_dm,
        const double& alpha)
    {
        ModuleBase::matrix f_exx_gs_diff(this->ucell_.nat, 3);
        auto& exx_lri_kernel = this->exx_lri_.lock()->get();
        exx_lri_kernel.set_Ds(dm_gs, this->exx_lri_.lock()->get_info().dm_threshold, "0");
        exx_lri_kernel.cal_Hs();  // using dm_gs
        // auto* lr_ptr = dynamic_cast<RI::LR<int, int, 3, TK>*>(&exx_lri_kernel);  // wrong: 
        // assert(lr_ptr != nullptr);
        RI::LR<int, int, 3, TK> lr_exx_kernel(std::move(exx_lri_kernel));
        std::cout << "post_2D adress moved: " << &(lr_exx_kernel.post_2D) << std::endl;
        std::cout << "begin RI::LR::cal_force" << std::endl;
        lr_exx_kernel.cal_force(relaxed_diff_dm); // using relaxed_diff_dm
        exx_lri_kernel = std::move(lr_exx_kernel);        // move back 
        std::cout << "post_2D adress after move back: " << &(exx_lri_kernel.post_2D) << std::endl;
        std::cout << "end RI::LR::cal_force" << std::endl;
        for (std::size_t idim = 0; idim < 3; ++idim)
            for (const auto& force_item : exx_lri_kernel.force[idim])
                f_exx_gs_diff(force_item.first, idim) = std::real(force_item.second);
        return f_exx_gs_diff * alpha;
    }
#endif
}

template class LR::LR_Force<double>;
template class LR::LR_Force<std::complex<double>>;