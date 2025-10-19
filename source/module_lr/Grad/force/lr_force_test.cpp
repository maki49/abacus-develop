#include "lr_force.h"
#include "cal_hs_grad.h"
#include "pulay_force_hcontainer.h"
#include "module_hamilt_lcao/hamilt_lcaodft/pulay_force_stress.h"   // only for gint terms
#include "module_lr/utils/lr_util_hcontainer.h"
namespace LR
{
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
        const elecstate::DensityMatrix<TK, double>& edm_gs)
    {
        this->gint_->reset_DMRGint(PARAM.inp.nspin);
        // local + Hartree + xc term, including Hellmann-Feynman and Pulay
        ModuleBase::matrix f_gs_hf_pulay = cal_force_hamilt_gs_dm_relaxed_diff(dm_gs, dm_gs); // pw+vnl+t_dphi+vl_dphi
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

    template<typename TK>
    void LR_Force<TK>::cal_H2_sz_center4_grad_hxc(const std::vector<double>& orb_cutoffs)
    {
        GlobalV::ofs_running << "  ==== Test H2_SZ_CENTER4_HXC (d_x(i) j | kl) ====" << std::endl;
        const std::vector<ModuleBase::Vector3<double>>& kvd_test = { ModuleBase::Vector3<double>(0.0, 0.0, 0.0) };
        auto init_dm_eff = [&, this](const int i, const int j) -> elecstate::DensityMatrix<TK, double>
            {
                std::vector<TK> dm_2d(4, 0.0);
                std::cout<<"i<<1 + j =" << ((i<<1) + j) << std::endl;
                dm_2d[i*2+j] = 1.0;
                elecstate::DensityMatrix<TK, double> dm(&this->pv_, 1, kvd_test, 1);
                dm.set_DMK_pointer(0, dm_2d.data());
                LR_Util::initialize_DMR(dm, this->pv_, this->ucell_, this->gd_, orb_cutoffs);
                dm.cal_DMR();
                return dm;
            };
        ModuleBase::matrix stress_tmp;  // dummy
        this->gint_->reset_DMRGint(1);

        for (auto&& i : { 0, 1 })
            for (auto&& j : { 0, 1 })
            {
                elecstate::DensityMatrix<TK, double> dm_ij = init_dm_eff(i, j);
                LR_Util::print_DMR(dm_ij, ucell_.nat, "DMR_" + std::to_string(i) + std::to_string(j));
                for (auto&& k : { 0, 1 })
                    for (auto&& l : { 0, 1 })
                    {
                        // 1. build dm(kl)
                        elecstate::DensityMatrix<TK, double> dm_kl = init_dm_eff(k, l);
                        // 2. build charge & potential 
                        elecstate::Potential pot_hxc_kl = dm_to_hxc_potential(dm_kl);
                        // 3. cal force
                        ModuleBase::matrix fvl_dphi(this->ucell_.nat, 3);
                        PulayForceStress::cal_pulay_fs(1/*nspin*/, fvl_dphi, stress_tmp,
                            dm_ij, this->ucell_, &pot_hxc_kl, *this->gint_, true, false);
                        ModuleIO::print_force(GlobalV::ofs_running, this->ucell_,
                            "H2_SZ_CENTER4_HXC_" + std::to_string(i) + std::to_string(j) + std::to_string(k) + std::to_string(l) + " FORCE (eV/Angstrom)",
                            fvl_dphi, false);
                    }
            }
    }
}

template class LR::LR_Force<double>;
template class LR::LR_Force<std::complex<double>>;