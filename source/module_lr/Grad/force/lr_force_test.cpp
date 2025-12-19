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
    ModuleBase::matrix LR_Force<TK>::reproduce_force_gs(const K_Vectors& kv,
        const elecstate::DensityMatrix<TK, double>& dm_gs,
        const elecstate::DensityMatrix<TK, double>& edm_gs)
    {
        const int& nspin = PARAM.inp.nspin;
        this->gint_->reset_DMRGint(nspin);
        // local + Hartree + xc term, including Hellmann-Feynman and Pulay
        ModuleBase::matrix f_gs_hf_pulay = cal_force_hamilt_gs_dm_relaxed_diff(dm_gs, dm_gs); // pw+vnl+t_dphi+vl_dphi
        // edm term
        ModuleBase::matrix f_nonortho = cal_force_overlap_edm(edm_gs); // overlap
        this->gint_->reset_DMRGint(1);
#ifdef __EXX
        const std::set<std::string> exx_kernel_list = { "hf", "hse" };
        if (exx_kernel_list.count(PARAM.inp.dft_functional))
        {
            const auto& Ds_gs = LR_Util::get_exx_Ds_gs(dm_gs, ucell_, kv, pv_);
            const auto& Ds_gs_2 = LR_Util::get_exx_Ds_gs(dm_gs, ucell_, kv, pv_);
            ModuleBase::matrix f_gs_exx(ucell_.nat, 3);
            // test the two function using the two spin channels respectively
            f_gs_exx += cal_force_exx_gs_dm_relaxed_diff(Ds_gs.at(0), Ds_gs_2.at(0), alpha_, std::to_string(0));    // test passed
            f_gs_exx += cal_force_exx_dm_trans(Ds_gs.at(1), alpha_, std::to_string(1));
            if (PARAM.inp.test_force)
                ModuleIO::print_force(GlobalV::ofs_running, ucell_, "EXX GS FORCE reproduce (eV/Angstrom)", f_gs_exx, false);
            f_gs_hf_pulay += f_gs_exx;
        }
#endif
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
    void LR_Force<TK>::cal_H2_sz_center4(const std::vector<double>& orb_cutoffs,
        const K_Vectors& kv, const bool is_grad)
    {
        const std::string label = is_grad ? "(d_x(i) j | kl)" : "(ij | kl)";;
        GlobalV::ofs_running << "  ==== Test H2_SZ_CENTER4_HXC " << label << " ====" << std::endl;
        const std::vector<ModuleBase::Vector3<double>>& kvd_test = { ModuleBase::Vector3<double>(0.0, 0.0, 0.0) };
        auto init_dm_eff = [&, this](const int i, const int j) -> elecstate::DensityMatrix<TK, double>
            {   // dm_{ij}=1, other elements = 0, i,j = 0,1
                std::vector<TK> dm_2d(4, 0.0);
                std::cout<<"i<<1 + j =" << ((i<<1) + j) << std::endl;
                dm_2d[i*2+j] = 1.0;
                elecstate::DensityMatrix<TK, double> dm(&this->pv_, 1, kvd_test, 1);
                dm.set_DMK_pointer(0, dm_2d.data());
                LR_Util::initialize_DMR(dm, this->pv_, this->ucell_, this->gd_, orb_cutoffs);
                dm.cal_DMR();
                return dm;
            };
        this->gint_->reset_DMRGint(1);
#ifdef __EXX
        std::vector<std::tuple<std::set<int>, std::set<int>>> judge = RI_2D_Comm::get_2D_judge(ucell_, pv_);
#endif

        for (auto&& i : { 0, 1 })
            for (auto&& j : { 0, 1 })
            {
                elecstate::DensityMatrix<TK, double> dm_ij = init_dm_eff(i, j);
                LR_Util::print_DMR(dm_ij, "DMR_" + std::to_string(i) + std::to_string(j));
                for (auto&& k : { 0, 1 })
                    for (auto&& l : { 0, 1 })
                    {
                        // 1. build dm(kl)
                        elecstate::DensityMatrix<TK, double> dm_kl = init_dm_eff(k, l);
                        if (is_grad)
                        {
                            // 2. build charge & potential 
                            elecstate::Potential pot_hxc_kl = dm_to_hxc_potential(dm_kl);
                            // 3. cal force
                            ModuleBase::matrix fvl_dphi(this->ucell_.nat, 3);
                            ModuleBase::matrix stress_tmp;  // dummy
                            PulayForceStress::cal_pulay_fs(1/*nspin*/, fvl_dphi, stress_tmp,
                                dm_ij, this->ucell_, &pot_hxc_kl, *this->gint_, true, false);
                            ModuleIO::print_force(GlobalV::ofs_running, this->ucell_,
                                "H2_SZ_CENTER4_HXC_(dx_i(" + std::to_string(i) + ")" + std::to_string(j) + "|" + std::to_string(k) + std::to_string(l) + ") FORCE (eV/Angstrom)",
                                fvl_dphi * (-0.5), false);   // F_Hxc = -2(dx_i(i)j| kl)
#ifdef __EXX
                            if (!this->exx_lri_.expired())
                            {   // match the Gint result with LibRI
                                auto ds_kl = LR_Util::get_exx_Ds_spin1(dm_kl, ucell_, kv, pv_); // returns ds_kl*0.5
                                auto ds_ij = LR_Util::get_exx_Ds_spin1(dm_ij, ucell_, kv, pv_); // returns ds_ij*0.5
                                ModuleBase::matrix f_exx = this->cal_force_exx_gs_dm_relaxed_diff(ds_kl, ds_ij, alpha_ * 4.0, ""); // cancel the two 0.5s in Ds
                                ModuleIO::print_force(GlobalV::ofs_running, ucell_,
                                    "H2_SZ_CENTER4_EXX_(dx_i(" + std::to_string(i) + ")" + std::to_string(k) + "|" + std::to_string(j) + std::to_string(l) + ") FORCE (eV/Angstrom)",
                                    f_exx, false);   // F_exx = (dx_i(i)k| jl)
                            }
#endif
                        }
                        else
                        {
                            //2. build charge & potential
                            elecstate::Potential pot_hxc_kl = dm_to_hxc_potential(dm_kl);
                            const Charge& charge_ij = this->dm_to_charge(dm_ij);
                            // 3. cal energy
                            double e_hxc = std::inner_product(charge_ij.rho[0],
                                charge_ij.rho[0] + this->rhopw_.nrxx,
                                pot_hxc_kl.get_effective_v(0), 0.0) * 0.5 * this->ucell_.omega / static_cast<double>(this->rhopw_.nrxx);
                            GlobalV::ofs_running << "  H2_SZ_CENTER4_COULOMB ("
                                << std::to_string(i) + std::to_string(j) + "|" + std::to_string(k) + std::to_string(l)
                                << ") by Gint: " << e_hxc * 2 << std::endl; // 2 for testing (ij|kl) instead of real Coulomb energy 0.5*(ij|kl)
#ifdef __EXX
                            if (!this->exx_lri_.expired())
                            {   // match the Gint result with LibRI
                                auto ds_kl = LR_Util::get_exx_Ds_spin1(dm_kl, ucell_, kv, pv_); // returns ds_kl*0.5
                                auto ds_ij = LR_Util::get_exx_Ds_spin1(dm_ij, ucell_, kv, pv_); // returns ds_ij*0.5
                                auto lri = this->exx_lri_.lock();
                                lri->get().set_Ds(std::move(ds_kl), lri->get_info().dm_threshold);
                                lri->get().cal_Hs();
                                lri->Hexxs[0] = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                                    lri->get_mpi_comm(), std::move(lri->get().Hs), std::get<0>(judge[0]), std::get<1>(judge[0]));
                                lri->post_process_Hexx(lri->Hexxs[0]);
                                TK e_exx = this->alpha_ * lri->get().post_2D.cal_energy(ds_ij, lri->Hexxs[0]) * 2.0; // 4 is to cancel two 0.5^2 in split_m2D_ktoR(nspin=1)`, and 0.5 for Fock energy
                                GlobalV::ofs_running << "  H2_SZ_CENTER4_COULOMB ("
                                    << std::to_string(i) + std::to_string(k) + "|" + std::to_string(j) + std::to_string(l)
                                    << ") by LibRI: " << -e_exx * 2.0 << ", where alpha = " << this->alpha_ << std::endl;  //-2 for Fock energy -> integral
                            }
#endif  
                        }

                    }
            }
    }
}



template class LR::LR_Force<double>;
template class LR::LR_Force<std::complex<double>>;