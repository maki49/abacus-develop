#include "Exx_LRI_interface.h"
#include "module_ri/exx_abfs-jle.h"
#include "module_ri/exx_opt_orb.h"
#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/op_exx_lcao.h"
#include "module_ri/exx_symmetry.h"
#include "module_elecstate/cal_dm.h"
#include "module_base/libm/libm.h"

template<typename Tdata>
void Exx_LRI_Interface<Tdata>::write_Hexxs(const std::string &file_name) const
{
	ModuleBase::TITLE("Exx_LRI","write_Hexxs");
	ModuleBase::timer::tick("Exx_LRI", "write_Hexxs");
	std::ofstream ofs(file_name, std::ofstream::binary);
	cereal::BinaryOutputArchive oar(ofs);
	oar(exx_lri->Hexxs);
	ModuleBase::timer::tick("Exx_LRI", "write_Hexxs");
}

template<typename Tdata>
void Exx_LRI_Interface<Tdata>::read_Hexxs(const std::string &file_name)
{
	ModuleBase::TITLE("Exx_LRI","read_Hexxs");
	ModuleBase::timer::tick("Exx_LRI", "read_Hexxs");
	std::ifstream ifs(file_name, std::ofstream::binary);
	cereal::BinaryInputArchive iar(ifs);
	iar(exx_lri->Hexxs);
	ModuleBase::timer::tick("Exx_LRI", "read_Hexxs");
}
template<typename Tdata>
void Exx_LRI_Interface<Tdata>::exx_beforescf(const K_Vectors& kv, const Charge_Mixing& chgmix)
{
#ifdef __MPI
		if ( GlobalC::exx_info.info_global.cal_exx )
		{
            if (GlobalC::ucell.atoms[0].ncpp.xc_func == "HSE" || GlobalC::ucell.atoms[0].ncpp.xc_func == "PBE0")
            {
                XC_Functional::set_xc_type("pbe");
            }
            else if (GlobalC::ucell.atoms[0].ncpp.xc_func == "SCAN0")
            {
                XC_Functional::set_xc_type("scan");
            }

			exx_lri->cal_exx_ions();
		}

		if (Exx_Abfs::Jle::generate_matrix)
		{
			//program should be stopped after this judgement
			Exx_Opt_Orb exx_opt_orb;
			exx_opt_orb.generate_matrix(kv);
			ModuleBase::timer::tick("ESolver_KS_LCAO", "beforescf");
			return;
		}
		
		// set initial parameter for mix_DMk_2D
		if(GlobalC::exx_info.info_global.cal_exx)
        {
            if (!GlobalV::GAMMA_ONLY_LOCAL && ModuleSymmetry::Symmetry::symm_flag == 1)
                exx_lri->mix_DMk_2D.set_nks(kv.nkstot_full, GlobalV::GAMMA_ONLY_LOCAL);
            else
                exx_lri->mix_DMk_2D.set_nks(kv.nks, GlobalV::GAMMA_ONLY_LOCAL);
			if(GlobalC::exx_info.info_global.separate_loop)
			{
				if(GlobalC::exx_info.info_global.mixing_beta_for_loop1==1.0)
					exx_lri->mix_DMk_2D.set_mixing_mode(Mixing_Mode::No);
				else
					exx_lri->mix_DMk_2D.set_mixing_mode(Mixing_Mode::Plain)
					                .set_mixing_beta(GlobalC::exx_info.info_global.mixing_beta_for_loop1);
			}
			else
			{
				if(chgmix.get_mixing_mode() == "plain")
					exx_lri->mix_DMk_2D.set_mixing_mode(Mixing_Mode::Plain);
				else if(chgmix.get_mixing_mode() == "pulay")
					exx_lri->mix_DMk_2D.set_mixing_mode(Mixing_Mode::Pulay);
				else
					throw std::invalid_argument(
						"mixing_mode = " + chgmix.get_mixing_mode() + ", mix_DMk_2D unsupported.\n"
						+ std::string(__FILE__) + " line " + std::to_string(__LINE__));
            }
        }
        // for exx two_level scf
        exx_lri->two_level_step = 0;
#endif // __MPI
}

///adjust S'(k) acording to orbital parity of the column
inline void orb_parity_Sk(
    const Parallel_2D& p2d,
    const ModuleSymmetry::Symmetry& symm,
    const std::map<int, ModuleBase::Vector3<double>>& kstar_ibz,
    std::vector<std::vector<std::complex<double>>>& sloc_kstar,
    const UnitCell& ucell,
    const LCAO_Orbitals& orb,
    bool& col_inside)
{
    int iks = 0;
    for (auto ks : kstar_ibz)
    {
        int isym = ks.first;
        if (static_cast<int>(symm.gmatrix[isym].Det()) == -1)// inverse
        {
            // S'(k)_ij *= (-1)^{l_j}
            for (int i = 0;i < p2d.get_row_size();++i)
            {
                int gj = 0; // global index of j
                for (int it_j = 0;it_j < GlobalC::ucell.ntype;++it_j)
                {
                    for (int ia_j = 0;ia_j < GlobalC::ucell.atoms[it_j].na;++ia_j)
                    {
                        std::cout << "lmax:" << GlobalC::ORB.Phi[it_j].getLmax() << std::endl;
                        std::cout << "ia=" << ia_j << std::endl;
                        for (int l = 0;l <= GlobalC::ORB.Phi[it_j].getLmax();++l)
                        {
                            for (int n = 0;n < GlobalC::ORB.Phi[it_j].getNchi(l);++n)
                            {
                                for (int m = 0;m < 2 * l + 1;++m)
                                {
                                    std::cout << "l=" << l << ",m=" << m << ",n=" << n << std::endl;
                                    int j = p2d.global2local_col(gj);
                                    if (l % 2 == 1 && j >= 0)
                                    {
                                        std::cout << "p-orbs: global_index=" << gj << ", local_index=" << j << std::endl;
                                        if (col_inside)
                                            sloc_kstar[iks][i * p2d.get_col_size() + j] *= -1;
                                        else
                                            sloc_kstar[iks][i + j * p2d.get_row_size()] *= -1;
                                    }
                                    ++gj;
                                }// end m
                            }//end n
                        }//end l
                    }//end ia_j
                } //end it_j
            }//end i
        }//end if inverse
        ++iks;
    }//end kstar
}

inline std::vector<std::complex<double>> folding_Srotk(
    const ModuleBase::Vector3<double>& gk,
    const ModuleBase::Vector3<double>& k,
    const LCAO_Hamilt& uhm,
    const UnitCell& ucell)
{
    ModuleBase::TITLE("Exx_LRI_Interface", "folding_Srotk");
    bool row_inside = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER();
    auto kphase = [k](const double& arg) -> std::complex<double> {
        double sinp, cosp;
        ModuleBase::libm::sincos(arg, &sinp, &cosp);
        return std::complex<double>(cosp, sinp);
        };
    std::vector<std::complex<double>> sloc_rot(uhm.LM->ParaV->get_local_size(), 0);
    for (int T1 = 0; T1 < ucell.ntype; ++T1)
    {
        Atom* atom1 = &ucell.atoms[T1];
        for (int I1 = 0; I1 < atom1->na; ++I1)
        {
            const int start1 = ucell.itiaiw2iwt(T1, I1, 0);
            ModuleBase::Vector3<double> tau1 = atom1->tau[I1];
            for (int T2 = 0;T2 < ucell.ntype;++T2)
            {
                Atom* atom2 = &ucell.atoms[T2];
                double rcut = GlobalC::ORB.Phi[T1].getRcut() + GlobalC::ORB.Phi[T2].getRcut();
                for (int I2 = 0;I2 < atom2->na;++I2)
                {
                    ModuleBase::Vector3<double> tau2 = atom2->tau[I2];
                    const int start2 = ucell.itiaiw2iwt(T2, I2, 0);
                    for (auto R1_int : uhm.LM->all_R_coor)
                    {
                        for (auto R2_int : uhm.LM->all_R_coor)
                        {
                            ModuleBase::Vector3<double> R1_d(static_cast<double>(R1_int.x), static_cast<double>(R1_int.y), static_cast<double>(R1_int.z));
                            ModuleBase::Vector3<double> R2_d(static_cast<double>(R2_int.x), static_cast<double>(R2_int.y), static_cast<double>(R2_int.z));
                            auto dr_c = (R1_d - R2_d) * ucell.latvec + tau1 - tau2;
                            double distance = dr_c.norm() * ucell.lat0;
                            if (distance < rcut)
                            {
                                const double arg1 = -(gk * R1_d) * ModuleBase::TWO_PI;
                                const double arg2 = (k * R2_d) * ModuleBase::TWO_PI;
                                auto dR_int = R1_int - R2_int;
                                for (int ii = 0; ii < atom1->nw * GlobalV::NPOL; ii++)
                                {
                                    const int iw1_all = start1 + ii;
                                    const int mu = uhm.LM->ParaV->global2local_row(iw1_all);
                                    if (mu < 0)continue;
                                    for (int jj = 0; jj < atom2->nw * GlobalV::NPOL; jj++)
                                    {
                                        int iw2_all = start2 + jj;
                                        const int nu = uhm.LM->ParaV->global2local_col(iw2_all);
                                        if (nu < 0)continue;
                                        long index = row_inside ? nu * uhm.LM->ParaV->get_row_size() + mu : mu * uhm.LM->ParaV->get_col_size() + nu;
                                        sloc_rot[index] += kphase(arg1) * kphase(arg2) * uhm.LM->SR_sparse[dR_int][iw1_all][iw2_all];
                                    }
                                }
                            }
                        }
                    }//R1_int
                }
            }
        }
    }
    for (auto& item : sloc_rot) item /= static_cast<double>(uhm.LM->all_R_coor.size());
    return sloc_rot;
}
template<typename Tdata>
void Exx_LRI_Interface<Tdata>::restore_dm(
    LCAO_Hamilt& uhm,
    Local_Orbital_Charge& loc,
    const K_Vectors& kv,
    const UnitCell& ucell,
    const psi::Psi<Tdata, psi::DEVICE_CPU>& psi,
    const ModuleSymmetry::Symmetry& symm,
    const ModuleBase::matrix& wg)
{
    ModuleBase::TITLE("Exx_LRI_Interface", "restore_dm");
    bool col_inside = !ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER();
    // prepare: calculate the index of inverse matirx of each symmetry operation
    std::vector<int> invgmat(symm.nrotk);
    symm.gmatrix_invmap(symm.gmatrix, symm.nrotk, invgmat.data());


    // if symmetry-on and multi-k, restore psi for all k-points in kstars before cal_dm
    // calculate =S'^{-1}(k)S(gk) from S(gk) (once for all)
    if (uhm.LM->invSkrot_Sgk.size() == 0)
    {
        uhm.calculate_SR_sparse(0);
        // std::map<int, std::vector<double>> SrotR; // store Srot(R) for subsequent kstars' use
        for (int ikibz = 0; ikibz < kv.nkstot; ++ikibz)
        {

            std::vector<std::vector<std::complex<double>>> sloc_kstar; //Srot(k) for each k current kstar

            // first trail: calculate Srot(k) from Srot(g, R)  (wrong)
            // for (auto& isym_kvecd : kv.kstars[ikibz])
            // {
            //     int isym = isym_kvecd.first;
            //     ModuleBase::Vector3<double> kvec_d = isym_kvecd.second;
            //     auto haveisym = SrotR.find(isym);
            //     if (haveisym == SrotR.end())
            //     {   //if SrotR[isym] is not calculated, calculate it. g^{-1} should be transformed to Cartesian coordinate
            //         Gint_inout inout(ucell.latvec.Inverse() * symm.gmatrix[invgmat[0]] * ucell.latvec, Gint_Tools::job_type::srot);
            //         uhm.GK.cal_gint(&inout);
            //         // store SrotR[isym] for other kstars' use
            //         std::vector<double> SrotR_isym(uhm.GK.gridt->nnrg);
            //         for (int i = 0;i < uhm.GK.gridt->nnrg;++i) SrotR_isym[i] = uhm.GK.get_pvpR(0, i);
            //         SrotR.insert(std::make_pair(isym, SrotR_isym));
            //     }
            //     sloc_kstar.push_back(ExxSym::rearrange_col(GlobalV::NLOCAL, *loc.ParaV, ucell, col_inside, symm.isym_rotiat_iat[isym],
            //         uhm.GK.folding_Srot_k(kvec_d, *loc.ParaV, SrotR[isym].data())/*Srot(R) -> Srot(k) whose col to be rearranged*/));
            // }

            //second trail: calculate S(k) from S(gk) (wrong)
            //S(gk) (S(R) have been calculated)
            // uhm.LM->folding_fixedH(ikibz, kv.kvec_d);
            //S'(k)
            // for (auto& isym_kvecd : kv.kstars[ikibz])
            // {
            //     uhm.LM->zeros_HSk('S');
            //     // uhm.LM->folding_fixedH(isym_kvecd.second);
            //     uhm.LM->folding_fixedH(kv.kvec_d[ikibz] + kv.kvec_d[ikibz] - isym_kvecd.second);
            //     //S'(k)
            //     sloc_kstar.push_back(ExxSym::rearrange_col(GlobalV::NLOCAL, *uhm.LM->ParaV, GlobalC::ucell, col_inside, symm.isym_rotiat_iat[isym_kvecd.first],
            //         uhm.LM->Sloc2));
            // }
            // orb_parity_Sk(*loc.ParaV, symm, kv.kstars[ikibz], sloc_kstar, GlobalC::ucell, GlobalC::ORB, col_inside);


            //third trail: calculate gint of 2 envelopes
            for (auto& isym_kvecd : kv.kstars[ikibz])
            {
                std::vector<std::complex<double>> sloc_ik_grid(uhm.GK.gridt->lgd * uhm.GK.gridt->lgd, 0);
                Gint_inout inout(kv.kvec_d[ikibz], isym_kvecd.second, &sloc_ik_grid, Gint_Tools::job_type::srotk);
                uhm.GK.cal_gint(&inout);
                // output sloc_ik
                GlobalV::ofs_running << "sloc_ik_grid of isym=" << isym_kvecd.first << ", kvec=" << isym_kvecd.second.x << " " << isym_kvecd.second.y << " " << isym_kvecd.second.z << std::endl;
                for (int i = 0;i < GlobalV::NLOCAL;++i)
                {
                    for (int j = 0;j < GlobalV::NLOCAL;++j)
                    {
                        GlobalV::ofs_running << sloc_ik_grid[j * GlobalV::NLOCAL + i] << " ";
                    }
                    GlobalV::ofs_running << std::endl;
                }
                sloc_kstar.push_back(ExxSym::rearrange_col(GlobalV::NLOCAL, *loc.ParaV, GlobalC::ucell, col_inside, symm.isym_rotiat_iat[isym_kvecd.first],
                    sloc_ik_grid));
                // uhm.GK.grid_to_2d(*loc.ParaV, sloc_ik_grid)));
            // }

            // //forth trail: calculate S(g, k) using 2-center integral 
            // for (auto& isym_kvecd : kv.kstars[ikibz])
            // {
                std::vector<std::complex<double>> sloc_ik = folding_Srotk(kv.kvec_d[ikibz], isym_kvecd.second, uhm, GlobalC::ucell);
                GlobalV::ofs_running << "sloc_ik_center2 of isym=" << isym_kvecd.first << ", kvec=" << isym_kvecd.second.x << " " << isym_kvecd.second.y << " " << isym_kvecd.second.z << std::endl;
                for (int i = 0;i < GlobalV::NLOCAL;++i)
                {
                    for (int j = 0;j < GlobalV::NLOCAL;++j)
                    {
                        GlobalV::ofs_running << sloc_ik[j * GlobalV::NLOCAL + i] << " ";
                    }
                    GlobalV::ofs_running << std::endl;
                }
                // sloc_kstar.push_back(ExxSym::rearrange_col(GlobalV::NLOCAL, *loc.ParaV, GlobalC::ucell, col_inside, symm.isym_rotiat_iat[isym_kvecd.first],
                //     sloc_ik));
            }
            //calculate S'^{-1}(k)S(gk)
#ifdef __MPI
            uhm.LM->invSkrot_Sgk.push_back(ExxSym::cal_invSkrot_Sgk_scalapack(kv.nkstot, uhm.LM->Sloc2, sloc_kstar, GlobalV::NLOCAL, *uhm.LM->ParaV));
#else
            uhm.LM->invSkrot_Sgk.push_back(ExxSym::cal_invSkrot_Sgk_lapack(kv.nkstot, uhm.LM->Sloc2, sloc_kstar, GlobalV::NLOCAL));
#endif
        }//end ikibz
        uhm.destroy_all_HSR_sparse();
    }

    //c(k)=S^{-1}(k)S(gk)c(gk)
    psi::Psi<std::complex<double >, psi::DEVICE_CPU> psi_full =
        ExxSym::restore_psik(kv.nkstot_full, psi, uhm.LM->invSkrot_Sgk,
            GlobalV::NLOCAL, GlobalV::NBANDS, *uhm.LM->ParaV);
    // test: output psifull
    GlobalV::ofs_running << "restored psi_full" << std::endl;
    for (int ik = 0;ik < kv.nkstot_full;++ik)
    {
        GlobalV::ofs_running << "ik=" << ik << std::endl;
        psi_full.fix_k(ik);
        for (int ir = 0;ir < GlobalV::NLOCAL;++ir)
        {
            for (int ic = 0;ic < GlobalV::NBANDS;++ic)
            {
                GlobalV::ofs_running << psi_full(ic, ir) << " ";
            }
            GlobalV::ofs_running << std::endl;
        }
    }
    // set wg_full
    ModuleBase::matrix wg_full(kv.nkstot_full, wg.nc);
    int ik_full = 0;
    for (int ikibz = 0; ikibz < kv.nkstot; ++ikibz)
    {
        int nkstar_ibz = kv.kstars[ikibz].size();
        for (int ikstar = 0;ikstar < nkstar_ibz;++ikstar)
        {
            for (int ib = 0; ib < GlobalV::NBANDS; ++ib) wg_full(ik_full, ib) = wg(ikibz, ib) / nkstar_ibz;
            ++ik_full;
        }
    }
    assert(ik_full = kv.nkstot_full);

    loc.dm_k.resize(kv.nkstot_full);
    elecstate::cal_dm(loc.ParaV, wg_full, psi_full, loc.dm_k);
}
template<typename Tdata>
void Exx_LRI_Interface<Tdata>::exx_eachiterinit(
    const Local_Orbital_Charge& loc,
    const Charge_Mixing& chgmix,
    const ModuleSymmetry::Symmetry& symm,
    const int& iter)
{
    if (GlobalC::exx_info.info_global.cal_exx)
    {
        if (!GlobalC::exx_info.info_global.separate_loop && exx_lri->two_level_step)
        {
            exx_lri->mix_DMk_2D.set_mixing_beta(chgmix.get_mixing_beta());
			if(chgmix.get_mixing_mode() == "pulay")
				exx_lri->mix_DMk_2D.set_coef_pulay(iter, chgmix);
			const bool flag_restart = (iter==1) ? true : false;
			if(GlobalV::GAMMA_ONLY_LOCAL)
				exx_lri->mix_DMk_2D.mix(loc.dm_gamma, flag_restart);
			else
				exx_lri->mix_DMk_2D.mix(loc.dm_k, flag_restart);

            exx_lri->cal_exx_elec(*loc.LOWF->ParaV, symm);
        }
    }
}

template<typename Tdata>
void Exx_LRI_Interface<Tdata>::exx_hamilt2density(elecstate::ElecState& elec, const Parallel_Orbitals& pv, const ModuleSymmetry::Symmetry& symm)
{
    // Peize Lin add 2020.04.04
    if (XC_Functional::get_func_type() == 4 || XC_Functional::get_func_type() == 5)
    {
        // add exx
        // Peize Lin add 2016-12-03
        elec.set_exx(this->get_Eexx());

        if (GlobalC::restart.info_load.load_H && GlobalC::restart.info_load.load_H_finish
            && !GlobalC::restart.info_load.restart_exx)
        {
            XC_Functional::set_xc_type(GlobalC::ucell.atoms[0].ncpp.xc_func);

            exx_lri->cal_exx_elec(pv, symm);
            GlobalC::restart.info_load.restart_exx = true;
        }
    }
    else
    {
        elec.f_en.exx = 0.;
    }
}

template<typename Tdata>
bool Exx_LRI_Interface<Tdata>::exx_after_converge(
    hamilt::Hamilt<double>& hamilt,
    LCAO_Matrix& lm,
    const Local_Orbital_Charge& loc,
    const K_Vectors& kv,
    const ModuleSymmetry::Symmetry& symm,
    int& iter)
{
    // Add EXX operator
    auto add_exx_operator = [&]() {
        if (GlobalV::GAMMA_ONLY_LOCAL)
        {
            hamilt::Operator<double>* exx
                = new hamilt::OperatorEXX<hamilt::OperatorLCAO<double>>(&lm,
                                                                        nullptr, // no explicit call yet
                                                                        &(lm.Hloc),
                                                                        kv);
            hamilt.opsd->add(exx);
        }
        else
        {
            hamilt::Operator<std::complex<double>>* exx
                = new hamilt::OperatorEXX<hamilt::OperatorLCAO<std::complex<double>>>(&lm,
                                                                                      nullptr, // no explicit call yet
                                                                                      &(lm.Hloc2),
                                                                                      kv);
            hamilt.ops->add(exx);
        }
    };
    
    if (GlobalC::exx_info.info_global.cal_exx)
    {
        // no separate_loop case
        if (!GlobalC::exx_info.info_global.separate_loop)
        {
            GlobalC::exx_info.info_global.hybrid_step = 1;

            // in no_separate_loop case, scf loop only did twice
            // in first scf loop, exx updated once in beginning,
            // in second scf loop, exx updated every iter

            if (exx_lri->two_level_step)
            {
                return true;
            }
            else
            {
                // update exx and redo scf
                XC_Functional::set_xc_type(GlobalC::ucell.atoms[0].ncpp.xc_func);
                iter = 0;
                std::cout << " Entering 2nd SCF, where EXX is updated" << std::endl;
                exx_lri->two_level_step++;

                add_exx_operator();

                return false;
            }
        }
        // has separate_loop case
        // exx converged or get max exx steps
        else if (exx_lri->two_level_step == GlobalC::exx_info.info_global.hybrid_step
                 || (iter == 1 && exx_lri->two_level_step != 0))
        {
            return true;
        }
        else
        {
            // update exx and redo scf
            if (exx_lri->two_level_step == 0)
            {
                add_exx_operator();
                XC_Functional::set_xc_type(GlobalC::ucell.atoms[0].ncpp.xc_func);
            }

			const bool flag_restart = (exx_lri->two_level_step==0) ? true : false;
			if (GlobalV::GAMMA_ONLY_LOCAL)
				exx_lri->mix_DMk_2D.mix(loc.dm_gamma, flag_restart);
			else
				exx_lri->mix_DMk_2D.mix(loc.dm_k, flag_restart);

            // GlobalC::exx_lcao.cal_exx_elec(p_esolver->LOC, p_esolver->LOWF.wfc_k_grid);
            exx_lri->cal_exx_elec(*loc.LOWF->ParaV, symm);
            iter = 0;
            std::cout << " Updating EXX and rerun SCF" << std::endl;
            exx_lri->two_level_step++;
            return false;
        }
    }
    return true;
}