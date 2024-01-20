#include "Exx_LRI_interface.h"
#include "module_ri/exx_abfs-jle.h"
#include "module_ri/exx_opt_orb.h"
#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/op_exx_lcao.h"

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::write_Hexxs(const std::string& file_name) const
{
	ModuleBase::TITLE("Exx_LRI","write_Hexxs");
	ModuleBase::timer::tick("Exx_LRI", "write_Hexxs");
	std::ofstream ofs(file_name, std::ofstream::binary);
	cereal::BinaryOutputArchive oar(ofs);
	oar(this->exx_ptr->Hexxs);
	ModuleBase::timer::tick("Exx_LRI", "write_Hexxs");
}

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::read_Hexxs(const std::string& file_name)
{
	ModuleBase::TITLE("Exx_LRI","read_Hexxs");
	ModuleBase::timer::tick("Exx_LRI", "read_Hexxs");
	std::ifstream ifs(file_name, std::ofstream::binary);
	cereal::BinaryInputArchive iar(ifs);
	iar(this->exx_ptr->Hexxs);
	ModuleBase::timer::tick("Exx_LRI", "read_Hexxs");
}
template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::exx_beforescf(const K_Vectors& kv, const Charge_Mixing& chgmix, const UnitCell& ucell, const Parallel_2D& pv)
{
#ifdef __MPI
		if ( GlobalC::exx_info.info_global.cal_exx )
		{
            if (ucell.atoms[0].ncpp.xc_func == "HF" || ucell.atoms[0].ncpp.xc_func == "PBE0" || ucell.atoms[0].ncpp.xc_func == "HSE")
            {
                XC_Functional::set_xc_type("pbe");
            }
            else if (ucell.atoms[0].ncpp.xc_func == "SCAN0")
            {
                XC_Functional::set_xc_type("scan");
            }

            this->exx_ptr->cal_exx_ions();

            // initialize the rotation matrix in AO representation
            this->exx_spacegroup_symmetry = (!GlobalV::GAMMA_ONLY_LOCAL && GlobalV::NSPIN < 4 && ModuleSymmetry::Symmetry::symm_flag == 1);
            if (this->exx_spacegroup_symmetry)
            {
                this->symrot_.cal_Ms(kv, ucell, pv);
                // test: irreducible atom pairs
                this->symrot_.find_irreducible_atom_pairs(ucell.symm);
                // test: irreducible R
                this->symrot_.find_irreducible_atom_pairs_set(ucell.symm);
                this->symrot_.find_irreducible_R(ucell.symm, ucell.atoms, ucell.st, kv);
                this->symrot_.output_irreducible_R(kv);
                this->symrot_.get_return_lattice_all(ucell.symm, ucell.atoms, ucell.st);
            }
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
            if (this->exx_spacegroup_symmetry)
                this->mix_DMk_2D.set_nks(kv.nkstot_full * (GlobalV::NSPIN == 2 ? 2 : 1), GlobalV::GAMMA_ONLY_LOCAL);
            else
                this->mix_DMk_2D.set_nks(kv.nks, GlobalV::GAMMA_ONLY_LOCAL);
			if(GlobalC::exx_info.info_global.separate_loop)
			{
                this->mix_DMk_2D.set_mixing(nullptr);
			}
			else
			{
				this->mix_DMk_2D.set_mixing(chgmix.mixing);
            }
        }
        // for exx two_level scf
        this->two_level_step = 0;
#endif // __MPI
}

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::exx_eachiterinit(const elecstate::DensityMatrix<T, double>& dm, const K_Vectors& kv, const int& iter)
{
    if (GlobalC::exx_info.info_global.cal_exx)
    {
        if (!GlobalC::exx_info.info_global.separate_loop && this->two_level_step)
        {
            const bool flag_restart = (iter == 1) ? true : false;
            if (this->exx_spacegroup_symmetry)
                this->mix_DMk_2D.mix(symrot_.restore_dm(kv, dm.get_DMK_vector(), *dm.get_paraV_pointer()), flag_restart);
            else
                this->mix_DMk_2D.mix(dm.get_DMK_vector(), flag_restart);
			const std::vector<std::map<int,std::map<std::pair<int, std::array<int, 3>>,RI::Tensor<Tdata>>>>
				Ds = GlobalV::GAMMA_ONLY_LOCAL
					? RI_2D_Comm::split_m2D_ktoR<Tdata>(*this->exx_ptr->p_kv, this->mix_DMk_2D.get_DMk_gamma_out(), *dm.get_paraV_pointer())
                : RI_2D_Comm::split_m2D_ktoR<Tdata>(*this->exx_ptr->p_kv, this->mix_DMk_2D.get_DMk_k_out(), *dm.get_paraV_pointer(), this->exx_spacegroup_symmetry);
            this->exx_ptr->cal_exx_elec(Ds, *dm.get_paraV_pointer());
        }
    }
}

template<typename T, typename Tdata>
void Exx_LRI_Interface<T, Tdata>::exx_hamilt2density(elecstate::ElecState& elec, const Parallel_Orbitals& pv)
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

			const std::vector<std::map<int,std::map<std::pair<int, std::array<int, 3>>,RI::Tensor<Tdata>>>>
				Ds = GlobalV::GAMMA_ONLY_LOCAL
					? RI_2D_Comm::split_m2D_ktoR<Tdata>(*this->exx_ptr->p_kv, this->mix_DMk_2D.get_DMk_gamma_out(), pv)
                : RI_2D_Comm::split_m2D_ktoR<Tdata>(*this->exx_ptr->p_kv, this->mix_DMk_2D.get_DMk_k_out(), pv, this->exx_spacegroup_symmetry);
            this->exx_ptr->cal_exx_elec(Ds, pv);
            GlobalC::restart.info_load.restart_exx = true;
        }
    }
    else
    {
        elec.f_en.exx = 0.;
    }
}

template<typename T, typename Tdata>
bool Exx_LRI_Interface<T, Tdata>::exx_after_converge(
    hamilt::Hamilt<T>& hamilt,
    LCAO_Matrix& lm,
    const elecstate::DensityMatrix<T, double>& dm,
    const K_Vectors& kv,
    int& iter)
{
    // Add EXX operator
    auto add_exx_operator = [&]() {
        if (GlobalV::GAMMA_ONLY_LOCAL)
        {
            hamilt::HamiltLCAO<double, double>* hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<double, double>*>(&hamilt);
            hamilt::Operator<double>* exx
                = new hamilt::OperatorEXX<hamilt::OperatorLCAO<double, double>>(&lm,
                                                                        hamilt_lcao->getHR(), 
                                                                        &(hamilt_lcao->getHk(&lm)),
                                                                        kv);
            hamilt_lcao->getOperator()->add(exx);
        }
        else
        {
            hamilt::Operator<std::complex<double>>* exx;
            if(GlobalV::NSPIN < 4)
            {
                hamilt::HamiltLCAO<std::complex<double>, double>* hamilt_lcao = 
                    dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, double>*>(&hamilt);
                exx = new hamilt::OperatorEXX<hamilt::OperatorLCAO<std::complex<double>, double>>(&lm,
                                                                                    hamilt_lcao->getHR(), 
                                                                                    &(hamilt_lcao->getHk(&lm)),
                                                                                    kv);
                hamilt_lcao->getOperator()->add(exx);
            }
            else
            {
                hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>* hamilt_lcao = 
                    dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>*>(&hamilt);
                exx = new hamilt::OperatorEXX<hamilt::OperatorLCAO<std::complex<double>, std::complex<double>>>(&lm,
                                                                                              hamilt_lcao->getHR(), 
                                                                                              &(hamilt_lcao->getHk(&lm)),
                                                                                              kv);
                hamilt_lcao->getOperator()->add(exx);
            }
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

            if (this->two_level_step)
            {
                return true;
            }
            else
            {
                // update exx and redo scf
                XC_Functional::set_xc_type(GlobalC::ucell.atoms[0].ncpp.xc_func);
                iter = 0;
                std::cout << " Entering 2nd SCF, where EXX is updated" << std::endl;
                this->two_level_step++;

                add_exx_operator();

                return false;
            }
        }
        // has separate_loop case
        // exx converged or get max exx steps
        else if (this->two_level_step == GlobalC::exx_info.info_global.hybrid_step
                 || (iter == 1 && this->two_level_step != 0))
        {
            return true;
        }
        else
        {
            // update exx and redo scf
            if (this->two_level_step == 0)
            {
                add_exx_operator();
                XC_Functional::set_xc_type(GlobalC::ucell.atoms[0].ncpp.xc_func);
            }

            const bool flag_restart = (this->two_level_step == 0) ? true : false;

            if (this->exx_spacegroup_symmetry)
                this->mix_DMk_2D.mix(symrot_.restore_dm(kv, dm.get_DMK_vector(), *dm.get_paraV_pointer()), flag_restart);
            else
                this->mix_DMk_2D.mix(dm.get_DMK_vector(), flag_restart);

            // GlobalC::exx_lcao.cal_exx_elec(p_esolver->LOC, p_esolver->LOWF.wfc_k_grid);
			const std::vector<std::map<int,std::map<std::pair<int, std::array<int, 3>>,RI::Tensor<Tdata>>>>
				Ds = GlobalV::GAMMA_ONLY_LOCAL
					? RI_2D_Comm::split_m2D_ktoR<Tdata>(*this->exx_ptr->p_kv, this->mix_DMk_2D.get_DMk_gamma_out(), *dm.get_paraV_pointer())
                : RI_2D_Comm::split_m2D_ktoR<Tdata>(*this->exx_ptr->p_kv, this->mix_DMk_2D.get_DMk_k_out(), *dm.get_paraV_pointer(), this->exx_spacegroup_symmetry);
            this->exx_ptr->cal_exx_elec(Ds, *dm.get_paraV_pointer());

            // check the rotation of Hexx
            this->symrot_.test_HR_rotation(GlobalC::ucell.symm, GlobalC::ucell.atoms, GlobalC::ucell.st, this->exx_ptr->Hexxs[0]);

            iter = 0;
            std::cout << " Updating EXX and rerun SCF" << std::endl;
            this->two_level_step++;
            return false;
        }
    }
    return true;
}