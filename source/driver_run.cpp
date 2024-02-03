#include "driver.h"
#include "module_cell/module_neighbor/sltk_atom_arrange.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_io/input.h"
#include "module_io/print_info.h"
#include "module_io/winput.h"
#include "module_md/run_md.h"
#include "module_io/para_json.h"

#ifdef __LCAO
#include "module_beyonddft/esolver_lrtd_lcao.h"
#endif
extern "C"
{
#include "module_base/blacs_connector.h"
}
/**
 * @brief This is the driver function which defines the workflow of ABACUS calculations.
 * It relies on the class Esolver, which is a class that organizes workflows of single point calculations.
 * 
 * For calculations involving change of configuration (lattice parameter & ionic motion),
 * this driver calls Esolver::Run and the configuration-changing subroutine in a alternating manner.
 * 
 * Information is passed between the two subroutines by class UnitCell
 * 
 * Esolver::Run takes in a configuration and provides force and stress, 
 * the configuration-changing subroutine takes force and stress and updates the configuration
 */
void Driver::driver_run(void)
{
    ModuleBase::TITLE("Driver", "driver_line");
    ModuleBase::timer::tick("Driver", "driver_line");

    //! 1: initialize the ESolver 
    ModuleESolver::ESolver *p_esolver = nullptr;
    ModuleESolver::init_esolver(p_esolver);

    //! 2: setup cell and atom information
#ifndef __LCAO
    if(GlobalV::BASIS_TYPE == "lcao_in_pw" || GlobalV::BASIS_TYPE == "lcao")
    {
        ModuleBase::WARNING_QUIT("driver","to use LCAO basis, compile with __LCAO");
    }
#endif
    GlobalC::ucell.setup_cell(GlobalV::stru_file, GlobalV::ofs_running);

    //! 3: initialize Esolver and fill json-structure 
#ifdef __LCAO
    if (GlobalV::ESOLVER_TYPE == "lr")s
        // use constructor rather than Init function to initialize reference (instead of pointers) to ucell
        if (GlobalV::GAMMA_ONLY_LOCAL)
            p_esolver = new ModuleESolver::ESolver_LRTD<double, double>(INPUT, GlobalC::ucell);
        else if (GlobalV::NSPIN < 4)
            p_esolver = new ModuleESolver::ESolver_LRTD<std::complex<double>, double>(INPUT, GlobalC::ucell);
        else
            throw std::runtime_error("LR-TDDFT is not implemented for spin polarized case");
    else
#endif
        p_esolver->init(INPUT, GlobalC::ucell);

#ifdef __RAPIDJSON
    Json::gen_stru_wrapper(&GlobalC::ucell);
#endif

    //! 4: md or relax calculations 
    if(GlobalV::CALCULATION == "md")
    {
        Run_MD::md_line(GlobalC::ucell, p_esolver, INPUT.mdp);
    }
    else //! scf; cell relaxation; nscf; etc
    {
        if (GlobalV::precision_flag == "single")
        {
            Relax_Driver<float, psi::DEVICE_CPU> rl_driver;
            rl_driver.relax_driver(p_esolver);
        }
        else
        {
            Relax_Driver<double, psi::DEVICE_CPU> rl_driver;
            rl_driver.relax_driver(p_esolver);
        }
    }

    //! 5: clean up esolver
    p_esolver->postprocess();

#ifdef __LCAO
    //---------beyond DFT: set up the next ESolver---------
    if (INPUT.esolver_type == "ks-lr")
    {
        std::cout << "setting up the esolver for excited state" << std::endl;
        // ModuleESolver::ESolver_KS_LCAO* p_esolver_lcao_tmp = dynamic_cast<ModuleESolver::ESolver_KS_LCAO<double, double>*>(p_esolver);
        ModuleESolver::ESolver* p_esolver_lr = nullptr;
        if (INPUT.gamma_only)
            p_esolver_lr = new ModuleESolver::ESolver_LRTD<double, double>(std::move(*dynamic_cast<ModuleESolver::ESolver_KS_LCAO<double, double>*>(p_esolver)), INPUT, GlobalC::ucell);
        else
            p_esolver_lr = new ModuleESolver::ESolver_LRTD<std::complex<double>, double>(std::move(*dynamic_cast<ModuleESolver::ESolver_KS_LCAO<std::complex<double>, double>*>(p_esolver)), INPUT, GlobalC::ucell);
        ModuleESolver::clean_esolver(p_esolver);
        p_esolver_lr->Run(0, GlobalC::ucell);
        p_esolver_lr->postprocess();

        std::cout << "before clean lr" << std::endl;
        ModuleESolver::clean_esolver(p_esolver_lr);
        std::cout << "after clean lr" << std::endl;
    } //----------------------beyond DFT------------------------
    else
        ModuleESolver::clean_esolver(p_esolver);

    if (INPUT.basis_type == "lcao")
        Cblacs_exit(1); // clean up blacs after all the esolvers are cleaned up without closing MPI
#else
    ModuleESolver::clean_esolver(p_esolver);
#endif
    ModuleBase::timer::tick("Driver", "driver_line");
    return;
}
