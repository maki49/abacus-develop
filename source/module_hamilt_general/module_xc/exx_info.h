#ifndef EXX_INFO_H
#define EXX_INFO_H

#include "module_ri/conv_coulomb_pot_k.h"
#include "xc_functional.h"
#include "module_parameter/input_parameter.h"
struct Exx_Info
{
    struct Exx_Info_Global
    {
        bool cal_exx = false;

        Conv_Coulomb_Pot_K::Ccp_Type ccp_type;
        double hybrid_alpha = 0.25;
        double hse_omega = 0.11;
        double mixing_beta_for_loop1 = 1.0;

        bool separate_loop = true;
        size_t hybrid_step = 1;
    };
    Exx_Info_Global info_global;

    struct Exx_Info_Lip
    {
        const Conv_Coulomb_Pot_K::Ccp_Type& ccp_type;
        const double& hse_omega;
        double lambda = 0.3;

        Exx_Info_Lip(const Exx_Info::Exx_Info_Global& info_global)
            :ccp_type(info_global.ccp_type),
            hse_omega(info_global.hse_omega) {}
    };
    Exx_Info_Lip info_lip;

    struct Exx_Info_RI
    {
        const Conv_Coulomb_Pot_K::Ccp_Type& ccp_type;
        const double& hse_omega;

        bool real_number = false;

        double pca_threshold = 0;
        std::vector<std::string> files_abfs;
        double C_threshold = 0;
        double V_threshold = 0;
        double dm_threshold = 0;
        double cauchy_threshold = 0;
        double C_grad_threshold = 0;
        double V_grad_threshold = 0;
        double cauchy_force_threshold = 0;
        double cauchy_stress_threshold = 0;
        double ccp_rmesh_times = 10;
        double kmesh_times = 4;

        int abfs_Lmax = 0; // tmp

        Exx_Info_RI(const Exx_Info::Exx_Info_Global& info_global)
            : ccp_type(info_global.ccp_type), hse_omega(info_global.hse_omega) {}
    };
    Exx_Info_RI info_ri;

    Exx_Info() : info_lip(this->info_global), info_ri(this->info_global) {}

    void set(const Input_para& inp)
    {
        std::string dft_functional_lower = inp.dft_functional;
        std::transform(inp.dft_functional.begin(),
            inp.dft_functional.end(),
            dft_functional_lower.begin(),
            tolower);
        if (dft_functional_lower == "hf" || dft_functional_lower == "pbe0" || dft_functional_lower == "scan0")
        {
            info_global.cal_exx = true;
            info_global.ccp_type
                = Conv_Coulomb_Pot_K::Ccp_Type::Hf;
        }
        else if (dft_functional_lower == "hse") {
            info_global.cal_exx = true;
            info_global.ccp_type
                = Conv_Coulomb_Pot_K::Ccp_Type::Hse;
        }
        else if (dft_functional_lower == "opt_orb") {
            info_global.cal_exx = false;
        }
        else {
            info_global.cal_exx = false;
        }

        if (info_global.cal_exx || dft_functional_lower == "opt_orb" || inp.rpa)
        {
            // EXX case, convert all EXX related variables
            // info_global.cal_exx = true;
            info_global.hybrid_alpha = std::stod(inp.exx_hybrid_alpha);
            info_global.hse_omega = inp.exx_hse_omega;
            info_global.separate_loop = inp.exx_separate_loop;
            info_global.hybrid_step = inp.exx_hybrid_step;
            info_global.mixing_beta_for_loop1 = inp.exx_mixing_beta;
            info_lip.lambda = inp.exx_lambda;

            info_ri.real_number = std::stoi(inp.exx_real_number);
            info_ri.pca_threshold = inp.exx_pca_threshold;
            info_ri.C_threshold = inp.exx_c_threshold;
            info_ri.V_threshold = inp.exx_v_threshold;
            info_ri.dm_threshold = inp.exx_dm_threshold;
            info_ri.cauchy_threshold = inp.exx_cauchy_threshold;
            info_ri.C_grad_threshold = inp.exx_c_grad_threshold;
            info_ri.V_grad_threshold = inp.exx_v_grad_threshold;
            info_ri.cauchy_force_threshold = inp.exx_cauchy_force_threshold;
            info_ri.cauchy_stress_threshold = inp.exx_cauchy_stress_threshold;
            info_ri.ccp_rmesh_times = std::stod(inp.exx_ccp_rmesh_times);
        }
    }
};

#endif
