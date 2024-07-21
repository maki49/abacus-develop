#include "parameter.h"

Parameter PARAM;

void Parameter::set_rank_nproc(const int& myrank, const int& nproc)
{
    sys.myrank = myrank;
    sys.nproc = nproc;
    input.mdp.my_rank = myrank;
}

void Parameter::set_start_time(const std::time_t& start_time)
{
    sys.start_time = start_time;
}

#ifdef __EXX
void Parameter::set_exx_postscf()
{
    exx_info_.info_global.ccp_type = Conv_Coulomb_Pot_K::Ccp_Type::Hf;
    exx_info_.info_global.hybrid_alpha = 1;
    exx_info_.info_ri.ccp_rmesh_times = inp.rpa_ccp_rmesh_times;
}
#endif