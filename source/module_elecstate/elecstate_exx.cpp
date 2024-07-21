#include "module_elecstate/elecstate.h"
#include "module_parameter/parameter.h"

namespace elecstate
{

#ifdef __EXX
#ifdef __LCAO
    /// @brief calculation if converged
    /// @date Peize Lin add 2016-12-03
    void ElecState::set_exx(const double& Eexx)
    {
        ModuleBase::TITLE("energy", "set_exx");

        if (PARAM.exx_info.info_global.cal_exx)
        {
            this->f_en.exx = PARAM.exx_info.info_global.hybrid_alpha * Eexx;
        }
        return;
    }
#endif //__LCAO
#endif //__EXX

}