#include "module_hamilt_lcao/hamilt_lcaodft/FORCE_STRESS.h"

template<typename T>
class LR_Force : public Force_Stress_LCAO<T>
{
public:
    LR_Force(ModulePW::PW_Basis* rhopw,
        const Structure_Factor& sf
    ) : rhopw(rhopw), sf(sf) {}

    /// 1. $Tr[H_{GS}^x * (T+D^Z)]
    void lr_force_gs_dm_diff_relaxed(const elecstate::DensityMatrix<T>& dm_diff_relaxed,
        ModuleBase::matrix& fcc);

    /// 2. $Tr[S^x * (EDM)]
    void lr_force_nonortho_edm(const elecstate::DensityMatrix<T>& mixed_edm,
        ModuleBase::matrix& fcc);

    /// 3. $\sum_{mnkl}(mn|f_{Hxc}|kl)^x *D^X *D^X$
    void lr_force_hxc_dm_trans(const elecstate::DensityMatrix<T>& dm_trans,
        ModuleBase::matrix& fcc);

#ifdef __EXX
    /// 4. $\sum_{mnkl}(mk|nl)^x *D^X *D^X$
    void lr_force_exx_dm_diff(const elecstate::DensityMatrix<T>& dm_diff,
        ModuleBase::matrix& fcc);
#endif

protected:
    ModulePW::PW_Basis* rhopw;
    const Structure_Factor& sf;
    TGint<T>::type* gint;
    
};