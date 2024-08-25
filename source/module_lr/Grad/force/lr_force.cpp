#include "lr_force.h"
#include "lr_util.h"
template<typename T>
Charge LR_Force<T>::dm_to_charge(const elecstate::DensityMatrix<T>& dm)
{
    Charge chr;
    chr.set_rhopw(this->rhopw);
    LR_Util::new_p2(chr.rho, this->nspin, this->rhopw->nrxx);
    // So huge a Charge class...
    // 1. Using a (private) `allocate_rho` to control whether to delete will definately cause memory leak here. No need for such judgement. 
    // 2. Charge-dependent interfaces need refactor: only rhopw and rho dependence are enough.

    this->gint->transfer_DM2DtoGrid(this->dm.get_DMR_vector());     // 2d block to grid
    Gint_inout inout_rho((double**)nullptr, chg.rho, Gint_Tools::job_type::rho, false);
    this->gint->cal_gint(&inout_rho);
    return chr;
}

template<typename T>
void LR_Force<T>::lr_force_gs_dm_diff_relaxed(const elecstate::DensityMatrix<T>& dm_diff_relaxed,
    ModuleBase::matrix& fcc)
{
    // Hamilt-term: 
    // vl: local pseudopotential
    // vnl: nonlocal pseudopotential

    // vlhxc: local pp + hartree + xc
    // ew: ewald (ion-ion)

    // type: 
    // dvx: hellmann-feynman
    // dphi: pulay
    // no: non-ortho

    const Charge chr_diff_relaxed = dm_to_charge(dm_diff_relaxed);

    ModuleBase::matrix fvl_dvl;
    ModuleBase::matrix few;
    ModuleBase::matrix fcc;
    this->calForcePwPart(fvl_dvl, few, fcc,
        /*fscc*/fcc, 0.0, /*vnew*/fcc, /*vnew_exist*/false,
        &chr_diff_relaxed, this->rhopw, this->sf);
}