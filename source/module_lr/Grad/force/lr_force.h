#include "force_funcs_lcao.h"
#include "module_lr/potentials/pot_hxc_lrtd.h"
#include "module_lr/utils/gint_template.h"
// free functions, usefull for both ground and excited state

namespace LR
{

    template<typename TK>
    class LR_Force
    {
    public:
        LR_Force(const UnitCell& ucell,
            const std::vector<ModuleBase::Vector3<double>>& kvec_d,
            const Parallel_Orbitals& pv,
            const ModulePW::PW_Basis& rhopw,
            const pseudopot_cell_vl& locpp,
            const Structure_Factor& sf,
            const Grid_Driver& gd,
            typename TGint<TK>::type* gint, ///<  for grid integrals
            const TwoCenterBundle& two_center_bundle)///<  for 2-center integrals
            : ucell_(ucell), kvec_d_(kvec_d), pv_(pv),
            rhopw_(rhopw), sf_(sf), locpp_(locpp), gd_(gd),
            gint_(gint), two_center_bundle_(two_center_bundle) {
        }

        /// 1. $Tr[H_{GS}^x * (T+D^Z)]$, where GS=groud state and $(T+D^Z)$ is the relaxed difference density matrix
        ModuleBase::matrix cal_force_hamilt_gs_dm_relaxed_diff(const elecstate::DensityMatrix<TK, double>& relaxed_diff_dm, const elecstate::Potential& pot_gs);

        /// 2. $Tr[S^x * (EDM)]
        ModuleBase::matrix cal_force_overlap_edm(const elecstate::DensityMatrix<TK, double>& edm);

        /// 3. $\sum_{mnkl}(mn|f_{Hxc}|kl)^x *D^X *D^X$
        ModuleBase::matrix cal_force_hxc_dmtrans(const elecstate::DensityMatrix<TK, double>& dm_trans, const PotHxcLR& pot_hxc);

#ifdef __EXX
        /// 4. $\alpha \sum_{mnkl}(mk|nl)^x *D^X *D^X$
        // ModuleBase::matrix cal_force_exx_dm_diff(const elecstate::DensityMatrix<TK, double>& dm_trans);
#endif

        /// test functions
        /// reproduce the force of the ground state
        ModuleBase::matrix reproduce_force_gs(const elecstate::DensityMatrix<TK, double>& dm_gs,
            const elecstate::DensityMatrix<TK, double>& edm_gs,
            const elecstate::Potential& pot_gs);
        /// repreduce the ground state local term
        ModuleBase::matrix reproduce_force_gs_loc(const elecstate::DensityMatrix<TK, double>& dm_gs,
            const elecstate::Potential& pot_gs);

    protected:
        const UnitCell& ucell_;
        const std::vector<ModuleBase::Vector3<double>>& kvec_d_;
        const Parallel_Orbitals& pv_;
        const ModulePW::PW_Basis& rhopw_;
        const pseudopot_cell_vl& locpp_;
        const Structure_Factor& sf_;
        const Grid_Driver& gd_;
        typename TGint<TK>::type* gint_;
        const TwoCenterBundle& two_center_bundle_;

        Charge dm_to_charge(const elecstate::DensityMatrix<TK, double>& dm);

        // probably move frome the ground state?
        // void build_dHS()
    };
}
