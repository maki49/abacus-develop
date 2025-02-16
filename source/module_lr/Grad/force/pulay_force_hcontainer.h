#pragma once 
#include "module_basis/module_nao/two_center_bundle.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_cell/unitcell.h"
#include "module_lr/utils/lr_util.h"

namespace PulayForceStress
{
// add stess later
/// for 2-center-integration terms, provided HS derivatives
    template<typename TK>
ModuleBase::matrix cal_pulay_fs(
    const elecstate::DensityMatrix<TK, double>& dm,  ///< [in] density matrix or energy density matrix
    const UnitCell& ucell,  ///< [in] unit cell
    const std::vector<hamilt::HContainer<double>>& dHS,  ///< [in] dHS x, y, z, for force
    const double& factor_force = 1.0)
{
    ModuleBase::matrix f(ucell.nat, 3);
    const Parallel_Orbitals& pv = *dHS[0].get_paraV();
    const int& npol = ucell.get_npol();
    const int nspin_dmr = dm.get_DMR_vector().size();
    for (int ixyz = 0;ixyz < 3;++ixyz)
    {
        for (int iat0 = 0;iat0 < ucell.nat;++iat0)
        {
            for (int iat1 = 0;iat1 < ucell.nat;++iat1)
            {
                hamilt::AtomPair<double>* ap = dHS[ixyz].find_pair(iat0, iat1);
                if (ap)
                {
                    for (int iR = 0;iR < ap->get_R_size();++iR)
                    {
                        const ModuleBase::Vector3<int>& R = ap->get_R_index(iR);
                        hamilt::BaseMatrix<double>& mat_dhs = ap->get_HR_values(R.x, R.y, R.z);
                        std::vector<const hamilt::BaseMatrix<double>*> mat_dmr;
                        for (int is = 0; is < nspin_dmr; ++is)
                        {
                            mat_dmr.push_back(dm.get_DMR_pointer(is + 1)->find_matrix(iat0, iat1, R.x, R.y, R.z));
                        }

                        for (int mu = 0; mu < pv.get_row_size(iat0); mu += npol)
                        {
                            for (int nu = 0; nu < pv.get_col_size(iat1); nu += npol)
                            {
                                double dm2d = 0.0;
                                for (int is = 0; is < nspin_dmr; ++is) { dm2d += mat_dmr[is]->get_value(mu, nu); }
                                f(iat0, ixyz) += dm2d * factor_force * 2.0 * mat_dhs.get_value(mu, nu);
                            }
                        }
                    }
                }
            }
        }
    }
    return f;
}

/// for grid-integration terms
template<typename TK>
ModuleBase::matrix cal_pulay_fs(
    const elecstate::DensityMatrix<TK, double>& dm,  ///< [in] density matrix or energy density matrix
    const UnitCell& ucell,  ///< [in] unit cell
    const LR::PotHxcLR* pot, ///< [in] potential on grid
    typename LR::TGint<TK>::type& gint ///< [in] Gint object
)
{
    ModuleBase::matrix force(ucell.nat, 3);
    ModuleBase::matrix stress_tmp;

    const int nspin = dm.get_DMR_vector().size();
    gint.transfer_DM2DtoGrid(dm.get_DMR_vector());

    // 1. dm->rho
    double** rho;
    const int& nrxx = pot->nrxx;
    LR_Util::_allocate_2order_nested_ptr(rho, nspin, nrxx);
    ModuleBase::GlobalFunc::ZEROS(rho[0], nrxx);
    Gint_inout inout_rho(rho, Gint_Tools::job_type::rho, 1, false);
    gint.cal_gint(&inout_rho);

    // 2. v_hxc = f_hxc * rho
    ModuleBase::matrix vr_hxc(1, nrxx);   //grid
    pot->cal_v_eff(rho, ucell, vr_hxc);
    LR_Util::_deallocate_2order_nested_ptr(rho, 1);

    // 3. v(r) -> force
#ifndef __NEW_GINT
    Gint_inout inout(0, &vr_hxc(0, 0), /*isforce=*/true, /*isstress=*/false, &force, &stress_tmp, Gint_Tools::job_type::force);
    gint.cal_gint(&inout);
#else
    std::vector<const double*> p_vr_eff(1, nullptr);
    p_vr_eff[0] = &vr_hxc(0, 0);
    ModuleGint::cal_gint_fvl(nspin, p_vr_hxc, dm.get_DMR_vector(), i/*isforce=*/true, /*isstress=*/false, &force, &stress_tmp);
#endif
    return force;
}
}
 