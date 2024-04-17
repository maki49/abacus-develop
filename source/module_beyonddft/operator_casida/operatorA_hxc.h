#pragma once
#include "module_hamilt_general/operator.h"
#include "module_hamilt_lcao/module_gint/gint_gamma.h"
#include "module_hamilt_lcao/module_gint/grid_technique.h"
#include "module_hamilt_lcao/module_gint/gint_k.h"
#include "module_beyonddft/potentials/pot_hxc_lrtd.hpp"

namespace hamilt
{

    /// @brief  Hxc part of A operator
    template<typename T = double, typename Device = psi::DEVICE_CPU>
    class OperatorA_Hxc : public Operator<T, Device>
    {
    public:
        OperatorA_Hxc(const int& nsk,
            const int& naos,
            const int& nocc,
            const int& nvirt,
            const psi::Psi<T, Device>* psi_ks_in,
            Gint_Gamma* gg_in,
            elecstate::PotHxcLR* pot_in,
            const std::vector<Parallel_2D*> p2d_in /*< 2d-block parallel info of {X, c, matrix}*/)
            : nsk(nsk), naos(naos), nocc(nocc), nvirt(nvirt),
            psi_ks(psi_ks_in), gg(gg_in), pot(pot_in),
            pX(p2d_in.at(0)), pc(p2d_in.at(1)), pmat(p2d_in.at(2))
        {
            ModuleBase::TITLE("OperatorA_Hxc", "OperatorA_Hxc(gamma)");
            this->cal_type = calculation_type::lcao_gint;
            this->is_first_node = true;
        };
        OperatorA_Hxc(const int& nsk,
            const int& naos,
            const int& nocc,
            const int& nvirt,
            const psi::Psi<T, Device>* psi_ks_in,
            Gint_k* gk_in,
            elecstate::PotHxcLR* pot_in,
            const std::vector<Parallel_2D*> p2d_in /*< 2d-block parallel info of {X, c, matrix}*/)
            : nsk(nsk), naos(naos), nocc(nocc), nvirt(nvirt),
            psi_ks(psi_ks_in), gk(gk_in), pot(pot_in),
            pX(p2d_in.at(0)), pc(p2d_in.at(1)), pmat(p2d_in.at(2))
        {
            ModuleBase::TITLE("OperatorA_Hxc", "OperatorA_Hxc(k)");
            this->cal_type = calculation_type::lcao_gint;
            this->is_first_node = true;
        };
        void init(const int ik_in) override {};

        virtual void act(const int nbands,
            const int nbasis,
            const int npol,
            const T* tmpsi_in,
            T* tmhpsi,
            const int ngk_ik = 0)const override {};
        //tmp, for only one state
        // virtual psi::Psi<T> act(const psi::Psi<T>& psi_in) const override;
        virtual void act(const psi::Psi<T>& psi_in, psi::Psi<T>& psi_out) const override;
    private:
        //global sizes
        int nsk;    //nspin*nkpoints
        int naos;
        int nocc;
        int nvirt;

        /// ground state wavefunction
        const psi::Psi<T, Device>* psi_ks = nullptr;
        //parallel info
        Parallel_2D* pc = nullptr;
        Parallel_2D* pX = nullptr;
        Parallel_2D* pmat = nullptr;

        elecstate::PotHxcLR* pot = nullptr;

        Gint_Gamma* gg = nullptr;
        Gint_k* gk = nullptr;
        /// \f[ \tilde{\rho}(r)=\sum_{\mu_j, \mu_b}\tilde{\rho}_{\mu_j,\mu_b}\phi_{\mu_b}(r)\phi_{\mu_j}(r) \f]
        void cal_rho_trans();
    };
}
#include "module_beyonddft/operator_casida/operatorA_hxc.hpp"