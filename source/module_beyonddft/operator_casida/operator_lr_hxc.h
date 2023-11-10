#pragma once
#include "module_hamilt_general/operator.h"
#include "module_hamilt_lcao/module_gint/gint_gamma.h"
#include "module_hamilt_lcao/module_gint/grid_technique.h"
#include "module_hamilt_lcao/module_gint/gint_k.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_beyonddft/potentials/pot_hxc_lrtd.hpp"

namespace hamilt
{
    template <typename T> struct TGint;
    template <>
    struct TGint<double> {
        using type = Gint_Gamma;
    };
    template <>
    struct TGint<std::complex<double>> {
        using type = Gint_k;
    };

    /// @brief  Hxc part of A operator for LR-TDDFT
    template<typename T = double, typename Device = psi::DEVICE_CPU>
    class OperatorLRHxc : public Operator<T, Device>
    {
    public:
        OperatorLRHxc(const int& nspin,
            const int& naos,
            const int& nocc,
            const int& nvirt,
            const psi::Psi<T, Device>* psi_ks_in,
            elecstate::DensityMatrix<T, double>* DM_trans_in,
            HContainer<double>* hR_in,
            typename TGint<T>::type* gint_in,
            elecstate::PotHxcLR* pot_in,
            const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
            const std::vector<Parallel_2D*> p2d_in /*< 2d-block parallel info of {X, c, matrix}*/)
            : nspin(nspin), naos(naos), nocc(nocc), nvirt(nvirt),
            psi_ks(psi_ks_in), DM_trans(DM_trans_in), hR(hR_in), gint(gint_in), pot(pot_in), kvec_d(kvec_d_in),
            pX(p2d_in.at(0)), pc(p2d_in.at(1)), pmat(p2d_in.at(2))
        {
            ModuleBase::TITLE("OperatorLRHxc", "OperatorLRHxc");
            this->nks = std::is_same<T, double>::value ? 1 : kvec_d.size();
            this->nsk = std::is_same<T, double>::value ? nspin : nks;
            this->cal_type = calculation_type::lcao_gint;
            this->act_type = 2;
            this->is_first_node = true;
        };

        void init(const int ik_in) override {};

        // virtual psi::Psi<T> act(const psi::Psi<T>& psi_in) const override;
        virtual void act(const psi::Psi<T>& psi_in, psi::Psi<T>& psi_out, const int nbands) const override;
    private:
        //global sizes
        int nks = 1;    // when nspin=2, nks is 2 times of real number of k-points.
        int nsk = 1; // nspin for gamma_only, nks for multi-k
        int nspin = 1;
        int naos;
        int nocc;
        int nvirt;
        const std::vector<ModuleBase::Vector3<double>>& kvec_d;
        /// ground state wavefunction
        const psi::Psi<T, Device>* psi_ks = nullptr;

        /// transition density matrix
        elecstate::DensityMatrix<T, double>* DM_trans;

        /// transition hamiltonian in AO representation
        hamilt::HContainer<double>* hR = nullptr;

        //parallel info
        Parallel_2D* pc = nullptr;
        Parallel_2D* pX = nullptr;
        Parallel_2D* pmat = nullptr;

        elecstate::PotHxcLR* pot = nullptr;

        typename TGint<T>::type* gint = nullptr;

        bool tdm_sym = false; ///< whether transition density matrix is symmetric
    };
}
#include "module_beyonddft/operator_casida/operator_lr_hxc.hpp"