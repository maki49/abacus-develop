#pragma once
#include "module_beyonddft/operator_casida/operator_lr_hxc.h"
namespace hamilt
{
    /// @brief  Kernel part of A+B for LR-TDDFT:
    /// \f[ H_{pq}[X] = \sum_{rs} [2(pq|f_{Hxc}|rs) + a_{EX}[(pr|qs)+(qr|ps)] ]\f]
    /// \f = Tr[(D+D^\dagger) \tilde{V}_{Hxc}]\f]
    template<typename T = double, typename Device = psi::DEVICE_CPU>
    class OperatorLRHxcCommon : public OperatorLRHxc<T, Device>    //herite first, replace later
    {
    public:

        /// @brief density matrix type:
        /// C: ground-state;
        /// X: transition amplitude or Z-vector
        /// Diff: difference density matrix
        enum class DM_TYPE { C, X, Diff };
        /// @brief AX type:
        /// CC: C_v^* C_o^T;
        /// CXC: C_v^* X^* C_v^T- C_o^* X^* C_o^T
        enum class AX_TYPE { CC, CXC };

        //when nspin=2, nks is 2 times of real number of k-points. else (nspin=1 or 4), nks is the real number of k-points
        OperatorLRHxcCommon(const int& nspin,
            const int& naos,
            const int& nocc,
            const int& nvirt,
            const psi::Psi<T, Device>* psi_ks_in,
            std::vector<elecstate::DensityMatrix<T, T>*>& DM_trans_in,
            typename TGint<T>::type* gint_in,
            elecstate::PotHxcLR* pot_in,
            const UnitCell& ucell_in,
            Grid_Driver& gd_in,
            const K_Vectors& kv_in,
            Parallel_2D* pX_in,
            Parallel_2D* pc_in,
            Parallel_Orbitals* pmat_in,
            DM_TYPE dm_rs_in,   ///< rs: the labels to be contracted with DM
            AX_TYPE dm_pq_in,   ///< pq: the labels of the final result
            const T factor_in)
            : dm_pq(dm_pq_in), dm_rs(dm_rs_in), factor(factor_in),
            OperatorLRHxc<T, Device>(nspin, naos, nocc, nvirt, psi_ks_in, DM_trans_in,
                gint_in, pot_in, ucell_in, gd_in, kv_in, pX_in, pc_in, pmat_in) {};

        ~OperatorLRHxcCommon() = default;

        // virtual psi::Psi<T> act(const psi::Psi<T>& psi_in) const override;
        virtual void act(const psi::Psi<T>& psi_in, psi::Psi<T>& psi_out, const int nbands) const override;

        /// range-independent version of hPsi (have been put into the base class)
        // virtual void hPsi(const psi::Psi<T>& psi_in, psi::Psi<T>& psi_out)const override;

    protected:

        void act_to_bfirst(const psi::Psi<T>& psi_in_bfirst, psi::Psi<T>& psi_out_bfirst, const int nbands) const;

        DM_TYPE dm_rs = DM_TYPE::X;
        AX_TYPE dm_pq = AX_TYPE::CC;
        const T factor = (T)1.0;
    };
}