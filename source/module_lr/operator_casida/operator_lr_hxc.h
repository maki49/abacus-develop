#pragma once
#include "module_cell/klist.h"
#include "module_hamilt_general/operator.h"
#include "module_lr/utils/gint_template.h"
#include "module_hamilt_lcao/module_gint/grid_technique.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_lr/potentials/pot_lr_base.h"
#include "module_lr/utils/lr_util.h"
#include "module_lr/utils/lr_util_hcontainer.h"
namespace LR
{
    /// @brief  Hxc part of A operator for LR-TDDFT
    template<typename T = double, typename Device = base_device::DEVICE_CPU>
    class OperatorLRHxc : public hamilt::Operator<T, Device>
    {
    public:
        /// @brief type of molecular orbital to atomic orbital transformation:
        /// CC_vo: MO = C_v^* AO  C_o;
        /// CC_oo: MO = C_o^* AO  C_o;
        /// CXC: MO = C_v^* AO X C_v- C_o^* X^* AO C_o
        /// CXC_o: MO = C_o^* X^* AO C_o
        enum class MO_TO_AO_TYPE { CC_vo, CC_oo, CXC, CXC_o };

        //when nspin=2, nks is 2 times of real number of k-points. else (nspin=1 or 4), nks is the real number of k-points
        OperatorLRHxc(const int& nspin,
            const int& naos,
            const std::vector<int>& nocc,
            const std::vector<int>& nvirt,
            const psi::Psi<T, Device>& psi_ks_in,
            elecstate::DensityMatrix<T, T>& DM_trans_in,
            typename TGint<T>::type* gint_in,
            std::weak_ptr<PotLRBase> pot_in,
            const UnitCell& ucell_in,
            const std::vector<double>& orb_cutoff,
            const Grid_Driver& gd_in,
            const K_Vectors& kv_in,
            const std::vector<Parallel_2D>& pX_in,
            const Parallel_2D& pc_in,
            const Parallel_Orbitals& pmat_in,
            const std::vector<int>& ispin_ks = { 0 },
            const T factor_in = (T)1.0,
            const MO_TO_AO_TYPE dm_pq_in = MO_TO_AO_TYPE::CC_vo)
          : nspin(nspin), naos(naos), nocc(nocc), nvirt(nvirt), nk(kv_in.get_nks() / nspin), psi_ks(psi_ks_in),
            DM_trans(DM_trans_in), gint(gint_in), pot(pot_in), ucell(ucell_in), orb_cutoff_(orb_cutoff), gd(gd_in),
            kv(kv_in), pX(pX_in), pc(pc_in), pmat(pmat_in), ispin_ks(ispin_ks),
            factor_(factor_in), dm_pq_(dm_pq_in)
        {
          ModuleBase::TITLE("OperatorLRHxc", "OperatorLRHxc");
          this->cal_type = hamilt::calculation_type::lcao_gint;
          this->is_first_node = true;
          this->hR = std::unique_ptr<hamilt::HContainer<T>>(new hamilt::HContainer<T>(&pmat_in));
          LR_Util::initialize_HR<T, T>(*this->hR, ucell_in, gd_in, orb_cutoff);
          assert(&pmat_in == this->hR->get_paraV());
        };
        ~OperatorLRHxc() { };

        void init(const int ik_in) override {};

        virtual void act(const int nbands,
                         const int nbasis,
                         const int npol,
                         const T* psi_in,
                         T* hpsi,
                         const int ngk_ik = 0,
                         const bool is_first_node = false) const override;

      private:
        void grid_calculation(const int& nbands)const;

        //global sizes
        const int& nspin;
        const int& naos;
        const int nk = 1;
        // const int nloc_per_band = 1;    ///< local size of each state of X  (passed by nbasis in act())
        const std::vector<int>& nocc;
        const std::vector<int>& nvirt;
        const std::vector<int> ispin_ks = { 0 };  ///< the index of spin of psi_ks used in {AX, DM_trans}
        const K_Vectors& kv;
        /// ground state wavefunction
        const psi::Psi<T, Device>& psi_ks = nullptr;

        /// transition density matrix
        elecstate::DensityMatrix<T, T>& DM_trans;

        /// transition hamiltonian in AO representation
        std::unique_ptr<hamilt::HContainer<T>> hR = nullptr;

        /// parallel info
        const Parallel_2D& pc;
        const std::vector<Parallel_2D>& pX; // output vector, OV/OO/VV
        const Parallel_Orbitals& pmat;

        std::weak_ptr<PotLRBase> pot;

        typename TGint<T>::type* gint = nullptr;

        const UnitCell& ucell;
        std::vector<double> orb_cutoff_;
        const Grid_Driver& gd;

        MO_TO_AO_TYPE dm_pq_ = MO_TO_AO_TYPE::CC_vo;
        const T factor_ = (T)1.0;

        /// test
        mutable bool first_print = true;
    };
}
