#pragma once
#include "module_hamilt_general/operator.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_ri/Exx_LRI.h"
namespace hamilt
{

    /// @brief  Hxc part of A operator
    template<typename T = double>
    class OperatorLREXX : public Operator<T, psi::DEVICE_CPU>
    {
        using TA = int;
        static const size_t Ndim = 3;
        using TC = std::array<int, Ndim>;
        using TAC = std::pair<TA, TC>;

    public:
        OperatorLREXX(const int& nspin,
            const int& naos,
            const int& nocc,
            const int& nvirt,
            const UnitCell& ucell_in,
            const psi::Psi<T>* psi_ks_in,
            std::vector<elecstate::DensityMatrix<T, double>*>& DM_trans_in,
            // HContainer<double>* hR_in,
            Exx_LRI<T>* exx_lri_in,
            const K_Vectors& kv_in,
            Parallel_2D* pX_in,
            Parallel_2D* pc_in,
            Parallel_Orbitals* pmat_in)
            : nspin(nspin), naos(naos), nocc(nocc), nvirt(nvirt),
            psi_ks(psi_ks_in), DM_trans(DM_trans_in), exx_lri(exx_lri_in), kv(kv_in),
            pX(pX_in), pc(pc_in), pmat(pmat_in), ucell(ucell_in)
        {
            ModuleBase::TITLE("OperatorLREXX", "OperatorLREXX");
            this->nks = std::is_same<T, double>::value ? 1 : this->kv.kvec_d.size();
            this->nsk = std::is_same<T, double>::value ? nspin : nks;
            this->cal_type = calculation_type::lcao_exx;
            this->act_type = 2;
            this->is_first_node = false;

            // reduce psi_ks for later use
            this->psi_ks_full.resize(this->nsk, this->psi_ks->get_nbands(), this->naos);
            LR_Util::gather_2d_to_full(*this->pc, this->psi_ks->get_pointer(), this->psi_ks_full.get_pointer(), false, this->naos, this->psi_ks->get_nbands());

            // get cells in BvK supercell
            const TC period = RI_Util::get_Born_vonKarmen_period(kv_in);
            this->BvK_cells = RI_Util::get_Born_von_Karmen_cells(period);

            this->allocate_Ds_onebase();
            this->exx_lri->Hexxs.resize(this->nspin);
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
        const K_Vectors& kv;
        /// ground state wavefunction
        const psi::Psi<T>* psi_ks = nullptr;
        psi::Psi<T> psi_ks_full;

        /// transition density matrix 
        std::vector<elecstate::DensityMatrix<T, double>*>& DM_trans;

        /// density matrix of a certain (i, a, k), with full naos*naos size for each key
        /// D^{iak}_{\mu\nu}(k): 1/N_k * c^*_{ak,\mu} c_{ik,\nu}
        /// D^{iak}_{\mu\nu}(R): D^{iak}_{\mu\nu}(k)e^{-ikR}
        // elecstate::DensityMatrix<T, double>* DM_onebase;
        mutable std::vector<std::map<TA, std::map<TAC, RI::Tensor<T>>>> Ds_onebase;

        // cells in the Born von Karmen supercell (direct)
        std::vector<std::array<int, Ndim>> BvK_cells;

        /// transition hamiltonian in AO representation
        // hamilt::HContainer<double>* hR = nullptr;

        /// C, V tensors of RI, and LibRI interfaces
        /// gamma_only: T=double, Tpara of exx (equal to Tpara of Ds(R) ) is also double 
        ///.multi-k: T=complex<double>, Tpara of exx here must be complex, because Ds_onebase is complex
        /// so TR in DensityMatrix and Tdata in Exx_LRI are all equal to T
        Exx_LRI<T>* exx_lri = nullptr;

        const UnitCell& ucell;

        ///parallel info
        Parallel_2D* pc = nullptr;
        Parallel_2D* pX = nullptr;
        Parallel_Orbitals* pmat = nullptr;


        // allocate Ds_onebase
        void allocate_Ds_onebase();

        void cal_DM_onebase(const int io, const int iv, const int ik, const int is) const;


    };
}
#include "module_beyonddft/operator_casida/operator_lr_exx.hpp"