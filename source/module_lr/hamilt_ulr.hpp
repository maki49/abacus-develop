#pragma once
#include "module_hamilt_general/hamilt.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_lr/operator_casida/operator_lr_diag.h"
#include "module_lr/operator_casida/operator_lr_hxc.h"
#ifdef __EXX
#include "module_lr/operator_casida/operator_lr_exx.h"
#endif
namespace LR
{
    /// Unristricted TDDFT (TDA) for open-shell systems
    /// The A matrix is diveded by 4 blocks: uu, ud, du, dd
    template<typename T>
    class HamiltULR
    {
    public:
        template<typename TGint>
        HamiltULR(std::string& xc_kernel,
            const int& nspin,
            const int& naos,
            const std::vector<int>& nocc,   ///< {up, down}
            const std::vector<int>& nvirt,   ///< {up, down}
            const UnitCell& ucell_in,
            const std::vector<double>& orb_cutoff,
            Grid_Driver& gd_in,
            const psi::Psi<T>& psi_ks_in,
            const ModuleBase::matrix& eig_ks,
#ifdef __EXX
            std::weak_ptr<Exx_LRI<T>> exx_lri_in,
            const double& exx_alpha,
#endif 
            TGint* gint_in,
            std::vector<std::shared_ptr<PotHxcLR>>& pot_in,
            const K_Vectors& kv_in,
            const std::vector<Parallel_2D>& pX_in,   ///< {up, down}
            const Parallel_2D& pc_in,
            const Parallel_Orbitals& pmat_in) :nocc(nocc), nvirt(nvirt), pX(pX_in), nk(kv_in.get_nks() / nspin),
            nloc_per_band(nk* pX[0].get_local_size() + nk * pX[1].get_local_size())
        {
            ModuleBase::TITLE("HamiltULR", "HamiltULR");
            this->DM_trans.resize(1);
            this->DM_trans[0] = LR_Util::make_unique<elecstate::DensityMatrix<T, T>>(&kv_in, &pmat_in, nspin);

            // how to change the index of eig_ks and psi_ks_In?
            // modify the interface of opetators to support different left- and right- spin-pairs

            this->ops.resize(4);

            this->ops[0] = new OperatorLRDiag<T>(eig_ks.c, pX_in[0], nk, nocc[0], nvirt[0]);
            this->ops[3] = new OperatorLRDiag<T>(eig_ks.c + nk * (nocc[0] + nvirt[0]), pX_in[1], nk, nocc[1], nvirt[1]);

            auto newHxc = [&](const int& sl, const int& sr) { return new OperatorLRHxc<T>(nspin, naos, nocc, nvirt, psi_ks_in,
                this->DM_trans, gint_in, pot_in[sl], ucell_in, orb_cutoff, gd_in, kv_in, pX_in, pc_in, pmat_in, { sl,sr }); };
            this->ops[0]->add(newHxc(0, 0));
            this->ops[1] = newHxc(0, 1);
            this->ops[2] = newHxc(1, 0);
            this->ops[3]->add(newHxc(1, 1));

#ifdef __EXX
            if (xc_kernel == "hf" || xc_kernel == "hse")
            {
                std::vector<psi::Psi<T>> psi_ks_spin = { LR_Util::get_psi_spin(psi_ks_in, 0, nk), LR_Util::get_psi_spin(psi_ks_in, 1, nk) };
                for (int is : {0, 1})
                {
                    this->ops[(is << 1) + is]->add(new OperatorLREXX<T>(nspin, naos, nocc[is], nvirt[is], ucell_in, psi_ks_spin[is],
                        this->DM_trans, exx_lri_in, kv_in, pX_in[is], pc_in, pmat_in,
                        xc_kernel == "hf" ? 1.0 : exx_alpha));
                }
            }
#endif
        }
        ~HamiltULR()
        {
            for (auto& op : ops) { delete op; }
        }
        void hPsi(const T* psi_in, T* hpsi, const int ld_psi, const int& nband) const
        {
            ModuleBase::TITLE("HamiltULR", "hPsi");
            assert(ld_psi == this->nloc_per_band);
            const std::vector<int64_t> xdim_is = { nk * pX[0].get_local_size(), nk * pX[1].get_local_size() };
            /// band-wise act (also works for close-shell, but not efficient)
            for (int ib = 0;ib < nband;++ib)
            {
                const int offset_b = ib * ld_psi;
                for (int is_bj : {0, 1})
                {
                    const int offset = offset_b + is_bj * xdim_is[0];
                    for (int is_ai : {0, 1})
                    {
                        hamilt::Operator<T>* node(this->ops[(is_ai << 1) + is_bj]);
                        while (node != nullptr)
                        {
                            node->act(/*nband=*/1, xdim_is[is_bj], /*npol=*/1, psi_in + offset, hpsi + offset);
                            node = (hamilt::Operator<T>*)(node->next_op);
                        }
                    }
                }
            }
        }
        std::vector<T> matrix()const
        {
            ModuleBase::TITLE("HamiltULR", "matrix");
            const std::vector<int> npairs = { this->nocc[0] * this->nvirt[0], this->nocc[1] * this->nvirt[1] };
            const std::vector<int64_t> ldim_is = { nk * pX[0].get_local_size(), nk * pX[1].get_local_size() };
            const std::vector<int> gdim_is = { nk * npairs[0], nk * npairs[1] };
            const int global_size = this->nk * (npairs[0] + npairs[1]);
            std::vector<T> Amat_full(global_size * global_size);
            for (int is_bj : {0, 1})
            {
                const int no = this->nocc[is_bj];
                const int nv = this->nvirt[is_bj];
                const auto& px = this->pX[is_bj];
                const int loffset_bj = is_bj * ldim_is[0];
                const int goffset_bj = is_bj * gdim_is[0];
                for (int ik_bj = 0;ik_bj < nk;++ik_bj)
                {
                    for (int j = 0;j < no;++j)
                    {
                        for (int b = 0;b < nv;++b)
                        {
                            const int gcol = goffset_bj + ik_bj * npairs[is_bj] + j * nv + b;//global
                            std::vector<T> X_bj(this->nloc_per_band, T(0));
                            const int lj = px.global2local_col(j);
                            const int lb = px.global2local_row(b);
                            const int lcol = loffset_bj + ik_bj * px.get_local_size() + lj * px.get_row_size() + lb;//local
                            if (px.in_this_processor(b, j)) { X_bj[lcol] = T(1); }
                            std::vector<T> Aloc_col(this->nloc_per_band, T(0)); // a col of A matrix (local)
                            for (int is_ai : {0, 1})
                            {
                                hamilt::Operator<T>* node(this->ops[(is_ai << 1) + is_bj]);
                                while (node != nullptr)
                                {
                                    node->act(1, ldim_is[is_bj], /*npol=*/1, X_bj.data() + loffset_bj, Aloc_col.data() + loffset_bj);
                                    node = (hamilt::Operator<T>*)(node->next_op);
                                }
                                const int goffset_ai = is_ai * gdim_is[0];
#ifdef __MPI
                                const int loffset_ai = is_ai * ldim_is[0];
                                for (int ik_ai = 0;ik_ai < this->nk;++ik_ai)
                                {
                                    LR_Util::gather_2d_to_full(px, Aloc_col.data() + loffset_ai + ik_ai * px.get_local_size(),
                                        Amat_full.data() + gcol * global_size /*col, bj*/ + goffset_ai + ik_ai * npairs[is_ai]/*row, ai*/,
                                        false, nv, no);
                                }
#else
                                std::memcpy(Amat_full.data() + gcol * global_size + goffset_ai, Aloc_col.data() + goffset_ai, gdim_is[is_ai] * sizeof(T));
#endif
                            }
                        }
                    }
                }
            }
            std::cout << "Full A matrix:" << std::endl;
            LR_Util::print_value(Amat_full.data(), global_size, global_size);
            return Amat_full;
        }

    private:
        const std::vector<int>& nocc;
        const std::vector<int>& nvirt;

        const std::vector<Parallel_2D>& pX;


        const int nk = 1;
        const int nloc_per_band = 1;
        // const std::vector<int> npairs;

        /// size per state
        // const int bsize_X[2] = { nk * pX[0].get_local_size(), nk * pX[1].get_local_size() };
        // const int bsize_psi = bsize_X[0] + bsize_X[1];

        // void allocate_X_AX(const size_t& nband)
        // {
        //     for (int is : {0, 1})
        //     {
        //         X.emplace_back(1, nband, nk * pX[is].get_local_size());
        //         AX.emplace_back(1, nband, nk * pX[is].get_local_size());
        //     }
        // }
        // void copy_psi_to_spin2X(const T* psi_in, const size_t& nband)
        // {
        //     for (int is : {0, 1}) { for (ib = 0;ib < nband;++ib) { std::memcpy(&X[is](0, 0, 0) + ib * bsize_X[is], psi_in + ib * bsize_psi + is * bsize_X[0], bsize_X[is] * sizeof(T)); } }
        // }

        // void copy_spin2X_to_hpsi(T* hpsi, const size_t& nband)
        // {
        //     for (int is : {0, 1}) { for (int ib = 0;ib < nband;++ib) { std::memcpy(hpsi + ib * bsize_psi + is * bsize_X[0], &X[is](0, 0, 0) + ib * bsize_X[is], bsize_X[is] * sizeof(T)); } }
        // }


        /// 4 operator lists: uu, ud, du, dd
        std::vector<hamilt::Operator<T>*> ops;

        /// transition density matrix in AO representation
        /// Hxc only: size=1, calculate on the same address for each bands
        /// Hxc+Exx: size=nbands, store the result of each bands for common use
        std::vector<std::unique_ptr<elecstate::DensityMatrix<T, T>>> DM_trans;
    };
}