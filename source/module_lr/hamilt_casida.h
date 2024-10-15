#pragma once
#include "module_hamilt_general/hamilt.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_lr/operator_casida/operator_lr_diag.h"
#include "module_lr/operator_casida/operator_lr_hxc.h"
#include "module_basis/module_ao/parallel_orbitals.h"
#ifdef __EXX
#include "module_lr/operator_casida/operator_lr_exx.h"
#include "module_lr/ri_benchmark/operator_ri_hartree.h"
#include "module_ri/LRI_CV_Tools.h"
#endif
namespace LR
{
    template<typename T>
    class HamiltLR
    {
    public:
        template<typename TGint>
        HamiltLR(std::string& xc_kernel,
            const int& nspin,
            const int& naos,
            const std::vector<int>& nocc,
            const std::vector<int>& nvirt,
            const UnitCell& ucell_in,
            const std::vector<double>& orb_cutoff,
            Grid_Driver& gd_in,
            const psi::Psi<T>* psi_ks_in,
            const ModuleBase::matrix& eig_ks,
#ifdef __EXX
            std::weak_ptr<Exx_LRI<T>> exx_lri_in,
            const double& exx_alpha,
#endif 
            TGint* gint_in,
            std::weak_ptr<PotHxcLR> pot_in,
            const K_Vectors& kv_in,
            std::vector<Parallel_2D>& pX_in,
            Parallel_2D* pc_in,
            Parallel_Orbitals* pmat_in,
            const std::string& spin_type,
            const std::string& ri_hartree_benchmark = "none",
            const std::vector<int>& aims_nbasis = {}) : nspin(nspin), nocc(nocc), nvirt(nvirt), pX(pX_in),
            nk(kv_in.get_nks() / nspin), openshell(spin_type == "up" || spin_type == "down"),
            nloc_per_band(nk* (openshell ? pX_in[0].get_local_size() + pX_in[1].get_local_size() : pX_in[0].get_local_size()))
        {
            ModuleBase::TITLE("HamiltLR", "HamiltLR");
            if (ri_hartree_benchmark != "aims") { assert(aims_nbasis.empty()); }
            this->DM_trans.resize(1);
            this->DM_trans[0] = LR_Util::make_unique<elecstate::DensityMatrix<T, T>>(pmat_in, 1, kv_in.kvec_d, nk);
            // add the diag operator  (the first one)
            this->ops = new OperatorLRDiag<T>(eig_ks.c, &pX[0], nk, nocc[0], nvirt[0]);
            //add Hxc operator
#ifdef __EXX
            using TAC = std::pair<int, std::array<int, 3>>;
            using TLRI = std::map<int, std::map<TAC, RI::Tensor<T>>>;
            const std::string& dir = PARAM.globalv.global_readin_dir;
            TLRI Cs_read; 
            TLRI Vs_read; 
#ifdef __DEBUG
            // TLRI Vs_compare = LRI_CV_Tools::read_Vs_abf<T>(dir + "Vs");
            // LRI_CV_Tools::write_Vs_abf(Vs_read, "Vs_read_from_coulomb");
            // LRI_CV_Tools::write_Cs_ao(Cs_read, "Cs_ao_read"); // ensure Cs_ao is read correctly
            // assert(RI_Benchmark::compare_Vs(Vs_read, Vs_compare));
#endif
            if (ri_hartree_benchmark != "none")
            {
#ifdef __EXX
                if (spin_type == "singlet")
                {
                    if (ri_hartree_benchmark == "aims") 
                    { 
                        Cs_read = LRI_CV_Tools::read_Cs_ao<T>(dir + "Cs_data_0.txt");
                        Vs_read = RI_Benchmark::read_coulomb_mat_general<T>(dir + "coulomb_mat_0.txt", Cs_read); 
                    }
                    else if (ri_hartree_benchmark == "abacus")
                    {
                        Cs_read = LRI_CV_Tools::read_Cs_ao<T>(dir + "Cs");
                        Vs_read = LRI_CV_Tools::read_Vs_abf<T>(dir + "Vs");
                    }
                    if (!std::set<std::string>({ "rpa", "hf" }).count(xc_kernel)) { throw std::runtime_error("ri_hartree_benchmark is only supported for xc_kernel rpa and hf"); }
                    RI_Benchmark::OperatorRIHartree<T>* ri_hartree_op
                        = new RI_Benchmark::OperatorRIHartree<T>(ucell_in, naos, nocc[0], nvirt[0], *psi_ks_in,
                            Cs_read, Vs_read, ri_hartree_benchmark == "aims", aims_nbasis);
                    this->ops->add(ri_hartree_op);
                }
                else if (spin_type == "triplet") { std::cout << "f_Hxc based on grid integral is not needed." << std::endl; }
#else
                ModuleBase::WARNING_QUIT("ESolver_LR", "RI benchmark is only supported when compile with LibRI.");
#endif
            }
            else
#endif
            {
                OperatorLRHxc<T>* lr_hxc = new OperatorLRHxc<T>(nspin, naos, nocc, nvirt, psi_ks_in,
                    this->DM_trans, gint_in, pot_in, ucell_in, orb_cutoff, gd_in, kv_in, pX_in, pc_in, pmat_in);
                this->ops->add(lr_hxc);
            }
#ifdef __EXX
            if (xc_kernel == "hf" || xc_kernel == "hse")
            {   //add Exx operator
                if (ri_hartree_benchmark != "none" && spin_type == "singlet")
                {
                    exx_lri_in.lock()->reset_Cs(Cs_read);
                    exx_lri_in.lock()->reset_Vs(Vs_read);
                }
                // std::cout << "exx_alpha=" << exx_alpha << std::endl; // the default value of exx_alpha is 0.25 when dft_functional is pbe or hse
                hamilt::Operator<T>* lr_exx = new OperatorLREXX<T>(nspin, naos, nocc[0], nvirt[0], ucell_in, psi_ks_in,
                    this->DM_trans, exx_lri_in, kv_in, &pX_in[0], pc_in, pmat_in,
                    xc_kernel == "hf" ? 1.0 : exx_alpha, //alpha
                    ri_hartree_benchmark != "none"/*whether to cal_dm_trans first here*/,
                    aims_nbasis);
                this->ops->add(lr_exx);
            }
#endif
        }
        ~HamiltLR()
        {
            if (this->ops != nullptr)
            {
                delete this->ops;
            }
        };

        virtual std::vector<T> matrix()const;

        virtual void hPsi(const T* psi_in, T* hpsi, const int ld_psi, const int& nband) const
        {
            assert(ld_psi == this->nloc_per_band);
            hamilt::Operator<T>* node(this->ops);
            if (openshell)
            {
                /// band-wise act (also works for close-shell, but not efficient)
                for (int ib = 0;ib < nband;++ib)
                {
                    const int offset = ib * ld_psi;
                    while (node != nullptr)
                    {
                        node->act(1, ld_psi, /*npol=*/1, psi_in + offset, hpsi + offset);
                        node = (hamilt::Operator<T>*)(node->next_op);
                    }
                }
            }
            else
            {
                while (node != nullptr)
                {
                    node->act(nband, ld_psi, /*npol=*/1, psi_in, hpsi);
                    node = (hamilt::Operator<T>*)(node->next_op);
                }
            }
        }

    private:
        const std::vector<int>& nocc;
        const std::vector<int>& nvirt;
        const int nspin = 1;
        const int nk = 1;
        const bool openshell = false;
        const int nloc_per_band = 1;
        std::vector<Parallel_2D>& pX;
        T one()const;
        /// transition density matrix in AO representation
        /// Hxc only: size=1, calculate on the same address for each bands
        /// Hxc+Exx: size=nbands, store the result of each bands for common use
        std::vector<std::unique_ptr<elecstate::DensityMatrix<T, T>>> DM_trans;

        /// first node operator, add operations from each operators
        hamilt::Operator<T, base_device::DEVICE_CPU>* ops = nullptr;
    };
}
