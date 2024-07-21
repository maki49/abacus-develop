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
    class HamiltCasidaLR : public hamilt::Hamilt<T, base_device::DEVICE_CPU>
    {
    public:
        template<typename TGint>
        HamiltCasidaLR(std::string& xc_kernel,
            const int& nspin,
            const int& naos,
            const int& nocc,
            const int& nvirt,
            const UnitCell& ucell_in,
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
            Parallel_2D* pX_in,
            Parallel_2D* pc_in,
            Parallel_Orbitals* pmat_in,
            const bool ri_hartree = false) : nocc(nocc), nvirt(nvirt), pX(pX_in), nk(kv_in.get_nks() / nspin)
        {
            ModuleBase::TITLE("HamiltCasidaLR", "HamiltCasidaLR");
            this->classname = "HamiltCasidaLR";
            this->DM_trans.resize(1);
            this->DM_trans[0] = LR_Util::make_unique<elecstate::DensityMatrix<T, T>>(&kv_in, pmat_in, nspin);
            // add the diag operator  (the first one)
            this->ops = new OperatorLRDiag<T>(eig_ks, pX_in, nk, nocc, nvirt);
            //add Hxc operator
#ifdef __EXX
            using TAC = std::pair<int, std::array<int, 3>>;
            using TLRI = std::map<int, std::map<TAC, RI::Tensor<T>>>;
            const std::string& dir = GlobalV::global_out_dir;
            const TLRI Cs_read = LRI_CV_Tools::read_Cs_ao<T>(dir + "Cs");
            const TLRI Vs_read = RI_Benchmark::read_coulomb_mat_general<T>(dir + "coulomb_mat_0.txt", Cs_read);
#ifdef __DEBUG
            // TLRI Vs_compare = LRI_CV_Tools::read_Vs_abf<T>(dir + "Vs");
            // LRI_CV_Tools::write_Vs_abf(Vs_read, "Vs_read_from_coulomb");
            // LRI_CV_Tools::write_Cs_ao(Cs_read, "Cs_ao_read"); // ensure Cs_ao is read correctly
            // assert(RI_Benchmark::compare_Vs(Vs_read, Vs_compare));
#endif
            if (ri_hartree)
            {
                assert(xc_kernel == "rpa" || xc_kernel == "hf");
                RI_Benchmark::OperatorRIHartree<T>* ri_hartree
                    = new RI_Benchmark::OperatorRIHartree<T>(ucell_in, naos, nocc, nvirt, *psi_ks_in,
                        Cs_read, Vs_read);
                this->ops->add(ri_hartree);
            }
            else
#endif
            {
                OperatorLRHxc<T>* lr_hxc = new OperatorLRHxc<T>(nspin, naos, nocc, nvirt, psi_ks_in,
                    this->DM_trans, gint_in, pot_in, ucell_in, gd_in, kv_in, pX_in, pc_in, pmat_in);
                this->ops->add(lr_hxc);
            }
#ifdef __EXX
            if (xc_kernel == "hf" || xc_kernel == "hse")
            {   //add Exx operator
                if (ri_hartree)
                {
                    exx_lri_in.lock()->reset_Cs(Cs_read);
                    exx_lri_in.lock()->reset_Vs(Vs_read);
                }
                // std::cout << "exx_alpha=" << exx_alpha << std::endl; // the default value of exx_alpha is 0.25 when dft_functional is pbe or hse
                hamilt::Operator<T>* lr_exx = new OperatorLREXX<T>(nspin, naos, nocc, nvirt, ucell_in, psi_ks_in,
                    this->DM_trans, exx_lri_in, kv_in, pX_in, pc_in, pmat_in, xc_kernel == "hf" ? 1.0 : exx_alpha);
                this->ops->add(lr_exx);
            }
#endif
        }
        ~HamiltCasidaLR()
        {
            if (this->ops != nullptr)
            {
                delete this->ops;
            }
        };

        virtual std::vector<T> matrix() override;

    private:
        int nocc;
        int nvirt;
        int nk;
        Parallel_2D* pX = nullptr;
        T one();
        /// transition density matrix in AO representation
        /// Hxc only: size=1, calculate on the same address for each bands
        /// Hxc+Exx: size=nbands, store the result of each bands for common use
        std::vector<std::unique_ptr<elecstate::DensityMatrix<T, T>>> DM_trans;
    };
}