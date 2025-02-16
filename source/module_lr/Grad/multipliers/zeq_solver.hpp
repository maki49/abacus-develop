#pragma once
#include "zeq_solver.h"
#include "module_base/opt_CG.h"
#include "module_lr/utils/lr_util.h"
#include "module_lr/utils/lr_util_print.h"

namespace LR
{
    // inline void solve_Z_CG(double* const Z, const double* const R, const int& ld, const int& nstates,
    //     std::function<void(const double* const, double* const)> f_LZ)
    // Opt_CG's interfaces have no const qualifier for the pointer
    inline void solve_Z_CG(double* const Z, double* R, const int& ld, const int& nstates,
        std::function<void(const double* const, double* const)> f_LZ)
    {
        ModuleBase::TITLE("Z_vector", "solve_Z_CG");
        ModuleBase::timer::tick("Z_vector", "solve_Z_CG");
        const int maxiter = 100;
        double tol = 1e-6;
        double residual = 10.;
        const int size = nstates * ld;
        container::Tensor P = LR_Util::newTensor<double>({ nstates, ld });   // step length
        container::Tensor LP = LR_Util::newTensor<double>({ nstates, ld }); //f_LZ(P)
        ModuleBase::zeros(Z, size);

        ModuleBase::Opt_CG cg;
        cg.allocate(size);
        cg.init_b(R);
        int final_iter = 0;
        std::cout << "Start solving Z-vector equaiton with CG method ..." << std::endl;
        for (int iter = 0; iter < maxiter; ++iter)
        {
            if (residual < tol)
            {
                final_iter = iter;
                break;
            }
            cg.next_direct(LP.data<double>(), 0, P.data<double>());
            std::cout << "iter=" << iter << " residual=" << cg.get_residual() << std::endl;
            // std::cout << "Z=" << std::endl;
            // LR_Util::print_value(P.data<double>(), nstates, ld);
            // std::cout << "LZ before=" << std::endl;
            // LR_Util::print_value(LP.data<double>(), nstates, ld);
            f_LZ(P.data<double>(), LP.data<double>());  // L: act each operators on P
            // std::cout << "LZ=" << std::endl;
            // LR_Util::print_value(LP.data<double>(), nstates, ld);
            int ifPD = 0;   //???
            double step = cg.step_length(LP.data<double>(), P.data<double>(), ifPD);
            // for (int i = 0; i < size; ++i) Z[i] += step * P[i];
            std::transform(Z, Z + size, P.data<double>(), Z, [step](const double& z, const double& p) { return z + step * p; });
            residual = cg.get_residual();
        }
        std::cout << "Final Z-vector:" << std::endl;
        LR_Util::print_value(Z, nstates, ld);
        ModuleBase::timer::tick("Z_vector", "solve_Z_CG");
    }

    inline void solve_Z_CG(std::complex<double>* const Z, std::complex<double>* R, const int& ld, const int& nstates,
        std::function<void(const std::complex<double>* const, std::complex<double>* const)> f_LZ)
    {
        throw std::runtime_error("complex Z-vector solver is not implemented yet");
    }


    template<typename T, typename TGint>
    void Z_vector_equation(const T* const X,
        T* const Z,
        const std::string& xc_kernel,
        const int& nstates,
        const int& nspin,
        const int& naos,
        const std::vector<int>& nocc,
        const std::vector<int>& nvirt,
        const UnitCell& ucell,
        const std::vector<double>& orb_cutoff,
        const Grid_Driver& gd,
        const psi::Psi<T>& psi_ks,
        const ModuleBase::matrix& eig_ks,
#ifdef __EXX
        std::weak_ptr<Exx_LRI<T>> exx_lri,
        const double& exx_alpha,
#endif 
        TGint* gint,
        std::weak_ptr<PotHxcLR> pot,
        const K_Vectors& kv,
        const std::vector<Parallel_2D>& px,
        const Parallel_2D& pc,
        const Parallel_Orbitals& pmat,
        const std::string& spin_type)
    {
        ModuleBase::TITLE("Z_vector", "Z_vector");
        const int nk = kv.get_nks() / nspin;
        // 1. the right-hand side of Z-vector equation
        const int nloc_per_band = nk * px[0].get_local_size();
        container::Tensor R = LR_Util::newTensor<T>({ nstates, nloc_per_band });
        Z_vector_R<T> ops_R(xc_kernel, nspin, naos, nocc, nvirt,
            ucell, orb_cutoff, gd, psi_ks, eig_ks,
#ifdef __EXX
            exx_lri, exx_alpha,
#endif 
            gint, pot, kv, px, pc, pmat, spin_type);
        ModuleBase::timer::tick("Z_vector", "Z_vector_R");
        ops_R.hPsi(X, R.data<T>(), nk * px[0].get_local_size(), nstates);  // act each operators on X
        ModuleBase::timer::tick("Z_vector", "Z_vector_R");

        // 2. the left-hand side of Z-vector equation
        // Z-vector (need a init?)
        Z_vector_L<T> ops_L(xc_kernel, nspin, naos, nocc, nvirt,
            ucell, orb_cutoff, gd, psi_ks, eig_ks,
#ifdef __EXX
            exx_lri, exx_alpha,
#endif 
            gint, pot, kv, px, pc, pmat, spin_type);

        // 3. solve Z-vector equation
        std::cout << "The right side of the Z-vector equation:" << std::endl;
        LR_Util::print_value(R.data<T>(), nstates, nloc_per_band);
        solve_Z_CG(Z, R.data<T>(), nloc_per_band, nstates, std::bind(&HamiltLR<T>::hPsi, &ops_L, std::placeholders::_1, std::placeholders::_2, nloc_per_band, nstates));
    }
}