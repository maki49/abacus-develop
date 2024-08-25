#pragma once
#include "zeq_solver.h"
#include "module_base/opt_CG.h"
#include "module_lr/utils/lr_util_print.h"

namespace LR
{
    inline void solve_Z_CG(psi::Psi<double>& Z, const psi::Psi<double>& R,
        std::function<void(const psi::Psi<double>&, psi::Psi<double>&)> f_LZ)
    {
        ModuleBase::TITLE("Z_vector", "solve_Z_CG");
        ModuleBase::timer::tick("Z_vector", "solve_Z_CG");
        const int maxiter = 100;
        double tol = 1e-6;
        double residual = 10.;

        psi::Psi<double> P(Z.get_nk(), Z.get_nbands(), Z.get_nbasis(), nullptr, false);   // step length
        psi::Psi<double> LP(Z.get_nk(), Z.get_nbands(), Z.get_nbasis(), nullptr, false); //f_LZ(P)
        assert(R.get_pointer() == &R(0, 0, 0)); // check reset
        assert(Z.get_pointer() == &Z(0, 0, 0)); // check reset
        Z.zero_out();

        ModuleBase::Opt_CG cg;
        cg.allocate(Z.get_nk() * Z.get_nbasis());
        cg.init_b(R.get_pointer());
        int final_iter = 0;
        for (int iter = 0; iter < maxiter; ++iter)
        {
            if (residual < tol)
            {
                final_iter = iter;
                break;
            }
            cg.next_direct(LP.get_pointer(), 0, P.get_pointer());
            f_LZ(P, LP);  // L: act each operators on P
            assert(P.get_pointer() == &P(0, 0, 0)); // check reset
            int ifPD = 0;
            double step = cg.step_length(LP.get_pointer(), P.get_pointer(), ifPD);
            for (int i = 0; i < Z.size(); ++i)
                Z.get_pointer()[i] += step * P.get_pointer()[i];
            residual = cg.get_residual();
        }
        for (int ib = 0;ib < Z.get_nbands();++ib)
            LR_Util::print_psi_bandfirst(Z, "Z-vector", ib);
        ModuleBase::timer::tick("Z_vector", "solve_Z_CG");
    }

    inline void solve_Z_CG(psi::Psi<std::complex<double>>& Z, const psi::Psi<std::complex<double>>& R,
        std::function<void(const psi::Psi<std::complex<double>>&, psi::Psi<std::complex<double>>&)> f_LZ)
    {
        throw std::runtime_error("complex Z-vector solver is not implemented yet");
    }

    template<typename T, typename TGint>
    psi::Psi<T> Z_vector(const psi::Psi<T>& X,
        const int& nstates,
        const int& nspin,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        const UnitCell& ucell,
        const std::vector<double>& orb_cutoff,
        Grid_Driver& gd,
        const psi::Psi<T>* psi_ks,
        const ModuleBase::matrix& eig_ks,
#ifdef __EXX
        Exx_LRI<T>* exx_lri,
#endif 
        TGint* gint,
        std::weak_ptr<PotHxcLR> pot,
        const K_Vectors& kv,
        Parallel_2D& px,
        Parallel_2D& pc,
        Parallel_Orbitals& pmat)
    {
        ModuleBase::TITLE("Z_vector", "Z_vector");
        // 1. the right-hand side of Z-vector equation
        psi::Psi<T> R(kv.get_nks(), nstates, px.get_local_size(), nullptr, false);
        Z_vector_R<T> ops_R(nspin, naos, nocc, nvirt,
            ucell, orb_cutoff, GlobalC::GridD, psi_ks, eig_ks,
#ifdef __EXX
            exx_lri,
#endif 
            gint, pot, kv, &px, &pc, &pmat);
        ModuleBase::timer::tick("Z_vector", "Z_vector_R");
        ops_R.ops->hPsi(X, R);  // act each operators on X
        ModuleBase::timer::tick("Z_vector", "Z_vector_R");

        // 2. the left-hand side of Z-vector equation
        // Z-vector (need a init?)
        Z_vector_L<T> ops_L(nspin, naos, nocc, nvirt,
            ucell, orb_cutoff, GlobalC::GridD, psi_ks, eig_ks,
#ifdef __EXX
            exx_lri,
#endif 
            gint, pot, kv, &px, &pc, &pmat);

        // 3. solve Z-vector equation
        psi::Psi<T> Z(kv.get_nks(), nstates, px.get_local_size(), nullptr, false);
        solve_Z_CG(Z, R,
            std::bind(static_cast<void(hamilt::Operator<T>::*)(const psi::Psi<T>&, psi::Psi<T>&) const>(&hamilt::Operator<T>::hPsi),
                *(ops_L.ops), std::placeholders::_1, std::placeholders::_2));
        return Z;
    }
}