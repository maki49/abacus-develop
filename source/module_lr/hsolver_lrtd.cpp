#include "hsolver_lrtd.h"
#include "module_parameter/parameter.h"
#include "module_hsolver/diago_david.h"
#include "module_hsolver/diago_dav_subspace.h"
#include "module_hsolver/diago_cg.h"
#include "module_lr/utils/lr_util.h"
#include "module_lr/utils/lr_util_print.h"

namespace LR
{
    inline double square(double x) { return x * x; };
    inline double square(std::complex<double> x) { return x.real() * x.real() + x.imag() * x.imag(); };
    template<typename T>
    inline void print_eigs(const std::vector<T>& eigs, const std::string& label = "", const double factor = 1.0)
    {
        std::cout << label << std::endl;
        for (auto& e : eigs) { std::cout << e * factor << " "; }
        std::cout << std::endl;
    }
    template<typename T>
    void HSolverLR<T>::solve(const HamiltLR<T>& hm,
        psi::Psi<T>& psi,
        ModuleBase::matrix& ekb,
        const std::string method,
        const bool hermitian)
    {
        ModuleBase::TITLE("HSolverLR", "solve");
        assert(psi.get_nk() == nk);
        const std::vector<std::string> spin_types = { "singlet", "triplet" };
        // note: if not TDA, the eigenvalues will be complex
        // then we will need a new constructor of DiagoDavid

        // 1. allocate precondition and eigenvalue
        std::vector<Real> precondition(psi.get_nk() * psi.get_nbasis());
        std::vector<Real> eigenvalue(psi.get_nbands());   //nstates
        // 2. select the method
#ifdef __MPI
        const hsolver::diag_comm_info comm_info = { POOL_WORLD, GlobalV::RANK_IN_POOL, GlobalV::NPROC_IN_POOL };
#else
        const hsolver::diag_comm_info comm_info = { GlobalV::RANK_IN_POOL, GlobalV::NPROC_IN_POOL };
#endif

        if (method == "lapack")
        {
            std::vector<T> Amat_full = hm.matrix();
            eigenvalue.resize(nk * npairs);
            if (hermitian) { LR_Util::diag_lapack(nk * npairs, Amat_full.data(), eigenvalue.data()); }
            else
            {
                std::vector<std::complex<double>> eig_complex(nk * npairs);
                LR_Util::diag_lapack_nh(nk * npairs, Amat_full.data(), eig_complex.data());
                print_eigs(eig_complex, "Right eigenvalues: of the non-Hermitian matrix: (Ry)");
                for (int i = 0; i < nk * npairs; i++) { eigenvalue[i] = eig_complex[i].real(); }
            }
            psi.fix_kb(0, 0);
            // copy eigenvectors
            for (int i = 0;i < psi.size();++i) { psi.get_pointer()[i] = Amat_full[i]; }
        }
        else
        {
            // 3. set precondition and diagethr
            for (int i = 0; i < psi.get_nk() * psi.get_nbasis(); ++i) { precondition[i] = static_cast<Real>(1.0); }

            // wrap band-first psi as k1-first psi_k1_dav
            psi::Psi<T> psi_k1_dav = LR_Util::bfirst_to_k1_wrapper(psi);
            assert(psi_k1_dav.get_nbands() == psi.get_nbands());
            assert(psi_k1_dav.get_nbasis() == psi.get_nbasis() * psi.get_nk());

            const int david_maxiter = hsolver::DiagoIterAssist<T>::PW_DIAG_NMAX;

            auto hpsi_func = [&hm](T* psi_in, T* hpsi, const int ld_psi, const int nvec) {hm.hPsi(psi_in, hpsi, ld_psi, nvec);};
            auto spsi_func = [&hm](const T* psi_in, T* spsi, const int ld_psi, const int nbands)
                { std::memcpy(spsi, psi_in, sizeof(T) * ld_psi * nbands); };

            if (method == "dav")
            {
                // Allow 5 tries at most. If ntry > ntry_max = 5, exit diag loop.
                const int ntry_max = 5;
                // In non-self consistent calculation, do until totally converged. Else allow 5 eigenvecs to be NOT
                // converged.
                const int notconv_max = ("nscf" == PARAM.inp.calculation) ? 0 : 5;
                // do diag and add davidson iteration counts up to avg_iter
                const int& dim = psi_k1_dav.get_nbasis();   //equals to leading dimension here
                const int& nband = psi_k1_dav.get_nbands();
                hsolver::DiagoDavid<T> david(precondition.data(), nband, dim, PARAM.inp.pw_diag_ndim, PARAM.inp.use_paw, comm_info);
                hsolver::DiagoIterAssist<T>::avg_iter += static_cast<double>(david.diag(hpsi_func, spsi_func,
                    dim, psi_k1_dav.get_pointer(), eigenvalue.data(), this->diag_ethr, david_maxiter, ntry_max, 0));
            }
            else if (method == "dav_subspace") //need refactor
            {
                hsolver::Diago_DavSubspace<T> dav_subspace(precondition,
                    psi_k1_dav.get_nbands(),
                    psi_k1_dav.get_nbasis(),
                    PARAM.inp.pw_diag_ndim,
                    this->diag_ethr,
                    david_maxiter,
                    false, //always do the subspace diag (check the implementation)
                    comm_info);
                std::vector<double> ethr_band(psi_k1_dav.get_nbands(), this->diag_ethr);
                hsolver::DiagoIterAssist<T>::avg_iter
                    += static_cast<double>(dav_subspace.diag(
                        hpsi_func, psi_k1_dav.get_pointer(),
                        psi_k1_dav.get_nbasis(),
                        eigenvalue.data(),
                        ethr_band.data(),
                        false /*scf*/));
            }
            else {throw std::runtime_error("HSolverLR::solve: method not implemented");}
        }

        // 5. copy eigenvalue to pes
        for (int ist = 0;ist < psi.get_nbands();++ist) { ekb(ispin_solve, ist) = eigenvalue[ist]; }


        // 6. output eigenvalues and eigenvectors
        print_eigs(eigenvalue, "eigenvalues: (Ry)");
        print_eigs(eigenvalue, "eigenvalues: (eV)", ModuleBase::Ry_to_eV);
        if (out_wfc_lr)
        {
            if (GlobalV::MY_RANK == 0)
            {
                std::ofstream ofs(PARAM.globalv.global_out_dir + "Excitation_Energy_" + spin_types[ispin_solve] + ".dat");
                ofs << std::setprecision(8) << std::scientific;
                for (auto& e : eigenvalue) {ofs << e << " ";}
                ofs.close();
            }
            LR_Util::write_psi_bandfirst(psi, PARAM.globalv.global_out_dir + "Excitation_Amplitude_" + spin_types[ispin_solve], GlobalV::MY_RANK);
        }

        // normalization is already satisfied
        // std::cout << "check normalization of eigenvectors:" << std::endl;
        // for (int ist = 0;ist < psi.get_nbands();++ist)
        // {
        //     double norm2 = 0;
        //     for (int ik = 0;ik < psi.get_nk();++ik)
        //     {
        //         for (int ib = 0;ib < psi.get_nbasis();++ib)
        //         {
        //             norm2 += square(psi(ist, ik, ib));
        //             // std::cout << "norm2_now=" << norm2 << std::endl;
        //         }
        //     }
        //     std::cout << "state " << ist << ", norm2=" << norm2 << std::endl;
        // }

        // output iters
        std::cout << "Average iterative diagonalization steps: " << hsolver::DiagoIterAssist<T>::avg_iter
            << " ; where current threshold is: " << hsolver::DiagoIterAssist<T>::PW_DIAG_THR << " . " << std::endl;
        // castmem_2d_2h_op()(cpu_ctx, cpu_ctx, pes->ekb.c, eigenvalues.data(), pes->ekb.nr * pes->ekb.nc);
    }
    template class HSolverLR<double>;
    template class HSolverLR<std::complex<double>>;
};