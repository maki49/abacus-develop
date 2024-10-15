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
        T* psi,
        const int& dim,
        const int& nband,
        ModuleBase::matrix& ekb,
        const std::string method,
        const bool hermitian)
    {
        ModuleBase::TITLE("HSolverLR", "solve");
        const std::vector<std::string> spin_types = { "singlet", "triplet" };
        // note: if not TDA, the eigenvalues will be complex
        // then we will need a new constructor of DiagoDavid

        // 1. allocate precondition and eigenvalue
        std::vector<Real> precondition(dim);
        std::vector<Real> eigenvalue(nband);   //nstates
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
            // copy eigenvectors
            std::memcpy(psi, Amat_full.data(), sizeof(T) * dim * nband);
        }
        else
        {
            // 3. set precondition and diagethr
            for (int i = 0; i < dim; ++i) { precondition[i] = static_cast<Real>(1.0); }

            const int david_maxiter = hsolver::DiagoIterAssist<T>::PW_DIAG_NMAX;

            auto hpsi_func = [&hm](T* psi_in, T* hpsi, const int ld_psi, const int nvec) {hm.hPsi(psi_in, hpsi, ld_psi, nvec);};
            auto spsi_func = [&hm](const T* psi_in, T* spsi, const int ld_psi, const int nband)
                { std::memcpy(spsi, psi_in, sizeof(T) * ld_psi * nband); };

            if (method == "dav")
            {
                // Allow 5 tries at most. If ntry > ntry_max = 5, exit diag loop.
                const int ntry_max = 5;
                // In non-self consistent calculation, do until totally converged. Else allow 5 eigenvecs to be NOT
                // converged.
                const int notconv_max = ("nscf" == PARAM.inp.calculation) ? 0 : 5;
                // do diag and add davidson iteration counts up to avg_iter
                hsolver::DiagoDavid<T> david(precondition.data(), nband, dim, PARAM.inp.pw_diag_ndim, PARAM.inp.use_paw, comm_info);
                hsolver::DiagoIterAssist<T>::avg_iter += static_cast<double>(david.diag(hpsi_func, spsi_func,
                    dim, psi, eigenvalue.data(), this->diag_ethr, david_maxiter, ntry_max, 0));
            }
            else if (method == "dav_subspace") //need refactor
            {
                hsolver::Diago_DavSubspace<T> dav_subspace(precondition,
                    nband,
                    dim,
                    PARAM.inp.pw_diag_ndim,
                    this->diag_ethr,
                    david_maxiter,
                    false, //always do the subspace diag (check the implementation)
                    comm_info);
                std::vector<double> ethr_band(nband, this->diag_ethr);
                hsolver::DiagoIterAssist<T>::avg_iter
                    += static_cast<double>(dav_subspace.diag(
                        hpsi_func, psi,
                        dim,
                        eigenvalue.data(),
                        ethr_band.data(),
                        false /*scf*/));
            }
            else {throw std::runtime_error("HSolverLR::solve: method not implemented");}
        }

        // 5. copy eigenvalue to pes
        for (int ist = 0;ist < nband;++ist) { ekb(ispin_solve, ist) = eigenvalue[ist]; }


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
            LR_Util::write_psi_bandfirst(psi, nband, nk, npairs, PARAM.globalv.global_out_dir + "Excitation_Amplitude_" + spin_types[ispin_solve], GlobalV::MY_RANK);
        }

        // normalization is already satisfied
        // std::cout << "check normalization of eigenvectors:" << std::endl;
        // for (int ist = 0;ist < nband;++ist)
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