#include "hsolver_lrtd.h"
#include "hamilt_casida.h"
#include "hamilt_ulr.h"
#include "source_io/module_parameter/parameter.h"
#include "source_hsolver/diag_comm_info.h"
#include "source_hsolver/diago_david.h"
#include "source_hsolver/diago_dav_subspace.h"
#include "source_hsolver/diago_cg.h"
#include "source_hsolver/diago_iter_assist.h"
#include "source_lcao/module_lr/lr_preconditioner.h"
#include "source_lcao/module_lr/utils/lr_util.h"
#include "source_lcao/module_lr/utils/lr_util_print.h"
#include "source_base/module_container/ATen/core/tensor_map.h"
#include "source_base/parallel_comm.h"
#include "source_base/parallel_device.h"

namespace LR
{


    namespace HSolver
    {
        /// The LR Hamiltonians (HamiltLR, HamiltULR) are not hamilt::Hamilt, so
        /// they get their own hsolver::HSOperator view. S is the identity.
        template <typename T, typename THamilt>
        class LRHSOperator : public hsolver::HSOperator<T>
        {
          public:
            explicit LRHSOperator(const THamilt& hm) : hm_(hm) {}
            void update_k(const int ik) override {}
            void hpsi(const T* x, T* hx, const int ld, const int nvec) const override
            {
                hm_.hPsi(x, hx, ld, nvec);
                ++operator_calls_;
                applied_vectors_ += nvec;
            }
            void spsi(const T* x, T* sx, const int ld, const int nvec) const override
            {
                std::memcpy(sx, x, sizeof(T) * static_cast<size_t>(ld) * static_cast<size_t>(nvec));
            }
            int operator_calls() const { return operator_calls_; }
            int applied_vectors() const { return applied_vectors_; }
          private:
            const THamilt& hm_;
            mutable int operator_calls_ = 0;
            mutable int applied_vectors_ = 0;
        };

        /// Recompute ||A X_n - omega_n X_n|| after Davidson, independently of its
        /// eigenvalue-change stopping test. Counts exclude this one verification block.
        template<typename T, typename THamilt>
        void report_davidson_residuals(const LRHSOperator<T, THamilt>& op, const T* vectors,
                                       const int dim, const int roots, const Real<T>* eigenvalues,
                                       const int iterations, const hsolver::diag_comm_info& comm)
        {
            ModuleBase::timer::start("HSolverLR", "report_davidson_residuals");
            const int operator_calls = op.operator_calls();
            const int applied_vectors = op.applied_vectors();
            const std::size_t size = static_cast<std::size_t>(dim) * roots;
            std::vector<T> image(size);
            op.hpsi(vectors, image.data(), dim, roots);
            std::vector<double> squared_norms(roots, 0.0);
            for (int n = 0; n < roots; ++n)
            {
                for (int j = 0; j < dim; ++j)
                {
                    const std::size_t index = static_cast<std::size_t>(n) * dim + j;
                    const T residual = image[index] - eigenvalues[n] * vectors[index];
                    squared_norms[n] += std::norm(residual);
                }
            }
#ifdef __MPI
            if (comm.nproc > 1) { Parallel_Common::reduce_data(squared_norms.data(), roots, comm.comm); }
#endif
            const auto precision = std::cout.precision();
            std::cout << std::setprecision(8) << "LR Davidson: iterations=" << iterations
                      << " operator calls=" << operator_calls << " applied vectors=" << applied_vectors << '\n';
            for (int n = 0; n < roots; ++n)
            {
                const double residual = std::sqrt(squared_norms[n]);
                std::cout << "LR Davidson root " << n << " true residual=" << residual << " Ry\n";
            }
            std::cout.precision(precision);
            ModuleBase::timer::end("HSolverLR", "report_davidson_residuals");
        }

        template<typename T>
        inline void print_eigs(const std::vector<T>& eigs, const std::string& label = "", const double factor = 1.0)
        {
            std::streamsize old = std::cout.precision();
            std::cout << label << std::setprecision(8) << std::endl;
            for (auto& e : eigs) { std::cout << e * factor << " "; }
            std::cout << std::endl;
            std::cout.precision(old);
        }

        /// eigensolver for common Hamilt
        template<typename T, typename THamilt>
        void solve(const THamilt& hm,
            T* psi,
            const int& dim, ///< local leading dimension (or nbasis)
            const int& nband,   ///< nstates in LR-TDDFT, not (nocc+nvirt)
            const int& nk,
            const std::vector<int>& nocc,
            const std::vector<int>& nvirt,
            const std::vector<Parallel_2D>& pX,
            double* eig,
            const std::string method,
            const Real<T>& diag_ethr, ///< threshold for diagonalization
            const std::vector<Real<T>>& precondition,
            const bool hermitian)
        {
            ModuleBase::TITLE("HSolverLR", "solve");
            ModuleBase::timer::start("HSolverLR", "solve");
            const std::vector<std::string> spin_types = { "singlet", "triplet" };
            // note: if not TDA, the eigenvalues will be complex
            // then we will need a new constructor of DiagoDavid

            // 1. allocate eigenvalue
            std::vector<Real<T>> eigenvalue(nband);   //nstates
            // 2. select the method
#ifdef __MPI
            const hsolver::diag_comm_info comm_info = { POOL_WORLD, GlobalV::RANK_IN_POOL, GlobalV::NPROC_IN_POOL };
#else
            const hsolver::diag_comm_info comm_info = { GlobalV::RANK_IN_POOL, GlobalV::NPROC_IN_POOL };
#endif

            if (method == "lapack")
            {
                std::vector<T> Amat_full = hm.matrix();
                const int gdim = std::sqrt(Amat_full.size());
                eigenvalue.resize(gdim);
                if (hermitian) { LR_Util::diag_lapack(gdim, Amat_full.data(), eigenvalue.data()); }
                else
                {
                    std::vector<std::complex<double>> eig_complex(gdim);
                    LR_Util::diag_lapack_nh(gdim, Amat_full.data(), eig_complex.data());
                    print_eigs(eig_complex, "Right eigenvalues: of the non-Hermitian matrix: (Ry)");
                    for (int i = 0; i < gdim; i++) { eigenvalue[i] = eig_complex[i].real(); }
                }
                bool openshell = std::is_same<THamilt, HamiltULR<T>>::value;
                // copy eigenvectors
#ifdef __MPI
                LR_Util::global2local_X(psi, Amat_full.data(), nband, nk,
                                        nocc, nvirt, pX, openshell);
#else
                std::memcpy(psi, Amat_full.data(), sizeof(T) * nband * gdim);
#endif
            }
            else
            {
                // 3. set maxiter and the operator
                const int maxiter = hsolver::DiagoIterAssist<T>::PW_DIAG_NMAX;
                const LRHSOperator<T, THamilt> op(hm);
                const int davidson_dimension = PARAM.inp.pw_diag_ndim;

                if (method == "dav")
                {
                    // Allow 5 tries at most. If ntry > ntry_max = 5, exit diag loop.
                    const int ntry_max = 5;
                    // do diag and add davidson iteration counts up to avg_iter
                    hsolver::DiagoDavid<T> david(precondition.data(),
                                                 nband,
                                                 dim,
                                                 davidson_dimension,
                                                 comm_info);
                    std::vector<double> ethr_band(nband, diag_ethr);
                    hsolver::DiagoIterAssist<T>::avg_iter += static_cast<double>(david.diag(op,
                        dim, psi, eigenvalue.data(), ethr_band, maxiter, ntry_max, 0));
                }
                else if (method == "dav_subspace")
                {
                    hsolver::Diago_DavSubspace<T> dav_subspace(precondition,
                        nband,
                        dim,
                        davidson_dimension,
                        diag_ethr,
                        maxiter,
                        comm_info,
                        PARAM.inp.diag_subspace,
                        PARAM.inp.nb2d);
                    std::vector<double> ethr_band(nband, diag_ethr);
                    // A - omega I is approximated by the signed orbital gaps.
                    // Cap only near-resonant denominators (1e-4 Ry = 1.36 meV).
                    const double denominator_floor = 1e-4;
                    const typename hsolver::Diago_DavSubspace<T>::Preconditioner inverse_gaps =
                        [&precondition, denominator_floor](T* residuals, const Real<T>* shifts,
                                                           int local_dim, int nvec)
                    {
                        apply_orbital_gap_preconditioner(residuals, shifts, local_dim, nvec,
                                                         precondition, denominator_floor);
                    };
                    std::cout << "LR Davidson preconditioner=signed-orbital-gap floor="
                              << denominator_floor << " Ry\n";
                    const int iterations = dav_subspace.diag(op, psi, dim, eigenvalue.data(), ethr_band, false,
                                                             inverse_gaps);
                    hsolver::DiagoIterAssist<T>::avg_iter += iterations;
                    report_davidson_residuals(op, psi, dim, nband, eigenvalue.data(), iterations, comm_info);
                }
                else if (method == "cg")
                {
                    // the subspace rotation of DiagoCG now works on any HSOperator, so it could be
                    // switched on here; it is kept off to leave the LR results unchanged
                    hsolver::DiagoCG<T> cg("lcao", "nscf", false, comm_info, diag_ethr, maxiter);

                    std::vector<double> ethr_band(nband, diag_ethr);
                    cg.diag(op,
                            dim,
                            nband,
                            dim,
                            psi,
                            eigenvalue.data(),
                            ethr_band,
                            precondition.data());
                }
                else { throw std::runtime_error("HSolverLR::solve: method not implemented"); }
            }

            // 5. copy eigenvalues
            for (int ist = 0;ist < nband;++ist) { eig[ist] = eigenvalue[ist]; }

            // 6. output eigenvalues and eigenvectors
            print_eigs(eigenvalue, "eigenvalues: (Ry)");
            print_eigs(eigenvalue, "eigenvalues: (eV)", ModuleBase::Ry_to_eV);

            // normalization is already satisfied
            // std::cout << "check normalization of eigenvectors:" << std::endl;
            // for (int ist = 0;ist < nband;++ist)
            // {
            //     double norm2 = 0;
            //     for (int ik = 0;ik < psi.get_nk();++ik)
            //     {
            //         for (int ib = 0;ib < psi.get_nbasis();++ib)
            //         {
            //             norm2 += std::norm(psi(ist, ik, ib));
            //             // std::cout << "norm2_now=" << norm2 << std::endl;
            //         }
            //     }
            //     std::cout << "state " << ist << ", norm2=" << norm2 << std::endl;
            // }

            // output iters
            std::cout << " Average iterative diagonalization steps: " << hsolver::DiagoIterAssist<T>::avg_iter
                << "; current threshold: " << diag_ethr << std::endl;
            ModuleBase::timer::end("HSolverLR", "solve");
        }
template void solve<double, HamiltLR<double>>(const HamiltLR<double>&, double*, const int&, const int&, const int&, const std::vector<int>&, const std::vector<int>&, const std::vector<Parallel_2D>&, double*, const std::string, const double&, const std::vector<double>&, const bool);
template void solve<double, HamiltULR<double>>(const HamiltULR<double>&, double*, const int&, const int&, const int&, const std::vector<int>&, const std::vector<int>&, const std::vector<Parallel_2D>&, double*, const std::string, const double&, const std::vector<double>&, const bool);
template void solve<std::complex<double>, HamiltLR<std::complex<double>>>(const HamiltLR<std::complex<double>>&, std::complex<double>*, const int&, const int&, const int&, const std::vector<int>&, const std::vector<int>&, const std::vector<Parallel_2D>&, double*, const std::string, const double&, const std::vector<double>&, const bool);
template void solve<std::complex<double>, HamiltULR<std::complex<double>>>(const HamiltULR<std::complex<double>>&, std::complex<double>*, const int&, const int&, const int&, const std::vector<int>&, const std::vector<int>&, const std::vector<Parallel_2D>&, double*, const std::string, const double&, const std::vector<double>&, const bool);

    }
}
