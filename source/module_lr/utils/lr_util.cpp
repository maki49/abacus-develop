#include "module_base/constants.h"
#include "lr_util.h"
#include "module_base/lapack_connector.h"
#include "module_base/scalapack_connector.h"
#include "module_base/blacs_connector.h"
#include "module_hsolver/genelpa/elpa_solver.h"
#include "module_base/module_container/base/third_party/lapack.h"
namespace LR_Util
{
    /// =================PHYSICS====================
    int cal_nocc(int nelec) { return nelec / ModuleBase::DEGSPIN + nelec % static_cast<int>(ModuleBase::DEGSPIN); }

    std::pair<ModuleBase::matrix, std::vector<std::pair<int, int>>>
        set_ix_map_diagonal(bool mode, int nocc, int nvirt)
    {
        int npairs = nocc * nvirt;
        ModuleBase::matrix ioiv2ix(nocc, nvirt, true);
        std::vector<std::pair<int, int>> ix2ioiv(npairs);
        int io = nocc - 1, iv = 0;    //start：leftup
        if (mode == 0)  // leftdown->rightup
        {
            for (int ix = 0;ix < npairs - 1;++ix)
            {
                // 1. set value
                ioiv2ix(io, iv) = ix;
                ix2ioiv[ix] = std::make_pair(io, iv);
                // 2. move
                if (io == nocc - 1 || iv == nvirt - 1)    // rightup bound
                {
                    int io_next = std::max(nocc - iv - 1 - (nocc - io), 0);
                    iv -= (io - io_next) - 1;
                    io = io_next;
                }
                else { ++io;++iv; }//move rightup
            }
        }
        else    //rightup->leftdown
        {
            for (int ix = 0;ix < npairs - 1;++ix)
            {
                // 1. set value
                ioiv2ix(io, iv) = ix;
                ix2ioiv[ix] = std::make_pair(io, iv);
                // 2. move
                if (io == 0 || iv == 0)    // leftdown bound
                {
                    int iv_next = std::min(nocc - io + iv, nvirt - 1);
                    io += (iv_next - iv) - 1;
                    iv = iv_next;
                }
                else { --iv;--io; }//move leftdown
            }
        }
        //final set: rightdown
        assert(io == 0);
        assert(iv == nvirt - 1);
        ioiv2ix(io, iv) = npairs - 1;
        ix2ioiv[npairs - 1] = std::make_pair(io, iv);
        return std::make_pair(std::move(ioiv2ix), std::move(ix2ioiv));
    }

    /// =================ALGORITHM====================

#ifdef __MPI
    template<>
    void matsym<double>(const double* in, const int n, const Parallel_2D& pmat, double* out)
    {
        std::copy(in, in + pmat.get_local_size(), out);
        const double alpha = 0.5, beta = 0.5;
        const int i1 = 1;
        pdtran_(&n, &n, &alpha, in, &i1, &i1, pmat.desc, &beta, out, &i1, &i1, pmat.desc);
    }
    template<>
    void matsym<double>(double* inout, const int n, const Parallel_2D& pmat)
    {
        std::vector<double> tmp(pmat.get_local_size());
        std::copy(inout, inout + pmat.get_local_size(), tmp.begin());
        const double alpha = 0.5, beta = 0.5;
        const int i1 = 1;
        pdtran_(&n, &n, &alpha, tmp.data(), &i1, &i1, pmat.desc, &beta, inout, &i1, &i1, pmat.desc);
    }
    template<>
    void matsym<std::complex<double>>(const std::complex<double>* in, const int n, const Parallel_2D& pmat, std::complex<double>* out)
    {
        std::copy(in, in + pmat.get_local_size(), out);
        const std::complex<double> alpha(0.5, 0.0), beta(0.5, 0.0);
        const int i1 = 1;
        pztranc_(&n, &n, &alpha, in, &i1, &i1, pmat.desc, &beta, out, &i1, &i1, pmat.desc);
    }
    template<>
    void matsym<std::complex<double>>(std::complex<double>* inout, const int n, const Parallel_2D& pmat)
    {
        std::vector<std::complex<double>> tmp(pmat.get_local_size());
        std::copy(inout, inout + pmat.get_local_size(), tmp.begin());
        const std::complex<double> alpha(0.5, 0.0), beta(0.5, 0.0);
        const int i1 = 1;
        pztranc_(&n, &n, &alpha, tmp.data(), &i1, &i1, pmat.desc, &beta, inout, &i1, &i1, pmat.desc);
    }
#endif

    // for the first matrix in the commutator
    void setup_2d_division(Parallel_2D& pv, int nb, int gr, int gc)
    {
        ModuleBase::TITLE("LR_Util", "setup_2d_division");
#ifdef __MPI
        pv.init(gr, gc, nb, MPI_COMM_WORLD);
#else
        pv.set_serial(gr, gc);
#endif
    }

#ifdef __MPI
    // for the other matrices in the commutator other than the first one
    void setup_2d_division(Parallel_2D& pv, int nb, int gr, int gc, const int& blacs_ctxt_in)
    {
        ModuleBase::TITLE("LR_Util", "setup_2d_division");
        pv.set(gr, gc, nb, blacs_ctxt_in);
    }
#endif

    void diag_lapack_zheev(const int& n, double* mat, double* eig)
    {
        ModuleBase::TITLE("LR_Util", "diag_lapack_zheev<double>");
        int info = 0;
        char jobz = 'V', uplo = 'U';
        double work_tmp;
        const int minus_one = -1;
        dsyev_(&jobz, &uplo, &n, mat, &n, eig, &work_tmp, &minus_one, &info);		// get best lwork
        const int lwork = work_tmp;
        double* work2 = new double[lwork];
        dsyev_(&jobz, &uplo, &n, mat, &n, eig, work2, &lwork, &info);
        if (info) { std::cout << "ERROR: Lapack solver, info=" << info << std::endl; }
        delete[] work2;
    }

    void diag_lapack_zheev(const int& n, std::complex<double>* mat, double* eig)
    {
        ModuleBase::TITLE("LR_Util", "diag_lapack_zheev<complex<double>>");
        int lwork = 2 * n;
        std::complex<double>* work2 = new std::complex<double>[lwork];
        double* rwork = new double[3 * n - 2];
        int info = 0;
        char jobz = 'V', uplo = 'U';
        zheev_(&jobz, &uplo, &n, mat, &n, eig, work2, &lwork, rwork, &info);
        if (info) { std::cout << "ERROR: Lapack solver, info=" << info << std::endl; }
        delete[] rwork;
        delete[] work2;
    }

    void diag_lapack_zheevx(const int& n, double* mat, double* eig)
    {
        ModuleBase::TITLE("LR_Util", "diag_lapack_zheevx<double>");
        int info = 0;
        char jobz = 'V', range = 'A', uplo = 'U';
        const double vl = 0.0, vu = 0.0, abstol = 0.0;
        const int il = 0, iu = 0;
        int m = 0;
        const int ldz = n;
        double* z = new double[ldz * n];
        const int lwork = std::max(8 * n, 1);
        const int lrwork = std::max(7 * n, 1);
        const int liwork = std::max(5 * n, 1);
        double* work = new double[lwork];
        double* rwork = new double[lrwork];
        int* iwork = new int[liwork];
        int* ifail = new int[n];
        dsyevx_(&jobz, &range, &uplo, &n, mat, &n, &vl, &vu, &il, &iu, &abstol, &m, eig, z, &ldz,
                work, &lwork, rwork, iwork, ifail, &info);
        if (info) { std::cout << "ERROR: Lapack solver dsyevx, info=" << info << std::endl; }
        std::copy(z, z + ldz * n, mat);
        delete[] ifail;
        delete[] iwork;
        delete[] rwork;
        delete[] work;
        delete[] z;
    }

    void diag_lapack_zheevx(const int& n, std::complex<double>* mat, double* eig)
    {
        ModuleBase::TITLE("LR_Util", "diag_lapack_zheevx<complex<double>>");
        int info = 0;
        char jobz = 'V', range = 'A', uplo = 'U';
        const double vl = 0.0, vu = 0.0, abstol = 0.0;
        const int il = 0, iu = 0;
        int m = 0;
        const int ldz = n;
        std::complex<double>* z = new std::complex<double>[ldz * n];
        const int lwork = std::max(2 * n, 1);
        const int lrwork = std::max(7 * n, 1);
        const int liwork = std::max(5 * n, 1);
        std::complex<double>* work = new std::complex<double>[lwork];
        double* rwork = new double[lrwork];
        int* iwork = new int[liwork];
        int* ifail = new int[n];
        zheevx_(&jobz, &range, &uplo, &n, mat, &n, &vl, &vu, &il, &iu, &abstol, &m, eig, z, &ldz,
                work, &lwork, rwork, iwork, ifail, &info);
        if (info) { std::cout << "ERROR: Lapack solver zheevx, info=" << info << std::endl; }
        std::copy(z, z + ldz * n, mat);
        delete[] ifail;
        delete[] iwork;
        delete[] rwork;
        delete[] work;
        delete[] z;
    }

    extern "C" {
        void dsyevr_(const char* jobz, const char* range, const char* uplo, const int* n,
                     double* a, const int* lda, const double* vl, const double* vu, const int* il, const int* iu,
                     const double* abstol, int* m, double* w, double* z, const int* ldz, int* isuppz,
                     double* work, const int* lwork, int* iwork, const int* liwork, int* info);
        void zheevr_(const char* jobz, const char* range, const char* uplo, const int* n,
                     std::complex<double>* a, const int* lda, const double* vl, const double* vu, const int* il, const int* iu,
                     const double* abstol, int* m, double* w, std::complex<double>* z, const int* ldz, int* isuppz,
                     std::complex<double>* work, const int* lwork, double* rwork, const int* lrwork, int* iwork, const int* liwork, int* info);
    }

    void diag_lapack_zheevr(const int& n, double* mat, double* eig)
    {
        ModuleBase::TITLE("LR_Util", "diag_lapack_zheevr<double>");
        int info = 0;
        char jobz = 'V', range = 'A', uplo = 'U';
        const double vl = 0.0, vu = 0.0, abstol = 0.0;
        const int il = 0, iu = 0;
        int m = 0;
        const int ldz = n;
        double* z = new double[ldz * n];
        int* isuppz = new int[2 * n];
        const int lwork = std::max(26 * n, 1);
        const int liwork = std::max(10 * n, 1);
        double* work = new double[lwork];
        int* iwork = new int[liwork];
        dsyevr_(&jobz, &range, &uplo, &n, mat, &n, &vl, &vu, &il, &iu, &abstol, &m, eig, z, &ldz, isuppz,
                work, &lwork, iwork, &liwork, &info);
        if (info) { std::cout << "ERROR: Lapack solver dsyevr, info=" << info << std::endl; }
        std::copy(z, z + ldz * n, mat);
        delete[] iwork;
        delete[] work;
        delete[] isuppz;
        delete[] z;
    }

    void diag_lapack_zheevr(const int& n, std::complex<double>* mat, double* eig)
    {
        ModuleBase::TITLE("LR_Util", "diag_lapack_zheevr<complex<double>>");
        int info = 0;
        char jobz = 'V', range = 'A', uplo = 'U';
        const double vl = 0.0, vu = 0.0, abstol = 0.0;
        const int il = 0, iu = 0;
        int m = 0;
        const int ldz = n;
        std::complex<double>* z = new std::complex<double>[ldz * n];
        int* isuppz = new int[2 * n];
        const int lwork = std::max(2 * n, 1);
        const int lrwork = std::max(24 * n, 1);
        const int liwork = std::max(10 * n, 1);
        std::complex<double>* work = new std::complex<double>[lwork];
        double* rwork = new double[lrwork];
        int* iwork = new int[liwork];
        zheevr_(&jobz, &range, &uplo, &n, mat, &n, &vl, &vu, &il, &iu, &abstol, &m, eig, z, &ldz, isuppz,
                work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
        if (info) { std::cout << "ERROR: Lapack solver zheevr, info=" << info << std::endl; }
        std::copy(z, z + ldz * n, mat);
        delete[] iwork;
        delete[] rwork;
        delete[] work;
        delete[] isuppz;
        delete[] z;
    }

    void diag_elpa(const int& n, double* mat, double* eig)
    {
        ModuleBase::TITLE("LR_Util", "diag_elpa<double>");
        #ifdef __MPI
            int ctxt = Csys2blacs_handle(MPI_COMM_WORLD);
            char layout = 'R';
            Cblacs_gridinit(&ctxt, &layout, 1, 1);
            int desc[9];
            int info = 0;
            const int m = n, nb = std::max(1, std::min(n, 128));
            const int rsrc = 0, csrc = 0;
            const int lda = n;
            descinit_(desc, &m, &m, &nb, &nb, &rsrc, &csrc, &ctxt, &lda, &info);
            if (info) { std::cout << "ERROR: descinit in diag_elpa<double>, info=" << info << std::endl; }
            ELPA_Solver es(true, MPI_COMM_WORLD, n, n, n, desc);
            std::vector<double> vec(n * n);
            es.eigenvector(mat, eig, vec.data());
            es.exit();
            std::copy(vec.begin(), vec.end(), mat);
            Cblacs_gridexit(&ctxt);
        #else
            ELPA_Solver es(true, MPI_COMM_WORLD, n, n, n, nullptr);
            std::vector<double> vec(n * n);
            es.eigenvector(mat, eig, vec.data());
            es.exit();
            std::copy(vec.begin(), vec.end(), mat);
        #endif
    }

    void diag_elpa(const int& n, std::complex<double>* mat, double* eig)
    {
        ModuleBase::TITLE("LR_Util", "diag_elpa<complex<double>>");
        #ifdef __MPI
            int ctxt = Csys2blacs_handle(MPI_COMM_WORLD);
            char layout = 'R';
            Cblacs_gridinit(&ctxt, &layout, 1, 1);
            int desc[9];
            int info = 0;
            const int m = n, nb = std::max(1, std::min(n, 128));
            const int rsrc = 0, csrc = 0;
            const int lda = n;
            descinit_(desc, &m, &m, &nb, &nb, &rsrc, &csrc, &ctxt, &lda, &info);
            if (info) { std::cout << "ERROR: descinit in diag_elpa<complex<double>>, info=" << info << std::endl; }
            ELPA_Solver es(false, MPI_COMM_WORLD, n, n, n, desc);
            std::vector<std::complex<double>> vec(n * n);
            es.eigenvector(mat, eig, vec.data());
            es.exit();
            std::copy(vec.begin(), vec.end(), mat);
            Cblacs_gridexit(&ctxt);
        #else
            ELPA_Solver es(false, MPI_COMM_WORLD, n, n, n, nullptr);
            std::vector<std::complex<double>> vec(n * n);
            es.eigenvector(mat, eig, vec.data());
            es.exit();
            std::copy(vec.begin(), vec.end(), mat);
        #endif
    }
    void diag_lapack_nh(const int& n, double* mat, std::complex<double>* eig)
    {
        ModuleBase::TITLE("LR_Util", "diag_lapack_nh<double>");
        int info = 0;
        char jobvl = 'N', jobvr = 'V';  //calculate right eigenvectors
        double work_tmp;
        const int minus_one = -1;
        std::vector<double> eig_real(n);
        std::vector<double> eig_imag(n);
        const int ldvl = 1, ldvr = n;
        std::vector<double> vl(ldvl * n), vr(ldvr * n);
        dgeev_(&jobvl, &jobvr, &n, mat, &n, eig_real.data(), eig_imag.data(),
            vl.data(), &ldvl, vr.data(), &ldvr, &work_tmp, &minus_one /*lwork*/, &info);		// get best lwork
        const int lwork = work_tmp;
        std::vector<double> work2(lwork);
        dgeev_(&jobvl, &jobvr, &n, mat, &n, eig_real.data(), eig_imag.data(),
            vl.data(), &ldvl, vr.data(), &ldvr, work2.data(), &lwork, &info);
        if (info) { std::cout << "ERROR: Lapack solver dgeev, info=" << info << std::endl; }
        for (int i = 0;i < n;++i) { eig[i] = std::complex<double>(eig_real[i], eig_imag[i]); }
    }

    void diag_lapack_nh(const int& n, std::complex<double>* mat, std::complex<double>* eig)
    {
        ModuleBase::TITLE("LR_Util", "diag_lapack_nh<complex<double>>");
        int lwork = 2 * n;
        std::vector<std::complex<double>> work2(lwork);
        std::vector<double> rwork(3 * n - 2);
        int info = 0;
        char jobvl = 'N', jobvr = 'V';
        const int ldvl = 1, ldvr = n;
        std::vector<std::complex<double>> vl(ldvl * n), vr(ldvr * n);
        zgeev_(&jobvl, &jobvr, &n, mat, &n, eig,
            vl.data(), &ldvl, vr.data(), &ldvr, work2.data(), &lwork, rwork.data(), &info);
        if (info) { std::cout << "ERROR: Lapack solver zgeev, info=" << info << std::endl; }
    }

    std::string tolower(const std::string& str)
    {
        std::string str_lower = str;
        std::transform(str_lower.begin(), str_lower.end(), str_lower.begin(), ::tolower);
        return str_lower;
    }
    std::string toupper(const std::string& str)
    {
        std::string str_upper = str;
        std::transform(str_upper.begin(), str_upper.end(), str_upper.begin(), ::toupper);
        return str_upper;
    }
}
