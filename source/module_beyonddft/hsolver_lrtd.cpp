#include "hsolver_lrtd.h"
#include "module_hsolver/diago_david.h"
#include "module_hsolver/diago_cg.h"
#include "module_beyonddft/utils/lr_util.h"

namespace hsolver
{
    inline double square(double x) { return x * x; };
    inline double square(std::complex<double> x) { return x.real() * x.real() + x.imag() * x.imag(); };
    template<typename T, typename Device>
    void HSolverLR<T, Device>::solve(hamilt::Hamilt<T, Device>* pHamilt,
        psi::Psi<T, Device>& psi,
        elecstate::ElecState* pes,
        const std::string method_in,
        const bool skip_charge)
    {
        // note: if not TDA, the eigenvalues will be complex
        // then we will need a new constructor of DiagoDavid

        // 1. allocate precondition and eigenvalue
        std::vector<Real> precondition(psi.get_nk() * psi.get_nbasis());
        std::vector<Real> eigenvalue(psi.get_nbands());   //nstataes
        // 2. select the method
        this->method = method_in;
        if (this->method == "dav")
        {
            DiagoDavid<T>::PW_DIAG_NDIM = GlobalV::PW_DIAG_NDIM;
            this->pdiagh = new DiagoDavid<T, Device>(precondition.data());      //waiting for complex<T> removement
            this->pdiagh->method = this->method;
        }
        else if (this->method == "cg")
        {
            this->pdiagh = new DiagoCG<T, Device>(precondition.data());
            this->pdiagh->method = this->method;
        }
        else if (this->method == "lapack")
        {
            std::vector<T> Amat_full = pHamilt->matrix();
            eigenvalue.resize(npairs);
            LR_Util::diag_lapack(npairs, Amat_full.data(), eigenvalue.data());
            psi.fix_kb(0, 0);
            // copy eigenvectors
            for (int i = 0;i < psi.size();++i) psi.get_pointer()[i] = Amat_full[i];
        }
        else
            throw std::runtime_error("HSolverLR::solve: method not implemented");

        if (this->method != "lapack")
        {
            // 3. set precondition and diagethr
            for (int i = 0;i < psi.get_nk() * psi.get_nbasis();++i)precondition[i] = static_cast<Real>(1.0);
            // T ethr = this->set_diagether(1, 1, static_cast<T>(1e-2));
            std::cout << "ethr: " << this->diag_ethr << std::endl;
            // 4. solve Hamiltonian
            this->pdiagh->diag(pHamilt, psi, eigenvalue.data());
        }

        // 5. copy eigenvalue to pes
        pes->ekb.create(1, psi.get_nbands());
        for (int ist = 0;ist < psi.get_nbands();++ist) pes->ekb(0, ist) = eigenvalue[ist];


        // 6. output eigenvalues and eigenvectors
        std::cout << "eigenvalues:" << std::endl;
        for (auto& e : eigenvalue)std::cout << e << " ";
        std::cout << std::endl;

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
        //             std::cout << "norm2_now=" << norm2 << std::endl;
        //         }
        //     }
        //     std::cout << "state " << ist << ", norm2=" << norm2 << std::endl;
        // }

        // output iters
        std::cout << "Average iterative diagonalization steps: " << DiagoIterAssist<T, Device>::avg_iter
            << " ; where current threshold is: " << DiagoIterAssist<T, Device>::PW_DIAG_THR << " . " << std::endl;
        // castmem_2d_2h_op()(cpu_ctx, cpu_ctx, pes->ekb.c, eigenvalues.data(), pes->ekb.nr * pes->ekb.nc);
    }
    template class HSolverLR<double, psi::DEVICE_CPU>;
    template class HSolverLR<std::complex<double>, psi::DEVICE_CPU>;
};