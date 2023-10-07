#include "hsolver_lrtd.h"
#include "module_hsolver/diago_david.h"

namespace hsolver
{
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
        std::vector<Real> precondition(psi.get_nbasis());
        std::vector<Real> eigenvalue(psi.get_nbands());   //nstataes
        // 2. select the method
        this->method = method_in;
        if (this->method == "dav")
        {
            DiagoDavid<T>::PW_DIAG_NDIM = GlobalV::PW_DIAG_NDIM;
            this->pdiagh = new DiagoDavid<T, Device>(precondition.data());      //waiting for complex<T> removement
            this->pdiagh->method = this->method;
        }
        else
            throw std::runtime_error("HSolverLR::solve: method not implemented");

        // 3. set precondition and diagethr
        for (int i = 0;i < psi.get_nbasis();++i)precondition[i] = static_cast<Real>(i + 1);
        // T ethr = this->set_diagether(1, 1, static_cast<T>(1e-2));
        this->diag_ethr = 1e-2;
        std::cout << "ethr: " << this->diag_ethr << std::endl;
        // 4. solve Hamiltonian
        this->pdiagh->diag(pHamilt, psi, eigenvalue.data());
        // 5. copy eigenvalue to pes
        std::cout << "eigenvalues:" << std::endl;
        for (auto& e : eigenvalue)std::cout << e << " ";
        std::cout << std::endl;

        // output iters
        std::cout << "Average iterative diagonalization steps: " << DiagoIterAssist<T, Device>::avg_iter
            << " ; where current threshold is: " << DiagoIterAssist<T, Device>::PW_DIAG_THR << " . " << std::endl;
        // castmem_2d_2h_op()(cpu_ctx, cpu_ctx, pes->ekb.c, eigenvalues.data(), pes->ekb.nr * pes->ekb.nc);
    }
    template class HSolverLR<double, psi::DEVICE_CPU>;
    template class HSolverLR<std::complex<double>, psi::DEVICE_CPU>;
};