#include "hamilt_casida.h"
namespace LR
{
    template<typename T>
    std::vector<T> HamiltCasidaLR<T>::matrix()
    {
        ModuleBase::TITLE("HamiltCasidaLR", "matrix");
        const int no = this->nocc[0];
        const int nv = this->nvirt[0];
        const auto& px = this->pX[0];
        int npairs = no * nv;
        std::vector<T> Amat_full(this->nk * npairs * this->nk * npairs, 0.0);
        for (int ik = 0;ik < this->nk;++ik)
            for (int j = 0;j < no;++j)
                for (int b = 0;b < nv;++b)
                {//calculate A^{ai} for each bj
                    int bj = j * nv + b;
                    int kbj = ik * npairs + bj;
                    psi::Psi<T> X_bj(1, 1, this->nk * px.get_local_size()); // k1-first, like in iterative solver
                    X_bj.zero_out();
                    // X_bj(0, 0, lj * px.get_row_size() + lb) = this->one();
                    int lj = px.global2local_col(j);
                    int lb = px.global2local_row(b);
                    if (px.in_this_processor(b, j)) X_bj(0, 0, ik * px.get_local_size() + lj * px.get_row_size() + lb) = this->one();
                    psi::Psi<T> A_aibj(1, 1, this->nk * px.get_local_size()); // k1-first
                    A_aibj.zero_out();

                    hamilt::Operator<T>* node(this->ops);
                    while (node != nullptr)
                    {   // act() on and return the k1-first type of psi
                        node->act(X_bj, A_aibj, 1);
                        node = (hamilt::Operator<T>*)(node->next_op);
                    }
                    // reduce ai for a fixed bj
                    A_aibj.fix_kb(0, 0);
#ifdef __MPI
                    for (int ik_ai = 0;ik_ai < this->nk;++ik_ai)
                        LR_Util::gather_2d_to_full(px, &A_aibj.get_pointer()[ik_ai * px.get_local_size()],
                            Amat_full.data() + kbj * this->nk * npairs /*col, bj*/ + ik_ai * npairs/*row, ai*/,
                            false, nv, no);
#endif
                }
        // output Amat
        std::cout << "Amat_full:" << std::endl;
        for (int i = 0;i < this->nk * npairs;++i)
        {
            for (int j = 0;j < this->nk * npairs;++j)
            {
                std::cout << Amat_full[i * this->nk * npairs + j] << " ";
            }
            std::cout << std::endl;
        }
        return Amat_full;
    }

    template<> double HamiltCasidaLR<double>::one() { return 1.0; }
    template<> std::complex<double> HamiltCasidaLR<std::complex<double>>::one() { return std::complex<double>(1.0, 0.0); }

    template class HamiltCasidaLR<double>;
    template class HamiltCasidaLR<std::complex<double>>;
}