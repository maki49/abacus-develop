#pragma once
#include <cstddef>
#include "lr_util.h"
#include <algorithm>
#include "module_cell/unitcell.h"
#include "module_base/constants.h"
#include "module_base/scalapack_connector.h"


namespace LR_Util
{

    /// =================ALGORITHM====================

    //====== newers and deleters========
    //(arbitrary dimention will be supported in the future)

    /// @brief  new 2d pointer
    /// @tparam T
    /// @param size1
    /// @param size2
    template <typename T>
    void new_p2(T**& p2, size_t size1, size_t size2)
    {
        p2 = new T * [size1];
        for (size_t i = 0; i < size1; ++i)
        {
            p2[i] = new T[size2];
        }
    };

    /// @brief  new 3d pointer
    /// @tparam T
    /// @param size1
    /// @param size2
    /// @param size3
    template <typename T>
    void new_p3(T***& p3, size_t size1, size_t size2, size_t size3)
    {
        p3 = new T * *[size1];
        for (size_t i = 0; i < size1; ++i)
        {
            new_p2(p3[i], size2, size3);
        }
    };

    /// @brief  delete 2d pointer 
    /// @tparam T 
    /// @param p2 
    /// @param size 
    template <typename T>
    void delete_p2(T** p2, size_t size)
    {
        if (p2 != nullptr)
        {
            for (size_t i = 0; i < size; ++i)
            {
                if (p2[i] != nullptr) delete[] p2[i];
            }
            delete[] p2;
        }
    };

    /// @brief  delete 3d pointer 
    /// @tparam T 
    /// @param p2 
    /// @param size1
    /// @param size2
    template <typename T>
    void delete_p3(T*** p3, size_t size1, size_t size2)
    {
        if (p3 != nullptr)
        {
            for (size_t i = 0; i < size1; ++i)
            {
                delete_p2(p3[i], size2);
            }
            delete[] p3;
        }
    };

    template <typename T>
    void matsym(const T* in, const int n, T* out)
    {
        for (int i = 0;i < n;++i)  out[i * n + i] = in[i * n + i];
        for (int i = 0;i < n;++i)
            for (int j = i + 1;j < n;++j)
                out[i * n + j] = out[j * n + i] = 0.5 * (in[i * n + j] + in[j * n + i]);
    }
    template void matsym<double>(const double* in, const int n, double* out);
    template void matsym<std::complex<double>>(const std::complex<double>* in, const int n, std::complex<double>* out);
    template <typename T>
    void matsym(T* inout, const int n)
    {
        for (int i = 0;i < n;++i)
            for (int j = i + 1;j < n;++j)
                inout[i * n + j] = inout[j * n + i] = 0.5 * (inout[i * n + j] + inout[j * n + i]);
    }
    template void matsym<double>(double* inout, const int n);
    template void matsym<std::complex<double>>(std::complex<double>* inout, const int n);
#ifdef __MPI
    template<>
    void matsym<double>(const double* in, const int n, const Parallel_2D& pmat, double* out)
    {
        for (int i = 0;i < pmat.get_local_size();++i)out[i] = in[i];
        const double alpha = 0.5, beta = 0.5;
        const int i1 = 1;
        pdtran_(&n, &n, &alpha, in, &i1, &i1, pmat.desc, &beta, out, &i1, &i1, pmat.desc);
    }
    template<>
    void matsym<double>(double* inout, const int n, const Parallel_2D& pmat)
    {
        std::vector<double> tmp(n * n);
        for (int i = 0;i < pmat.get_local_size();++i)tmp[i] = inout[i];
        const double alpha = 0.5, beta = 0.5;
        const int i1 = 1;
        pdtran_(&n, &n, &alpha, tmp.data(), &i1, &i1, pmat.desc, &beta, inout, &i1, &i1, pmat.desc);
    }
    template<>
    void matsym<std::complex<double>>(const std::complex<double>* in, const int n, const Parallel_2D& pmat, std::complex<double>* out)
    {
        for (int i = 0;i < pmat.get_local_size();++i)out[i] = in[i];
        const std::complex<double> alpha(0.5, 0.0), beta(0.5, 0.0);
        const int i1 = 1;
        pztranu_(&n, &n, &alpha, in, &i1, &i1, pmat.desc, &beta, out, &i1, &i1, pmat.desc);
    }
    template<>
    void matsym<std::complex<double>>(std::complex<double>* inout, const int n, const Parallel_2D& pmat)
    {
        std::vector<std::complex<double>> tmp(n * n);
        for (int i = 0;i < pmat.get_local_size();++i)tmp[i] = inout[i];
        const std::complex<double> alpha(0.5, 0.0), beta(0.5, 0.0);
        const int i1 = 1;
        pztranu_(&n, &n, &alpha, tmp.data(), &i1, &i1, pmat.desc, &beta, inout, &i1, &i1, pmat.desc);
    }
#endif
    container::Tensor mat2ten_double(ModuleBase::matrix& m)
    {
        container::Tensor t(DAT::DT_DOUBLE, DEV::CpuDevice, { m.nr, m.nc });
        for (int i = 0;i < t.NumElements();++i)t.data<double>()[i] = m.c[i];
        return t;
    }
    std::vector<container::Tensor> mat2ten_double(std::vector<ModuleBase::matrix>& m)
    {
        std::vector<container::Tensor> t;
        for (int i = 0;i < m.size();++i) t.push_back(mat2ten_double(m[i]));
        return t;
    }
    ModuleBase::matrix ten2mat_double(container::Tensor& t)
    {
        ModuleBase::matrix m(t.shape().dims()[0], t.shape().dims()[1]);
        for (int i = 0;i < t.NumElements();++i)m.c[i] = t.data<double>()[i];
        return m;
    }
    std::vector<ModuleBase::matrix> ten2mat_double(std::vector<container::Tensor>& t)
    {
        std::vector<ModuleBase::matrix> m;
        for (int i = 0;i < t.size();++i) m.push_back(ten2mat_double(t[i]));
        return m;
    }
    container::Tensor mat2ten_complex(ModuleBase::ComplexMatrix& m)
    {
        container::Tensor t(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { m.nr, m.nc });
        for (int i = 0;i < t.NumElements();++i)t.data<std::complex<double>>()[i] = m.c[i];
        return t;
    }
    std::vector<container::Tensor> mat2ten_complex(std::vector<ModuleBase::ComplexMatrix>& m)
    {
        std::vector<container::Tensor> t;
        for (int i = 0;i < m.size();++i) t.push_back(mat2ten_complex(m[i]));
        return t;
    }
    ModuleBase::ComplexMatrix ten2mat_complex(container::Tensor& t)
    {
        ModuleBase::ComplexMatrix m(t.shape().dims()[0], t.shape().dims()[1]);
        for (int i = 0;i < t.NumElements();++i)m.c[i] = t.data<std::complex<double>>()[i];
        return m;
    }
    std::vector<ModuleBase::ComplexMatrix> ten2mat_complex(std::vector<container::Tensor>& t)
    {
        std::vector<ModuleBase::ComplexMatrix> m;
        for (int i = 0;i < t.size();++i) m.push_back(ten2mat_complex(t[i]));
        return m;
    }

    ModuleBase::matrix vec2mat(const std::vector<double>& v, const int nr, const int nc)
    {
        assert(v.size() == nr * nc);
        ModuleBase::matrix m(nr, nc, false);
        for (int i = 0;i < v.size();++i) m.c[i] = v[i];
        return m;
    }
    ModuleBase::ComplexMatrix vec2mat(const std::vector<std::complex<double>>& v, const int nr, const int nc)
    {
        assert(v.size() == nr * nc);
        ModuleBase::ComplexMatrix m(nr, nc, false);
        for (int i = 0;i < v.size();++i) m.c[i] = v[i];
        return m;
    }
    std::vector<ModuleBase::matrix> vec2mat(const std::vector<std::vector<double>>& v, const int nr, const int nc)
    {
        std::vector<ModuleBase::matrix> m(v.size());
        for (int i = 0;i < v.size();++i) m[i] = vec2mat(v[i], nr, nc);
        return m;
    }
    std::vector<ModuleBase::ComplexMatrix> vec2mat(const std::vector<std::vector<std::complex<double>>>& v, const int nr, const int nc)
    {
        std::vector<ModuleBase::ComplexMatrix> m(v.size());
        for (int i = 0;i < v.size();++i) m[i] = vec2mat(v[i], nr, nc);
        return m;
    }

    /// psi(nk=1, nbands=nb, nk * nbasis) -> psi(nb, nk, nbasis) without memory copy
    template<typename T, typename Device>
    psi::Psi<T, Device> k1_to_bfirst_wrapper(const psi::Psi<T, Device>& psi_kfirst, int nk_in, int nbasis_in)
    {
        assert(psi_kfirst.get_nk() == 1);
        assert(nk_in * nbasis_in == psi_kfirst.get_nbasis());
        int ib_now = psi_kfirst.get_current_b();
        psi_kfirst.fix_b(0);    // for get_pointer() to get the head pointer
        psi::Psi<T, Device> psi_bfirst(psi_kfirst.get_pointer(), nk_in, psi_kfirst.get_nbands(), nbasis_in, psi_kfirst.get_ngk_pointer(), false);
        psi_kfirst.fix_b(ib_now);
        return psi_bfirst;
    }

    ///  psi(nb, nk, nbasis) -> psi(nk=1, nbands=nb, nk * nbasis)  without memory copy
    template<typename T, typename Device>
    psi::Psi<T, Device> bfirst_to_k1_wrapper(const psi::Psi<T, Device>& psi_bfirst)
    {
        int ib_now = psi_bfirst.get_current_b();
        int ik_now = psi_bfirst.get_current_k();
        psi_bfirst.fix_kb(0, 0);    // for get_pointer() to get the head pointer
        psi::Psi<T, Device> psi_kfirst(psi_bfirst.get_pointer(), 1, psi_bfirst.get_nbands(), psi_bfirst.get_nk() * psi_bfirst.get_nbasis(), psi_bfirst.get_ngk_pointer(), true);
        psi_bfirst.fix_kb(ik_now, ib_now);
        return psi_kfirst;
    }

    // for the first matrix in the commutator
    void setup_2d_division(Parallel_2D& pv, int nb, int gr, int gc)
    {
        ModuleBase::TITLE("ESolver_LRTD", "setup_2d_division");
        std::ofstream ofs("");
        pv.set_block_size(nb);
#ifdef __MPI
        int nprocs;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        pv.set_proc_dim(nprocs);
        pv.mpi_create_cart(MPI_COMM_WORLD);
        pv.set_local2global(gr, gc, ofs, ofs);
        pv.set_desc(gr, gc, pv.get_row_size());
        pv.set_global2local(gr, gc, true, ofs);
#else
        pv.set_proc_dim(1);
        pv.set_serial(gr, gc);
        pv.set_global2local(gr, gc, false, ofs);
#endif
    };

#ifdef __MPI
    // for the other matrices in the commutator other than the first one
    void setup_2d_division(Parallel_2D& pv, int nb, int gr, int gc,
        const MPI_Comm& comm_2D_in, const int& blacs_ctxt_in)
    {
        ModuleBase::TITLE("ESolver_LRTD", "setup_2d_division");
        std::ofstream ofs("");
        pv.set_block_size(nb);

        int nprocs, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        pv.set_proc_dim(nprocs);
        pv.comm_2D = comm_2D_in;
        pv.blacs_ctxt = blacs_ctxt_in;
        pv.set_local2global(gr, gc, ofs, ofs);
        pv.set_desc(gr, gc, pv.get_row_size(), false);
        pv.set_global2local(gr, gc, true, ofs);
    };
#endif


#ifdef __MPI
    template <typename T>
    void gather_2d_to_full(const Parallel_2D& pv, const T* submat, T* fullmat, bool col_first, int global_nrow, int global_ncol)
    {
        auto get_mpi_datatype = []() -> MPI_Datatype {
            if (std::is_same<T, int>::value) { return MPI_INT; }
            if (std::is_same<T, float>::value) { return MPI_FLOAT; }
            else if (std::is_same<T, double>::value) { return MPI_DOUBLE; }
            if (std::is_same<T, std::complex<float>>::value) { return MPI_COMPLEX; }
            else if (std::is_same<T, std::complex<double>>::value) { return MPI_DOUBLE_COMPLEX; }
            else { throw std::runtime_error("gather_2d_to_full: unsupported type"); }
            };

        // zeros
        for (int i = 0;i < global_nrow * global_ncol;++i) fullmat[i] = 0.0;
        //copy
        for (int i = 0;i < pv.get_row_size();++i)
            for (int j = 0;j < pv.get_col_size();++j)
                if (col_first)
                    fullmat[pv.local2global_row(i) * global_ncol + pv.local2global_col(j)] = submat[i * pv.get_col_size() + j];
                else
                    fullmat[pv.local2global_col(j) * global_nrow + pv.local2global_row(i)] = submat[j * pv.get_row_size() + i];

        //reduce to root
        MPI_Allreduce(MPI_IN_PLACE, fullmat, global_nrow * global_ncol, get_mpi_datatype(), MPI_SUM, pv.comm_2D);
    };
#endif
}