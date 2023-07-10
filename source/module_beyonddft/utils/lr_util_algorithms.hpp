#pragma once
#include <cstddef>
#include "lr_util.h"
#include <algorithm>
#include "module_cell/unitcell.h"
#include "module_base/constants.h"


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

    // for the first matrix in the commutator
    void setup_2d_division(Parallel_2D& pv, int nb, int gr, int gc)
    {
        ModuleBase::TITLE("ESolver_LRTD", "setup_2d_division");
        pv.set_block_size(nb);
#ifdef __MPI
        int nprocs;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        pv.set_proc_dim(nprocs);
        pv.mpi_create_cart(MPI_COMM_WORLD);
        pv.set_local2global(gr, gc, GlobalV::ofs_running, GlobalV::ofs_warning);
        pv.set_desc(gr, gc, pv.get_row_size());
        pv.set_global2local(gr, gc, true, GlobalV::ofs_running);
#else
        pv.set_proc_dim(1);
        pv.set_serial(gr, gc);
        pv.set_global2local(gr, gc, false, GlobalV::ofs_running);
#endif
    };

#ifdef __MPI
    // for the other matrices in the commutator other than the first one
    void setup_2d_division(Parallel_2D& pv, int nb, int gr, int gc,
        const MPI_Comm& comm_2D_in, const int& blacs_ctxt_in)
    {
        ModuleBase::TITLE("ESolver_LRTD", "setup_2d_division");
        pv.set_block_size(nb);

        int nprocs, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        pv.set_proc_dim(nprocs);
        pv.comm_2D = comm_2D_in;
        pv.blacs_ctxt = blacs_ctxt_in;
        pv.set_local2global(gr, gc, GlobalV::ofs_running, GlobalV::ofs_warning);
        pv.set_desc(gr, gc, pv.get_row_size(), false);
        pv.set_global2local(gr, gc, true, GlobalV::ofs_running);
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