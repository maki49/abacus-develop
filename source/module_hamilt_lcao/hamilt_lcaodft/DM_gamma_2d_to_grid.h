#pragma once
#include <vector>
#include "module_base/matrix.h"
#ifdef __MPI
#include <mpi.h>
#endif



class DMgamma_2dtoGrid
{
public:
    DMgamma_2dtoGrid();
    ~DMgamma_2dtoGrid();
#ifdef __MPI
    int setAlltoallvParameter(MPI_Comm comm_2D, const int nbasis, const int blacs_ctxt, const int nblk, const int& loc_grid_dim, const int* global2local_grid);
#endif
    void cal_dk_gamma_from_2D(
        const std::vector<ModuleBase::matrix>& dm_gamma_2d,
        double*** dm_gamma_grid,
        const int& nspin,
        const int& nbasis,
        const int& loc_grid_dim,
        std::ofstream& ofs_running);
private:
    // Buffer parameters for tranforming 2D block-cyclic distributed DM matrix 
// to grid distributed DM matrix
    int* sender_2D_index;
    int sender_size;
    int* sender_size_process;
    int* sender_displacement_process;
    double* sender_buffer;

    int* receiver_local_index;
    int receiver_size;
    int* receiver_size_process;
    int* receiver_displacement_process;
    double* receiver_buffer;
#ifdef __MPI
    MPI_Comm comm_2D;
#endif
};