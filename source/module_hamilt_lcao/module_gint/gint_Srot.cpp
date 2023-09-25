#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "gint_k.h"
#include "module_basis/module_ao/ORB_read.h"
#include "grid_technique.h"
#include "module_base/ylm.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_base/tool_threading.h"
#include "module_cell/module_neighbor/sltk_grid_driver.h"
#include "module_base/libm/libm.h"

void Gint::gint_kernel_Srot(
    const ModuleBase::Matrix3& ginv,
    const int na_grid,
    const int grid_index,
    const double delta_r,
    const double dv,
    const int LD_pool,
    double* pvpR_in)
{
    //prepare block information
    int* block_iw, * block_index, * block_size;
    bool** cal_flag;
    Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, block_iw, block_index, block_size, cal_flag);

    //evaluate phi and phirot on grids
    Gint_Tools::Array_Pool<double> psir_ylm(this->bxyz, LD_pool);
    Gint_Tools::Array_Pool<double> psir_ylmrot(this->bxyz, LD_pool);
    Gint_Tools::cal_psir_ylm(*this->gridt,
        this->bxyz, na_grid, grid_index, delta_r,
        block_index, block_size,
        cal_flag,
        psir_ylm.ptr_2D);
    //phirot(r) = f(r)*Ylm(g^{-1}r), here calculate phirot(r)*dv
    Gint_Tools::cal_psir_ylmrot_dv(*this->gridt, ginv,
        this->bxyz, na_grid, grid_index, delta_r, dv,
        block_index, block_size,
        cal_flag,
        psir_ylmrot.ptr_2D);


    //integrate (psi_mu*v(r)*dv) * psi_nu on grid
    //and accumulates to the corresponding element in Hamiltonian
    this->cal_meshball_vlocal_k(
        na_grid, LD_pool, grid_index, block_size, block_index, block_iw, cal_flag,
        psir_ylm.ptr_2D, psir_ylmrot.ptr_2D, pvpR_in);

    //release memories
    delete[] block_iw;
    delete[] block_index;
    delete[] block_size;
    for (int ib = 0; ib < this->bxyz; ++ib)
    {
        delete[] cal_flag[ib];
    }
    delete[] cal_flag;

    return;
}


void Gint::gint_kernel_Srot(
    const int na_grid,
    const int grid_index,
    const double delta_r,
    const double dv,
    const int LD_pool,
    Gint_inout* inout)
{
    const int& lgd = this->gridt->lgd;
    // ModuleBase::GlobalFunc::ZEROS(inout->Srotk_grid->data(), lgd * lgd);
    //prepare block information
    // block_index[ia]: start orbital index of atom ia in na_grid
    int* block_iw, * block_index, * block_size;
    bool** cal_flag;
    Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, block_iw, block_index, block_size, cal_flag);

    //evaluate psi on grids
    Gint_Tools::Array_Pool<double> psir_ylm(this->bxyz, LD_pool);
    Gint_Tools::cal_psir_ylm(*this->gridt,
        this->bxyz, na_grid, grid_index, delta_r,
        block_index, block_size,
        cal_flag,
        psir_ylm.ptr_2D);

    for (int ia1 = 0; ia1 < na_grid; ++ia1)
    {
        const int mcell_index1 = this->gridt->bcell_start[grid_index] + ia1;
        const int iat1 = this->gridt->which_atom[mcell_index1];
        const int T1 = GlobalC::ucell.iat2it[iat1];
        Atom* atom1 = &GlobalC::ucell.atoms[T1];
        const int I1 = GlobalC::ucell.iat2ia[iat1];
        //find R by which_unitcell and cal kphase
        const int id1_ucell = this->gridt->which_unitcell[mcell_index1];
        const int R1x = this->gridt->ucell_index2x[id1_ucell] + this->gridt->minu1;
        const int R1y = this->gridt->ucell_index2y[id1_ucell] + this->gridt->minu2;
        const int R1z = this->gridt->ucell_index2z[id1_ucell] + this->gridt->minu3;
        ModuleBase::Vector3<double> R1((double)R1x, (double)R1y, (double)R1z);
        const double arg1 = -(inout->kd1 * R1) * ModuleBase::TWO_PI;
        const std::complex<double> kphase1 = std::complex <double>(cos(arg1), sin(arg1));
        // get the start index of local orbitals.
        const int start1 = GlobalC::ucell.itiaiw2iwt(T1, I1, 0);
        const int* iw1_lo = &this->gridt->trace_lo[start1];
        for (int ia2 = 0;ia2 < na_grid;++ia2)
        {
            const int mcell_index2 = this->gridt->bcell_start[grid_index] + ia2;
            const int iat2 = this->gridt->which_atom[mcell_index2];
            const int T2 = GlobalC::ucell.iat2it[iat2];
            Atom* atom2 = &GlobalC::ucell.atoms[T2];
            const int I2 = GlobalC::ucell.iat2ia[iat2];
            //find R by which_unitcell and cal kphase
            const int id2_ucell = this->gridt->which_unitcell[mcell_index2];
            const int R2x = this->gridt->ucell_index2x[id2_ucell] + this->gridt->minu1;
            const int R2y = this->gridt->ucell_index2y[id2_ucell] + this->gridt->minu2;
            const int R2z = this->gridt->ucell_index2z[id2_ucell] + this->gridt->minu3;
            ModuleBase::Vector3<double> R2((double)R2x, (double)R2y, (double)R2z);
            if (grid_index == 0 && inout->kd1.norm() < 1e-5 && inout->kd2.norm() < 1e-5)
            {
                GlobalV::ofs_running << "R1_int=" << R1x << " " << R1y << " " << R1z << std::endl;
                GlobalV::ofs_running << "R2_int=" << R2x << " " << R2y << " " << R2z << std::endl;
            }
            const double arg2 = (inout->kd2 * R2) * ModuleBase::TWO_PI;
            const std::complex<double> kphase2 = std::complex <double>(cos(arg2), sin(arg2));
            // get the start index of local orbitals.
            const int start2 = GlobalC::ucell.itiaiw2iwt(T2, I2, 0);
            const int* iw2_lo = &this->gridt->trace_lo[start2];
            for (int iw1 = 0; iw1 < atom1->nw; ++iw1)
            {
                for (int iw2 = 0; iw2 < atom2->nw; ++iw2)
                {
                    for (int ib = 0; ib < this->bxyz; ib++)
                    {
                        if (cal_flag[ib][ia1] && cal_flag[ib][ia2])
                        {
                            double* psi1 = &psir_ylm.ptr_2D[ib][block_index[ia1]];
                            double* psi2 = &psir_ylm.ptr_2D[ib][block_index[ia2]];
                            inout->Srotk_grid->at(iw1_lo[iw1] * lgd + iw2_lo[iw2]) += std::complex<double>(psi1[iw1], 0.0) * psi2[iw2] * kphase1 * kphase2 * dv;
                        }// cal_flag
                    }//ib
                }//iw2
            }//iw1
        }//ia2
    }// ia1
    delete[] block_iw;
    delete[] block_index;
    delete[] block_size;
    for (int ib = 0; ib < this->bxyz; ++ib)
    {
        delete[] cal_flag[ib];
    }
    delete[] cal_flag;
}

std::vector<std::complex<double>> Gint_k::grid_to_2d(
    const Parallel_2D& p2d,
    const std::vector<std::complex<double>>& mat_grid)const
{
    ModuleBase::TITLE("Gint_k", "grid_to_2d");
    std::cout << "lgd=" << this->gridt->lgd << std::endl;
    const int& lgd = this->gridt->lgd;
    const int& nlocal = GlobalV::NLOCAL;
    assert(mat_grid.size() == lgd * lgd);

    std::vector<std::complex<double>> mat_2d(p2d.get_local_size(), 0);
    for (int i = 0; i < nlocal; ++i)
    {
        std::vector<std::complex<double>> tmp(nlocal, 0);
        int mug = this->gridt->trace_lo[i];
        if (mug > 0)
            for (int j = 0; j < nlocal; ++j)
            {
                int nug = this->gridt->trace_lo[j];
                if (nug > 0) tmp[j] = mat_grid[mug * lgd + nug];
            }
        Parallel_Reduce::reduce_complex_double_pool(tmp.data(), tmp.size());
        for (int j = 0; j < nlocal; j++)
        {
            if (p2d.in_this_processor(i, j))
            {   //set the value
                const int li = p2d.global2local_row(i);
                const int lj = p2d.global2local_col(j);
                long index = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER() ?
                    lj * p2d.get_row_size() + li : lj * p2d.get_row_size() + li;
                mat_2d[index] += tmp[j];
            }
        }
    }
    return mat_2d;
}

std::vector<std::complex<double>> Gint_k::folding_Srot_k(
    const ModuleBase::Vector3<double>& kvec_d,
    const Parallel_2D& p2d,
    const double* data)
{
    ModuleBase::TITLE("Gint_k", "folding_Srot_k");
    ModuleBase::timer::tick("Gint_k", "folding_Srot_k");
    if (!pvpR_alloc_flag)
    {
        ModuleBase::WARNING_QUIT("Gint_k::destroy_pvpR", "pvpR hasnot been allocated yet!");
    }
    int lgd = this->gridt->lgd;
    std::complex<double>** pvp = new std::complex<double>*[lgd];
    std::complex<double>* pvp_base = new std::complex<double>[lgd * lgd];
    for (int i = 0; i < lgd; i++)
    {
        pvp[i] = pvp_base + i * lgd;
    }

    auto init_pvp = [&](int num_threads, int thread_id)
        {
            int beg, len;
            ModuleBase::BLOCK_TASK_DIST_1D(num_threads, thread_id, lgd * lgd, 256, beg, len);
            ModuleBase::GlobalFunc::ZEROS(pvp_base + beg, len);
        };
    ModuleBase::OMP_PARALLEL(init_pvp);
#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        ModuleBase::Vector3<double> tau1, dtau, dR;
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int iat = 0; iat < GlobalC::ucell.nat; ++iat)
        {
            const int T1 = GlobalC::ucell.iat2it[iat];
            const int I1 = GlobalC::ucell.iat2ia[iat];
            {
                // atom in this grid piece.
                if (this->gridt->in_this_processor[iat])
                {
                    Atom* atom1 = &GlobalC::ucell.atoms[T1];
                    const int start1 = GlobalC::ucell.itiaiw2iwt(T1, I1, 0);

                    // get the start positions of elements.
                    const int DM_start = this->gridt->nlocstartg[iat];

                    // get the coordinates of adjacent atoms.
                    tau1 = GlobalC::ucell.atoms[T1].tau[I1];
                    //GlobalC::GridD.Find_atom(tau1);	
                    AdjacentAtomInfo adjs;
                    GlobalC::GridD.Find_atom(GlobalC::ucell, tau1, T1, I1, &adjs);
                    // search for the adjacent atoms.
                    int nad = 0;

                    for (int ad = 0; ad < adjs.adj_num + 1; ad++)
                    {
                        // get iat2
                        const int T2 = adjs.ntype[ad];
                        const int I2 = adjs.natom[ad];
                        const int iat2 = GlobalC::ucell.itia2iat(T2, I2);


                        // adjacent atom is also on the grid.
                        if (this->gridt->in_this_processor[iat2])
                        {
                            Atom* atom2 = &GlobalC::ucell.atoms[T2];
                            dtau = adjs.adjacent_tau[ad] - tau1;
                            double distance = dtau.norm() * GlobalC::ucell.lat0;
                            double rcut = GlobalC::ORB.Phi[T1].getRcut() + GlobalC::ORB.Phi[T2].getRcut();

                            // for the local part, only need to calculate <phi_i | phi_j> within range
                            // mohan note 2012-07-06
                            if (distance < rcut)
                            {
                                const int start2 = GlobalC::ucell.itiaiw2iwt(T2, I2, 0);

                                // calculate the distance between iat1 and iat2.
                                dR.x = adjs.box[ad].x;
                                dR.y = adjs.box[ad].y;
                                dR.z = adjs.box[ad].z;

                                // calculate the phase factor exp(ikR).
                                const double arg = (kvec_d * dR) * ModuleBase::TWO_PI;
                                double sinp, cosp;
                                ModuleBase::libm::sincos(arg, &sinp, &cosp);
                                const std::complex<double> phase = std::complex<double>(cosp, sinp);
                                int ixxx = DM_start + this->gridt->find_R2st[iat][nad];

                                for (int iw = 0; iw < atom1->nw; iw++)
                                {
                                    std::complex<double>* vij = pvp[this->gridt->trace_lo[start1 + iw]];
                                    const int* iw2_lo = &this->gridt->trace_lo[start2];
                                    // get the <phi | V | phi>(R) Hamiltonian.
                                    // const double* vijR = &pvpR_reduced[0][ixxx];
                                    const double* vijR = &data[ixxx];
                                    for (int iw2 = 0; iw2 < atom2->nw; ++iw2)
                                    {
                                        vij[iw2_lo[iw2]] += vijR[iw2] * phase;
                                    }
                                    ixxx += atom2->nw;
                                }
                                ++nad;
                            }// end distance<rcut
                        }
                    }// end ad
                }
            }// end ia
        }// end it
#ifdef _OPENMP
    }
#endif

    // Distribution of data.
    ModuleBase::timer::tick("Gint_k", "Distri");
    const int nlocal = GlobalV::NLOCAL;
    std::vector<std::complex<double>> tmp(nlocal);
    std::vector<std::complex<double>> Srotk(p2d.get_local_size(), 0);  //result

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        //loop each row with index i, than loop each col with index j 
        for (int i = 0; i < nlocal; i++)
        {
#ifdef _OPENMP
#pragma omp for
#endif
            for (int j = 0; j < nlocal; j++)
            {
                tmp[j] = std::complex<double>(0.0, 0.0);
            }
            int i_flag = i & 1; // i % 2 == 0
            const int mug = this->gridt->trace_lo[i];
            // if the row element is on this processor.
            if (mug >= 0)
            {
#ifdef _OPENMP
#pragma omp for
#endif
                for (int j = 0; j < nlocal; j++)
                {
                    const int nug = this->gridt->trace_lo[j];
                    // if the col element is on this processor.
                    if (nug >= 0)
                    {
                        tmp[j] = pvp[mug][nug];
                    }
                }
            }
#ifdef _OPENMP
#pragma omp single
            {
#endif
                // collect the matrix after folding.
                Parallel_Reduce::reduce_complex_double_pool(tmp.data(), tmp.size());
#ifdef _OPENMP
            }
#endif

            //-----------------------------------------------------
            // NOW! Redistribute the Hamiltonian matrix elements
            // according to the HPSEPS's 2D distribution methods.
            //-----------------------------------------------------
#ifdef _OPENMP
#pragma omp for
#endif
            for (int j = 0; j < nlocal; j++)
            {
                if (p2d.in_this_processor(i, j))
                {   //set the value
                    const int li = p2d.global2local_row(i);
                    const int lj = p2d.global2local_col(j);
                    long index;
                    if (ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER())
                    {
                        index = lj * p2d.get_row_size() + li;
                    }
                    else
                    {
                        index = li * p2d.get_col_size() + lj;
                    }
                    Srotk[index] += tmp[j];
                }
            }
        }
#ifdef _OPENMP
    }
#endif

    // delete the tmp matrix.
    delete[] pvp;
    delete[] pvp_base;
    ModuleBase::timer::tick("Gint_k", "Distri");

    ModuleBase::timer::tick("Gint_k", "folding_Srot_k");

    return Srotk;
}

