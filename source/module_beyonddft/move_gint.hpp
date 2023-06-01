#include "utils/lr_util.h"
#include "module_hamilt_lcao/module_gint/gint_gamma.h"
#include "module_hamilt_lcao/module_gint/gint_k.h"
#include "module_hamilt_lcao/module_gint/grid_technique.h"

// Here will be  the only place where GlobalCs are used (to be moved) in module_beyonddft
#include "module_hamilt_pw/hamilt_pwdft/global.h"

template <typename T>
using D2 = void(*) (T**, size_t);
template <typename T>
using D3 = void(*) (T***, size_t, size_t);
template <typename T>
D2<T> d2 = LR_Util::delete_p2<T>;
template <typename T>
D3<T> d3 = LR_Util::delete_p3<T>;

Gint& Gint::operator=(Gint&& rhs)
{
    if (this == &rhs)return *this;
    
    this->nbx = rhs.nbx;
    this->nby = rhs.nby;
    this->nbz = rhs.nbz;
    this->ncxyz = rhs.ncxyz;
    this->nbz_start = rhs.nbz_start;
    this->bx = rhs.bx;
    this->by = rhs.by;
    this->bz = rhs.bz;
    this->bxyz = rhs.bxyz;
    this->nbxx = rhs.nbxx;
    this->ny = rhs.ny;
    this->nplane = rhs.nplane;
    this->startz_current = rhs.startz_current;
    rhs.nbx = 0;
    rhs.nby = 0;
    rhs.nbz = 0;
    rhs.ncxyz = 0;
    rhs.nbz_start = 0;
    rhs.bx = 0;
    rhs.by = 0;
    rhs.bz = 0;
    rhs.bxyz = 0;
    rhs.nbxx = 0;
    rhs.ny = 0;
    rhs.nplane = 0;
    rhs.startz_current = 0;
    
    if (this->pvpR_reduced != nullptr) d2<double>(this->pvpR_reduced, GlobalV::NSPIN);  //nspin*gridt.nnrg
    if (this->pvdpRx_reduced != nullptr) d2<double>(this->pvdpRx_reduced, GlobalV::NSPIN);
    if (this->pvdpRy_reduced != nullptr) d2<double>(this->pvdpRy_reduced, GlobalV::NSPIN);
    if (this->pvdpRz_reduced != nullptr) d2<double>(this->pvdpRz_reduced, GlobalV::NSPIN);
    if (this->pvpR_grid != nullptr) delete[] this->pvpR_grid;   //lgd*lgd
    this->pvpR_alloc_flag = rhs.pvpR_alloc_flag;
    rhs.pvpR_alloc_flag = false;
    this->pvpR_reduced = rhs.pvpR_reduced;
    this->pvdpRx_reduced = rhs.pvdpRx_reduced;
    this->pvdpRy_reduced = rhs.pvdpRy_reduced;
    this->pvdpRz_reduced = rhs.pvdpRz_reduced;
    this->pvpR_grid = rhs.pvpR_grid;
    rhs.pvpR_reduced = nullptr;
    rhs.pvdpRx_reduced = nullptr;
    rhs.pvdpRy_reduced = nullptr;
    rhs.pvdpRz_reduced = nullptr;
    rhs.pvpR_grid = nullptr;

    return *this;
}

Gint_Gamma& Gint_Gamma::operator=(Gint_Gamma&& rhs)
{
    if (this == &rhs)return *this;
    Gint::operator=(std::move(rhs));
    
    // DM may not needed in beyond DFT ESolver
    // if (this->DM != nullptr) d3<double>(this->DM, GlobalV::NSPIN, gridt.lgd);
    assert (this->DM == nullptr);
    // this->DM = rhs.DM;
    // rhs.DM = nullptr;
    this->sender_index_size = rhs.sender_index_size;
    this->sender_size = rhs.sender_size;
    rhs.sender_index_size = 0;
    rhs.sender_size = 0;
    if (this->sender_local_index != nullptr) delete[] this->sender_local_index;
    if (this->sender_size_process != nullptr) delete[] this->sender_size_process;
    if (this->sender_displacement_process != nullptr) delete[] this->sender_displacement_process;
    if (this->sender_buffer != nullptr) delete[] this->sender_buffer;
    this->sender_local_index = rhs.sender_local_index;
    this->sender_size_process = rhs.sender_size_process;
    this->sender_displacement_process = rhs.sender_displacement_process;
    this->sender_buffer = rhs.sender_buffer;
    rhs.sender_local_index = nullptr;
    rhs.sender_size_process = nullptr;
    rhs.sender_displacement_process = nullptr;
    rhs.sender_buffer = nullptr;
    
    this->receiver_index_size = rhs.receiver_index_size;
    this->receiver_size = rhs.receiver_size;
    rhs.receiver_index_size = 0;
    rhs.receiver_size = 0;
    if (this->receiver_global_index != nullptr) delete[] this->receiver_global_index;
    if (this->receiver_size_process != nullptr) delete[] this->receiver_size_process;
    if (this->receiver_displacement_process != nullptr) delete[] this->receiver_displacement_process;
    if (this->receiver_buffer != nullptr) delete[] this->receiver_buffer;
    this->receiver_global_index = rhs.receiver_global_index;
    this->receiver_size_process = rhs.receiver_size_process;
    this->receiver_displacement_process = rhs.receiver_displacement_process;
    this->receiver_buffer = rhs.receiver_buffer;
    rhs.receiver_global_index = nullptr;
    rhs.receiver_size_process = nullptr;
    rhs.receiver_displacement_process = nullptr;
    rhs.receiver_buffer = nullptr;
    
    return *this;
}

Gint_k& Gint_k::operator=(Gint_k&& rhs)
{
    if (this == &rhs)return *this;
    this->Gint::operator=(std::move(rhs));
    return *this;
}

Grid_MeshK& Grid_MeshK::operator=(Grid_MeshK&& rhs)
{
    if (this == &rhs)return *this;
    
    this->maxu1 = rhs.maxu1;
    this->maxu2 = rhs.maxu2;
    this->maxu3 = rhs.maxu3;
    this->minu1 = rhs.minu1;
    this->minu2 = rhs.minu2;
    this->minu3 = rhs.minu3;
    this->nu1 = rhs.nu1;
    this->nu2 = rhs.nu2;
    this->nu3 = rhs.nu3;
    this->nutot = rhs.nutot;
    
    if (this->ucell_index2x != nullptr) delete[] this->ucell_index2x;
    if(this->ucell_index2y != nullptr) delete[] this->ucell_index2y;
    if(this->ucell_index2z != nullptr) delete[] this->ucell_index2z;
    this->ucell_index2x = rhs.ucell_index2x;
    this->ucell_index2y = rhs.ucell_index2y;
    this->ucell_index2z = rhs.ucell_index2z;
    rhs.ucell_index2x = nullptr;
    rhs.ucell_index2y = nullptr;
    rhs.ucell_index2z = nullptr;
    return *this;
}

Grid_MeshCell& Grid_MeshCell::operator=(Grid_MeshCell&& rhs)
{
    if (this == &rhs)return *this;
    this->Grid_MeshK::operator=(std::move(rhs));
    
    this->ncx = rhs.ncx;
    this->ncy = rhs.ncy;
    this->ncz = rhs.ncz;
    this->ncxyz = rhs.ncxyz;
    this->bx = rhs.bx;
    this->by = rhs.by;
    this->bz = rhs.bz;
    this->bxyz = rhs.bxyz;
    this->nbx = rhs.nbx;
    this->nby = rhs.nby;
    this->nbz = rhs.nbz;
    this->nbxyz = rhs.nbxyz;
    this->nbxx = rhs.nbxx;
    this->nbzp_start = rhs.nbzp_start;
    this->nbzp = rhs.nbzp;
    for (int i = 0;i < 3;++ i)
    {
        this->meshcell_vec1[i] = rhs.meshcell_vec1[i];
        this->meshcell_vec2[i] = rhs.meshcell_vec2[i];
        this->meshcell_vec3[i] = rhs.meshcell_vec3[i];
    }
    this->meshcell_latvec0 = std::move(rhs.meshcell_latvec0);
    this->meshcell_GT = std::move(rhs.meshcell_GT);
    
    if (this->allocate_pos && this->meshcell_pos != nullptr) d2<double>(this->meshcell_pos, this->bxyz);
    this->meshcell_pos = rhs.meshcell_pos;
    rhs.meshcell_pos = nullptr;
    this->allocate_pos = rhs.allocate_pos;
    rhs.allocate_pos = false;
    return *this;
}

Grid_BigCell& Grid_BigCell::operator=(Grid_BigCell&& rhs)
{
    if (this == &rhs)return *this;
    this->Grid_MeshCell::operator=(std::move(rhs));
    this->orbital_rmax = rhs.orbital_rmax;
    this->bigcell_dx = rhs.bigcell_dx;
    this->bigcell_dy = rhs.bigcell_dy;
    this->bigcell_dz = rhs.bigcell_dz;
    this->dxe = rhs.dxe;
    this->dye = rhs.dye;
    this->dze = rhs.dze;
    this->nxe = rhs.nxe;
    this->nye = rhs.nye;
    this->nze = rhs.nze;
    this->nxyze = rhs.nxyze;
    for (int i = 0;i < 3;++ i)
    {
        this->bigcell_vec1[i] = rhs.bigcell_vec1[i];
        this->bigcell_vec2[i] = rhs.bigcell_vec2[i];
        this->bigcell_vec3[i] = rhs.bigcell_vec3[i];
    }
    this->bigcell_latvec0 = std::move(rhs.bigcell_latvec0);
    this->bigcell_GT = std::move(rhs.bigcell_GT);

    if (this->flag_tib && this->tau_in_bigcell != nullptr) d2<double>(this->tau_in_bigcell, GlobalC::ucell.nat);
    this->tau_in_bigcell = rhs.tau_in_bigcell;
    rhs.tau_in_bigcell = nullptr;
    this->flag_tib = rhs.flag_tib;
    rhs.flag_tib = false;
    if (this->index_atom != nullptr) delete[] this->index_atom;
    this->index_atom = rhs.index_atom;
    rhs.index_atom = nullptr;
    return *this;
}

Grid_MeshBall& Grid_MeshBall::operator=(Grid_MeshBall&& rhs)
{
    if (this == &rhs)return *this;
    this->Grid_BigCell::operator=(std::move(rhs));

    this->meshball_radius = rhs.meshball_radius;
    this->meshball_ncells = rhs.meshball_ncells;
    
    if (this->flag_mp && this->meshball_positions != nullptr) d2<double>(this->meshball_positions, this->meshball_ncells);
    this->meshball_positions = rhs.meshball_positions;
    rhs.meshball_positions = nullptr;
    this->flag_mp = rhs.flag_mp;
    rhs.flag_mp = false;
    
    if (this->index_ball != nullptr) delete[] this->index_ball;
    this->index_ball = rhs.index_ball;
    rhs.index_ball = nullptr;
    return *this;
}

Grid_Technique& Grid_Technique::operator=(Grid_Technique&& rhs)
{
    if (this == &rhs)return *this;
    this->Grid_MeshBall::operator=(std::move(rhs));
    // so many members to move
    // following the order rather than by type
    if (this->how_many_atoms != nullptr) delete[] this->how_many_atoms;
    this->how_many_atoms = rhs.how_many_atoms;
    rhs.how_many_atoms = nullptr;


    this->max_atom = rhs.max_atom;
    this->total_atoms_on_grid = rhs.total_atoms_on_grid;

    if (this->start_ind != nullptr) delete[] this->start_ind;
    this->start_ind = rhs.start_ind;
    rhs.start_ind = nullptr;

    if (this->bcell_start != nullptr) delete[] this->bcell_start;
    this->bcell_start = rhs.bcell_start;
    rhs.bcell_start = nullptr;

    if (this->which_atom != nullptr) delete[] this->which_atom;
    this->which_atom = rhs.which_atom;
    rhs.which_atom = nullptr;

    if (this->which_bigcell != nullptr) delete[] this->which_bigcell;
    this->which_bigcell = rhs.which_bigcell;
    rhs.which_bigcell = nullptr;

    if (this->which_unitcell != nullptr) delete[] this->which_unitcell;
    this->which_unitcell = rhs.which_unitcell;
    rhs.which_unitcell = nullptr;

    if (this->in_this_processor != nullptr) delete[] this->in_this_processor;
    this->in_this_processor = rhs.in_this_processor;
    rhs.in_this_processor = nullptr;

    this->lnat = rhs.lnat;
    this->lgd = rhs.lgd;

    if (this->trace_lo != nullptr) delete[] this->trace_lo;
    this->trace_lo = rhs.trace_lo;
    rhs.trace_lo = nullptr;

    this->nnrg = rhs.nnrg;

    if (this->nlocdimg != nullptr) delete[] this->nlocdimg;
    this->nlocdimg = rhs.nlocdimg;
    rhs.nlocdimg = nullptr;

    if (this->nlocstartg != nullptr) delete[] this->nlocstartg;
    this->nlocstartg = rhs.nlocstartg;
    rhs.nlocstartg = nullptr;

    if (this->nad != nullptr) delete[] this->nad;
    this->nad = rhs.nad;
    rhs.nad = nullptr;

    if (this->allocate_find_R2)
    {
        if (this->find_R2 != nullptr) d2<int>(this->find_R2, GlobalC::ucell.nat);
        if (this->find_R2_sorted_index != nullptr) d2<int>(this->find_R2_sorted_index, GlobalC::ucell.nat);
        if (this->find_R2st != nullptr) d2<int>(this->find_R2st, GlobalC::ucell.nat);
    }
    this->find_R2 = rhs.find_R2;
    rhs.find_R2 = nullptr;
    this->find_R2_sorted_index = rhs.find_R2_sorted_index;
    rhs.find_R2_sorted_index = nullptr;
    this->find_R2st = rhs.find_R2st;
    rhs.find_R2st = nullptr;
    this->allocate_find_R2 = rhs.allocate_find_R2;
    rhs.allocate_find_R2 = false;

    this->nnrg_index = std::move(rhs.nnrg_index);

    this->maxB1 = rhs.maxB1;
    this->maxB2 = rhs.maxB2;
    this->maxB3 = rhs.maxB3;
    this->minB1 = rhs.minB1;
    this->minB2 = rhs.minB2;
    this->minB3 = rhs.minB3;
    this->nB1 = rhs.nB1;
    this->nB2 = rhs.nB2;
    this->nB3 = rhs.nB3;
    this->nbox = rhs.nbox;
    return *this;
}