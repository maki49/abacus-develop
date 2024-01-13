#pragma once
#include "module_base/vector3.h"
#include "module_base/complexmatrix.h"
#include "module_base/matrix3.h"
#include <vector>
#include <map>
#include "module_basis/module_ao/parallel_2d.h"
#include "module_cell/unitcell.h" 
#include "module_cell/module_symmetry/symmetry.h"
#include "module_cell/klist.h"
namespace ModuleSymmetry
{
    class Symmetry_rotation
    {
    public:
        Symmetry_rotation() {};
        ~Symmetry_rotation() {};

        /// The top-level calculation interface of this class. calculate the rotation matrix in AO representation: M
        /// only need once call in each ion step (decided by the configuration)
        /// @param kstars  equal k points to each ibz-kpont, corresponding to a certain symmetry operations. 
        void cal_Ms(const K_Vectors& kv,
            //const std::vector<std::map<int, ModuleBase::Vector3<double>>>& kstars,
            const UnitCell& ucell, const Parallel_2D& pv);

        /// Use calculated M matrix to recover D(k) from D(k_ibz): D(k) = M(R, k)^\dagger D(k_ibz) M(R, k)
        /// the link "ik_ibz-isym-ik" can be found in kstars: k_bz = gmat[isym](k)
        std::vector<std::vector<std::complex<double>>>restore_dm(const K_Vectors& kv,
            const std::vector<std::vector<std::complex<double>>>& dm_k_ibz,
            const Parallel_2D& pv)const;
        std::vector<std::vector<double>>restore_dm(const K_Vectors& kv,
            const std::vector<std::vector<double>>& dm_k_ibz,
            const Parallel_2D& pv)const;
        std::vector<std::complex<double>> rot_matrix_ao(const std::vector<std::complex<double>>& DMkibz,
            const int ik_ibz, const int kstar_size, const int isym, const Parallel_2D& pv, const bool TRS_conj = false) const;

        /// calculate Wigner D matrix
        double wigner_d(const double beta, const int l, const int m1, const int m2) const;
        std::complex<double> wigner_D(const ModuleBase::Vector3<double>& euler_angle, const int l, const int m1, const int m2, const bool inv) const;

        /// c^l_{m1, m2}=<Y_l^m1|S_l^m2>
        std::complex<double> ovlp_Ylm_Slm(const int l, const int m1, const int m2) const;

        /// calculate euler angle from rotation matrix
        ModuleBase::Vector3<double> get_euler_angle(const ModuleBase::Matrix3& gmatc) const;

        /// T_mm' = [c^\dagger D c]_mm', the rotation matrix in the representation of real sphere harmonics
        void cal_rotmat_Slm(const ModuleBase::Matrix3* gmatc, const int lmax);

        /// Perfoming {R|t} to atom position r in the R=0 lattice, we get Rr+t, which may get out of R=0 lattice, 
        /// whose image in R=0 lattice is r'=Rr+t-O. This function is to get O for each atom and each symmetry operation.
        /// the range of direct position is [-0.5, 0.5).
        ModuleBase::Vector3<double> get_return_lattice(const Symmetry& symm,
            const ModuleBase::Matrix3& gmatd, const ModuleBase::Vector3<double>gtransd,
            const ModuleBase::Vector3<double>& posd_a1, const ModuleBase::Vector3<double>& posd_a2)const;

        /// exp(-ik_ibz*O)
        std::complex<double> cal_phase_factor_from_return_attice(const Symmetry& symm,
            const ModuleBase::Vector3<double>& pos_a1, const ModuleBase::Vector3<double>& pos_a2,
            int isym, ModuleBase::Vector3<double>kvec_d_ibz) const;

        /// set a block matrix onto a 2d-parallelized matrix, at the position (starti, startj) 
        void set_block_to_mat2d(const int starti, const int startj, const ModuleBase::ComplexMatrix& block, std::vector<std::complex<double>>& obj_mat, const Parallel_2D& pv) const;

        /// 2d-block parallized rotation matrix in AO-representation, denoted as M.
        /// finally we will use D(k)=M(R, k)^\dagger*D(Rk)*M(R, k) to recover D(k) from D(Rk).
        std::vector<std::complex<double>> contruct_2d_rot_mat_ao(const Symmetry& symm, const Atom* atoms, const Statistics& cell_st,
            const ModuleBase::Vector3<double>& kvec_d_ibz, int isym, const Parallel_2D& pv) const;

        ModuleBase::Matrix3 direct_to_cartesian(const ModuleBase::Matrix3& d, const ModuleBase::Matrix3& latvec);

        std::vector<std::vector<ModuleBase::ComplexMatrix>>& get_rotmat_Slm() { return this->rotmat_Slm_; }

    private:

        int nsym_ = 1;

        double eps_ = 1e-6;

        bool TRS_first_ = true; //if R(k)=-k, firstly use TRS to restore D(k) from D(R(k)), i.e conjugate D(R(k)).

        /// the rotation matrix under the basis of S_l^m. size: [nsym][lmax][nm*nm]
        std::vector<std::vector<ModuleBase::ComplexMatrix>> rotmat_Slm_;
        // [natom][nsym], phase factor corresponding to a certain kvec_d_ibz
        // std::vector<std::vector<std::complex<double>>> phase_factor_;

        /// The unitary matrix associate D(Rk) with D(k) for each ibz-kpoint Rk and each symmetry operation. 
        /// size: [nks_ibz][nsym][nbasis*nbasis], only need to calculate once.
        std::vector<std::map<int, std::vector<std::complex<double>>>> Ms_;
    };
}