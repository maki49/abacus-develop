#include "module_base/constants.h"
#include <cmath>
#include "module_base/parallel_reduce.h"
#include "module_base/scalapack_connector.h"
#include "module_base/tool_title.h"
#include "module_base/timer.h"
#include "symmetry_rotation.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

namespace ModuleSymmetry
{
    void Symmetry_rotation::cal_Ms(const K_Vectors& kv,
        //const std::vector<std::map<int, ModuleBase::Vector3<double>>>& kstars,
        const UnitCell& ucell, const Parallel_2D& pv)
    {
        ModuleBase::TITLE("Symmetry_rotation", "cal_Ms");
        ModuleBase::timer::tick("Symmetry_rotation", "cal_Ms");

        this->nsym_ = ucell.symm.nrotk;
        this->eps_ = ucell.symm.epsilon;

        // 1. calculate the rotation matrix in real spherical harmonics representation for each symmetry operation: [T_l (isym)]_mm'
        std::vector<ModuleBase::Matrix3> gmatc(nsym_);
        for (int i = 0;i < nsym_;++i) gmatc[i] = this->direct_to_cartesian(ucell.symm.gmatrix[i], ucell.latvec);
        this->cal_rotmat_Slm(gmatc.data(), ucell.lmax);

        // 2. calculate the rotation matrix in AO-representation for each ibz_kpoint and symmetry operation: M(k, isym)
        auto restrict_kpt = [](const ModuleBase::Vector3<double>& kvec, const double& symm_prec) -> ModuleBase::Vector3<double>
            {// in (-0.5, 0.5]
                ModuleBase::Vector3<double> kvec_res;
                kvec_res.x = fmod(kvec.x + 100.5 - 0.5 * symm_prec, 1) - 0.5 + 0.5 * symm_prec;
                kvec_res.y = fmod(kvec.y + 100.5 - 0.5 * symm_prec, 1) - 0.5 + 0.5 * symm_prec;
                kvec_res.z = fmod(kvec.z + 100.5 - 0.5 * symm_prec, 1) - 0.5 + 0.5 * symm_prec;
                if (std::abs(kvec_res.x) < symm_prec) kvec_res.x = 0.0;
                if (std::abs(kvec_res.y) < symm_prec) kvec_res.y = 0.0;
                if (std::abs(kvec_res.z) < symm_prec) kvec_res.z = 0.0;
                return kvec_res;
            };
        int nks_ibz = kv.kstars.size(); // kv.nks = 2 * kv.nks_ibz when nspin=2
        this->Ms_.resize(nks_ibz);
        for (int ik_ibz = 0;ik_ibz < nks_ibz;++ik_ibz)
        {
            // const ModuleBase::Vector3<double>& kvec_d_ibz = restrict_kpt((*kstars[ik_ibz].begin()).second * ucell.symm.kgmatrix[(*kstars[ik_ibz].begin()).first], ucell.symm.epsilon);
            for (auto& isym_kvd : kv.kstars[ik_ibz])
                if (isym_kvd.first < nsym_)
                    this->Ms_[ik_ibz][isym_kvd.first] = this->contruct_2d_rot_mat_ao(ucell.symm, ucell.atoms, ucell.st, kv.kvec_d[ik_ibz], isym_kvd.first, pv);
        }

        // output Ms of isym=1
        // std::ofstream ofs("Ms_kibz7_sym7.dat");
        // for (int i = 0;i < pv.get_row_size();++i)
        // {
        //     for (int j = 0;j < pv.get_col_size();++j)
        //     {
        //         ofs << std::setprecision(10) << this->Ms_[7][7][j * pv.get_col_size() + i] << " ";
        //     }
        //     ofs << std::endl;
        // }
        // ofs << std::endl;
        // ofs.close();

        ModuleBase::timer::tick("Symmetry_rotation", "cal_Ms");
    }

    std::vector<std::vector<std::complex<double>>> Symmetry_rotation::restore_dm(const K_Vectors& kv,
        const std::vector<std::vector<std::complex<double>>>& dm_k_ibz, const Parallel_2D& pv)const
    {
        ModuleBase::TITLE("Symmetry_rotation", "restore_dm");
        ModuleBase::timer::tick("Symmetry_rotation", "restore_dm");
        auto vec3_eq = [](const ModuleBase::Vector3<double>& v1, const ModuleBase::Vector3<double>& v2, const double& prec) -> bool
            {
                return (std::abs(v1.x - v2.x) < prec) && (std::abs(v1.y - v2.y) < prec) && (std::abs(v1.z - v2.z) < prec);
            };
        auto  vec_conj = [](const std::vector<std::complex<double>>& z, const double scal = 1.0) -> std::vector<std::complex<double>>
            {
                std::vector<std::complex<double>> z_conj(z.size());
                for (int i = 0;i < z.size();++i) z_conj[i] = std::conj(z[i]) * scal;
                return z_conj;
            };
        std::vector<std::vector<std::complex<double>>> dm_k_full;
        int nspin0 = GlobalV::NSPIN == 2 ? 2 : 1;
        dm_k_full.reserve(kv.nkstot_full * nspin0); //nkstot_full didn'nt doubled by spin
        int nk = kv.nkstot / nspin0;
        for (int is = 0;is < nspin0;++is)
            for (int ik_ibz = 0;ik_ibz < nk;++ik_ibz)
                for (auto& isym_kvd : kv.kstars[ik_ibz])
                    if (isym_kvd.first == 0)
                    {
                        double factor = 1.0 / static_cast<double>(kv.kstars[ik_ibz].size());
                        std::vector<std::complex<double>> dm_scaled(pv.get_local_size());
                        for (int i = 0;i < pv.get_local_size();++i) dm_scaled[i] = factor * dm_k_ibz[ik_ibz + is * nk][i];
                        dm_k_full.push_back(dm_scaled);
                    }
                    else if (vec3_eq(isym_kvd.second, -kv.kvec_d[ik_ibz], this->eps_) && this->TRS_first_)
                        dm_k_full.push_back(vec_conj(dm_k_ibz[ik_ibz + is * nk], 1.0 / static_cast<double>(kv.kstars[ik_ibz].size())));
                    else if (isym_kvd.first < nsym_) //space group operations
                        dm_k_full.push_back(this->rot_matrix_ao(dm_k_ibz[ik_ibz + is * nk], ik_ibz, kv.kstars[ik_ibz].size(), isym_kvd.first, pv));
                    else    // TRS*spacegroup operations
                        dm_k_full.push_back(this->rot_matrix_ao(dm_k_ibz[ik_ibz + is * nk], ik_ibz, kv.kstars[ik_ibz].size(), isym_kvd.first - nsym_, pv, true));


        // test for output
/*
        std::ofstream ofs("DM.dat");
        int ik = 0;
        for (int ikibz = 0;ikibz < kv.nkstot / nspin0;++ikibz)
            for (auto& isym_kvd : kv.kstars[ikibz])
            {
                ofs << "isym=" << isym_kvd.first << std::endl;
                ofs << " k = " << isym_kvd.second.x << " " << isym_kvd.second.y << " " << isym_kvd.second.z << std::endl;
                ofs << "DM(k):" << std::endl;
                for (int i = 0;i < pv.get_row_size();++i)
                {
                    for (int j = 0;j < pv.get_col_size();++j)
                    {
                        ofs << dm_k_full[ik][j * pv.get_row_size() + i] << " ";
                    }
                    ofs << std::endl;
                }
                ++ik;
                ofs << std::endl;
            }
        ofs.close();
*/
        ModuleBase::timer::tick("Symmetry_rotation", "restore_dm");
        return dm_k_full;
    }
    std::vector<std::vector<double>> Symmetry_rotation::restore_dm(const K_Vectors& kv,
        const std::vector<std::vector<double>>& dm_k_ibz, const Parallel_2D& pv)const
    {
        return dm_k_ibz;// do nothing for gamma_only
    }

    // calculate Wigner D matrix
    double Symmetry_rotation::wigner_d(const double beta, const int l, const int m1, const int m2) const
    {
        auto factorial = [](int n) -> int {
            int result = 1;
            for (int i = 1;i <= n;++i) result *= i;
            return result;
            };
        double result = 0.0;
        for (int i = std::max(0, m2 - m1);i <= std::min(l - m1, l + m2);++i)
            result += std::pow(-1, i) * std::sqrt(factorial(l + m1) * factorial(l - m1) * factorial(l + m2) * factorial(l - m2))
            * std::pow(std::cos(beta / 2), 2 * l + m2 - m1 - 2 * i) * std::pow(-std::sin(beta / 2), m1 - m2 + 2 * i)
            / (factorial(i) * factorial(l - m1 - i) * factorial(l + m2 - i) * factorial(i - m2 + m1));
        return result;
    }

    std::complex<double> Symmetry_rotation::wigner_D(const ModuleBase::Vector3<double>& euler_angle, const int l, const int m1, const int m2, const bool inv) const
    {
        std::complex<double> prefac(inv ? std::pow(-1, l) : 1, 0);
        return std::exp(-ModuleBase::IMAG_UNIT * static_cast<double>(m1) * euler_angle.x)
            * std::exp(-ModuleBase::IMAG_UNIT * static_cast<double>(m2) * euler_angle.z)
            * wigner_d(euler_angle.y, l, m1, m2) * prefac;
    }

    // c^l_{m1, m2}=<Y_l^m1|S_l^m2>
    std::complex<double> Symmetry_rotation::ovlp_Ylm_Slm(const int l, const int m1, const int m2) const
    {
        if (m1 == m2)
        {
            if (m1 == 0) return 1.0;
            if (m1 > 0) return 1 / std::sqrt(2);
            if (m1 < 0) return std::pow(-1, m1) * ModuleBase::IMAG_UNIT / std::sqrt(2);
        }
        else if (m1 == -m2)
        {
            if (m1 > 0) return -ModuleBase::IMAG_UNIT / std::sqrt(2);
            if (m1 < 0) return std::pow(-1, m1) / std::sqrt(2);
        }
        return 0.0;
    }

    // reference: https://github.com/minyez/abf_trans/blob/f9e68e68069a94610d89e077bfe6e8ffac0b097d/src/rotate.cpp#L118
    // because the atom position here is row vector, the original gmatrix(eular angle) is transposed.
    // gmatc: the rotation matrix under the basis of cartesian coordinates
    // gmatc should be a rotation matrix, i.e. det(gmatc)=1
    ModuleBase::Vector3<double> Symmetry_rotation::get_euler_angle(const ModuleBase::Matrix3& gmatc) const
    {
        double threshold = 1e-8;
        double alpha, beta, gamma;
        if (std::fabs(gmatc.e32) > threshold || std::fabs(gmatc.e31) > threshold) // sin(beta) is not zero
        {
            // use the 2-angle elements to get alpha and gamma
            alpha = std::atan2(gmatc.e32, gmatc.e31);
            if (alpha < 0) alpha += 2 * ModuleBase::PI;
            gamma = std::atan2(gmatc.e23, -gmatc.e13);
            if (gamma < 0) gamma += 2 * ModuleBase::PI;
            // use the larger one of 2-angle elements to calculate beta
            if (std::fabs(gmatc.e32) > std::fabs(gmatc.e31))
                beta = std::atan2(gmatc.e32 / std::sin(alpha), gmatc.e33);
            else
                beta = std::atan2(gmatc.e31 / std::cos(alpha), gmatc.e33);
        }
        else
        {//sin(beta)=0, beta = 0 or pi, only (alpha+gamma) or (alpha-gamma) is important. now assign this to alpha.
            alpha = std::atan2(gmatc.e12, gmatc.e11);
            if (alpha < 0) alpha += 2 * ModuleBase::PI;
            // if beta=0, gmatc.e11=cos(alpha+gamma), gmatc.e21=sin(alpha+gamma)
            // if beta=pi, gmatc.e11=cos(pi+alpha-gamma), gmatc.e21=sin(pi+alpha-gamma)
            if (gmatc.e33 > 0)
            {
                beta = 0;
                gamma = 0;  //alpha+gamma=alpha => gamma=0
            }
            else
            {
                beta = ModuleBase::PI;
                gamma = ModuleBase::PI;// pi+alpha-gamma=alpha  => gamma=pi
            }
        }
        return ModuleBase::Vector3<double>(alpha, beta, gamma);
    }

    // in: the real value of m in range {-l, -l+1, ..., 0, ..., l-1, l}
    // out: the index of the orbital in a fixed {n， l}, i.e. the index in array [0, 1, -1, 2, -2, ...]
    inline int m2im(int m)
    {
        return (m > 0 ? 2 * m - 1 : -2 * m);
    }

    /// T_mm' = [c^\dagger D c]_mm'
    void Symmetry_rotation::cal_rotmat_Slm(const ModuleBase::Matrix3* gmatc, const int lmax)
    {
        auto set_integer = [](ModuleBase::ComplexMatrix& mat) -> void
            {
                double zero_thres = 1e-10;
                for (int i = 0;i < mat.nr;++i)
                    for (int j = 0;j < mat.nc;++j)
                    {
                        if (std::abs(mat(i, j).real() - std::round(mat(i, j).real())) < zero_thres) mat(i, j).real(std::round(mat(i, j).real()));
                        if (std::abs(mat(i, j).imag() - std::round(mat(i, j).imag())) < zero_thres) mat(i, j).imag(std::round(mat(i, j).imag()));
                    }
            };
        this->rotmat_Slm_.resize(nsym_);
        // c matrix is independent on isym
        std::vector<ModuleBase::ComplexMatrix> c_mm(lmax + 1);
        for (int l = 0;l <= lmax;++l)
        {
            c_mm[l].create(2 * l + 1, 2 * l + 1);
            for (int m1 = -l;m1 <= l;++m1)
                for (int m2 = -l;m2 <= l;++m2)
                    c_mm[l](m2im(m1), m2im(m2)) = ovlp_Ylm_Slm(l, m1, m2);
        }
        for (int isym = 0;isym < nsym_;++isym)
        {
            // if R is a reflection operation, calculate D^l(R)=(-1)^l*D^l(IR), so the euler angle of (IR) is needed.
            ModuleBase::Vector3<double>euler_angle = get_euler_angle(gmatc[isym].Det() > 0 ?
                gmatc[isym] : gmatc[isym] * ModuleBase::Matrix3(-1, 0, 0, 0, -1, 0, 0, 0, -1));

            this->rotmat_Slm_[isym].resize(lmax + 1);
            for (int l = 0;l <= lmax;++l)
            {// wigner D matrix                
                ModuleBase::ComplexMatrix D_mm(2 * l + 1, 2 * l + 1);
                for (int m1 = -l;m1 <= l;++m1)
                    for (int m2 = -l;m2 <= l;++m2)
                        D_mm(m2im(m1), m2im(m2)) = wigner_D(euler_angle, l, m1, m2, (gmatc[isym].Det() < 0));
                this->rotmat_Slm_[isym][l] = transpose(c_mm[l], /*conj=*/true) * D_mm * c_mm[l];
                // set_integer(this->rotmat_Slm_[isym][l]);
            }
        }
/*
        std::vector<ModuleBase::Vector3<double>> euler_angles_test(nsym_);
        for (int isym = 0;isym < nsym_;++isym) euler_angles_test[isym] =
            get_euler_angle(gmatc[isym].Det() > 0 ? gmatc[isym] : gmatc[isym] * ModuleBase::Matrix3(-1, 0, 0, 0, -1, 0, 0, 0, -1));

        auto test_Tmm = [&]()-> void
            {
                std::ofstream ofs("Tlm.dat");
                for (int isym = 0;isym < nsym_;++isym)
                {
                    ofs << "isym=" << isym << std::endl;
                    ofs << "gmatrix_cart=" << std::endl;
                    ofs << gmatc[isym].e11 << " " << gmatc[isym].e12 << " " << gmatc[isym].e13 << std::endl;
                    ofs << gmatc[isym].e21 << " " << gmatc[isym].e22 << " " << gmatc[isym].e23 << std::endl;
                    ofs << gmatc[isym].e31 << " " << gmatc[isym].e32 << " " << gmatc[isym].e33 << std::endl;
                    ofs << "gmatrix_direct=" << std::endl;
                    ofs << GlobalC::ucell.symm.gmatrix[isym].e11 << " " << GlobalC::ucell.symm.gmatrix[isym].e12 << " " << GlobalC::ucell.symm.gmatrix[isym].e13 << std::endl;
                    ofs << GlobalC::ucell.symm.gmatrix[isym].e21 << " " << GlobalC::ucell.symm.gmatrix[isym].e22 << " " << GlobalC::ucell.symm.gmatrix[isym].e23 << std::endl;
                    ofs << GlobalC::ucell.symm.gmatrix[isym].e31 << " " << GlobalC::ucell.symm.gmatrix[isym].e32 << " " << GlobalC::ucell.symm.gmatrix[isym].e33 << std::endl;
                    ofs << "kgmatrix_direct=" << std::endl;
                    ofs << GlobalC::ucell.symm.kgmatrix[isym].e11 << " " << GlobalC::ucell.symm.kgmatrix[isym].e12 << " " << GlobalC::ucell.symm.kgmatrix[isym].e13 << std::endl;
                    ofs << GlobalC::ucell.symm.kgmatrix[isym].e21 << " " << GlobalC::ucell.symm.kgmatrix[isym].e22 << " " << GlobalC::ucell.symm.kgmatrix[isym].e23 << std::endl;
                    ofs << GlobalC::ucell.symm.kgmatrix[isym].e31 << " " << GlobalC::ucell.symm.kgmatrix[isym].e32 << " " << GlobalC::ucell.symm.kgmatrix[isym].e33 << std::endl;
                    ofs << "euler_angle/pi: " << euler_angles_test[isym].x / ModuleBase::PI << " "
                        << euler_angles_test[isym].y / ModuleBase::PI << " " << euler_angles_test[isym].z / ModuleBase::PI << std::endl;
                    for (int l = 0;l <= lmax;++l)
                        for (int i = 0;i < 2 * l + 1;++i)
                        {
                            for (int j = 0;j < 2 * l + 1;++j) ofs << this->rotmat_Slm_[isym][l](i, j) << " ";
                            ofs << std::endl;
                        }
                }
                ofs.close();
            };
        test_Tmm();
    */
    }

    // Perfoming {R|t} to atom position r in the R=0 lattice, we get Rr+t, which may get out of R=0 lattice, 
    // whose image in R=0 lattice is r'=Rr+t-O. This function is to get O for each atom and each symmetry operation.
    // the range of direct position is [-0.5, 0.5).
    ModuleBase::Vector3<double> Symmetry_rotation::get_return_lattice(const Symmetry& symm,
        const ModuleBase::Matrix3& gmatd, const ModuleBase::Vector3<double>gtransd,
        const ModuleBase::Vector3<double>& posd_a1, const ModuleBase::Vector3<double>& posd_a2)const
    {
        auto restrict_center = [&symm](const ModuleBase::Vector3<double>& v) -> ModuleBase::Vector3<double> {
            // in [-0.5, 0.5)
            ModuleBase::Vector3<double> vr;
            vr.x = fmod(v.x + 100.5 + 0.5 * symm.epsilon, 1) - 0.5 - 0.5 * symm.epsilon;
            vr.y = fmod(v.y + 100.5 + 0.5 * symm.epsilon, 1) - 0.5 - 0.5 * symm.epsilon;
            vr.z = fmod(v.z + 100.5 + 0.5 * symm.epsilon, 1) - 0.5 - 0.5 * symm.epsilon;
            if (std::abs(vr.x) < symm.epsilon) vr.x = 0.0;
            if (std::abs(vr.y) < symm.epsilon) vr.y = 0.0;
            if (std::abs(vr.z) < symm.epsilon) vr.z = 0.0;
            return vr;
            };
        auto check_integer = [&symm](const double x) -> void {
            assert(symm.equal(x, std::round(x)));
            };
        ModuleBase::Vector3<double> rotpos1 = restrict_center(posd_a1) * gmatd + restrict_center(gtransd);  // row vector
        ModuleBase::Vector3<double> return_lattice_double = rotpos1 - restrict_center(posd_a2);
#ifdef __DEBUG
        check_integer(return_lattice_double.x);
        check_integer(return_lattice_double.y);
        check_integer(return_lattice_double.z);
#endif
        return ModuleBase::Vector3<double>(std::round(return_lattice_double.x), std::round(return_lattice_double.y), std::round(return_lattice_double.z));
    }

    std::complex<double> Symmetry_rotation::cal_phase_factor_from_return_attice(const Symmetry& symm,
        const ModuleBase::Vector3<double>& pos_a1, const ModuleBase::Vector3<double>& pos_a2,
        int isym, ModuleBase::Vector3<double>kvec_d_ibz)const
    {
        ModuleBase::Vector3<double> return_lattice = get_return_lattice(symm, symm.gmatrix[isym], symm.gtrans[isym], pos_a1, pos_a2);
        if (isym == 1)
        {
            auto restrict_center = [&symm](const ModuleBase::Vector3<double>& v) -> ModuleBase::Vector3<double> {
                // in [-0.5, 0.5)
                ModuleBase::Vector3<double> vr;
                vr.x = fmod(v.x + 100.5 + 0.5 * symm.epsilon, 1) - 0.5 - 0.5 * symm.epsilon;
                vr.y = fmod(v.y + 100.5 + 0.5 * symm.epsilon, 1) - 0.5 - 0.5 * symm.epsilon;
                vr.z = fmod(v.z + 100.5 + 0.5 * symm.epsilon, 1) - 0.5 - 0.5 * symm.epsilon;
                if (std::abs(vr.x) < symm.epsilon) vr.x = 0.0;
                if (std::abs(vr.y) < symm.epsilon) vr.y = 0.0;
                if (std::abs(vr.z) < symm.epsilon) vr.z = 0.0;
                return vr;
                };
        }
        double arg = 2 * ModuleBase::PI * kvec_d_ibz * return_lattice;
        return std::complex<double>(std::cos(arg), std::sin(arg));
    }

    void Symmetry_rotation::set_block_to_mat2d(const int starti, const int startj, const ModuleBase::ComplexMatrix& block, std::vector<std::complex<double>>& obj_mat, const Parallel_2D& pv) const
    {   // caution: ComplaxMatrix is row-major(col-continuous), but obj_mat is col-major(row-continuous)
        for (int j = 0;j < block.nr;++j)//outside dimension
            for (int i = 0;i < block.nc;++i) //inside dimension
                if (pv.in_this_processor(starti + i, startj + j))
                {
                    int index = pv.global2local_col(startj + j) * pv.get_row_size() + pv.global2local_row(starti + i);
                    obj_mat[index] = block(j, i);
                }
    }

    // 2d-block parallized rotation matrix in AO-representation, denoted as M.
    // finally we will use D(k)=M(R, k)^\dagger*D(Rk)*M(R, k) to recover D(k) from D(Rk) in cal_Ms.
    std::vector<std::complex<double>> Symmetry_rotation::contruct_2d_rot_mat_ao(const Symmetry& symm, const Atom* atoms, const Statistics& cell_st,
        const ModuleBase::Vector3<double>& kvec_d_ibz, int isym, const Parallel_2D& pv) const
    {
        std::vector<std::complex<double>> M_isym(pv.get_local_size(), 0.0);
        for (int iat2 = 0;iat2 < cell_st.nat;++iat2)
        {
            int it = cell_st.iat2it[iat2];  // it1=it2
            int ia2 = cell_st.iat2ia[iat2];
            int iat1 = symm.get_rotated_atom(isym, iat2); //iat1=rot(iat2)
            int ia1 = cell_st.iat2ia[iat1];
            std::complex<double>phase_factor = cal_phase_factor_from_return_attice(symm, atoms[it].taud[ia2], atoms[it].taud[ia1], isym, kvec_d_ibz);   //Si2 wrong
            int iw1start = atoms[it].stapos_wf + ia1 * atoms[it].nw;
            int iw2start = atoms[it].stapos_wf + ia2 * atoms[it].nw;
            int iw = 0;
            while (iw < atoms[it].nw)
            {
                int l = atoms[it].iw2l[iw];
                int nm = 2 * l + 1;
                //caution: the order of m in orbitals may be different from increasing
                set_block_to_mat2d(iw1start + iw, iw2start + iw,
                    phase_factor * this->rotmat_Slm_[isym][l], M_isym, pv);
                iw += nm;
            }
        }
        return M_isym;
    }

    // void cal_Ms (kstar), maybe use map to stare Ms

    // D(k) = M^*(R, k) D(k_ibz) M^T(R, k)
    // the link  ik_ibz-isym-ik can be found in kstars.
    std::vector<std::complex<double>> Symmetry_rotation::rot_matrix_ao(const std::vector<std::complex<double>>& DMkibz,
        const int ik_ibz, const int kstar_size, const int isym, const Parallel_2D& pv, const bool TRS_conj) const
    {
        std::vector<std::complex<double>> DMk(pv.nloc, 0.0);
        std::vector<std::complex<double>> DMkibz_M(pv.nloc, 0.0);    // intermediate result
        const char dagger = 'C';
        const char transpose = 'T';
        const char notrans = 'N';
        std::complex<double> alpha(1.0, 0.0);
        const std::complex<double> beta(0.0, 0.0);
        const int nbasis = GlobalV::NLOCAL;
        const int i1 = 1;

        // if TRS_conj, calculate [M^* D M^T]^* = [D^\dagger M^T]^T M^\dagger
        // else, calculate M^* D M^T = [D^T M^\dagger]^T M^T

        // step 1
        pzgemm_(TRS_conj ? &dagger : &transpose, TRS_conj ? &transpose : &dagger, &nbasis, &nbasis, &nbasis,
            &alpha, DMkibz.data(), &i1, &i1, pv.desc, this->Ms_[ik_ibz].at(isym).data(), &i1, &i1, pv.desc,
            &beta, DMkibz_M.data(), &i1, &i1, pv.desc);

        //step 2
        alpha.real(1.0 / static_cast<double>(kstar_size));
        pzgemm_(&transpose, TRS_conj ? &dagger : &transpose, &nbasis, &nbasis, &nbasis,
            &alpha, DMkibz_M.data(), &i1, &i1, pv.desc, this->Ms_[ik_ibz].at(isym).data(), &i1, &i1, pv.desc,
            &beta, DMk.data(), &i1, &i1, pv.desc);

        // but if we store D^*(=D^T, col-maj/row-inside), we sould calculate D^*(k) = M D^*(k_ibz) M^dagger
        // step 1
        // if (TRS_conj)
        //     pzgemm_(&dagger, &dagger, &nbasis, &nbasis, &nbasis,
        //         &alpha, this->Ms_[ik_ibz].at(isym).data(), &i1, &i1, pv.desc, DMkibz.data(), &i1, &i1, pv.desc,
        //         &beta, DMkibz_M.data(), &i1, &i1, pv.desc);
        // else
        //     pzgemm_(&notrans, &notrans, &nbasis, &nbasis, &nbasis,
        //         &alpha, DMkibz.data(), &i1, &i1, pv.desc, this->Ms_[ik_ibz].at(isym).data(), &i1, &i1, pv.desc,
        //         &beta, DMkibz_M.data(), &i1, &i1, pv.desc);

        // //step 2
        // alpha.real(1.0 / static_cast<double>(kstar_size));
        // pzgemm_(TRS_conj ? &transpose : &dagger, TRS_conj ? &transpose : &notrans, &nbasis, &nbasis, &nbasis,
        //     &alpha, this->Ms_[ik_ibz].at(isym).data(), &i1, &i1, pv.desc, DMkibz_M.data(), &i1, &i1, pv.desc,
        //     &beta, DMk.data(), &i1, &i1, pv.desc);

        return DMk;
    }

    ModuleBase::Matrix3 Symmetry_rotation::direct_to_cartesian(const ModuleBase::Matrix3& d, const ModuleBase::Matrix3& latvec)
    {
        return latvec.Inverse() * d * latvec;
    }

}