#include "symmetry_rotation.h"
#include "module_ri/RI_Util.h"
#include "module_base/blas_connector.h"
#include <array>
#include <RI/global/Global_Func-2.h>

namespace ModuleSymmetry
{
    template<typename Tdata>
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> Symmetry_rotation::restore_HR(
        const Symmetry& symm, const Atom* atoms, const Statistics& st,
        std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> HR_irreduceble)
    {
        ModuleBase::TITLE("Symmetry_rotation", "restore_HR");
        // get invmap
        // std::vector<int> invmap(symm.nrotk, -1);
        // symm.gmatrix_invmap(symm.gmatrix, symm.nrotk, invmap.data());

        std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> HR_full;

        for (auto& apstar : this->atompair_stars_)
        {
            std::pair<int, int>& irap = apstar.at(0);
            for (auto& ap : apstar)
            {
                const int& iat1 = ap.second.first, & iat2 = ap.second.second;
                for (auto& R_isym_irR : this->final_map_to_irreducible_sector_[iat1 * st.nat + iat2])
                {
                    const std::array<int, 3>& R = R_isym_irR.first;
                    const int& isym = R_isym_irR.second.first;
                    const std::array<int, 3>& irR = R_isym_irR.second.second;
                    // rotate the matrix and pack data
                    // H_12(R)=T^\dagger(V)H_1'2'(VR+O_1-O_2)T(V)
                    HR_full[iat1][{iat2, R}] = rotate_atompair_tensor(HR_irreduceble.at(irap.first).at({ irap.second, irR }), isym, atoms[st.iat2it[irap.first]], atoms[st.iat2it[irap.second]], 'H');
                }
            }
        }
        return HR_full;
    }

    /*
    template<typename Tdata>
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> Symmetry_rotation::restore_HR(
        const Symmetry& symm, const Atom* atoms, const Statistics& st,
        std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> HR_irreduceble)
    {
        ModuleBase::TITLE("Symmetry_rotation", "restore_HR");
        // get invmap
        std::vector<int> invmap(symm.nrotk, -1);
        symm.gmatrix_invmap(symm.gmatrix, symm.nrotk, invmap.data());

        auto it2index_ap = [this](std::vector<std::map<int, std::pair<int, int>>>::iterator it) ->int {return it - this->atompair_stars_.begin(); };
        auto find_iap = [this](std::pair<int, int> ap) -> int
            {
                for (int iap = 0;iap < this->atompair_stars_.size();++iap)
                    if (this->atompair_stars_[iap].at(0) == ap)return iap;
                return -1;
            };
        auto find_iR = [this](int iap, std::array<int, 3> R) -> int
            {
                for (int iR = 0;iR < this->R_stars_[iap].size();++iR)
                    if (this->R_stars_[iap][iR].at(0) == R)return iR;
                return -1;
            };
        auto round2int = [&](const double v)->int
            {return v > 0 ? static_cast<int>(v + symm.epsilon) : static_cast<int>(v - symm.epsilon);};

        auto transform_back_R_method1 = [&](const int isym12, const std::pair<int, int>& irap,
            const std::pair<int, int>& ap0, const std::array<int, 3>& irR)->std::array<int, 3>
        {
            int irat1 = irap.first, irat2 = irap.second;    //irreducible atom pair
            int at1 = ap0.first, at2 = ap0.second;  //original atom pair
            int irt = st.iat2it[irat1], ira1 = st.iat2ia[irat1], ira2 = st.iat2ia[irat2];
            int it = st.iat2it[at1], ia1 = st.iat2ia[at1], ia2 = st.iat2ia[at2];
            ModuleBase::Vector3<int> irRv = RI_Util::array3_to_Vector3(irR);
            ModuleBase::Vector3<double> irRv_double(irRv.x, irRv.y, irRv.z);
            ModuleBase::Vector3<double> R_back_double =
                (atoms[irt].taud[ira1] - atoms[irt].taud[ira2] + irRv_double) * symm.gmatrix[invmap[isym12]]
                - (atoms[it].taud[ia1] - atoms[it].taud[ia2]);
#ifdef __DEBUG
            assert(symm.equal(R_back_double.x, std::round(R_back_double.x)));
            assert(symm.equal(R_back_double.y, std::round(R_back_double.y)));
            assert(symm.equal(R_back_double.z, std::round(R_back_double.z)));
#endif
            ModuleBase::Vector3<int> R_back(round2int(R_back_double.x),
                round2int(R_back_double.y), round2int(R_back_double.z));
            return RI_Util::Vector3_to_array3(R_back);
        };

        auto transform_back_R_method2 = [&, this](const int isym12,
            const std::pair<int, int>& ap0, const std::array<int, 3>& irR)->std::array<int, 3>
        {
            int at1 = ap0.first, at2 = ap0.second;  //original atom pair
            // int it = st.iat2it[at1], ia1 = st.iat2ia[at1], ia2 = st.iat2ia[at2];
            ModuleBase::Vector3<int> irRv = RI_Util::array3_to_Vector3(irR);
            ModuleBase::Vector3<double> irRv_double(irRv.x, irRv.y, irRv.z);
            ModuleBase::Vector3<double> R_back_double =
                (irRv_double + this->return_lattice_[at2][isym12] - this->return_lattice_[at1][isym12])
                * symm.gmatrix[invmap[isym12]];
#ifdef __DEBUG
            assert(symm.equal(R_back_double.x, std::round(R_back_double.x)));
            assert(symm.equal(R_back_double.y, std::round(R_back_double.y)));
            assert(symm.equal(R_back_double.z, std::round(R_back_double.z)));
#endif
            ModuleBase::Vector3<int> R_back(round2int(R_back_double.x),
                round2int(R_back_double.y), round2int(R_back_double.z));
            return RI_Util::Vector3_to_array3(R_back);
        };


        std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> HR_full;

        // traverse irreducible {IJR}
        for (auto& iHR_ia1 : HR_irreduceble)
        {
            int iat1 = iHR_ia1.first;
            for (auto& iHR_ia12R : iHR_ia1.second)
            {
                int iat2 = iHR_ia12R.first.first;
                std::array<int, 3> irR = iHR_ia12R.first.second;
                // ModuleBase::Vector3<int> R = RI_Util::array3_to_Vector3(iHR_ia12R.first.second);
                std::pair<int, int> irap = { iat1, iat2 };    //irreducible atom pair
                int iap = find_iap(irap);
                if (iap >= 0) //irreducible atom pair exists
                {
                    for (auto& isym_ap : this->atompair_stars_[iap])// traverse the atom pair star of irreducible atom pair
                    {
                        int isym1 = isym_ap.first;
                        std::pair<int, int> ap = isym_ap.second;
                        int iap_all = ap.first * st.nat + ap.second;
                        int iR = find_iR(iap_all, irR);
                        if (iR >= 0)// irreducible R in atom pair exists
                        {
                            for (auto& isym_R : this->R_stars_[iap_all][iR]) //traverse the R star of irreducible R in atom pair
                            {
                                int isym2 = isym_R.first;
                                // std::array<int, 3> R_in_star = isym_R.second;
                                // ModuleBase::Vector3<int> irR = RI_Util::array3_to_Vector3(isym_R.second);

                                // get the original R
                                int isym12 = group_multiply(symm, isym1, isym2);
                                std::array<int, 3> R = transform_back_R_method2(isym12, ap, irR);
#ifdef __DEBUG
                                std::array<int, 3> R_ref = transform_back_R_method1(isym12, irap, ap, irR);
                                assert(R == R_ref);
#endif
                                // rotate the matrix and pack data
                                // H_12(R)=T^\dagger(V)H_1'2'(VR+O_1-O_2)T(V)
                                HR_full[ap.first][{ap.second, R}] = rotate_atompair_tensor(iHR_ia12R.second, invmap[isym12], atoms[st.iat2it[iat1]], atoms[st.iat2it[iat2]],'H');
                            }
                        }
                    }
                }
            }
        }
        return HR_full;
    }
    */
    template<typename Tdata>
    RI::Tensor<Tdata> Symmetry_rotation::rotate_atompair_tensor(const RI::Tensor<Tdata>& A, const int isym,
        const Atom& a1, const Atom& a2, const char mode)
    {   // due to col-contiguous, actually what we know is T^T and H^T (or D^T), 
        // and what we calculate is(H'^T = T ^ T * H ^ T * T^*) or (D'^T = T ^ \dagger * D ^ T * T)
        auto set_block = [](const int starti, const int startj, const ModuleBase::ComplexMatrix& block,
            RI::Tensor<std::complex<double>>& obj_tensor)->void
            {   // both ComplexMatrix and RI::Tensor are row-major (col-contiguous)
                for (int i = 0;i < block.nr;++i)
                    for (int j = 0;j < block.nc;++j)
                        obj_tensor(starti + i, startj + j) = block(i, j);
            };
        auto set_rotation_matrix = [&, this](const Atom& a) -> RI::Tensor<std::complex<double>>
            {
                RI::Tensor<std::complex<double>> T({ static_cast<size_t>(a.nw), static_cast<size_t>(a.nw) }); // check if zero
                int iw = 0;
                while (iw < a.nw)
                {
                    int l = a.iw2l[iw];
                    int nm = 2 * l + 1;
                    set_block(iw, iw, this->rotmat_Slm_[isym][l], T);
                    iw += nm;
                }
                return T;
            };

        bool sametype = (a1.type == a2.type);
        assert(A.shape[0] == a2.nw);//col
        assert(A.shape[1] == a1.nw);//row
        // contrut T matrix 
        const RI::Tensor<std::complex<double>>& T1 = set_rotation_matrix(a1);
        const RI::Tensor<std::complex<double>>& T2 = sametype ? T1 : set_rotation_matrix(a2);

        // A*T_2 (atom 2 is contiguous)
        const char notrans = 'N', transpose = 'T', dagger = 'C';
        const std::complex<double> alpha(1.0, 0.0), beta(0.0, 0.0);
        const RI::Tensor<std::complex<double>>& A_complex = RI::Global_Func::convert<std::complex<double>>(A);

        RI::Tensor<std::complex<double>> TAT(A.shape);
        RI::Tensor<std::complex<double>> AT2(A.shape);
        if (mode == 'H')
        {   // H'^T = T2^T * H^T * T1^*
            BlasConnector::gemm(notrans, notrans, a2.nw, a1.nw, a2.nw, alpha, T2.ptr(), a2.nw, A_complex.ptr(), a2.nw, beta, AT2.ptr(), a2.nw);
            BlasConnector::gemm(notrans, dagger, a2.nw, a1.nw, a1.nw, alpha, AT2.ptr(), a2.nw, T1.ptr(), a1.nw, beta, TAT.ptr(), a2.nw);
        }
        else if (mode == 'D')
        {   //T2^\dagger * D^T * T1 = [(D^T)^T * (T2^T)^\dagger]^T * (T1^T)^T
            BlasConnector::gemm(transpose, dagger, a1.nw, a2.nw, a2.nw, alpha, A_complex.ptr(), a2.nw, T2.ptr(), a2.nw, beta, AT2.ptr(), a1.nw);
            BlasConnector::gemm(transpose, transpose, a2.nw, a1.nw, a1.nw, alpha, AT2.ptr(), a1.nw, T1.ptr(), a1.nw, beta, TAT.ptr(), a2.nw);
        }
        else throw std::invalid_argument("Symmetry_rotation::rotate_atompair_tensor: invalid mode.");
        return RI::Global_Func::convert<Tdata>(TAT);
    }

    template<typename Tdata>
    void Symmetry_rotation::test_HR_rotation(const Symmetry& symm, const Atom* atoms, const Statistics& st,
        const std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>>& HR_full)
    {
        ModuleBase::TITLE("Symmetry_rotation", "test_HR_rotation");

        auto print_tensor = [](const RI::Tensor<Tdata>& t, const std::string& name)
            {
                std::cout << name << ":\n";
                for (int i = 0;i < t.shape[0];++i)
                {
                    for (int j = 0;j < t.shape[1];++j)
                        std::cout << t(i, j) << " ";
                    std::cout << std::endl;
                }
            };

        // 1. pick out H(R) in the irreducible sector from full H(R)
        std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> HR_irreduceble;
        for (int iap = 0;iap < this->atompair_stars_.size();++iap)
        {
            std::pair<int, int> irap = this->atompair_stars_[iap].at(0);
            for (int iR = 0;iR < this->R_stars_[iap].size();++iR)
            {
                std::array<int, 3> irR = this->R_stars_[iap][iR].at(0);
                std::pair<int, std::array<int, 3>> a2_irR = { irap.second, irR };
                HR_irreduceble[irap.first][a2_irR]
                    = HR_full.at(irap.first).at(a2_irR);
            }
        }
        // 2. rotate
        std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> HR_rotated = restore_HR(symm, atoms, st, HR_irreduceble);
        // 3. compare
        for (auto& HR_ia1 : HR_rotated)
        {
            int iat1 = HR_ia1.first;
            for (auto& HR_ia12R : HR_ia1.second)
            {
                int iat2 = HR_ia12R.first.first;
                std::array<int, 3> R = HR_ia12R.first.second;
                const RI::Tensor<Tdata>& HR_rot = HR_ia12R.second;
                if (HR_full.at(iat1).count({ iat2, R }) == 0)// rot back but not found
                {
                    std::cout << "R_rot not found in atom pair (" << iat1 << ", " << iat2 << "):  R=(" << R[0] << "," << R[1] << "," << R[2] << "):\n";
                    continue;
                }
                const RI::Tensor<Tdata>& HR_ref = HR_full.at(iat1).at({ iat2, R });
                assert(HR_rot.shape[0] == HR_ref.shape[0]);
                assert(HR_rot.shape[1] == HR_ref.shape[1]);
                // output 
                if (iat1 == 0 && iat2 == 0) //&& R[0] == 0 && R[1] == -1 && R[2] == 0)
                {
                    std::cout << "atom pair (" << iat1 << ", " << iat2 << "), R=(" << R[0] << "," << R[1] << "," << R[2] << "):\n";
                    print_tensor(HR_rot, "HR_rot");
                    print_tensor(HR_ref, "HR_ref");
                }
            }
        }

    }
}