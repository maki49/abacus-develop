#pragma once
#include "module_io/restart_exx_csr.h"
#include "module_cell/unitcell.h"
#include "module_io/csr_reader.h"
#include "module_io/write_HS_sparse.h"
#include "module_ri/serialization_cereal.h"
#include <RI/global/Tensor.h>
#include <map>
#include <set>
#include <RI/ri/Cell_Nearest.h>
#include "module_ri/RI_Util.h"
namespace ModuleIO
{
    template<typename Tdata>
    void read_Hexxs_csr(const std::string& file_name, const UnitCell& ucell,
        const int nspin, const int nbasis,
        std::vector<std::map<int, std::map<TAC, RI::Tensor<Tdata>>>>& Hexxs)
    {
        ModuleBase::TITLE("Exx_LRI", "read_Hexxs_csr");
        Hexxs.resize(nspin);
        for (int is = 0;is < nspin;++is)
        {
            ModuleIO::csrFileReader<Tdata> csr(file_name + "_" + std::to_string(is) + ".csr");
            int nR = csr.getNumberOfR();
            int nbasis = csr.getMatrixDimension();
            // allocate Hexxs[is]
            for (int iat1 = 0; iat1 < ucell.nat; ++iat1) {
                for (int iat2 = 0;iat2 < ucell.nat;++iat2) {
                    for (int iR = 0;iR < nR;++iR)
                    {
                        const std::vector<int>& R = csr.getRCoordinate(iR);
                        TC dR({ R[0], R[1], R[2] });
                        Hexxs[is][iat1][{iat2, dR}] = RI::Tensor<Tdata>({ static_cast<size_t>(ucell.atoms[ucell.iat2it[iat1]].nw), static_cast<size_t>(ucell.atoms[ucell.iat2it[iat2]].nw) });
                    }
                }
            }
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
            for (int iR = 0;iR < nR;++iR)
            {
                const std::vector<int>& R = csr.getRCoordinate(iR);
                const SparseMatrix<Tdata>& matrix = csr.getMatrix(iR);
                for (auto& ijv : matrix.getElements())
                {
                    const int& npol = ucell.get_npol();
                    const int& i = ijv.first.first * npol;
                    const int& j = ijv.first.second * npol;
                    Hexxs.at(is).at(ucell.iwt2iat[i]).at({ ucell.iwt2iat[j], { R[0], R[1], R[2] } })(ucell.iwt2iw[i] / npol, ucell.iwt2iw[j] / npol) = ijv.second;
                }
            }
        }
    }

    template<typename Tdata>
    void read_Hexxs_cereal(const std::string& file_name, 
        std::vector<std::map<int, std::map<TAC, RI::Tensor<Tdata>>>>& Hexxs)
    {
        ModuleBase::TITLE("Exx_LRI", "read_Hexxs_cereal");
        ModuleBase::timer::tick("Exx_LRI", "read_Hexxs_cereal");
        std::ifstream ifs(file_name, std::ios::binary);
        cereal::BinaryInputArchive iar(ifs);
        iar(Hexxs);
        ModuleBase::timer::tick("Exx_LRI", "read_Hexxs_cereal");
    }

    template<typename Tdata>
    std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, Tdata>>>
        calculate_RI_Tensor_sparse(const double& sparse_threshold,
            const std::map<int, std::map<TAC, RI::Tensor<Tdata>>>& Hexxs,
            const UnitCell& ucell,
            const RI::Cell_Nearest<int, int, 3, double, 3>& cell_nearest)
    {
        ModuleBase::TITLE("Exx_LRI_Interface", "calculate_HContainer_sparse_d");
        std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, Tdata>>> target;
        for (auto& a1_a2R_data : Hexxs)
        {
            int iat1 = a1_a2R_data.first;
            for (auto& a2R_data : a1_a2R_data.second)
            {
                int iat2 = a2R_data.first.first;
                int nw1 = ucell.atoms[ucell.iat2it[iat1]].nw;
                int nw2 = ucell.atoms[ucell.iat2it[iat2]].nw;
                int start1 = ucell.atoms[ucell.iat2it[iat1]].stapos_wf / ucell.get_npol() + ucell.iat2ia[iat1] * nw1;
                int start2 = ucell.atoms[ucell.iat2it[iat2]].stapos_wf / ucell.get_npol() + ucell.iat2ia[iat2] * nw2;

                const TC& R = cell_nearest.get_cell_nearest_discrete(iat1, iat2, a2R_data.first.second);
                auto& matrix = a2R_data.second;
                Abfs::Vector3_Order<int> dR(R[0], R[1], R[2]);
                for (int i = 0;i < nw1;++i) {
                    for (int j = 0;j < nw2;++j) {
                        target[dR][start1 + i][start2 + j] = ((std::abs(matrix(i, j)) > sparse_threshold) ? matrix(i, j) : static_cast<Tdata>(0));
                    }
                }
            }
        }
        return target;
    }
    template<typename Tdata>
    inline std::set<Abfs::Vector3_Order<int>> get_R_range(
        const std::map<int, std::map<TAC, RI::Tensor<Tdata>>>& Hs,
        const RI::Cell_Nearest<int, int, 3, double, 3>& cell_nearest)
    {
        std::set<Abfs::Vector3_Order<int>> all_R_coor;
        for (const auto& H1 : Hs)
        {
            const int iat1 = H1.first;
            for (const auto& H2 : H1.second)
            {
                const int iat2 = H2.first.first;
                all_R_coor.insert(
                    RI_Util::array3_to_Vector3(
                        cell_nearest.get_cell_nearest_discrete(iat1, iat2, H2.first.second)));
            }
        }
        return all_R_coor;
    }

    inline RI::Cell_Nearest<int, int, 3, double, 3> init_cell_nearest(const UnitCell& ucell, const int(&nmp)[3])
    {
        RI::Cell_Nearest<int, int, 3, double, 3> cell_nearest;
        std::map<int, std::array<double, 3>> atoms_pos;
        for (int iat = 0; iat < ucell.nat; ++iat)
        {
            atoms_pos[iat] = RI_Util::Vector3_to_array3(ucell.atoms[ucell.iat2it[iat]].tau[ucell.iat2ia[iat]]);
        }
        const std::array<std::array<double, 3>, 3> latvec
            = { RI_Util::Vector3_to_array3(ucell.a1),
               RI_Util::Vector3_to_array3(ucell.a2),
               RI_Util::Vector3_to_array3(ucell.a3) };
        cell_nearest.init(atoms_pos, latvec, { nmp[0],nmp[1], nmp[2] });
        return cell_nearest;
    }

    template<typename Tdata>
    void write_Hexxs_csr(const std::string& file_name, const UnitCell& ucell,
        const std::vector<std::map<int, std::map<TAC, RI::Tensor<Tdata>>>>& Hexxs,
        const int(&nmp)[3])
    {
        ModuleBase::TITLE("Exx_LRI", "write_Hexxs_csr");
        double sparse_threshold = 1e-10;
        const auto& cell_nearest = init_cell_nearest(ucell, nmp);
        for (int is = 0;is < Hexxs.size();++is)
        {
            std::set<Abfs::Vector3_Order<int>> all_R_coor = get_R_range(Hexxs[is], cell_nearest);
            ModuleIO::save_sparse(
                calculate_RI_Tensor_sparse(sparse_threshold, Hexxs[is], ucell, cell_nearest),
                all_R_coor,
                sparse_threshold,
                false, //binary
                file_name + "_" + std::to_string(is) + ".csr",
                Parallel_Orbitals(),
                "Hexxs_" + std::to_string(is),
                -1,
                false);  //no reduce, one file for each process
        }
    }
}
