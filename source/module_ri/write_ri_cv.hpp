#include "module_ri/LRI_CV_Tools.h"
namespace LRI_CV_Tools
{
    template<typename TR>
    TLRI<TR> read_Cs_ao(const std::string& file_path)
    {
        int natom, ncell, ia1, ia2, ic_1, ic_2, ic_3, nw1, nw2, nabf;
        std::ifstream infile;
        infile.open(file_path);
        infile >> natom >> ncell;   // no use of ncell

        TLRI<TR> Cs;
        while (infile.peek() != EOF)
        {
            infile >> ia1 >> ia2 >> ic_1 >> ic_2 >> ic_3 >> nw1 >> nw2 >> nabf;
            const TC& box = { ic_1, ic_2, ic_3 };
            RI::Tensor<TR> tensor_cs({ nabf, nw1, nw2 });
            for (int i = 0; i != nw1; i++) { for (int j = 0; j != nw2; j++) { for (int mu = 0; mu != nabf; mu++) { infile >> tensor_cs(mu, i, j); } } }
            // no screening for data-structure consistency
            // if (loc_atp_index.count(ia1) && (*cs_ptr).absmax() >= threshold)
            Cs[ia1][{ia2, box}] = tensor_cs;
            // else ++cs_discard;
        }
        return Cs;
    }

    template<typename TR>
    void write_Cs_ao(const TLRI<TR>& Cs, const std::string& file_path)
    {
        std::ofstream outfile;
        outfile.open(file_path);
        outfile << Cs.size() << " " << Cs.at(0).size() / Cs.size() << std::endl;    //natom, ncell
        for (auto& it1 : Cs)
        {
            const int& ia1 = it1.first;
            for (auto& it2 : it1.second)
            {
                const int& ia2 = it2.first.first;
                const auto& box = it2.first.second;
                const auto& tensor_cs = it2.second;
                outfile << ia1 << " " << ia2 << " " << box[0] << " " << box[1] << " " << box[2] << std::endl;
                const int& nw1 = tensor_cs.shape[1], nw2 = tensor_cs.shape[2], nabf = tensor_cs.shape[0];
                outfile << nw1 << " " << nw2 << " " << nabf << std::endl;
                for (int i = 0; i != nw1; i++)
                {
                    for (int j = 0; j != nw2; j++)
                    {
                        for (int mu = 0; mu != nabf; mu++) { outfile << tensor_cs(mu, i, j) << " "; }
                        outfile << std::endl;
                    }
                }
            }
        }
    }

    template<typename TR>
    TLRI<TR> read_Vs_abf(const std::string& file_path)
    {
        int natom, ncell, ia1, ia2, ic_1, ic_2, ic_3, nabf1, nabf2;
        std::ifstream infile;
        infile.open(file_path);
        infile >> natom >> ncell;   // no use of ncell

        TLRI<TR> Vs;
        while (infile.peek() != EOF)
        {
            infile >> ia1 >> ia2 >> ic_1 >> ic_2 >> ic_3 >> nabf1 >> nabf2;
            const TC& box = { ic_1, ic_2, ic_3 };
            RI::Tensor<TR> tensor_vs({ nabf1, nabf2 });
            for (int i = 0; i != nabf1; i++)
                for (int j = 0; j != nabf2; j++)
                    infile >> tensor_vs(i, j);
            // no screening for data-structure consistency
            // if (loc_atp_index.count(ia1) && (*cs_ptr).absmax() >= threshold)
            Vs[ia1][{ia2, box}] = tensor_vs;
            // else ++cs_discard;
        }
        return Vs;
    }

    template <typename TR>
    void write_Vs_abf(const TLRI<TR>& Vs, const std::string& file_path)
    {
        std::ofstream outfile;
        outfile.open(file_path);
        outfile << Vs.size() << " " << Vs.at(0).size() / Vs.size() << std::endl;    //natom, ncell
        for (const auto& it1 : Vs)
        {
            const int& ia1 = it1.first;
            for (const auto& it2 : it1.second)
            {
                const int& ia2 = it2.first.first;
                const auto& box = it2.first.second;
                const auto& tensor_v = it2.second;
                outfile << ia1 << " " << ia2 << " " << box[0] << " " << box[1] << " " << box[2] << std::endl;
                outfile << tensor_v.shape[0] << " " << tensor_v.shape[1] << std::endl;
                for (int i = 0; i != tensor_v.shape[0]; i++)
                {
                    for (int j = 0; j != tensor_v.shape[1]; j++)
                        outfile << tensor_v(i, j) << " ";
                    outfile << std::endl;
                }
            }
        }
    }
}