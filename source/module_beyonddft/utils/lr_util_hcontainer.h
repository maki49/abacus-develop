#pragma once
#include "module_elecstate/module_dm/density_matrix.h"
namespace LR_Util
{
    template<typename TR>
    void print_HR(const hamilt::HContainer<TR>& HR, const int& nat, const std::string& label)
    {
        std::cout << label << "\n";
        for (int ia = 0;ia < nat;ia++)
            for (int ja = 0;ja < nat;ja++)
            {
                auto ap = HR.find_pair(ia, ja);
                for (int iR = 0;iR < ap->get_R_size();++iR)
                {
                    std::cout << "atom pair (" << ia << ", " << ja << "),  "
                        << "R=(" << ap->get_R_index(iR)[0] << ", " << ap->get_R_index(iR)[1] << ", " << ap->get_R_index(iR)[2] << "): \n";
                    auto ptr = ap->get_HR_values(iR).get_pointer();
                    for (int i = 0;i < ap->get_size();++i)std::cout << ptr[i] << " ";
                    std::cout << std::endl;
                }
            }
    }
    template <typename TK, typename TR>
    void print_DMR(const elecstate::DensityMatrix<TK, TR>& DMR, const int& nat, const std::string& label)
    {
        std::cout << label << "\n";
        int is = 0;
        for (auto& dr : DMR.get_DMR_vector())
            print_HR(*dr, nat, "DMR[" + std::to_string(is++) + "]");
    }
    void get_DMR_real_imag_part(const elecstate::DensityMatrix<std::complex<double>, std::complex<double>>& DMR,
        elecstate::DensityMatrix<std::complex<double>, double>& DMR_real,
        const int& nat,
        const char& type = 'R');
    void set_HR_real_imag_part(const hamilt::HContainer<double>& HR_real,
        hamilt::HContainer<std::complex<double>>& HR,
        const int& nat,
        const char& type = 'R');
}