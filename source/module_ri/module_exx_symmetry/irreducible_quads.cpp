#include "irreducible_quads.h"
#include <RI/global/Array_Operator.h>
#include <numeric> 

namespace ModuleSymmetry
{
    inline bool sector_eq(const TapR& a, const TapR& b) { return a.first == b.first && a.second == b.second; }
    
    void Irreducible_Quads::find_irsector_invariant_operations(const Irreducible_Sector& irs, const Symmetry& symm)
    {
        const auto& irreducible_sector = irs.get_irreducible_sector();
        for (auto& irap : irreducible_sector)
            for (auto& irR : irap.second)
            {
                const TapR& irapR = { irap.first, irR };
                std::set<int> isym_set;
                for (int isym = 0;isym < symm.nrotk;++isym)
                    if (sector_eq(irs.rotate_apR_by_formula(symm, isym, irapR), irapR))
                        isym_set.insert(isym);
                this->irsector_invariant_ops_[irapR] = isym_set;
            }
        // print 
        std::cout << "sector-invariant operations:" << std::endl;
        for (auto& irs_ops : this->irsector_invariant_ops_)
        {
            std::cout << "irreducible sector (" << irs_ops.first.first.first << ", " << irs_ops.first.first.second << "), R = (" << irs_ops.first.second[0] << ", " << irs_ops.first.second[1] << ", " << irs_ops.first.second[2] << "): ";
            for (auto& isym : irs_ops.second)
                std::cout << isym << " ";
            std::cout << std::endl;
        }
    }

    void Irreducible_Quads::find_irreducible_quads(const Irreducible_Sector& irs,
        const Symmetry& symm, const Atom* atoms, const Statistics& st,
        const std::vector<TC>& Rs, const TC& period, const Lattice& lat)
    {
        using namespace RI::Array_Operator;
        const auto& irreducible_sector = irs.get_irreducible_sector();
        if (irreducible_sector.empty()) { throw std::runtime_error("Please find the irreducible sector first."); }
        
        this->find_irsector_invariant_operations(irs, symm);

        const std::set<TC, Irreducible_Sector::len_less_func> Rs_sorted(Rs.begin(), Rs.end());
        for (auto& IJR_ops : this->irsector_invariant_ops_)
        {
            const TapR& IJR = IJR_ops.first;
            // step1: construct quad set for each irreducible sector IJR
            std::set<Tquad_IJR> all_quads_IJR;   // need a compare func?
            for (int iat1 = 0;iat1 < st.nat;++iat1)
                for (int iat2 = 0; iat2 < st.nat; ++iat2)
                    for (auto& R : Rs_sorted)
                    {
                        const TapR& KLR = { {iat1, iat2}, R };
                        for (const TC& R_IK : Rs_sorted)
                        {
                            const TC& R_JL = R_IK + KLR.second - IJR.second;
                            if (R_JL == R_JL % period)   //in BvK supercell
                                // all_quads_IJR.insert({ KLR, R_IK, R_JL });
                                all_quads_IJR.insert({ KLR, R_IK });
                        }
                    }
            // step2: find irreducible quads
            while (!all_quads_IJR.empty())
            {
                const Tquad_IJR irquad = *all_quads_IJR.begin();
                const TapR& KLR = std::get<0>(irquad);
                const TC& R_IK = std::get<1>(irquad);
                // const TC& R_JL = std::get<2>(irquad);
                std::map<int, Tquad_IJR> quad_star_IJR;
                for (const int& isym : IJR_ops.second)
                {
                    const TapR& KLR_rot = irs.rotate_apR_by_formula(symm, isym, KLR);
                    const TC& R_IK_rot = irs.rotate_R(symm, isym, IJR.first.first, KLR.first.first, R_IK);
                    const TC& R_JL_rot = R_IK_rot + KLR_rot.second - IJR.second;
                    // rot check 
                    const TC& R_JL = R_IK + KLR.second - IJR.second;
                    assert(R_JL_rot == irs.rotate_R(symm, isym, IJR.first.second, KLR.first.second, R_JL));
                    if (R_JL_rot != R_JL_rot % period) { continue; } // exceed BvK supercell after rot
                    // const Tquad_IJR& rotquad = { KLR_rot, R_IK_rot, R_JL_rot };
                    const Tquad_IJR& rotquad = { KLR_rot, R_IK_rot };
                    if (all_quads_IJR.count(rotquad))  // if different isym gives the same quad, only keep one
                    {
                        quad_star_IJR.insert({ isym, rotquad });
                        all_quads_IJR.erase(rotquad);
                        if (all_quads_IJR.empty()) break;
                    }
                }
                if (!quad_star_IJR.empty())
                {
                    this->quads_stars_[IJR].push_back(quad_star_IJR);
                    this->irreducible_quads_weight_[IJR][irquad] = quad_star_IJR.size();
                    if (this->irreducible_quads_.count(IJR))
                        this->irreducible_quads_[IJR].insert(irquad);
                    else
                        this->irreducible_quads_.insert({ IJR, {irquad} });
                }
            }
        }
        // test
        this->print_quads_stars();
        this->print_irreducible_quads();
    }

    inline std::string str_R(const TC& R)
    {
        std::stringstream ss;
        ss << "(" << R[0] << ", " << R[1] << ", " << R[2] << ")";
        return ss.str();
    }
    inline std::string str_sector(const TapR& s)
    {
        std::stringstream ss;
        ss << "atom pair (" << s.first.first << ", " << s.first.second << "), R=" << str_R(s.second);
        return ss.str();
    }
    inline int get_total_size(const std::vector<std::map<int, Tquad_IJR>>& stars)
    {
        return std::accumulate(stars.begin(), stars.end(), 0,
            [](int sum, const std::map<int, Tquad_IJR>& star) { return sum + star.size(); });
    }
    void Irreducible_Quads::print_quads_stars()const
    {
        for (auto& irquad_star_IJRs : this->quads_stars_)
        {
            std::cout << irquad_star_IJRs.second.size() << " quad stars with invariant IJR_ij: " << str_sector(irquad_star_IJRs.first) << "\n";
            for (int istar = 0;istar < irquad_star_IJRs.second.size();++istar)
            {
                std::cout << "quad star " << istar << " with size" << irquad_star_IJRs.second[istar].size() << ":\n";
                for (auto& isym_quad : irquad_star_IJRs.second[istar])
                    std::cout << "isym=" << isym_quad.first
                    << ", KLR = " << str_sector(std::get<0>(isym_quad.second))
                    << ", R_IK=" << str_R(std::get<1>(isym_quad.second)) << "\n";
                // << ", R_JL=" << str_R(std::get<2>(isym_quad.second)) << std::endl;
            }
        }
    }
    void Irreducible_Quads::print_irreducible_quads()const
    {
        for (auto& irquad : this->irreducible_quads_)
        {
            std::cout << "irreducible IJR_ij: " << str_sector(irquad.first) << " has " << irquad.second.size() << " irreducible quads reduced from "
                << get_total_size(this->quads_stars_.at(irquad.first)) << " full quads: \n";
                for (auto& quad : irquad.second)
                std::cout << "KLR = " << str_sector(std::get<0>(quad)) << ", R_IK=" << str_R(std::get<1>(quad)) << "\n";
            // << ", R_JL=" << str_R(std::get<2>(quad)) << std::endl;
        }
    }

}