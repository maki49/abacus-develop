#include "symmetry_rotation.h"
#include <set>
#include "RI_Util.h"
namespace ModuleSymmetry
{
    int Symmetry_rotation::group_multiply(const Symmetry& symm, const int isym1, const int isym2) const
    {   // row_vec * gmat1*gmat2
        ModuleBase::Matrix3 g12 = symm.gmatrix[isym1] * symm.gmatrix[isym2];
        int isym = 0;
        for (;isym < symm.nrotk;++isym)
            if (symm.equal(g12.e11, symm.gmatrix[isym].e11) && symm.equal(g12.e12, symm.gmatrix[isym].e12) && symm.equal(g12.e13, symm.gmatrix[isym].e13) &&
                symm.equal(g12.e21, symm.gmatrix[isym].e21) && symm.equal(g12.e22, symm.gmatrix[isym].e22) && symm.equal(g12.e23, symm.gmatrix[isym].e23) &&
                symm.equal(g12.e31, symm.gmatrix[isym].e31) && symm.equal(g12.e32, symm.gmatrix[isym].e32) && symm.equal(g12.e33, symm.gmatrix[isym].e33))
                break;
        return isym;
    }

    int Symmetry_rotation::round2int(const double x) const
    {
        return x > 0 ? static_cast<int>(x + eps_) : static_cast<int>(x - eps_);
    }

    std::array<int, 3> Symmetry_rotation::rotate_R_by_formula(const Symmetry& symm,
        const int isym, const int iat1, const int iat2, const std::array<int, 3>& R, const char gauge) const
    {
        const ModuleBase::Vector3<double> R_double(static_cast<double>(R[0]), static_cast<double>(R[1]), static_cast<double>(R[2]));
        const ModuleBase::Vector3<double> Rrot_double = (gauge == 'L')
            ? R_double * symm.gmatrix[isym] + this->return_lattice_[iat1][isym] - this->return_lattice_[iat2][isym]
            : R_double * symm.gmatrix[isym] + this->return_lattice_[iat2][isym] - this->return_lattice_[iat1][isym];
        return { round2int(Rrot_double.x), round2int(Rrot_double.y), round2int(Rrot_double.z) };
    }

    ModuleBase::Vector3<double> Symmetry_rotation::get_aRb_direct(const Atom* atoms, const Statistics& st,
        const int iat1, const int iat2, const ModuleBase::Vector3<double>& R, const char gauge)const
    {
        return atoms[st.iat2it[iat1]].taud[st.iat2ia[iat1]] - atoms[st.iat2it[iat2]].taud[st.iat2ia[iat2]] + (gauge == 'L' ? R : -R);
    }

    ModuleBase::Vector3<double> Symmetry_rotation::get_aRb_direct(const Atom* atoms, const Statistics& st,
        const int iat1, const int iat2, const std::array<int, 3>& R, const char gauge) const
    {
        const ModuleBase::Vector3<double> R_double(static_cast<double>(R[0]), static_cast<double>(R[1]), static_cast<double>(R[2]));
        return get_aRb_direct(atoms, st, iat1, iat2, R_double);
    }

    void Symmetry_rotation::find_irreducible_atom_pairs(const Symmetry& symm)
    {
        ModuleBase::TITLE("Symmetry_rotation", "find_irreducible_atom_pairs");
        this->eps_ = symm.epsilon;
        for (int iat1 = 0;iat1 < symm.nat;++iat1)
            for (int iat2 = 0;iat2 < symm.nat;++iat2)
            {
                std::pair<int, int> pair = { iat1, iat2 };
                bool exist = false;
                for (int isym = 0;isym < symm.nrotk;++isym)
                {
                    std::pair<int, int> rotpair = { symm.get_rotated_atom(isym,iat1), symm.get_rotated_atom(isym,iat2) };
                    for (int ip = 0;ip < this->atompair_stars_.size();++ip) // current irreduceble pairs
                    {
                        if (rotpair == this->atompair_stars_[ip].at(0))
                        {
                            this->atompair_stars_[ip].insert({ isym, pair });
                            exist = true;
                            break;
                        }
                    }
                    if (exist)break;
                }
                if (!exist)this->atompair_stars_.push_back({ {0, pair} });
            }
    }

    struct ap_less_func
    {
        bool operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const
        {
            if (lhs.first < rhs.first)return true;
            else if (lhs.first > rhs.first)return false;
            else return lhs.second < rhs.second;
        }
    };
    void Symmetry_rotation::find_irreducible_atom_pairs_set(const Symmetry& symm)
    {
        ModuleBase::TITLE("Symmetry_rotation", "find_irreducible_atom_pairs_set");
        this->eps_ = symm.epsilon;
        std::vector<int> invmap(symm.nrotk, -1);
        symm.gmatrix_invmap(symm.gmatrix, symm.nrotk, invmap.data());
        // contruct initial ap-set
        std::set<std::pair<int, int>, ap_less_func> ap_set;
        for (int iat1 = 0; iat1 < symm.nat; ++iat1)
            for (int iat2 = 0; iat2 < symm.nat; ++iat2)
                ap_set.insert({ iat1, iat2 });
        while (!ap_set.empty())
        {
            std::pair<int, int> ap = *ap_set.begin();
            std::map<int, std::pair<int, int>> ap_star;
            for (int isym = 0; isym < symm.nrotk; ++isym)
            {
                std::pair<int, int> rotpair = { symm.get_rotated_atom(isym,ap.first), symm.get_rotated_atom(isym,ap.second) };
                if (ap_set.find(rotpair) != ap_set.end())
                {
                    ap_star.insert({ invmap[isym], rotpair });
                    ap_set.erase(rotpair);
                }
            }
            this->atompair_stars_.push_back(ap_star);
        }
    }

    inline void output_atompair_stars(const std::vector<std::map<int, std::pair<int, int>>>& ap_stars)
    {
        std::cout << "stars of irreducible atom pairs: " << std::endl;
        for (int ip = 0; ip < ap_stars.size(); ++ip)
        {
            std::cout << "atom pair star " << ip << ": " << std::endl;
            for (const auto& ap : ap_stars[ip])
                std::cout << "isym=" << ap.first << ", atompair=(" << ap.second.first << ", " << ap.second.second << ") " << std::endl;
        }
    }

    void Symmetry_rotation::test_irreducible_atom_pairs(const Symmetry& symm)
    {
        std::cout << "Algorithm 1: find irreducible atom pairs by set" << std::endl;
        this->find_irreducible_atom_pairs_set(symm);
        output_atompair_stars(this->atompair_stars_);

        std::cout << std::endl;
        this->atompair_stars_.clear();
        std::cout << "Algorithm 2: find irreducible atom pairs by judge" << std::endl;
        this->find_irreducible_atom_pairs(symm);
        output_atompair_stars(this->atompair_stars_);
    }


    void Symmetry_rotation::find_irreducible_R(const Symmetry& symm, const Atom* atoms, const Statistics& st, const K_Vectors& kv)
    {
        ModuleBase::TITLE("Symmetry_rotation", "find_irreducible_R");
        std::array<int, 3> period = RI_Util::get_Born_vonKarmen_period(kv);
        std::vector<std::array<int, 3>> BvK_cells = RI_Util::get_Born_von_Karmen_cells(period);

        this->R_stars_.resize(st.nat * st.nat);

        auto no_rotation = [](const int isym, const int iat1, const int iat2, const std::array<int, 3>& R)-> int
            {return isym; };
        auto rotate_to_current_irR = [&st, atoms, &symm, this](const int isym, const int iat1, const int iat2, const std::array<int, 3>& R)-> int
            {
                const ModuleBase::Vector3<double> aRb = this->get_aRb_direct(atoms, st, iat1, iat2, R);
                for (auto& Rstar : this->R_stars_[iat1 * st.nat + iat2])
                {
                    const std::array<int, 3>& R_dest = Rstar.at(0);
                    const ModuleBase::Vector3<double> aRb_dest = this->get_aRb_direct(atoms, st, iat1, iat2, R_dest);
                    // If current R int Rstar_append can be rotated into Rstar of the same irreduceble atom pair, 
                    // update isym: V=V1*V2. Finally only the Rs with V=0 will be added to the irreducible sector.
                    if (symm.equal((aRb_dest * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det(),
                        (aRb * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det()))
                        for (int isym_to_iR = 0;isym_to_iR < symm.nrotk;++isym_to_iR)
                        {
                            ModuleBase::Vector3<double>  aRb_rot = aRb * symm.gmatrix[isym_to_iR];
                            if (symm.equal(aRb_rot.x, aRb_dest.x) && symm.equal(aRb_rot.y, aRb_dest.y) && symm.equal(aRb_rot.z, aRb_dest.z))
                                return this->group_multiply(symm, isym, isym_to_iR);
                        }
                }

                // test
                if ((aRb * GlobalC::ucell.latvec).norm() > 10) std::cout << "isym = " << isym << ", aRb length=" << (aRb * GlobalC::ucell.latvec).norm() << std::endl;
                return isym;
            };
        auto add_R_to_Rstar = [&, this](std::vector<std::map<int, std::array<int, 3>>>& Rstars, const int iat1, const int iat2, const std::array<int, 3>& R,
            std::function<int(const int, const int, const int, const std::array<int, 3>&) > rotate_first)
            {
                const ModuleBase::Vector3<double> aRb_d = this->get_aRb_direct(atoms, st, iat1, iat2, R);
                bool exist = false;
                for (int iiR = 0;iiR < Rstars.size();++iiR)    // for each R_star in current atom pair
                { // compare with the aRb of irreducible R (current R_star[isym==0]) of current atom pair
                    // std::array<int, 3>& irR = Rstars[iiR].at(0);
                    const int isym0 = Rstars[iiR].begin()->first;
                    std::array<int, 3>& irR = Rstars[iiR].begin()->second;
                    ModuleBase::Vector3<double> aRb_d_irR = this->get_aRb_direct(atoms, st, iat1, iat2, irR);
                    if (symm.equal((aRb_d_irR * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det(),
                        (aRb_d * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det()))   // the  length of aRb in the same Rstar should be the same
                    {
                        for (int isym = 0; isym < symm.nrotk; ++isym)
                        {
                            ModuleBase::Vector3<double> rot_aRb_d = aRb_d * symm.gmatrix[isym];
                            if (symm.equal(rot_aRb_d.x, aRb_d_irR.x) &&
                                symm.equal(rot_aRb_d.y, aRb_d_irR.y) &&
                                symm.equal(rot_aRb_d.z, aRb_d_irR.z))
                            {
                                exist = true;
                                Rstars[iiR].insert({ isym0 == 0 ? isym : this->group_multiply(symm,isym,isym0), R });
                                break;
                            }
                        }// isym
                        if (exist) break;
                    }
                }// Rstar
                if (!exist) Rstars.push_back({ {rotate_first(0, iat1,iat2,R), R} });
            };

        // 1. find Rstars for each atom pair
        for (int iat1 = 0;iat1 < st.nat;++iat1)
            for (int iat2 = 0;iat2 < st.nat;++iat2)
                for (auto& R : BvK_cells) add_R_to_Rstar(this->R_stars_[iat1 * st.nat + iat2], iat1, iat2, R, no_rotation);

        // 2. find irreducible sector
        // 2.1. contruct appendix Rstar: irreducible R in other atom pairs that cannot rotate into R_stars_[irreducebule_ap]
        auto exceed_range = [&](const std::array<int, 3>& irR)->bool
            {
                for (auto& cell : BvK_cells) if (irR == cell) return false;
                return true;
            };
        for (auto apstar : this->atompair_stars_)
        {
            for (std::map<int, std::pair<int, int>>::iterator it = ++apstar.begin();it != apstar.end();++it)
            {
                const  int isym_to_irap = it->first;
                const std::pair<int, int>& ap = it->second;
                const std::pair<int, int>& irap = apstar.begin()->second;
                for (auto& Rstar : R_stars_[ap.first * st.nat + ap.second])
                {
                    const std::array<int, 3>& irR = Rstar.at(0);
                    const std::array<int, 3> irR_rot_to_irap = this->rotate_R_by_formula(symm, isym_to_irap, ap.first, ap.second, irR);// R'=VR+O_1-O_2
                    // if irR_rot exceeds the range of Rstar, put it into irreducible sector and the appendix Rstar
                    if (exceed_range(irR_rot_to_irap))
                        if (this->R_stars_irap_append_.find(irap) == this->R_stars_irap_append_.end())
                            this->R_stars_irap_append_[irap] = { {{{rotate_to_current_irR(0,irap.first,irap.second, irR_rot_to_irap), irR_rot_to_irap}}} };
                        else
                            add_R_to_Rstar(this->R_stars_irap_append_.at(irap), irap.first, irap.second, irR_rot_to_irap, rotate_to_current_irR);
                }// Rstar
            }// atom pair
        }// atom pair star

        // 2.2. add all the irreducible Rs (including appendix) in irreducible atom pairs
        for (auto& apstar : this->atompair_stars_)
        {
            std::pair<int, int> irap = apstar.at(0);
            for (auto& Rstar : R_stars_[irap.first * st.nat + irap.second])
                this->irreducible_sector_[irap].insert(Rstar.at(0));
            if (this->R_stars_irap_append_.find(irap) != this->R_stars_irap_append_.end())
                for (auto& Rstar : this->R_stars_irap_append_.at(irap))
                    if (Rstar.begin()->first == 0)  // only the Rs forming new stars
                        this->irreducible_sector_[irap].insert(Rstar.begin()->second);
        }
    }

    void Symmetry_rotation::output_irreducible_R(const K_Vectors& kv, const Atom* atoms, const Statistics& st)
    {
        // output BvK cells
        std::array<int, 3> period = RI_Util::get_Born_vonKarmen_period(kv);
        std::vector<std::array<int, 3>> BvK_cells = RI_Util::get_Born_von_Karmen_cells(period);

        std::cout << "Number of irreducible atom pairs: " << this->atompair_stars_.size() << std::endl;
        std::cout << "Irreducible atom pairs: " << std::endl;
        for (int iap = 0; iap < this->atompair_stars_.size(); ++iap)
            std::cout << " (" << this->atompair_stars_[iap].at(0).first << ", "
            << this->atompair_stars_[iap].at(0).second << "), " << std::endl;
        std::cout << std::endl;

        auto print_Rstars = [&st, atoms, this](const std::vector<std::map<int, std::array<int, 3>>>& Rstars, const int iat1, const int iat2)
            {
                for (int iiR = 0; iiR < Rstars.size(); ++iiR)
                {
                    std::cout << "R_star " << iiR << ": cartesian length of aRb = "
                        << (this->get_aRb_direct(atoms, st, iat1, iat2, Rstars[iiR].begin()->second) * GlobalC::ucell.latvec).norm() << ":" << std::endl;
                    for (const auto& R : Rstars[iiR])
                        std::cout << "isym=" << R.first << ", R=(" << R.second[0] << ", " << R.second[1] << ", " << R.second[2] << ") " << std::endl;
                }
                std::cout << std::endl;
            };
        std::cout << "Rstar in each atom pair: " << std::endl;
        for (int iat1 = 0;iat1 < st.nat;++iat1)
            for (int iat2 = 0;iat2 < st.nat;++iat2)
            {
                std::cout << "in atompair (" << iat1 << ", " << iat2 << "):" << std::endl;
                print_Rstars(this->R_stars_[iat1 * st.nat + iat2], iat1, iat2);
            }

        std::cout << "appendix Rstar:" << std::endl;
        for (auto& ap_Rsapd : this->R_stars_irap_append_)
        {
            std::pair<int, int> irap = ap_Rsapd.first;
            std::cout << "In irreducible atom pair: (" << irap.first << ", " << irap.second << "): " << std::endl;
            print_Rstars(ap_Rsapd.second, irap.first, irap.second);
        }

        std::cout << "Irreducible Sector: " << std::endl;
        for (auto& irap_Rs : this->irreducible_sector_)
        {
            std::pair<int, int> irap = irap_Rs.first;
            std::cout << "In irreducible atom pair: (" << irap.first << ", " << irap.second << "): " << std::endl;
            for (auto& R : irap_Rs.second)
                std::cout << "R=(" << R[0] << ", " << R[1] << ", " << R[2] << ") " << std::endl;
        }
    }

    void Symmetry_rotation::get_final_map_to_irreducible_sector(const Symmetry& symm, const Atom* atoms, const Statistics& st)
    {
        ModuleBase::TITLE("Symmetry_rotation", "get_final_map_to_irreducible_sector");
        this->final_map_to_irreducible_sector_.resize(st.nat * st.nat);
        for (auto& apstar : this->atompair_stars_)
            for (auto& ap : apstar)
            {
                int iat1 = ap.second.first, iat2 = ap.second.second;
                int iap = iat1 * st.nat + iat2;
                // for irreducible atom pairs, the map is by its symmetry operation to the irreduceble R in its star
                if (ap.first == 0)
                    for (auto& Rstar : this->R_stars_[iap])
                        for (auto& isym_R : Rstar)
                            this->final_map_to_irreducible_sector_[iap][isym_R.second] = { isym_R.first, Rstar.at(0) };
                else
                {    // for other atom pairs, the map is:  [ab,R]->V2_ab->[ab,irR_ap]->V1->[a'b',R'_irap]->V2_a'b'->[a'b',irR'_irap]
                    std::pair<int, int> irap = apstar.at(0);
                    const int& isym_to_irap = ap.first;  // V1
                    for (auto& Rstar : this->R_stars_[iap])
                        for (auto& symR_ap_R : Rstar)
                        {
                            const int& isymR_ap = symR_ap_R.first;  //V2_ab
                            const std::array<int, 3>& R = symR_ap_R.second;
                            const std::array<int, 3>& irR = Rstar.begin()->second;
                            const std::array<int, 3> irR_rot_to_irap = this->rotate_R_by_formula(symm, isym_to_irap, iat1, iat2, irR);// R'=VR+O_1-O_2
                            const ModuleBase::Vector3<double> aRb_irap = this->get_aRb_direct(atoms, st, irap.first, irap.second, irR_rot_to_irap);// s_a' + R' - s_b

                            // 1. try Rstars_[irap]
                            bool found = false;
                            for (auto& Rstar_irap : this->R_stars_[irap.first * st.nat + irap.second])
                            {
                                const std::array<int, 3>& irR_irap = Rstar_irap.begin()->second;
                                const ModuleBase::Vector3<double> aRb_irR_irap = this->get_aRb_direct(atoms, st, irap.first, irap.second, irR_irap); // s_a' + irR' - s_b'
                                if (symm.equal((aRb_irR_irap * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det(),
                                    (aRb_irap * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det()))   // the  length of aRb in the same Rstar should be the same
                                    for (auto& symR_irap_R : Rstar_irap)
                                        if (symR_irap_R.second == irR_rot_to_irap) // found R' in irreducible atom pair's Rstar, the corresponding R in ireducible sector is the ireducible R of current Rstar.
                                        {
                                            const int& isymR_irap = symR_irap_R.first;  //V2_a'b'
                                            const int isym = this->group_multiply(symm, group_multiply(symm, isymR_ap, isym_to_irap), isymR_irap);  // V = V2_ab * V1 * V2_a'b'
                                            this->final_map_to_irreducible_sector_[iap][R] = { isym, Rstar_irap.begin()->second };
                                            found = true;
                                            break;
                                        }
                                if (found)break;
                            }   // Rstar_irap

                            // 2. try Rstars_irap_append, usually small so we skip the length check
                            for (auto& Rstars_irap_append : this->R_stars_irap_append_[irap])
                                for (auto& symR_irap_R : Rstars_irap_append)
                                    if (!found && symR_irap_R.second == irR_rot_to_irap) // found
                                    {
                                        const int& isymR_irap = symR_irap_R.first;  //V2_a'b'
                                        const int isym = this->group_multiply(symm, group_multiply(symm, isymR_ap, isym_to_irap), isymR_irap);
                                        if (Rstars_irap_append.begin()->first == 0) // irreducible R is from current appendix star
                                            this->final_map_to_irreducible_sector_[iap][R] = { isym, Rstars_irap_append.begin()->second };
                                        else    //find the irreducible R of the Rstar it belongs to. no need to traverse Rstar again, just do a rotation
                                            this->final_map_to_irreducible_sector_[iap][R] = { isym,
                                            rotate_R_by_formula(symm, isymR_irap, irap.first, irap.second,irR_rot_to_irap) };
                                        found = true;
                                        break;
                                    }

                            if (!found) throw std::runtime_error("Symmetry_rotation::get_final_map_to_irreducible_sector: \
                                cannot find irreducible sector for atom pair(" + std::to_string(iat1) + ", " + std::to_string(iat2) + ") with R = (" + std::to_string(R[0]) + ", " + std::to_string(R[1]) + ", " + std::to_string(R[2]) + ")");
                        }// Rstar[iap]
                }  //if irreducible atom pair
            } // atom pair star
    }

    void Symmetry_rotation::output_final_map_to_irreducible_sector(const int nat)
    {
        std::cout << "Final map to irreducible sector: " << std::endl;
        for (int iat1 = 0;iat1 < nat;++iat1)
            for (int iat2 = 0;iat2 < nat;++iat2)
            {
                std::cout << "in atompair (" << iat1 << ", " << iat2 << "):" << std::endl;
                for (auto& R_isym_irR : this->final_map_to_irreducible_sector_[iat1 * nat + iat2])
                    std::cout << "R=(" << R_isym_irR.first[0] << ", " << R_isym_irR.first[1] << ", " << R_isym_irR.first[2] << "),  isym=" << R_isym_irR.second.first
                    << ",  R in irreducible atom pair:" << R_isym_irR.second.second[0] << ", " << R_isym_irR.second.second[1] << ", " << R_isym_irR.second.second[2] << ")" << std::endl;
            }
    }
}