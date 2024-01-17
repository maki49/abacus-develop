#include "symmetry_rotation.h"
#include <set>
#include "RI_Util.h"
namespace ModuleSymmetry
{
    void Symmetry_rotation::find_irreducible_atom_pairs(const Symmetry& symm)
    {
        ModuleBase::TITLE("Symmetry_rotation", "find_irreducible_atom_pairs");
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
        this->R_stars_.resize(this->atompair_stars_.size());
        this->irreducible_aRb_d_.resize(this->atompair_stars_.size());
        for (int iap = 0;iap < this->atompair_stars_.size();++iap)
        {
            std::pair<int, int> ap = this->atompair_stars_[iap].at(0);   // irreducible atom pair
            std::array<int, 3> period = RI_Util::get_Born_vonKarmen_period(kv);
            std::vector<std::array<int, 3>> BvK_cells = RI_Util::get_Born_von_Karmen_cells(period);

            for (auto& R : BvK_cells)
            {
                ModuleBase::Vector3<double> aRb_d = atoms[st.iat2it[ap.first]].taud[st.iat2ia[ap.first]]
                    + ModuleBase::Vector3<double>(static_cast<double>(R[0]), static_cast<double>(R[1]), static_cast<double>(R[2]))
                    - atoms[st.iat2it[ap.second]].taud[st.iat2ia[ap.second]];    // the relative position vector (direct) from b0 to aR
                bool exist = false;

                for (int isym = 0; isym < symm.nrotk; ++isym)
                {
                    ModuleBase::Vector3<double> rot_aRb_d = aRb_d * symm.gmatrix[isym];

                    // test=======
                    if (ap.first == 0 && ap.second == 1 && R[0] == 0 && R[1] == -1 && R[2] == 0)
                        std::cout << "isym=" << isym << ", rot_aRb_d=(" << rot_aRb_d.x << ", " << rot_aRb_d.y << ", " << rot_aRb_d.z << ") " << std::endl;
                    // test=======
                    for (int iiR = 0;iiR < this->R_stars_[iap].size();++iiR)    // for each R_star in current atom pair
                    { // compare with the aRb of irreducible R (current R_star[isym==0]) of current atom pair
                        if (symm.equal(rot_aRb_d.x, this->irreducible_aRb_d_[iap][iiR].x) &&
                            symm.equal(rot_aRb_d.y, this->irreducible_aRb_d_[iap][iiR].y) &&
                            symm.equal(rot_aRb_d.z, this->irreducible_aRb_d_[iap][iiR].z))
                        {
                            // why failed? debug!!
                            assert(symm.equal((this->irreducible_aRb_d_[iap][iiR] * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det(),
                                (aRb_d * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det()));   // the  length of aRb in the same Rstar should be the same
                            exist = true;
                            this->R_stars_[iap][iiR].insert({ isym, R });
                            break;
                        }
                    }
                    if (exist) break;
                }//isym
                if (!exist)
                {
                    this->R_stars_[iap].push_back({ {0, R} });
                    this->irreducible_aRb_d_[iap].push_back(aRb_d);
                }
            }// R
        }// iap
    }

    void Symmetry_rotation::output_irreducible_R(const K_Vectors& kv)
    {
        // output BvK cells
        std::array<int, 3> period = RI_Util::get_Born_vonKarmen_period(kv);
        std::vector<std::array<int, 3>> BvK_cells = RI_Util::get_Born_von_Karmen_cells(period);
        std::cout << "Born von Karmen cells: " << std::endl;
        for (const auto& R : BvK_cells)
            std::cout << "(" << R[0] << ", " << R[1] << ", " << R[2] << ") " << std::endl;

        for (int iap = 0; iap < this->atompair_stars_.size(); ++iap)
        {
            std::cout << "in irreducible atom pair " << iap << ":  (" << this->atompair_stars_[iap].at(0).first << ", "
                << this->atompair_stars_[iap].at(0).second << ")" << std::endl;
            for (int iiR = 0; iiR < this->R_stars_[iap].size(); ++iiR)
            {
                std::cout << "R_star " << iiR << ", irreducible aRb =(" <<
                    this->irreducible_aRb_d_[iap][iiR].x << ", " << this->irreducible_aRb_d_[iap][iiR].y
                    << ", " << this->irreducible_aRb_d_[iap][iiR].z << ") , cartesian length = "
                    << (this->irreducible_aRb_d_[iap][iiR] * GlobalC::ucell.latvec).norm() << ":" << std::endl;
                for (const auto& R : this->R_stars_[iap][iiR])
                    std::cout << "isym=" << R.first << ", R=(" << R.second[0] << ", " << R.second[1] << ", " << R.second[2] << ") " << std::endl;
            }
            std::cout << std::endl;
        }
    }

}