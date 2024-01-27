#include "symmetry_rotation.h"
#include <set>
#include "RI_Util.h"
#include "module_base/timer.h"
namespace ModuleSymmetry
{
    void Symmetry_rotation::find_irreducible_sector(const Symmetry& symm, const Atom* atoms, const Statistics& st, const std::vector<TC>& Rs)
    {
        ModuleBase::TITLE("Symmetry_rotation", "find_irreducible_sector");
        ModuleBase::timer::tick("Symmetry_rotation", "find_irreducible_sector");
        this->eps_ = symm.epsilon;
        if (this->return_lattice_.empty()) this->get_return_lattice_all(symm, atoms, st);
        // 1. find irreducible atom pairs
        if (this->atompair_stars_.empty())this->find_irreducible_atom_pairs(symm);

        // 2. find irreducible R
        this->find_irreducible_R(symm, atoms, st, Rs);
        this->output_irreducible_R(atoms, st, Rs);

        // 3. find final map to irreducible sector
        this->get_final_map_to_irreducible_sector(symm, atoms, st);
        this->output_final_map_to_irreducible_sector(st.nat);

        // 4. find sector star from final map
        this->find_sector_star_from_final_map(symm, atoms, st);
        this->output_sector_star();

        // 5. find transpose list from sector star
        this->find_transpose_list_from_sector_star(symm, atoms, st);
        this->output_transpose_list();
        ModuleBase::timer::tick("Symmetry_rotation", "find_irreducible_sector");
    }

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

    TC Symmetry_rotation::rotate_R_by_formula(const Symmetry& symm,
        const int isym, const int iat1, const int iat2, const TC& R, const char gauge) const
    {
        const TCdouble R_double(static_cast<double>(R[0]), static_cast<double>(R[1]), static_cast<double>(R[2]));
        const TCdouble Rrot_double = (gauge == 'L')
            ? R_double * symm.gmatrix[isym] + this->return_lattice_[iat1][isym] - this->return_lattice_[iat2][isym]
            : R_double * symm.gmatrix[isym] + this->return_lattice_[iat2][isym] - this->return_lattice_[iat1][isym];
        return { round2int(Rrot_double.x), round2int(Rrot_double.y), round2int(Rrot_double.z) };
    }
    TapR Symmetry_rotation::rotate_R_by_formula(const Symmetry& symm,
        const int isym, const TapR& apR, const char gauge) const
    {
        return { apR.first,  this->rotate_R_by_formula(symm, isym, apR.first.first, apR.first.second, apR.second, gauge) };
    }

    TCdouble Symmetry_rotation::get_aRb_direct(const Atom* atoms, const Statistics& st,
        const int iat1, const int iat2, const TCdouble& R, const char gauge)const
    {
        return TCdouble(atoms[st.iat2it[iat1]].taud[st.iat2ia[iat1]] - atoms[st.iat2it[iat2]].taud[st.iat2ia[iat2]]) + (gauge == 'L' ? R : TCdouble(-R));
    }

    TCdouble Symmetry_rotation::get_aRb_direct(const Atom* atoms, const Statistics& st,
        const int iat1, const int iat2, const TC& R, const char gauge) const
    {
        const TCdouble R_double(static_cast<double>(R[0]), static_cast<double>(R[1]), static_cast<double>(R[2]));
        return get_aRb_direct(atoms, st, iat1, iat2, R_double);
    }

    void Symmetry_rotation::find_irreducible_atom_pairs(const Symmetry& symm)
    {
        ModuleBase::TITLE("Symmetry_rotation", "find_irreducible_atom_pairs");
        this->eps_ = symm.epsilon;
        for (int iat1 = 0;iat1 < symm.nat;++iat1)
            for (int iat2 = 0;iat2 < symm.nat;++iat2)
            {
                Tap pair = { iat1, iat2 };
                bool exist = false;
                for (int isym = 0;isym < symm.nrotk;++isym)
                {
                    Tap rotpair = { symm.get_rotated_atom(isym,iat1), symm.get_rotated_atom(isym,iat2) };
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

    void Symmetry_rotation::find_irreducible_atom_pairs_set(const Symmetry& symm)
    {
        ModuleBase::TITLE("Symmetry_rotation", "find_irreducible_atom_pairs_set");
        this->eps_ = symm.epsilon;
        std::vector<int> invmap(symm.nrotk, -1);
        symm.gmatrix_invmap(symm.gmatrix, symm.nrotk, invmap.data());
        // contruct initial ap-set
        std::set<Tap, ap_less_func> ap_set;
        for (int iat1 = 0; iat1 < symm.nat; ++iat1)
            for (int iat2 = 0; iat2 < symm.nat; ++iat2)
                ap_set.insert({ iat1, iat2 });
        while (!ap_set.empty())
        {
            Tap ap = *ap_set.begin();
            std::map<int, Tap> ap_star;
            for (int isym = 0; isym < symm.nrotk; ++isym)
            {
                Tap rotpair = { symm.get_rotated_atom(isym,ap.first), symm.get_rotated_atom(isym,ap.second) };
                if (ap_set.find(rotpair) != ap_set.end())
                {
                    ap_star.insert({ invmap[isym], rotpair });
                    ap_set.erase(rotpair);
                }
            }
            this->atompair_stars_.push_back(ap_star);
        }
    }

    inline void output_atompair_stars(const std::vector<std::map<int, Tap>>& ap_stars)
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

    std::vector<TC> Symmetry_rotation::get_Rs_from_BvK(const K_Vectors& kv) const
    {
        const TC& period = RI_Util::get_Born_vonKarmen_period(kv);
        return RI_Util::get_Born_von_Karmen_cells(period);
    }
    std::vector<TC> Symmetry_rotation::get_Rs_from_adjacent_list(const UnitCell& ucell, Grid_Driver& gd, const Parallel_Orbitals& pv) const
    {
        // find the union set of Rs for all the atom pairs
        std::set<TC> Rs_set;
        for (int iat1 = 0;iat1 < ucell.nat;++iat1)
        {
            auto tau1 = ucell.get_tau(iat1);
            int it1 = ucell.iat2it[iat1], ia1 = ucell.iat2ia[iat1];
            AdjacentAtomInfo adjs;
            gd.Find_atom(ucell, tau1, it1, ia1, &adjs);
            for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
            {
                const int it2 = adjs.ntype[ad];
                const int ia2 = adjs.natom[ad];
                int iat2 = ucell.itia2iat(it2, ia2);
                if (pv.get_row_size(iat1) && pv.get_col_size(iat2))
                {
                    const ModuleBase::Vector3<int>& R_index = adjs.box[ad];
                    if (ucell.cal_dtau(iat1, iat2, R_index).norm() * ucell.lat0
                        < ucell.atoms[it1].Rcut + ucell.atoms[it2].Rcut)
                        Rs_set.insert({ R_index.x, R_index.y, R_index.z });
                }
            }
        }
        // set to vector
        std::vector<TC> Rs(Rs_set.size());
        for (auto& R : Rs_set) Rs.push_back(R);
        return Rs;
    }

    void Symmetry_rotation::find_irreducible_R(const Symmetry& symm, const Atom* atoms, const Statistics& st, const std::vector<TC>& Rs)
    {
        ModuleBase::TITLE("Symmetry_rotation", "find_irreducible_R");
        this->R_stars_.clear();
        this->R_stars_.resize(st.nat * st.nat);
        this->irreducible_sector_.clear();

        auto no_rotation = [](const int isym, const int iat1, const int iat2, const TC& R)-> int
            {return isym; };
        auto rotate_to_current_irR = [&st, atoms, &symm, this](const int isym, const int iat1, const int iat2, const TC& R)-> int
            {
                const TCdouble aRb = this->get_aRb_direct(atoms, st, iat1, iat2, R);
                for (auto& Rstar : this->R_stars_[iat1 * st.nat + iat2])
                {
                    const TC& R_dest = Rstar.at(0);
                    const TCdouble aRb_dest = this->get_aRb_direct(atoms, st, iat1, iat2, R_dest);
                    // If current R int Rstar_append can be rotated into Rstar of the same irreduceble atom pair, 
                    // update isym: V=V1*V2. Finally only the Rs with V=0 will be added to the irreducible sector.
                    if (symm.equal((aRb_dest * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det(),
                        (aRb * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det()))
                        for (int isym_to_iR = 0;isym_to_iR < symm.nrotk;++isym_to_iR)
                        {
                            TCdouble  aRb_rot = aRb * symm.gmatrix[isym_to_iR];
                            if (symm.equal(aRb_rot.x, aRb_dest.x) && symm.equal(aRb_rot.y, aRb_dest.y) && symm.equal(aRb_rot.z, aRb_dest.z))
                                return this->group_multiply(symm, isym, isym_to_iR);
                        }
                }

                // test
                if ((aRb * GlobalC::ucell.latvec).norm() > 10) std::cout << "isym = " << isym << ", aRb length=" << (aRb * GlobalC::ucell.latvec).norm() << std::endl;
                return isym;
            };
        auto add_R_to_Rstar = [&, this](std::vector<std::map<int, TC>>& Rstars, const int iat1, const int iat2, const TC& R,
            std::function<int(const int, const int, const int, const TC&) > rotate_first)
            {
                const TCdouble aRb_d = this->get_aRb_direct(atoms, st, iat1, iat2, R);
                bool exist = false;
                for (int iiR = 0;iiR < Rstars.size();++iiR)    // for each R_star in current atom pair
                { // compare with the aRb of irreducible R (current R_star[isym==0]) of current atom pair
                    // TC& irR = Rstars[iiR].at(0);
                    const int isym0 = Rstars[iiR].begin()->first;
                    TC& irR = Rstars[iiR].begin()->second;
                    TCdouble aRb_d_irR = this->get_aRb_direct(atoms, st, iat1, iat2, irR);
                    if (symm.equal((aRb_d_irR * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det(),
                        (aRb_d * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det()))   // the  length of aRb in the same Rstar should be the same
                    {
                        for (int isym = 0; isym < symm.nrotk; ++isym)
                        {
                            // if (symm.gmatrix[isym].Det() < 0) continue; // only consider pure rotation
                            TCdouble rot_aRb_d = aRb_d * symm.gmatrix[isym];
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
                for (auto& R : Rs) add_R_to_Rstar(this->R_stars_[iat1 * st.nat + iat2], iat1, iat2, R, no_rotation);

        // 2. find irreducible sector
        // 2.1. contruct appendix Rstar: irreducible R in other atom pairs that cannot rotate into R_stars_[irreducebule_ap]
        auto exceed_range = [&](const TC& irR)->bool
            {
                for (auto& cell : Rs) if (irR == cell) return false;
                return true;
            };
        for (auto apstar : this->atompair_stars_)
        {
            for (std::map<int, Tap>::iterator it = ++apstar.begin();it != apstar.end();++it)
            {
                const  int isym_to_irap = it->first;
                const Tap& ap = it->second;
                const Tap& irap = apstar.begin()->second;
                for (auto& Rstar : R_stars_[ap.first * st.nat + ap.second])
                {
                    const TC& irR = Rstar.at(0);
                    const TC irR_rot_to_irap = this->rotate_R_by_formula(symm, isym_to_irap, ap.first, ap.second, irR);// R'=VR+O_1-O_2
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
            Tap irap = apstar.at(0);
            for (auto& Rstar : R_stars_[irap.first * st.nat + irap.second])
                this->irreducible_sector_[irap].insert(Rstar.at(0));
            if (this->R_stars_irap_append_.find(irap) != this->R_stars_irap_append_.end())
                for (auto& Rstar : this->R_stars_irap_append_.at(irap))
                    if (Rstar.begin()->first == 0)  // only the Rs forming new stars
                        this->irreducible_sector_[irap].insert(Rstar.begin()->second);
        }
    }

    void Symmetry_rotation::output_irreducible_R(const Atom* atoms, const Statistics& st, const std::vector<TC>& Rs)
    {
        std::cout << "Number of irreducible atom pairs: " << this->atompair_stars_.size() << std::endl;
        std::cout << "Irreducible atom pairs: " << std::endl;
        for (int iap = 0; iap < this->atompair_stars_.size(); ++iap)
            std::cout << " (" << this->atompair_stars_[iap].at(0).first << ", "
            << this->atompair_stars_[iap].at(0).second << "), " << std::endl;
        std::cout << std::endl;

        auto print_Rstars = [&st, atoms, this](const std::vector<std::map<int, TC>>& Rstars, const int iat1, const int iat2)
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
            Tap irap = ap_Rsapd.first;
            std::cout << "In irreducible atom pair: (" << irap.first << ", " << irap.second << "): " << std::endl;
            print_Rstars(ap_Rsapd.second, irap.first, irap.second);
        }

        std::cout << "Irreducible Sector: " << std::endl;
        for (auto& irap_Rs : this->irreducible_sector_)
        {
            Tap irap = irap_Rs.first;
            std::cout << "In irreducible atom pair: (" << irap.first << ", " << irap.second << "): " << std::endl;
            for (auto& R : irap_Rs.second)
                std::cout << "R=(" << R[0] << ", " << R[1] << ", " << R[2] << ") " << std::endl;
        }
    }

    void Symmetry_rotation::get_final_map_to_irreducible_sector(const Symmetry& symm, const Atom* atoms, const Statistics& st)
    {
        ModuleBase::TITLE("Symmetry_rotation", "get_final_map_to_irreducible_sector");
        this->final_map_to_irreducible_sector_.clear();
        for (auto& apstar : this->atompair_stars_)
            for (auto& isym_ap : apstar)
            {
                const Tap& ap = isym_ap.second;
                const int iap = ap.first * st.nat + ap.second;
                // for irreducible atom pairs, the map is by its symmetry operation to the irreduceble R in its star
                if (isym_ap.first == 0)
                    for (auto& Rstar : this->R_stars_[iap])
                        for (auto& isym_R : Rstar)
                            this->final_map_to_irreducible_sector_[{ap, isym_R.second}] = { isym_R.first,{ap, Rstar.at(0)} };
                else
                {    // for other atom pairs, the map is:  [ab,R]->V2_ab->[ab,irR_ap]->V1->[a'b',R'_irap]->V2_a'b'->[a'b',irR'_irap]
                    Tap irap = apstar.at(0);
                    const int& isym_to_irap = isym_ap.first;  // V1
                    for (auto& Rstar : this->R_stars_[iap])
                        for (auto& symR_ap_R : Rstar)
                        {
                            const int& isymR_ap = symR_ap_R.first;  //V2_ab
                            const TC& R = symR_ap_R.second;
                            const TC& irR = Rstar.begin()->second;
                            const TC irR_rot_to_irap = this->rotate_R_by_formula(symm, isym_to_irap, ap.first, ap.second, irR);// R'=VR+O_1-O_2
                            const TCdouble aRb_irap = this->get_aRb_direct(atoms, st, irap.first, irap.second, irR_rot_to_irap);// s_a' + R' - s_b

                            // 1. try Rstars_[irap]
                            bool found = false;
                            for (auto& Rstar_irap : this->R_stars_[irap.first * st.nat + irap.second])
                            {
                                const TC& irR_irap = Rstar_irap.begin()->second;
                                const TCdouble aRb_irR_irap = this->get_aRb_direct(atoms, st, irap.first, irap.second, irR_irap); // s_a' + irR' - s_b'
                                if (symm.equal((aRb_irR_irap * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det(),
                                    (aRb_irap * GlobalC::ucell.latvec).norm() / GlobalC::ucell.latvec.Det()))   // the  length of aRb in the same Rstar should be the same
                                    for (auto& symR_irap_R : Rstar_irap)
                                        if (symR_irap_R.second == irR_rot_to_irap) // found R' in irreducible atom pair's Rstar, the corresponding R in ireducible sector is the ireducible R of current Rstar.
                                        {
                                            const int& isymR_irap = symR_irap_R.first;  //V2_a'b'
                                            const int isym = this->group_multiply(symm, group_multiply(symm, isymR_ap, isym_to_irap), isymR_irap);  // V = V2_ab * V1 * V2_a'b'
                                            this->final_map_to_irreducible_sector_[{ap, R}] = { isym, {irap, Rstar_irap.begin()->second} };
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
                                            this->final_map_to_irreducible_sector_[{ap, R}] = { isym, {irap, Rstars_irap_append.begin()->second } };
                                        else    //find the irreducible R of the Rstar it belongs to. no need to traverse Rstar again, just do a rotation
                                            this->final_map_to_irreducible_sector_[{ap, R}] = { isym,
                                            {irap,this->rotate_R_by_formula(symm, isymR_irap, irap.first, irap.second,irR_rot_to_irap) } };
                                        found = true;
                                        break;
                                    }

                            if (!found) throw std::runtime_error("Symmetry_rotation::get_final_map_to_irreducible_sector: \
                                cannot find irreducible sector for atom pair(" + std::to_string(ap.first) + ", " + std::to_string(ap.second) + ") with R = (" + std::to_string(R[0]) + ", " + std::to_string(R[1]) + ", " + std::to_string(R[2]) + ")");
                        }// Rstar[iap]
                }  //if irreducible atom pair
            } // atom pair star
        // clear Rstars 
        this->R_stars_.clear();
        this->R_stars_irap_append_.clear();
    }
    void Symmetry_rotation::output_final_map_to_irreducible_sector(const int nat)
    {
        std::cout << "Final map to irreducible sector: " << std::endl;
        for (auto& apR_isym_irapR : this->final_map_to_irreducible_sector_)
            {
            const Tap& ap = apR_isym_irapR.first.first;
            const TC& R = apR_isym_irapR.first.second;
            const Tap& irap = apR_isym_irapR.second.second.first;
            const TC& irR = apR_isym_irapR.second.second.second;
            std::cout << "atompair (" << ap.first << ", " << ap.second << "), R=(" << R[0] << ", " << R[1] << ", " << R[2] << ") -> "
                << "isym=" << apR_isym_irapR.second.first << " -> irreducible atompair (" << irap.first << ", " << irap.second << "), irreducible R=("
                << irR[0] << ", " << irR[1] << ", " << irR[2] << ")" << std::endl;
        }
    }

    void Symmetry_rotation::find_sector_star_from_final_map(const Symmetry& symm, const Atom* atoms, const Statistics& st)
    {
        this->sector_stars_.clear();
        for (auto& apR_isym_irapR : this->final_map_to_irreducible_sector_)
        {
            const TapR& apR = apR_isym_irapR.first;
            const int& isym = apR_isym_irapR.second.first;
            const TapR& irapR = apR_isym_irapR.second.second;
            bool exist = false;
            for (auto& irs_star : this->sector_stars_)  //search previous stars
            {
                const TapR& irapR_star = this->final_map_to_irreducible_sector_.at(irs_star.begin()->second).second;
                if (irapR_star.first == irapR.first && irapR_star.second == irapR.second)   // map to the same irreducible {abR} <=> in the same star
                {
                    exist = true;
                    irs_star.insert({ isym, apR });
                    break;
                }
            }
            if (!exist) this->sector_stars_.push_back({ { isym, apR} });
        }
        // check size
        int irreducible_sector_size = 0;
        for (auto& irap_Rs : this->irreducible_sector_) irreducible_sector_size += irap_Rs.second.size();
        assert(irreducible_sector_size >= this->sector_stars_.size());
        int total_apR_in_star = 0;
        for (auto& secstar : this->sector_stars_) total_apR_in_star += secstar.size();
        assert(total_apR_in_star == this->final_map_to_irreducible_sector_.size());
    }

    void Symmetry_rotation::output_sector_star()
    {
        std::cout << "Found " << this->sector_stars_.size() << " irreducible sector stars:" << std::endl;
        // for (auto& irs_star : this->sector_stars_)
        for (int istar = 0;istar < this->sector_stars_.size();++istar)
        {
            std::cout << "in star " << istar << " with size " << this->sector_stars_[istar].size() << ":\n";
            for (auto& isym_ap_R : this->sector_stars_[istar])
                std::cout << "isym=" << isym_ap_R.first << ", atompair=(" << isym_ap_R.second.first.first << ", " << isym_ap_R.second.first.second << "), R=("
                << isym_ap_R.second.second[0] << ", " << isym_ap_R.second.second[1] << ", " << isym_ap_R.second.second[2] << ")" << std::endl;
        }

    }

    inline double cartesian_dot(const TC& a, const TC& b)
    {
        const TCdouble& cart_a = TCdouble(a[0], a[1], a[2]) * GlobalC::ucell.latvec;
        const TCdouble& cart_b = TCdouble(b[0], b[1], b[2]) * GlobalC::ucell.latvec;
        return cart_a * cart_b;
    }

    void Symmetry_rotation::find_transpose_list_from_sector_star(const Symmetry& symm, const Atom* atoms, const Statistics& st)
    {
        this->transpose_list_.clear();
        for (auto& irs_star : this->sector_stars_)
        {
            const TapR& irap_R = irs_star.count(0) ? irs_star.at(0) : this->rotate_R_by_formula(symm, irs_star.begin()->first, irs_star.begin()->second);
            if (irap_R.first.first != irap_R.first.second) continue; // only consider the aa-pairs
            std::set<TapR> transpose_set;
            std::set<TapR> rotation_set;
            for (auto& isym_ap_R : irs_star)
            {
                const TapR& ap_R = isym_ap_R.second;
                double dot = cartesian_dot(ap_R.second, irap_R.second);
                (dot > -eps_) ? rotation_set.insert(ap_R) : transpose_set.insert(ap_R);
            }
            for (auto& apR : transpose_set)
            {
                const TC& R = apR.second;
                for (auto& apR_ref : rotation_set)
                {
                    const TC& R_ref = apR_ref.second;
                    const bool same_atom = (apR.first.first == apR_ref.first.first);
                    if (same_atom && R[0] == -R_ref[0] && R[1] == -R_ref[1] && R[2] == -R_ref[2])
                        this->transpose_list_.insert({ apR, apR_ref });
                }
            }
        }
    }

    void Symmetry_rotation::output_transpose_list()
    {
        std::cout << "transpose R list of aa-pairs: " << std::endl;
        std::cout << "calculated by transpose   <-  calculated by rotation" << std::endl;
        for (auto& item : this->transpose_list_)
        {
            const TC& R = item.first.second;
            const TC& R_ref = item.second.second;
            std::cout << "in atom pair(" << item.first.first.first << ", " << item.first.first.second << "): (" << R[0] << ", " << R[1] << ", " << R[2] << ")   <-   ("
                << R_ref[0] << ", " << R_ref[1] << ", " << R_ref[2] << ")" << std::endl;
        }
    }
}