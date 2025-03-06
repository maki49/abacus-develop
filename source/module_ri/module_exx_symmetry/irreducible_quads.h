#include "irreducible_sector.h"

namespace ModuleSymmetry
{
    // using Tquad_IJR = std::tuple<TapR, TC, TC>;    // {KLR, R_IK, R_JL}
    using Tquad_IJR = std::pair<TapR, TC>;    // {KLR, R_IK}
    class Irreducible_Quads
    {
    public:
        void find_irreducible_quads(const Irreducible_Sector& irs,
            const Symmetry& symm, const Atom* atoms, const Statistics& st,
            const std::vector<TC>& Rs, const TC& period, const Lattice& lat);
        void print_quads_stars()const;
        void print_irreducible_quads()const;
        const std::map<TapR, std::set<Tquad_IJR>>& get_irreducible_quads()const { return this->irreducible_quads_; }
        const std::map<TapR, std::vector<std::map<int, Tquad_IJR>>>& get_quads_stars()const { return this->quads_stars_; }
        const std::map<TapR, std::map<Tquad_IJR, int>>& get_irreducible_quads_weight()const { return this->irreducible_quads_weight_; }
        const std::set<int> get_invariant_ops(const TapR& irsec)const
        {
            try
            {
                return this->irsector_invariant_ops_.at(irsec);
            }
            catch (const std::out_of_range& oor)
            {
                const TC& R = irsec.second;
                std::cerr << "Cannot find the irreducible sector: (" << irsec.first.first << ", " << irsec.first.second << "), R=(" << R[0] << ", " << R[1] << ", " << R[2] << ")\n";
                return std::set<int>({ });
            }
        }
    private:
        void find_irsector_invariant_operations(const Irreducible_Sector& irs, const Symmetry& symm);
        std::map<TapR, std::set<int>> irsector_invariant_ops_;
        std::map<TapR, std::vector<std::map<int, Tquad_IJR>>> quads_stars_;   // irIJR, {isym, {KLR, R_IK, R_JL}}
        std::map<TapR, std::set<Tquad_IJR>> irreducible_quads_;
        std::map<TapR, std::map<Tquad_IJR, int>> irreducible_quads_weight_;
    };
}