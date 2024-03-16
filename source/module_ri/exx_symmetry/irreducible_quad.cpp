#include "irreducible_quad.h"
namespace ModuleSymmetry
{
    inline bool sector_eq(const TapR& a, const TapR& b) { return a.first == b.first && a.second == b.second; }
    void Irreducible_Quad::find_irsector_invariant_operations(const Symmetry& symm)
    {
        for (auto& irap : this->irreducible_sector_)
            for (auto& irR : irap.second)
            {
                const TapR& irapR = { irap.first, irR };
                std::set<int> isym_set;
                for (int isym = 0;isym < symm.nrotk;++isym)
                {
                    if (sector_eq(this->rotate_apR_by_formula(symm, isym, irapR), irapR))
                }
            }
    }
}