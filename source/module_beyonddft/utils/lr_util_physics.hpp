#pragma once
#include <cstddef>
#include <vector>
#include <utility>
#include "module_base/constants.h"
#include "module_base/matrix.h"

namespace LR_Util
{
    template <typename TCell>
    const size_t cal_nelec(const TCell& ucell) {
        size_t nelec = 0;
        for (size_t it = 0; it < ucell.ntype; ++it)
            nelec += ucell.atoms[it].ncpp.zv * ucell.atoms[it].na;
        return nelec;
    }

    const size_t cal_nocc(size_t nelec) { return nelec / ModuleBase::DEGSPIN; }

    std::pair<ModuleBase::matrix, std::vector<std::pair<int, int>>>
        set_ix_map_diagonal(bool mode, int nc, int nv)
    {
        int npairs = nc * nv;
        ModuleBase::matrix iciv2ix(nc, nv, true);
        std::vector<std::pair<int, int>> ix2iciv(npairs);
        int ic = nc - 1, iv = 0;    //start：leftup
        if (mode == 0)  // leftdown->rightup
        {
            for (int ix = 0;ix < npairs - 1;++ix)
            {
                // 1. set value
                iciv2ix(ic, iv) = ix;
                ix2iciv[ix] = std::make_pair(ic, iv);
                // 2. move
                if (ic == nc - 1 || iv == nv - 1)    // rightup bound
                {
                    int ic_next = std::max(nc - iv - 1 - (nc - ic), 0);
                    iv -= (ic - ic_next) - 1;
                    ic = ic_next;
                }
                else { ++ic;++iv; }//move rightup
            }
        }
        else    //rightup->leftdown
        {
            for (int ix = 0;ix < npairs - 1;++ix)
            {
                // 1. set value
                iciv2ix(ic, iv) = ix;
                ix2iciv[ix] = std::make_pair(ic, iv);
                // 2. move
                if (ic == 0 || iv == 0)    // leftdown bound
                {
                    int iv_next = std::min(nc - ic + iv, nv - 1);
                    ic += (iv_next - iv) - 1;
                    iv = iv_next;
                }
                else { --iv;--ic; }//move leftdown
            }
        }
        //final set: rightdown
        assert(ic == 0);
        assert(iv == nv - 1);
        iciv2ix(ic, iv) = npairs - 1;
        ix2iciv[npairs - 1] = std::make_pair(ic, iv);
        return std::make_pair(std::move(iciv2ix), std::move(ix2iciv));
    }
}