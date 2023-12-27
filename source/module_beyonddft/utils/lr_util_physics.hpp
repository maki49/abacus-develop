#pragma once
#include <cstddef>
#include <vector>
#include <utility>
#include "module_base/constants.h"
#include "module_base/matrix.h"

namespace LR_Util
{
    template <typename TCell>
    const int cal_nelec(const TCell& ucell) {
        int nelec = 0;
        for (int it = 0; it < ucell.ntype; ++it)
            nelec += ucell.atoms[it].ncpp.zv * ucell.atoms[it].na;
        return nelec;
    }

    const int cal_nocc(int nelec) { return nelec / ModuleBase::DEGSPIN + nelec % static_cast<int>(ModuleBase::DEGSPIN); }

    std::pair<ModuleBase::matrix, std::vector<std::pair<int, int>>>
        set_ix_map_diagonal(bool mode, int nocc, int nvirt)
    {
        int npairs = nocc * nvirt;
        ModuleBase::matrix ioiv2ix(nocc, nvirt, true);
        std::vector<std::pair<int, int>> ix2ioiv(npairs);
        int io = nocc - 1, iv = 0;    //start：leftup
        if (mode == 0)  // leftdown->rightup
        {
            for (int ix = 0;ix < npairs - 1;++ix)
            {
                // 1. set value
                ioiv2ix(io, iv) = ix;
                ix2ioiv[ix] = std::make_pair(io, iv);
                // 2. move
                if (io == nocc - 1 || iv == nvirt - 1)    // rightup bound
                {
                    int io_next = std::max(nocc - iv - 1 - (nocc - io), 0);
                    iv -= (io - io_next) - 1;
                    io = io_next;
                }
                else { ++io;++iv; }//move rightup
            }
        }
        else    //rightup->leftdown
        {
            for (int ix = 0;ix < npairs - 1;++ix)
            {
                // 1. set value
                ioiv2ix(io, iv) = ix;
                ix2ioiv[ix] = std::make_pair(io, iv);
                // 2. move
                if (io == 0 || iv == 0)    // leftdown bound
                {
                    int iv_next = std::min(nocc - io + iv, nvirt - 1);
                    io += (iv_next - iv) - 1;
                    iv = iv_next;
                }
                else { --iv;--io; }//move leftdown
            }
        }
        //final set: rightdown
        assert(io == 0);
        assert(iv == nvirt - 1);
        ioiv2ix(io, iv) = npairs - 1;
        ix2ioiv[npairs - 1] = std::make_pair(io, iv);
        return std::make_pair(std::move(ioiv2ix), std::move(ix2ioiv));
    }
}