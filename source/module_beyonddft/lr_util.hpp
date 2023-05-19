#pragma once
#include <cstddef>
#include "module_cell/unitcell.h"
#include "module_base/constants.h"
namespace LR_Util
{
    template <typename T>
    void delete_p2(T** p2, size_t size)
    {
        if (p2 != nullptr)
        {
            for (size_t i = 0; i < size; ++i)
            {
                if(p2[i] != nullptr) delete[] p2[i];
            }
            delete[] p2;
        }
    };

    template <typename T>
    void delete_p3(T*** p3, size_t size1, size_t size2)
    {
        if (p3 != nullptr)
        {
            for (size_t i = 0; i < size1; ++i)
            {
                delete_p2(p3[i], size2);
            }
            delete[] p3;
        }
    };
    
    template <typename TCell>
    const size_t cal_nelec(const TCell& ucell)
    {
        size_t nelec = 0;
        for (size_t it = 0; it < ucell.ntype; ++it)
            nelec += ucell.atoms[it].ncpp.zv * ucell.atoms[it].na;
        return nelec;
    }
    const size_t cal_nocc(size_t nelec)
    {
        return nelec / ModuleBase::DEGSPIN;
    }

    
}