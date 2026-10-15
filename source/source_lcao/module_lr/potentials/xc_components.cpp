#include "xc_components.h"
#ifdef __LIBXC
#include <xc.h>
#include <stdexcept>

std::vector<LR::XCComponent> LR::kernel_components(std::vector<xc_func_type>& functionals)
{
    std::vector<XCComponent> components;
    for (auto& parent : functionals)
    {
        if (parent.info->number != XC_HYB_GGA_XC_HSE06)
        {
            components.push_back({&parent, &parent, 1.0, false});
            continue;
        }
        if (parent.mix_coef == nullptr || parent.n_func_aux == 0)
        {
            throw std::runtime_error("HSE06 kernel requires initialized LibXC mixed components");
        }
        for (int index = 0; index < parent.n_func_aux; ++index)
        {
            xc_func_type* component = parent.func_aux[index];
            const bool wpbeh = component->info->number == XC_GGA_X_WPBEH;
            components.push_back({component, &parent, parent.mix_coef[index], wpbeh});
        }
    }
    return components;
}
#endif
