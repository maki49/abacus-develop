#ifndef ABACUS_LR_XC_COMPONENTS_H
#define ABACUS_LR_XC_COMPONENTS_H
#ifdef __LIBXC
#include <vector>
struct xc_func_type;
namespace LR
{
// Borrow initialized components: the parent retains ownership and screening parameters.
// Keep its cutoff policy so isolating the sigma safeguard does not change density masks.
struct XCComponent
{
    xc_func_type* functional;
    xc_func_type* cutoff_functional;
    double weight;
    bool stabilize_exchange;
};
std::vector<XCComponent> kernel_components(std::vector<xc_func_type>& functionals);
}
#endif
#endif
