#ifndef ABACUS_LR_SPECTRUM_WEIGHTS_H
#define ABACUS_LR_SPECTRUM_WEIGHTS_H

namespace LR_Util
{
/// KS weights include a factor two only for nspin=1. Optical singlet
/// amplitudes account for that degeneracy separately through sqrt(2).
inline double spatial_k_weight(const double ks_weight, const int nspin_global)
{
    return nspin_global == 1 ? 0.5 * ks_weight : ks_weight;
}
}

#endif
