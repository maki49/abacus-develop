#ifndef ABACUS_LR_GRADIENT_AMPLITUDES_H
#define ABACUS_LR_GRADIENT_AMPLITUDES_H
#include "utils/lr_util.h"
#include "source_base/parallel_reduce.h"
#include <ostream>
namespace LR
{
template <typename T>
void follow_root(const T* Xall, const int n, const int nstates, const int initial_state,
    int& target_state, std::vector<T>& previous, std::ostream& ofs)
{

    if (target_state < 0) { target_state = initial_state; }

    if (!previous.empty())
    {
        std::vector<T> ov(nstates, T(0));
        for (int j = 0; j < nstates; ++j)
        {
            const T* const Xj = Xall + j * n;
            T acc = T(0);
            for (int i = 0; i < n; ++i) { acc += LR_Util::get_conj(previous[i]) * Xj[i]; }
            ov[j] = acc;
        }
        // X is distributed over the same 2D grid as the particle-hole pairs, so the inner
        // product is only complete after summing over that grid. Reduce the accumulators
        // themselves (real and imaginary parts alike) and take the modulus afterwards --
        // reducing |partial| would be wrong. `reduce_all` is the guarded wrapper and is a no-op
        // in a serial build, so no `#ifdef __MPI` is needed around it.
        Parallel_Reduce::reduce_all(ov.data(), nstates);
        int best = 0;
        double best_ov = -1.0;
        for (int j = 0; j < nstates; ++j)
        {
            const double a = std::abs(ov[j]);
            if (a > best_ov) { best_ov = a; best = j; }
        }

        if (best != target_state)
        {
            ofs << " EXCITED-STATE RELAX: followed root moved from index "
                << target_state << " to " << best << " (overlap " << best_ov
                << "); the states crossed and the index no longer names the same state."
                << std::endl;
        }
        // A low best overlap means no current root resembles the one being followed -- the step
        // was too large, or the state left the solved window. Say so: the relaxation continues
        // but the surface it follows is no longer guaranteed continuous.
        if (best_ov < 0.5)
        {
            ofs << " WARNING: largest amplitude overlap with the previous step is"
                " only " << best_ov << ". The followed state may have left the window spanned by"
                " lr_nstates; consider raising lr_nstates or reducing the ionic step." << std::endl;
        }
        target_state = best;
    }

    previous.assign(Xall + target_state * n, Xall + (target_state + 1) * n);
}

template <typename T>
ct::Tensor pad_amplitudes(const std::vector<ct::Tensor>& X, const bool openshell,
    const std::vector<Parallel_2D>& px, const std::vector<Parallel_2D>& px_z,
    const std::vector<int>& nocc, const std::vector<int>& nvirt,
    const int nk, const int nloc, const int nloc_z,
    const int ispin, const int istate_begin, const int nst)
{
    ct::Tensor Xz = LR_Util::newTensor<T>({ nst, nloc_z });
    Xz.zero();
    // Closed shell: one channel, and `ispin` selects the spin COMBINATION (singlet/triplet)
    // whose X is being widened -- `nocc`/`nvirt`/`paraX_` are the same for both.
    // Open shell: one eigenvector holding both channels back to back, so both sub-blocks are
    // widened and the second one is re-based, because widening the first moves where it starts.
    const std::vector<int> chan = openshell ? std::vector<int>{ 0, 1 }
                                                  : std::vector<int>{ ispin };
    const int ix = openshell ? 0 : ispin;   // which entry of `X`
    int soff_ch = 0;
    int zoff_ch = 0;                 // channel offsets inside one state's block
    for (const int is : chan)
    {
        const Parallel_2D& pxs = px[is];
        const Parallel_2D& pxz = px_z[is];
        for (int ist = 0;ist < nst;++ist)
        {
            const T* const src = X[ix].template data<T>()
                + (istate_begin + ist) * nloc + soff_ch;
            T* const dst = Xz.data<T>() + ist * nloc_z + zoff_ch;
            for (int ik = 0;ik < nk;++ik)
            {
                const int soff = ik * pxs.get_local_size();
                const int zoff = ik * pxz.get_local_size();
                // Both windows are block-cyclic with nb2d = 1 on the same process grid and share
                // the occupied dimension, so a global (virt, occ) element lives on the same
                // process in both -- only its local index differs. The widening is therefore a
                // purely local copy.
                for (int o = 0;o < nocc[is];++o)
                {
                    for (int v = 0;v < nvirt[is];++v)
                    {
                        if (!pxs.in_this_processor(v, o)) { continue; }
                        dst[zoff + pxz.global2local_col(o) * pxz.get_row_size() + pxz.global2local_row(v)]
                            = src[soff + pxs.global2local_col(o) * pxs.get_row_size() + pxs.global2local_row(v)];
                    }
                }
            }
        }
        soff_ch += nk * pxs.get_local_size();
        zoff_ch += nk * pxz.get_local_size();
    }
    return Xz;
}

}
#endif
