#ifndef ABACUS_SOURCE_LCAO_MODULE_LR_HXC_SPIN_ACCUMULATOR_H
#define ABACUS_SOURCE_LCAO_MODULE_LR_HXC_SPIN_ACCUMULATOR_H

#include "source_base/matrix.h"
#include <array>
#include <stdexcept>
#include <vector>

namespace hamilt
{
template<typename T, typename Device> class Operator;
}

namespace LR
{
template<typename T, typename Device> class OperatorLRHxc;

/// Grid potential for one output spin and one trial vector:
/// v_sigma(r) = sum_tau integral f_Hxc^{sigma,tau}(r,r') rho_tau(r') dr'.
/// Owns only values (real/imaginary parts); no operator or density references survive add().
class HxcSpinAccumulator
{
  public:
    /// Add rho_tau -> v_{sigma,tau}(r). The caller supplies the same output_projection
    /// for every contribution and for integrate_and_project(); it must be a compatible VO map.
    template<typename T, typename Device>
    void add(const OperatorLRHxc<T, Device>& contribution,
             const OperatorLRHxc<T, Device>& output_projection,
             typename OperatorLRHxc<T, Device>::TransitionDensityCache& density_cache)
    {
        using Hxc = OperatorLRHxc<T, Device>;
        if (contribution.dm_pq_ != Hxc::MO_TO_AO_TYPE::CC_vo
            || output_projection.dm_pq_ != Hxc::MO_TO_AO_TYPE::CC_vo
            || contribution.ispin_ks.size() != 2 || output_projection.ispin_ks.size() != 2)
        {
            throw std::logic_error("Hxc spin fusion requires an unrestricted VO projection");
        }
        if (contribution.ispin_ks[0] != output_projection.ispin_ks[0]
            || contribution.factor_ != output_projection.factor_
            || &contribution.psi_ks != &output_projection.psi_ks
            || &contribution.pX != &output_projection.pX)
        {
            throw std::logic_error("Hxc spin contributions have incompatible projections");
        }
        contribution.accumulate_potential(density_cache, potential_);
    }

    /// Integrate the sum once into V_AO, then add factor * C_v^dagger V_AO C_o to hpsi.
    /// The explicit output operator supplies its AO workspace, orbitals and distribution.
    template<typename T, typename Device>
    void integrate_and_project(const OperatorLRHxc<T, Device>& output_projection, T* hpsi) const
    {
        if (!potential_.empty())
        {
            output_projection.integrate_potential(potential_);
            output_projection.project(nullptr, hpsi); // CC_vo does not read psi_in.
        }
    }

  private:
    std::vector<ModuleBase::matrix> potential_;
};

/// Select Hxc_{00} and Hxc_{11} from the diagonal chains of ops[2 * out + in].
/// Their VO maps depend only on the output spin; the input spin was consumed in v(r).
template<typename T, typename Device>
std::array<const OperatorLRHxc<T, Device>*, 2>
hxc_spin_projections(const std::vector<hamilt::Operator<T, Device>*>& ops)
{
    if (ops.size() != 4)
    {
        throw std::logic_error("Hxc spin fusion requires four spin-block chains");
    }
    std::array<const OperatorLRHxc<T, Device>*, 2> projections{};
    for (int spin = 0; spin < 2; ++spin)
    {
        for (auto* node = ops[3 * spin]; node != nullptr; node = node->next_op)
        {
            const auto* hxc = dynamic_cast<const OperatorLRHxc<T, Device>*>(node);
            if (hxc != nullptr)
            {
                if (projections[spin] != nullptr)
                {
                    throw std::logic_error("Hxc spin fusion requires one Hxc per diagonal chain");
                }
                projections[spin] = hxc;
            }
        }
        if (projections[spin] == nullptr)
        {
            throw std::logic_error("Hxc spin fusion is missing a diagonal Hxc operator");
        }
    }
    return projections;
}

/// Defer Hxc integration/projection until both input spins have contributed.
/// Diagonal and exchange nodes still act while their input-spin density is available.
template<typename T, typename Device>
void act_with_spin_sum(hamilt::Operator<T, Device>* node,
                       const int nbasis,
                       const T* psi_in,
                       T* hpsi,
                       typename OperatorLRHxc<T, Device>::TransitionDensityCache& density_cache,
                       const OperatorLRHxc<T, Device>& output_projection,
                       HxcSpinAccumulator& spin_potential)
{
    const auto* hxc = dynamic_cast<const OperatorLRHxc<T, Device>*>(node);
    if (hxc != nullptr)
    {
        spin_potential.add(*hxc, output_projection, density_cache);
    }
    else
    {
        node->act(1, nbasis, 1, psi_in, hpsi);
    }
}
}

#endif // ABACUS_SOURCE_LCAO_MODULE_LR_HXC_SPIN_ACCUMULATOR_H
