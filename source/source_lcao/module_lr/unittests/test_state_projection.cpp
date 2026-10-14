#include "source_lcao/module_lr/state_projection.h"
#include "source_base/parallel_2d.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_basis/module_nao/two_center_integrator.h"
#include "source_cell/unitcell.h"
#include "source_psi/psi.h"
#include <gtest/gtest.h>
#include <complex>

namespace
{
template <typename T> T conjugate(T value) { return value; }
template <> std::complex<double> conjugate(std::complex<double> value) { return std::conj(value); }

template <typename T>
void check_spin_projection(T phase)
{
    const int bands = 4;
    const int block_size = 1;
    Parallel_Orbitals pc;
#ifdef __MPI
    pc.init(bands, bands, block_size, MPI_COMM_WORLD);
#else
    pc.set_serial(bands, bands);
#endif
    std::vector<Parallel_2D> px(2);
    const std::vector<int> occupied{1, 2};
    const std::vector<int> virtuals{3, 2};
    LR::OrbitalBasisSnapshot<T> old;
    LR::OrbitalBasisSnapshot<T> current;
    std::vector<double> ao(pc.get_local_size(), 0.0);
    for (int spin = 0; spin < 2; ++spin)
    {
#ifdef __MPI
        px[spin].set(virtuals[spin], occupied[spin], block_size, pc.blacs_ctxt);
#else
        px[spin].set_serial(virtuals[spin], occupied[spin]);
#endif
        old.coefficients.emplace_back(pc.get_local_size(), T(0));
        current.coefficients.emplace_back(pc.get_local_size(), T(0));
        for (int col = 0; col < pc.get_col_size(); ++col)
        {
            const int global_col = pc.local2global_col(col);
            for (int row = 0; row < pc.get_row_size(); ++row)
            {
                const int global_row = pc.local2global_row(row);
                if (global_row != global_col) { continue; }
                const int index = col * pc.get_row_size() + row;
                old.coefficients[spin][index] = T(1);
                current.coefficients[spin][index] = global_col % 2 == 0 ? phase : T(1);
                ao[index] = 0.7 + 0.05 * global_col;
            }
        }
    }
    UnitCell cell;
    TwoCenterIntegrator integrator;
    psi::Psi<T> orbitals;
    const std::vector<double> cutoffs;
    const LR::TransitionBasisView<T> view{cell, cutoffs, integrator, orbitals, pc, pc,
        px, occupied, virtuals, true, 0};
    const int local_dimension = px[0].get_local_size() + px[1].get_local_size();
    std::vector<std::vector<T>> vectors(3, std::vector<T>(local_dimension));
    auto expected = vectors;
    std::vector<std::vector<T>*> references;
    for (int root = 0; root < 3; ++root)
    {
        int offset = 0;
        for (int spin = 0; spin < 2; ++spin)
        {
            for (int i = 0; i < px[spin].get_col_size(); ++i)
            {
                const int occupied_band = px[spin].local2global_col(i);
                for (int a = 0; a < px[spin].get_row_size(); ++a)
                {
                    const int virtual_band = occupied[spin] + px[spin].local2global_row(a);
                    const int index = offset + i * px[spin].get_row_size() + a;
                    const T value = T(1 + root + spin + occupied_band + virtual_band);
                    const T occupied_phase = occupied_band % 2 == 0 ? phase : T(1);
                    const T virtual_phase = virtual_band % 2 == 0 ? phase : T(1);
                    vectors[root][index] = value;
                    expected[root][index] = value * occupied_phase * conjugate(virtual_phase)
                        * (0.7 + 0.05 * occupied_band) * (0.7 + 0.05 * virtual_band);
                }
            }
            offset += px[spin].get_local_size();
        }
        references.push_back(&vectors[root]);
    }
    LR::project_transition_vectors(view, old, current, ao, references);
    for (int root = 0; root < 3; ++root)
    {
        for (int i = 0; i < local_dimension; ++i)
        {
            const double error = std::abs(vectors[root][i] - expected[root][i]);
            EXPECT_NEAR(error, 0.0, 1e-12);
        }
    }
}
}

TEST(StateProjection, UnequalSpinWindowsAndMultipleRoots) { check_spin_projection(-1.0); }
TEST(StateProjection, ComplexOrbitalGauge)
{
    const std::complex<double> phase(0.6, 0.8);
    check_spin_projection(phase);
}

TEST(StateProjection, RejectsChangedCellWindowAndDistribution)
{
    LR::OrbitalBasisSnapshot<double> old;
    old.lat0 = 8.0;
    old.nocc = {1, 2};
    old.nvirt = {3, 2};
    old.layout = {1, 2, 2};
    auto current = old;
    EXPECT_TRUE(LR::compatible_orbital_bases(old, current));
    current.lat0 = 8.1;
    EXPECT_FALSE(LR::compatible_orbital_bases(old, current));
    current = old;
    current.nocc[0] = 2;
    EXPECT_FALSE(LR::compatible_orbital_bases(old, current));
    current = old;
    current.layout[0] = 2;
    EXPECT_FALSE(LR::compatible_orbital_bases(old, current));
}
