#include "../minimal_gaussian_ri.h"
#include "../minimal_gaussian_basis.h"
#include "source_basis/module_ao/orb_read.h"
#include "source_cell/unitcell.h"
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#include <RI/global/MPI_Wrapper.h>
#include <gtest/gtest.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <set>

namespace
{
class MinimalGaussianRI : public ::testing::Test
{
  protected:
    std::vector<Atom> atoms;
    UnitCell cell;
    LCAO_Orbitals orbitals;
    std::vector<int> atomic_numbers;
    LR::MinimalGaussianRISettings settings;
    const LR::RICell period{{1, 1, 1}};

    void load(const bool include_oxygen)
    {
        const int rank = RI::MPI_Wrapper::mpi_get_rank(MPI_COMM_WORLD);
        const int ntype = include_oxygen ? 2 : 1;
        const std::string filename = "minimal_gaussian_ri_rank" + std::to_string(rank) + ".log";
        std::ofstream log(filename);
        const std::string directory = std::string(RI_TEST_ORBITAL_DIR) + "/";
        const std::vector<std::string> filenames{"H_gga_8au_60Ry_2s1p.orb", "O_gga_7au_60Ry_2s2p1d.orb"};
        const std::string* orbital_files = filenames.data();
        orbitals.init(log, ntype, directory, orbital_files, "", 2, 60.0, 0.01, 0.01, 30.0,
                      false, 0, false, false, rank);
        atoms.resize(ntype);
        cell.atoms = atoms.data();
        cell.ntype = ntype;
        cell.nat = ntype;
        cell.lat0 = 80.0; // Isolated atoms/molecule: no overlap with periodic images.
        cell.a1 = {1.0, 0.0, 0.0};
        cell.a2 = {0.0, 1.0, 0.0};
        cell.a3 = {0.0, 0.0, 1.0};
        cell.iat2it.resize(ntype);
        cell.iat2ia.resize(ntype, 0);
        atomic_numbers = include_oxygen ? std::vector<int>{1, 8} : std::vector<int>{1};
        for (int a = 0; a < ntype; ++a)
        {
            cell.iat2it[a] = a;
            atoms[a].na = 1;
            atoms[a].type = a;
            atoms[a].tau.resize(1);
            atoms[a].tau[0] = {0.025 * a, 0.0, 0.0};
        }
        settings.coulomb_cutoff = 20.0;
        settings.kmesh_multiplier = 1.0;
    }

    template<typename T> LR::RIAtomPairMap<T> probe_density(const LR::RIGeometry& geometry)
    {
        LR::RIAtomPairMap<T> density;
        const int rank = RI::MPI_Wrapper::mpi_get_rank(MPI_COMM_WORLD);
        if (rank == 0)
        {
            const std::size_t nao = geometry.orbitals_per_atom[0];
            const LR::RIAtomCell pair{0, {{0, 0, 0}}};
            density[0][pair] = RI::Tensor<T>({nao, nao});
            density[0][pair](0, 0) = 1.0;
        }
        return density;
    }

    template<typename T> T onsite(const LR::RIAtomPairMap<T>& local)
    {
        const std::set<int> atom{0};
        const auto all = RI::Communicate_Tensors_Map_Judge::comm_map2_first(MPI_COMM_WORLD, local, atom, atom);
        const LR::RIAtomCell pair{0, {{0, 0, 0}}};
        return all.at(0).at(pair)(0, 0);
    }
};

TEST_F(MinimalGaussianRI, GaussianSelfIntegralAndScreenedExchange)
{
    load(false);
    // Replace the first H s orbital by a normalized Gaussian whose square lies
    // exactly in the minimal auxiliary space. This gives an analytic absolute-unit reference.
    LR::MinimalGaussianBasisSettings ao_settings;
    ao_settings.theta = settings.theta * 0.5;
    ao_settings.lmax = 0;
    ao_settings.nk = orbitals.get_kmesh();
    ao_settings.dk = orbitals.get_dk();
    const auto gaussian_ao = LR::make_minimal_gaussian_basis(atomic_numbers, ao_settings);
    orbitals.Phi[0].chi()[0] = gaussian_ao[0][0][0];
    const double beta = LR::minimal_gaussian_exponent(1, ao_settings.theta);
    const double pi = std::acos(-1.0);
    const double unscreened = 2.0 * std::sqrt(beta / pi);
    settings.exchange_omega = 0.11;
    const double omega = settings.exchange_omega;
    const double screened = unscreened * (1.0 - omega / std::sqrt(beta + omega * omega));
    const auto fits = LR::build_minimal_gaussian_ri<double>(
        MPI_COMM_WORLD, cell, orbitals, atomic_numbers, period, settings);
    LR::RIHartreeFock<double> kernel(MPI_COMM_WORLD, fits.geometry, fits.hartree, fits.exchange);
    const auto density = probe_density<double>(fits.geometry);
    const auto j = kernel.apply_hartree(density);
    const auto k = kernel.apply_exchange(density);
    const double computed_j = onsite(j);
    const double computed_k = onsite(k);
    std::cout << "Gaussian self J/K (Ha): " << std::setprecision(12) << computed_j << " " << computed_k
              << "; analytic " << unscreened << " " << screened << std::endl;
    EXPECT_NEAR(computed_j, unscreened, 2e-6);
    EXPECT_NEAR(computed_k, screened, 2e-6);
}

TEST_F(MinimalGaussianRI, NumericalAOsWithSeparateSPDAndSFits)
{
    load(true);
    settings.exchange_omega = 0.11;
    const auto start = std::chrono::steady_clock::now();
    const auto fits = LR::build_minimal_gaussian_ri<std::complex<double>>(
        MPI_COMM_WORLD, cell, orbitals, atomic_numbers, period, settings);
    EXPECT_EQ(fits.hartree.auxiliary_per_atom, (std::vector<int>{1, 9}));
    EXPECT_EQ(fits.exchange.auxiliary_per_atom, (std::vector<int>{1, 1}));
    LR::RIHartreeFock<std::complex<double>> kernel(MPI_COMM_WORLD, fits.geometry, fits.hartree, fits.exchange);
    const auto density = probe_density<std::complex<double>>(fits.geometry);
    const auto j = kernel.apply_hartree(density);
    const auto k = kernel.apply_exchange(density);
    const auto j0 = onsite(j);
    const auto k0 = onsite(k);
    EXPECT_GT(j0.real(), 0.0);
    EXPECT_GT(k0.real(), 0.0);
    EXPECT_NEAR(j0.imag(), 0.0, 1e-12);
    EXPECT_NEAR(k0.imag(), 0.0, 1e-12);
    const std::complex<double> phase(0.3, -0.7);
    auto response = density;
    for (auto& row : response)
    {
        for (auto& block : row.second) { block.second = block.second * phase; }
    }
    const auto j_response = kernel.apply_hartree(response);
    const auto k_response = kernel.apply_exchange(response);
    const auto j1 = onsite(j_response);
    const auto k1 = onsite(k_response);
    EXPECT_NEAR(std::abs(j1 - phase * j0), 0.0, 1e-11);
    EXPECT_NEAR(std::abs(k1 - phase * k0), 0.0, 1e-11);
    const auto finish = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(finish - start).count();
    std::cout << "Numerical H/O AO minimal RI setup and four actions: " << seconds << " s" << std::endl;
}
}
