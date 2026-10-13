#include "minimal_gaussian_ri.h"

#include "exx_abfs_ctor_orbs.h"
#include "lri_cv.h"
#include "minimal_gaussian_basis.h"
#include <RI/distribute/Distribute_Equally.h>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace LR
{
namespace
{
struct GaussianFittingInputs
{
    const MPI_Comm& comm;
    const UnitCell& cell;
    const LCAO_Orbitals& orbitals;
    const NumericalAuxiliaryBasis& atomic_orbitals;
    const std::vector<int>& atomic_numbers;
    const MinimalGaussianRISettings& settings;
    const MinimalGaussianBasisSettings& basis_mesh;
    std::shared_ptr<ORB_gaunt_table> gaunt;
};

std::string decimal(const double value)
{
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

MinimalGaussianBasisSettings gaussian_mesh(const LCAO_Orbitals& orbitals,
                                           const MinimalGaussianRISettings& settings)
{
    MinimalGaussianBasisSettings mesh;
    mesh.theta = settings.theta;
    const double expanded_kmesh = orbitals.get_kmesh() * settings.kmesh_multiplier;
    if (!std::isfinite(expanded_kmesh) || expanded_kmesh > std::numeric_limits<int>::max() - 1)
    {
        throw std::invalid_argument("Minimal Gaussian RI momentum mesh is too large");
    }
    mesh.nk = static_cast<int>(expanded_kmesh) | 1;
    mesh.dk = orbitals.get_dk();
    mesh.dr_uniform = orbitals.get_dr_uniform();
    // Center2_Orb::Orb21 multiplies radial arrays point by point, without interpolation.
    // Therefore the Gaussian step must equal the AO step for each species, not an arbitrary mesh.
    for (int species = 0; species < orbitals.get_ntype(); ++species)
    {
        const auto& atom = orbitals.Phi[species];
        const double step = atom.PhiLN(0, 0).getRab(0);
        for (int l = 0; l <= atom.getLmax(); ++l)
        {
            for (int n = 0; n < atom.getNchi(l); ++n)
            {
                const auto& orbital = atom.PhiLN(l, n);
                for (int ir = 0; ir < orbital.getNr(); ++ir)
                {
                    const double expected_radius = ir * step;
                    if (std::abs(orbital.getRadial(ir) - expected_radius) > 1e-10
                        || std::abs(orbital.getRab(ir) - step) > 1e-10)
                    {
                        throw std::invalid_argument("Minimal Gaussian RI requires uniform, matching AO radial grids");
                    }
                }
            }
        }
        mesh.species_radial_steps.push_back(step);
    }
    return mesh;
}

template<typename T> RIFittingTensors<T> fit_gaussian_products(
    const GaussianFittingInputs& input, const int lmax, const double omega)
{
    auto basis_settings = input.basis_mesh;
    basis_settings.lmax = lmax;
    const auto auxiliary = make_minimal_gaussian_basis(input.atomic_numbers, basis_settings);
    const auto auxiliary_cutoffs = Exx_Abfs::Construct_Orbs::get_Rcut(auxiliary);
    const double largest_auxiliary_cutoff = *std::max_element(auxiliary_cutoffs.begin(), auxiliary_cutoffs.end());
    const double smallest_auxiliary_cutoff = *std::min_element(auxiliary_cutoffs.begin(), auxiliary_cutoffs.end());
    // A Gaussian shell is cut at alpha*r^2=36. The convolved radial functions must extend
    // past rc + r_aux(A) + r_aux(B), even when an auxiliary orbital is longer than an AO.
    const double kernel_range = omega == 0.0 ? input.settings.coulomb_cutoff : 6.0 / omega;
    const double metric_range = kernel_range + 2.0 * largest_auxiliary_cutoff;
    const double convolution_mesh_multiplier = metric_range / smallest_auxiliary_cutoff + 1.0;
    for (const auto& species : auxiliary)
    {
        const double radial_points = species[0][0].getNr() * convolution_mesh_multiplier;
        if (!std::isfinite(radial_points) || radial_points > 1000000.0)
        {
            throw std::invalid_argument("Minimal Gaussian RI convolution mesh exceeds one million points");
        }
    }
    CoulombParam coulomb;
    const auto kernel = omega == 0.0 ? Conv_Coulomb_Pot_K::Coulomb_Type::Fock
                                    : Conv_Coulomb_Pot_K::Coulomb_Type::Erfc;
    // For screened exchange use the standard HSE Fourier kernel. Applying a sharp cutoff
    // would enter the legacy complex-erf implementation, whose tail truncation is inaccurate.
    // Numerical radial/image ranges instead extend to omega*r=6 (erfc tail below 3e-17).
    const std::string coulomb_method = omega == 0.0 ? "spencer" : "limits";
    coulomb[kernel] = {{{"alpha", "1"}, {"singularity_correction", coulomb_method},
                         {"Rcut", decimal(input.settings.coulomb_cutoff)}, {"omega", decimal(omega)}}};
    const auto convolved = Conv_Coulomb_Pot_K::cal_orbs_ccp(auxiliary, coulomb, convolution_mesh_multiplier);
    Exx_Info_RI fitting_info;
    fitting_info.Cs_inv_thr = input.settings.metric_relative_cutoff;
    LRI_CV<T> integrals;
    integrals.set_info_ri(&fitting_info);
    integrals.set_orbitals(input.cell, input.orbitals, input.atomic_orbitals, auxiliary,
                          convolved, input.settings.kmesh_multiplier, input.gaunt, true);

    std::vector<int> atoms(input.cell.nat);
    for (int a = 0; a < input.cell.nat; ++a) { atoms[a] = a; }
    const auto orbital_cutoffs = Exx_Abfs::Construct_Orbs::get_Rcut(input.atomic_orbitals);
    const auto coefficient_cells = LRI_CV_Tools::cal_latvec_range<int>(2.0, input.cell, orbital_cutoffs);
    const std::vector<double> metric_cutoffs(input.cell.ntype, metric_range);
    const auto metric_cells = LRI_CV_Tools::cal_latvec_range<int>(1.0, input.cell, metric_cutoffs);
    const auto coefficient_pairs = RI::Distribute_Equally::distribute_atoms_periods(
        input.comm, atoms, coefficient_cells, 2, false);
    const auto metric_pairs = RI::Distribute_Equally::distribute_atoms_periods(
        input.comm, atoms, metric_cells, 2, false);
    const std::map<std::string, bool> metric_flags{{"writable_Vws", true}};
    const std::map<std::string, bool> coefficient_flags{
        {"cal_dC", false}, {"writable_Cws", true}, {"writable_dCws", false},
        {"writable_Vws", true}, {"writable_dVws", false}};
    RIFittingTensors<T> fit;
    fit.Vs = integrals.cal_Vs(input.cell, metric_pairs.first, metric_pairs.second[0], metric_flags);
    auto coefficients = integrals.cal_Cs_dCs(
        input.cell, coefficient_pairs.first, coefficient_pairs.second[0], coefficient_flags);
    fit.Cs = std::move(coefficients.first);
    fit.auxiliary_per_atom.resize(input.cell.nat);
    for (int a = 0; a < input.cell.nat; ++a)
    {
        const int species = input.cell.iat2it[a];
        const int shells = auxiliary[species].size();
        fit.auxiliary_per_atom[a] = shells * shells;
    }
    return fit;
}
}

template<typename T> MinimalGaussianRIFits<T> build_minimal_gaussian_ri(
    const MPI_Comm& comm, const UnitCell& cell, const LCAO_Orbitals& orbitals,
    const std::vector<int>& atomic_numbers, const RICell& period,
    const MinimalGaussianRISettings& settings)
{
    if (cell.ntype != orbitals.get_ntype() || atomic_numbers.size() != cell.ntype || cell.nat <= 0
        || !std::isfinite(settings.coulomb_cutoff) || settings.coulomb_cutoff <= 0.0
        || !std::isfinite(settings.exchange_omega) || settings.exchange_omega < 0.0
        || !std::isfinite(settings.kmesh_multiplier) || settings.kmesh_multiplier < 1.0
        || !std::isfinite(settings.metric_relative_cutoff) || settings.metric_relative_cutoff <= 0.0
        || settings.metric_relative_cutoff >= 1.0)
    {
        throw std::invalid_argument("Invalid minimal Gaussian RI species, cutoff, kernel, or integration settings");
    }
    for (int d = 0; d < 3; ++d)
    {
        if (period[d] <= 0) { throw std::invalid_argument("Gaussian RI requires a positive BvK period"); }
    }
    const auto mesh = gaussian_mesh(orbitals, settings);
    const auto atomic_orbitals = Exx_Abfs::Construct_Orbs::change_orbs(orbitals, settings.kmesh_multiplier);
    const auto gaunt = std::make_shared<ORB_gaunt_table>();
    const GaussianFittingInputs input{comm, cell, orbitals, atomic_orbitals, atomic_numbers, settings, mesh, gaunt};
    MinimalGaussianRIFits<T> result;
    result.hartree = fit_gaussian_products<T>(input, settings.hartree_lmax, 0.0);
    result.exchange = fit_gaussian_products<T>(input, settings.exchange_lmax, settings.exchange_omega);
    result.geometry.period = period;
    result.geometry.lattice = {{RI_Util::Vector3_to_array3(cell.a1), RI_Util::Vector3_to_array3(cell.a2),
                               RI_Util::Vector3_to_array3(cell.a3)}};
    result.geometry.orbitals_per_atom.resize(cell.nat);
    for (int a = 0; a < cell.nat; ++a)
    {
        const auto& position = cell.get_tau(a);
        result.geometry.positions[a] = RI_Util::Vector3_to_array3(position);
        const int species = cell.iat2it[a];
        int nao = 0;
        for (int l = 0; l <= orbitals.Phi[species].getLmax(); ++l)
        {
            nao += (2 * l + 1) * orbitals.Phi[species].getNchi(l);
        }
        result.geometry.orbitals_per_atom[a] = nao;
    }
    return result;
}

template<typename T> std::unique_ptr<RIHartreeFock<T>> make_minimal_gaussian_ri(
    const MPI_Comm& comm, const UnitCell& cell, const LCAO_Orbitals& orbitals,
    const std::vector<int>& atomic_numbers, const RICell& period,
    const MinimalGaussianRISettings& settings)
{
    const auto fits = build_minimal_gaussian_ri<T>(comm, cell, orbitals, atomic_numbers, period, settings);
    return std::unique_ptr<RIHartreeFock<T>>(new RIHartreeFock<T>(comm, fits.geometry, fits.hartree, fits.exchange));
}

template MinimalGaussianRIFits<double> build_minimal_gaussian_ri(
    const MPI_Comm&, const UnitCell&, const LCAO_Orbitals&, const std::vector<int>&,
    const RICell&, const MinimalGaussianRISettings&);
template MinimalGaussianRIFits<std::complex<double>> build_minimal_gaussian_ri(
    const MPI_Comm&, const UnitCell&, const LCAO_Orbitals&, const std::vector<int>&,
    const RICell&, const MinimalGaussianRISettings&);
template std::unique_ptr<RIHartreeFock<double>> make_minimal_gaussian_ri(
    const MPI_Comm&, const UnitCell&, const LCAO_Orbitals&, const std::vector<int>&,
    const RICell&, const MinimalGaussianRISettings&);
template std::unique_ptr<RIHartreeFock<std::complex<double>>> make_minimal_gaussian_ri(
    const MPI_Comm&, const UnitCell&, const LCAO_Orbitals&, const std::vector<int>&,
    const RICell&, const MinimalGaussianRISettings&);
}
