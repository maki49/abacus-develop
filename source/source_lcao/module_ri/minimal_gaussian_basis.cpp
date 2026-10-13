#include "minimal_gaussian_basis.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace LR
{
double minimal_gaussian_exponent(const int atomic_number, const double theta)
{
    // Ghosh et al., J. Mol. Struct.: THEOCHEM 865 (2008), 60-67.
    // Numeric radii (angstrom) and conversion used in the authors' parameter.py:
    // https://github.com/John-zzh/pyscf_TDDFT_ris/blob/main/pyscf_TDDFT_ris/parameter.py
    // Indexed by Z-1, H through Lr (103 entries).
    static const double radii[] = {
        0.5292, 0.3113, 1.6283, 1.0855, 0.8141, 0.6513, 0.5428, 0.4652, 0.4071, 0.3618,
        2.165, 1.6711, 1.3608, 1.1477, 0.9922, 0.8739, 0.7808, 0.7056, 3.293, 2.5419,
        2.4149, 2.2998, 2.1953, 2.1, 2.0124, 1.9319, 1.8575, 1.7888, 1.725, 1.6654,
        1.4489, 1.2823, 1.145, 1.0424, 0.9532, 0.8782, 3.8487, 2.9709, 2.8224, 2.688,
        2.5658, 2.4543, 2.352, 2.2579, 2.1711, 2.0907, 2.016, 1.9465, 1.6934, 1.4986,
        1.344, 1.2183, 1.1141, 1.0263, 4.2433, 3.2753, 2.6673, 2.2494, 1.9447, 1.7129,
        1.5303, 1.383, 1.2615, 1.1596, 1.073, 0.9984, 0.9335, 0.8765, 0.8261, 0.7812,
        0.7409, 0.7056, 0.6716, 0.6416, 0.6141, 0.589, 0.5657, 0.5443, 0.5244, 0.506,
        1.867, 1.6523, 1.4818, 1.3431, 1.2283, 1.1315, 4.4479, 3.4332, 3.2615, 3.1061,
        2.2756, 1.9767, 1.7473, 1.4496, 1.2915, 1.296, 1.1247, 1.0465, 0.9785, 0.9188,
        0.8659, 0.8188, 0.8086};
    if (atomic_number < 1 || atomic_number > 103 || !std::isfinite(theta) || theta <= 0.0)
    {
        throw std::invalid_argument("Minimal Gaussian basis requires Z=1..103 and finite theta > 0");
    }
    const double radius_bohr = radii[atomic_number - 1] * 1.8897259885789;
    return theta / (radius_bohr * radius_bohr);
}

NumericalAuxiliaryBasis make_minimal_gaussian_basis(
    const std::vector<int>& atomic_numbers, const MinimalGaussianBasisSettings& settings)
{
    if (atomic_numbers.empty() || settings.lmax < 0 || settings.lmax > 2
        || !std::isfinite(settings.radial_step) || settings.radial_step <= 0.0
        || !std::isfinite(settings.tail_exponent) || settings.tail_exponent < 20.0
        || settings.nk < 3 || settings.nk % 2 == 0
        || !std::isfinite(settings.dk) || settings.dk <= 0.0
        || !std::isfinite(settings.dr_uniform) || settings.dr_uniform <= 0.0)
    {
        throw std::invalid_argument("Invalid minimal Gaussian basis mesh or angular cutoff");
    }
    if (!settings.species_radial_steps.empty() && settings.species_radial_steps.size() != atomic_numbers.size())
    {
        throw std::invalid_argument("Gaussian radial steps must match the number of species");
    }
    NumericalAuxiliaryBasis basis(atomic_numbers.size());
    for (std::size_t species = 0; species < atomic_numbers.size(); ++species)
    {
        const int atomic_number = atomic_numbers[species];
        const double exponent = minimal_gaussian_exponent(atomic_number, settings.theta);
        const double radial_step = settings.species_radial_steps.empty()
                                       ? settings.radial_step : settings.species_radial_steps[species];
        if (!std::isfinite(radial_step) || radial_step <= 0.0)
        {
            throw std::invalid_argument("Invalid species Gaussian radial step");
        }
        const int lmax = atomic_number == 1 ? 0 : settings.lmax;
        const double cutoff = std::sqrt(settings.tail_exponent / exponent);
        const double mesh_intervals = std::ceil(cutoff / radial_step);
        if (mesh_intervals > 1000000.0)
        {
            throw std::invalid_argument("Minimal Gaussian radial mesh exceeds one million points");
        }
        const int nr = (static_cast<int>(mesh_intervals) + 1) | 1;
        std::vector<double> radial(nr);
        std::vector<double> rab(nr, radial_step);
        for (int ir = 0; ir < nr; ++ir) { radial[ir] = ir * radial_step; }
        basis[species].resize(lmax + 1);
        for (int l = 0; l <= lmax; ++l)
        {
            // Integral_0^infinity |R_l(r)|^2 r^2 dr = 1; Y_lm has unit angular norm.
            const double gamma_argument = l + 1.5;
            const double twice_exponent = 2.0 * exponent;
            const double normalization_squared
                = 2.0 * std::pow(twice_exponent, gamma_argument) / std::tgamma(gamma_argument);
            const double normalization = std::sqrt(normalization_squared);
            std::vector<double> radial_gaussian(nr);
            for (int ir = 0; ir < nr; ++ir)
            {
                const double radius = radial[ir];
                radial_gaussian[ir] = normalization * std::pow(radius, l)
                                      * std::exp(-exponent * radius * radius);
            }
            basis[species][l].resize(1);
            const std::string label = "minimal_Gaussian_Z" + std::to_string(atomic_number);
            const int species_index = species;
            const double* rab_data = rab.data();
            const double* radial_data = radial.data();
            const double* gaussian_data = radial_gaussian.data();
            basis[species][l][0].set_orbital_info(label, species_index, l, 0, nr, rab_data,
                                                radial_data, Numerical_Orbital_Lm::Psi_Type::Psi,
                                                gaussian_data, settings.nk, settings.dk,
                                                settings.dr_uniform, false, true, false);
        }
    }
    return basis;
}
}
