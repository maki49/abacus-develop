#include "module_hamilt_lcao/module_gint/gint_gamma.h"
#include "module_hamilt_lcao/module_gint/gint_k.h"
#include "module_psi/psi.h"
#include "module_elecstate/module_dm/density_matrix.h"

template <typename T> struct TGint;
template <>
struct TGint<double> {
    using type = Gint_Gamma;
};
template <>
struct TGint<std::complex<double>> {
    using type = Gint_k;
};

template<typename T>
class LR_Spectrum
{
public:
    LR_Spectrum(const double* eig, const psi::Psi<T>& X, const int& nspin, const int& naos, const int& nocc, const int& nvirt,
        typename TGint<T>::type* gint, const ModulePW::PW_Basis& rho_basis, psi::Psi<T>& psi_ks,
        const UnitCell& ucell, const K_Vectors& kv_in, Parallel_2D& pX_in, Parallel_2D& pc_in, Parallel_Orbitals& pmat_in) :
        eig(eig), X(X), nspin(nspin), naos(naos), nocc(nocc), nvirt(nvirt),
        gint(gint), rho_basis(rho_basis), psi_ks(psi_ks),
        ucell(ucell), kv(kv_in), pX(pX_in), pc(pc_in), pmat(pmat_in) {};
    /// $$2/3\Omega\sum_{ia\sigma} |\braket{\psi_{i}|\mathbf{r}|\psi_{a}} |^2\int \rho_{\alpha\beta}(\mathbf{r}) \mathbf{r} d\mathbf{r}$$
    void oscillator_strength();
    /// @brief calculate the optical absorption spectrum
    void optical_absorption(const std::vector<double>& freq, const double eta);
    /// @brief print out the transition dipole moment and the main contributions to the transition amplitude
    void transition_analysis();
private:
    const int nspin;
    const int naos;
    const int nocc;
    const int nvirt;
    const double* eig;
    const psi::Psi<T>& X;
    const K_Vectors& kv;
    const psi::Psi<T>& psi_ks;
    Parallel_2D& pX;
    Parallel_2D& pc;
    Parallel_Orbitals& pmat;
    typename TGint<T>::type* gint = nullptr;
    const ModulePW::PW_Basis& rho_basis;
    const UnitCell& ucell;

    void cal_gint_rho(double** rho, const int& nspin, const int& nrxx);

    std::vector<ModuleBase::Vector3<T>> transition_dipole_;   // \braket{ \psi_{i} | \mathbf{r} | \psi_{a} }
    std::vector<double> oscillator_strength_;// 2/3\Omega |\sum_{ia\sigma} \braket{\psi_{i}|\mathbf{r}|\psi_{a}} |^2
};