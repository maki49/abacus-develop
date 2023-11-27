#pragma once
#include "module_psi/psi.h"
#include "module_elecstate/module_dm/density_matrix.h"
#include "module_beyonddft/utils/lr_util.h"
#include "module_hamilt_lcao/module_gint/gint_gamma.h"
#include "module_hamilt_lcao/module_gint/gint_k.h"

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
        ucell(ucell), kv(kv_in), pX(pX_in), pc(pc_in), pmat(pmat_in),
        nsk(std::is_same<T, double>::value ? nspin : kv_in.kvec_d.size()) {};
    /// $$2/3\Omega\sum_{ia\sigma} |\braket{\psi_{i}|\mathbf{r}|\psi_{a}} |^2\int \rho_{\alpha\beta}(\mathbf{r}) \mathbf{r} d\mathbf{r}$$
    void oscillator_strength();
    /// @brief calculate the optical absorption spectrum
    void optical_absorption(const std::vector<double>& freq, const double eta);
    /// @brief print out the transition dipole moment and the main contributions to the transition amplitude
    void transition_analysis();
private:
    const int nspin;
    const int nsk;
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

    std::vector<ModuleBase::Vector3<double>> transition_dipole_;   // \braket{ \psi_{i} | \mathbf{r} | \psi_{a} }
    std::vector<double> oscillator_strength_;// 2/3\Omega |\sum_{ia\sigma} \braket{\psi_{i}|\mathbf{r}|\psi_{a}} |^2
};

template<typename T>
void LR_Spectrum<T>::oscillator_strength()
{
    ModuleBase::TITLE("LR_Spectrum", "oscillator_strength");
    std::vector<double>& osc = this->oscillator_strength_;
    osc.resize(X.get_nbands(), 0.0);
    // const int nspin0 = (this->nspin == 2) ? 2 : 1;   use this in NSPIN=4 implementation
    double osc_tot = 0.0;
    elecstate::DensityMatrix<T, double> DM_trans(&this->kv, &this->pmat, this->nspin);
    DM_trans.init_DMR(&GlobalC::GridD, &this->ucell);
    this->transition_dipole_.resize(X.get_nbands(), ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    for (int istate = 0;istate < X.get_nbands();++istate)
    {
        X.fix_b(istate);

        GlobalV::ofs_running << "final X: " << std::endl;
        for (int j = 0;j < pX.get_col_size();++j)  //nbands
        {
            for (int i = 0;i < pX.get_row_size();++i)  //nlocal
                GlobalV::ofs_running << X.get_pointer()[j * pX.get_row_size() + i] << " ";
            GlobalV::ofs_running << std::endl;
        }

        //1. transition density 
#ifdef __MPI
        std::vector<container::Tensor>  dm_trans_2d = hamilt::cal_dm_trans_pblas(X, this->pX, this->psi_ks, this->pc, this->naos, this->nocc, this->nvirt, this->pmat);
        // if (this->tdm_sym) for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos, pmat);
#else
        std::vector<container::Tensor>  dm_trans_2d = hamilt::cal_dm_trans_blas(X, this->psi_ks, this->nocc, this->nvirt);
        // if (this->tdm_sym) for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos);
#endif
        for (int isk = 0;isk < this->nsk;++isk)DM_trans.set_DMK_pointer(isk, dm_trans_2d[isk].data<T>());
        DM_trans.cal_DMR();
        this->gint->transfer_DM2DtoGrid(DM_trans.get_DMR_vector());

        // 2. transition density
        double** rho_trans;
        LR_Util::new_p2(rho_trans, nspin, this->rho_basis.nrxx);
        for (int is = 0;is < nspin;++is)ModuleBase::GlobalFunc::ZEROS(rho_trans[is], this->rho_basis.nrxx);
        Gint_inout inout_rho((double**)nullptr, rho_trans, Gint_Tools::job_type::rho, false);
        this->gint->cal_gint(&inout_rho);

        // 3. transition dipole moment
        for (int ir = 0; ir < rho_basis.nrxx; ++ir)
        {
            int i = ir / (rho_basis.ny * rho_basis.nplane);
            int j = ir / rho_basis.nplane - i * rho_basis.ny;
            int k = ir % rho_basis.nplane + rho_basis.startz_current;
            ModuleBase::Vector3<double> rd(static_cast<double>(i) / rho_basis.nx, static_cast<double>(j) / rho_basis.ny, static_cast<double>(k) / rho_basis.nz);  //+1/2 better?
            rd -= ModuleBase::Vector3<double>(0.5, 0.5, 0.5);   //shift to the center of the grid (need ?)
            ModuleBase::Vector3<double> rc = rd * ucell.latvec * ucell.lat0; // real coordinate
            for (int is = 0;is < nspin;++is) transition_dipole_[istate] += rc * rho_trans[is][ir];
        }
        Parallel_Reduce::reduce_all(transition_dipole_[istate].x);
        Parallel_Reduce::reduce_all(transition_dipole_[istate].y);
        Parallel_Reduce::reduce_all(transition_dipole_[istate].z);
        osc[istate] = transition_dipole_[istate].norm2() * eig[istate] * 2 / 3;
        osc_tot += osc[istate];
    }


    // check sum rule
    // if (std::abs(osc_tot - GlobalV::nelec) > 1e-3)
    //     ModuleBase::WARNING("LR_Spectrum::oscillator_strength",
    //         "sum rule is not satisfied, try more nstates if needed: total oscillator strength = "
    //         + std::to_string(osc_tot) + "nelec = " + std::to_string(GlobalV::nelec));
}

template<typename T>
void LR_Spectrum<T>::optical_absorption(const std::vector<double>& freq, const double eta)
{
    ModuleBase::TITLE("LR_Spectrum", "optical_absorption");
    std::vector<double>& osc = this->oscillator_strength_;
    std::ofstream ofs(GlobalV::global_out_dir + "absorption.dat");
    if (GlobalV::MY_RANK == 0) ofs << "Frequency (eV) | wave length(nm) | Absorption coefficient (a.u.)" << std::endl;
    for (int f = 0;f < freq.size();++f)
    {
        std::complex<double> f_complex = std::complex<double>(freq[f], eta);
        double abs = 0.0;
        for (int i = 0;i < osc.size();++i)  //nstates
            abs += (osc[i] / (f_complex * f_complex - eig[i] * eig[i])).imag();
        if (GlobalV::MY_RANK == 0)ofs << freq[f] * ModuleBase::Ry_to_eV << "\t" << 91.126664 / freq[f] << "\t" << std::abs(abs) << std::endl;
    }
    ofs.close();
}

template<typename T>
void LR_Spectrum<T>::transition_analysis()
{
    ModuleBase::TITLE("LR_Spectrum", "transition_analysis");
    std::ofstream& ofs = GlobalV::ofs_running;
    ofs << "==================================================================== " << std::endl;
    ofs << std::setw(8) << "State" << std::setw(30) << "Excitation Energy (Ry, eV)" <<
        std::setw(45) << "Transition dipole x, y, z (a.u.)" << std::setw(30) << "Oscillator strength(a.u.)" << std::endl;
    ofs << "------------------------------------------------------------------------------------ " << std::endl;
    for (int istate = 0;istate < X.get_nbands();++istate)
        ofs << std::setw(8) << istate << std::setw(15) << std::setprecision(3) << eig[istate] << std::setw(15) << eig[istate] * ModuleBase::Ry_to_eV
        << std::setw(15) << transition_dipole_[istate].x << std::setw(15) << transition_dipole_[istate].y << std::setw(15) << transition_dipole_[istate].z
        << std::setw(30) << oscillator_strength_[istate] << std::endl;
    ofs << "------------------------------------------------------------------------------------ " << std::endl;
    ofs << std::setw(8) << "State" << std::setw(20) << "Occupied orbital" <<
        std::setw(20) << "Virtual orbital" << std::setw(30) << "Excitation amplitude" << std::endl;
    ofs << "------------------------------------------------------------------------------------ " << std::endl;
    for (int istate = 0;istate < X.get_nbands();++istate)
    {
        /// find the main contributions (> 0.5)
        X.fix_b(istate);
        psi::Psi<T> X_full(1, X.get_nk(), nocc * nvirt, nullptr, false);
        X_full.zero_out();
        for (int isk = 0;isk < X.get_nk();++isk)
        {
            X.fix_k(isk);
            X_full.fix_k(isk);
            LR_Util::gather_2d_to_full(this->pX, X.get_pointer(), X_full.get_pointer(), false, nvirt, nocc);
        }
        std::map<double, int, std::greater<double>> abs_order;
        X_full.fix_k(0);
        for (int i = 0;i < X.get_nk() * nocc * nvirt;++i) { double abs = std::abs(X_full.get_pointer()[i]);if (abs > 0.5) abs_order[abs] = i; }
        if (abs_order.size() > 0)
            for (auto it = abs_order.cbegin();it != abs_order.cend();++it)
            {
                int ik = it->second % X.get_nk();
                int ipair = it->second - ik * nocc * nvirt;
                ofs << std::setw(8) << (it == abs_order.cbegin() ? std::to_string(istate) : " ")
                    << std::setw(20) << ipair / nvirt + 1 << std::setw(20) << ipair % nvirt + nocc + 1// iocc and ivirt
                    << std::setw(30) << X_full(ik, ipair) << std::endl;
            }
    }
    ofs << "==================================================================== " << std::endl;
}