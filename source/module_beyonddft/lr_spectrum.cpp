#include "lr_spectrum.h"
#include "module_beyonddft/utils/lr_util.h"
#include "module_beyonddft/dm_trans/dm_trans.h"
#include "module_base/parallel_reduce.h"

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
    ofs << std::setw(8) << "State" << std::setw(20) << "Occupied orbital"
        << std::setw(20) << "Virtual orbital" << std::setw(30) << "Excitation amplitude"
        << std::setw(30) << "Excitation rate"
        << std::setw(10) << "k-point" << std::endl;
    ofs << "------------------------------------------------------------------------------------ " << std::endl;
    for (int istate = 0;istate < X.get_nbands();++istate)
    {
        /// find the main contributions (> 0.5)
        X.fix_b(istate);
        psi::Psi<T> X_full(X.get_nk(), 1, nocc * nvirt, nullptr, false);// one-band
        X_full.zero_out();
        for (int isk = 0;isk < X.get_nk();++isk)
        {
            X.fix_k(isk);
            X_full.fix_k(isk);
            LR_Util::gather_2d_to_full(this->pX, X.get_pointer(), X_full.get_pointer(), false, nvirt, nocc);
        }
        std::map<double, int, std::greater<double>> abs_order;
        X_full.fix_k(0);
        for (int i = 0;i < X.get_nk() * nocc * nvirt;++i) { double abs = std::abs(X_full.get_pointer()[i]);if (abs > 0.3) abs_order[abs] = i; }
        if (abs_order.size() > 0)
            for (auto it = abs_order.cbegin();it != abs_order.cend();++it)
            {
                int ik = it->second / (nocc * nvirt);
                int ipair = it->second - ik * nocc * nvirt;
                ofs << std::setw(8) << (it == abs_order.cbegin() ? std::to_string(istate) : " ")
                    << std::setw(20) << ipair / nvirt + 1 << std::setw(20) << ipair % nvirt + nocc + 1// iocc and ivirt
                    << std::setw(30) << X_full(ik, ipair)
                    << std::setw(30) << (std::conj(X_full(ik, ipair)) * X_full(ik, ipair)).real()
                    << std::setw(10) << ik << std::endl;
            }
    }
    ofs << "==================================================================== " << std::endl;
    X.fix_kb(0, 0);
}

template class LR_Spectrum<double>;
template class LR_Spectrum<std::complex<double>>;