#include "operator_lr_hxc.h"
#include <vector>
#include "module_base/blas_connector.h"
#include "module_base/timer.h"
#include "module_beyonddft/utils/lr_util.h"
#include "module_beyonddft/utils/lr_util_hcontainer.h"
// #include "module_hamilt_lcao/hamilt_lcaodft/DM_gamma_2d_to_grid.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer_funcs.h"
#include "module_beyonddft/dm_trans/dm_trans.h"
#include "module_beyonddft/AX/AX.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

inline double conj(double a) { return a; }
inline std::complex<double> conj(std::complex<double> a) { return std::conj(a); }

template<typename T>
inline void print_psi_bandfirst(const psi::Psi<T>& psi, const std::string& label, const int& ib)
{
    assert(psi.get_k_first() == 0);
    std::cout << label << ": band " << ib << "\n";
    for (int ik = 0;ik < psi.get_nk();++ik)
    {
        std::cout << "iks=" << ik << "\n";
        for (int i = 0;i < psi.get_nbasis();++i)std::cout << psi(ib, ik, i) << " ";
        std::cout << "\n";
    }
}
template<typename T>
inline void print_tensor(const container::Tensor& t, const std::string& label, const Parallel_Orbitals* pmat)
{
    std::cout << label << "\n";
    for (int j = 0; j < pmat->get_col_size();++j)
    {
        for (int i = 0;i < pmat->get_row_size();++i)
            std::cout << t.data<T>()[j * pmat->get_row_size() + i] << " ";
        std::cout << std::endl;
    }
    std::cout << "\n";
}
inline void print_grid_nonzero(double* rho, const int& nrxx, const int& nnz, const std::string& label, const bool& threshold = 1e-5)
{
    std::cout << "first " << nnz << " non-zero elements of " << label << "\n";
    int inz = 0;int i = 0;
    while (inz < nnz && i < nrxx)
        if (rho[++i] - 0.0 > threshold) { std::cout << rho[i] << " ";++inz; };
}

namespace hamilt
{
    template<typename T, typename Device>
    void OperatorLRHxc<T, Device>::act(const psi::Psi<T>& psi_in, psi::Psi<T>& psi_out, const int nbands) const
    {
        ModuleBase::TITLE("OperatorLRHxc", "act");
        assert(nbands <= psi_in.get_nbands());
        const int& nks = this->kv.nks;

        this->init_DM_trans(nbands, this->DM_trans);    // initialize transion density matrix

        psi::Psi<T> psi_in_bfirst = LR_Util::k1_to_bfirst_wrapper(psi_in, nks, this->pX->get_local_size());
        psi::Psi<T> psi_out_bfirst = LR_Util::k1_to_bfirst_wrapper(psi_out, nks, this->pX->get_local_size());

        const int& lgd = gint->gridt->lgd;
        for (int ib = 0;ib < nbands;++ib)
        {
            // if Hxc-only, the memory of single-band DM_trans is enough.
            // if followed by EXX, we need to allocate memory for all bands.
            int ib_dm = (this->next_op == nullptr) ? 0 : ib;
            psi_in_bfirst.fix_b(ib);
            psi_out_bfirst.fix_b(ib);

            // 1. transition density matrix
#ifdef __MPI
            std::vector<container::Tensor>  dm_trans_2d = cal_dm_trans_pblas(psi_in_bfirst, *pX, *psi_ks, *pc, naos, nocc, nvirt, *pmat);
            if (this->tdm_sym) for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos, *pmat);
#else
            std::vector<container::Tensor>  dm_trans_2d = cal_dm_trans_blas(psi_in_bfirst, psi_ks, nocc, nvirt);
            if (this->tdm_sym) for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos);
#endif
            // tensor to vector, then set DMK
            for (int isk = 0;isk < nks;++isk)this->DM_trans[ib_dm]->set_DMK_pointer(isk, dm_trans_2d[isk].data<T>());

            // use cal_DMR to get DMR form DMK by FT
            this->DM_trans[ib_dm]->cal_DMR();  //DM_trans->get_DMR_vector() is 2d-block parallized
            // LR_Util::print_DMR(*this->DM_trans[0], ucell.nat, "DM(R) (complex)");

            // ========================= begin grid calculation=========================
            this->grid_calculation(nbands, ib_dm);   //DM(R) to H(R)
            // ========================= end grid calculation =========================

            // V(R)->V(k)
            std::vector<ct::Tensor> v_hxc_2d(this->kv.nks,
                ct::Tensor(ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<psi::DEVICE_CPU>::value,
                    { pmat->get_col_size(), pmat->get_row_size() }));
            for (auto& v : v_hxc_2d) v.zero();
            int nrow = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER() ? this->pmat->get_row_size() : this->pmat->get_col_size();
            for (int isk = 0;isk < nks;++isk)
                hamilt::folding_HR(*this->hR, v_hxc_2d[isk].data<T>(), this->kv.kvec_d[isk], nrow, 1);            // V(R) -> V(k)
            // LR_Util::print_HR(*this->hR, this->ucell.nat, "4.VR");
            // print_tensor<T>(v_hxc_2d[0], "4.V(k)", this->pmat);

            // 5. [AX]^{Hxc}_{ai}=\sum_{\mu,\nu}c^*_{a,\mu,}V^{Hxc}_{\mu,\nu}c_{\nu,i}
#ifdef __MPI
            cal_AX_pblas(v_hxc_2d, *this->pmat, *this->psi_ks, *this->pc, naos, nocc, nvirt, *this->pX, psi_out_bfirst);
#else
            cal_AX_blas(v_hxc_2d, *this->psi_ks, nocc, nvirt, psi_out_bfirst);
#endif
            // print_psi_bandfirst(psi_out_bfirst, "5.AX", ib);
        }
    }


    template<>
    void OperatorLRHxc<double, psi::DEVICE_CPU>::grid_calculation(const int& nbands, const int& iband_dm) const
    {
        ModuleBase::TITLE("OperatorLRHxc", "grid_calculation(real)");
        ModuleBase::timer::tick("OperatorLRHxc", "grid_calculation");
        this->gint->transfer_DM2DtoGrid(this->DM_trans[iband_dm]->get_DMR_vector());     // 2d block to grid

        // 2. transition electron density
        // \f[ \tilde{\rho}(r)=\sum_{\mu_j, \mu_b}\tilde{\rho}_{\mu_j,\mu_b}\phi_{\mu_b}(r)\phi_{\mu_j}(r) \f]
        double** rho_trans;
        LR_Util::new_p2(rho_trans, nspin, this->pot->nrxx);
        for (int is = 0;is < nspin;++is)ModuleBase::GlobalFunc::ZEROS(rho_trans[is], this->pot->nrxx);
        Gint_inout inout_rho((double**)nullptr, rho_trans, Gint_Tools::job_type::rho, false);
        this->gint->cal_gint(&inout_rho);

        // 3. v_hxc = f_hxc * rho_trans
        ModuleBase::matrix vr_hxc(nspin, this->pot->nrxx);   //grid
        this->pot->cal_v_eff(rho_trans, &GlobalC::ucell, vr_hxc);
        LR_Util::delete_p2(rho_trans, nspin);

        // 4. V^{Hxc}_{\mu,\nu}=\int{dr} \phi_\mu(r) v_{Hxc}(r) \phi_\mu(r)
        // V(R) for each spin
        for (int is = 0;is < nspin;++is)
        {
            double* vr_hxc_is = &vr_hxc.c[is * this->pot->nrxx];   //v(r) at current spin
            Gint_inout inout_vlocal(vr_hxc_is, is, Gint_Tools::job_type::vlocal);
            this->gint->get_hRGint()->set_zero();
            this->gint->cal_gint(&inout_vlocal);
        }
        this->hR->set_zero();   // clear hR for each bands
        this->gint->transfer_pvpR(this->hR);    //grid to 2d block
        ModuleBase::timer::tick("OperatorLRHxc", "grid_calculation");
    }

    template<>
    void OperatorLRHxc<std::complex<double>, psi::DEVICE_CPU>::grid_calculation(const int& nbands, const int& iband_dm) const
    {
        ModuleBase::TITLE("OperatorLRHxc", "grid_calculation(complex)");
        ModuleBase::timer::tick("OperatorLRHxc", "grid_calculation");

        elecstate::DensityMatrix<std::complex<double>, double> DM_trans_real_imag(&kv, pmat, nspin);
        DM_trans_real_imag.init_DMR(*this->hR);
        HContainer<double> HR_real_imag(GlobalC::ucell, this->pmat);
        this->initialize_HR(&HR_real_imag, ucell, gd, this->pmat);

        auto dmR_to_hR = [&, this](const int& iband_dm, const char& type) -> void
            {
                LR_Util::get_DMR_real_imag_part(*this->DM_trans[iband_dm], DM_trans_real_imag, ucell.nat, type);
                // LR_Util::print_DMR(DM_trans_real_imag, ucell.nat, "DMR(2d, real)");
                this->gint->transfer_DM2DtoGrid(DM_trans_real_imag.get_DMR_vector());
                // LR_Util::print_HR(*this->gint->get_DMRGint()[0], this->ucell.nat, "DMR(grid, real)");

                // 2. transition electron density
                double** rho_trans;
                LR_Util::new_p2(rho_trans, nspin, this->pot->nrxx);
                for (int is = 0;is < nspin;++is)ModuleBase::GlobalFunc::ZEROS(rho_trans[is], this->pot->nrxx);
                Gint_inout inout_rho((double**)nullptr, rho_trans, Gint_Tools::job_type::rho, false);
                this->gint->cal_gint(&inout_rho);
                // print_grid_nonzero(rho_trans[0], this->pot->nrxx, 10, "rho_trans");

                // 3. v_hxc = f_hxc * rho_trans
                ModuleBase::matrix vr_hxc(nspin, this->pot->nrxx);   //grid
                this->pot->cal_v_eff(rho_trans, &GlobalC::ucell, vr_hxc);
                // print_grid_nonzero(vr_hxc.c, this->pot->nrxx, 10, "vr_hxc");

                LR_Util::delete_p2(rho_trans, nspin);

                // 4. V^{Hxc}_{\mu,\nu}=\int{dr} \phi_\mu(r) v_{Hxc}(r) \phi_\mu(r)
                for (int is = 0;is < nspin;++is)
                {
                    double* vr_hxc_is = &vr_hxc.c[is * this->pot->nrxx];   //v(r) at current spin
                    Gint_inout inout_vlocal(vr_hxc_is, is, Gint_Tools::job_type::vlocal);
                    this->gint->get_hRGint()->set_zero();
                    this->gint->cal_gint(&inout_vlocal);
                }
                // LR_Util::print_HR(*this->gint->get_hRGint(), this->ucell.nat, "VR(grid)");
                HR_real_imag.set_zero();
                this->gint->transfer_pvpR(&HR_real_imag);
                // LR_Util::print_HR(HR_real_imag, this->ucell.nat, "VR(real, 2d)");
                LR_Util::set_HR_real_imag_part(HR_real_imag, *this->hR, GlobalC::ucell.nat, type);
            };
        this->hR->set_zero();
        dmR_to_hR(iband_dm, 'R');   //real
        if (kv.nks / this->nspin > 1)dmR_to_hR(iband_dm, 'I');   //imag for multi-k
        ModuleBase::timer::tick("OperatorLRHxc", "grid_calculation");
    }

    template class OperatorLRHxc<double>;
    template class OperatorLRHxc<std::complex<double>>;
}