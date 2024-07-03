//==========================================================
// AUTHOR : Peize Lin
// DATE : 2015-03-10
//==========================================================
#ifndef EXX_LIP_H
#define EXX_LIP_H

#include "module_base/complexmatrix.h"
#include "module_base/vector3.h"
#include "module_hamilt_general/module_xc/exx_info.h"
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_elecstate/elecstate.h"
#include "module_cell/module_symmetry/symmetry.h"
#include "module_hamilt_pw/hamilt_pwdft/wfinit.h"
class K_Vectors;
class UnitCell;

// template<typename T>
// struct RemoveComplex;
// template<>
// struct RemoveComplex<std::complex<double>> {
//     using type = double;
// };
// template<>
// struct RemoveComplex<std::complex<float>> {
//     using type = float;
// };

template<typename T, typename Device = base_device::DEVICE_CPU>
class Exx_Lip
{
    using Real = typename GetTypeReal<T>::type;
public:
    // Exx_Lip(const Exx_Info::Exx_Info_Lip& info_in);
    // use default constructor to let the instance to be a member of ESolver
    Exx_Lip() {};
    ~Exx_Lip();

    Exx_Info::Exx_Info_Lip info;

    void init(const Exx_Info::Exx_Info_Lip& info_in,
        const ModuleSymmetry::Symmetry& symm,
        K_Vectors* kv_ptr_in,
        //   wavefunc* wf_ptr_in,
        psi::WFInit<T, Device>* wf_ptr_in,
        psi::Psi<T, Device>* kspw_psi_ptr_in,
        const ModulePW::PW_Basis_K* wfc_basis_in,
        const ModulePW::PW_Basis* rho_basis_in,
        const Structure_Factor& sf,
        const UnitCell* ucell_ptr_in,
        const elecstate::ElecState* pelec_in);
    // void cal_exx(const int& nks);
    void cal_exx();
    const T* const* const* get_exx_matrix() const
    {
        return exx_matrix;
    }
    Real get_exx_energy() const
    {
        return exx_energy;
    }

    void write_q_pack() const;

    void set_hvec(const int ik, T* hvec)
    {
        memcpy(&(*this->k_pack->hvec_array)(ik, 0, 0), hvec, sizeof(T) * GlobalV::NLOCAL * GlobalV::NBANDS);
    }

private:

    int gzero_rank_in_pool;

    // template<typename T, typename Device = base_device::DEVICE_CPU>
    struct k_package
    {
        K_Vectors* kv_ptr;
        // wavefunc* wf_ptr;
        psi::Psi<T, Device>* kspw_psi_ptr;
        psi::WFInit<T, Device>* wf_ptr;
        ModuleBase::matrix wf_wg;

        /// @brief LCAO wavefunction, the eigenvectors from lapack diagonalization
        psi::Psi<T, Device>* hvec_array;
        const elecstate::ElecState* pelec;
    } *k_pack, * q_pack;

    int iq_vecik;

    T** phi;
    T*** psi;
    Real* recip_qkg2;
    Real sum2_factor;
    T* b;
    T* b0;
    T* sum1;
    T** sum3;

    T*** exx_matrix = nullptr;
    Real exx_energy = 0.0;

	void wf_wg_cal();
    void phi_cal(k_package* kq_pack, int ikq);
    void psi_cal();
    void judge_singularity( int ik);
	void qkg2_exp(int ik, int iq);
	void b_cal(int ik, int iq, int ib);
	void sum3_cal(int iq, int ib);
	void b_sum(int iq, int ib);
	void sum_all(int ik);
	void exx_energy_cal();
    // void read_q_pack(const ModuleSymmetry::Symmetry& symm,
    //                  const ModulePW::PW_Basis_K* wfc_basis,
    //                  const Structure_Factor& sf);

    //2*pi*i
    T two_pi_i = Real(ModuleBase::TWO_PI) * T(0.0, 1.0);
public:
    const ModulePW::PW_Basis* rho_basis;
    const ModulePW::PW_Basis_K* wfc_basis;

    const UnitCell* ucell_ptr;
};


#endif
