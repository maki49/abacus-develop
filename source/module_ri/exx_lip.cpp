//==========================================================
// AUTHOR : Peize Lin
// DATE : 2015-03-10
//==========================================================

#include "exx_lip.h"
#include "module_base/global_function.h"
#include "module_base/vector3.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_cell/klist.h"
#include "module_hamilt_pw/hamilt_pwdft/wavefunc.h"
#include "module_hamilt_lcao/hamilt_lcaodft/wavefunc_in_pw.h"
#include "module_base/lapack_connector.h"
#include <limits>
#include "module_base/parallel_global.h"

template <typename T, typename Device>
void Exx_Lip<T, Device>::cal_exx()
{
    ModuleBase::TITLE("Exx_Lip", "cal_exx");

    wf_wg_cal();
    //cout_t("wf_wg_cal",my_time(t));
    psi_cal();
    //cout_t("psi_cal",my_time(t));
    for (int ik = 0; ik < k_pack->kv_ptr->get_nks(); ++ik)
    {
        phi_cal(k_pack, ik);
        //t_phi_cal += my_time(t);

        judge_singularity(ik);
        for (int iw_l = 0; iw_l < GlobalV::NLOCAL; ++iw_l)
            for (int iw_r = 0; iw_r < GlobalV::NLOCAL; ++iw_r)
                sum1[iw_l * GlobalV::NLOCAL + iw_r] = T(0.0, 0.0);
        if (Conv_Coulomb_Pot_K::Ccp_Type::Ccp == info.ccp_type || Conv_Coulomb_Pot_K::Ccp_Type::Hf == info.ccp_type)
        {
            sum2_factor = 0.0;
            if (gzero_rank_in_pool == GlobalV::RANK_IN_POOL)
                for (int iw_l = 0; iw_l < GlobalV::NLOCAL; ++iw_l)
                    for (int iw_r = 0; iw_r < GlobalV::NLOCAL; ++iw_r)
                        sum3[iw_l][iw_r] = T(0.0, 0.0);
        }

        for (int iq_tmp = iq_vecik; iq_tmp < iq_vecik + q_pack->kv_ptr->get_nks() / GlobalV::NSPIN; ++iq_tmp)					// !!! k_point
            // parallel incompleted.need to loop iq in other pool
        {
            int iq = (ik < (k_pack->kv_ptr->get_nks() / GlobalV::NSPIN)) ? (iq_tmp % (q_pack->kv_ptr->get_nks() / GlobalV::NSPIN)) :
                (iq_tmp % (q_pack->kv_ptr->get_nks() / GlobalV::NSPIN) + (q_pack->kv_ptr->get_nks() / GlobalV::NSPIN)); 			qkg2_exp(ik, iq);
            //t_qkg2_exp += my_time(t);
            for (int ib = 0; ib < GlobalV::NBANDS; ++ib)
            {
                b_cal(ik, iq, ib);
                //t_b_cal += my_time(t);
                if (Conv_Coulomb_Pot_K::Ccp_Type::Ccp == info.ccp_type || Conv_Coulomb_Pot_K::Ccp_Type::Hf == info.ccp_type)
                    if (iq == iq_vecik)
                        sum3_cal(iq, ib);
                //t_sum3_cal += my_time(t);
                b_sum(iq, ib);
                //t_b_sum += my_time(t);
            }
        }
        sum_all(ik);
        //t_sum_all += my_time(t);
    }
    exx_energy_cal();

    auto print_Hexxk = [&]()
        {
            static int istep = 1;
            for (int ik = 0; ik != q_pack->kv_ptr->get_nks(); ++ik)
            {
                std::ofstream
                    ofs("Hexxk_" + ModuleBase::GlobalFunc::TO_STRING(istep++) + "_" + ModuleBase::GlobalFunc::TO_STRING(ik) + "_" + ModuleBase::GlobalFunc::TO_STRING(GlobalV::MY_RANK));
                for (int i = 0; i != GlobalV::NLOCAL; ++i)
                {
                    for (int j = 0; j != GlobalV::NLOCAL; ++j)
                        ofs << exx_matrix[ik][i][j] << "\t";
                    ofs << std::endl;
                }
            };
        };
}

/*
void Exx_Lip::cal_exx()
{
	ModuleBase::TITLE("Exx_Lip","cal_exx");
	wf_wg_cal();
	psi_cal();
	for( int ik=0; ik<k_pack->kv_ptr->get_nks(); ++ik)
	{
		phi_cal(k_pack, ik);

		judge_singularity(ik);
		for( int iw_l=0; iw_l<GlobalV::NLOCAL; ++iw_l)
			for( int iw_r=0; iw_r<GlobalV::NLOCAL; ++iw_r)
				sum1[iw_l*GlobalV::NLOCAL+iw_r] = std::complex<double>(0.0,0.0);
		if( Exx_Info::Hybrid_Type::HF==info.hybrid_type || Exx_Info::Hybrid_Type::PBE0==info.hybrid_type || Exx_Info::Hybrid_Type::SCAN0==info.hybrid_type )
		{
			sum2_factor = 0.0;
			if(gzero_rank_in_pool==GlobalV::RANK_IN_POOL)
				for( int iw_l=0; iw_l<GlobalV::NLOCAL; ++iw_l)
					for( int iw_r=0; iw_r<GlobalV::NLOCAL; ++iw_r)
						sum3[iw_l][iw_r] = std::complex<double>(0.0,0.0);
		}

		for( int iq_tmp=iq_vecik; iq_tmp<iq_vecik+q_pack->kv_ptr->get_nks()/GlobalV::NSPIN; ++iq_tmp)					// !!! k_point parallel incompleted. need to loop iq in other pool
		{
			int iq = (ik<(k_pack->kv_ptr->get_nks()/GlobalV::NSPIN)) ? (iq_tmp%(q_pack->kv_ptr->get_nks()/GlobalV::NSPIN)) : (iq_tmp%(q_pack->kv_ptr->get_nks()/GlobalV::NSPIN)+(q_pack->kv_ptr->get_nks()/GlobalV::NSPIN));
			qkg2_exp(ik, iq);
			for( int ib=0; ib<GlobalV::NBANDS; ++ib)
			{
				b_cal(ik, iq, ib);
				if( Exx_Info::Hybrid_Type::HF==info.hybrid_type || Exx_Info::Hybrid_Type::PBE0==info.hybrid_type || Exx_Info::Hybrid_Type::SCAN0==info.hybrid_type )
					if(iq==iq_vecik)
						sum3_cal(iq,ib);
					b_sum(iq, ib);
			}
		}
		sum_all(ik);
	}
	exx_energy_cal();
}
*/
template <typename T, typename Device>
void Exx_Lip<T, Device>::init(
    const Exx_Info::Exx_Info_Lip& info_in,
    const ModuleSymmetry::Symmetry& symm,
    K_Vectors* kv_ptr_in,
    psi::WFInit<T, Device>* wf_ptr_in,
    psi::Psi<T, Device>* kspw_psi_ptr_in,
    //    wavefunc* wf_ptr_in,
    const ModulePW::PW_Basis_K* wfc_basis_in,
    const ModulePW::PW_Basis* rho_basis_in,
    const Structure_Factor& sf,
    const UnitCell* ucell_ptr_in,
    const elecstate::ElecState* pelec_in)
{
    ModuleBase::TITLE("Exx_Lip", "init");
    this->info = info_in;
    try
	{
		k_pack = new k_package;
		k_pack->kv_ptr = kv_ptr_in;
		k_pack->wf_ptr = wf_ptr_in;
        k_pack->pelec = pelec_in;
        k_pack->kspw_psi_ptr = kspw_psi_ptr_in;
        wfc_basis = wfc_basis_in;
        rho_basis = rho_basis_in;
        ucell_ptr = ucell_ptr_in;

        int gzero_judge(-1);
        if (rho_basis->gg_uniq[0] < 1e-8)
        {
            gzero_judge = GlobalV::RANK_IN_POOL;
        }
#ifdef __MPI
        MPI_Allreduce(&gzero_judge, &gzero_rank_in_pool, 1, MPI_INT, MPI_MAX, POOL_WORLD);
	#endif
		k_pack->wf_wg.create(k_pack->kv_ptr->get_nks(),GlobalV::NBANDS);

        k_pack->hvec_array = new psi::Psi<T, Device>(k_pack->kv_ptr->get_nks(), GlobalV::NBANDS, GlobalV::NLOCAL);
        // k_pack->hvec_array = new ModuleBase::ComplexMatrix[k_pack->kv_ptr->get_nks()];
        // for( int ik=0; ik<k_pack->kv_ptr->get_nks(); ++ik)
        // {
        // 	k_pack->hvec_array[ik].create(GlobalV::NLOCAL,GlobalV::NBANDS);
        // }

        // if (GlobalV::init_chg=="atomic")
		{
			q_pack = k_pack;
		}
        // else if(GlobalV::init_chg=="file")
        // {
        //     read_q_pack(symm, wfc_basis, sf);
        // }

        phi = new T * [GlobalV::NLOCAL];
        for (int iw = 0; iw < GlobalV::NLOCAL; ++iw)
        {
            phi[iw] = new T[rho_basis->nrxx];
        }

        psi = new T * *[q_pack->kv_ptr->get_nks()];
        for (int iq = 0; iq < q_pack->kv_ptr->get_nks(); ++iq)
        {
            psi[iq] = new T * [GlobalV::NBANDS];
			for( int ib=0; ib<GlobalV::NBANDS; ++ib)
			{
                psi[iq][ib] = new T[rho_basis->nrxx];
			}
        }

        recip_qkg2 = new Real[rho_basis->npw];

        b = new T[GlobalV::NLOCAL * rho_basis->npw];

        sum1 = new T[GlobalV::NLOCAL * GlobalV::NLOCAL];

        if (Conv_Coulomb_Pot_K::Ccp_Type::Ccp == info.ccp_type || Conv_Coulomb_Pot_K::Ccp_Type::Hf == info.ccp_type)
			if(gzero_rank_in_pool==GlobalV::RANK_IN_POOL)
			{
                b0 = new T[GlobalV::NLOCAL];
                sum3 = new T * [GlobalV::NLOCAL];
				for( int iw_l=0; iw_l<GlobalV::NLOCAL; ++iw_l)
				{
                    sum3[iw_l] = new T[GlobalV::NLOCAL];
				}
			}
			else
			{
				b0 = NULL;
				sum3 = NULL;
			}
		else
		{
			b0 = NULL;
			sum3 = NULL;
		}

        exx_matrix = new T * *[k_pack->kv_ptr->get_nks()];
		for( int ik=0; ik<k_pack->kv_ptr->get_nks(); ++ik)
		{
            exx_matrix[ik] = new T * [GlobalV::NLOCAL];
			for( int iw_l=0; iw_l<GlobalV::NLOCAL; ++iw_l)
			{
                exx_matrix[ik][iw_l] = new T[GlobalV::NLOCAL];
			}
		}
	}
	catch(const std::bad_alloc &ex)
	{
		ModuleBase::WARNING_QUIT("exact_exchange","Memory");
    }
}

template <typename T, typename Device>
Exx_Lip<T, Device>::~Exx_Lip()
{
    for (int iw = 0; iw < GlobalV::NLOCAL; ++iw)
    {
        delete[] phi[iw];	phi[iw] = NULL;
    }
    delete[] phi;		phi = NULL;

    for (int iq = 0;iq < q_pack->kv_ptr->get_nks();++iq)
    {
        for (int ib = 0;ib < GlobalV::NBANDS;++ib)
        {
            delete[] psi[iq][ib];	psi[iq][ib] = NULL;
        }
        delete[] psi[iq];	psi[iq] = NULL;
    }
    delete[] psi;		psi = NULL;

    delete[] recip_qkg2;	recip_qkg2 = NULL;

    delete[] b;		b = NULL;

    delete[] sum1;		sum1 = NULL;

    if (Conv_Coulomb_Pot_K::Ccp_Type::Ccp == info.ccp_type || Conv_Coulomb_Pot_K::Ccp_Type::Hf == info.ccp_type)
        if (gzero_rank_in_pool == GlobalV::RANK_IN_POOL)
        {
            delete[] b0;	b0 = NULL;
            for (int iw_l = 0; iw_l < GlobalV::NLOCAL; ++iw_l)
            {
                delete[] sum3[iw_l];	sum3[iw_l] = NULL;
            }
            delete[] sum3;		sum3 = NULL;
        }

    for (int ik = 0; ik < k_pack->kv_ptr->get_nks(); ++ik)
    {
        for (int iw_l = 0; iw_l < GlobalV::NLOCAL; ++iw_l)
        {
            delete[] exx_matrix[ik][iw_l];	exx_matrix[ik][iw_l] = NULL;
        }
        delete[] exx_matrix[ik];	exx_matrix[ik] = NULL;
    }
    delete[] exx_matrix;	exx_matrix = NULL;

    // delete[] k_pack->hvec_array;	k_pack->hvec_array=NULL;
    delete k_pack->hvec_array;
    delete k_pack;

    if (GlobalV::init_chg == "atomic")
    {
        q_pack = NULL;
    }
    else if (GlobalV::init_chg == "file")
    {
        delete q_pack->kv_ptr;	q_pack->kv_ptr = NULL;
        delete q_pack->wf_ptr;	q_pack->wf_ptr = NULL;
        // delete[] q_pack->hvec_array;	q_pack->hvec_array=NULL;
        delete q_pack;	q_pack = NULL;
    }
}

template <typename T, typename Device>
void Exx_Lip<T, Device>::wf_wg_cal()
{
	ModuleBase::TITLE("Exx_Lip","wf_wg_cal");
	if(GlobalV::NSPIN==1)
		for( int ik=0; ik<k_pack->kv_ptr->get_nks(); ++ik)
			for( int ib=0; ib<GlobalV::NBANDS; ++ib)
				k_pack->wf_wg(ik,ib) = k_pack->pelec->wg(ik,ib)/2;
	else if(GlobalV::NSPIN==2)
		for( int ik=0; ik<k_pack->kv_ptr->get_nks(); ++ik)
			for( int ib=0; ib<GlobalV::NBANDS; ++ib)
				k_pack->wf_wg(ik,ib) = k_pack->pelec->wg(ik,ib);
}

template <typename T, typename Device>
void Exx_Lip<T, Device>::phi_cal(k_package* kq_pack, int ikq)
{
    T* porter = new T[wfc_basis->nrxx];
	for( int iw=0; iw< GlobalV::NLOCAL; ++iw)
	{
        // wfc_basis->recip2real(&kq_pack->wf_ptr->wanf2[ikq](iw,0), porter, ikq);
        wfc_basis->recip2real(&(kq_pack->wf_ptr->get_psig().lock()->operator()(ikq, iw, 0)), porter, ikq);
        int ir(0);
		for( int ix=0; ix<rho_basis->nx; ++ix)
		{
            const Real phase_x = kq_pack->kv_ptr->kvec_d[ikq].x * ix / rho_basis->nx;
			for( int iy=0; iy<rho_basis->ny; ++iy)
			{
                const Real phase_xy = phase_x + kq_pack->kv_ptr->kvec_d[ikq].y * iy / rho_basis->ny;
				for( int iz=rho_basis->startz_current; iz<rho_basis->startz_current+rho_basis->nplane; ++iz)
				{
                    const Real phase_xyz = phase_xy + kq_pack->kv_ptr->kvec_d[ikq].z * iz / rho_basis->nz;
                    const T exp_tmp = exp(phase_xyz * two_pi_i);
                    phi[iw][ir] = porter[ir] * exp_tmp;
					++ir;
				}
			}
		}
	}
	delete [] porter;
}
template <typename T, typename Device>
void Exx_Lip<T, Device>::psi_cal()
{
    ModuleBase::TITLE("Exx_Lip", "psi_cal");
    if (GlobalV::init_chg == "atomic")
    {
        T* porter = new T[wfc_basis->nrxx];
        for (int iq = 0; iq < q_pack->kv_ptr->get_nks(); ++iq)
        {
            for (int ib = 0; ib < GlobalV::NBANDS; ++ib)
            {
                wfc_basis->recip2real(&(q_pack->kspw_psi_ptr->operator()(iq, ib, 0)), porter, iq);

                int ir(0);
                for (int ix = 0; ix < rho_basis->nx; ++ix)
                {
                    const Real phase_x = q_pack->kv_ptr->kvec_d[iq].x * ix / rho_basis->nx;
                    for (int iy = 0; iy < rho_basis->ny; ++iy)
                    {
                        const Real phase_xy = phase_x + q_pack->kv_ptr->kvec_d[iq].y * iy / rho_basis->ny;
                        for (int iz = rho_basis->startz_current; iz < rho_basis->startz_current + rho_basis->nplane; ++iz)
                        {
                            const Real phase_xyz = phase_xy + q_pack->kv_ptr->kvec_d[iq].z * iz / rho_basis->nz;
                            const T exp_tmp = exp(phase_xyz * two_pi_i);
                            psi[iq][ib][ir] = porter[ir] * exp_tmp;
                            ++ir;
                        }
                    }
                }
            }
        }
        delete[] porter;
    }
    else if (GlobalV::init_chg == "file")
    {
        for (int iq = 0; iq < q_pack->kv_ptr->get_nks(); ++iq)
        {
            phi_cal(q_pack, iq);
            for (int ib = 0; ib < GlobalV::NBANDS; ++ib)
            {
                ModuleBase::GlobalFunc::ZEROS(psi[iq][ib], rho_basis->nrxx);
                for (int iw = 0; iw < GlobalV::NLOCAL; ++iw)
                {
                    for (int ir = 0; ir < rho_basis->nrxx; ++ir)
                    {
                        psi[iq][ib][ir] += (*q_pack->hvec_array)(iq, ib, iw) * phi[iw][ir];
                    }
                }
            }
        }
    }
    ///////////////////////////////////////////////////////////////////////////
    //			!!! k_point parallel incompleted. need to loop iq in other pool
    ///////////////////////////////////////////////////////////////////////////
}

template <typename T, typename Device>
void Exx_Lip<T, Device>::judge_singularity(int ik)
{
	if (GlobalV::init_chg=="atomic")
	{
		iq_vecik = ik;
	}
	else if(GlobalV::init_chg=="file")
	{
        Real min_q_minus_k(std::numeric_limits<Real>::max());
		for( int iq=0; iq<q_pack->kv_ptr->get_nks(); ++iq)
		{
            const Real q_minus_k((q_pack->kv_ptr->kvec_c[iq] - k_pack->kv_ptr->kvec_c[ik]).norm2());
			if(q_minus_k < min_q_minus_k)
			{
				min_q_minus_k = q_minus_k;
				iq_vecik = iq;
			}
		}
	}
}

template <typename T, typename Device>
void Exx_Lip<T, Device>::qkg2_exp(int ik, int iq)
{
	for( int ig=0; ig<rho_basis->npw; ++ig)
	{
        const Real qkg2 = ((q_pack->kv_ptr->kvec_c[iq] - k_pack->kv_ptr->kvec_c[ik] + rho_basis->gcar[ig]) * (ModuleBase::TWO_PI / ucell_ptr->lat0)).norm2();
        if (Conv_Coulomb_Pot_K::Ccp_Type::Ccp == info.ccp_type || Conv_Coulomb_Pot_K::Ccp_Type::Hf == info.ccp_type)
		{
			if( std::abs(qkg2)<1e-10 )
				recip_qkg2[ig] = 0.0;												// 0 to ignore bb/qkg2 when qkg2==0
			else
				recip_qkg2[ig] = 1.0/qkg2;
            sum2_factor += recip_qkg2[ig] * exp(-info.lambda * qkg2);
			recip_qkg2[ig] = sqrt(recip_qkg2[ig]);
		}
        else if (Conv_Coulomb_Pot_K::Ccp_Type::Hse == info.ccp_type)
		{
			if( std::abs(qkg2)<1e-10 )
                recip_qkg2[ig] = 1.0 / (2 * info.hse_omega);
			else
                recip_qkg2[ig] = sqrt((1 - exp(-qkg2 / (4 * info.hse_omega * info.hse_omega))) / qkg2);
		}
	}
}

template <typename T, typename Device>
void Exx_Lip<T, Device>::b_cal(int ik, int iq, int ib)
{
    const ModuleBase::Vector3<double> q_minus_k = q_pack->kv_ptr->kvec_d[iq] - k_pack->kv_ptr->kvec_d[ik];
    std::vector<T > mul_tmp(rho_basis->nrxx);
	for( size_t ir=0,ix=0; ix<rho_basis->nx; ++ix)
	{
        const Real phase_x = q_minus_k.x * ix / rho_basis->nx;
		for( size_t iy=0; iy<rho_basis->ny; ++iy)
		{
            const Real phase_xy = phase_x + q_minus_k.y * iy / rho_basis->ny;
			for( size_t iz=rho_basis->startz_current; iz<rho_basis->startz_current+rho_basis->nplane; ++iz)
			{
                const Real phase_xyz = phase_xy + q_minus_k.z * iz / rho_basis->nz;
                mul_tmp[ir] = exp(-phase_xyz * two_pi_i);
				mul_tmp[ir] *= psi[iq][ib][ir];
				++ir;
			}
		}
	}

    T* const porter = new T[rho_basis->nrxx];

	for(size_t iw=0; iw< GlobalV::NLOCAL; ++iw)
	{
        const T* const phi_w = phi[iw];
		for( size_t ir=0; ir<rho_basis->nrxx; ++ir)
		{
			porter[ir] = conj(phi_w[ir]) * mul_tmp[ir] ;
//			porter[ir] = phi_w[ir] * psi_q_b[ir] *exp_tmp[ir] ;
		}
        T* const b_w = b + iw * rho_basis->npw;
		rho_basis->real2recip( porter, b_w);
        if (Conv_Coulomb_Pot_K::Ccp_Type::Ccp == info.ccp_type || Conv_Coulomb_Pot_K::Ccp_Type::Hf == info.ccp_type)
			if((iq==iq_vecik) && (gzero_rank_in_pool==GlobalV::RANK_IN_POOL))							/// need to check while use k_point parallel
				b0[iw] = b_w[rho_basis->ig_gge0];

		for( size_t ig=0; ig<rho_basis->npw; ++ig)
			b_w[ig] *= recip_qkg2[ig];
	}
	delete [] porter;
}

template <typename T, typename Device>
void  Exx_Lip<T, Device>::sum3_cal(int iq, int ib)
{
	if( gzero_rank_in_pool == GlobalV::RANK_IN_POOL )
		for( int iw_l=0; iw_l<GlobalV::NLOCAL; ++iw_l)
			for( int iw_r=0; iw_r<GlobalV::NLOCAL; ++iw_r)
                sum3[iw_l][iw_r] += b0[iw_l] * conj(b0[iw_r]) * (Real)q_pack->wf_wg(iq, ib);
}

template <typename T, typename Device>
void Exx_Lip<T, Device>::b_sum(int iq, int ib)			// Peize Lin change 2019-04-14
{
	// sum1[iw_l,iw_r] += \sum_{ig} b[iw_l,ig] * conj(b[iw_r,ig]) * q_pack->wf_wg(iq,ib)
    LapackConnector::herk(
		'U','N',
		GlobalV::NLOCAL, rho_basis->npw,
        (Real)q_pack->wf_wg(iq, ib), b, rho_basis->npw,
		1.0, sum1, GlobalV::NLOCAL);
//	cblas_zherk( CblasRowMajor, CblasUpper, CblasNoTrans,
//				GlobalV::NLOCAL, rho_basis->npw,
//				q_pack->wf_wg(iq,ib), static_cast<void*>(b), rho_basis->npw,
//				1.0, static_cast<void*>(sum1), GlobalV::NLOCAL);
}

template <typename T, typename Device>
void Exx_Lip<T, Device>::sum_all(int ik)
{
    Real sum2_factor_g(0.0);
    Real fourpi_div_omega = 4 * (Real)(ModuleBase::PI / ucell_ptr->omega);
    Real spin_fac = 2.0;
#ifdef __MPI
    if (Conv_Coulomb_Pot_K::Ccp_Type::Ccp == info.ccp_type || Conv_Coulomb_Pot_K::Ccp_Type::Hf == info.ccp_type)
		MPI_Reduce( &sum2_factor, &sum2_factor_g, 1, MPI_DOUBLE, MPI_SUM, gzero_rank_in_pool, POOL_WORLD);
	#endif
	for( size_t iw_l=1; iw_l<GlobalV::NLOCAL; ++iw_l)
		for( size_t iw_r=0; iw_r<iw_l; ++iw_r)
			sum1[iw_l*GlobalV::NLOCAL+iw_r] = conj(sum1[iw_r*GlobalV::NLOCAL+iw_l]);		// Peize Lin add conj 2019-04-14

	for( int iw_l=0; iw_l<GlobalV::NLOCAL; ++iw_l)
	{
		for( int iw_r=0; iw_r<GlobalV::NLOCAL; ++iw_r)
		{
            exx_matrix[ik][iw_l][iw_r] = -fourpi_div_omega * sum1[iw_l * GlobalV::NLOCAL + iw_r] * spin_fac;
            if (Conv_Coulomb_Pot_K::Ccp_Type::Ccp == info.ccp_type || Conv_Coulomb_Pot_K::Ccp_Type::Hf == info.ccp_type)
				if(gzero_rank_in_pool==GlobalV::RANK_IN_POOL)
				{
                    exx_matrix[ik][iw_l][iw_r] += spin_fac * (fourpi_div_omega * sum3[iw_l][iw_r] * sum2_factor_g);
                    exx_matrix[ik][iw_l][iw_r] += spin_fac * (-1 / (Real)sqrt(info.lambda * ModuleBase::PI) * (Real)(q_pack->kv_ptr->get_nks() / GlobalV::NSPIN) * sum3[iw_l][iw_r]);
				}
		}
	}
}

template <typename T, typename Device>
void Exx_Lip<T, Device>::exx_energy_cal()
{
	ModuleBase::TITLE("Exx_Lip","exx_energy_cal");

    Real exx_energy_tmp = 0.0;

	for( int ik=0; ik<k_pack->kv_ptr->get_nks(); ++ik)
	{
		for( int iw_l=0; iw_l<GlobalV::NLOCAL; ++iw_l)
		{
			for( int iw_r=0; iw_r<GlobalV::NLOCAL; ++iw_r)
			{
				for( int ib=0; ib<GlobalV::NBANDS; ++ib)
				{
                    exx_energy_tmp += (exx_matrix[ik][iw_l][iw_r] * conj((*k_pack->hvec_array)(ik, ib, iw_l)) * (*k_pack->hvec_array)(ik, ib, iw_r)).real() * k_pack->wf_wg(ik, ib);
				}
			}
		}
	}
#ifdef __MPI
	MPI_Allreduce( &exx_energy_tmp, &exx_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);				// !!! k_point parallel incompleted. different pools have different kv.set_nks(>) deadlock
#endif
	exx_energy *= (GlobalV::NSPIN==1) ? 2 : 1;
	exx_energy /= 2;										// ETOT = E_band - 1/2 E_exx

	#if TEST_EXX==1
	{
		std::ofstream ofs("exx_matrix.dat",std::ofstream::app);
		static int istep=0;
		ofs<<"istep:\t"<<istep++<<std::endl;
		for( int ik=0; ik<k_pack->kv_ptr->get_nks(); ++ik)
		{
			ofs<<"ik:\t"<<ik<<std::endl;
			for( int iw_l=0; iw_l<GlobalV::NLOCAL; ++iw_l)
			{
				for( int iw_r=0; iw_r<GlobalV::NLOCAL; ++iw_r)
				{
					ofs<<exx_matrix[ik][iw_l][iw_r]<<"\t";
				}
				ofs<<std::endl;
			}
			ofs<<std::endl;
		}
		ofs.close();
	}
	{
		std::ofstream ofs("DM.dat",std::ofstream::app);
		static int istep=0;
		ofs<<"istep:\t"<<istep++<<std::endl;
		for( int ik=0; ik<k_pack->kv_ptr->get_nks(); ++ik)
		{
			ofs<<"ik:\t"<<ik<<std::endl;
			for( int iw_l=0; iw_l<GlobalV::NLOCAL; ++iw_l)
			{
				for( int iw_r=0; iw_r<GlobalV::NLOCAL; ++iw_r)
				{
                    T DM = { 0,0 };
					for( int ib=0; ib<GlobalV::NBANDS; ++ib )
                        DM += conj((*k_pack->hvec_array)(ik, ib, iw_l)) * (*k_pack->hvec_array)(ik, ib, iw_r) * k_pack->wf_wg(ik, ib);
					ofs<<DM<<"\t";
				}
				ofs<<std::endl;
			}
			ofs<<std::endl;
		}
		ofs.close();
	}
	#elif TEST_EXX==-1
		#error
	#endif

	return;
}

template <typename T, typename Device>
void Exx_Lip<T, Device>::write_q_pack() const
{
    if (GlobalV::out_chg==0)
		return;

	if(!GlobalV::RANK_IN_POOL)
	{
		const std::string exx_q_pack = "exx_q_pack/";
		int return_value=0;
		const std::string command_mkdir = "test -d " + GlobalV::global_out_dir + exx_q_pack + " || mkdir " + GlobalV::global_out_dir + exx_q_pack;
        return_value = system(command_mkdir.c_str());
        assert(return_value == 0);

        const std::string command_kpoint = "test -f " + GlobalV::global_out_dir + exx_q_pack + GlobalV::global_kpoint_card + " || cp " + GlobalV::global_kpoint_card + " " + GlobalV::global_out_dir + exx_q_pack + GlobalV::global_kpoint_card;
        return_value = system(command_kpoint.c_str());
		assert(return_value==0);

		std::stringstream ss_wf_wg;
		ss_wf_wg << GlobalV::global_out_dir << exx_q_pack << "wf_wg_" << GlobalV::MY_POOL;
		std::ofstream ofs_wf_wg(ss_wf_wg.str().c_str());
		for( int iq = 0; iq < q_pack->kv_ptr->get_nks(); ++iq)
		{
			for( int ib=0; ib<GlobalV::NBANDS; ++ib)
			{
				ofs_wf_wg<<q_pack->wf_wg(iq,ib)<<"\t";
			}
			ofs_wf_wg<<std::endl;
		}
		ofs_wf_wg.close();

		std::stringstream ss_hvec;
		ss_hvec	<< GlobalV::global_out_dir << exx_q_pack << "hvec_" << GlobalV::MY_POOL;
		std::ofstream ofs_hvec(ss_hvec.str().c_str());
		for( int iq=0; iq<q_pack->kv_ptr->get_nks(); ++iq)
		{
			for( int iw=0; iw<GlobalV::NLOCAL; ++iw)
			{
				for( int ib=0; ib<GlobalV::NBANDS; ++ib)
				{
                    ofs_hvec << (*q_pack->hvec_array)(iq, ib, iw).real() << " " << (*q_pack->hvec_array)(iq, ib, iw).imag() << " ";
				}
				ofs_hvec<<std::endl;
			}
		}
		ofs_hvec.close();
	}
	return;
}

// void Exx_Lip::read_q_pack(const ModuleSymmetry::Symmetry& symm,
//                           const ModulePW::PW_Basis_K* wfc_basis,
//                           const Structure_Factor& sf)
// {
// 	const std::string exx_q_pack = "exx_q_pack/";

// 	q_pack = new k_package();

// 	q_pack->kv_ptr = new K_Vectors();
// 	const std::string exx_kpoint_card = GlobalV::global_out_dir + exx_q_pack + GlobalV::global_kpoint_card;
// 	q_pack->kv_ptr->set( symm, exx_kpoint_card, GlobalV::NSPIN, ucell_ptr->G, ucell_ptr->latvec, GlobalV::ofs_running );

// 	q_pack->wf_ptr = new wavefunc();
//     q_pack->wf_ptr->allocate(q_pack->kv_ptr->get_nkstot(),
//                              q_pack->kv_ptr->get_nks(),
//                              q_pack->kv_ptr->ngk.data(),
//                              wfc_basis->npwk_max); // mohan update 2021-02-25
//     //	q_pack->wf_ptr->init(q_pack->kv_ptr->get_nks(),q_pack->kv_ptr,ucell_ptr,old_pwptr,&ppcell,&GlobalC::ORB,&hm,&Pkpoints);
//     q_pack->wf_ptr->table_local.create(GlobalC::ucell.ntype, GlobalC::ucell.nmax_total, GlobalV::NQX);
// //	q_pack->wf_ptr->table_local.create(q_pack->wf_ptr->ucell_ptr->ntype, q_pack->wf_ptr->ucell_ptr->nmax_total, GlobalV::NQX);
// #ifdef __LCAO
// 	Wavefunc_in_pw::make_table_q(GlobalC::ORB.orbital_file, q_pack->wf_ptr->table_local);
// //	Wavefunc_in_pw::make_table_q(q_pack->wf_ptr->ORB_ptr->orbital_file, q_pack->wf_ptr->table_local, q_pack->wf_ptr);
// 	for(int iq=0; iq<q_pack->kv_ptr->get_nks(); ++iq)
// 	{
//         Wavefunc_in_pw::produce_local_basis_in_pw(iq,
//                                                   wfc_basis,
//                                                   sf,
//                                                   q_pack->wf_ptr->wanf2[iq],
//                                                   q_pack->wf_ptr->table_local);
//         //		Wavefunc_in_pw::produce_local_basis_in_pw(iq, q_pack->wf_ptr->wanf2[iq], q_pack->wf_ptr->table_local,
//         // q_pack->wf_ptr);
//     }
// #endif
// 	q_pack->wf_wg.create(q_pack->kv_ptr->get_nks(),GlobalV::NBANDS);
// 	if(!GlobalV::RANK_IN_POOL)
// 	{
// 		std::stringstream ss_wf_wg;
// 		ss_wf_wg << GlobalV::global_out_dir << exx_q_pack << "wf_wg_" << GlobalV::MY_POOL;
// 		std::ifstream ifs_wf_wg(ss_wf_wg.str().c_str());
// 		for( int iq = 0; iq < q_pack->kv_ptr->get_nks(); ++iq)
// 		{
// 			for( int ib=0; ib<GlobalV::NBANDS; ++ib)
// 			{
// 				ifs_wf_wg>>q_pack->wf_wg(iq,ib);
// 			}
// 		}
// 		ifs_wf_wg.close();
// 	}
// 	#ifdef __MPI
// 	MPI_Bcast( q_pack->wf_wg.c, q_pack->kv_ptr->get_nks()*GlobalV::NBANDS, MPI_DOUBLE, 0, POOL_WORLD);
// 	#endif

// 	q_pack->hvec_array = new ModuleBase::ComplexMatrix [q_pack->kv_ptr->get_nks()];
// 	for( int iq=0; iq<q_pack->kv_ptr->get_nks(); ++iq)
// 	{
// 		q_pack->hvec_array[iq].create(GlobalV::NLOCAL,GlobalV::NBANDS);
// 	}
// 	if(!GlobalV::RANK_IN_POOL)
// 	{
// 		std::stringstream ss_hvec;
// 		ss_hvec	<< GlobalV::global_out_dir << exx_q_pack << "hvec_" << GlobalV::MY_POOL;
// 		std::ifstream ifs_hvec(ss_hvec.str().c_str());
// 		for( int iq=0; iq<q_pack->kv_ptr->get_nks(); ++iq)
// 		{
// 			for( int iw=0; iw<GlobalV::NLOCAL; ++iw)
// 			{
// 				for( int ib=0; ib<GlobalV::NBANDS; ++ib)
// 				{
// 					double a,b;
// 					ifs_hvec>>a>>b;
// 					q_pack->hvec_array[iq](iw,ib) = {a,b};
// 				}
// 			}
// 		}
// 		ifs_hvec.close();
// 	}
// 	#ifdef __MPI
// 	for( int iq=0; iq<q_pack->kv_ptr->get_nks(); ++iq)
// 	{
// 		MPI_Bcast( q_pack->hvec_array[iq].c, GlobalV::NLOCAL*GlobalV::NBANDS, MPI_DOUBLE_COMPLEX, 0, POOL_WORLD);
// 	}
// 	#endif

// 	return;
// }

template class Exx_Lip<std::complex<float>, base_device::DEVICE_CPU>;
template class Exx_Lip<std::complex<double>, base_device::DEVICE_CPU>;