#include "CVCX.h"
#include "module_base/blas_connector.h"
#include "module_base/tool_title.h"
#include "module_beyonddft/utils/lr_util.h"
namespace hamilt
{
    //=====================occ========================
    template <>
    void CVCX_occ_forloop_serial(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<double, psi::DEVICE_CPU>& c,
        const psi::Psi<double, psi::DEVICE_CPU>& X_istate,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<double, psi::DEVICE_CPU>& AX_istate)
    {
        ModuleBase::TITLE("hamilt_lrtd", "CVCX_occ_forloop_serial");
        int nks = c.get_nk();
        assert(V_istate.size() == nks);
        assert(naos == c.get_nbasis());
        AX_istate.fix_k(0);
        ModuleBase::GlobalFunc::ZEROS(AX_istate.get_pointer(), nks * nocc * nvirt);
        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            AX_istate.fix_k(isk);
            for (int i = 0;i < nocc;++i)
                for (int a = 0;a < nvirt;++a)
                    for (int nu = 0;nu < naos;++nu)
                        for (int mu = 0;mu < naos;++mu)
                            for (int j = 0;j < nocc;++j)
                                AX_istate(i * nvirt + a) += X_istate(j * nvirt + a) * c(i, mu) * V_istate[isk].data<double>()[nu * naos + mu] * c(j, nu);
        }
    }
    template <>
    void CVCX_occ_forloop_serial(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>& AX_istate)
    {
        ModuleBase::TITLE("hamilt_lrtd", "CVCX_occ_forloop_serial");
        int nks = c.get_nk();
        assert(V_istate.size() == nks);
        assert(naos == c.get_nbasis());
        AX_istate.fix_k(0);
        ModuleBase::GlobalFunc::ZEROS(AX_istate.get_pointer(), nks * nocc * nvirt);
        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            AX_istate.fix_k(isk);
            for (int i = 0;i < nocc;++i)
                for (int a = 0;a < nvirt;++a)
                    for (int nu = 0;nu < naos;++nu)
                        for (int mu = 0;mu < naos;++mu)
                            for (int j = 0;j < nocc;++j)
                                AX_istate(i * nvirt + a) += std::conj(X_istate(j * nvirt + a) * c(i, mu)) * V_istate[isk].data<std::complex<double>>()[nu * naos + mu] * c(j, nu);
        }
    }

    template <>
    void CVCX_occ_blas(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<double, psi::DEVICE_CPU>& c,
        const psi::Psi<double, psi::DEVICE_CPU>& X_istate,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<double, psi::DEVICE_CPU>& AX_istate,
        const bool add_on)
    {
        ModuleBase::TITLE("hamilt_lrtd", "CVCX_occ_AX_blas");
        int nks = c.get_nk();
        assert(V_istate.size() == nks);
        assert(naos == c.get_nbasis());

        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            AX_istate.fix_k(isk);
            const char trans = 'T';
            const char notrans = 'N';  //c is col major
            const double one = 1.0;
            const double zero = 0.0;

            // c^TV[nocc*naos]
            container::Tensor cv(DAT::DT_DOUBLE, DEV::CpuDevice, { naos, nocc });
            dgemm_(&trans, &notrans, &nocc, &naos, &naos, &one,
                c.get_pointer(), &naos, V_istate[isk].data<double>(), &naos, &zero,
                cv.data<double>(), &nocc);

            // cX^T[naos*nvirt]
            container::Tensor cx(DAT::DT_DOUBLE, DEV::CpuDevice, { nvirt, naos });
            dgemm_(&notrans, &trans, &naos, &nvirt, &nocc, &one,
                c.get_pointer(), &naos, X_istate.get_pointer(), &nvirt, &zero,
                cx.data<double>(), &naos);

            //AX_istate=[cX^T]^T[c^TV]^T (nvirt major)
            dgemm_(&trans, &trans, &nvirt, &nocc, &naos, &one,
                cx.data<double>(), &naos, cv.data<double>(), &nocc, add_on ? &one : &zero,
                AX_istate.get_pointer(), &nvirt);
        }
    }

    template <>
    void CVCX_occ_blas(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>& AX_istate,
        const bool add_on)
    {
        ModuleBase::TITLE("hamilt_lrtd", "CVCX_occ_AX_blas");
        int nks = c.get_nk();
        assert(V_istate.size() == nks);
        assert(naos == c.get_nbasis());

        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            AX_istate.fix_k(isk);
            const char trans = 'T';
            const char notrans = 'N';  //c is col major
            const char dagger = 'C';
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);

            // c^TV[nocc*naos]
            container::Tensor cv(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { naos, nocc });
            zgemm_(&dagger, &notrans, &nocc, &naos, &naos, &one,
                c.get_pointer(), &naos, V_istate[isk].data<std::complex<double>>(), &naos, &zero,
                cv.data<std::complex<double>>(), &nocc);

            // cX^T[naos*nvirt]
            container::Tensor cx(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { nvirt, naos });
            zgemm_(&notrans, &dagger, &naos, &nvirt, &nocc, &one,
                c.get_pointer(), &naos, X_istate.get_pointer(), &nvirt, &zero,
                cx.data<std::complex<double>>(), &naos);

            //AX_istate=[cX^T]^T[c^TV]^T (nvirt major)
            zgemm_(&trans, &trans, &nvirt, &nocc, &naos, &one,
                cx.data<std::complex<double>>(), &naos, cv.data<std::complex<double>>(), &nocc, add_on ? &one : &zero,
                AX_istate.get_pointer(), &nvirt);
        }
    }


    //=====================virt========================
    template <>
    void CVCX_virt_forloop_serial(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<double, psi::DEVICE_CPU>& c,
        const psi::Psi<double, psi::DEVICE_CPU>& X_istate,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<double, psi::DEVICE_CPU>& AX_istate)
    {
        ModuleBase::TITLE("hamilt_lrtd", "CVCX_virt_forloop_serial");
        int nks = c.get_nk();
        assert(V_istate.size() == nks);
        assert(naos == c.get_nbasis());
        AX_istate.fix_k(0);
        ModuleBase::GlobalFunc::ZEROS(AX_istate.get_pointer(), nks * nocc * nvirt);
        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            AX_istate.fix_k(isk);
            for (int i = 0;i < nocc;++i)
                for (int a = 0;a < nvirt;++a)
                    for (int nu = 0;nu < naos;++nu)
                        for (int mu = 0;mu < naos;++mu)
                            for (int b = 0;b < nvirt;++b)
                                AX_istate(i * nvirt + a) += X_istate(i * nvirt + b) * c(nocc + b, mu) * V_istate[isk].data<double>()[nu * naos + mu] * c(nocc + a, nu);
        }
    }
    template <>
    void CVCX_virt_forloop_serial(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>& AX_istate)
    {
        ModuleBase::TITLE("hamilt_lrtd", "CVCX_virt_forloop_serial");
        int nks = c.get_nk();
        assert(V_istate.size() == nks);
        assert(naos == c.get_nbasis());
        AX_istate.fix_k(0);
        ModuleBase::GlobalFunc::ZEROS(AX_istate.get_pointer(), nks * nocc * nvirt);
        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            AX_istate.fix_k(isk);
            for (int i = 0;i < nocc;++i)
                for (int a = 0;a < nvirt;++a)
                    for (int nu = 0;nu < naos;++nu)
                        for (int mu = 0;mu < naos;++mu)
                            for (int b = 0;b < nvirt;++b)
                                AX_istate(i * nvirt + a) += std::conj(X_istate(i * nvirt + b) * c(nocc + b, mu)) * V_istate[isk].data<std::complex<double>>()[nu * naos + mu] * c(nocc + a, nu);
        }
    }

    template <>
    void CVCX_virt_blas(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<double, psi::DEVICE_CPU>& c,
        const psi::Psi<double, psi::DEVICE_CPU>& X_istate,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<double, psi::DEVICE_CPU>& AX_istate,
        const bool add_on)
    {
        ModuleBase::TITLE("hamilt_lrtd", "CVCX_virt_AX_blas");
        int nks = c.get_nk();
        assert(V_istate.size() == nks);
        assert(naos == c.get_nbasis());

        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            AX_istate.fix_k(isk);
            const char trans = 'T';
            const char notrans = 'N';  //c is col major
            const double one = 1.0;
            const double zero = 0.0;

            // VC[naos*nvirt]
            container::Tensor cv(DAT::DT_DOUBLE, DEV::CpuDevice, { nvirt, naos });
            dgemm_(&notrans, &notrans, &naos, &nvirt, &naos, &one,
                V_istate[isk].data<double>(), &naos, c.get_pointer(nocc), &naos, &zero,
                cv.data<double>(), &naos);

            // X^TC^T[nocc*naos]
            container::Tensor cx(DAT::DT_DOUBLE, DEV::CpuDevice, { naos, nocc });
            dgemm_(&trans, &trans, &nocc, &naos, &nvirt, &one,
                X_istate.get_pointer(), &nvirt, c.get_pointer(nocc), &naos, &zero,
                cx.data<double>(), &nocc);

            //AX_istate=[VC]^T[X^TC^T]^T (nvirt major)
            dgemm_(&trans, &trans, &nvirt, &nocc, &naos, &one,
                cv.data<double>(), &naos, cx.data<double>(), &nocc, add_on ? &one : &zero,
                AX_istate.get_pointer(), &nvirt);
        }
    }

    template <>
    void CVCX_virt_blas(
        const std::vector<container::Tensor>& V_istate,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& c,
        const psi::Psi<std::complex<double>, psi::DEVICE_CPU>& X_istate,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        psi::Psi<std::complex<double>, psi::DEVICE_CPU>& AX_istate,
        const bool add_on)
    {
        ModuleBase::TITLE("hamilt_lrtd", "CVCX_virt_AX_blas");
        int nks = c.get_nk();
        assert(V_istate.size() == nks);
        assert(naos == c.get_nbasis());

        for (int isk = 0;isk < nks;++isk)
        {
            c.fix_k(isk);
            X_istate.fix_k(isk);
            AX_istate.fix_k(isk);
            const char trans = 'T';
            const char notrans = 'N';  //c is col major
            const char dagger = 'C';
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);

            // VC[naos*nvirt]
            container::Tensor cv(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { nvirt, naos });
            zgemm_(&notrans, &notrans, &naos, &nvirt, &naos, &one,
                V_istate[isk].data<std::complex<double>>(), &naos, c.get_pointer(nocc), &naos, &zero,
                cv.data<std::complex<double>>(), &naos);

            // X^TC^T[nocc*naos]
            container::Tensor cx(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { naos, nocc });
            zgemm_(&dagger, &dagger, &nocc, &naos, &nvirt, &one,
                X_istate.get_pointer(), &nvirt, c.get_pointer(nocc), &naos, &zero,
                cx.data<std::complex<double>>(), &nocc);

            //AX_istate=[VC]^T[X^TC^T]^T (nvirt major)
            zgemm_(&trans, &trans, &nvirt, &nocc, &naos, &one,
                cv.data<std::complex<double>>(), &naos, cx.data<std::complex<double>>(), &nocc, add_on ? &one : &zero,
                AX_istate.get_pointer(), &nvirt);
        }
    }
}