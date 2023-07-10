#include "dm_trans.h"
#include "module_base/blas_connector.h"
#include "module_base/tool_title.h"
namespace hamilt
{
    // void out_mat(const ModuleBase::matrix& mat)
    // {
    //     // for (size_t i = 0;i < mat.nc;++i)
    //     // {
    //     //     for (size_t j = 0;j < mat.nr;++j)
    //     //         std::cout << mat(j, i) << " ";
    //     //     std::cout << std::endl;
    //     // }
    //     for (size_t i = 0;i < mat.nr * mat.nc;++i)
    //         std::cout << mat.c[i] << " ";
    //     std::cout << std::endl;
    // }
    // void cxc_out_test_isk(const int isk, const psi::Psi<double, psi::DEVICE_CPU>& X_istate, const psi::Psi<double, psi::DEVICE_CPU>& c)
    // {
    //     int naos = c.get_nbasis();
    //     int nocc = X_istate.get_nbands();
    //     int nvirt = X_istate.get_nbasis();
    //     // c*
    //     c.fix_k(isk);
    //     X_istate.fix_k(isk);
    //     std::cout << "c(j, mu): " << std::endl;
    //     for (size_t j = 0;j < nocc;++j)
    //     {
    //         for (size_t mu = 0;mu < naos;++mu)
    //             std::cout << c(j, mu) << " ";
    //         std::cout << std::endl;
    //     }
    //     std::cout << "c serial " << std::endl;
    //     for (size_t i = 0;i < nocc * naos;++i)
    //         std::cout << c.get_pointer()[i] << " ";
    //     std::cout << "x(j, b)" << std::endl;
    //     for (size_t j = 0;j < nocc;++j)
    //     {
    //         for (size_t b = 0;b < nvirt;++b)
    //             std::cout << X_istate(j, b) << " ";
    //         std::cout << std::endl;
    //     }
    //     std::cout << "c*(b, nu)" << std::endl;
    //     for (size_t b = nocc;b < nocc + nvirt;++b)
    //     {
    //         for (size_t mu = 0;mu < naos;++mu)
    //             std::cout << c(b, mu) << " ";
    //         std::cout << std::endl;
    //     }
    // }
    // void cxc_out_test(const psi::Psi<double, psi::DEVICE_CPU>& X_istate, const psi::Psi<double, psi::DEVICE_CPU>& c)
    // {
    //     int nsk = c.get_nk();
    //     assert(nsk == X_istate.get_nk());
    //     // c*
    //     for (size_t isk = 0;isk < nsk;++isk)
    //     {
    //         cxc_out_test_isk(isk, X_istate, c);
    //     }
    // }
    std::vector<ModuleBase::matrix> cal_dm_trans_forloop_serial(const psi::Psi<double, psi::DEVICE_CPU>& X_istate, const psi::Psi<double, psi::DEVICE_CPU>& c)
    {
        // cxc_out_test(X_istate, c);
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_forloop");
        int nsk = c.get_nk();
        assert(nsk == X_istate.get_nk());
        int naos = c.get_nbasis();
        int nocc = X_istate.get_nbands();
        int nvirt = X_istate.get_nbasis();
        std::vector<ModuleBase::matrix> dm_trans(nsk);
        // loop for AOs
        for (size_t isk = 0;isk < nsk;++isk)
        {
            dm_trans[isk].create(naos, naos);
            c.fix_k(isk);
            X_istate.fix_k(isk);
            for (size_t mu = 0;mu < naos;++mu)
            {
                for (size_t nu = 0;nu < naos;++nu)
                {
                    // loop for ks states
                    for (size_t j = 0;j < nocc;++j)
                    {
                        for (size_t b = 0; b < nvirt;++b)
                            dm_trans[isk](mu, nu) += c(j, mu) * X_istate(j, b) * c(nocc + b, nu);
                    }
                }
            }
        }
        return dm_trans;
    }

    std::vector<ModuleBase::matrix> cal_dm_trans_blas(const psi::Psi<double, psi::DEVICE_CPU>& X_istate, const psi::Psi<double, psi::DEVICE_CPU>& c)
    {
        ModuleBase::TITLE("hamilt_lrtd", "cal_dm_trans_blas");
        int nsk = c.get_nk();
        assert(nsk == X_istate.get_nk());
        int naos = c.get_nbasis();
        int nocc = X_istate.get_nbands();
        int nvirt = X_istate.get_nbasis();
        std::vector<ModuleBase::matrix> dm_trans(nsk);
        for (size_t isk = 0;isk < nsk;++isk)
        {
            dm_trans[isk].create(naos, naos);
            c.fix_k(isk);
            X_istate.fix_k(isk);
            // Xc^*, c^*=c [nocc, naos] for gamma_only
            char transa = 'T';
            char transb = 'T';
            const double alpha = 1.0;
            const double beta = 0.0;
            ModuleBase::matrix Xc(nocc, naos, true);
            dgemm_(&transa, &transb, &nocc, &naos, &nvirt, &alpha,
                X_istate.get_pointer(), &nvirt, c.get_pointer(nocc), &naos,
                &beta, Xc.c, &nocc);
            // cXc^*
            dgemm_(&transa, &transb, &naos, &naos, &nocc, &alpha,
                Xc.c, &nocc, c.get_pointer(), &naos, &beta,
                dm_trans[isk].c, &naos);
        }
        return dm_trans;
    }

}
