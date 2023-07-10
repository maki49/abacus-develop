#include "moduel_hamilt_general/operator.h"
#include "module_hamilt_lcao/module_gint/gint_gamma.h"
#include "module_hamilt_lcao/module_gint/grid_technique.h"
#include "module_hamilt_lcao/module_gint/gint_k.h"


namespace hamilt
{
    // separate in two classes because the algorithms for 
    // gamma_only and multi_k are completely different

    /// @brief  kernel for Hxc operator (gama_only)
    /// @tparam Device 
    class OperatorA_Hxc_Double : public Operator<double, psi::DEVICE_CPU>
    {
    public:
        OperatorA_Hxc_Double(const int& ns, const Gint_Gamma& gg, const Grid_Technique& gt,
            const Parallel_2D* pc_in, const Parallel_2D& px_in)
            : nspin(ns), gint_g(&gg), gridt(&gt), pc(pc_in), px(px_in)
        {
            this->cal_type = calculation_type::lcao_gint;
            this->is_first_node = true;
        };
        virtual void init(const int ik_in) override;
        virtual void act() const override;  // call gint

    private:
        int nspin;
        int naos;
        int naos_local_grid;    ///< number of ao basis on local grid  (lgd)
        int naos_local_row;     ///< row number of local ao basis (2d-block)
        int naos_local_col;     ///< col number of local ao basis (2d-block)
        int nocc;
        int nvirt;
        Parallel_2D* pc = nullptr;
        Parallel_2D* px = nullptr;

        elecstate::Potential* pot = nullptr;

        Gint_Gamma* gint_g;
        /// \f[ \tilde{\rho}(r)=\sum_{\mu_j, \mu_b}\tilde{\rho}_{\mu_j,\mu_b}\phi_{\mu_b}(r)\phi_{\mu_j}(r) \f]
        void cal_rho_trans();
    };

    /// @brief  kernel for Hxc operator (multi-k)
    class OperatorA_Hxc_Complex : public Operator<std::complex<double>, psi::DEVICE_CPU>
    {
    public:
        OperatorA_Hxc_Complex(const int& nk, cosnt Gint_k& gk, const Grid_Technique& gt,
            const Parallel_2D* pc_in, const Parallel_2D& px_in)
            : nspin(ns), gint_g(&gg), gridt(&gt), pc(pc_in), px(px_in)
        {
            this->cal_type = calculation_type::lcao_gint;
            this->is_first_node = true;
        };
        virtual void init(const int ik_in) override;
        virtual void act() const override;  // call gint
    private:
        int nkpt;
        Gint_k* gint_k;

    };


}
#include "operatorA_hxc_double.hpp"