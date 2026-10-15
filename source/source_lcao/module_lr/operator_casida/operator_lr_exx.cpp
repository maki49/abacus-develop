#ifdef __EXX
#include "operator_lr_exx.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <type_traits>
#include "source_lcao/module_lr/dm_trans/dm_trans.h"
#include "source_lcao/module_lr/utils/lr_util.h"
#include "source_lcao/module_lr/exx_proj.h"
#include "source_base/parallel_reduce.h"
namespace LR
{
    namespace
    {
        template<typename T>
        T projection_phase(const ModuleBase::Vector3<double>& k,
                           const std::array<int, 3>& cell,
                           const ModuleBase::Matrix3& lattice);

        template<>
        double projection_phase<double>(const ModuleBase::Vector3<double>&,
                                        const std::array<int, 3>&,
                                        const ModuleBase::Matrix3&)
        {
            return 1.0;
        }

        template<>
        std::complex<double> projection_phase<std::complex<double>>(
            const ModuleBase::Vector3<double>& k,
            const std::array<int, 3>& cell,
            const ModuleBase::Matrix3& lattice)
        {
            const auto translation = RI_Util::array3_to_Vector3(cell) * lattice;
            const double angle = ModuleBase::TWO_PI * (k * translation);
            // The legacy probe contraction conjugated the negative Bloch phase.
            return std::exp(ModuleBase::IMAG_UNIT * angle);
        }

        template<typename T>
        using AtomTensorMap = std::map<RI_2D_Comm::TA,
                                       std::map<RI_2D_Comm::TAC, RI::Tensor<T>>>;

        // Transform the current MO transition density into LibRI atom blocks.
        // Borrow DM(k): copying it once per trial vector is unnecessary.
        template<typename T>
        AtomTensorMap<T> split_transition_density(const UnitCell& ucell,
                                                  const K_Vectors& kv,
                                                  const module_dm::DensityMatrix<T, T>& density,
                                                  const Parallel_Orbitals& pmat,
                                                  const int nk)
        {
            ModuleBase::timer::start("OperatorLREXX", "split_transition_density");
            const std::vector<std::vector<T>>& dmk = density.get_dmk_vec();
            std::vector<const std::vector<T>*> density_k(nk);
            for (int ik = 0; ik < nk; ++ik)
            {
                density_k[ik] = &dmk[ik];
            }
            std::vector<AtomTensorMap<T>> density_r
                = RI_2D_Comm::split_m2D_ktoR<T>(ucell, kv, density_k, pmat, 1);
            ModuleBase::timer::end("OperatorLREXX", "split_transition_density");
            return std::move(density_r[0]);
        }

        // Install one unscreened LR density without changing the SCF threshold.
        template<typename T>
        void set_transition_density(Exx_LRI<T>& lri, AtomTensorMap<T>&& density_r)
        {
            ModuleBase::timer::start("OperatorLREXX", "set_transition_density");
            constexpr double density_threshold = 0.0;
            lri.get().set_Ds(std::move(density_r), density_threshold);
            ModuleBase::timer::end("OperatorLREXX", "set_transition_density");
        }

        // Apply the unchanged local-RI exchange contraction K[D] to atom blocks.
        template<typename T>
        void contract_transition_exchange(Exx_LRI<T>& lri)
        {
            ModuleBase::timer::start("OperatorLREXX", "contract_transition_exchange");
            lri.get().cal_Hs();
            ModuleBase::timer::end("OperatorLREXX", "contract_transition_exchange");
        }

        // Communicate K[D] blocks to the ranks owning their AO elements.
        template<typename T>
        void communicate_transition_exchange(Exx_LRI<T>& lri,
                                             const std::tuple<std::set<int>, std::set<int>>& judge)
        {
            ModuleBase::timer::start("OperatorLREXX", "communicate_transition_exchange");
            lri.Hexxs[0] = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                lri.get_mpi_comm(), std::move(lri.get().Hs), std::get<0>(judge), std::get<1>(judge));
            lri.post_process_Hexx(lri.Hexxs[0]);
            ModuleBase::timer::end("OperatorLREXX", "communicate_transition_exchange");
        }
    }

    template<typename T>
    void OperatorLREXX<T>::project_k(const T* psi_in, T* hpsi) const
    {
        ModuleBase::timer::start("OperatorLREXX", "project_k");
        const bool occupied = dm_pq_ == MO_TO_AO_TYPE::CC_oo;
        const int nright = occupied ? nocc : nvirt;
        // Keep the historical aims probe indices, including its complex CC_oo
        // convention, without retaining per-pair density construction.
        const bool complex_occupied = occupied && std::is_same<T, std::complex<double>>::value;
        const bool benchmark_probe = !aims_nbasis.empty()
                                  && (dm_pq_ == MO_TO_AO_TYPE::CC_vo || complex_occupied);
        if (benchmark_probe)
        {
            if (aims_nbasis.size() != static_cast<std::size_t>(ucell.ntype))
            {
                throw std::invalid_argument("EXX benchmark probe requires one basis index per atom type");
            }
            for (const int index : aims_nbasis)
            {
                if (index < 0 || index >= naos)
                {
                    throw std::out_of_range("EXX benchmark probe basis index is outside the AO matrix");
                }
            }
        }
        std::vector<T> h(static_cast<std::size_t>(naos) * naos);
        std::vector<T> result(static_cast<std::size_t>(nocc) * nright);
        std::vector<T> scratch(static_cast<std::size_t>(naos) * nocc);
        const double factor = 2.0 * alpha;
        const auto lri = this->exx_lri.lock();
        psi::Psi<T> coxt_full;
        psi::Psi<T> cvx_full;
        if (dm_pq_ == MO_TO_AO_TYPE::CXC || dm_pq_ == MO_TO_AO_TYPE::CXC_o)
        {
            coxt_full.resize(this->nk, nvirt, this->naos);
            cvx_full.resize(this->nk, nocc, this->naos);
            this->cal_coxt_cvx(psi_in, coxt_full, cvx_full);
        }
        for (int ik = 0; ik < nk; ++ik)
        {
            std::fill(h.begin(), h.end(), T(0));
            std::fill(result.begin(), result.end(), T(0));
            // Hexxs contains communicated atom blocks. Keep uniquely owned AO
            // elements used by the probe contraction; reduce the smaller MO result below.
            for (const auto& atom : lri->Hexxs[0])
            {
                const int it1 = ucell.iat2it[atom.first];
                const int ia1 = ucell.iat2ia[atom.first];
                for (const auto& pair : atom.second)
                {
                    const int it2 = ucell.iat2it[pair.first.first];
                    const int ia2 = ucell.iat2ia[pair.first.first];
                    const T phase = projection_phase<T>(kv.kvec_c[ik], pair.first.second, ucell.latvec);
                    const auto& block = pair.second;
                    for (int mu = 0; mu < ucell.atoms[it1].nw; ++mu)
                    {
                        const int native_row = ucell.itiaiw2iwt(it1, ia1, mu);
                        const int row = benchmark_probe ? aims_nbasis[it1] : native_row;
                        for (int nu = 0; nu < ucell.atoms[it2].nw; ++nu)
                        {
                            const int native_col = ucell.itiaiw2iwt(it2, ia2, nu);
                            const int col = benchmark_probe ? aims_nbasis[it2] : native_col;
                            if (pmat.in_this_processor(row, col))
                            {
                                h[static_cast<std::size_t>(col) * naos + row] += phase * block(mu, nu);
                            }
                        }
                    }
                }
            }
            if (projection_workspace_) { projection_workspace_->prepare(h.data()); }
            const auto project = [&](const T* left, const T* right, const double scale)
            {
                if (projection_workspace_)
                {
                    projection_workspace_->accumulate(left, right, scale);
                }
                else
                {
                    project_exx(h.data(), left, right, naos, nocc, nright, scale,
                                scratch.data(), result.data());
                }
            };
            const T* co = &psi_ks_full(ik, 0, 0);
            const T* cv = &psi_ks_full(ik, nocc, 0);
            if (dm_pq_ == MO_TO_AO_TYPE::CC_vo || occupied)
            {
                const T* right = occupied ? co : cv;
                project(co, right, factor);
            }
            else
            {
                const T* coxt = &coxt_full(ik, 0, 0);
                if (dm_pq_ == MO_TO_AO_TYPE::CXC)
                {
                    const T* cvx = &cvx_full(ik, 0, 0);
                    project(cvx, cv, factor);
                }
                const double occ_factor = dm_pq_ == MO_TO_AO_TYPE::CXC ? -factor : factor;
                project(co, coxt, occ_factor);
            }
            if (projection_workspace_) { projection_workspace_->download(result.data()); }
            const int result_size = static_cast<int>(result.size());
            Parallel_Reduce::reduce_all(result.data(), result_size);
            const bool transpose = occupied && std::is_same<T, std::complex<double>>::value;
            const int start = ik * pX.get_local_size();
            for (int io = 0; io < nocc; ++io)
            {
                for (int iv = 0; iv < nright; ++iv)
                {
                    if (pX.in_this_processor(iv, io))
                    {
                        const int local = pX.global2local_col(io) * pX.get_row_size()
                                        + pX.global2local_row(iv);
                        // The legacy complex CC_oo probe orders (io,iv), unlike
                        // CC_vo/CXC_o; preserve it even for nonsymmetric H(k).
                        const int index = transpose ? iv * nocc + io : io * nright + iv;
                        hpsi[start + local] += result[index];
                    }
                }
            }
        }
        ModuleBase::timer::end("OperatorLREXX", "project_k");
    }

    template<typename T>
    void OperatorLREXX<T>::cal_Hs() const
    {
        ModuleBase::TITLE("OperatorLREXX", "cal_Hs");
        ModuleBase::timer::start("OperatorLREXX", "cal_Hs");
        // Cs and Vs are prepared for this geometry; DM_trans belongs to the current trial vector.
        const std::vector<std::tuple<std::set<int>, std::set<int>>> judge
            = RI_2D_Comm::get_2D_judge(ucell, pmat);
        const std::shared_ptr<Exx_LRI<T>> lri = exx_lri.lock();
        AtomTensorMap<T> density_r
            = split_transition_density(ucell, kv, DM_trans, pmat, nk);
        // All LR exchange contractions retain every input-density block. Screening
        // small Krylov directions would break K[a D1 + b D2] = a K[D1] + b K[D2].
        // The ground-state SCF keeps its configured threshold.
        set_transition_density(*lri, std::move(density_r));
        contract_transition_exchange(*lri);
        communicate_transition_exchange(*lri, judge[0]);
        ModuleBase::timer::end("OperatorLREXX", "cal_Hs");
    }

    template<typename T>
    void OperatorLREXX<T>::act(const int nbands,
                               const int nbasis,
                               const int npol,
                               const T* psi_in,
                               T* hpsi,
                               const int ngk_ik,
                               const bool is_first_node) const
    {
        ModuleBase::TITLE("OperatorLREXX", "act");
        ModuleBase::timer::start("OperatorLREXX", "act");
        this->cal_Hs();
        this->project_k(psi_in, hpsi);
        ModuleBase::timer::end("OperatorLREXX", "act");
    }
    template class OperatorLREXX<double>;
    template class OperatorLREXX<std::complex<double>>;
}
#endif
