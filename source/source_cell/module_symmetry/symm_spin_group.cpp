#include "symmetry.h"
using namespace ModuleSymmetry;
#include "symm_rot_spin.h"

void Symmetry::analyze_spin_space_group_nspin4(const Atom* atoms, const Statistics& st, const ModuleBase::Matrix3& latvec)
{
    // (nspin=4 noncollinear, SOC off) Spin space group: the spin rotation R_spin decouples from the
    // spatial op g. For each g we FIT an independent proper R_spin from the moment configuration
    // {m_i} -> {m_{g(i)}} (SpinRotation::fit_spin_rotation) instead of locking it to spin_so3(gmatc).
    // Operation g is kept as UNITARY if R_spin maps +m (store spin_rotation_ssg, aligned with gmatrix),
    // as ANTIUNITARY Theta*g if R_spin maps -m (store spin_rotation_anti_ssg, aligned with gmatrix_anti),
    // else dropped. Arrays are populated EXACTLY like analyze_magnetic_group_nspin4 so all downstream
    // consumers (k-reduction, density, EXX) are unchanged; only the spin-rotation source differs.

    const ModuleBase::Matrix3 ilatvec = latvec.Inverse();
    std::vector<int> keep;                       // unitary SSG operations
    keep.reserve(this->nrotk);
    std::vector<ModuleBase::Matrix3> keep_R;     // spin rotation R_spin aligned with keep (-> spin_rotation_ssg)
    keep_R.reserve(this->nrotk);
    std::vector<int> anti;                       // antiunitary SSG coset (rank>=2 only)
    anti.reserve(this->nrotk);
    std::vector<ModuleBase::Matrix3> anti_R;     // R_spin aligned with anti (-> spin_rotation_anti_ssg)
    anti_R.reserve(this->nrotk);
    int nrot_new = 0;

    // Moment rank: nonmagnetic (rank 0), collinear (rank 1, all moments (anti)parallel along one axis n_hat),
    // or noncollinear (rank >= 2). The independent-spin fit needs rank >= 2; the collinear and
    // nonmagnetic cases are handled separately below.
    ModuleBase::Vector3<double> axis(0.0, 0.0, 0.0);   // n_hat (unnormalized) of the collinear moments
    bool has_moment = false;
    bool rank_ge_2 = false;
    for (int iat = 0; iat < this->nat; ++iat)
    {
        const ModuleBase::Vector3<double>& m = atoms[st.iat2it[iat]].m_loc_[st.iat2ia[iat]];
        if (m.norm() < this->epsilon) { continue; }
        has_moment = true;
        if (axis.norm() < this->epsilon) { axis = m; }
        // "^" is the cross product operator of Vector3
        else if ((m ^ axis).norm() > 100.0 * this->epsilon) { rank_ge_2 = true; break; }   // cross!=0 => not (anti)parallel
    }

    if (!has_moment)
    {
        // rank 0 (nonmagnetic): the fit is meaningless; the magnetic path keeps the full group. Fall back.
        this->analyze_magnetic_group_nspin4(atoms, st, latvec);
        return;
    }

    // diagnostic: how many ops the magnetic (spin-locked-to-space) subgroup WOULD keep as unitary, for
    // the "SSG order > magnetic order" comparison printed below. Pure counting, no state change.
    int mag_unitary_count = 0;
    for (int isym = 0; isym < this->nrotk; ++isym)
    {
        const ModuleBase::Matrix3 gmatc = ilatvec * this->gmatrix[isym] * latvec;
        const ModuleBase::Matrix3 W = ModuleSymmetry::SpinRotation::spin_so3(gmatc);
        bool mag_ok = true;
        for (int iat = 0; iat < this->nat && mag_ok; ++iat)
        {
            const ModuleBase::Vector3<double>& m = atoms[st.iat2it[iat]].m_loc_[st.iat2ia[iat]];
            const ModuleBase::Vector3<double> mrot = W * m;
            const int jat = this->get_rotated_atom(isym, iat);
            const ModuleBase::Vector3<double>& mj = atoms[st.iat2it[jat]].m_loc_[st.iat2ia[jat]];
            if (!this->equal(mrot.x, mj.x) || !this->equal(mrot.y, mj.y) || !this->equal(mrot.z, mj.z)) { mag_ok = false; }
        }
        if (mag_ok) { ++mag_unitary_count; }
    }

    if (rank_ge_2)
    {
        // NONCOLLINEAR: fit an independent proper R_spin per operation.
        const double fit_tol = 100.0 * this->epsilon;   // moment-match tolerance for the fit verification
        for (int isym = 0; isym < this->nrotk; ++isym)
        {
            std::vector<ModuleBase::Vector3<double>> from(this->nat), to_plus(this->nat), to_minus(this->nat);
            for (int iat = 0; iat < this->nat; ++iat)
            {
                const ModuleBase::Vector3<double>& m = atoms[st.iat2it[iat]].m_loc_[st.iat2ia[iat]];
                const int jat = this->get_rotated_atom(isym, iat);
                const ModuleBase::Vector3<double>& mj = atoms[st.iat2it[jat]].m_loc_[st.iat2ia[jat]];
                from[iat] = m;
                to_plus[iat] = mj;
                to_minus[iat] = mj * (-1.0);
            }
            // unitary SSG test: fit a proper R_spin mapping +m.
            bool ok = false;
            ModuleBase::Matrix3 R = ModuleSymmetry::SpinRotation::fit_spin_rotation(from, to_plus, ok, fit_tol);
            if (ok)
            {
                keep.push_back(isym);
                keep_R.push_back(R);
                if (isym < this->nrot) { ++nrot_new; }
            }
            else
            {
                // antiunitary SSG coset: R_spin maps -m, i.e. Theta*g (Theta = trs/sigma_y downstream).
                bool anti_ok = false;
                ModuleBase::Matrix3 Ra = ModuleSymmetry::SpinRotation::fit_spin_rotation(from, to_minus, anti_ok, fit_tol);
                if (anti_ok) { anti.push_back(isym); anti_R.push_back(Ra); }
            }
        }
    }
    else
    {
        // COLLINEAR (rank 1): the spin part is the discrete Z2 group {I, C2_perp}. For a moment
        // configuration +-M n_hat, an operation g that PRESERVES the sign pattern (m_{g(i)}=+m_i) gets
        // R_spin=I; one that REVERSES it (m_{g(i)}=-m_i) gets R_spin=C2 about a FIXED axis e_hat perp
        // n_hat (a proper 180-deg rotation sending n_hat -> -n_hat, purely UNITARY, no time reversal).
        // Using the SAME e_hat for every reversing op makes g->R_spin a valid Z2 homomorphism (C2_e^2=I),
        // so the group closes. Unlike the magnetic path (spin_so3 locks spin to space and DISCARDS ops
        // whose spatial rotation tilts n_hat off-axis), the decoupled R_spin recovers those ops.
        // All kept operations are UNITARY (anti stays empty); downstream is identical to the rank>=2 case.
        ModuleBase::Vector3<double> nhat = axis * (1.0 / axis.norm());
        const ModuleBase::Vector3<double> ref = (std::fabs(nhat.x) < 0.9)
            ? ModuleBase::Vector3<double>(1.0, 0.0, 0.0) : ModuleBase::Vector3<double>(0.0, 1.0, 0.0);
        ModuleBase::Vector3<double> ehat = ref ^ nhat;
        ehat = ehat * (1.0 / ehat.norm());   // fixed unit vector perpendicular to n_hat
        // C2 about ehat:  R = 2 ehat (x) ehat - I  (symmetric; maps ehat->ehat, anything perp ehat -> -it,
        // in particular n_hat -> -n_hat since n_hat perp ehat).
        const ModuleBase::Matrix3 Rflip(
            2.0 * ehat.x * ehat.x - 1.0, 2.0 * ehat.x * ehat.y,       2.0 * ehat.x * ehat.z,
            2.0 * ehat.y * ehat.x,       2.0 * ehat.y * ehat.y - 1.0, 2.0 * ehat.y * ehat.z,
            2.0 * ehat.z * ehat.x,       2.0 * ehat.z * ehat.y,       2.0 * ehat.z * ehat.z - 1.0);
        const ModuleBase::Matrix3 Iden(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        for (int isym = 0; isym < this->nrotk; ++isym)
        {
            bool preserve = true, reverse = true;
            for (int iat = 0; iat < this->nat && (preserve || reverse); ++iat)
            {
                const ModuleBase::Vector3<double>& m = atoms[st.iat2it[iat]].m_loc_[st.iat2ia[iat]];
                const int jat = this->get_rotated_atom(isym, iat);
                const ModuleBase::Vector3<double>& mj = atoms[st.iat2it[jat]].m_loc_[st.iat2ia[jat]];
                if (!this->equal(mj.x, m.x) || !this->equal(mj.y, m.y) || !this->equal(mj.z, m.z)) { preserve = false; }
                if (!this->equal(mj.x, -m.x) || !this->equal(mj.y, -m.y) || !this->equal(mj.z, -m.z)) { reverse = false; }
            }
            if (preserve)      { keep.push_back(isym); keep_R.push_back(Iden);  if (isym < this->nrot) { ++nrot_new; } }
            else if (reverse)  { keep.push_back(isym); keep_R.push_back(Rflip); if (isym < this->nrot) { ++nrot_new; } }
        }
    }

    // capture the antiunitary coset BEFORE the unitary arrays are compacted below.
    this->magnetic_nspin4 = has_moment;
    this->spin_space_group_nspin4 = true;
    this->nrotk_anti = static_cast<int>(anti.size());
    if (this->nrotk_anti > 0)
    {
        this->isym_rotiat_anti_.resize(this->nrotk_anti);
        for (int j = 0; j < this->nrotk_anti; ++j)
        {
            const int isym = anti[j];
            this->gmatrix_anti[j] = this->gmatrix[isym];
            this->kgmatrix_anti[j] = this->kgmatrix[isym];
            this->gtrans_anti[j] = this->gtrans[isym];
            this->isym_rotiat_anti_[j] = this->isym_rotiat_[isym];
            this->spin_rotation_anti_ssg[j] = anti_R[j];
        }
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,
            "SSG ANTIUNITARY OPERATIONS (Theta*g, nspin=4)", this->nrotk_anti);
    }

    const int nrotk_new = static_cast<int>(keep.size());
    // store the fitted unitary spin rotations aligned with the compacted gmatrix[0..nrotk_new).
    for (int i = 0; i < nrotk_new; ++i) { this->spin_rotation_ssg[i] = keep_R[i]; }

    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "MAGNETIC SUBGROUP UNITARY OPS (for comparison)", mag_unitary_count);
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "SSG UNITARY OPERATIONS (nspin=4)", nrotk_new);

    if (nrotk_new == this->nrotk) { return; }   // nothing removed

    for (int i = 0; i < nrotk_new; ++i)
    {
        const int isym = keep[i];
        if (i != isym)
        {
            this->gmatrix[i] = this->gmatrix[isym];
            this->kgmatrix[i] = this->kgmatrix[isym];
            this->gtrans[i] = this->gtrans[isym];
            this->isym_rotiat_[i] = this->isym_rotiat_[isym];
        }
    }
    this->isym_rotiat_.resize(nrotk_new);
    this->nrot = nrot_new;
    this->nrotk = nrotk_new;

    this->pointgroup(this->nrot, this->pgnumber, this->pgname, this->gmatrix, GlobalV::ofs_running, nullptr);
    this->pointgroup(this->nrotk, this->spgnumber, this->spgname, this->gmatrix, GlobalV::ofs_running, nullptr);
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "SSG POINT GROUP (unitary, nspin=4)", this->pgname);
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "SSG POINT GROUP IN SPACE GROUP", this->spgname);
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "SSG SPACE GROUP OPERATIONS (nspin=4)", this->nrotk);
}

void Symmetry::analyze_spin_space_group_nspin2(const Atom* atoms, const Statistics& st)
{
    // (nspin=2 collinear spin space group) The FULL chemical space group is already in
    // gmatrix[0..nrotk). Split it, using the scalar collinear moments mag[iat], into:
    //   - unitary magnetic subgroup : mag[iat] == mag[g(iat)]  for every atom  -> kept in gmatrix
    //   - unitary spin-flip coset   : mag[iat] == -mag[g(iat)] for every atom  -> gmatrix_flip[]
    // The spin-flip coset elements are [C2_perp||g]: they swap the up/down spin channels while
    // rotating space by g. There is NO time reversal here (the collinear Hamiltonian is real, the two
    // spin blocks are independent), so this is a plain channel swap - distinct from the antiunitary
    // Theta*g coset built by analyze_magnetic_group_nspin4. Operations that neither preserve nor exactly
    // flip the moment pattern are dropped.
    std::vector<int> keep;   // unitary magnetic subgroup
    keep.reserve(this->nrotk);
    std::vector<int> flip;   // unitary spin-flip coset [C2_perp||g]
    flip.reserve(this->nrotk);
    int nrot_new = 0;
    for (int isym = 0; isym < this->nrotk; ++isym)
    {
        bool preserve = true;
        for (int iat = 0; iat < this->nat && preserve; ++iat)
        {
            const double mi = atoms[st.iat2it[iat]].mag[st.iat2ia[iat]];
            const int jat = this->get_rotated_atom(isym, iat);
            const double mj = atoms[st.iat2it[jat]].mag[st.iat2ia[jat]];
            if (!this->equal(mi, mj)) { preserve = false; }
        }
        if (preserve)
        {
            keep.push_back(isym);
            if (isym < this->nrot) { ++nrot_new; }   // pure point-group rotations are the first nrot ops
            continue;
        }
        // g does not preserve the moment pattern; check whether it exactly FLIPS it, i.e.
        // mag[iat] = -mag[g(iat)] for every atom. Then [C2_perp||g] (spatial g + up<->down swap) is a symmetry.
        bool flip_ok = true;
        for (int iat = 0; iat < this->nat && flip_ok; ++iat)
        {
            const double mi = atoms[st.iat2it[iat]].mag[st.iat2ia[iat]];
            const int jat = this->get_rotated_atom(isym, iat);
            const double mj = atoms[st.iat2it[jat]].mag[st.iat2ia[jat]];
            if (!this->equal(mi, -mj)) { flip_ok = false; }
        }
        if (flip_ok) { flip.push_back(isym); }
    }

    // Capture the spin-flip coset BEFORE the unitary arrays are compacted in place below.
    this->nrotk_flip = static_cast<int>(flip.size());
    this->spin_flip_nspin2 = (this->nrotk_flip > 0);
    if (this->nrotk_flip > 0)
    {
        this->isym_rotiat_flip_.resize(this->nrotk_flip);
        for (int j = 0; j < this->nrotk_flip; ++j)
        {
            const int isym = flip[j];
            this->gmatrix_flip[j] = this->gmatrix[isym];
            this->kgmatrix_flip[j] = this->kgmatrix[isym];
            this->gtrans_flip[j] = this->gtrans[isym];
            this->isym_rotiat_flip_[j] = this->isym_rotiat_[isym];
        }
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,
            "SPIN-FLIP COSET OPERATIONS (nspin=2 SSG)", this->nrotk_flip);
    }

    const int nrotk_new = static_cast<int>(keep.size());
    if (nrotk_new != this->nrotk)
    {
        // compact the unitary subgroup in ascending order (keeps the rotations-first layout).
        for (int i = 0; i < nrotk_new; ++i)
        {
            const int isym = keep[i];
            if (i != isym)
            {
                this->gmatrix[i] = this->gmatrix[isym];
                this->kgmatrix[i] = this->kgmatrix[isym];
                this->gtrans[i] = this->gtrans[isym];
                this->isym_rotiat_[i] = this->isym_rotiat_[isym];
            }
        }
        this->isym_rotiat_.resize(nrotk_new);
        this->nrot = nrot_new;
        this->nrotk = nrotk_new;

        // refresh the point-/space-group labels for the reduced (unitary magnetic) group
        this->pointgroup(this->nrot, this->pgnumber, this->pgname, this->gmatrix, GlobalV::ofs_running, nullptr);
        this->pointgroup(this->nrotk, this->spgnumber, this->spgname, this->gmatrix, GlobalV::ofs_running, nullptr);
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "MAGNETIC POINT GROUP (unitary, nspin=2)", this->pgname);
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "MAGNETIC SPACE GROUP OPERATIONS", this->nrotk);
    }
    if (this->spin_flip_nspin2)
    {
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,
            "SPIN SPACE GROUP OPERATIONS (unitary + spin-flip)", this->nrotk + this->nrotk_flip);
    }
}

int Symmetry::spin_flip_sym_ops(std::vector<ModuleBase::Matrix3>& kgmat,
                                std::vector<ModuleBase::Vector3<double>>& gtr,
                                std::vector<double>& flip_sign) const
{
    // (nspin=2 collinear SSG) Assemble the full spin space group used to symmetrize the collinear
    // density and to fold the k-points: the nrotk unitary operations (flip_sign +1) followed by the
    // nrotk_flip spatial parts of the spin-flip coset [C2_perp||g] (flip_sign -1). The combined set
    // is the full chemical space group -- a group, closed under inverse -- so the invmap/grouping in
    // rhog_symmetry* remains valid. In the (charge, mag) basis the charge is invariant (all ops act
    // as ordinary space-group operations) and the magnetization flips sign under the coset.
    const int nu = this->nrotk;
    const int nf = (this->spin_flip_nspin2 ? this->nrotk_flip : 0);
    kgmat.resize(nu + nf);
    gtr.resize(nu + nf);
    flip_sign.assign(nu + nf, 1.0);
    for (int i = 0; i < nu; ++i)
    {
        kgmat[i] = this->kgmatrix[i];
        gtr[i] = this->gtrans[i];
    }
    for (int j = 0; j < nf; ++j)
    {
        kgmat[nu + j] = this->kgmatrix_flip[j];
        gtr[nu + j] = this->gtrans_flip[j];
        flip_sign[nu + j] = -1.0;
    }
    return nu + nf;
}
