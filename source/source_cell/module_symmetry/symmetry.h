#ifndef SYMMETRY_H
#define SYMMETRY_H

#include "source_cell/unitcell_data.h"
#include "source_cell/atom_spec.h"
#include "source_base/timer.h"
#include "source_base/mathzone.h"
#include "source_base/constants.h"
#include "symmetry_basic.h"

namespace ModuleSymmetry
{

class Symmetry : public Symmetry_Basic
{

public:

	Symmetry() 
	{
		this->epsilon = 1e-6;
	};
    ~Symmetry() {};

	//symmetry flag for levels
	//-1 : no symmetry at all, k points would be total nks in KPT
	//0 : only basic time-reversal symmetry is considered, point k and -k would fold to k
	//1 : point group symmetry is considered
    static int symm_flag;
    static bool symm_autoclose; // controled by INPUT
    static bool pricell_loop;   ///< whether to loop primitive cell in rhog_symmetry, Only for AFM

    /// @brief analyze the symmetry of the system
    /// @param lat structure of lattice
    /// @param st 
    /// @param atoms all atoms
    /// @param ofs_running 
	/// get the symmetry information of the system, gmatries (rotation 3*3 matrixs), gtrans (transfer a collections vector3), etc.
    void analy_sys(const Lattice& lat, const Statistics& st, Atom* atoms, std::ofstream& ofs_running);

	ModuleBase::Vector3<double> s1, s2, s3;
	ModuleBase::Vector3<double> a1, a2, a3;	//primitive cell vectors(might be changed during the process of the program)
	ModuleBase::Vector3<double>	p1, p2, p3;	//primitive cell vectors
	
	int ntype=0;	  //the number of atomic species
	int nat  =0; 	  //the number of all atoms
 	int *na  =nullptr;//number of atoms for each species
	int *istart=nullptr; //start number of atom.
	int itmin_type=0; //the type has smallest number of atoms
	int itmin_start=0;

	// direct coordinates of atoms.
	double *newpos=nullptr;
	// positions of atoms after rotation.
	double *rotpos=nullptr;
	
	
	std::vector<ModuleBase::Vector3<double>> ptrans; // the translation vectors of the primitive cell in the input structure
    int ncell=1;	//the number of primitive cells within one supercell
	int *index=nullptr;
	
	double cel_const[6]={0.0};
	double pcel_const[6]={0.0};	//cel_const of primitive cell
	double pre_const[6]={0.0};	//cel_const of input configuration, first 3 is moduli of a1, a2, a3, last 3 is eular angle

	bool symflag_fft[48]={false};
	int sym_test=0;
	int pbrav=0;		//ibrav of primitive cell
	int real_brav=0;    // the real ibrav for the cell     pengfei Li 3-15-2022
	std::string ilattname;	//the bravais lattice type of the supercell
	std::string plattname;	//the bravais lattice type of the primitive cell

	ModuleBase::Matrix3 gmatrix[48];	//the rotation matrices for all space group operations
	ModuleBase::Matrix3 kgmatrix[48];	//the rotation matrices in reciprocal space
	ModuleBase::Vector3<double> gtrans[48];

	/// (nspin=4, magnetic) Spatial parts of the ANTIUNITARY elements of the Shubnikov (magnetic) group:
	/// operations g that REVERSE the magnetization, so that g alone is not a symmetry but Theta*g is (Theta = time reversal).
	/// Since an operation either preserves or reverses a non-zero moment, 
	/// this set is a coset of the unitary subgroup and is DISJOINT from
	/// gmatrix[0..nrotk); when non-empty it has exactly nrotk elements.
	/// Index convention used downstream (k-stars, restore_dm): isym < nrotk  -> unitary gmatrix[isym],
	/// isym >= nrotk -> antiunitary Theta*gmatrix_anti[isym-nrotk].
	ModuleBase::Matrix3 gmatrix_anti[48];
	ModuleBase::Matrix3 kgmatrix_anti[48];
	ModuleBase::Vector3<double> gtrans_anti[48];
	int nrotk_anti = 0;         ///< number of antiunitary elements; 0 = none (or non-magnetic)
	/// nspin=4 with at least one non-zero local moment. Deliberately independent of lspinorb:
	/// without SOC the spinor Hamiltonian is still complex whenever the moment has a y-component
	/// (H^{up,dn} = B_x - i B_y), so plain conjugation K is not a symmetry there either and the
	/// antiunitary operation must be the full Theta = -i*sigma_y*K. 
	/// Treating the noncollinear no-SOC case with the Shubnikov group is therefore correct (though conservative: 
	/// the exact symmetry there is the larger spin space group, where spin and space rotations decouple).
	bool magnetic_nspin4 = false;

	ModuleBase::Matrix3 symop[48];	//the rotation matrices for the pure bravais lattice
    int nop=0;	//the number of point group operations of the pure bravais lattice without basis
	int nrot=0;	//the number of pure point group rotations
    int nrotk = -1; 	//the number of all space group operations, >0 means the nrotk has been analyzed
    int max_nrotk = -1;  ///< record the maximum number of symmetry operations during cell-relax
    int pgnumber=0;	//the serial number of point group
	int spgnumber=0;	//the serial number of point group in space group
	std::string pgname;	//the Schoenflies name of the point group R in {R|0}
	std::string spgname;	//the Schoenflies name of the point group R in the space group {R|t}

	ModuleBase::Matrix3 optlat;		//the optimized-symmetry lattice
	ModuleBase::Matrix3 plat;		//the primitive lattice

    bool all_mbl = true;    ///< whether all the atoms are movable in all the directions

    int standard_lat(ModuleBase::Vector3<double>& a, 
                     ModuleBase::Vector3<double>& b, 
                     ModuleBase::Vector3<double>& c, 
                     double* celconst)const;

	void lattice_type(ModuleBase::Vector3<double> &v1,
                      ModuleBase::Vector3<double> &v2,
					  ModuleBase::Vector3<double> &v3, 
					  ModuleBase::Vector3<double> &v01, 
                      ModuleBase::Vector3<double> &v02, 
                      ModuleBase::Vector3<double> &v03,
					  double* cel_const, 
                      double* pre_const, 
                      int& real_brav, 
                      std::string& bravname, 
                      const Atom* atoms,
					  bool convert_atoms, 
                      double* newpos = nullptr)const;

	void getgroup(int& nrot, 
			int& nrotk, 
			std::ofstream& ofs_running, 
			const int& nop,
			const ModuleBase::Matrix3* symop, 
			ModuleBase::Matrix3* gmatrix, 
			ModuleBase::Vector3<double>* gtrans,
			double* pos, double* rotpos, int* index, 
			const int ntype, const int itmin_type, const int itmin_start, 
			int* istart, int* na)const;

	bool checksym(const ModuleBase::Matrix3 &s, 
			ModuleBase::Vector3<double>& gtrans,
			double* pos, double* rotpos, int* index, 
			const int itmin_type, const int ntype, const int itmin_start, 
			int* istart, int* na)const;

    /// @brief  primitive cell analysis
    void pricell(double* pos, const Atom* atoms);

	/// -----------------------
	/// Symmetrize the charge density, the forces, and the stress
	/// -----------------------
	void rho_symmetry(double *rho, const int &nr1, const int &nr2, const int &nr3);

	void rhog_symmetry(std::complex<double> *rhogtot, int* ixyz2ipw, const int &nx,
			const int &ny, const int &nz, const int & fftnx, const int &fftny, const int &fftnz);

	/// @brief Symmetrize the nspin=4 (non-collinear/SOC) spin density in reciprocal space.
	/// The three Pauli spin components (rho^x, rho^y, rho^z) are processed TOGETHER because
	/// each symmetry operation g couples the spatial map with a spin rotation W(g):
	///     m_sym(G) = (1/|G|) sum_g W(g) * m(g^{-1} G) * phase(g).
	/// The spatial bookkeeping (grouping/phase) is identical to rhog_symmetry; the only
	/// difference is that the per-g spin rotation W(g) is applied to the 3-vector. `wspin`
	/// is the precomputed array (size nrotk) of spin-rotation matrices, with
	///     wspin[s] = SpinRotation::spin_so3(direct_to_cartesian(gmatrix[s], latvec)),
	/// such that m'^i = sum_j wspin[s]_{ij} m^j under symmetry operation s.
	void rhog_symmetry_soc(std::complex<double>* rhogtot_x, std::complex<double>* rhogtot_y,
			std::complex<double>* rhogtot_z, const ModuleBase::Matrix3* wspin,
			int* ixyz2ipw, const int &nx, const int &ny, const int &nz,
			const int & fftnx, const int &fftny, const int &fftnz);

    /// symmetrize a vector3 with nat elements, which can be forces or variation of atom positions in relax
    void symmetrize_vec3_nat(double* v)const;   // force

    /// symmetrize a 3*3 tensor, which can be stress or variation of unitcell in cell-relax
    void symmetrize_mat3(ModuleBase::matrix& sigma, const Lattice& lat)const; // stress

	//convert n rotation-matrices from sa on basis {a1, a2, a3} to sb on basis {b1, b2, b3}
	void gmatrix_convert(const ModuleBase::Matrix3* sa, ModuleBase::Matrix3* sb, 
			const int n, const ModuleBase::Matrix3 &a, const ModuleBase::Matrix3 &b)const;

	void gmatrix_convert_int(const ModuleBase::Matrix3* sa, ModuleBase::Matrix3* sb, 
			const int n, const ModuleBase::Matrix3 &a, const ModuleBase::Matrix3 &b)const;

	//convert n translation-vectors from va on basis {a1, a2, a3} to vb on basis {b1, b2, b3}
	void gtrans_convert(const ModuleBase::Vector3<double>* va, ModuleBase::Vector3<double>* vb, 
			const int n, const ModuleBase::Matrix3 &a, const ModuleBase::Matrix3 &b)const;

	void gmatrix_invmap(const ModuleBase::Matrix3* s, const int n, int* invmap) const;

	void hermite_normal_form(const ModuleBase::Matrix3 &s, ModuleBase::Matrix3 &H, ModuleBase::Matrix3 &b) const;

    int get_rotated_atom(int isym, int iat)const
    {
        if (!this->isym_rotiat_.empty()) { return this->isym_rotiat_[isym][iat]; }
        else { return -1; }
    }

    /// atom map for the j-th ANTIUNITARY operation (spatial part gmatrix_anti[j]).
    int get_rotated_atom_anti(int j, int iat)const
    {
        if (!this->isym_rotiat_anti_.empty()) { return this->isym_rotiat_anti_[j][iat]; }
        else { return -1; }
    }

	private:

    /// atom-map for each symmetry operation: isym_rotiat[isym][iat]=rotiat
    std::vector<std::vector<int>> isym_rotiat_;

    /// atom-map for each ANTIUNITARY operation: isym_rotiat_anti_[j][iat]=rotiat.
    /// Captured in analyze_magnetic_group_nspin4 before the unitary arrays are compacted.
    std::vector<std::vector<int>> isym_rotiat_anti_;

    /// @brief  set atom map for each symmetry operation
    void set_atom_map(const Atom* atoms);
    /// @brief check if all the atoms are movable
    ///  delta_pos symmetrization in relax is only meaningful when all the atoms are movable in all the directions.
    bool is_all_movable(const Atom* atoms, const Statistics& st)const;

    // to be called in lattice_type
	void get_shortest_latvec(ModuleBase::Vector3<double> &a1, 
			ModuleBase::Vector3<double> &a2, ModuleBase::Vector3<double> &a3)const;

	void get_optlat(ModuleBase::Vector3<double> &v1, ModuleBase::Vector3<double> &v2, 
			ModuleBase::Vector3<double> &v3, ModuleBase::Vector3<double> &w1, 
			ModuleBase::Vector3<double> &w2, ModuleBase::Vector3<double> &w3, 
        int& real_brav, double* cel_const, double* tmp_const)const;

    /// Loop the magmom of each atoms in its type when NSPIN>1. 
    /// If not all the same, primitive cells should not be looped in rhog_symmetry.
    bool magmom_same_check(const Atom* atoms)const;

    /// Analyze magnetic group without time-reversal symmetry
    /// (because currently the charge density symmetrization does not support it)
    /// Method: treat atoms with different magmom as atoms of different type
    void analyze_magnetic_group(const Atom* atoms, const Statistics& st, int& nrot_out, int& nrotk_out);

    /// (nspin=4 / SOC) Restrict the already-built space group to the unitary magnetic
    /// subgroup: keep operation g only if it preserves the magnetization as a pseudovector,
    /// W(g) m_i = m_{g(i)} with W(g)=SpinRotation::spin_so3(gmatc). This prevents operations
    /// that reverse the moment (which are only symmetries when combined with time reversal)
    /// from being applied in k-reduction and density symmetrization.
    /// See ref/2026-07-SOC磁群对称性判据.md. Non-magnetic (m_i=0) keeps all operations.
    void analyze_magnetic_group_nspin4(const Atom* atoms, const Statistics& st, const ModuleBase::Matrix3& latvec);
};
}

#endif
