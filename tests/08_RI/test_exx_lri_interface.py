#!/usr/bin/env python3
"""Compare two-step diamond DDH cell-relax against a fresh DDH single point.

Requires C_ONCV_PBE-1.0.upf and Orb-v2's
6_C_DZP/C_gga_7au_100Ry_2s2p1d.orb in the supplied data directories.
Run after loading the binary's MPI/library environment, for example:
  python3 tests/08_RI/test_exx_lri_interface.py --binary build/abacus_std_para \
    --pseudo-dir /path/to/SG15_ONCV_v1.0_upf --orbital-dir /path/to/Orb-v2 \
    --work-dir /tmp/exx-cell-regression
Both separate and simultaneous EXX loops are exercised. A common FFT grid
removes grid-selection differences between the continuous and fresh runs.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess


STRUCTURE = """ATOMIC_SPECIES
C 12.011 C_ONCV_PBE-1.0.upf
NUMERICAL_ORBITAL
6_C_DZP/C_gga_7au_100Ry_2s2p1d.orb
LATTICE_CONSTANT
1.8897268777743552
LATTICE_VECTORS
0 1.7440225 1.7440225
1.7440225 0 1.7440225
1.7440225 1.7440225 0
ATOMIC_POSITIONS
Direct
C
0.0
2
0 0 0 1 1 1
0.25 0.25 0.25 1 1 1
"""
KPOINTS = "K_POINTS\n0\nGamma\n2 2 2 0 0 0\n"
COMMON = """basis_type lcao
gamma_only 0
symmetry 1
ecutwfc 100
nx 36
ny 36
nz 36
scf_thr 1e-9
scf_nmax 200
nspin 1
smearing_method gaussian
smearing_sigma 0.002
dft_functional pbe0
exx_fock_alpha 0.18
exx_singularity_correction spencer
cal_force 1
cal_stress 1
"""
RELAX = """calculation cell-relax
relax_nmax 2
force_thr_ev 0.001
stress_thr 0.1
fixed_axes shape
relax_scale_force 1.0
"""


def run_case(args, directory, structure, calculation, separate_loop):
    directory.mkdir()
    settings = ("INPUT_PARAMETERS\n"
                f"pseudo_dir {args.pseudo_dir}\n"
                f"orbital_dir {args.orbital_dir}\n"
                f"exx_separate_loop {separate_loop}\n")
    (directory / "INPUT").write_text(settings + COMMON + calculation)
    (directory / "STRU").write_text(structure)
    (directory / "KPT").write_text(KPOINTS)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [args.mpi_launcher, "-np", str(args.mpi_ranks), str(args.binary)]
    with (directory / "run.log").open("w") as output:
        subprocess.run(command, cwd=directory, env=environment,
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    log_name = "running_cell-relax.log" if "cell-relax" in calculation else "running_scf.log"
    log = (directory / "OUT.ABACUS" / log_name).read_text()
    if re.search(r"\b(?:nan|inf)\b", log, re.IGNORECASE):
        raise AssertionError(f"Non-finite result in {directory}")
    return log


def verify_mode(args, separate_loop):
    relax_dir = args.work_dir / f"relax-loop{separate_loop}"
    log = run_case(args, relax_dir, STRUCTURE, RELAX, separate_loop)
    if log.count("#SCF IS CONVERGED#") != 2:
        raise AssertionError("Both ionic steps must converge their electronic SCF")
    checkpoint = (relax_dir / "OUT.ABACUS/STRU_NOW").read_text()
    match = re.search(r"# RELAX STEP 2, Energy: ([-+0-9.eE]+) eV", checkpoint)
    if match is None:
        raise AssertionError("STRU_NOW must describe the evaluated second step")
    energy_relax = float(match.group(1))
    second_step = log.split("RELAX STEP: 2", 1)[1]
    exchange = re.search(r"E_exx\s+[-+0-9.eE]+\s+([-+0-9.eE]+)", second_step)
    if exchange is None or abs(float(exchange.group(1))) < 1e-8:
        raise AssertionError("Second-step hybrid Hamiltonian is missing its EXX energy")
    single_dir = args.work_dir / f"single-loop{separate_loop}"
    single_log = run_case(args, single_dir, checkpoint, "calculation scf\n", separate_loop)
    if single_log.count("#SCF IS CONVERGED#") != 1:
        raise AssertionError("Independent DDH single point must converge")
    energy_single = float(re.search(r"!FINAL_ETOT_IS\s+([-+0-9.eE]+)", single_log).group(1))
    delta = energy_relax - energy_single
    result = dict(separate_loop=separate_loop, energy_relax_ev=energy_relax,
                  energy_single_ev=energy_single, delta_ev=delta,
                  first_outer_exchange_ev=float(exchange.group(1)))
    print(json.dumps(result), flush=True)
    if not math.isfinite(delta) or abs(delta) > 1e-5:
        raise AssertionError(f"Second-step DDH energy mismatch: {delta:.12g} eV")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--pseudo-dir", type=Path, required=True)
    parser.add_argument("--orbital-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--mpi-launcher", default="mpirun")
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--separate-loop", type=int, choices=[0, 1], nargs="+", default=[1, 0])
    args = parser.parse_args()
    for name in ["binary", "pseudo_dir", "orbital_dir", "work_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.work_dir.mkdir(parents=True)
    results = [verify_mode(args, mode) for mode in args.separate_loop]
    (args.work_dir / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
