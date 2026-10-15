# HSE06 component-clamping experiment (2026-10-14)

This branch preserves an experiment that is **not included in PR #8069**.
The data below are a snapshot of seven completed full-window cases; three
additional cases were still running or pending when this snapshot was written.
These are diagnostic calculations, not integration-test reference updates.

## Scope

The production safeguard floors all sigma invariants passed to the combined
HSE06 functional. This also changes its PBE correlation inputs, including signed
mixed-spin sigma. That is a known limitation, not a physically justified
correlation regularization.

The experiment expands only HSE06's already initialized LibXC components,
retaining their pointers, weights and screening parameters. Only the two WPBEH
exchange components keep the existing 1e-6 floor on diagonal sigma. Correlation
receives raw sigma, and signed mixed-spin sigma is preserved. VXC, FXC and KXC
are accumulated with the initialized component weights. The parent masking
policy is retained. Independent PBE, EXX and GS calculations are unchanged.

This does not make the exchange floor an energy/derivative-consistent
regularization. Component isolation alone is not a completed HSE force fix.

## Configuration and comparison

Both versions use LibXC 5.2.3 with KXC, two MPI ranks, and one OpenMP/BLAS thread.
Original full-window sweep inputs are retained, including 100 Ry cutoff,
per-case SCF settings, windows and solvers. At each geometry both versions share
the same freshly computed GS. Each version has its own new finite difference
of its excitation energy: atom index 1, z displacement +/-0.0005 Angstrom.
FD endpoints omit force calculations. The displayed forces are -dOmega/dR,
without the GS force.

Errors compare multiplet means using the previous sweep's grouping convention
(2e-3 Ry), excluding the final group because it may be incomplete at the
requested-root boundary. MAE averages retained group errors; Max/scale divides
the maximum retained error by the largest absolute analytic/FD force over all
roots, including the boundary group. Nonpolarized cases include singlets only.

| System | Max absolute excitation-energy change (eV) | Before MAE (eV/Angstrom) | After MAE (eV/Angstrom) | Before Max/scale | After Max/scale |
|---|---:|---:|---:|---:|---:|
| 02_Li2 | 0.0619479 | 0.02206132 | 0.04343897 | 3.491044% | 5.816603% |
| 01_Si | 0.0006305 | 0.0003779477 | 0.0006979599 | 0.02268434% | 0.04150005% |
| 03_C_diamond | 2e-07 | 0.005272094 | 0.005272075 | 0.08467557% | 0.08467537% |
| 04_hBN | 1.9e-05 | 0.001502652 | 0.0009454134 | 0.04594744% | 0.03300553% |
| 05_BH | 0.001237 | 0.002808594 | 0.003346718 | 0.03016422% | 0.0378886% |
| 06_N2 | 4.45e-05 | 0.0003594958 | 0.0004079269 | 0.003225279% | 0.004104868% |
| 07_LiF | 0.010168 | 0.07879367 | 0.2200816 | 1.766818% | 8.249134% |

All completed stages returned successfully without SCF/Z nonconvergence.
This does not establish step-size convergence of the FD itself.

## Important individual checks

For Li2, first-singlet analytic/central-FD forces change from
0.30192387/0.29820000 to 0.16585666/0.24020000 eV/Angstrom. The isolated model's
left/right FD slopes are 0.2398/0.2406 eV/Angstrom, both substantially separated
from its analytic force. Its first excitation energy changes by +0.0619479 eV.

For LiF, the first-singlet-doublet mean error changes from 0.09678863 to
0.45172256 eV/Angstrom. However, the isolated model's first-root left/right FD
slopes differ considerably (1.8366/2.1648 eV/Angstrom). This indicates significant
finite-step sensitivity; its entire discrepancy must not be attributed to an
analytic derivative error without a smaller-step check. Doublet means avoid
comparing arbitrary individual degenerate eigenvectors.

## Decision and limits

The experiment removes unintended correlation-input modification, but does not
show a general improvement in force/energy derivative agreement. It is therefore
kept on this separate branch rather than introduced as a completed production
fix in PR #8069. The lower errors of the original safeguard do not prove that
its correlation modification is correct; cancellation of errors is possible
but has not been established. A consistent derivative of the actually
regularized kernel remains follow-up work. The current PR retains its existing
approximate HSE safeguard and its limitations; this review concern is not
claimed fully resolved.

## Verification

- GNU/MPI executable built with the experiment.
- Component tests cover initialized ownership/weights, independent PBE/PBE0,
  raw component reconstruction and all nine VXC/FXC/KXC derivative arrays at
  well-conditioned inputs. These do not validate the floor as a differentiable
  energy model.
- Sigma tests cover signed mixed-spin invariants and diagonal-only floors.
- Kernel-binding tests cover interleaved functional construction.
- Independent PBE comparison preserved printed excitation energies and all 36
  signed force components.
- The seven full-window comparisons above use new GS, LR and version-specific
  FD endpoints, not archived FD values for a different model.
