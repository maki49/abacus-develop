# Finite virtual-window response in TDA gradients

The real Gamma-point gradient paths include the response of the canonical
virtual subspace selected by the LR window, for both spin-adapted and
spin-unrestricted TDA. The correction differentiates the energy of the
selected window; it does not replace it with a full-window energy.

Retain all occupied orbitals and compute enough ground-state bands for the
response. The Z equation spans every available virtual band. Response of
omitted occupied orbitals or ground-state bands that were never computed is
not recovered. Fractional occupations are not covered.

## Window multiplier

For each unrestricted spin, let $a$ be a retained virtual, $r$ a discarded
virtual, and $i$ an occupied orbital. Zero-pad the amplitudes into the Z
space and apply the coupled Casida operator once. Its outside-window action
is the residual $K^{\mathrm{es}}_{ri\sigma}[D^X]$. Then

$$
W^{\mathrm{win}}_{ra\sigma}
=\frac{2\sum_iX_{ai\sigma}(AX)_{ri\sigma}}
       {\epsilon_{a\sigma}-\epsilon_{r\sigma}}.
$$

The denominator is spin diagonal; the residual contains both spin sources.
Use the jointly normalized unrestricted amplitude, without normalizing the
two spin blocks separately. The transition density remains nonsymmetric.

Define the symmetric window density

$$
D^{\mathrm{win}}_\sigma
=\mathrm{sym}(C_{R\sigma}W^{\mathrm{win}}_\sigma C_{A\sigma}^{\mathsf T}).
$$

Add $-H^{\mathrm{gs}}[D^{\mathrm{win}}]$ to the Z right-hand side, retaining
the ground-state Hessian even if the excitation kernel is different.
The final force contracts $T+D^Z+D^{\mathrm{win}}$ with the ground-state
Hamiltonian derivative. In addition to evaluating the existing energy
density with the updated Z, add

$$
\Delta\Lambda_\sigma
=\mathrm{sym}\!\left(
 \sum_{ra}c_{r\sigma}\epsilon_{a\sigma}W^{\mathrm{win}}_{ra\sigma}
 c_{a\sigma}^{\mathsf T}\right)
+C_{o\sigma}\left(\tfrac12H^{\mathrm{gs}}_{oo\sigma}
 [D^{\mathrm{win}}]\right)C_{o\sigma}^{\mathsf T}.
$$

The occupied-metric contribution is required along with the direct
energy-weighted window density.

## Implementation and limits

The LR module owns the per-state window densities and direct energy-weighted
densities. ESolver constructs them before solving Z. The Z right-hand side
merges the window density into the existing difference density and evaluates
$-H^{\mathrm{gs}}[T+D^{\mathrm{win}}]$ with the existing operators. The occupied
multiplier reuses the relaxed density $T+D^Z+D^{\mathrm{win}}$ already assembled
for the final force. Thus the occupied-metric contribution in the equation above
is included through the existing multiplier calculation and must not be added
again. Only the direct $\epsilon_a W^{\mathrm{win}}_{ra}$ energy-weighted density
is added separately; no separate Hessian application to the window density is
needed. The full-space Casida action used to construct the window multiplier is
retained.

In the unrestricted path, local Hartree/xc actions sum over all four input/output
spin blocks; exact exchange acts only on equal-spin blocks. The merged density
uses the ground-state kernel even when the LR functional differs, and leaves the
transition-density and third-order xc terms unchanged.

For a symmetric source, the unrestricted local-kernel factors are -2 for the
Z source and 1 for the occupied multiplier. The spin-adapted factors are -4
and 2 because that path uses a different local-kernel normalization.

A full virtual window returns an empty correction. A boundary gap smaller
than $10^{-8}$ Ry raises an error: expand the window to retain the complete
degenerate orbital group rather than clipping the denominator. Complex
truncated-window gradients are not implemented. The correction is quadratic
in the amplitudes and also applies to the combinations used to construct
degenerate-state gradient matrices.

The focused MODULE_LR_window_response test covers a moving generalized
eigenproblem, full-window cancellation, a degenerate boundary, amplitude
quadraticity, and a coupled-spin finite difference with joint normalization.
Nuclear finite differences must use separately converged ground states at
each displaced geometry and the same excitation window as the analytic force.
