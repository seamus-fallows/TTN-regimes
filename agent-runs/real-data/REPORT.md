# real-data: do the teacher-student TTN phenomena survive on real images? (B5)

Status: working - Branch: agent/real-data - Updated: 2026-10-09 21:19 UTC

## Summary
In progress. Task: train TTNs (quadtree binary, snake MPS, scrambled binary) by gradient flow on
downsampled MNIST with cos/sin site features, and test plateaus, stepwise bond growth, the locality
rule and the regime map.

## Results
Interim (pilot, seed 0 only; main sweep relaunched):
- Quadtree binary, raw features, sigma = 0.2: loss drops at tau = 15, 84, 225, 369 (rho = 7, 15, 16):
  saddle-to-saddle staircase; first rank growth on the sibling bottom edges (3, 4) (a k = 3 route) in all
  three sigma; gap to the 2nd drop 29, 45, 69 for sigma = 0.5, 0.3, 0.2 (~ A/sigma + B, k = 3).
- Scrambled binary: same mechanism (first rank growth on sibling edges 10, 11), but after it the loss
  stalls at ~0.29-0.33 for thousands of time units (test acc 0.81 vs 0.95 for the quadtree at sigma = 0.2).
- MPS (snake): one edge (5) gains rank during the first escape itself; later decrease smooth (rho <= 3.5).
- Route amplitudes predict the first sibling pair to grow in 6/6 binary pilot runs; the largest amplitudes
  belong to longer routes (k = 5, 7), which escape later: degree beats amplitude on real data.

## What was done
1. Reverted the launch commit b622158 (in-progress library edits), `uv sync`, pushed (98042d5).
2. Data download + downsampling script `experiments/real_data/download.py` (MNIST, Fashion-MNIST; 4x4 and 8x8).
3. Problem builder `data.py`, two-phase GF runner `run.py`, analysis module `analysis.py`; merged main (39ca8d7).

## Decisions
- **Task: MNIST 3 vs 5 (+-1), n_train = 4000, n_test = 1600, 4x4** (central 24x24 crop, 6x6 block means).
  Options: MNIST/Fashion, one-vs-rest or a pair. Chose a classic confusable, balanced pair, fixed a priori
  (not tuned to make locality look good); Fashion-MNIST is kept as a possible robustness check.
- **Features "white" = per-site whitening with the uncentred second moment** (floor 1e-3 of the top
  eigenvalue). Centring changes the function class (span{cos, sin} does not contain constants), so the
  two variants are raw and whitened; this keeps the network's function class identical.
- **Bond spectra on real data are data-weighted** (`data.function_spectra`: Schmidt values w.r.t. the
  empirical marginals of the two sides of each edge, from per-sample messages). Reason: after the escape
  the Frobenius norm of W grows to ~1e6-1e7 along directions the data barely sees (probe, sigma = 0.5),
  so Frobenius spectra say nothing about the learned function. Checked: equal to the Frobenius spectra
  to 2e-12 on an exactly enumerated product measure with identity second moments.
- **Event references = the run's own final spectra** (no teacher), learned if s_k(T) >= 0.02 s_1(T).
- **Integrator: two phases.** DP5 through the first plateau (cheap, O(log) steps), then LSODA on its own
  clock tau = t - t_a (t_a ~ sigma^-(K-2) up to 1e15: absolute times cannot resolve later gaps in
  float64). After the escape the flow is stiff (lambda_max ~ 57): DP5 would need ~1e9 steps; GD likewise.
  LSODA's Newton matrix: Gauss-Newton part J^T J / n (per-sample Jacobians via vmap, 0.1-1 s) instead of
  autograd's exact Hessian (20 s). The error control is unchanged; GN was faster in wall time than the
  exact Hessian (also implemented, dense residual tensor, 1.2 s) on the probe.
- **Loss drops on real data = peaks of the loss rate -dL/dt** (FWHM width, valley factor 3, min size 1% of
  L(0); `analysis.loss_drops`), not `find_plateaus`' progress widths. Reason: between learning events the
  loss keeps creeping down (ill-conditioned directions within the current bond-rank profile), so a
  10%->90%-of-drop width measures the tail (widths of ~900 for a drop whose rate peak is ~17 wide).
  Separation ratios rho and the S2S rule (i)-(ii) are then applied as in notes/diagnostics.md.
- **Spectral events = s_k/s_1 on an edge crossing 0.05** (log-interpolated), used for *which* edges gain
  rank in each loss drop; widths from final-value references are not used (the references creep).
- **Phase 1 restarts its clock in chunks** (adaptive, ~0.3 x the remaining time estimated from the speed's
  growth): at sigma <= 0.1 the escape happens at t ~ 1e15-1e19 where O(1) steps underflow (all 41 runs at
  sigma <= 0.1 of the first main launch failed this way; relaunched).
- **Route amplitudes** (`route_amplitudes.py`): at the bond-rank-1 plateau of a binary run,
  lambda_uw = |P_u M_uw P_w| with M_uw = (1/n) sum_j r_j h_uw(x_j) Phi_u Phi_w^T, for every pair of bottom
  cores; the route degree is the Steiner size of (u, w). Used to test "lowest k first, then largest
  amplitude" and to see where locality enters.
- **Parallelism: ~60 processes, not 126.** The CPU is 64 physical cores (SMT 2) and these runs are
  memory-bandwidth bound: at 126 processes one gradient took 127 ms (14 ms idle), at ~60 it takes 38 ms.
  The per-sample Jacobian is now a closed form from per-sample messages (0.4 s vs 27 s for vmap under load;
  equal to 4e-18). The first two launches of the sweeps were killed (no valid run lost: all finished ones
  had failed in phase 1).
- **Merged main (34c76ad) at the lead's request**; event times use `merged_rows` (log + fine rows; fine rows
  also carry the data-weighted spectra).

## Files left on the machine
- `/root/outputs/real_data/data/` (~25 MB): raw IDX files and downsampled npz (regenerate with download.py).

## Caveats and next steps
(pending)

## Log
- 21:19 reverted launch commit, read brief/notes/library, downloaded data, launched cost pilot
- 21:37 cost probes: post-escape stiffness (lambda_max ~ 57); LSODA + Gauss-Newton Jacobian chosen; function-space spectra added; merged main; pilot (18 runs) launched
- 22:08 pilot analysed (staircases on quadtree binary, route amplitudes predict first route); main sweep failed at sigma <= 0.1 (clock underflow) -> chunked phase 1; merged main 6c42d85 (gauge fix; no lambda_min reported yet)
- 22:59 phase-1 clock re-anchoring (escape at t ~ 1e18-1e20 now resolved); closed-form Jacobian; sweeps relaunched with 48 + 12 workers (machine memory-bound)
