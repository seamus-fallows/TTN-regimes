# real-data: do the teacher-student TTN phenomena survive on real images? (B5)

Status: working (interim) - Branch: agent/real-data - Updated: 2026-10-10 00:02 UTC

## Summary
**Interim report (steps 1-3 on 4x4 data; 2026-10-10 00:02 UTC).** Gradient flow on MNIST 3-vs-5 (4x4 pixels, cos/sin
features, n_train 4000) with chi = 4 TTNs. The teacher-student picture survives where it follows from
the tree structure. The first plateau scales as sigma^-(K-2) exactly. Bond ranks grow in discrete events.
The second event's plateau scales as predicted by the route degree: a sibling pair of bottom cores on
binary trees (k = 3: gap ∝ sigma^-1.0) and a single edge on the MPS (k = 2: gap ∝ log(1/sigma)). Route
amplitudes computed at the rank-1 plateau predict which sibling pair goes first in 34/35 runs. What
does not carry over: loss plateaus are not flat, and the MPS beats the local binary tree at small sigma.

## Results
Interim: main sweep 80/280 runs (raw features: quadtree 35, MPS 35, scrambled partial), plus the pilot.
Gradient flow (DP5 then LSODA, rtol 1e-8, atol 1e-14), sigma in {0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.5},
seeds 0-4, tau = t - t_a up to 1e4. Numbers from `summarize.py` / `scaling.py` / `tables.py` on
`/root/outputs/real_data/main` (interim summary copied to `/root/outputs/real_data/analysis/main_summary_interim.json`).

| prediction (teacher-student) | real data (interim) | verdict |
|---|---|---|
| first plateau t1 ∝ sigma^-(K-2) (any input law; theory_covariance C4) | per-seed exponent 13.000 ± 0.000 (binary16, 5 seeds), 14.000 ± 0.000 (mps16, 5 seeds); t1 from 1e6 to 1e20 | yes (exact: homogeneity) |
| bond ranks grow stepwise, one route at a time | discrete s_k/s_1 escapes from 1e-5 to O(1) on route edges (figure bond_spectra) | yes |
| binary at rank 1: next route = sibling bottom pair (k = 3) | first rank increase on exactly two sibling bottom edges in 35/35 quadtree runs | yes |
| MPS: next route = one edge (k = 2) | single edge (edge 5) in 35/35 MPS runs, during/just after the first escape | yes |
| k = 3 gap ∝ sigma^-1 | quadtree 2nd drop - 1st drop: seed 0: 298, 209, 143, 100, 69, 45, 29 (sigma 0.05...0.5), exponent 1.01; mean over seeds 1.11 ± 0.13 (n = 5, two seeds switch sibling pair at large sigma); power law rms 0.15 vs log law 0.87 | yes |
| k = 2 gap ∝ log(1/sigma) | MPS 2nd drop: seed 0: 51.0, 44.5, 38.5, 32.2, 25.9, 19.2 (sigma 0.05...0.3): increments 16-19 per unit of ln(1/sigma) | yes (preliminary) |
| later routes k >= 4 give sigma^-(k-2) | quadtree seed 0, 4th drop gap: 5637, 1818, 626, 144, 44 (sigma 0.07...0.3), exponent 3.3 (k = 5 would give 3) | partly (one seed; systematic table in progress) |
| route with smallest k first, ties by amplitude | largest sibling amplitude predicts the first sibling pair in 34/35 quadtree runs; the largest amplitudes overall sit on k = 5, 7 routes, which escape later | yes |
| flat plateaus at partial fits (S2S signature) | plateaus creep (loss keeps decreasing slowly); with rate-peak drops: quadtree S2S fraction 1.0 at sigma <= 0.07 (min rho median 22 at sigma = 0.05); MPS mostly overlapping drops | partly |
| locality: quadtree faster than scrambled, MPS in between | scrambled stalls on long plateaus (pilot: L ≈ 0.29 to tau = 3000 at sigma = 0.2 vs 0.10 quadtree); but the MPS is FASTEST at small sigma (log-time routes) and the quadtree slows as sigma^-1, sigma^-3 (at tau = 1e4, sigma = 0.05: quadtree L = 0.18, acc 0.89; MPS L = 0.090, acc 0.956) | partly: quadtree beats scrambled; MPS vs binary is set by route degree, not locality |

Figures (interim, `experiments/real_data/results/`): `loss_curves_raw`, `bond_spectra_raw_s{0.05,0.1,0.5}`,
`growth_order_raw_s*`, `locality_raw_s*`, `scaling_raw`, `route_amplitudes`.

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
- **Reduced grids for robustness variants** (decided mid-sweep, ~1 h saved): scrambled1/2 at
  sigma in {0.05, 0.1, 0.2, 0.5}, whitened features at {0.05, 0.07, 0.1, 0.2, 0.5}; the full 7-value grid is
  kept for raw quadtree / MPS / scrambled0 (the scaling claims). Implemented by placing empty placeholder
  files for the dropped cases (the runner skips existing outputs; list in
  /root/outputs/real_data/main_placeholders.txt; removed after the sweep).
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
- 00:02 interim report: 80/280 main runs; t1 exponents exact; k = 3 sigma^-1 and k = 2 log gaps; route amplitudes 34/35; regime sweep relaunched at sigma >= 0.5
- 00:45 main sweep 124/220 runs; regime sweep 9/35 (sigma 0.5-1: not lazy, kernel scale 1e-13..3e-5 at init: raw features are small-init in function space up to sigma ~ 1); plateau Hessians and route amplitudes running; chi8 + fashion queued after main
