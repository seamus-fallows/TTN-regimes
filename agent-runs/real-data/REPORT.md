# real-data: do the teacher-student TTN phenomena survive on real images? (B5)

> **Addendum (fix round 1, local, 2026-10-10; branch wip/real-data-fix).** After the round-1 review
> (`notes/reviews/real_data_round1.md`) several statements below are corrected in `notes/real_data.md`
> section 8, which supersedes this report where they differ. In short: the k = 3 first-route gap had an
> attribution bug (corrected: A/sigma + B fits all 11 clean series, free exponent 1.22 ± 0.03, not "1.01");
> both gap laws are now fitted with relative residuals (log-law rms 0.11-0.30, still rejected); the
> "locality" result is narrowed to a pixel-order/orientation effect measured by route amplitudes (equally
> local orders such as the transposed quadtree are not near the best: ranks 13-21 of 27 in the pre-registered
> test); "the MPS is fastest at every sigma / fails
> on every task" is narrowed to the tau = t - t_a clock on MNIST with the snake order (slower on the
> absolute clock, slowest on Fashion, shape and order confounded); the MPS first route is merged with the
> escape (10/35 rank events are the threshold at tau = 0), so its verdict is "partly"; the regime map's
> "rich" rests on one sigma, the lazy runs are unconverged and the lazy floor is not above feature learning
> for every seed; counts over sigma are de-duplicated into critical-point units; chi = 8 and Fashion
> exponents that no committed output produces are withdrawn. **Round 2** (raw outputs local): every pending
> item was rerun with the fixed code (first-route fits over all 124 binary runs: 1.20 ± 0.03, A/sigma + B with
> B < 0 in 10/10; with fair fits 6 of 9 later MPS edges rise in log time, so the review's "2 of 9" was a fit
> artifact), and a pre-registered order test was run (`notes/real_data.md` section 9) over 27 pixel
> orders: a data-only route-amplitude share predicts the final loss (rho = -0.60 [-0.82, -0.22]) but not the time
> to reach L <= 0.25. **Round-2 verification** (`notes/reviews/real_data_round2_verification.md`): the
> pre-registration is intact and the main numbers reproduce; its errata E1-E9 are applied to the notes, the README
> and this report, whose Summary, table and Caveats below now state the narrowed claims (the round-0 versions are
> in the git history).

Status: done - Branch: agent/real-data - Updated: 2026-10-10 04:25 UTC

## Summary
Task: train TTNs by gradient flow on real images (MNIST 3 vs 5, 4x4 pixels, cos/sin features, chi = 4) and
test the teacher-student predictions. Main sweep: 234 runs; quadtree binary, snake MPS and 3 scrambled
binary trees; raw and whitened features; sigma 0.05-0.5; 5 seeds. Plus a 65-run regime sweep and checks
on Fashion-MNIST, chi = 8 and 8x8 images (102 runs). Findings: `notes/real_data.md` (section 5 = table).

The structural predictions survive where they follow from the tree structure; several round-0 claims were
narrowed in the fix rounds (`notes/real_data.md` section 8 and the round-2 verification errata). The first
plateau scales as sigma^-(K-2) exactly, an identity of small-init flow (an integrator check, not evidence).
Ranks grow in discrete route escapes. On binary trees the first route after the escape is a sibling pair of
bottom cores (k = 3) in all 20 (order, seed) pairs; its gap follows A/sigma + B with raw features (free
exponent 1.20-1.22, not robust) and the log law is rejected. On the MPS the first rank rise (one edge) is
merged with the escape, and 6 of 9 later edges rise in log time (28 of 29 seed fits). The MPS plateau after
the first route is a strict saddle, the binary rank-1 plateau a higher-order one (lambda_min exponent
0.87-0.89 on three seeds, 1.15 on a 3-point seed). Route amplitudes predict which sibling pair goes first in
about 80-85% of critical-point units (chance 25%). Regimes: S2S at small sigma, rich at one sigma (1.4), lazy
from sigma = 2, where the runs follow the linearized model but are unconverged at their horizon; the
boundaries depend on the thresholds.

Pixel order (not locality as such): at small sigma the quadtree order ends lowest at every seed, but equally
local orders rank only 13-21 of 27 and early learning is not faster. In a pre-registered test over 27 orders a
data-only route-amplitude share predicts the final loss (Spearman -0.60 [-0.82, -0.22]) but not the time to
L <= 0.25; that the order acts through short vs long routes is a conjecture. MPS vs binary: the snake MPS
reaches each loss level first only on the tau = t - t_a clock on MNIST; on the absolute clock its first escape
is later (4 of 5 seeds at sigma = 0.1, 5 of 5 at sigma <= 0.07), and on Fashion it is the slowest tree to
L <= 0.3-0.15 (the only one to reach L <= 0.12 early). Tree shape and pixel order are confounded (snake chains
only), so the brief's "MPS between quadtree and scrambled" is undetermined, not refuted. Loss plateaus are not
flat on real data: between drops the loss falls by a median 0.76 of the next drop's size.

## Results
Full table of predictions vs observations: `notes/real_data.md` section 5 (short version below, updated to
the fix rounds and the round-2 verification; the round-0 table is in the git history). Counts are per
(order, seed) or per critical-point unit, not per run. Gradient
flow (DP5 to the escape, then LSODA, rtol 1e-8, atol 1e-14), MNIST 3 vs 5 4x4, n_train 4000, chi = 4, raw
features unless stated; sigma = 0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.5; seeds 0-4. Produced by
`summarize.py` -> `scaling.py`, `tables.py`, `events_table.py`, `route_amplitudes.py`, `plateau_hessians.py`,
`regime_analysis.py`, `make_figures.py`, then `fix1_analysis.py` and the order test (README "How to run");
outputs in `experiments/real_data/results/`.

| prediction | observed | verdict |
|---|---|---|
| t1 ∝ sigma^-(K-2) | 13.000 / 13.000 / 14.000 per seed (quadtree, scrambled, MPS): an identity of small-init flow, so an integrator check | identity (proven) |
| ranks grow one route at a time | discrete escapes of s_k/s_1 from 1e-5-1e-3 to O(1); one negative Hessian direction at most binary small-sigma plateaus | yes |
| flat plateaus | plateaus creep (between drops the loss falls by a median 0.76 of the next drop); S2S signature holds with rate-peak drops | partly |
| first route on binary trees: sibling pair (k = 3) | all 20 (order, seed) pairs; the two siblings cross together in 96/96 rank-1 states | yes |
| MPS: first route on one edge (k = 2) | edge 5 alone in 25/25 resolved runs, but merged with the escape (before the first drop's peak); 10/35 recorded rises are the threshold at tau = 0 | partly |
| k = 3 gap ∝ sigma^-1 | raw features: A/sigma + B fits all 11 clean and all 10 window-checked series (B < 0); free exponent 1.22 ± 0.03 / 1.20 ± 0.03 (not robust); log law rms 0.11-0.30; the one whitened series has B = +10.8 | yes, as A/sigma + B |
| k = 2 gap ∝ log(1/sigma) | later MPS edges 3, 4, 6-9 prefer the log law (28/29 seed fits; edge 4 slope 14.7 ± 1.5); edges 1, 2 power-like, edge 0 mixed | yes for 6 of 9 edges |
| strict iff k = 2 (lambda_min vs sigma) | MPS plateau after the first route: exponent 0.047 ± 0.016 (strict); binary rank-1 plateau 0.87-0.89 on three seeds, 1.15 on one 3-point seed (mean 0.95 ± 0.07; higher order, predicted 1) | yes |
| route choice: lowest k, then largest amplitude | 22/26 critical-point units (19-22 of 23-26 with other groupings; chance 1/4); 11/11 review runs (10 new orders or seeds) | yes |
| locality: local pixel orders learn faster | quadtree lowest final loss at every seed (sigma <= 0.14) but not faster early; equally local orders rank 13-21 of 27; pre-registered: a data-only share predicts the final loss over 27 orders (rho -0.60 [-0.82, -0.22]), not the time to L <= 0.25 | no as locality; an order effect (final loss) |
| MPS between quadtree and scrambled | snake MPS first on the tau clock (MNIST, e.g. L <= 0.2 at tau = 56 vs 333 quadtree at sigma = 0.1); later on the absolute clock (4/5 seeds at sigma = 0.1); slowest on Fashion to L <= 0.3-0.15; shape and order confounded | undetermined |
| regimes S2S -> rich -> lazy | S2S at small sigma; rich at sigma = 1.4 only; lazy (>= 2) = linearized model, kernel rank 304 = R2 formula; boundaries threshold-dependent; lazy runs unconverged; lazy floor not above feature learning for every seed | partly |
| (Fashion-MNIST sneaker vs boot) same structure | t1 exact; first two rising edges siblings 15/15, 14/15, but rising together only 3/15, 2/15; a rank-1 strict saddle (lambda_min ∝ sigma^0.03±0.09) left by a rank-preserving rotation; no order advantage (task ~solved at rank 1) | partly / order: no |
| (chi = 8) same structure | t1 exact; sibling first route 14/15, 14/15; 23/45 runs wall-limited (losses at tau_end; MPS lowest L(tau_end) at small sigma); first-route exponents withdrawn (two-point slopes 1.27 / 1.20) | yes (structure) |
| (8x8 images, N = 64) same structure | t1 exponents 61 / 62 (two-point identity); sibling first route 4/4, 4/4; MPS single edge 4/4; one permutation, two seeds, tau = 3000 | yes (small) |
| integrator accuracy | drift <= 4.6e-8 (234 runs); 100x tighter tolerances: drop times within 0.1-0.9 units, L(T) to 6 digits | ok |

Figures (`experiments/real_data/results/`): `loss_curves_raw` (train/test per tree and sigma),
`bond_spectra_raw_s{0.05,0.1,0.5}`, `growth_order_raw_s*`, `scaling_raw` (t1, k = 3 gaps, MPS log law),
`plateau_hessians`, `route_amplitudes`, `locality_raw_s{0.05,0.1,0.5}`, `regime_sweep`; fix rounds:
`first_route_gaps`, `scaling_fix1`, `locality_orders`, `order_test`. Tables:
`tables.md`, `events_table.md`, `scaling.json`, `regime_analysis.json`, `routes_raw.json`, `hessians_raw.json`;
fix rounds: `fix1_analysis.md`, `order_test.md`, `extra_orders_summary.csv`, `extra_orders_predictor.json`.

## What was done
1. Reverted the launch commit b622158, `uv sync`, pushed (98042d5); merged main twice (34c76ad library final,
   6c42d85 gauge fix).
2. Data: `download.py` (MNIST / Fashion-MNIST IDX files -> 4x4, 8x8). Problem builder `data.py` (pixel orders,
   features, trees, fast Jacobians, data-weighted spectra). Runner `run.py` (two-phase GF, re-anchored clock).
3. Pilot (18 runs) -> design: data-weighted spectra, rate-peak loss drops, horizon 1e4 (commits 0209996-cc66b04).
4. Main sweep: 234 runs (`configs/main.yaml`); two failed launches before it (clock underflow; memory-bound
   oversubscription), see Decisions. Interim report bb33029.
5. Analyses: per-run summaries, classified scaling fits, route amplitudes (124 runs), plateau Hessians (67 runs),
   tables and figures (commits after bb33029).
6. Regime sweep: 35 centred runs (`configs/regime.yaml`) + 30 plain runs; linearized reference.
7. Robustness: Fashion-MNIST sneaker vs ankle boot (45 runs), chi = 8 (45 runs, two runners), 8x8 images (12 runs),
   tolerance check (2 runs at 100x tighter tolerances), Fashion plateau Hessians (30 runs).

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
- **Second dataset: Fashion-MNIST sneaker (7) vs ankle boot (9)**, chosen a priori for a visible shape
  difference at 4x4 (T-shirt vs shirt looked near-indistinguishable at that resolution). Not screened for
  locality.
- **chi = 8 sweep split across two runners** (`chi8.yaml` cases 1-24, `chi8b.yaml` cases 25-45 into
  /root/outputs/real_data/chi8b, placeholders in chi8/ removed afterwards) to use the cores freed by the
  Fashion sweep; same cases, same code.
- **Merged main (34c76ad) at the lead's request**; event times use `merged_rows` (log + fine rows; fine rows
  also carry the data-weighted spectra).

## Files left on the machine
All under `/root/outputs/real_data/` (~0.6 GB; none in git):
| path | size | what | needed to reproduce |
|---|---|---|---|
| `data/` | 65 MB | raw IDX files + 4x4 / 8x8 npz (`download.py`) | yes (or re-download) |
| `main/` | 367 MB | 234 trajectories of the main sweep (npz: phase-1/2 rows, spectra, theta at phase ends) + `runs.jsonl` | yes for every main-sweep number |
| `regime/` | 68 MB | 35 centred-model trajectories | yes for the regime map |
| `fashion/` (45), `chi8/` + `chi8b/` (24 + 21; `chi8_runs/` = symlinks to both), `img8/` (12), `tol_check/` (2) | ~230 MB | robustness sweeps | for section 7 of the notes |
| `analysis/` | 2 MB | summaries, route amplitudes, Hessians, regime analysis (JSON; copies of the small ones in `experiments/real_data/results/`) | regenerable from the runs |
| `pilot/`, `smoke*/` | 38 MB | development runs (pre-final settings) | no |
| `main_placeholders.txt`, `old/` | - | list of trimmed cases; logs of the killed launches | no |

## Caveats and next steps
- **Scope.** Exponent fits come from one task (MNIST 3 vs 5, 4x4, chi = 4, raw features). Fashion-MNIST, chi = 8 and
  8x8 runs confirm the structure (t1, first routes, the pixel-order direction on MNIST) but have 2-3 sigma values
  only.
- **The pixel-order effect is task dependent.** On MNIST 3 vs 5 at small sigma the pixel order changes the final
  loss, and a data-only route-amplitude share predicts the ranking of 27 orders (pre-registered); on Fashion
  sneaker vs boot, which a rank-1 (product) function almost solves, there is no order advantage. Locality in the
  image is neither necessary nor sufficient (equally local orders rank 13-21 of 27), and that the order acts
  through residual interactions on short vs long routes is a conjecture. The brief's "MPS in between" is
  undetermined, not refuted: tree shape and pixel order are confounded (snake chains only); the snake MPS is
  fastest only on the tau clock on MNIST, later on the absolute clock and slowest on Fashion to L <= 0.3-0.15;
  that its 2-core (log-time) routes make it fast is a conjecture.
- **Adapted diagnostics.** No teacher: events are loss-rate peaks and s_k/s_1 = 0.05 crossings of data-weighted
  spectra. Plateaus creep, so the notes' progress-based widths were replaced by rate-peak FWHM (documented in
  `notes/real_data.md` section 3). Later events can merge; routes of later events are only partly identified.
- **Horizon.** tau = 1e4 cuts late (k >= 5) escapes at sigma < 0.1, which lowers the small-sigma losses and
  accuracies at fixed time. 23/45 chi = 8 runs stopped at the 2 h wall limit (tau >= 4700).
- **Open.** (i) The binary rank-1 plateau's lambda_min exponent is 0.87-0.89 on the three seeds with 5-6 sigma
  values and 1.15 on one 3-point seed (mean 0.95 ± 0.07): whether it is the k = 3 value 1 is not established.
  (ii) MPS edges 1 and 2 rise with power-law-like sigma dependence where counting predicts k = 2 (with fair fits
  edges 3, 4, 6-9 rise in log time, edge 0 is mixed). (iii) Whitened features give weaker, noisier scaling (the
  small-init asymptote moves to smaller sigma; the one whitened first-route series with the k = 3 form has
  B > 0).
- **Next steps (suggested).** Route amplitudes at higher-rank MPS states to explain (ii); longer horizons at
  sigma <= 0.07 to measure the k = 5 escapes (sigma^-3) systematically; tasks with more interaction structure
  (one-vs-rest, 10-class one-hot targets) to map when the pixel order matters; MPS chains in several pixel
  orders to separate tree shape from order; compare with sweep (DMRG) training.
- **Library.** No library changes; experiment code only (`experiments/real_data/`). Merged main 34c76ad and
  6c42d85 (gauge fix; all lambda_min here use the fixed projection, raw values reported alongside).
- **Process note.** Twice I killed my own processes by pattern (`pkill -f`/`pgrep -f`), which also hit my own
  shell; only processes I had started were affected, but from then on I killed by explicit PID only.

## Log
- 21:19 reverted launch commit, read brief/notes/library, downloaded data, launched cost pilot
- 21:37 cost probes: post-escape stiffness (lambda_max ~ 57); LSODA + Gauss-Newton Jacobian chosen; function-space spectra added; merged main; pilot (18 runs) launched
- 22:08 pilot analysed (staircases on quadtree binary, route amplitudes predict first route); main sweep failed at sigma <= 0.1 (clock underflow) -> chunked phase 1; merged main 6c42d85 (gauge fix; no lambda_min reported yet)
- 22:59 phase-1 clock re-anchoring (escape at t ~ 1e18-1e20 now resolved); closed-form Jacobian; sweeps relaunched with 48 + 12 workers (machine memory-bound)
- 00:02 interim report: 80/280 main runs; t1 exponents exact; k = 3 sigma^-1 and k = 2 log gaps; route amplitudes 34/35; regime sweep relaunched at sigma >= 0.5
- 00:45 main sweep 124/220 runs; regime sweep 9/35 (sigma 0.5-1: not lazy, kernel scale 1e-13..3e-5 at init: raw features are small-init in function space up to sigma ~ 1); plateau Hessians and route amplitudes running; chi8 + fashion queued after main
- 01:39 main sweep (234 runs) and regime sweep (65 runs) analysed; notes section 5 table; final figures; chi8 / fashion / 8x8 running
- 02:24 Fashion (45 runs) + tolerance check analysed; Fashion rank-1 rotation through a strict saddle; chi8 split to a second runner; chi8 / 8x8 running
- 04:24 chi = 8 (45 runs) analysed: same structure; 8x8 runs still running
- 04:25 all sweeps analysed (main, regime, Fashion, chi = 8, 8x8, tolerance check); report complete
