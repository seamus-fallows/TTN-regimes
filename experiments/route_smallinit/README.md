# route_smallinit: the route theory against training from small random init (star targets)

**Question.** The paper `report/ttn_saddle_order_and_aligned_training_dynamics.pdf` predicts how a
tree tensor network (TTN) learns a pattern target W* = sum_a s_a U_a (one-hot product tensors
"patterns", weights s_a) under gradient flow: plateaus at the partial targets sum over learned a of
s_a U_a; the next pattern is the one with the smallest route depth c_J(r) (number of cores that must
change from zero, given the learned set J; ties to the larger s_r A_r(J)); plateau durations
~ sigma^-(c-2); growth with effective depth c; dynamics on the aligned manifold M*. Most of it is
proven only on M* or conditional on Assumption C.1. Does real training from a small random init
behave like that, and where not? Task: `notes/briefs/route_theory.md` 4.1. Pre-registration, results
and deviations: **`notes/route_smallinit.md`**. Report: `agent-runs/route-smallinit/REPORT.md`.

## Setup

- 10 star-satisfying targets (`configs/targets.yaml`, table in the note, predictions for every J in
  `results/predictions.csv`): decoupled (D4 mps4, DB4 binary4), shared entries with a light short
  route (B4), splices on the greedy path (S5A, S5B mps5; S7 mps7; H4 ht4; B8 binary8), a depth tie
  (TIE ht4; also inside S5A, S7, D4), the pair-star spectral shortcut (PS mps6).
- Network: bond = blocks + 2 per edge (capped at the cut rank); fan-in Gaussian init
  `spec.init(sigma, seed)` on all parameters; exact population loss with a dense, entrywise residual
  (`model.py`); exact Hessian as Jacobian.
- Gradient flow: step-driven LSODA (rtol 1e-10, atol 1e-15) with exact time bookkeeping across
  clock restarts, VODE-BDF fallback, trap rule (`driver.py`; pitfalls of `notes/saddle_dyn.md` sec. 8
  built in).
- Grid: sigma = 10^-1.5 ... 10^-5 in quarter decades (15 values) x 40 seeds x 10 targets = 6000 runs;
  plus the reference run on M* per (target, sigma) (exact reduced flow from uniform entries sigma) and
  a floor-check duplicate of every run at the 3 smallest sigma with different floating-point
  arithmetic (library norm-form gradient).

## How to run (from the repo root)

    uv run python experiments/route_smallinit/targets.py                 # predictions -> results/
    cd experiments/route_smallinit && uv run python sweep.py configs/sweep.yaml --procs 300
    cd ../.. && uv run python experiments/route_smallinit/analyze.py     # CSVs -> results/
    uv run python experiments/route_smallinit/figures.py                 # figures -> results/

The sweep is resumable (existing JSON skipped) and writes one JSON + one small npz trajectory
(t, L, P, off-target mass, gradient norm per solver step) per run to
`/root/outputs/route-smallinit/runs/{dense,lib,ref}/<target>/ls<log10 sigma>_seed<k>.json` (not in
git; ~1.5 GB). Variants: `--only S7,B8`, `--variants dense`; other grids by editing `configs/sweep.yaml`.
A single run: `run.run_full(cfg, sigma, seed)` then `run.reduce_run(...)`.

## Files

| file | contents |
|---|---|
| `model.py` | `Model`: dense contraction, velocity, Hessian, amplitude gradients, roundoff bound; `SpareManifold` (AlignedManifold with blocks + 2 per edge) |
| `driver.py` | step-driven stiff integrator with exact time offsets, restarts, stall fallback |
| `run.py` | one run (full network or reference on M*) and its reduction to events and plateau diagnostics; the three prediction rules; the scalar-route-equation time |
| `targets.py` | `results/targets.csv`, `results/predictions.csv` |
| `sweep.py`, `configs/sweep.yaml` | the registered grid |
| `analyze.py` | tidy CSVs, floor check, per-seed exponents, per-cell decisions |
| `figures.py` | draft figures `results/fig_*.png/pdf` |
| `headline.py` | `results/headline.md`: every number quoted in the note and the report |
| `diag_trapped.py` | post-hoc: D4 and B4 without the bond cap (spare channels on the end edges) -> `results/diag_trapped.csv` (columns as runs.csv plus `bond_cap`, `bonds`) |
| `configs/sweep_ps_long.yaml` | rerun of the 20 PS runs unfinished after 1 h (3 h cap, BDF integrator) |

## CSV columns (all in `results/`)

Identifying columns everywhere: `target` (name in `configs/targets.yaml`), `tree` (mps4, binary8, ...),
`d` (physical dimension per site), `chi` (largest bond dimension of the network; per-edge bonds are
blocks + 2 capped at the cut rank, listed in `targets.csv`), `sigma` (init scale), `log10_sigma`,
`seed` (init seed; -1 for reference runs), `variant` (`dense` = training run, `ref` = reduced flow on
the aligned manifold M*, `lib` = floor-check duplicate). Pattern indices refer to the order in
`configs/targets.yaml`; `*_str` columns give the pattern strings. Times are gradient-flow time
(dimensionless). J sets are written `{0,2}`.

**targets.csv** (one row per target): `kind` (brief's categories i-v), `K` (cores), `patterns`,
`weights`, `bonds` (per-edge bond dims, edge order of the tree), `n_params`, `star`,
`max_min_depth` (largest c*_J over nonempty proper J), `greedy_seq_from_heaviest`.

**predictions.csv** (one row per target, learned set J, missing pattern r; no training):
`J`, `J_str`, `missing`, `missing_str`, `weight` (s_r), `depth` (route depth c_J(r)), `pairwise`
(min over a in J of |St(V_ar)|), `region` (nodes of R_J(r)), `gamma` (A_r(J) at the balanced theta_J),
`coef` (s_r A_r(J)), `pred_greedy` / `pred_pairwise` / `pred_weight` (next pattern by each rule),
`pred_lib_greedy` (`RouteTarget.greedy_next`, weight-only tie-break), `tie` (two or more missing
patterns share the smallest depth), `min_depth` (c*_J), `one_component_order` and `_mechanism`
(Theorem 3.2 order at |J| = 1).

**runs.csv** (one row per run): `status` (done: L < 1e-8; trapped: every unlearned |P_r|/s_r < 1e-35
for a decade of time; unfinished_wall: 3600 s; failed; error), `behaviour` ("clean" or "+"-joined
tags: status if not done, `two_at_once` (an event where another unlearned pattern was above 10% of
its weight when the learned one passed 90%, or a plateau never reached its entry point),
`forgetting`, `noncanonical_plateau` (a resolved plateau with canonical distance >= 0.01),
`floor_flag`), `n_learned`, `order` / `order_idx` (patterns in learning-event order), `first_pattern`,
`all_greedy` (every transition after the first matched the greedy rule), `n_forgotten`, `forgotten`
(pattern@time), `n_unseparated`, `n_no_entry` (plateaus without an entry point), `n_noncanonical`,
`L_final`, `t_final`, `max_offtarget` (max over steps of the off-target fraction, includes the random
init), `floor_flag` (floor-check duplicate disagreed; see floor_check.csv), `floor_checked`,
`floor_flag_bound` (informational worst-case roundoff bound below 100x the escape signal; see the
note: far too pessimistic), `steps`, `segments` (local clocks), `restarts`, `vode_segments` (stall or
failure fallbacks), `fails`, `secs` (wall time).

**events.csv** (one row per learning event; event 0 = first pattern learned, from the origin):
`event`, `pattern`, `pattern_str`, `pattern_weight`, `t_event` (absolute time of the last upward
crossing of s/2), `n_up` (number of upward s/2 crossings), `J_before` / `J_before_str` (learned set
just before the event, J(t) definition), `J_prev_events` (patterns of earlier events),
`n_learned_before`, `transition` (`J_before->pattern`), `duration` (event time minus previous event
time; for event 0: from t = 0), `t_escape` (from the previous event to the first L < L_J - 1e-3),
`depth` (c_J(r) of the learned pattern; K for event 0), `pairwise`, `min_depth` (c*_J),
`pred_greedy`/`pred_pairwise`/`pred_weight` (+ `_str`), `pred_lib_greedy`, `tie`, `agree_greedy`,
`agree_pairwise`, `agree_weight`, `gamma` (A_r(J)), `one_component_order` (Thm 3.2, |J| = 1),
`eff_depth` (effective depth from `ttn.routes.effective_depth` over the plateau from its entry
point to the event; nan if fewer than 6 points with P/s in [1e-4, 1e-2]), `eff_slope`, `eff_points`,
`eff_depth_full` (same over all steps since the previous event), `peak_offtarget` (max off-target
fraction of ||W||^2 between the time midpoints of the plateaus before and after; for event 0 it
includes the random init), `sep_other_at_r90` (largest P_q/s_q over other unlearned q when the learned
pattern first passes 0.9 s), `unseparated` (that value > 0.1), `has_entry` (plateau reached its entry
point: every learned amplitude within 1% of its weight), `L_minus_LJ_mg` (L - L_J at the
minimum-gradient point of the plateau), `gradnorm_mg`, `rel_dist_mg` (||W - sum_J s_a U_a|| / ||W*||
there), `canonical` (rel_dist_mg < 0.01), `t_entry_to_event`, `P_r_entry` / `P_r_mg` (signed amplitude
of the learned pattern at entry / minimum-gradient point), `onestep_pred_time` (predictor A: eq. 6.15
time from P_r(entry) to s_r/2 with depth c and A_r(J); inf if P_r(entry) <= 0), `onestep_ratio`
(observed t_entry_to_event / predictor A), `onestep_winner` (missing pattern with the smallest
predicted time A; -1 if none), `agree_onestep`, `onestep_pred_time_B` / `onestep_ratio_B` /
`onestep_winner_B` (predictor B: same equation started from the geometric mean of the route-entry
norms instead of P_r), `floor_flag` (run's floor-check flag), `floor_flag_bound`, `run_status`.

**plateaus.csv** (one row per plateau with J nonempty and missing pattern q; dense runs): `event`
(the event that ends the plateau), `plateau` (= event - 1), `J`, `missing`, `missing_str`,
`learned_next` (q is the pattern learned at the end of the plateau), `depth`, `pairwise`, `gamma`
(A_q(J)), `onestep_pred_time` (A), `onestep_pred_time_B`; at the entry point (`_entry`) and at the
minimum-gradient point (`_mg`): `P` (signed P_q), `ren_min`, `ren_max`, `ren_gmean` (min / max /
geometric mean over the nodes of R_J(q) of the route-entry norms from `align` + `route_entry_norms`,
divided by sigma), `align_q` (min over edges of `align`'s quality, 1 = perfectly aligned),
`gamma_obs` (product of q's active entries in the aligned cores), `floor_ratio` (escape signal over
worst-case roundoff bound), `floor_flag`; `gamma_ratio_mg` (gamma_obs_mg / gamma), `ren_spread_entry`
(ren_max / ren_min at entry).

**floor_check.csv** (one row per duplicated run): `order_dense`, `order_lib`, `status_dense`,
`status_lib`, `max_rel_duration_diff` (largest relative difference of event durations > 1),
`floor_flag` (order or status differ, or difference > 1e-3).

**exponents.csv** (one row per variant, target, cell (J_before, pattern) and seed with >= 5 window
points spanning >= 1.5 decades in sigma in [1e-5, 1e-3], status done, not floor-flagged):
`transition`, `depth`, `pairwise`, `one_component_order`, `gamma`, `n_points`, `log10_sigma_min/max`,
`exponent` (OLS slope of -log duration on log sigma), `pred_route` (c - 2), `pred_pairwise`,
`log_slope` (OLS slope of duration on ln(1/sigma), for c = 2), `pred_log_slope` (1/(s_r A_r(J))).

**cells.csv** (one row per variant, target and transition): counts (`n_events`, `n_events_small` at
sigma <= 1e-3 without floor flag, `n_seeds`, `n_fitted`), exponent quartiles (`exp_q1`, `exp_median`,
`exp_q3`), `frac_within_0p3_route` / `_pairwise` / `_order`, `log_slope_median`, `pred_log_slope`,
`decision_exponent` (registered rule), and at sigma <= 1e-3: `eff_depth_median`, `frac_eff_match`
(|eff - c| <= 0.3), `frac_canonical`, `frac_unseparated`, `peak_off_median`,
`onestep_ratio_median`, `onestep_ratio_B_median`, `onestep_winner_agree`, `frac_P_entry_negative`.

**c1.csv** (Assumption C.1 items 1-2; one row per target, J, missing pattern; dense runs with
sigma <= 1e-2): `slope_log_ren_vs_log_sigma` (pooled OLS of log(route-entry geometric mean at entry)
on log sigma; item 1 predicts 1), `ren_gmean_over_sigma_median`, `spread_max_over_min_median`,
`gamma_ratio_mg_median`, `frac_gamma_within_5pct` (sigma <= 1e-3), `frac_P_entry_negative`,
`learned_next_frac`, `n_rows`.

## Results (details, status tags and decisions: `notes/route_smallinit.md` Part II)

Numbers at sigma <= 1e-3 unless stated; `results/headline.md` lists them all (generated).

| question | answer | evidence |
|---|---|---|
| (a) plateaus at the canonical partial targets? | yes: 98.7% within 1% of ||W*|| (median distance 8e-9) | `events.csv` `rel_dist_mg`, `canonical` |
| (b) next pattern = greedy rule? | always when there is no tie in route depth (6867/6867); the pairwise rule is wrong in all 747 transitions where it differs (splices); ties go to the heavier pattern only 62-100% of the time, decided by the seed | `events.csv` `agree_*`, `fig_rules`, `fig_rules_vs_sigma` |
| (c) duration ~ sigma^-(c-2)? | yes on clean transitions: 98% of 648 per-seed fits (c >= 3) within 0.3, log law with the predicted slope for c = 2 in 6 of 7 cells; exceptions: lighter-first plateaus are strict saddles (log law, Theorem 3.2 / Remark 8.5), PS after 000000 follows Theorem 3.2's order 4 (sigma^-2, not the route depth's sigma^-3), B4 log slope 0.70 x prediction | `exponents.csv`, `cells.csv`, `fig_durations`, `fig_exponents`, `fig_shortcut_PS` |
| (d) effective depth = c? | yes: medians 2.02, 3.02, 4.09, 5.16; first escape K | `events.csv` `eff_depth`, `fig_effective_depth` |
| (e) close to the aligned manifold? | yes between and during clean transitions (peak off-target mass median 8e-9, ~sigma^2); not during takeovers and mixed first escapes (up to 0.5) | `peak_offtarget`, `fig_offtarget` |
| (f) Assumption C.1 | (1) dormant entries ~ sigma at entry in 36/39 cells (B4: sigma^0.70); (2) active factors = A_r(J) (ratio 1.000); (3-4) separated in 98% of events (PS 79%), but takeovers erase learned patterns | `c1.csv`, `plateaus.csv` |
| one-step-ahead (eq. 6.15) | from P_r at entry: fails (ratio 2.7, IQR 0.9-579); from the dormant-entry size: works (1.04, IQR 0.93-1.39) for c <= 4 | `onestep_ratio`, `onestep_ratio_B`, `fig_onestep` |
| runs doing something else | forgetting 807 runs (13%; S5A, S5B, S7, PS), two at once later 793 (PS pair), trapped 270 (D4 13 seeds, B4 5 seeds, at every sigma), mixed first escape 1391 | `runs.csv` `behaviour`, `history`, `fig_census`, `fig_example_forgetting` |

Figures (`results/`): `fig_example_run` (one B8 run, plateau by plateau), `fig_rules`, `fig_rules_vs_sigma`,
`fig_durations`, `fig_exponents`, `fig_shortcut_PS`, `fig_example_forgetting`, `fig_effective_depth`,
`fig_onestep`, `fig_offtarget`, `fig_census`. The reference flow on M* starts from uniform entries
sigma, the training runs from fan-in Gaussians of scale sigma (entries ~sigma/sqrt(fan-in)), so their
durations differ by a constant factor; the slopes agree.

`passes.csv` (one row per upward crossing of half the weight, also the non-final ones of runs with
forgetting): `n_pass`, `pattern`, `J_before` (J(t) just before), `t_pass`, `duration` (since the
previous pass), `time_resolution` (float64 spacing of t / duration), `precise` (< 1e-3), `final` (it is
a learning event), the predictions and `agree_*` as in events.csv. `exponents.csv` and `cells.csv` have
a `basis` column: `event` (registered: all learning events), `event_clean` (events whose interval since
the previous event contains no other crossing and no forgetting, pattern crossed once, plateau
resolved), `pass` (passes with precise times). events.csv also has `contaminated`, `clean`; runs.csv
has `history` (the full sequence of crossings of half the weight, + up / - down).
