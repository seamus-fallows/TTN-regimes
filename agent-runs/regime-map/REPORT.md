# regime-map: phase diagrams of TTN training regimes (Q5)

> **2026-10-10, verification of round 2:** errata E1-E19 of `notes/reviews/regime_map_round3_verification.md`
> are applied below (summary items 4 and 7, items (j) and (k), decisions 22, 24 and 25, caveats, two log
> lines), marked "(E#)"; the full list is in `notes/regime_map.md` section 4.9.

Status: done - Branch: agent/regime-map - Updated: 2026-10-10 15:36 UTC

> This body was rewritten after round 2 (2026-10-10).
> - It now matches `notes/regime_map.md` section 3 (fix round 1, every changed claim old -> new) and
>   section 3.4 (errata E1-E6 of `notes/reviews/regime_map_round2_verification.md`).
> - Withdrawn claims are not repeated here: "every boundary predicted quantitatively",
>   "sigma_c -> sqrt(d), not 1 (H4 refined)", "chi^-0.84 as P16 predicts" and "S2S widest at
>   lambda = sqrt(chi)".
> - Round 2 (realizable sigma_c(K), d = 2 and 3) and the tensor-kernel relabel of the round-1 maps are
>   items (j) and (k).

## Summary
Round 2 asked: the realizable S2S boundary sigma_c(K) at large K (d = 2, 3), the verification errata, and
tensor-kernel labels for the N = 4 maps (items (j), (k)). Round 1 asked: phase diagrams of saddle-to-saddle (S2S), rich and lazy gradient-flow dynamics of TTNs over the
init scale sigma, bond dimension chi, number of cores K and core imbalance, with boundaries compared to
theory and threshold sensitivity. Round 1: about 9,000 runs, 6 seeds per point, per-seed statistics.
Status tags as in the notes.

1. **Binary N = 4 (K = 3).**
   - Per seed, rho ~ sigma^-(1.02-1.05) on the S2S side, against -(K - 2) = -1 (empirical).
   - So each seed's S2S boundary follows from its deepest run. This is a consistency check, not an
     a-priori prediction.
   - With the tensor-space kernel the lazy boundary lies inside theory P5's bracket at chi = 24-96 and
     0.06% below it at chi = 128 (sigma_lazy sqrt(chi) = 6.40-6.56 vs 6.50; empirical, confirms derived
     P5).
2. **Both N = 4 boundaries scale ~ chi^-1/2** (chi = 16-128, under one decade). The map collapses in
   sigma sqrt(chi). sigma^K chi^(K-1) is disfavoured at 2.7-2.9 SE (per-seed fits).
   - This is the TTN analogue of known width scales (Tu et al. 2024, Geiger et al. 2020). Empirical,
     N = 4 and fan-in only.
3. **No lazy regime below the cut dimension.** The centred target is unrealizable there, and runs stall
   at the Eckart-Young floor (derived R7; 44/44 runs).
4. **The S2S boundary rises with K:** 0.17, 0.49, 0.61, 0.68, 0.75 at K = 3-7 (round 1, centred, d = 2,
   chi = 4). Round 1 could not decide its large-K limit.
   - Round 2 (realizable problems, d = 2 and 3, item (j)): in the plain model with spare channels, at
     K >= 8, 85 of 96 seeds are censored from above (E2). 76 stay S2S-like at every sampled sigma up
     to the sigma where the init function itself overlaps a teacher component at 0.1 (the event
     rule's domain limit), and 9 were still S2S-like when the budget ended. The other 11 cross, all
     below sqrt(d): 0.86-1.37 at d = 2, and one seed of 48 at d = 3, at 1.08 (E1, E5).
   - The domain limit has median about sqrt(d) at K = 8-15 but tends to about 1.15 sqrt(d), not
     sqrt(d) (E4). The lower bounds at K >= 8 (1.07-2.03, E12) lie above 1 and above both candidate
     laws, sigma^K = const and sigma^K / sqrt(D) = const, extrapolated from the resolved cells.
   - So sigma_c does not tend to 1 there (H4 rejected for the plain model at K <= 15, empirical; the
     comparison uses the domain censoring and was not pre-registered).
   - A candidate mechanism is the plain model unlearning W_0 on the plateau (H_W). It is consistent
     with the data but not confirmed: the measured core shrinkage is required by an exact norm
     identity, and the causal step to the boundary is untested (conjecture; E3).
   - In the centred model with full bonds (K <= 8 at d = 2, K <= 4 at d = 3), the two laws are not
     resolved (beta = -0.07 +- 0.51).
   - sigma_c -> sqrt(d) remains a conjecture.
5. **Large chi** (N = 4, tensor kernel): only muP-TTN keeps an O(1) kernel change (delta_K -> 6.2).
   - SP: delta_K ~ chi^-3.02.
   - NTK: ~ chi^-1 at chi >= 64.
   - All as theory P16 predicts.
6. **Imbalance.**
   - A light root makes training lazy, with delta_K ~ lambda^2 (local slopes 1.97, 1.85; P20).
   - delta_K saturates at ~0.6, the lower edge of P20's 0.6-0.7.
   - A heavy root is lazy from alpha = 100 (tensor-kernel labels), as at balance.
7. **Tensor-kernel labels** (item (k)): relabelling the N = 4 maps with the tensor-space kernel instead
   of the 128-probe kernel changes few labels and moves the lazy boundaries up by about 5% (1-9%;
   14-15% at chi = 6-9; E18).

## Results
All numbers: whitened inputs, exact population loss, odeco teacher [3, 2, 1] (d = 3) or [3, 2]
(d = 2), fan-in init, gradient flow with the shared library, every event re-integrated on a dense
linear grid (last upward 1/2-crossing before the component first reaches 0.9), 6 seeds per point.
- Raw runs: `/root/outputs/regime_map/<exp>/` (round 1) and `/root/outputs/regime_map_r2/` (round 2).
- Tables: `experiments/regime_map/results/` (fix round: `results/fix1/`; report tables:
  `results/report/`; round 2: `results/round2/`).
- Derivations and status tags: `notes/regime_map.md`.

**(a) sigma x chi, binary N = 4** (`sigma_chi_b4`, centred; notes 2.1, 3.2 item 1; tables
`fix1/table_2_1_s2s.csv`, `fix1/lazy_summary_chi_b4.csv`).

| chi | 3 | 4 | 6 | 9 | 12 | 18 | 24 | 36 |
|---|---|---|---|---|---|---|---|---|
| S2S boundary (per-seed rho = 3 crossings, geometric mean) | 0.237 | 0.099 | 0.137 | 0.136 | 0.075 | 0.072 | 0.067 | 0.043 |

- Per-seed rho exponent on the S2S side: -1.02 to -1.05 (53/54 chi_b4 seeds: -1.015 to -1.031). Four
  seeds with transients above 1/2 are steeper (-1.09 to -1.27).
- Deep-S2S event widths are 0.466, 0.798 and 2.010 for s = 3, 2, 1, against the aligned theory's 0.46,
  0.80 and 2.01.
- Threshold sensitivity:
  - rho* from 2 to 5 moves the S2S boundary by x2.13-2.34 (asymptotic prediction 2.5);
  - delta_K* from 0.05 to 0.2 moves the lazy boundary by x1.30-1.35 at chi >= 32 (P7) and by x1.97 at
    chi = 9.
- Below the cut dimension (chi < 9) every stalled centred run ends at the Eckart-Young floor of
  W_0 + W*. A run ending on a floor below 1e-3 counts as converged.

**(b) Boundaries vs chi** (`chi_b4`, chi = 9-128; notes 2.1b, 3.2 item 2).

| | slope vs chi | sigma sqrt(chi) at the boundary | theory |
|---|---|---|---|
| S2S boundary | -0.528 +- 0.053 (chi 16-128) | 0.27-0.33 (measured medians) | not derived |
| lazy boundary, tensor kernel | -0.504 +- 0.011 (chi 32-128) | 6.40-6.56 | -1/2; 6.50 (P5) |
| lazy boundary, two-copy, tensor kernel | -0.499 +- 0.014 (chi 32-128) | 5.06-5.21 | -1/2; 5.16 (P6) |

- The two-copy / centred lazy boundary ratio is 0.79 (P6: 2^(-1/3)).
- The rich window (lazy / S2S boundary) is 18-23x in sigma.

**(c) sigma x K, round 1** (centred, d = 2, chi = 4; notes 2.3, 3.2 item 4, 3.4 E1-E2).
- Resolved label boundaries: 0.17, 0.49, 0.61, 0.68 (4/6 seeds) and 0.75 at K = 3-7.
- K = 8: 1.01 (5/6 seeds; floor-stalled runs classified by rho at deficient bonds).
- The K = 10 and 12 brackets end at no-learning stalls, so they are not boundaries.
- sigma^K = const and sigma^K / sqrt(D) = const are indistinguishable (chi^2 0.8 vs 0.6 on K = 4-7;
  7.7 vs 6.9 on K = 4-10 with crossing-terminated brackets).

**(d) K = 7 maps (binary N = 8)** (notes 2.6).

| tree / bonds | S2S side | large sigma |
|---|---|---|
| d = 3, chi = 24 (deficient) | sigma_c <= 0.65 (bracket [0.45, 0.65]; one S2S run per seed in 5/6 seeds) | rich at 0.65-0.93; floor-stalled from 1.3 |
| d = 2, bonds 8/32 (2x cut dimension) | 0.53 | lazy from ~1.4 (kernel boundary 1.45) |

**(e) Large chi** (N = 4, centred, sigma = 1.5, chi = 16-128; notes 2.5, 3.2 item 5).

| slope vs chi | delta_K (tensor kernel) | delta_theta | root change |
|---|---|---|---|
| SP (theory -3, -3/2, -1) | -3.02 +- 0.06 | -1.63 +- 0.04 | -1.02 +- 0.03 |
| NTK (-1, -1/2, -1) | -0.79 +- 0.04 (-1.01 between chi = 64 and 128; levels 0.250, 0.124 vs R4's 0.240, 0.120) | -0.52 +- 0.03 | -0.93 +- 0.02 |
| muP-TTN (0, 0, -1/2) | +0.085 +- 0.024 (levels 5.1-6.2 -> 6.2) | -0.06 +- 0.02 | -0.50 +- 0.01 |

**(f) Imbalance** (N = 4; notes 2.4, 3.2 item 6; `fix1/imbalance_P20.csv`).
- At chi = 16 and alpha = 3.375, the per-seed local slopes of the tensor-kernel delta_K vs lambda are
  1.97 +- 0.01, 1.85 +- 0.06, 1.55 +- 0.11 and 1.22 +- 0.09.
- delta_K saturates at 0.60 (seeds 0.51-0.75), the lower edge of P20's 0.6-0.7.
- At chi = 24 a light root lowers the lazy boundary by ~100x in alpha. With tensor-kernel labels
  (item (k)) the boundary is alpha = 1 for a light root and 100 for balance and heavy roots.

**(g) Stalls at chi = teacher rank** (K = 7, 8; notes 2.2): fan-in init leaves dead routes in
12/18 (binary 8) and 16/18 (MPS 8) runs. Gauge-balanced init removes them, but the route still decays
(P14).

**(h) Two-copy model** (notes 2.x, 3.2 item 7):
- It has the same lazy residual as the centred model and a second-order kernel change, which gives the
  0.79 boundary ratio.
- At small sigma, events 1 and 2 merge. The largest min_i rho_i at sigma <= 0.016 is 7.1, 14.4 and 4.8
  at chi = 18, 24, 36, so the two-copy model is not a cross-check of the S2S boundary.

**(i) Checks** (notes 2.y, 3):
- scipy RK45 vs library DP5 on 1937 runs: labels agree in 98.1%.
- Conservation drift <= 1e-9.
- Local reruns vs remote runs (1,323 runs): identical labels.
- The tensor kernel matches the library to 2e-16 (verified independently).

**(j) Round 2: sigma_c(K) in realizable problems** (pre-registered in notes section 4 before any run,
f57a28c at 12:26 UTC (E7); amendments 4.6, results 4.7, answer 4.8, errata 4.9; tables and figures in
`results/round2/`).
- **Design.**
  - Arm P: plain model (target W*, realizable), chi = 4 = 2R, teacher (3, 2) at both d = 2 and 3,
    binary_tree(N) with N = 4-16 (K = 3-15), 6 seeds. Adaptive per-seed sigma walker, 1,070 runs.
  - Arm C: centred model with full bonds (W_0 subtracted, realizable), d = 2 for K = 3-8 and d = 3 for
    K = 3-4, 360 runs.
  - Statistic: y = ln(sigma_c^K / u_0c(K)) = -gamma + beta ln sqrt(D). H_A (sigma^K = const) is
    beta = 0; H_B (sigma^K / sqrt(D) = const) is beta = 1.

| | arm P (plain, chi = 4) | arm C (centred, full bonds) |
|---|---|---|
| median sigma_c, d = 2, K = 3 / 5 / 7 / 8 | 0.16 / 0.59 / 0.78 / 1.12 (2 of 6 seeds) | 0.16 / 0.50 / 0.81 / 0.83 |
| seeds censored from above (S2S-like at every sampled sigma up to the event-rule domain limit, or open at the end of the budget) | 97/156 (88 domain, 9 open); 85/96 at K >= 8 (76 domain, 9 open); d = 3, K >= 8: 47/48, one seed crosses at 1.08 (E1, E2) | 0 |
| lower bounds at K >= 11 (median) | 1.45 (d = 2), 1.69 (d = 3) | - |
| pre-registered beta, rho* = 3 (cells with <= 2 censored seeds) | 1.21 +- 0.22 ("H_B favoured"; 0.61 +- 0.51 without shrinkage cells; 0.72 +- 0.56 at rho* = 2) | -0.07 +- 0.51 (unresolved) |
| censored (Tobit) beta over all seeds, not pre-registered | 3.15 +- 0.23 (rejects both laws) | -0.17 +- 0.55 |
| d-ratio sigma_c(3) / sigma_c(2) | 1.27, 1.13, 1.57, 0.89 at K = 3-6 (CIs at K = 3-5 contain 1 and H_B's 1.27-1.31; at K = 6 [0.64, 1.22] excludes 1.27, with 3 of 6 d = 3 seeds censored; E15) | 0.93, 0.73 at K = 3, 4 (chi differs with d) |

- **Answer.**
  - In the plain model sigma_c does not tend to 1. At K >= 8, for 85 of 96 seeds the S2S region
    reaches the sigma at which W_0 itself overlaps a teacher component at 0.1 (76 seeds) or was still
    open when the budget ended (9); above that sigma the event rule does not apply. The other 11 seeds
    cross below sqrt(d) (E1, E2, E5). The domain limit's median is about sqrt(d) at K = 8-15 and tends
    to about 1.15 sqrt(d) (E4).
  - All 97 censored lower bounds exceed both laws' predictions. H4 (sigma_c -> 1) is rejected for the
    plain model at K <= 15 (empirical). The bounds hold at the sampled sigmas only (E13).
  - H_W: before the first escape the plain model shrinks its cores (unlearns W_0). The median loss of
    squared core norm is 0.1% at x_W = sigma^K sqrt(D) / (s_2 d) < 0.01, 10% at x_W ~ 1, 52% at 1e2
    and 92% above 1e4, and 0 in the centred model. Near the top of the S2S block rho stops falling
    with sigma. These numbers are empirical. That the shrinkage lifts the boundary is consistent with
    them but untested; the shrinkage itself follows from an exact norm identity, and H_W's registered
    signature "arm C near beta = 1" failed. H_W stays a conjecture (E3).
  - Centred, realizable: unresolved up to K = 8. sigma_c -> sqrt(d) stays a conjecture.
- **Not resolved:**
  - the centred boundary beyond K = 8, because full bonds cost about 1 h per run at P = 2349;
  - the d-dependence at fixed K;
  - the regime above the domain limit.
- **Run time** (arm P, under about 330 concurrent runs):
  - median per run 100-250 s at K = 3-5, 300-650 s at K = 6-10 and 700-1,650 s at K = 13-15
    (timing_P.csv);
  - 5-7 sequential walker steps per seed, so at K >= 13 some walkers had not finished at the cut-off;
    by the analysis statuses 9 seeds are open (d = 2, K = 15: 2; d = 3, K = 13: 3, K = 14: 1,
    K = 15: 3), not three to six per cell (E16);
  - the largest K with a resolved seed is K = 12 at d = 2 (4/6 seeds, also fit-eligible) and K = 10 at
    d = 3 (1/6); the largest fit-eligible d = 3 cell is K = 5 (6/6) (E17).

**(k) Tensor-kernel relabel of the round-1 N = 4 maps** (notes section 5;
`results/tensor_relabel/`).
- The labels of sigma_chi_b4, chi_b4 and imbalance_b4 now use the tensor-space kernel, with the same
  rules (delta_K < 0.1, rho > 3).
- 65 of 5,298 labels change (1.2%). Almost all are lazy -> rich at the lazy boundary; the plain model
  also has 3 rich -> lazy and 3 s2s -> lazy.
- Per-seed lazy boundaries move up by a median of 1-9% (E18; first written as 4-9%, the centred
  range): centred chi_b4 1.04-1.09 at chi = 12-128, plain 1.04-1.07, two-copy 1.01-1.07 at chi >= 9;
  14-15% at chi = 6-9 (centred sigma_chi_b4 1.14 at chi = 9, two-copy 1.15 at chi = 6, 2 seeds).
- The S2S side is unchanged.
- The imbalance map at chi = 24 changes in 7 of 189 cells: balance and heavy roots are lazy from
  alpha = 100 instead of 10, and a light root (lambda / sqrt(chi) = 0.01) from alpha = 1 instead of
  0.1.
- Regenerated with tensor labels:
  - `results/report/runs_sigma_chi_b4.csv`, `floor_sigma_chi_b4.csv`, `imbalance_map_b4.csv` and
    `numbers.csv` (3 of 50 numbers change);
  - `report_map.pdf` and `report_other.pdf`, also copied to `report/figures/regime_map/`;
  - the phase, sensitivity and boundary figures and CSVs of these sweeps.

## What was done
Commit hashes on `agent/regime-map`:
1. Theory draft and pilots: lazy residual, lazy coefficients vs chi, two-copy second-order kernel
   change (49f8580, 60bf327, 9696394, 7a71905).
2. Interim scipy driver and first N = 4 map; interim report (e995bb7, d675177, 7832b4a).
3. Library driver `run_regime.py`, `sweep.py` and YAML configs, with linear-time event refinement
   (28d1826).
4. Stall tests, gauge-balanced init, output-dir options and per-seed analysis (b556c44, a910af2,
   0e62c41).
5. Fixes and restarts:
   - convergence reference 1/2 ||W*||^2, single-threaded BLAS (2230955);
   - cost-controlled configs, sigma x K at d = 2 (c874913, 49f3f53, 1465852, f7cc0e0, e326b9a);
   - centred K sweep (8da6cde);
   - direct centred loss (80022b0);
   - sigma-descending order (6b20cb7).
6. Analysis and figures: thresholds, chi boundaries, large chi, imbalance, sigma_c vs K, resolution
   flags, summary (671643b, b276eca, cf7f272, a8bea1b, bb84947).
7. Notes `notes/regime_map.md` sections 0-2 (769db99, f5cbaef, de31177, f501a14, 23fa63b,
   5a41232, d10e93f) and README (7790c16).

8. Round 2 (2026-10-10):
   - merged main (0804eb7) and pre-registered the hypotheses and decision rule (f57a28c);
   - adaptive per-seed driver `round2.py` (c8cae51); event-rule domain, time budget, forked pool
     (ece2871); arm C (bc28dd7, 32f24f7);
   - crash fixes: broken pool restart (5533a6c), tensor-kernel OOM guard (1c88e3d), walker stall
     (8ebba5d);
   - analysis `round2_analysis.py` with Tobit, H_W shrinkage, timing (8ebba5d, ee25939); results in
     `results/round2/`;
   - notes 4.6-4.8 and report (j).
9. Lead's extra items:
   - merged main again;
   - verification errata E1-E6 in the notes (d5da436, section 3.4), README E1 and the rewritten
     report body (b579cc9);
   - tensor-kernel relabel of the N = 4 maps (`relabel_tensor.py`, `REGIME_MAP_LABEL_KERNEL`;
     f5c4783, c0eff35, 22ddd33), with the report tables, numbers and figures regenerated and
     `report/figures/regime_map/{map,other}.pdf` updated.

## Decisions
1. **d = 3, odeco teacher [3, 2, 1]** (minimal d for 3 odeco components; keeps the cut dimension
   small: 9 for N = 4, 81 for the top cut of N = 8, so full bonds are reachable). Alternative d = 4/5
   would make full bonds unaffordable for N = 8.
2. **N = 4 bond dimensions uncapped** (`cap=False`), chi in {3,4,6,9,12,18,24,36}: chi = 9 is the
   cut dimension (full, square cores, ill-conditioned), chi > 9 are overparameterized bonds, which is
   what the brief's "up to 8x the teacher rank" requires. N = 8 uses capped bonds (bottom bonds <= 9),
   since uncapped middle cores chi^3 are unaffordable at chi = 81.
3. **Interim integrator.** The library's dynamics/runner had not landed when the first diagram was
   due, so the interim sweeps use scipy RK45 (rtol 1e-9, atol 1e-14) on the flat parameter vector,
   as the brief allows for pilots, with an explicit time budget (README). They will be rerun or
   spot-checked with `ttn.dynamics` when it lands.
4. **All pilot results computed before the core-fix merge (teacher stream bug) were discarded**; the
   pilots had used teacher seed 1000 + s vs init seed s, but every number in this report is from
   runs after merging 57d5262.
5. **Stalled runs are split** (in figures only; the official label stays "stalled") into
   delta_K < 0.1 (kernel-like stall at the lazy residual, a representability effect) and
   delta_K >= 0.1 (feature-learning stall), because below the cut dimension the centred target
   W* + W_0 is unrealizable (theory R6) and "stalled" otherwise mixes two different mechanisms.
6. **Teacher seed = 1000 + init seed**, so seeds vary both; by the O(d) invariance of the init the
   teacher basis is statistically irrelevant, so this costs nothing and avoids any shared stream.
7. **Event widths are re-measured in linear time.** A log-time recording grid (1% spacing) cannot
   resolve O(1)-wide events at late times (at t ~ 4e5 on K = 7 the spacing is ~1e4), which made rho
   come out 1000x too small. Each final rise is re-integrated from the saved parameters on a
   200-point linear grid (`run_regime.refine_events`); deep-S2S widths then match the aligned theory
   to 3 digits (0.466, 0.798, 2.010 vs 0.46, 0.80, 2.01 for s = 3, 2, 1, K = 3).
8. **Integrators.** LSODA (exact Hessian) where P <= ~2500 (all of sigma x K; N = 8 with chi <= 12),
   dopri5 otherwise; N = 8 runs with large P start at sigma = 0.3-0.4, where later plateaus are still
   ~1e2-1e3 time units (deep S2S for K = 7: rho ~ 3 (0.9/sigma)^5).
9. **sigma_min(K) = 10^(-8/K)** in the sigma x K sweep so that initial component amplitudes ~sigma^K
   stay ~1e-8 or larger, far above the float64 floor of the gradient (lead's note); runs record the
   smallest dormant overlap at the first event (`dormant_min`).
10. **Interim scipy sweep superseded.** The first N = 4 sigma x chi map (2460 runs, scipy RK45) is
    kept in /root/outputs/regime_map/b4_sigma_chi as an integrator cross-check; all reported numbers
    come from the library runs (ttn.dynamics) with event refinement.
11. **Convergence is judged against 1/2 ||W*||^2, not L(0).** For the centred and two-copy models
    these are equal. For the plain model at large sigma, L(0) ~ 1/2 ||W_0||^2 >> 1/2 ||W*||^2, and the
    literal rule "L(T) < 1e-3 L(0)" (and a stop floor relative to L(0)) would call a run converged
    after it merely shrank W_0, before any teacher component is learned (seen: K = 15, sigma = 2, no
    events). Plain rows with L(0) > 100 x 1/2 ||W*||^2 computed under the old floor were discarded and
    rerun; the others were rescaled (the analysis recomputes labels from l_end).
12. **Gradient-flow budget is not unlimited**: runs stop at the time budget T or at max_steps
    (1e5; 3e4 for N = 8), whichever comes first. Runs stopped there unconverged are 'stalled' by the
    rule and are reported as such, with the stall type from the bond spectra (dead route: R-th bond
    singular value < 1e-12; long plateau: >= 1e-4; decaying: in between).
13. **K = 7 lazy region mapped at d = 2.** At d = 3 the top cut of binary N = 8 has dimension 81: full
    bonds mean square 81 x 81 cores (lazy fit needs tau ~ 1e5, theory P9) or 1.3e5 parameters for
    2x-overparameterized bonds (~1 s per gradient under load). The d = 3 map (teacher [3, 2, 1]) keeps
    the deficient bonds chi = 3..24 (centred and plain); a second map at d = 2 (teacher [3, 2], cut
    dimensions 4 and 16) has chi = 2, 4, 8, 16 (full, square) and 8 / 32 (full, 2x), where the lazy
    regime exists. Same for the large-chi study at K = 7 (d = 2, as in theory P17).
14. **sigma x K uses the plain model with chi = 6 (= 2 x teacher rank) plus a chi = 3 slice**, after
    theory v2 and the stall test: at chi = 3 most K >= 7 runs never learn the last component (dead
    routes), and the centred target is floored near the expected boundary. For K >= 10 at chi = 6 the
    sigma grid is a window around the predicted boundary (0.4-1.6) integrated with dopri5, because
    LSODA's Hessian costs ~5 min per evaluation for P ~ 1800 under full load.
    *Revised 22:10:* those dopri5 runs stopped at their step cap mid-plateau (stiff after the first
    escape), so the full K range is now mapped at **d = 2** (teacher [3, 2], chi = 4 = 2R and chi = 2 = R;
    <= 600 parameters, LSODA everywhere; `configs/sigma_K_d2.yaml`), as in theory P15. The d = 3 sweep
    keeps K <= 8 (chi = 3, 6) and the chi = 3 stall slice up to K = 15.
15. **Centred model with small W_0 uses the direct loss** 1/2 ||W - (W* + W_0)||^2 (alpha = prod sigma_v
    < 1e-2; `run_regime.ShiftedCP`). Its cancellation error is ~1e-16 alpha^2, and the cancellation-free
    doubled-bond form costs 4.5x per gradient and 14x per Hessian (binary 16 under load: 0.21 s vs 0.05 s,
    47 s vs 3.3 s). Checked on two configurations: identical labels, event times to 1e-4, delta_K to
    6 digits. Overlaps and spectra then refer to W instead of W - W_0 (difference O(alpha / sqrt(D))).
16. **Dropped the rest of the d = 3 sigma x K sweep for K <= 8** (252 runs done) when the heavy pool
    was projected at > 5 h; the d = 2 plain, full-bond and centred sweeps cover the K dependence.
17. **Cost.** The full program is ~9,000 gradient-flow runs; the heavy part (K >= 7 with LSODA or the
    centred loss) dominates: 20-60 min per run under full load. Machine time so far ~8.5 h on 368 vCPUs.
18. **Round 2: one teacher spectrum (3, 2) and chi = 4 at both d = 2 and d = 3.** Options: (3, 2, 1) at
    d = 3 as in round 1, or the same rank-2 teacher at both d. Chosen: the same teacher and chi, so
    that only D = d^N changes between d (the cleanest test of sigma^K vs sigma^K / sqrt(D)).
19. **Round 2: binary_tree(N), N = 4..16, as the single tree family for K = 3..15.** MPS was dropped:
    in round 1 MPS trees stalled on the S2S side at K >= 8 even with chi = 4.
20. **Round 2 primary arm = plain model with spare channels (chi = 2R).** It is realizable and reaches
    K = 15 at both d. The alternative realizable version, the centred model with full bonds (arm C),
    costs P ~ d^(N/2) chi and runs only up to K = 9 (d = 2) and K = 5 (d = 3) in the budget; it is
    kept as a cross-check.
21. **Round 2: adaptive per-seed sigma walker instead of a fixed grid.** It brackets each seed's
    crossing within 2^(1/(K-2)) in sigma at a fraction of the cost.
22. **Round 2: out-of-domain runs.** At large sigma the plain model's W_0 already projects on the
    teacher components with |a_r(0)| >= 0.1 (up to 50 in arm P; the binary-16 pilot had 8.85, first
    written as ~40, E8). Options: redefine events after the overlap first drops below 0.1, or classify
    such runs as outside the event rule's domain. Chosen: the latter (no change to the merged event
    rule; seeds S2S-like up to that point are reported as censored, i.e. as lower bounds on sigma_c).
23. **Round 2: time budget T = max(50 t_esc + 1e4, 1e6) and max_steps 3e4.** The default T was too
    short after the W_0 collapse.
24. **Round 2: secondary analyses that were not pre-registered.**
    - A censored-normal (Tobit) fit over all seeds, because most K >= 7 cells have more than 2 of 6
      seeds censored and the pre-registered rule excludes them.
    - A median-based variant, because single outlying seeds (y ~ -10) move cell means.
    - The pre-escape core-shrinkage diagnostic for H_W.
    - Birge scaling of the WLS standard errors (by sqrt(chi^2 / dof) when that exceeds 1). It was not
      in the pre-registration, and it decides rho* = 5: "H_B favoured" with it, unresolved without
      (E6, found by the verification).
    These are labelled as such in the notes.
25. **Round 2: arm C uses LSODA and only trees with P <= ~3300.** The dopri5 pilots had not finished
    after 20 min under load. Two larger cells, (K, d) = (9, 2) and (5, 3), were added at 14:22
    (commit 32f24f7) on cores left idle by arm P's slow walkers. (5, 3) ran one sigma per seed
    (rho 2.7-6.6) and enters the arm-C Tobit fit as 5 lower bounds and 1 upper bound; (9, 2) produced
    no rows (E11; now amendment 6 in the notes).
26. **Round 2: crashes and restarts.**
    - Fork instead of spawn: spawned workers ran out of memory.
    - The tensor kernel is skipped in arm P and its memory guard is fixed (D x D identity, OOM kills).
    - The walker restarted after a stall below the domain limit.
    Every relaunch resumes from the saved rows; in-flight runs were resubmitted.
27. **Round 2 budget: no new runs after 15:25 UTC** (about 3 h after the first round-2 run).

## Files left on the machine
Round 2:
- `/root/outputs/regime_map_r2/r2_plain/` (339 MB): arm P rows and trajectories, needed for
  `round2_analysis.py`.
- `/root/outputs/regime_map_r2/r2_centred_full/` (78 MB): arm C.
- `/root/outputs/regime_map_r2/r2_plain_discarded_eventbug/` (12 MB): 22 discarded rows, not needed.
- `/root/outputs/regime_map_r2_pilot/` (4 MB): timing pilots, not needed.
- `/root/outputs/regime_map/tensor_kernel/` (0.4 MB): dK_tensor cache of all 5,298 N = 4 runs, needed
  for the tensor relabel (recomputable with `relabel_tensor.py compute`).

Round 1:
`/root/outputs/regime_map/` (1.2 GB) holds one directory per experiment with `results.jsonl` (one
summary row per run) and `traj/*.npz` (library `Trajectory` per run). All are needed to regenerate the
figures with `make_figures.py`, and all can be regenerated by `sweep.py` (seeded; ~10 h on 368 vCPUs).

| experiment | size |
|---|---|
| chi_b4 | 184 MB |
| sigma_chi_b4 | 171 MB |
| sigma_K_d2 | 156 MB |
| sigma_K_centred_d2 | 155 MB |
| sigma_chi_b8_d2 | 132 MB |
| imbalance_b4 | 90 MB |
| sigma_chi_b8 | 75 MB |
| sigma_K_full_d2 | 60 MB |
| sigma_K | 53 MB |
| large_chi_b4 | 49 MB |
| b4_sigma_chi (interim scipy runs) | 44 MB |
| large_chi_b8 | 30 MB |
| stall_test | 11 MB |
| pilots, test* (throwaway) | 2 MB |

## Caveats and next steps
- **Round 2.**
  - Arm P's large-K boundary is censored by the event rule's domain (max |a_r(0)| >= 0.1).
    Above it W_0 already contains the teacher components, and S2S vs rich is undefined there.
  - 9 seeds were still open when the budget ended (15:34 UTC): d = 2, K = 15: 2; d = 3, K = 13: 3,
    K = 14: 1, K = 15: 3 (E16; first written as 3-6 per cell at K >= 13). Their lower bounds are
    reported.
  - Censored bounds hold at the sampled sigmas only (E13): a median of 4 in-domain runs per censored
    seed at K >= 8, gaps up to a factor 1.86 in sigma, and rho is not monotone in sigma; for the 7 seeds
    with an S2S-like rho < 6 a rich window in an unsampled gap cannot be excluded.
  - Arm C reached only K = 8 (d = 2) and K = 4 (d = 3). Its d-ratio confounds d with the full-bond chi.
  - The Tobit fit, the median variant, the shrinkage diagnostic and the Birge scaling of the WLS
    SEs were not pre-registered (decision 24; E6, E19).
  - Runs at d = 3 and K >= 12 where the second component never learned by t = 1e6 count as S2S-like
    (rho lower bound, rule 4.1). These are 14 of the 97 bounds, and a stall would be classified the
    same way. The rule's (t_end - t_1) / w_1 bounds rho = gap / max(w_1, w_2) only if w_2 <= w_1 (E14).
  - Next: full-bond centred runs at K >= 9 need a cheaper integrator (dopri5 with a Krylov stiff solver,
    or GPU); measure the boundary above the domain limit with a rule based on W - W_0.
- **Tensor-kernel labels** apply to sigma_chi_b4, chi_b4 and imbalance_b4 (and the report tables built
  from them). The K sweeps, N = 8 maps and stall tests keep the probe kernel.
- **Unfinished runs at the cut-off** (06:56 UTC; deep-S2S points were run last on purpose):

  | experiment | finished | what is missing |
  |---|---|---|
  | sigma_chi_b8 | 227/240 | |
  | sigma_chi_b8_d2 | 375/390 | 8/32 bonds at large sigma: 9 runs |
  | sigma_K_centred_d2 | 827/912 | full-bond binary 8 and MPS 8; binary 16 at sigma 0.76-0.92, so its boundary is only bracketed |
  | sigma_K_full_d2 | 220/228 | |
  | large_chi_b8 | 71/72 | |
  | sigma_K_d2_large | 333/408 | |
  | sigma_K_large | 67/168 | |
  | stall_test chi = 6 variants | 66/126 | |

  Finished runs are analysed as they are; the per-seed tables give the seed counts behind every number.
- **Design limits.**
  - The d = 3 sigma x K sweep was dropped beyond the 252 finished runs; the K dependence comes from
    d = 2.
  - The sqrt(d) limit is tested at d = 2 only.
  - Large-chi results are for N = 4; the K = 7 large-chi runs at d = 2 were noisy and are not
    interpreted.
- **Time budgets.** "Stalled" mixes three causes: floors (unrealizable), dead or long routes, and
  unfinished lazy fits at square cores. The notes separate them case by case. A fixed budget
  necessarily truncates the escape times that grow with K.
- **Interim numbers** of the first report (scipy, log-time grid) are superseded; they are kept in
  `results/interim/`.
- **Next steps.**
  - Test sigma_c -> sqrt(d) at d = 3 with spare channels.
  - Measure the effective amplitude c_K directly from first escapes.
  - Derive the chi^-1/2 scaling of the S2S boundary.
  - Run deeper trees at large chi under muP-TTN.
  - Add a two-copy S2S comparison at small chi.

## Log
- 18:18 started; read brief and notes; merged origin/main; machine has ~368 CPUs (cgroup quota), 125 GB RAM
- 18:25 theory draft in notes/regime_map.md: centred-model lazy limit only fits W* at maximal bond dimension (lazy residual), delta_K ~ kappa/sigma^K, rho ~ sigma^-(K-2), imbalance rule
- 18:30 pilot (scipy RK45) binary N=4 sigma x chi, 2 seeds: S2S at small sigma, lazy at large sigma for chi >= 9; final loss = best rank-chi approx of W0+W* (exact); delta_K matches kappa/sigma^3 to <1% at sigma >= 3 when converged
- 18:45 merged main (core fixes: teacher stream, cancellation-free centred loss); discarded old pilots; read literature.md (H4 corrections) and draft theory_regimes.md (agrees with my derivations; adds muP-TTN, exact aligned S2S boundary)
- 18:52 launched interim sweep b4_sigma_chi (N=4, centred/plain/two-copy, 8 chi x 19 sigma x 6 seeds = 2736 runs)
- 19:10 interim N=4 map: S2S / rich / lazy (chi >= 12) / stalled (chi < 9, unrealizable); per-seed S2S boundary predicted within one grid step by aligned-mode extrapolation from the deepest run; two-copy merges events 1-2 (mirror copies escape together)
- 19:20 merged library v0; wrote run_regime.py + sweep.py + YAML configs; found log-grid width resolution problem, added linear-time event refinement
- 19:54 launched production sweeps (library): sigma_chi_b8 (462), sigma_K (768), sigma_chi_b4 (2736); queued imbalance_b4, large_chi_b4, large_chi_b8
- 20:05 LSODA fails at huge sigma^K (K = 15, sigma = 10); added auto method (LSODA below sigma 0.7) with dopri5 fallback, max_steps caps, trimmed sigma ranges where deficient-bond runs only stall; relaunched (390 + 666 + 2736 runs)
- 20:15 interim report: N = 4 sigma x chi map with predicted S2S, lazy and realizability boundaries
- 20:25 all K >= 7 runs at chi = 3 (= teacher rank) stall; paused sweeps; added gauge-balanced init (balance.py); stall test (binary/MPS N = 8, fan-in vs balanced, chi = 3 vs 6, centred vs plain)
- 20:30 merged analysis standards + theory_regimes v2 (stalls at chi = R, centred floor at deficient bonds, second-order lazy boundary); revised configs: sigma x K in the plain model with chi = 3 and 6; chi_b4 to chi = 128; imbalance check point; added output-dir options, per-seed boundary functions, stall classification from bond spectra
- 20:38 stall test (plain, K = 7/8, chi = 3, 3 sigma x 6 seeds): fan-in 12/17 (binary) and 11/13 (MPS) dead routes (3rd bond singular value < 1e-12); balanced init: 0 dead but 12/17 still unconverged (route decaying, s_3 ~ 3e-5) - agrees with theory P14
- 20:55 library vs scipy integrators on 1937 matched N = 4 runs: labels agree 98.1%, delta_K 0.08%, drift <= 1e-9
- 21:02 load ~1900: LSODA workers spawned BLAS threads; pinned OMP/OpenBLAS/MKL threads to 1 in sweep.py and restarted (resumable)
- 21:08 plain-model convergence/stop floor now relative to 1/2 ||W*||^2 (decision 11); dropped 140 affected plain rows; relaunched all sweeps (5516 runs to do)
- 21:45 library N = 4 data: per-seed rho exponent on the S2S side -1.03 (seed sd 0.001-0.09) vs predicted -(K-2) = -1 (interim log-grid data had given -0.5..-0.7); continuous per-seed threshold sensitivity
- 21:55 heavy runs infeasible under load (binary16 chi = 6 Hessian 330 s, N = 8 chi = 81 gradient 0.2 s); stopped and restructured (decisions 13-14); relaunched 5749 runs
- 22:10 d = 3 K >= 10 runs at chi = 6 stopped at step caps mid-plateau; moved the K sweep to d = 2 (sigma_K_d2, 1428 runs, P <= 600, LSODA); relaunched
- 22:15 added full-bond K subset at d = 2 (sigma_K_full_d2); README rewritten
- 22:38 N = 8 runs 10-30+ min each and N = 4 studies queued behind them: restarted as two pools (N = 4: 140 procs; heavy: 215 procs) with trimmed N = 8 grids (sigma >= 0.3 at K = 7, chi = 12 dropped at d = 3, MPS 10 dropped from the full-bond set)
- 23:25 heavy pool was spending its slots on K >= 10 stall runs; restarted it with the N = 8 maps and K <= 8 first (220 procs); K >= 10 slices (step cap 1e4) chained to start when the N = 4 pool ends
- 23:45 plain-model K sweep: at large sigma the run first collapses W_0 (loss / (1/2||W*||^2) from 187 to ~1 by t ~ 0.1 at K = 15, sigma = 1.36) and then learns saddle-to-saddle from the small residual scale, so the plain-model S2S region extends beyond sigma = 1 (binary 16, chi = 4: S2S 5/6 at sigma = 1.36) - not a test of H4; added a centred K sweep at d = 2 (sigma_K_centred_d2), chained after the N = 4 pool
- 00:35 chi_b4 (N = 4, chi = 9..128, partial): lazy boundary 0.903 / 0.769 / 0.627 / 0.535 at chi = 48 / 64 / 96 / 128 vs theory P5 6.50 chi^-1/2 = 0.938 / 0.812 / 0.663 / 0.575 (two-copy 0.87 / 0.61 / 0.42 vs 0.91 / 0.65 / 0.46); threshold ratio 1.30-1.33 (P7); S2S boundary also ~chi^-1/2 at large chi: both collapse in sigma sqrt(chi) (S2S ~0.28, lazy ~6.1)
- 00:45 large_chi_b4 (partial): exponents of delta_K vs chi: SP -2.95 +- 0.16 (theory -3), NTK -0.76 +- 0.11 (-1), muP-TTN +0.15 +- 0.16 (0)
- 01:25 chi_b4 complete: lazy boundary slope -0.522 +- 0.020 (chi >= 32; theory -1/2), S2S boundary slope -0.524 +- 0.045 (chi 16-128): collapse in sigma sqrt(chi) (per-bond Gram scale); large_chi_b4 complete (sigma = 1.5): SP -3.09 +- 0.09, NTK -0.84 +- 0.06, muP-TTN +0.03 +- 0.06 for delta_K vs chi
- 01:37 started the centred sigma x K sweep (d = 2) alongside the N = 4 tail (two-copy deficient escapes at large sigma)
- 01:52 heavy pool too slow (~1350 runs left at ~4/min): dropped the remaining d = 3 sigma x K runs (K <= 8; 252 done, d = 2 sweeps cover K); restarted heavy pool with the N = 8 maps, the d = 2 plain and full-bond K sweeps and large-chi K = 7
- 02:55 centred loss with small W_0 switched to the direct form (decision 15, verified); restarted the heavy pool with the centred sigma x K sweep included (1885 runs)
- 04:55 heavy pool down to ~1 run/min (in-flight K = 10-12 deep-S2S centred runs); restarted it in sigma-descending order (boundary and lazy regions first, deep S2S points last)
- 06:20 centred sigma x K: boundary set by the per-component amplitude sigma^K / sqrt(D) ~ 0.01 -> sigma_c -> sqrt(d)
- 06:56 cut-off: stopped the remaining sweeps (deep-S2S points, some K = 7 large-sigma runs); listed unfinished runs per config
- 06:58 per-seed S2S brackets; summary figure; all figures regenerated from the final data; notes sections 0, 2.3, 2.6 and report finalized
- 12:22 ROUND 2 started (lead: realizable sigma_c(K) at large K, d = 2 and 3); merged origin/main (fix round 1: last-crossing events, tensor kernel, counters)
- 12:50 pre-registered hypotheses H_A / H_B / H_W and decision rule in notes section 4 (f57a28c); timing pilots running [E7: f57a28c is from 12:26:25 UTC, before any round-2 run; 12:50 is the arm-C launch]
- 12:34 pilots: plain binary 10/16 runs take 25-160 s (LSODA); found that at large sigma W_0 itself projects on the teacher components (a_r(0) up to ~40 [E8: the pilot had 8.85; arm P runs reach 50]), outside the event rule's domain -> runs with max|a_r(0)| >= 0.1 classified 'D' (amendment 4.6); longer time budget; forked pool (spawned 330 workers ran out of memory); arm P relaunched (156 walkers = 13 trees x 2 d x 6 seeds)
- 12:48 centred full-bond pilots with dopri5 did not finish in 20 min under load: arm C runs with LSODA on d = 2 K = 3-8 and d = 3 K = 3-4 only (48 workers); arm P at 325 runs: K >= 8 (d = 2) seeds are S2S-like up to the domain limit
- 12:54 arm P driver died (BrokenProcessPool: a worker was killed, cause unknown; 284 orphaned workers of the killed second launch were still alive, my kill loop had run after their parent died); Claude Code session restarted
- 13:33 arm C finished (48 walkers, all brackets resolved); killed the 284 orphans (by PID); arm P relaunched from its 390 rows with automatic pool restart, stop at 15:25 UTC
- 13:38 arm C (centred, full bonds; d = 2 K = 3-8, d = 3 K = 3-4): beta = -0.07 +- 0.51 (H_A 0.1 SE, H_B 2.1 SE: unresolved by the pre-registered rule); per-seed rho slopes -1.1, -2.0, -3.1, -4.8, -5.3, -6.6 for K = 3-8 (R9: -(K-2))
- 13:50 second arm P crash traced to run_regime.tensor_kernel (D x D identity, 2-9 GB per run at D >= 2^14): memory guard added, arm P skips the tensor kernel; relaunched
- 14:08 11 walkers stalled below the domain limit (walker step would leave the domain): fixed, arm P restarted (in-flight runs resubmitted); interim: at K >= 7 most plain-model seeds are S2S-like up to the domain limit (sigma_dom ~ 1.2-1.9 at d = 2, 1.6-2.4 at d = 3), above the H_A and H_B predictions
- 14:42 lead's extra items arrived: merged main (report scripts, verification); errata E1-E7 applied (notes 3.4, README, report body rewritten)
- 14:55 tensor kernel computed for all 5,298 N = 4 runs; relabel: 65 labels change (almost all lazy -> rich), lazy boundaries +4-9% [E18: 1-9%, centred 4-9%; 14-15% at chi = 6-9], imbalance map at chi = 24: lazy from alpha = 100 at balance / heavy root, 1 for a light root; report tables, numbers and figures regenerated
- 15:34 round-2 runs stopped at the 3 h budget (arm P 1,070 runs, arm C 360); final analysis: plain model S2S up to the domain limit at K >= 8 (97/156 seeds censored, all above both laws; H_W shrinkage confirmed); centred full bonds unresolved (beta = -0.07 +- 0.51) [corrected by the verification: 85/96 censored at K >= 8 (76 domain, 9 open), 11 cross; H_W consistent, not confirmed (E1-E3)]
- 15:36 notes 4.7-4.8, README, report finalized
