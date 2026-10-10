# data-teacher: do input covariance and teacher structure change the saddle-to-saddle picture? (Q6, Q7)

Status: working (round 2; interim report after item 1) - Branch: agent/data-teacher - Updated: 2026-10-09 21:55 UTC

## Summary
**Round 2 (in progress; interim report after item 1).**

- **Item 1, tree-shape contrast, the same generic teacher on different trees.** The saddle-start protocol
  (exact rank-1 saddle + noise ε) was run for 12 teachers on mps4/8, binary4/8 and ht4/8.
  - Binary and HT students escape like ε⁻ᵖ with p = 1.01–1.03 (predicted 1); the log law is never
    preferred.
  - MPS students follow a log law (preferred in 59/60 seeds). For 10 teachers the slope matches
    1/|λ_min| of the exact-saddle Hessian within 0.5–6%.
  - The exact-saddle Hessian is strict on every MPS student and has no negative eigenvalue on any
    binary or HT student.
  - Random-init sweeps on mps4 and binary4, and an ε extension (down to 1e-8 on MPS, 1e-5 on
    binary/HT), are running.
- **Item 3, correlated inputs.** The MPS-4 later plateau follows the predicted log law: 2.131 ± 0.013
  per halving of σ vs 2.128 predicted at ρ = 0.4; 14.89 ± 0.22 vs 14.39 at ρ = 0.2. Binary-4 stays σ⁻¹.
- **Item 2, stars.** The exact Hessians match the predicted λ_min to 1e-15 (−0.3716, −0.3310, −0.4311).
  The dynamics runs are queued.
- **Review of round 1.** The claims that changed are listed in `notes/data_teacher.md` section R2.0:
  - **Withdrawn:** the single-σ plateau durations (they were grid artefacts).
  - **Corrected:** the 4a slopes, now fitted per seed on major-saddle runs: 36.6 ± 1.2 vs 34.1 and
    5.98 ± 0.07 vs 5.68.
  - **Narrowed:** "critical point, not HSVD" (teacher 105 only), 4c (20/24 sibling-first; P11 tested
    only by teacher 203), early alignment (binary-4 only).
  - **Fixed:** escape degree, Λ slope, wordings.
  - **Reproducibility:** output and results directories are now configurable.

**Round 1 (complete; see the review response above).** Covariance aligned with an odeco teacher only
reweights: the race follows Λ·s and the landscape follows √Λ·s. Breaking orthogonality changes the
later plateaus of MPS students from σ^−(K−2) to log(1/σ).

## Results

Dynamics: gradient flow, float64. Steps 1 to 3 and single-sigma runs use scipy DOP853 (rtol 1e-9,
atol 1e-14), the pilot integrator. Sigma sweeps and TTN teachers use the library
`ttn.dynamics.integrate_gf` (LSODA, rtol 1e-9, atol 1e-14) on a dense two-stage grid. The two
integrators agree to 7e-7 relative (`results/crosscheck_library.csv`). Conservation drift is
< 1e-9 for the pilot and < 6e-10 for the library runs. Teachers use N = 4 (N = 8 in 4c), with 6 to 40
seeds per setting. Paths are relative to `experiments/data_teacher/`.

| claim | evidence | status |
|---|---|---|
| **1. Aligned covariance.** Aligned init: Pi_r(t; Sigma) = Pi_r^white(Lambda_r t) | half-times match the scalar ODE to 1e-9; the substitution s -> Lambda s gives widths wrong by up to 45% (`results/e1_exact_aligned.{csv,png}`) | derived + empirical |
| First escape = whitened problem with weights Lambda s | same first component in 95-100% of 960 paired runs; times within 0.02-0.7% (max 1.3%) (`results/e1_paired_agg.csv`, `e1_paired_times.png`) | derived + empirical |
| Order reversal by the covariance | smaller-s component first in 90-98% (binary-4) / 78% (MPS-4) of runs; P(first) identical to (I, Lambda s) (`results/e1_order_table.csv`, `e1_order_probs.png`) | empirical |
| Swap criterion = Lambda s^2 (landscape) | P(swap given the "wrong" first) = 0.33 / 0.68 / 0.94 at sigma = 0.1 / 0.05 / 0.02, P(swap given the "right" first) = 0; no swaps on MPS-4 (960 runs); exact-saddle Hessian lambda_min = derived formula to 1e-15, strict/non-strict correct in 24/24 (`results/e1_examples.png`, `e1_hessian_check.csv`) | derived + empirical |
| Race set by the flow metric, landscape by Fact B | same landscape and same start (mapped init): different first component in 10-53% of seeds, 11-23% slower | empirical |
| Exponents unchanged | t_1 ~ sigma^-1.010+-0.002 (binary-4), sigma^-1.999+-0.001 (MPS-4); t_1 ~ (Lambda s)^-0.96+-0.007 / -1.009+-0.001 over 3.6 decades (`results/e1_scaling{.png,_fits.csv}`) | empirical |
| **2. Misaligned covariance.** Plateau = rank-1 critical point of the Sigma-loss; early alignment with Sigma W*, then realignment | levels to 1e-6 - 1e-3; |cos| with race tensor 0.90-0.997 early, with saddle >= 0.998 on the plateau (`results/e2_misaligned_summary.png`) | empirical |
| Later plateau law changes on MPS-4 | theta = 0: sigma^-1.98+-0.005; theta = 20 deg: Dt_2 = 15.0(+-0.4) log(1/sigma) + B (`results/e2_sigma_law{.png,_fits.csv}`) | empirical |
| **3. Cross-site correlations.** S2S survives; plateaus = rank-1 critical points of the non-product loss | levels to <= 5e-3; binary-4's first saddle blends both components and non-teacher pairing features (level 0.33 -> 0.10 L(0) at rho = 0.5); MPS-4 later plateau shrinks 137 -> 0.14 (`results/e3_correlated_summary.png`) | empirical |
| Sampled data agrees | online SGD (eta 0.01/0.002, B 256/4096) vs exact population GF: event times within 0.2-1.4 (`results/e3_sgd_check.{png,csv}`) | empirical |
| **4a. Non-orthogonal CP.** Number of rank-1 saddles | MPS-4: 2 for c < 1/2, 1 above (equal weights; fold ~0.4 at s2 = 0.7); binary-4: always 1 (`results/rank1_predictions_*.csv`, `e4a_nonorth_summary.png`) | derived + numerical |
| Later plateau law (MPS-4) | c = 0: sigma^-1.97+-0.01; c = 0.2: A = 31.9 +- 1.6 (predicted 34.1); c = 0.45: A = 4.7 +- 0.3 (medians ~5.6; predicted 5.68) (`results/e4a_sigma_law{.png,_fits.csv}`, `e4a_theory_slopes.csv`) | derived (theory_teachers R6) + empirical |
| **4b. Random MPS teachers.** Saddles = critical points, not HSVD | rank-1 plateau 3x closer to the critical point (median 0.005 vs 0.016 of ||W*||); distance shrinks with sigma (`results/e4b_ttn_plateaus.csv`) | empirical |
| Bond-by-bond growth, log plateaus | slopes 2.8/2.9/0.65/5.9/3.3 vs predicted 2.5/3.0/0.87/5.2/3.7; first bond = predicted in 43-100% (`results/e4b_law.png`, `e4b_order_fits.csv`) | derived + empirical |
| **4c. Cross-tree.** Binary student, MPS teacher | bottom-up in 24/24 runs: sibling pairs first, root bipartition last; "largest teacher Schmidt value grows first" holds in 0/24 (binary student) and 6/24 (MPS student) (`results/e4c_cross.png`, `e4c_cross_summary.csv`) | empirical (confirms theory_teachers P11) |
| MPS student, binary teacher | first bond is the predicted one in only 13-38% of runs at sigma = 0.45, where log(1/sigma) = 0.8; the asymptotic regime is out of reach at K = 8 (`results/e4c_theory.csv`) | empirical (negative at accessible sigma) |

## What was done
1. Theory (`notes/data_teacher.md` sections 0-4):
   - Fact A: binary-4 is a depth-3 DLN.
   - Fact B: the whitened frame; this landscape map was also proposed by the lead.
   - Exact aligned-init solution.
   - Race vs landscape substitutions and the swap Hessian.
   - The rank-1 saddle count for non-orthogonal teachers.
   - The non-product moment metric for correlated inputs.

   Commits a613ec8, 0620017.
2. Tools:
   - config-driven runners `run_aligned.py`, `run_generic.py`, `run_dense.py`;
   - the rank-constrained critical-point finder (`constrained.py`);
   - exact Isserlis moment matrices (`correlated.py`);
   - analysis scripts.

   Commits 0e23928, f696ffe, 1d421c6, c8e7c28.
3. Step 1: 2800 runs plus checks; interim report. Commits 60a648d, 98c1ed5, 4a444ad.
4. Steps 2, 3, 4a at one sigma: 1600 runs. Commits 07d390c, d288514.
5. Library v0 cross-check and float64-floor scan. Plateau-law sigma sweeps and steps 4b/4c with the
   library integrator: 618 dense runs. Commits c8e7c28, c587d51, ec98d94, c1185e2, 0e08d43.

## Decisions
- **Pilot integrator first, library later.** I used scipy DOP853 on the flat vector (allowed by the
  brief) until library v0 landed, then cross-checked against the library (7e-7). Every
  measurement that needs to resolve short plateaus was redone with the library (LSODA, dense grid).
  Steps 1-3 and the single-sigma runs were not rerun: the cross-check makes that unnecessary.
- **Paired variants from the same init** (Sigma; I with Lambda s; I with sqrt(Lambda) s; the last from
  the mapped init). This separates race from landscape, and landscape from gradient-flow metric, as
  the lead asked. It is a much sharper test than comparing distributions.
- **Teachers with controlled conflicts** between the orderings by s, Lambda s and Lambda s^2.
  Covariance exponent 0.25 gives a Lambda ratio of 2, enough to reverse the order without stiffness.
- **binary-4 and MPS-4 as the two student trees.** binary-4 is a depth-3 DLN (only one cut), MPS-4
  is genuinely tensorial (three cuts). Many effects differ between the two, so both are always shown.
- **Correlated inputs via the exact moment matrix** (N = 4) rather than SGD only, plus an online-SGD
  check. The library `Problem` has no non-product metric, so step 3 stays on the pilot integrator.
- **Plateau detection for generic teachers.** The odeco baseline's event rule (a_r = 1/2) has no
  meaning for blended saddles. I used bond-spectrum and loss-level events and speed minima
  (diagnostics.md), with the theory note's small fixed s_2 threshold where it applies. At
  sigma = 0.3 the init's own singular values are ~1e-2, so I switched to loss levels there; this is
  documented in each script.
- **Effective rank at plateaus** counts s_k^e > 0.1 x the teacher's own s_k^e, instead of 0.1 x the
  smallest teacher value: with teacher values ~1e-2, the diagnostics rule counted still-rising
  singular values.
- **4c at moderate sigma** (0.4/0.45 for K = 7/8). sigma^K <= 2e-3 is small in function space, but
  log(1/sigma) < 1, so log-time races are not in their asymptotic regime. Smaller sigma costs
  ~sigma^-(K-2); reported as a limitation.
- **Deleted my own superseded outputs:** the pilot versions of the sigma sweeps (under-resolved) and
  stale result files. Nothing created by others was touched.

## Files left on the machine
All under `/root/outputs/data_teacher/` (7.2 GB total). None of it is needed to read the results:
summary CSVs and figures are in git. All of it is regenerated by the commands in
`experiments/data_teacher/README.md` (about 1.5 h on 28 cores).
- `e1_order/` 2.7 GB, `e1_scaling_sigma/` 0.4 GB, `e1_scaling_lambda/` 0.6 GB, `e1_exact/`: step 1.
- `e2_misaligned/` 0.9 GB, `e3_correlated/` 0.3 GB, `e4a_nonorth/` 0.7 GB: single-sigma runs.
- `e4a_sigma_dense/` 0.56 GB, `e2_sigma_dense/` 0.41 GB, `e4b_ttn_dense/` 0.25 GB,
  `e4c_cross_dense/` 0.17 GB: library-integrator runs (`Trajectory` npz with parameters).
- `e4b_ttn/`, `e4c_cross/` (0.25 GB): pilot runs, superseded by the dense ones.
- `e3_sgd/`, `pilots/`, `rank1_predictions_*.pt` (predicted critical tensors).

## Caveats and next steps
- **Theory dependence.** The slope predictions come from `notes/theory_teachers.md`, a draft still under
  verification. My agreement there is 3-25% (4b), within the error bar (4a, c = 0.2), and ~20%
  (4a, c = 0.45). The derivations in `notes/data_teacher.md` (Facts A/B, aligned solution, swap
  Hessian, c* threshold) are mine and are checked numerically.
- **Untested sigma law.** For correlated inputs the later-plateau law in sigma was not measured: no
  non-product metric in the library. That it is logarithmic is a conjecture from the same mechanism.
- **Single-sigma durations.** Durations below ~1 time unit (steps 2, 3, 4a; pilot grid) are
  resolution-limited. The sigma sweeps on dense grids supersede them.
- **Plateau detection in 4b** found 42 plateaus in 150 runs. Later plateaus are short at these sigma.
  HSVD and critical points differ only at rank 1 for these teachers.
- **4c, MPS student**: the asymptotic first-bond rule is not reached at accessible sigma.
- **Next steps:**
  - extend the sigma-law measurement to binary-8/HT students, where theory_teachers.md predicts
    polynomial later plateaus even for generic teachers;
  - run the misaligned case through a preconditioned version of R6 to predict its slope;
  - measure a sigma law for correlated inputs with a dense pilot grid.
- **Out of scope but relevant to `order`/`locality`:** the swap in step 1 is the H1b correction, with
  the criterion generalized to Lambda s^2. Its probability rises toward 1 as sigma -> 0 on binary-4
  (strict saddle vs polynomial learning).

## Log
- 18:20 Read brief, project map, model, diagnostics, plan, library code. Machine: CPU quota 30.7
  vCPUs (cgroup cpu.max 3071999/100000), 503 GB RAM. Merged origin/main (report skeleton).
- 18:22 Theory for aligned covariance written (notes/data_teacher.md s.1): exact per-component
  clock Lambda_r t for aligned init; first escape = whitened problem with weights Lambda s; swap
  (piggyback) criterion = Lambda s^2 (function-space norm). binary-4 = depth-3 DLN (Fact A).
- 18:25 Pilot (8 seeds, binary-4 + MPS-4) confirms: first escape matches (I, Lambda s) paired runs;
  binary-4 shows swaps exactly when Lambda s and Lambda s^2 orderings disagree; MPS-4 never swaps.
- 18:27 E1.1 exact aligned-init check: TTN GF matches scalar ODE to 1e-9 in half-times; s -> Lambda s
  substitution gets half-times to 1-2% but widths wrong by up to 45%.
- 18:30 Launched e1_order sweep (2 trees x 3 sigma x 2 teachers x 4 variants x 40 seeds).
- 18:44 Lead messages: literature v1 merged; added variant "white_sqrtLs_mapped" (absorbed white
  problem from the mapped init: same landscape and start, different GF metric) to separate landscape
  from flow metric. Library fix 57d5262 merged: my teachers (standard basis, deterministic) and
  spec.init are unaffected (checked: saved init identical to spec.init after merge).
- 18:50 Theory drafts merged (theory_teachers.md covers step 4 predictions P1-P11); step 2 and 4a
  configs written, rank-1 prediction tool (constrained.py) tested.
- 18:55 Recorder bug found (terminal state not stored when the stop event fell between output
  points; 99 step-1 runs missed their last event); fixed, affected runs rerun.
- 19:00 Step-1 sweeps done (order 1920 runs, sigma scaling 360, Lambda scaling 520); analysis and
  interim report. Steps 2-4 sweeps queued.
- 19:10 Steps 2 (800 runs), 4(a) (560), 3 (240) done and analyzed: plateaus = rank-1 critical points
  of the (Sigma- or M-) loss in every case; misalignment / overlap / correlation collapse the MPS-4
  rank-1 plateau (2-core strict-saddle route), not binary-4's.
- 19:20 Library v0 merged. Cross-check pilot vs ttn.dynamics.integrate_gf (LSODA): max relative
  difference 7e-7 in event times and losses (results/crosscheck_library.csv). Float64-floor scan
  (lead's note): no dormant signal below 1e-11 in any CP run (results/float_floor_check.csv).
- 19:25 Found that the 2000-point log grid cannot resolve the short (log-time) later plateaus of
  generic teachers; new two-stage dense runner on the library integrator (run_dense.py); 4(b), 4(c)
  and the sigma sweeps (e4a_sigma, e2_sigma) rerun with it (pilot versions discarded).
- 19:35 Plateau laws: e4a_sigma and e2_sigma (dense, library LSODA) show polynomial -> logarithmic
  change with overlap / misalignment, slopes as predicted. Exact Hessian check of the swap criterion.
- 19:50 Steps 4(b), 4(c) analyzed (critical points not HSVD; log plateaus with predicted slopes;
  bottom-up order on binary students). Summary table, overview figure, README, final report.
- 19:58 Round 2 started (lead: tree-shape contrast for generic teachers; star teachers; correlated-input
  sigma law). Merged main (theory_saddles v2). Saddle-start protocol written (exact balanced rank-1
  saddle + sigma noise, library LSODA); exact-saddle Hessians: strict on mps4, non-strict on binary4
  and ht4 for all three teacher types; test run: mps4 slope 6.6 vs 1/|lambda_min| = 6.63, binary4 and
  ht4 sigma^-1. Full sweep launched (6 trees x 12 teachers x 6 sigma x 5 seeds).
- 20:15 Lead: theory_teachers v2 / theory_covariance v2 merged. Exact-saddle Hessians for 36 (teacher, tree)
  combos: strict on every MPS student, non-strict on every binary/HT student.
- 20:40 Review of round 1 received and merged. Reproducibility fixes (DATA_TEACHER_RESULTS/OUT, PROCS);
  re-analysis tooling: lawfit.py (per-seed, same-space log vs power), analyze_law_v2.py (saddle
  classification, unresolved flags, plateau Hessian), analyze_cross_v2.py (f-rule events); sigma
  extensions queued (saddle-start eps to 1e-8 / 1e-5; random init to 3e-4 / 1e-4); new 4c teachers.
- 21:00 Correlated-input sigma law (dense pilot LSODA) done: MPS-4 increments 2.131 / 14.89 per halving
  vs 2.128 / 14.39 predicted; binary-4 sigma^-1. Escape-degree and Lambda-slope re-analyses done.
- 21:30 Lead: diagnostics rules (f = 0.05 onset events; sweep-level saddle type), final library
  (fine rows / merged_rows), gauge-projection fix merged. Two-stage runner fixed for fine rows;
  exact-saddle Hessian tables recomputed with the fix (unchanged; raw = projected); raw lambda_min now
  reported everywhere.
- 21:55 Saddle-start sweep (1080 runs) analyzed: tree contrast confirmed for 12 teachers (table in
  notes R2.1). Interim report.
- 22:00 Random-init item-1 sweep (mps4, binary4, 3 decades) analyzed: mps4 log law with R6 slopes
  (0.1-6%), strict plateau lambda_min; binary4 sigma^-1, lambda_min ~ sigma.
- 22:20 Balanced N = 8 misaligned teacher: MPS A = 57.5 vs 57.4 predicted; binary8/ht8 sigma^-1.
- 22:40 Saddle-start eps extension: MPS slopes over 6.7 decades within 0.04-4% of 1/|lambda_min| for 12
  of 13 teachers (nonorth8_c0.45: +10%, slow crossover). Review text corrections in the notes.
- 23:30 eps extension for binary/HT (to 1e-5) still running; queued: round-1 sigma extensions (4a, 2,
  4b), star dynamics, new 4c teachers, resolution check.
- 00:35 Item 1 final: saddle-start over 6.7 decades (MPS) / 3.7 decades (binary, HT). Binary/HT p =
  0.999-1.013 in all 26 cases (log never preferred); MPS log law in 65/65 seeds, A within 0.04-4% of
  1/|lambda_min| for 11 teachers (nonorth8_c0.45 +11%, slow crossover; misaligned8_20 degenerate).
  14 very long ht8 runs (nonorth8_c0.45, eps <= 3e-4, t > 1e8) stopped to free the machine.
