# data-teacher: do input covariance and teacher structure change the saddle-to-saddle picture? (Q6, Q7)

Status: done (round 2 and the fixes requested by the round-1 review) - Branch: agent/data-teacher - Updated: 2026-10-10 01:25 UTC

## Summary
**Round 2** (lead's three items, plus the review of round 1). Notes: `notes/data_teacher.md` sections R2.0–R2.3;
summary table at the top of that file.

1. **Tree-shape contrast for generic teachers.** For 13 generic teachers (non-orthogonal CP,
   misaligned covariance, random TTN teachers; N = 4 and 8), the escape from the exact rank-1 saddle
   (saddle-start protocol, ε over 6.7 decades on MPS and 3.7 on binary/HT):
   - **MPS students** (mps4, mps8): strict saddle, log law in 65/65 seeds. Slope = 1/|λ_min| of the
     exact-saddle Hessian within 0.04–4% for 11 teachers. One slow crossover is at +11%; one teacher is
     degenerate and was replaced.
   - **Binary and HT students** (binary4/8, ht4/8): no negative eigenvalue at the saddle, and the escape
     time scales as ε^−p with p = 0.999–1.013 (prediction 1). The log law is never preferred.
   - Random init on mps4/binary4 over 3 decades of σ gives the same contrast. mps4 slopes match R6
     (with head start) within 0.1–6%. The plateau λ_min is σ-independent on mps4 and ∝ σ on binary4
     (the sweep-level saddle-type rule).
2. **Star teachers.**
   - The exact Hessian at "q learned" matches −a^(K−2)(ρ(S) − s_q) to 1e-15 (−0.3716, −0.3310,
     −0.4311).
   - The threshold ρ(S) = s_q is confirmed to 3e-4 at ε = 1e-6. The escape from θ_q is log-time with
     slope 1/|λ_min| (0.2%).
   - With random init, super-critical stars forget q down to min a_q ∝ σ^(5.7 ± 0.3), and sub-critical
     ones never do.
   - New: on ht4 the forgetting stalls at a blended rank-1 point that is left only in ~1/ε time, the
     same MPS/HT contrast as item 1.
3. **Correlated inputs.**
   - On MPS-4 the second plateau follows the predicted log law: 2.131 ± 0.013 per halving of σ at
     ρ = 0.4 (predicted 2.128), 14.89 ± 0.22 at ρ = 0.2 (predicted 14.39).
   - Binary-4 stays σ⁻¹.
   - The plateaus stay separated on MPS-4, but only logarithmically (separation ratio 2.6 → 11 at
     ρ = 0.4).

**Review of round 1** (`notes/reviews/data_teacher_round1.md`). Section R2.0 of the notes lists every
claim that changed:
- **Withdrawn:** the single-σ plateau durations of steps 2, 3, 4a ("540 → <1", "137 → 0.14", …). They
  were grid artefacts.
- **Corrected:** 4a slopes, now per seed over 3 decades of σ, major-saddle runs only: 5.76 ± 0.04
  (predicted 5.68) and 36.5 ± 0.8 (predicted 34.1).
- **4b and the HSVD claim:** slopes are per seed by the edge that grows first (within 0.1–3%).
  "Critical point, not SVD-type truncation" is shown for 3 of 5 teachers; for the other two a TT-SVD
  truncation is within 0.003 of the critical point.
- **4c:** P11 now holds 125/128 over 6 discriminating teachers at 3 σ. "Sibling pair first" does not
  generalize.
- **Narrowed:** early alignment then realignment (binary-4 only).
- **Fixed:**
  - escape degree (now 1.32 and 1.465 at the smallest σ, vs 2 − 2/K);
  - Λ slope (−1.000 within each teacher direction);
  - swap wording;
  - float-floor scan on the dense runs (only the odeco baselines at σ ≤ 3e-3 are flagged; refits
    unchanged).
- **Reproducibility:** output and results folders are configurable, `PROCS` is an option, and the run
  order is documented.

**Round 1 in one line.** Covariance aligned with an odeco teacher only reweights: the race follows the
whitened teacher with weights Λ·s; the landscape follows the one with √Λ·s. Making the teacher's
components non-orthogonal (misalignment, CP overlap, correlations, generic TTN teachers) changes the law
of the later plateaus on MPS students from σ^−(K−2) to log(1/σ). Binary and HT students keep σ⁻¹.

## Results
Dynamics:
- **Round 2 and every re-analysis:** gradient flow on the library integrator (`integrate_gf`, LSODA,
  rtol 1e-9, atol 1e-14), with event times from merged fine rows.
- **Correlated inputs:** scipy LSODA with the exact Hessian on the dense moment-matrix loss.
- **Round-1 pilot runs:** scipy DOP853. Cross-checked against the library to 7e-7.
- Log-row vs fine-row escape times agree to ≤ 0.5%.

| claim | number | evidence (paths under `experiments/data_teacher/`) | status |
|---|---|---|---|
| Exact rank-1 saddle strict on MPS, non-strict on binary/HT | 13 teachers × 3 trees | `results/r2_saddle_hessian.csv`, `r2_saddle_b_hessian.csv` | derived + computed |
| MPS escape: log law, slope 1/\|λ_min\| | 65/65 seeds log; 11 teachers within 0.04–4% | `results/r2_saddle_fits.csv`, `r2_saddle_collapse.png`, `r2_saddle_law.png` | empirical, confirms R9/R13 |
| Binary/HT escape: ε^−1 | p = 0.999–1.013, 26 tree × teacher cases | same | empirical |
| Random init: mps4 log law (R6), binary4 σ⁻¹, plateau λ_min scaling | slopes within 0.1–6%; λ_min const vs ∝ σ | `results/r2_random_v2_summary.csv`, `r2_random_v2.png` | empirical |
| Correlated inputs, MPS-4 increments per halving | 2.131 ± 0.013 (2.128), 14.89 ± 0.22 (14.39) | `results/r2_corr_fits.csv`, `r2_corr_law.png` | empirical, confirms theory_covariance |
| Star Hessian / threshold / escape rate | 1e-15 / 3e-4 / 0.2% | `results/r2_star_hessian.csv`, `r2_star_threshold_scan.csv`, `r2_star_rotation_law.csv`, `r2_star.png` | derived + empirical |
| Star forgetting depth (random init) | min a_q ∝ σ^(5.7 ± 0.3) (mps4), σ^(5.4 ± 0.4) (binary4) | `results/r2_star_depth_exponents.csv` | empirical |
| ht4 star: two-stage forgetting | strict rotation in ~40 time units, then ∝ 1/ε | notes R2.2 item 4 | empirical |
| 4a log law (round 1, re-analysed) | A = 5.76 ± 0.04 (5.68), 36.5 ± 0.8 (34.1); c = 0: p = 2.004 | `results/e4a_sigma_v2_summary.csv` | empirical |
| 4b plateaus → rank-constrained critical point, not SVD truncations | distance ∝ σ^0.4–0.85; separated for 3/5 teachers | `results/e4b_ttn_plateaus.csv` | empirical |
| 4c P11 (root bipartition last) | 125/128 runs, 6 discriminating teachers | `results/r2_cross_v2_summary.csv`, `e4c_cross_v2_summary.csv` | empirical |
| Round-1 step-1 results (race Λ·s, landscape √Λ·s, swaps) | unchanged (see the round-1 table in the notes) | `results/e1_*` | derived + empirical |

## What was done (round 2)
1. Merged library and theory updates as announced. Saddle-start protocol (`r2_common.py`,
   `r2_saddle_runs.py`): 1935 completed runs, ε to 1e-8 / 1e-5. Commits 9ec283a…eee6deb.
2. Random-init item-1 sweep (`configs/r2_random.yaml`, 540 dense runs). Commit ca563e7.
3. Correlated-input dense runner and σ law (`r2_corr.py`, 300 runs). Commit f1f9c9d.
4. Stars (`r2_star.py`: Hessian, 463 saddle-start runs including a 48-run threshold scan, 240 random-init runs).
   Commit a8b93d3.
5. Review fixes:
   - `lawfit.py`, `analyze_law_v2.py`, `analyze_cross_v2.py`, TT-SVD comparison;
   - σ extensions of 4a, 2, 4b (300 new dense runs) and new 4c teachers (120 dense runs);
   - float-floor and resolution checks;
   - reproducibility options;
   - notes and README.

   Commits e3b6ad8, a08a658, 0e1b9ba, c4b9087, 90f366d and earlier.

## Decisions
- **d = 2 for all round-2 teachers.** These are the theory notes' conventions, and they keep N = 8
  cheap.
- **N = 8 teacher choices:**
  - Overlaps c = 0.45 and 0.7 instead of 0.2, because the MPS 2-core rate scales like c^(N−2).
  - A balanced misaligned teacher (θ = 45°, α = 2). The θ = 20° one is degenerate over 8 sites; it is
    kept in the tables and flagged.
- **Random TTN teachers:** binary-tree teachers without decay, plus one MPS teacher with decay 1.5.
  Random MPS teachers with decay < 1 are nearly rank 1.
- **Saddle-start protocol for every tree** (the lead's suggestion). Random init was used where the
  asymptotic regime is reachable (mps4, binary4).
- **Escape event = onset rule** s₂ᵉ ≥ 0.05·s₂ᵉ* (diagnostics.md), with ≥ 10× growth over the start
  value. The sensitivity over f = 0.03–0.1 is < 1.5% on MPS slopes.
- **Both laws fitted and scored in log space, per seed.** Errors are standard errors over seeds; runs
  are classified by visited saddle, and unresolved ones are excluded and counted.
- **Stopped very long runs** that could not change a conclusion:
  - 14 ht8 runs (c = 0.45, ε ≤ 3e-4, t > 1e8);
  - the last degenerate misaligned8_20 run;
  - 5 ht4 star runs at the threshold (ε = 1e-6), whose second, polynomial stage takes ~1e9.

  Each is noted in the notes.
- **Star "leaving q" event = a_q < 0.9.** On ht4 the strict escape ends at a_q ≈ 0.67, so a_q < 1/2
  would time the second, polynomial stage.
- **BLAS threads capped at 1 per process** (`env1.sh`). LSODA otherwise oversubscribed the machine
  (load ~175).
- **Superseded files** (the old overview figure and round-1 analysis CSVs): the figure was deleted (it
  is in git history); the CSVs are kept and labelled in the README.

## Files left on the machine
All under `/root/outputs/data_teacher/` (9.5 GB). None of it is needed to read the results: the
summary CSVs and figures are in git. All of it is regenerated by the commands in
`experiments/data_teacher/README.md`.
- Round 2: `r2_saddle/` (1.1 GB, 1935 saddle-start runs incl. supplement and extension),
  `r2_saddle_fine/`, `r2_random_dense/`, `r2_corr/`, `r2_star_saddle/`, `r2_star_random/`,
  `r2_cross_dense/`.
- Round 1 and its extensions: `e1_order/` (2.7 GB), `e1_scaling_*`, `e2_misaligned/`,
  `e3_correlated/`, `e4a_nonorth/`, `e4a_sigma_dense/`, `e2_sigma_dense/`, `e4b_ttn_dense/`,
  `e4c_cross_dense/`.
- Superseded: `e4b_ttn/` and `e4c_cross/` (pilot versions).
- Rank-1 predictions: `rank1_predictions_*.pt`.

## Caveats and next steps
- **Slow crossover:** nonorth8_c0.45 on MPS-8 (rate ∝ c⁶) still sits +11% above its asymptotic slope
  at ε = 1e-8.
- **Unfitted cases:** two random mps4 teachers (randbin_303, randmps_301) have unresolved rank-1
  plateaus with random init in most runs (large κ), so their R6 fits rest on 1–3 seeds.
- **4c:** at K = 7–8 the MPS-8 student's first-bond rule is not reached at accessible σ. P11 is tested
  at σ = 0.3–0.5 only.
- **Unpredicted laws:** the star forgetting-depth law (σ^5.7) and the relearning time (~σ^−1.4) are
  empirical. No theory predicts them yet.
- **Correlated inputs:** studied at N = 4 only (dense moment matrix, d = 3).
- **Theory still under review:** slope predictions come from theory_teachers v2, theory_saddles v2
  and theory_covariance v2.
- **Next:**
  - a theory for the forgetting depth;
  - correlated inputs at N = 6 (moment matrix 4096², feasible);
  - a sweep-level saddle-type test for the ht8 star.

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
- 00:33 Round-1 4a / 2 re-analysed over 3 decades (per seed): A = 5.76 / 36.5 vs 5.68 / 34.1; theta = 20: 12.6.
- 01:13 Stars: threshold to 3e-4, escape slope 1/|lambda_min| to 0.2%, ht4 two-stage forgetting, depth ~sigma^5.7.
- 01:16 4b (sigma to 1e-3): per-seed slopes by realized edge within 0.1-3%; critical point vs all SVD-type
  truncations separated for 3/5 teachers. 4c with 5 discriminating teachers: P11 117/120.
- 01:24 Float-floor scan on dense runs (odeco baselines at sigma <= 3e-3 flagged, refits unchanged);
  resolution check (log rows vs fine rows <= 0.5%). README, notes, report finalized.
- 01:25 Report checked against the raw outputs (run counts, commit range, log times corrected). Done.
