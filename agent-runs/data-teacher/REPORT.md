# data-teacher: do input covariance and teacher structure change the saddle-to-saddle picture? (Q6, Q7)

Status: done (fix round 2: response to the review of round 2) - Branch: agent/data-teacher - Updated: 2026-10-10 10:11 UTC

## Summary
**Fix round 2** (review: `notes/reviews/data_teacher_round2.md`). The review reproduced every group of
claims and found no wrong result. This round changes how slopes are fitted, fixes several counts and
labels, rereads the star random-init runs and narrows the scope statements. **The list of claims that
changed is section R2.4 of `notes/data_teacher.md`**; the main ones:
- **Slopes are now asymptotic per-seed fits** (a line through the 3 smallest σ or 5 smallest ε, the
  theory_teachers §9 protocol). The global fits over all σ included pre-asymptotic points. With the
  asymptotic fits the agreement with theory is tighter:
  - saddle start, MPS students: slope = 1/γ of the edge that actually grows, to ≤ 0.4% for all 13
    teachers (the "+11% slow crossover" of nonorth8_c0.45 is withdrawn: −0.05%);
  - random init, mps4: within 1.3% of R6 (with the head start) for all 5 teachers with a prediction,
    5 seeds each (c = 0.2: 33.95 vs 34.14, was 36.3);
  - 4a: c = 0.2 34.40 vs 34.14 (+0.8%, was +7%); step 2, θ = 5°: 194.9 vs the bound 193.2 (was 251);
  - binary / HT students: ε^−1 with p = 0.9998–1.0036 (24 cases).
- **misaligned8_20 is not degenerate.** Its escape ends at an intermediate saddle whose s₂ is below the
  f = 0.05 event threshold, so the event timed the next escape. Re-timed at that level: slope 1979.6 vs
  1974.8 (+0.24%). This is a failure mode of the f-rule worth knowing.
- **4c / P11 count corrected.** The "discriminating" flag was too loose and one set of runs was counted
  twice: 46/48 runs on 2 discriminating teachers. 120 new runs with 5 teachers whose root cut is the
  strongest cut (a weight-order rule predicts the root first): P11 holds in 117/120. Total over the
  7 discriminating teachers: **163/168**.
- **Stars:**
  - The random-init "forgetting depth σ^5.7" is withdrawn. Those runs never sit at "q learned", two
    kinds of runs were pooled, and the decay of a_q is steep but not a power law.
  - New: near the threshold the mps4 escape from θ_q takes two strict stages; the slope is
    1/|λ_q| + 1/|λ_b| to 0.1–0.2%. This explains the +50% deviation seen in round 2.
  - The ht4 threshold bracket at ε = 1e-6 is completed.
- **Correlated inputs:** errors are now per seed. Increments per halving: 2.133 ± 0.003 (predicted
  2.128) and 14.67 ± 0.06 (14.39, still falling). This replicates theory_covariance N14.
- **Scope:** "binary / HT keep σ⁻¹" and "MPS log-time" are measured for the escape from the rank-one
  saddle only (N = 4, 8).
- **Fixed:** Hessian labels by sign, realized-edge predictions, P1 exponents for all 4b teachers,
  float-floor scan of the star and correlated runs, stored saddles (reproducibility), spawn-safe
  runners, README run order.

**Answer to Q6/Q7.**
- *Aligned covariance:* with an odeco teacher it only reweights. The race follows Λ·s, the landscape
  √Λ·s.
- *Non-orthogonality in the loss metric:* misalignment, CP overlap, cross-site correlations and generic
  TTN teachers all make the rank-one saddle of MPS students strict. It is escaped in log(1/σ) time with
  slope 1/γ of the growing edge, instead of σ^−(K−2).
- *Binary and HT students:* for the same teachers they keep a higher-order rank-one saddle with σ⁻¹
  escape (a 3-core route through an internal core).
- *Plateaus:* they sit at critical points of the bond-rank-constrained loss, not at SVD-type
  truncations.
- *Binary students and cross-tree teachers:* they grow the root bipartition only after a bottom bond on
  each half (P11).
- *Star teachers:* they lose a learned component exactly when ρ(S) > s_q. This happens in log time on
  MPS students and in polynomial time on HT students.

Status: empirical, confirming the theory notes. The tree contrast is shown for the escape from the
rank-one saddle.

## Results
Dynamics: gradient flow with the library integrator (`integrate_gf`, LSODA, rtol 1e-9, atol 1e-14) and
events from merged fine rows; correlated inputs: scipy LSODA with the exact Hessian on the dense
moment-matrix loss. Paths are under `experiments/data_teacher/`.

| claim | number | evidence | status |
|---|---|---|---|
| Rank-1 saddle strict (by sign) on MPS, no negative eigenvalue on binary / HT | 13 MPS (5 / 17 negatives on mps4 / mps8), 26 binary / HT | `results/r2_saddle_hessian.csv`, `r2_saddle_b_hessian.csv` | derived + computed |
| Saddle start, MPS: log law, slope 1/γ of the realized edge | ≤ 0.4% for 13 teachers (asymptotic, ε ≤ 1e-4); log law preferred in 65/65 seeds | `results/r2_saddle_fits.csv`, `r2_saddle_perseed_local.csv`, `r2_saddle_collapse.png` | empirical, confirms R9 / R13 |
| Saddle start, binary / HT: ε^−1 | p = 0.9998–1.0036 (24 cases, ε ≤ 1e-3) | same | empirical |
| misaligned8_20 re-timed (intermediate saddle) | 1979.6 ± 1.8 vs 1974.8 | same; `configs/r2_saddle_m20.yaml` | empirical |
| Random init mps4 = R6 with head start; binary4 σ⁻¹ | within 1.3% (5 teachers); p = 1.001–1.005 | `results/r2_random_v2_summary.csv`, `r2_random_v2.png` | empirical |
| 4a, step 2 (round 1, asymptotic) | 34.40 ± 0.20 (34.14), 5.686 ± 0.016 (5.682); θ = 5°: 194.9 ± 1.2 (bound 193.2) | `results/e4a_sigma_v2_summary.csv`, `e2_sigma_v2_summary.csv` | empirical |
| 4b single-route slopes; P1 exponent | within 0.3–1.6% of R6; measured α 0.07–0.15 above min(α_r, α_a), same order over 5 teachers | `results/e4b_ttn_v2_summary.csv`, `e4b_p1_exponents.csv` | empirical |
| 4c P11 (root bipartition after a bottom bond on each half) | 163/168 runs over 7 discriminating teachers (117/120 with the root cut strongest) | `results/p11_totals.csv`, `p11_counts.csv` | empirical |
| Correlated inputs, MPS-4 increments per halving | 2.133 ± 0.003 (2.128), 14.67 ± 0.06 (14.39) | `results/r2_corr_fits.csv`, `r2_corr_increments_perseed.csv` | empirical (replicates N14) |
| Star Hessian at θ_q, threshold ρ(S) = s_q | 1e-15; to 3e-4 (mps4, binary4), between w = 1.40 and 1.42 (ht4) | `results/r2_star_hessian.csv`, `r2_star_threshold.csv` | replication (C11) + empirical |
| Star escape from θ_q | 1/\|λ_q\| to ≤ 0.2% (one stage); 1/\|λ_q\| + 1/\|λ_b\| to 0.1–0.2% (mps4 near threshold); ε^−1 stage on ht4 | `results/r2_star_rotation_law.csv`, `r2_star_blended.csv` | derived + empirical |
| Star random init | never sits at "q learned"; seed classes; steep decay of a_q, no power law | `results/r2_star_random_classes.csv`, `r2_star_depth_exponents.csv`, `r2_star.png` | empirical |
| Float64 floor | only the odeco baselines at σ ≤ 3e-3 flagged (refits unchanged); star / correlated runs: no flag | `results/float_floor_check.csv` | check |
| Round-1 step 1 (race Λ·s, landscape √Λ·s, swaps) | unchanged | `results/e1_*`, notes §1 | derived + empirical |

## What was done
**Fix round 2** (after `notes/reviews/data_teacher_round2.md`):
1. Re-analyses: asymptotic per-seed fits (`lawfit.per_seed_local`) in `analyze_r2_saddle.py`,
   `analyze_law_v2.py` (r2_random, e4a_sigma, e2_sigma, e4b_ttn), `analyze_r2_star.py`; realized-edge
   predictions from per-edge Hessian rates; sign-based saddle labels; per-seed increments for the
   correlated inputs; P11 flag fixed and runs de-duplicated (`count_p11.py`); float-floor scan extended.
   Commit a9bbfa9.
2. New runs:
   - 120 step-4(c) runs with 5 teachers whose root cut is the strongest (`configs/r2_cross_b.yaml`,
     chosen by `scan_cross_teachers.py`);
   - 15 misaligned8_20 saddle-start runs at ε = 1e-6–1e-4 (`configs/r2_saddle_m20.yaml`);
   - 5 ht4 star runs at w = 1.42, ε = 1e-6.

   New analyses: `star_blended.py` (two-stage escape near the star threshold) and `p1_exponents.py`
   (P1 exponent per 4b teacher). Commit 2348106.
3. Reproducibility:
   - the exact saddles are stored (`results/r2_saddle_cores.pt`); recomputing them on this machine
     reproduces committed escape times to 2e-10;
   - run parameters are passed with each job (spawn-safe);
   - 1 torch thread everywhere;
   - `run_round2.sh` replaces `r2_chain*.sh`; README run order and methods updated.
4. Notes rewritten: R2.1–R2.3, the new R2.4 (claims changed), the summary table and the R2.0 rows.
   Commit d4cb229. README and this report: final commits.

**Round 2** (earlier):
1. Saddle-start protocol (`r2_common.py`, `r2_saddle_runs.py`): 1935 runs, ε to 1e-8 / 1e-5. Commits
   9ec283a…eee6deb.
2. Random-init sweep (`configs/r2_random.yaml`, 540 dense runs). Commit ca563e7.
3. Correlated inputs (`r2_corr.py`, 300 runs). Commit f1f9c9d.
4. Stars: 463 saddle-start runs incl. a 48-run threshold scan, 240 random-init runs. Commit a8b93d3.
5. Round-1 review fixes: σ extensions (300 dense runs), new 4c teachers (120 runs), float-floor and
   resolution checks, reproducibility options. Commits e3b6ad8, a08a658, 0e1b9ba, c4b9087, 90f366d.

## Decisions
**Fix round 2:**
- **Asymptotic window: the 3 smallest σ (random init) or the 5 smallest ε (saddle start).** This is the
  review's and theory_teachers §9's protocol. The global fits stay in the CSVs, labelled.
- **misaligned8_20 is re-timed, not replaced.** The event is s₂ ≥ 1.8e-4 (half the intermediate level),
  used only where s₂(0) ≤ 1.8e-5. I added 15 runs at ε ≤ 1e-4 so the fit covers 3.7 decades; the
  ε ≥ 1e-2 runs start above the level and are excluded.
- **P11 "discriminating"** means the root s₂ beats both bottom bonds of at least one half, as the review
  defines it.
- **New 4c teachers:** the 5 seeds (of seeds 200–999) where the root cut beats every bottom bond by the
  largest margins (5–13%).
- **Star thresholds also use the departure rule a_q < 0.99.** On ht4 near the threshold the strict
  escape stops at a_q = 0.98, and a_q < 0.9 would need the ~1/ε stage. Wherever both rules fire they
  give identical outcomes.
- **Stopped my 5 ht4 star runs** that used a stop at a_q < 0.85: they were heading into the ~1/ε stage.
  I reran them with the departure stop.
- **Star random-init depth:** I report per-seed local exponents and classes instead of fitting a power
  law, as the review recommended.
- **Not computed:** predictions for the correlated inputs at ρ = 0.1 and 0.3. The theory scripts are
  not in the repo, and this is the theory owner's job.
- **Stored saddles:** I kept the saddle copies the committed runs used, rather than switching to a
  canonical gauge. A canonical gauge would have meant rerunning about 1900 runs.
- **Deleted my own `r2_chain*.sh` queue scripts** (they are in git history); `run_round2.sh` replaces
  them.

**Round 2 (unchanged):**
- d = 2 teachers; N = 8 overlaps c = 0.45, 0.7; a balanced misaligned8_45 teacher was added.
- Saddle-start protocol for every tree; random init on mps4 / binary4.
- Onset events f = 0.05 with ≥ 10× growth; per-seed fits in log space.
- Very long runs stopped where they could not change a conclusion (14 ht8 runs at nonorth8_c0.45 with
  ε ≤ 3e-4).
- BLAS threads capped at 1 (`env1.sh`).

## Files left on the machine
All under `/root/outputs/data_teacher/` (about 9.9 GB). None of it is needed to read the results: the
summary CSVs and figures are in git, and every file is regenerated by `experiments/data_teacher/README.md`.
- Round 2 and fix round 2: `r2_saddle/` (1950 saddle-start runs incl. the 15 misaligned8_20
  supplements), `r2_saddle_fine/`, `r2_random_dense/`, `r2_corr/`, `r2_star_saddle/` (468 runs),
  `r2_star_random/`, `r2_cross_dense/`, `r2_cross_b_dense/` (new, 120 runs).
- Round 1 and its extensions: `e1_order/` (2.7 GB), `e1_scaling_*`, `e2_misaligned/`,
  `e3_correlated/`, `e4a_nonorth/`, `e4a_sigma_dense/`, `e2_sigma_dense/`, `e4b_ttn_dense/`,
  `e4c_cross_dense/`.
- Superseded: `e4b_ttn/` and `e4c_cross/` (pilot versions).
- Rank-1 predictions: `rank1_predictions_*.pt`.

## Caveats and next steps
- **Asymptotic fits use few points:** 3 smallest σ (random init) or 5 smallest ε. Their errors are
  between seeds; the fit residuals are small but not propagated.
- **Not yet asymptotic:**
  - 4a c = 0.05 (+4.4% at σ ≥ 3e-4);
  - the correlated ρ = 0.2 increments (+2%, still falling);
  - misaligned8_20 on binary8 / ht8 (p = 1.07 / 1.05 over 1.3 decades).
- **Scope:**
  - The tree contrast is shown for the escape from the rank-one saddle (N = 4, 8; d = 2).
  - Random init: N = 4 only.
  - Correlated inputs: N = 4 MPS / binary only (no HT).
  - 4c: P11 tested at σ = 0.3–0.5 only.
- **No theory yet:**
  - the measured P1 exponents sit 0.07–0.15 above the heuristic min(α_r, α_a) for every 4b teacher;
  - the star random-init decay of a_q is steep but not described by any theory.
- **The f-rule can miss an escape** whose end point has s₂ < f·s₂* (misaligned8_20). Other analyses
  that use the f-rule should check for intermediate levels.
- **Next:**
  - predictions for the correlated inputs at ρ = 0.1, 0.3 (theory owner);
  - correlated inputs at N = 6;
  - escapes from higher-rank saddles on binary / HT (theory_saddles §5.6);
  - a theory for the P1 offset.

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
- 09:55 Fix round 2 started: main merged, review of round 2 read (7 must-fix, 4 should-fix items).
- 09:58 New 4c teachers (root cut strongest) run: P11 117/120. Saddles stored; determinism check (t_esc to 2e-10).
- 10:01 Sign-based saddle labels and per-edge rates; misaligned8_20 small-eps runs launched; asymptotic fits added.
- 10:04 Asymptotic slopes: saddle start <= 0.4% (13 teachers), random init <= 1.3%; P1 exponents for all 4b teachers.
- 10:05 Stars: classes, departure rule, two-stage escape near threshold (0.1-0.2%); ht4 eps = 1e-6 bracket completed.
- 10:06 misaligned8_20 re-timed: 1979.6 vs 1974.8. Float-floor scan extended (no new flags).
- 10:11 Notes (R2.1-R2.4, summary table), README, report updated. Done.
