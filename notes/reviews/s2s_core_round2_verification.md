# s2s-core fix round 1: verification (round 2), 2026-10-10

Independent verifier. Scope: the fix round merged as 5647a0f (fix list = MUST FIX 1-8, SHOULD FIX 1-3
sent after `notes/reviews/s2s_core_round1.md`), i.e. `notes/s2s_core.md` (sections 2-4),
`agent-runs/s2s-core/REPORT.md`, `experiments/s2s_core/` (code, `results/*.csv`, figures). Raw data:
`data/s2s_core/` after the refresh (3417 files, identical byte count to the remote). No sweep was re-run;
about 30 CPU-minutes of new computation in total (max 8 processes). My scripts and outputs are in the
session scratchpad (`.../scratchpad/verify_s2s_r2/`: `sphere_T.py`, `strict_hessian.py`, `mid_hess.py`,
`decay_refit.py`, `gap_refit.py`, `drift0.py`, `cont*/`); nothing in the repo was changed except this file.

**Bottom line.** Every MUST FIX item is fixed or fixed with small residual errors; no headline claim
of the fix round fails. Independent recomputations agree with the committed numbers: the corrected
correlation-flow T matches to 1e-10 or better, the strict-saddle eigenvalue is exact, the trap decay
rate is sigma-independent, and the gap fits are rule-based and reproduce exactly. What remains are
wording and number slips (section 2): "100% first component" is 1847/1848; the stop rule never met
its direction criterion on the close teacher, so those jobs stopped at the amplitude cap (harmless for
T); "1.1-5.6x below the single-end estimate" cannot be reproduced; the "derived bound" for stalled runs
is exceeded in 4 of 16 escapes; the R10 magnitude agreement holds for cell medians only; and several
bookkeeping numbers are wrong.

## 0. What I recomputed

- **Correlation-flow blow-up time T, independently.** I used the spherical decomposition of note 1.1:
  u' = grad g(u) - K g(u) u, G' = g(u), t' = rho_0^-(K-2) exp(-(K-2) K G), integrated in tau until
  (K-2) K G > 45. This has no amplitude stop and no remaining-time extrapolation. The contraction is my
  own; only theta_hat (`s2s.make_init`) and the teacher vectors come from the repo. I ran it for all
  160 binary4 and mps4 first-escape jobs (both teachers, chi = 3, 6, fan-in and balanced).
- **Exact Hessian at the balanced partial fits W_S of binary4** (both teachers, chi = 3, 6, all learned
  sets). I also computed my own raw Hessian at 12 saved plateau midpoints (`mid_params`) of the
  close-teacher wrong-order runs.
- **Trap decay rate, refitted** from the raw `slice_norms` of all mps4 chi = 3 runs that do not
  complete. I used a different window from the agent's (t > 2 t_last; 1e-10 < slice < 1e-2 x its value
  at the window start), and two estimators: least squares, and the median of local rates.
- **Per-seed exponents, refitted** from `main_runs.csv` with my own implementation of the agent's
  run-set rule (rule A), plus two alternatives: B, the same without the rho > 3 cut; C, greedy order
  only.
- **Continuations.**
  - Re-ran 4 committed continuations with the committed code: 3 from the resolved set and 1 from the
    random set.
  - Re-ran the mps8 chi = 6 run that did not escape to 3x its estimate.
  - Rebuilt the k78 candidate pool and the seeded random pick.
  - Recomputed all 151 k78/large slice-trend rows from raw npz with my own code.
- **Tolerance check.** Reclassified the 9 `k78_tight` runs from raw data and compared their events with
  the loose runs.
- **Conservation drift from t = 0** (Q_e(theta_end) - Q_e(theta_0)) for the 737 runs that saved both
  states.
- **The counts and numbers quoted in notes 2.2b-2.7, 3, 4 and the REPORT**, checked against the CSVs. This
  includes the class tables, edge steps, escape drops, plateau levels, R10 ratios, KH ratios, d and chi
  tables, and the order table.

## 1. Verdict per item

| item (fix list) | verdict | evidence (my recomputation) |
|---|---|---|
| M1 strict saddle; classify runs before fitting | **fixed** (small residual: E6, E7) | **Prediction:** the exact Hessian at the balanced partial fit W_{1,3} (close teacher, chi 3 and 6) has lambda_min = -0.133186. This equals s_q^((K-2)/K)(s_q - s_r) to 6 digits; the separated sets also match (-1, -1.259921, -2). **Measurement:** my raw Hessians at the saved midpoints reproduce `lam_min_raw` to 6 digits. The 9 chi = 3 wrong-order plateaus (seed 8 at 7 sigmas, seed 5 at 2) give projected lambda_min -0.106 .. -0.146, with loss 0.3-3.8% from L_S and rho_next 0.76-1.25. Seed 8: slope 0.025, gap-2 exponent -0.08. **Fits:** the run-set rule reproduces exactly; binary4 close gap 1 / gap 2 = 1.004 +- 0.010 / 1.016 +- 0.011 (chi 3), 0.999 / 1.014 (chi 6). |
| M2 scope of "S2S on every tree"; mps8 gaps | **fixed** (residual: E8) | Reproduced from the CSVs: mps8 0/284 complete, 132 learn two; K = 7 at chi = R 14/180 and 12/150 complete; ht4 per-seed exponents 3.93-5.78; mps8 gap 1 5.95 +- 0.05 (7 seeds, chi 6) and 6.0 +- 0.3 (3 seeds, chi 3). |
| M3 trap decay fit above the float64 floor | **fixed for the rates**; magnitude statement **not verifiable** (E4, E5) | **Rates:** my window and estimators give a sigma-independent rate / sigma^2 per seed at both end cores: 0.259-0.265 (seed 3), 0.807-0.821 (5), 0.275-0.276 (6), 0.98-1.04 (7). The spread over sigma = 0.003-0.1 is <= 2.4% for seeds 3, 5, 6; for seed 7 it is <= 5.3% by least squares and <= 3% by local-rate median. **Floor:** dead slices sit at 1e-16 .. 1e-12 absolute, at least 3 decades below the 1e-9 cut. The 20-100x "switch" is gone. **Close teacher:** seed 1 varies by 1.36x and seed 7 by 1.23x for sigma <= 0.03 (agent: 1.33x, 1.25x). |
| M4 correlation-flow stop; r1 and T | **fixed** for T and r1 (residual: E1, E2, E3) | **T:** the independent sphere-flow T agrees with the committed T to <= 1.5e-11 (binary4) and <= 1.7e-10 (mps4) relative, on all 160 jobs; r1 agrees 160/160. The v1 values were off by up to 2.2e-3 (close) and 3.0e-5 (separated). All 5 r1 changes since v1 go to the largest component. **Stop rule:** the direction criterion was never met on the close teacher; all 40 of those jobs ended at the 1e30 cap (harmless for T, E3). **First component:** GF learns the correlation flow's r1 in 1847/1848 runs, not 100% (E1). |
| M5 stalled runs = ever-longer plateaus | **partly fixed**: wording is now hedged, the sampling is honest, the evidence is thin (E12, E13) | **Reproduction:** my reruns of 4 continuations reproduce the zoom-resolved escape times to <= 5.5e-6. The seeded random pick (`--seed 1`) reproduces exactly. Slice trends reproduce exactly (151 rows). The mps8 chi = 6 run reported "not escaped at 2x" does not escape by 3x either (my rerun). **Residuals:** 4 of the 16 measured escapes exceed the "derived bound" (1.08, 1.37, 1.81, 2.19x). Only 15/106 k78 stalled runs and 8/43 large mps8 stalled or unresolved runs were continued. |
| M6 trap fractions as lower bounds; K = 4 by teacher | **fixed** (residual: E15) | All counts reproduce from `main_classes.csv`. Separated teacher: seeds 3, 5, 6, 7 trap at 8/8 sigmas. Close teacher: seed 1 traps at 8/8, seed 7 at 6/8, seed 5 stalls at 8/8. **Tolerance check:** 9/9 runs keep their class and order; events agree to <= 3.6e-6; at the tight tolerance, trapped slices reach ~1e-15 sigma and stalled ones stay >= 1e-7 sigma. |
| M7 balanced init / chi = 2R counts | **fixed** | Reproduced: 0/120 vs 12/30 and 47/90; 69/90 balanced K = 7, 8 runs stall; 5 runs complete with fan-in but stall with balanced init; ht4 chi = 6 5/50 (seed 9 at all 5 sigmas); mps8 chi = 6 7/50 (5 seeds). |
| M8 numbers vs CSVs | **fixed**, with new slips (E10, E11, E16-E18) | **Reproduced:** edge steps 497/531 (332 untested); the escape-drop table, all 20 entries; KH ratios 3.1-4.8 / 7.9-13.7 / 95-210 / 184-2650 / 1460; mps4 plateau level 2.8e-3 (8.1e-3 balanced); mps4 close 8 stalled. **Slips:** R10 holds only as cell medians (E11); some plateau-level numbers are wrong (E10). |
| S1 tag replications of theory_escape | **fixed** (residual: E7) | E1, E2, E5, E7, E8 are tagged in notes 2.2b, 2.2c and 3 and in the REPORT; P3 is tagged new (heuristic). |
| S2 20 directions; sd ln T; drift | **fixed** | **20 directions:** binary4 largest-first in 380/380 runs = 20 init directions. **sd ln T** recomputed: 0.218 +- 0.010, 0.157 +- 0.009, 0.135 +- 0.005. **Drift:** 11 runs exceed 1e-6 per chunk (max 3.3e-6). My drift from t = 0 agrees (max 3.8e-6, large mps8), which is <= 8.6e-4 of the initial Q(0) scale for fan-in K = 7, 8. |
| S3 README / code | **mostly fixed** (residual: E13, E14, E19) | **analyze_main:** `--hess` has no default, and all 2270 plateaus have recomputed Hessians (no silent fallback). **Outputs:** `--out` is required in every writer. **aligned.py:** one file per integrator, `--workers`, `--raw`; the new `aligned.csv` matches round-1 reviewer A's independent values (1.97e-10, 3.25e-10, 2.26e-9, 2.59e-9, 2.68e-9). **README** lists every command. **Remaining:** the README spread command does not reproduce `continued_large_spread.csv`. |

**New problems found** (all minor; details in the errata):

- The `first_summary.csv` column `r1_agree_corrflow` tests only the smallest sigma (E1).
- The decay fit window starts inside the post-event transient, so the per-core levels are biased by
  about +-4% (E4).
- The continuation candidate pool admits trapped runs that ran to t_max. One is in the k78 pool (none
  was sampled) (E12).
- `dormant_decay.csv` has no sweep column, and main and k78 rows duplicate one another at K = 7, 8 (E19).
- The binary8 chi = 3 first-gap fit is the only K = 7 fit spanning about 1 decade, and it is not
  reported (E21).

## 2. Errata (apply when citing `notes/s2s_core.md`, the REPORT or the README)

**First escape**

- **E1. First component, "100% of runs".** Applies to notes 2.2b and 4.5, REPORT section 4 and the
  fix-round table.
  - Correct figure: GF's first component equals the correlation flow's r1 in **1847/1848** first-escape
    runs, over all sigmas.
  - The exception is binary4, chi = 3, close teacher, balanced init, seed 6 at sigma = 0.1. There GF
    learns the 0.7 component first: a finite-sigma wrong first escape, the type theory_escape E9
    describes.
  - Replace "gradient flow learns the largest component first at every sigma" with: "in 639/640 binary4
    first-escape runs and in all 380 main/balanced runs". What is withdrawn is E9 as the cause of the
    old r1 mismatches, not E9 itself.
  - `first_summary.csv` `r1_agree_corrflow` = 1.0 tests the smallest sigma only.
- **E2. Size of the old bias in T.** Applies to notes 2.2b and 4.5 and the REPORT ("T moved by <=
  1.7e-3").
  - Binary4 close: the old (amplitude 1e6) T was off by up to **2.2e-3 relative** (balanced init, chi 3,
    seed 6). The 1.7e-3 figure is the fan-in maximum (1.75e-3).
  - Binary4 separated: off by up to 3.0e-5 (balanced, chi 6), not 2.4e-5.
  - mps4 and K = 7, 8 changed by <= 2e-7.
- **E3. How the new correlation-flow stop actually ended.** Applies to the notes 2.2b stop description,
  the REPORT "Decisions" entry and the `s2s.correlation_flow_blowup` docstring.
  - The direction criterion (second/first amplitude <= 1e-6) was **never met for the 40 binary4
    close-teacher jobs**. All of them stopped at the amplitude cap 1e30, with the ratio still between
    2e-6 and 7e-3; 2 separated binary4 chi = 6 jobs also reached the cap.
  - T is converged regardless: the time remaining beyond amplitude A is ~A^-(K-2)/K ~ 1e-10, and the
    independent sphere-flow T agrees to <= 1.5e-11.
  - Describe this case as "stopped at amplitude 1e30 (direction converging slowly)", not as direction
    convergence.
- **E16. Residual bound.** Applies to the REPORT summary and notes 3.3. "residual <= 2e-3" should read
  **<= 2.2e-3** (binary4 close, balanced, chi 6: 2.15e-3).
- **E17. Offset-removed exponents.** Applies to notes 2.2b, which states 0.9996 without a teacher.
  - 0.9996 +- 0.0001 is the **separated** teacher.
  - The close teacher gives 0.990-0.996 per cell (0.993 +- 0.003 for fan-in chi 3), which is 2.6-3.8 SE
    below 1. This is consistent with the unconverged O(1) offset; the REPORT table already shows 0.993.

**Saddles and gap exponents**

- **E6. Strict-saddle range.** Applies to notes 2.5, 3.2 and 4.1 and REPORT sections 5 and 2.
  - "-0.118 .. -0.146 over sigma 0.005-0.1, median 0.98" is **seed 8 only**.
  - All 9 wrong-order plateaus at chi = 3, adding seed 5 at sigma = 0.07 and 0.1: projected
    **-0.106 .. -0.146** (raw -0.113 .. -0.149); median ratio 0.98 (seed 8) and 0.93 (seed 5).
  - The exact eigenvalue at W_{1,3} is -0.133186 (this verification).
  - Add: the wrong order, and with it the strict saddle, is **not visited at the smallest sigma**. Seed 8
    is greedy at sigma = 0.003; chi = 6 seed 4 is wrong-order only at sigma >= 0.03. "Flat in sigma"
    therefore covers sigma = 0.005-0.1.
- **E7. Novelty tags.**
  - binary4 is a depth-3 DLN (theory_saddles 4.2). Its S2S sequence, the sigma^-1 plateaus and the
    strict wrong-order saddle are **known DLN results**: [Achour2024] (non-top rank-r critical points
    are strict saddles) and [Jacot2021] Lemma 15, both in `notes/literature.md`.
  - Tag them "replication (known for DLNs; prediction = theory_saddles 4.1)", not as new. New for TTNs:
    the same measurements on mps4, binary8, ht4 and mps8.
- **E8. mps8 gap rows.** Applies to the notes 2.4 table and the REPORT table.
  - The sigma column "0.4-0.03" for mps8 chi = 3 is the t_1 range. The gap-1 fits use **sigma = 0.1 or
    0.15 to 0.4** (4-5 points, <= 0.6 decade).
  - The close-teacher mps8 chi = 3 gap 1 (6.19 +- 0.60, range 5.14-7.21, 3 seeds) is omitted.
- **E9. Binary4 close gap fits.** Applies to notes 2.4 ("within +-0.1 over 1.5 decades").
  - The rho > 3 cut leaves some seeds with only 4-5 sigmas (0.003-0.02, about 0.8 decade). Example:
    chi 3, seed 4, gap 1 = 1.064 from 4 sigmas.
  - The cut is rule-based and applied identically everywhere (my reimplementation reproduces every
    value). Without it, binary4 close gives:
    - chi 3: gap 1 = 1.00 +- 0.04 [0.76, 1.30], gap 2 = 0.92 +- 0.11;
    - chi 6: gap 1 = 1.04 +- 0.04 [0.90, 1.25], gap 2 = **1.12 +- 0.05 [0.99, 1.45]**.
  - Every other cell changes by <= 0.002 without the cut.
- **E11. R10 magnitude.** Applies to notes 2.5, 3.4 and 4.9 and the REPORT.
  - "0.87-1.15 at K = 3, 4" holds for the cell medians of the main fan-in cells. The balanced binary4
    chi 3 cell median is 1.19.
  - **Per-seed medians reach 20**: binary4 separated chi 3 seed 8, second plateau, where lambda_min is
    about 20x the route value at every sigma. Balanced mps4 chi 3 reaches 7.9.
  - At K = 7, 8 the cell medians are 1.36-3.15 (main, large) and up to 3.9 (k78); per seed up to 20.
  - Cite R10 as a cell-median statement only. The slopes are unaffected.
- **E21 (addition).** binary8 at chi = 3 (main): gap 1 = 4.93 +- 0.05 (separated) and 4.92 +- 0.04
  (close), 10 seeds each, per seed 4.65-5.13, mostly over sigma = 0.03-0.4 (up to 1.1 decades). This is
  the only K = 7 gap fit spanning about 1 decade; it is in `main_exponents.csv` but not in the notes.

**Plateaus and dormant components**

- **E10. Plateau levels.** Applies to notes 2.3 and the REPORT section 2.
  - "binary4 2.0e-2 (both teachers, sigma = 0.1)" should read: separated **8.5e-3** (sigma = 0.1);
    close **2.0e-2 at sigma = 0.003** (chi 6, seed 4, rho_2 = 5.1).
  - "K = 7 <= 2.1e-5" holds at chi = 6 (large). At chi = 3 (main) the level error reaches **1.5e-4**
    (ht4 close, sigma = 0.4).
  - "falls to 6e-7 at sigma = 0.003" holds for the separated teacher and mps4, not for the close
    teacher.
- **E4. Decay-rate levels.** Applies to notes 2.7c and REPORT section 6.
  - The fit starts at 1.5x the last event, inside the post-event transient. Example: seed 3, sigma 0.01,
    where core 0 still grows at first and core 3 decays at 0.7 sigma^2. This biases the per-core values
    by about +-4% (0.251 vs 0.270 for seed 3).
  - The asymptotic local rate is the **same at both end cores**: 0.259 (seed 3), 0.81 (5), 0.276 (6),
    1.00-1.04 (7).
  - "constant to 1-3%" should read "<= 2.4% (seeds 3, 5, 6), <= 5.3% (seed 7; <= 3% by local-rate median), sigma = 0.003-0.1".
- **E5. Single-end estimate.** "measured rates are 0.18-0.87 of the single-end estimate, i.e. 1.1-5.6x
  below it" appears in notes 2.7c, 3.5 and 4.4 and twice in the REPORT.
  - **Not reproducible:** no script, CSV or log computes Q_nu nu or the estimate. It rests on 2 re-run
    seeds, and the choice of s_q (2 or 3) alone spans a factor 1.5.
  - Do not cite the magnitude. Cite only the sign rule and the sigma^2 scaling.
- **E18. Escape-drop percentages.** Applies to notes 2.7a and 3.5 and the README.
  - ">100x tail 3-8% at K = 7, 8, chi = 3 (later)" should read **2.0-8.3%** (ht4 later escapes 2.0%).
  - "0.6-2% at chi = 6" should read 0-2%.
  - The README "3-11%" should read 2-8%, or 11% for the mps8 separated later escapes only.

**Traps and stalled runs**

- **E15. Trap fraction at K = 4.** Applies to the REPORT summary ("40% (separated) and 18% (close) at
  K = 4, chi = 3, fixed per init direction").
  - For the close teacher the fraction is not fixed per direction: seed 7 traps at 6 of 8 sigmas, and
    seed 5 stalls at all 8.
- **E12. Stalled runs and the "derived bound".** Applies to notes 2.7 and 3.3 and the REPORT.
  - The balanced-route estimate is exceeded in **4 of the 16 measured escapes** (1.08, 1.37, 1.81 and
    2.19x). When quoting data, call it an estimate, not a bound.
  - The "74 k78 stalled runs with estimate <= 100x" are **73 stalled + 1 trapped**. The trapped one,
    mps8 fan-in sigma = 0.3 seed 5, ran to t_max and has a slice of 1.4e-10 sigma; it was not drawn.
  - Overall, 23 distinct stalled or unresolved runs were continued and 16 escaped: 14/15 k78 and 2/8
    large.
  - 91/106 k78 stalled runs (estimate / elapsed: cell medians 17-64x) and 35/43 large mps8
    stalled/unresolved runs (median about 390x) are untested.
  - "Stalls are ever-longer plateaus" is therefore a derived prediction checked on 16 escapes. For the
    untested majority it is a conjecture, supported only by the slice trends: pinned in binary8 and in
    balanced mps8; still decaying in 50% of balanced ht4 runs and 19-32% of fan-in mps8 runs.
- **E13. mps8 chi = 6 spread sample.** Applies to notes 2.7, the REPORT and the README.
  - The run "not escaped at 2x" (s0.2 seed 9) **does not escape by 3x either** (my rerun with
    `--factor 3`).
  - The factor used for the committed CSV is not recorded.
  - The README command (`--spread --n 8 --max_ratio 1000 --factor 3`) picks a different set: the last
    3 of the 8 runs differ. So `continued_large_spread.csv` is a round-1 artefact the README does not
    regenerate.
  - "4 hit the wall limit before reaching the estimate" cannot be checked: the CSV has no reached-time
    column.

**Bookkeeping**

- **E14. Run and job counts.**
  - REPORT "T recomputed for all 238 first-escape jobs" and "first/ (238 jobs)": there are **231**.
  - "Sweep main (1600 runs)": `main/` has **1126** runs (2252 files).
  - The K = 7, 8 first-escape sweep is described as "separated teacher, fan-in only". It also contains
    21 binary8 jobs with the close teacher or balanced init, which are in `first_summary.csv`.
- **E19. `dormant_decay.csv` sweep column.** The table has no sweep column. 13 K = 7, 8 keys appear in
  both main and k78; one pair differs 6x (mps8 sigma 0.15 seed 8). No claim depends on it.
- **E20. ht4 decay example.** The example "ht4 seed 0: 0.0057-0.0060 over sigma = 0.15-0.3" omits
  sigma = 0.4 (main), where the value is 0.0017. This strengthens the "not sigma^2-clean at K = 7, 8"
  statement.

## 3. Safe to cite (final claims after the errata)

**Status tags:**
- **proven, derived, empirical, conjecture** follow CLAUDE.md.
- **known / replication:** the result exists in the literature or in an earlier project theory note,
  and this agent measured it again.
- **new:** not found in `notes/literature.md` for TTNs.

All sources are in `experiments/s2s_core/results/`.

**Aligned init and first escape**

| # | claim | status | known / new | source |
|---|---|---|---|---|
| 1 | Aligned balanced init on odeco teachers follows the exact balanced Saxe modes on binary4, mps4, binary8, ht4 and mps8. Max deviation of the overlaps: 1.97e-10, 3.25e-10, 2.26e-9, 2.59e-9, 2.68e-9 (DP5, rtol 1e-10). | derived + empirical | known (Saxe 2014, DLN); explicit for TTNs | `aligned.csv` |
| 2 | **First-escape law per seed, no free parameter:** t_1 = sigma^-(K-2) T(theta_hat) + c_K. Relative residual at the smallest sigma: <= 8.5e-6 (binary4 separated), <= 2.2e-3 (binary4 close), <= 6e-5 (mps4), <= 4e-8 (binary8, ht4, mps8 at sigma = 0.01). Exponents with c_K removed: 0.9996 (binary4 separated), 0.990-0.996 (binary4 close), 2.0000 (mps4), 4.998-4.999 (K = 7), 5.9995 (mps8). T is confirmed by an independent method to <= 2e-10 (binary4, mps4). | derived (prediction) + empirical | replication of theory_escape E1; the exponent is known (Jacot 2021; Kumar & Haupt 2025) | `first_summary.csv`, `first_fits.csv`, `first_runs.csv` |
| 3 | **Which component goes first:** GF's first component equals the correlation flow's r1 in 1847/1848 runs over all sigmas, and in 100% at the smallest sigma. On tensor-like trees the first component is the same at every sigma for each seed (10/10 seeds in every cell). | empirical | replication (theory_escape E1, E8) | `first_runs.csv`, `main_order.csv` |
| 4 | **Kumar-Haupt time is a lower bound on T.** Median T / t_KH: 3.1-4.8 (binary4), 7.9-13.7 (mps4), 95-210 (ht4), 184-2650 (binary8), 1460 (mps8). | proven (bound) + empirical (ratios) | known (Kumar & Haupt 2025; theory_escape E2) | `first_summary.csv` |
| 5 | **No d-dependence at full bond (chi = d^2).** binary4 median t_hat_1 = 1.544, 1.569, 1.563 for d = 2, 3, 4 (about pi/2), while t_KH falls 1.9x. sd ln t_hat_1 = 0.218 +- 0.010 (d = 2; predicted 0.250 +- 0.02), 0.157, 0.135. mps4: 1.39, 1.41, 1.50. | empirical | replication of theory_escape E5 | `lo_d.csv` |
| 6 | **chi dependence of the median prefactor.** Slope -0.335 +- 0.028 (binary4) and -0.264 +- 0.037 (mps4) over chi = 4-16; -0.76, -0.69, -0.72 (binary8, ht4, mps8) over chi = 2-16 (2-12 for binary8). t_KH slopes are -0.47 to -2.5. | empirical | replication of theory_escape E7 | `lo_chi.csv`, `lo_chi_slopes.csv` |
| 7 | **RMS-overlap heuristic P3** = d^(N(K-2)/(2K)) / ((K-2) s_1) tracks the median t_hat_1 within 0.87-2.13x over K = 3-15 (17 trees). | derived heuristic + empirical | new (project) | `lo_K_table.csv`, `lo_K.csv` |
| 18 | Gauge-balancing the same init changes T per seed by 0.88-1.92x (median 0.99-1.18 per cell) and keeps r1 in 88/94 seeds. | empirical | new (minor) | `first_runs.csv` |

**S2S sequence, gap exponents and saddle type**

| # | claim | status | known / new | source |
|---|---|---|---|---|
| 8 | **Complete S2S sequences from fan-in init** (odeco, d = 4): binary4 320/320; mps4 160/160 at chi 6 and 106/160 at chi 3; binary8 50/50 and ht4 45/50 at chi = 6 (sigma 0.07-0.3). Plateau loss = L_S at bracketed plateaus: <= 8.5e-3 (binary4 separated), 2.8e-3 (mps4), <= 2.1e-5 (K = 7, chi 6). The top singular values on every bond equal the learned weights. One rank step per event on every edge in 497/531 runs with event spectra. mps8: 0/284 runs learn the third component (132 learn two). At chi = R, K = 7 rarely completes: 14/180 (binary8), 12/150 (ht4). | empirical | binary4 known (depth-3 DLN); mps4, binary8, ht4 new | `main_runs.csv`, `main_classes.csv`, `main_plateaus.csv` |
| 9 | **Gap exponents per seed**, on runs that follow a resolved S2S sequence (rho > 3, modal order); predicted K - 2. binary4: cell means 0.999-1.016 (per seed 0.95-1.09). mps4: 1.990-2.004 (per seed 1.97-2.01). binary8 at chi 6: 5.020 +- 0.009 / 4.969 +- 0.022 (per seed 4.88-5.09, 0.6 decade). binary8 at chi 3, gap 1: 4.93 +- 0.05 (up to 1.1 decades). ht4 at chi 6: 5.10 +- 0.08 / 4.75 +- 0.14 (per seed 3.93-5.78; not a clean confirmation). mps8 gap 1: 5.95 +- 0.05 (chi 6, 7 seeds, 0.6 decade). | empirical (prediction derived) | binary4 known (DLN); others new | `main_exponents.csv`, `main_gap_fits.csv` |
| 10 | **Higher-order saddles.** Cell means of the per-seed slope of log(-lambda_min) vs log sigma: 1.02-1.18 (binary4), 1.99-2.02 (mps4), 5.07-5.10 (binary8, chi 6), 4.88-5.13 (ht4, chi 6), 5.99 (mps8, chi 6, plateau 1); predicted K - 2. The magnitude agrees with R10 only as a cell median (E11). | empirical (prediction derived, theory_saddles R10) | binary4 known (DLN); others new | `main_exponents.csv`, `hess_main.csv`, `hess_other.csv` |
| 11 | **Strict wrong-order saddle on binary4** (close teacher). The exact lambda_min at W_{1,3} is -0.133186 = s_q^((K-2)/K)(s_q - s_r). At 9 plateau midpoints it measures -0.106 .. -0.146 (projected), flat over sigma = 0.005-0.1 (slope 0.03). Its gap does not grow (exponent -0.08). The saddle is not visited at sigma = 0.003. | proven at W_S (theory_saddles 4.1) + empirical | known for DLNs (Achour 2024; Jacot 2021 Lemma 15); replication | `main_strict_saddles.csv`, `main_plateaus.csv` |
| 19 | **Learning order on binary4:** largest component first in all 380 main/balanced runs (20 init directions) and in 639/640 first-escape runs. Full order greedy in 160/160 runs (separated teacher) and 89-95% (close teacher). | empirical | known for DLNs (Jacot 2021 Lemma 15) | `main_order.csv`, `first_runs.csv` |

**Dormant components, traps and stalled runs**

| # | claim | status | known / new | source |
|---|---|---|---|---|
| 12 | **Conservation-law trap at K = 4** (mps4, chi 3, separated teacher). The same 4/10 seeds trap at all 8 sigmas = 0.003-0.1 (32/80). The end-slice decay rate / sigma^2 is constant per seed (<= 2.4%; seed 7 <= 5.3%). Levels are seed-dependent (0.26-1.04) and equal at both route ends. Close teacher: 14/80 trapped + 8/80 stalled; seed 7 traps at 6/8 sigmas. | empirical (mechanism derived, theory_saddles R12) | new | `main_classes.csv`, `main_runs.csv`, `dormant_decay.csv` (per-core levels carry a +-4% window bias) |
| 13 | **No trap at K = 3** (0/380 runs). | empirical | new | `main_classes.csv` |
| 14 | **Gauge-balanced init.** 0/120 runs trap in the cells where fan-in traps (fan-in: 12/30 and 47/90). But 69/90 balanced K = 7, 8 runs stall. 17 of the 47 fan-in-trapped K = 7, 8 runs complete when balanced; 5 fan-in-complete runs stall. | empirical | new (supports theory_saddles R12) | `main_runs.csv`, `main_classes.csv` |
| 15 | **Trap fractions at K = 7, 8 are lower bounds.** chi = 3 (k78, sigma 0.15-0.3, fan-in): at least 9/30 (binary8), 24/30 (ht4), 14/30 (mps8). chi = 6: 0/50 (binary8), 5/50 (ht4, seed 9 at every sigma), 7/50 (mps8). Classes are stable at 100x tighter atol (9/9). | empirical | new | `main_classes.csv`, `tolcheck_k78.csv`, `slice_trends.csv` |
| 16 | **Escapes shrink dormant components only mildly.** Median factor: x2.1-3.0 (growth) at K = 3; x0.64-0.74 at K = 4; x0.12-0.75 at K = 7, 8. Drops by more than 100x: <= 1.3% (K = 4), 0-2% (K = 7, 8, chi 6), 2-8% (K = 7, 8, chi 3; 11% for mps8 separated later escapes). | empirical | new | `dormant_escape_drops.csv` |
| 17 | **Stalled runs.** Of 23 continued stalled/unresolved runs, 16 escape at 0.24-2.19x the balanced-route estimate: all 8 shortest-estimate k78 runs (0.50-0.87, zoom-resolved), 6 of 7 new runs of a seeded random sample, and 2 of 8 mps8 chi 6 runs. 2 do not escape by 3x, 4 hit the wall limit, 1 failed. The ever-longer-plateau reading of the untested majority is a **conjecture**. | empirical (prediction derived) | new | `continued_k78_resolved.csv`, `continued_k78_random.csv`, `continued_large_spread.csv`, `slice_trends.csv` |

**Not safe to cite as written:**
- "1.1-5.6x below the single-end estimate" (E5).
- "first component 100% / largest first at every sigma" (E1).
- Per-seed R10 magnitudes (E11).
- The balanced-route estimate as an upper bound (E12).
- K = 7, 8 decay rates and their sigma dependence (E19, E20).
- Any K = 7, 8 gap exponent as meeting the project's 1-decade rule, except binary8 chi 3 gap 1 (E21).
- "T moved by <= 1.7e-3" (E2).
- Plateau-level statements in the form given in E10.
