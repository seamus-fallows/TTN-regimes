# Verification of the data-teacher fix round 2 (merge fe95253), 2026-10-10

Independent verifier. Inputs: notes/reviews/data_teacher_round2.md (and round 1), the fix list sent to the
agent (7 MUST FIX, 4 SHOULD FIX), notes/data_teacher.md (summary table, R2.0-R2.4), agent-runs/data-teacher/REPORT.md,
experiments/data_teacher/ (code, README, results/*.csv).

**How checked.**
- Every fix-round number below was recomputed from the committed CSVs with my own code: value-based fit windows,
  per-seed realized edges, P11 flags and de-duplication, star classes, correlated-input regressions.
- Independent of the agent's code: teacher Schmidt values of all 12 step-4(c) teachers by my own SVD of the dense
  teacher; raw Hessians (my own loss incl. the covariance metric, autograd, my own threshold) at 12 of the stored
  exact saddles (mps4/8, binary4/8, ht4/8).
- One Windows rerun with the agent's runner (8 randbin_302 saddle-start runs, spawn workers, stored saddles; 2 min,
  4 processes), outputs in my scratchpad.
- Raw data used once, marked [raw]: the 5 local misaligned8_20 mps8 files at eps = 1e-3, to read the intermediate level.
- Total compute about 5 minutes. No repo file other than this one was written; nothing was committed.

**Overall.**
- All 7 MUST-FIX items are fixed except the scope narrowing (M6), which is applied in R2.1, R2.4 and the REPORT
  caveats but not in several summary and body sentences. M6 is partly fixed.
- All 4 SHOULD-FIX items are fixed.
- Every fix-round number I recomputed matches the CSVs to the quoted digits.
- New small problems:
  - a false "same ordering" claim for the P1 exponents;
  - one HT asymptotic window that silently extends to eps = 2e-2;
  - a few slightly wrong figures or attributions;
  - round-1 body text that is superseded but not marked as such.
- No result changes direction. Section 2 lists the errata.

## 1. Verdict per item

| # | item (fix list) | verdict | key evidence (my recomputation; details below) |
|---|---|---|---|
| M1 | 4c / P11 counts and the discriminating flag | **fixed** | flag = root s_2 above both bottom bonds of some half (the review's definition), confirmed by my own SVD; de-duplication 264 -> 256 runs; 46/48 (209, 212), 117/120 (5 new teachers), 163/168 (7 teachers), 249/256 (12 teachers) |
| M2 | asymptotic per-seed slope fits | **fixed** (error bars statistical only) | windows are rank-based and uniform; every quoted slope is reproduced to all digits by value-based refits; one HT window deviates (E1); errors are s.e.m. over seeds, window systematics 1-3% |
| M3 | misaligned8_20 not degenerate; re-time; tallies | **fixed** | 1979.56 +- 1.78 vs 1974.82 (+0.24%); per decade 4531 / 4564 / 4541 vs 4547 (seed 0); log wins 60/60 + 5/5; binary/HT 24 cases; [raw] level 3.74-3.87e-4, not 3.6e-4 (E2) |
| M4 | star random-init "depth sigma^5.7" | **fixed** | classes by peak a_q reproduced (binary4 7/3; mps4 1/4/5); power law and "sigma^-1.4" withdrawn everywhere; one overstatement (E7) |
| M5 | correlated-input statistics | **fixed** | per-seed regression reproduced: 2.1442 +- 0.0069 / 2.1329 +- 0.0025 (rho 0.4), 14.969 +- 0.296 / 14.672 +- 0.064 (rho 0.2); N14 replication stated |
| M6 | scope: escape from the rank-one saddle only | **partly fixed** | narrowed in the one-line answer, R2.1, R2.4, REPORT, README; still unscoped in summary rows 4a/4b, the 4(b) closing sentence, and "binary and HT ... for the same teachers" (HT never run with correlated inputs) (E12-E16) |
| M7 | saddle labels, realized edges, row-2 number, f-sensitivity | **fixed** | sign-based labels confirmed by independent Hessians (5 / 17 / 17 negatives); edge rates identical; 53.04 +- 0.06 asymptotic (53.90 global); f-sensitivity <= 0.38% asymptotic, up to 1.83% global |
| S1 | 4b route mixing; P1 for all teachers | **fixed; new problem** | single-edge slopes and P1 exponents reproduced; "same ordering" is false (102 and 105 swap) (E4) |
| S2 | star rotation at w = 1.48, 1.42/1.44; ht4 bracket | **fixed** | one-stage slopes within 0.21%; near-threshold mps4 = 1/abs(lambda_q) + 1/abs(lambda_b) within 0.12-0.17%; ht4 eps = 1e-6 bracket 1.40-1.42 with 5 new runs |
| S3 | escape degree, float-floor coverage, replication tags | **fixed** (wording) | e1_kexp.csv reproduced ("9-29 s.e." is 7-29) (E8); star/correlated runs scanned, 0 flags; C11 and DLN tags present |
| S4 | randbin_302 reproducibility, spawn bug, threads, README | **fixed** | Windows spawn rerun with the stored saddles reproduces committed t_esc to <= 4e-9 (mps4) and <= 4e-5 (binary4), where the review saw offsets up to 50% |
| opt | optional runs | done | 15 misaligned8_20 runs (eps 1e-6...1e-4); 5 ht4 star runs (w = 1.42, eps = 1e-6); 120 runs on 5 new step-4(c) teachers |

### Details

**M1 (P11).**
- `analyze_cross_v2.py` now flags a teacher as discriminating if the root s_2 exceeds both bottom bonds of some half.
- My own SVD of the 12 dense teachers gives the same flags:
  - 209 and 212: root above every bottom bond by only 0.34% / 0.53% (above the weaker half's maximum by 1.7% / 0.7%);
  - new teachers 491, 548, 643, 694, 846: root / largest bottom = 1.055, 1.132, 1.096, 1.068, 1.062 ("5-13%" holds);
  - 201-204 and 218: not discriminating.
- `count_p11.py` keeps distinct (teacher, sigma, seed). The 8 duplicated runs (teacher 203, sigma = 0.4) have identical
  event orders in both sweeps. No run is dropped as undecided.
- New teachers: 40/40, 39/40, 38/40 at sigma = 0.3 / 0.4 / 0.5. All discriminating teachers: 56/56 at sigma = 0.3.
- All 5 discriminating exceptions grow the root bond after a bottom bond on one half only.
- Binary-8 edge labels used for this check: root 6; edges 2 and 5 carry the root cut; bottoms are {0, 1} and {3, 4}.

**M2 (asymptotic fits).**
- Windows, by rank (`lawfit.per_seed_local`):
  - random init (r2_random, e4a_sigma, e2_sigma, e4b_ttn): the 3 smallest sigma with a resolved run. In every seed of
    every case these are the grid's 3 smallest: mps4 3e-4, 1e-3, 3e-3 (one decade; the next grid point is 0.03);
    binary4 1e-4, 3e-4, 1e-3; 4b 1e-3, 3.16e-3, 1e-2.
  - saddle start: the 5 smallest eps with an event: MPS 1e-8...1e-4, binary/HT 1e-5...1e-3, with two exceptions.
    nonorth8_c0.45/ht8: 14 small-eps runs were stopped, so the per-seed windows reach 5e-3...2e-2.
    misaligned8_20: the t_int validity rule gives MPS 1e-6...2e-3 and binary/HT 1e-3...2e-2.
- The rule follows the review's recommendation and satisfies theory_teachers §9 (>= 3 sigma spaced by >= 2x). It was
  adopted after the review, not pre-registered, but it is applied uniformly. The 3-point variant ("loc3") is also in
  the CSV and agrees within 0.6%.
- My value-based refits reproduce to all digits: the 13 MPS saddle-start slopes, the 24 binary/HT exponents, the 7
  random-init slopes, 4a (c = 0.05, 0.2, 0.45), step 2 (theta = 5, 20) and 4b.
- Error bars:
  - they are s.e.m. over 4-10 seeds of per-seed slopes; per-seed fit errors are not propagated (the REPORT says so).
  - Two-point slopes inside the window (smallest pair / next pair):
    - converged at about 1-2%: random c = 0.2: 34.11 / 33.78 (pred 34.14); 4a c = 0.2: 34.18 / 34.40;
      theta = 5: 194.1 / 195.8 (bound 193.2); 4a c = 0.45: 5.64 / 5.74; theta = 20: 12.06 / 12.24;
    - spread by 2-3%, more than the quoted +-0.10 / +-0.07: misaligned_20 51.9 / 53.1 and randbin_302 12.18 / 11.81;
    - still drifting, as the note says: c = 0.05: 561 / 633 / 676.
  - So the "within 1-2%" agreement statements stand, but the +- values are statistical only.

**M3 (misaligned8_20).**
- Event t_int: max_e s_2 >= 1.8e-4, valid where s_2(0) <= 1.8e-5 (s_2(0) grows like eps^2 on mps8).
- Fit over eps = 1e-6...2e-3 (5 points per seed): 1979.56 +- 1.78 vs 1974.82 (per seed +0.07% to +0.53%).
- My own Hessian gives 1/gamma = 1974.82 on all 7 edges.
- f-rule times are about 2.6e5 at eps = 1e-3, against t_int = 6935. Review B's rerun gives 7135 to s_2 = 2e-4; consistent.
- [raw] At eps = 1e-3, for t in [2e4, 8e4], 5 seeds: max_e s_2 = 3.74-3.87e-4 on every bond (0.375-0.379e-3 per edge at
  t = 5e4), loss flat at 0.0023258-0.0023259.
- binary8 / ht8 re-timed: p = 1.071 / 1.048 over eps 1e-3...2e-2, and 1.028 / 1.019 over the 3 smallest eps.

**M4 (stars, random init).** All quoted numbers reproduce from `r2_star_random_classes.csv` and `r2_star_depth_exponents.csv`:
- peak a_q: blended 0.549-0.684; near q 0.775-0.996 (mps4) and 0.847-0.967 (binary4);
- time at the "q learned" level: <= 0.127 (one mps4 near-q seed: 1.67); at the "q forgotten" level: up to 538;
- whole-range depth exponents: 4.47-6.72 (blended), 2.36-6.25 (near q), 4.70-5.89 (binary4);
- local exponents: 3.76 -> 5.31 (mps4 seed 0), 0.03 -> 11.3 (seed 6);
- seeds 2 and 7 cross zero at sigma <= 0.0158;
- forgotten-plateau exponents: 1.22-1.59 / 2.41-3.67 / 1.31-2.32.
- No stale "sigma^5.7" or "sigma^-1.4" anywhere in notes/, report/ or CLAUDE.md, outside the R2.4 "withdrawn" row
  and the REPORT's historical log.

**M5 (correlated).**
- sigma grid 0.0016-0.05; neighbour ratios 1.937, 2.032, 1.984, 2, 2.
- Seed 2 is on the minor saddle at every rho on MPS-4 and is excluded.
- Smallest-pair increments: 2.1313 +- 0.0017 (rho 0.4) and 14.618 +- 0.053 (rho 0.2).
- Separation ratios reproduce as quoted (MPS 103 -> 530, 64 -> 94, 20 -> 45, 3.9 -> 19, 2.6 -> 11.4; binary 14-19 -> 139-185).
- Predictions for rho = 0.1 and 0.3 are still not computed; the agent declares this. The measured 113.9 and 4.616 are
  therefore untested numbers.

**M7 (labels, numbers).** Independent raw Hessians at the stored saddles. Negative eigenvalues use a threshold of -1e-6;
the next eigenvalue is about -1e-10 in every case.

| teacher, tree | negatives | lambda_min | per-edge 1/gamma |
|---|---|---|---|
| randbin_302 mps4 | 5 | -0.0763729 | e0 13.094, e1 402.5, e2 27.76 |
| nonorth_c0.45 mps4 | 5 | -0.1508901 | 6.627 on every edge |
| misaligned_20 mps4 | 5 | -0.0188879 | 52.944 on every edge |
| misaligned8_20 mps8 | 17 | -5.0638e-4 | |
| nonorth8_c0.45 mps8 | 17 | -4.7687e-3 | |
| randbin8_303 mps8 | 17 | -0.18972 | e4 5.474, e6 5.271 |
| randmps8_303 mps8 | 17 | -0.16424 | e1 6.181, e6 6.088 |

- No eigenvalue below -1e-8 on nonorth_c0.45 binary4 and ht4, misaligned_20 ht4, misaligned8_20 binary8, or
  nonorth8_c0.45 ht8.
- Realized-edge deviations +0.08% (randbin8_303) and +0.34% (randmps8_303) are seed means. Each teacher has one
  route-switching seed in the mean; for randmps8_303 that seed's own deviation is -2.0%.
- f-sensitivity of the asymptotic slopes between f = 0.03 and 0.1: at most 0.38% (randmps8_303). For global fits it
  reaches 1.83% (misaligned8_45, f = 0.1 vs 0.05).

**S1 (4b).**
- Single-edge asymptotic slopes: 2.4996 +- 0.0073, 3.0236 +- 0.0163, 0.8536 +- 0.0069, 5.1883 +- 0.0083,
  3.6788 +- 0.0185, against 2.488, 3.010, 0.868, 5.175, 3.657.
- Seeds 3-4 of teacher 102 and seed 0 of teacher 104 switch edge inside the window and are excluded.
- At sigma = 1e-3 every resolved seed of every teacher fires on e*. Log wins 26/26.
- Measured P1 exponents: 0.720 +- 0.012, 0.832 +- 0.019, 0.410 (1 seed), 0.927 +- 0.026, 0.802 +- 0.029, against
  0.643, 0.681, 0.343, 0.798, 0.722. The offsets are 0.07-0.15 as stated, and alpha_r for 104 is arithmetically
  consistent (0.798).
- New problem: the measured order 103 < 101 < 105 < 102 < 104 is not the predicted 103 < 101 < 102 < 105 < 104.

**S2 (stars, saddle start).**
- One stage (`r2_star_rotation_law.csv`): mps4 w = 1.48 / 1.6: -0.12% / -0.21%; ht4: +0.17% / -0.12%; binary4, all w: within 0.15%.
- Two stages (`r2_star_blended.csv`): mps4 w = 1.42: 130.03 +- 0.22 vs 129.88; w = 1.44: 29.39 +- 0.02 vs 29.34.
  The blended point has a_q = 0.984 / 0.933 (> 0.9) and is strict on mps4. On ht4 it has no negative eigenvalue
  (lambda_b ~ -1e-15); there p = 0.995 / 0.998.
- Threshold:
  - eps = 1e-6: 0/3 at w = 1.4138 and 3/3 at 1.4146 on mps4 and binary4;
  - ht4 (departure rule): step between 1.40 and 1.42 at eps = 1e-6, 1e-4 and 1e-3;
  - the two rules agree in every run with eps <= 1e-3; all 7 disagreements are at eps = 1e-2;
  - 43/45 sub-critical ht4 runs with eps <= 1e-3 reach t = 1.34e9 with no event; 2 learn r.
- ht4 w = 1.6 forgetting times: 3.36e3 / 3.34e4 / 3.34e6 at eps = 1e-3 / 1e-4 / 1e-6.

**S3.**
- `e1_kexp.csv`: binary4 at sigma = 0.02: 1.319-1.326 (7.4-9.3 s.e. below 4/3). MPS4 at sigma = 0.05: 1.459-1.466
  (16.5-28.6 s.e. below 1.5).
- Float floor (`float_floor_check.csv`): star random 545 rows and correlated 300 rows, 0 flags (minima 2.4e-12 and
  6.7e-10). Only the odeco rows of e2_sigma_dense (28) and e4a_sigma_dense (16) are flagged.
- The closest non-flagged case: binary4 TTN-teacher runs at sigma = 1e-4 reach max_e s_2 = 1.4e-13 (threshold 1e-14).

**S4.**
- `saddle_for` loads `results/r2_saddle_cores.pt` (39 saddles).
- `t_cap`, `stop_s2` and `stop_aq` now travel with each job; pools use spawn.
- `r2_star.py hessian` runs with 1 thread; `recompute_saddle_hessians.py` takes `--procs`.
- `run_round2.sh` (PROCS, DATA_TEACHER_LOGS) replaces `r2_chain*.sh`.
- The README documents the environment variables, the full run order (incl. star threshold / complete_ht) and spawn safety.
- My Windows run (`r2_saddle_runs.py`, 4 spawn workers, `DATA_TEACHER_OUT` / `DATA_TEACHER_RESULTS` in the scratchpad):
  Hessian rows are identical, and t_esc reproduces for mps4 (eps 0.02, 0.05; seeds 0, 1) to <= 4e-9 relative and for
  binary4 to <= 4e-5.
- On binary4 the recorded "edge" differs in 2/4 runs. Both binary4 edges carry the same cut, so this is a tie and is
  immaterial.

## 2. Errata

Exact corrections for anyone citing notes/data_teacher.md (N), agent-runs/data-teacher/REPORT.md (R) or
experiments/data_teacher/README.md (M). E1-E8 are new in or after the fix round; E9-E11 are round-1 or round-2 text
that is still unmarked; E12-E17 concern scope.

**Wrong or imprecise numbers**

- **E1** (N summary row "R2. Same generic teacher"; R2.1 verdict; R2.4 row "Binary / HT"; R results table).
  - Was: "asymptotic p = 0.9998-1.0036 (eps <= 1e-3), 24 cases".
  - Now: p = 0.9998-1.0009 in 23 cases over eps <= 1e-3. nonorth8_c0.45 on ht8 (small-eps runs stopped) is fitted over
    each seed's 5 smallest available eps, up to 2e-2, giving 1.0036 +- 0.0006 (1.0009 +- 0.0002 from the 2 seeds with
    3 points at eps <= 1e-3).
- **E2** (N R2.1 misaligned8_20 paragraph; R2.4 row; comment in configs/r2_saddle_m20.yaml).
  - Was: intermediate level "s_2 ~ 3.6e-4 (0.5% of the teacher value)". Now: s_2 = 3.7-3.9e-4 on every bond (about 0.6%).
  - Was: "A = 1979.6 +- 1.8 ... over eps = 1e-6 ... 5e-3". Now: fitted over eps = 1e-6 ... 2e-3 (the 5 smallest valid
    eps; 5e-3 is valid but outside the fit).
- **E3** (N R2.1 verdict). Was: global-fit exponents "p = 1.003-1.013". Now: p = 0.999-1.013. Also (N R2.0 row 4a):
  for c = 0 the power law has p = 2.004 (7 seeds), not "p = 2.000".
- **E4** (N R2.4 row on the P1 exponent; R table row 4b).
  - Delete "same ordering". Measured: 103 < 101 < 105 < 102 < 104. Predicted: 103 < 101 < 102 < 105 < 104.
  - 102 and 105 swap, within errors (0.832 +- 0.019 vs 0.802 +- 0.029).
- **E5** (N summary row 4c).
  - Was: "'sibling pair first' in 30-98% depending on the teacher".
  - Now: 30-48% per sigma for the r2_cross teachers, 90-98% for the new teachers; 8-100% per teacher.
- **E6** (N summary row 4c; R table row 4c).
  - Was: "117/120 for 5 teachers whose root cut carries the largest teacher Schmidt value of all cuts" / "(117/120 with
    the root cut strongest)".
  - Now: 117/120 for the 5 new teachers whose root s_2 exceeds every bottom bond by 5.5-13%.
  - Teachers 209 and 212 also have the strongest root cut, by only 0.3-0.5%. The class "root above all bottoms" is all
    7 teachers, 163/168 (p11_totals.csv).
- **E7** (N R2.2 item 5).
  - Was: "What happens is fixed by the init direction (seed), at every sigma". Now: the class (peak a_q) is fixed by the
    seed, but the outcome is not. Near-q mps4 seeds 3, 5 and 8 keep q (min a_q > 0.5) at sigma = 0.1, and seed 6 at
    sigma >= 0.025.
  - Was: "min a_q >= 0.993" (sub-critical). Now: >= 0.9918 (mps4, sigma = 0.1).
  - Was: "smallest abs(a_q) while forgotten, 2.4e-12 (envelope)". Now: 3.1e-12 (envelope). The 2.4e-12 is a dormant
    a_r' in a sub-critical w = 1.2 run.
- **E8** (N R2.0 and R2.4 escape-degree rows). Was: "9-29 s.e. below". Now: 7-9 s.e. below (binary4, sigma = 0.02) and
  17-29 s.e. below (MPS4, sigma = 0.05).

**Round-1 / round-2 text that is superseded but not marked** (cite R2.0 / R2.4 instead)

- **E9** (N 4(a).2 header bracket). It names 5.76 +- 0.04 (c = 0.45) and 36.5 +- 0.8 (c = 0.2) as the superseding
  values. These are pre-asymptotic global fits. Cite 5.686 +- 0.016 and 34.40 +- 0.20.
- **E10** (N body).
  - 1.4 item 3: "P(first) under (Sigma, s) equals ... in every setting" holds except binary T2 at sigma = 0.02 (0.800 vs 0.825).
  - 1.4 item 8: the -0.96 is not "the O(1) saturation time". It is a prefactor jump between the e1 and e4 sub-series;
    the slope within each is -1.000.
  - 1.4 item 10: "1.26-1.28 / 1.32-1.35 ... limited by the output grid" is replaced by the R2.0 row: the degree
    approaches 2 - 2/K but does not reach it (E8).
  - 2.1 item 4: "theta = 20: A = 15.0 +- 0.4; theta = 5: crossover, A = 264 +- 15" is replaced by 12.14 +- 0.03 and
    194.9 +- 1.2 (asymptotic).
  - 3.1 item 1: "every run has the plateau-escape-plateau structure" is replaced by the R2.0 row. On MPS-4 the
    separation ratio grows only like ln(1/sigma) and is below 3 at sigma = 0.05, rho = 0.4.
  - 4(b) items 1-3 are replaced by R2.0 and E4: slopes "3-25%", median distances "0.042 / 0.005 / 0.004" (a
    composition artefact), first-edge percentages at sigma >= 0.02.
  - 4(b) closing sentence: see E12.
- **E11** (N R2.0 row 4b; R2.1 random-init verdict).
  - "0.0002-0.0008 from the critical point at sigma = 1e-3" are per-teacher medians. Per run the range is
    0.0001-0.0025, still below the nearest truncation in every run.
  - "follows the log law in every seed" means all 23 seeds with >= 4 resolved sigma. randbin_303 has 1 such seed and
    randmps_301 has 3.

**Scope (M6 residuals)**

- **E12** (N section 4(b) closing sentence).
  - Was: "every plateau after the first a strict saddle with a log(1/sigma) duration".
  - Now: the first rank increase after the first escape (the escape from the rank-one saddle) is log-time on MPS-4;
    later plateaus were not timed against sigma.
- **E13** (N summary rows).
  - Row 4b: "MPS students: later plateaus logarithmic" should read: the escape from the rank-one saddle (the first rank
    increase) is logarithmic.
  - Row 4a: "Binary-4: sigma^-1 for every c" should read: for c = 0.2 and 0.45 (random init and saddle start); no
    binary-4 sigma law at other c.
- **E14** (N one-line answer; R "Answer to Q6/Q7").
  - "Binary and HT students keep ... for the same teachers": add that for cross-site correlations only binary-4 was
    run, never HT.
  - "escaped in log(1/sigma) time with slope 1/gamma of the edge that grows": add that for cross-site correlations the
    MPS-4 escape is two-stage, with ln2 [(1 - kappa)/lambda_2 + 1/lambda''] per halving (theory_covariance 7.4).
- **E15** (R "Plateaus"; M Results "Plateaus are always critical points ... (not HSVD truncations)").
  - Rank-one plateaus converge to the bond-rank-1 critical point.
  - They are distinguishable from every SVD-type truncation for 3 of 5 random MPS-4 teachers (102, 104, 105).
  - At higher ranks the two coincide for these teachers.
- **E16** (R "Binary students and cross-tree teachers ... only after"; R "Star teachers ... lose a learned component
  exactly when rho(S) > s_q"; M "with slopes as predicted by theory_teachers.md (... <= 1.3% from random init)").
  - P11 holds in 163/168 discriminating runs (N = 8, sigma 0.3-0.5), not in every run.
  - Stars: started at theta_q with noise eps <= 1e-3, the flow leaves theta_q iff rho(S) > s_q (to 3e-4 at eps = 1e-6).
    At eps = 1e-2 the apparent threshold moves to w ~ 1.43-1.44. Random-init runs never sit at theta_q.
  - The README's "<= 0.4% / <= 1.3%" covers misalignment, CP overlap and TTN teachers only. Cross-site correlations
    follow theory_covariance: +0.2% at rho = 0.4, +2% at rho = 0.2, still falling.
- **E17** (CLAUDE.md "Key findings", the lead's data-teacher line).
  - Was: "turns MPS later plateaus into strict log-time saddles".
  - Now: turns the escape of MPS students from the rank-one saddle (N = 4, 8) into a strict, log-time escape;
    binary/HT students keep sigma^-1 there.

## 3. Safe to cite

How to read the table:
- Status tags as in CLAUDE.md.
- "new" means not reported in the literature review (notes/literature.md) as the notes state it. "replication" means
  it repeats a check already in a theory note.
- All sources are in experiments/data_teacher/results/.
- The +- values are s.e.m. over seeds and statistical only (M2): cite agreements at the 1-2% level.

| # | claim | status | new / known | source |
|---|---|---|---|---|
| 1 | At the exact bond-rank-1 saddle of 13 generic teachers (non-orthogonal CP, misaligned covariance, random binary/MPS TTN teachers; N = 4, 8; d = 2), every MPS student has a strict saddle with 5 negative eigenvalues on mps4 and 17 on mps8 (the sum over edges of (d-1)(chi_e-1)). None of the 26 binary/HT students has an eigenvalue below -1e-8 | derived (theory_teachers R9, theory_saddles R13) + computed (independent Hessians here) | new | r2_saddle_hessian.csv, r2_saddle_b_hessian.csv, r2_saddle_m20_hessian.csv |
| 2 | Saddle start, MPS students: t_esc = A ln(1/eps) + B with A = 1/gamma_e of the edge that grows, within 0.4% (mean over 5 seeds) for all 13 teachers over eps = 1e-8 ... 1e-4 (misaligned8_20 re-timed at its intermediate level, eps 1e-6 ... 2e-3). The log law beats a power law in 65/65 seeds; f = 0.03-0.1 changes A by <= 0.4% | empirical, confirming derived R9/R13 | new | r2_saddle_fits.csv, r2_saddle_perseed_local.csv, r2_saddle_runs_all.csv |
| 3 | Saddle start, binary/HT students (N = 4, 8): t_esc ∝ eps^-1, p = 0.9998-1.0009 over eps = 1e-5 ... 1e-3 in 23 of 24 cases (nonorth8_c0.45/ht8: 1.0036 over a wider window); log law never preferred | empirical, confirming derived (route through an internal core, k* = 3) | new | r2_saddle_fits.csv |
| 4 | Event-rule failure mode: an onset rule s_2 >= f s_2(teacher) misses an escape that ends at an intermediate saddle with s_2 < f s_2(teacher). misaligned8_20: level 3.7-3.9e-4, 9x below the threshold; the rule then times the next escape (slope 24x) | empirical | new (methodological) | r2_saddle_runs_all.csv (t_int vs t_esc_0.05), r2_saddle_m20_runs.csv |
| 5 | Random fan-in init, mps4 (d = 2): second-escape slope over the 3 smallest sigma (3e-4 ... 3e-3; 4-5 seeds) = R6 with head start, within 1.3% for 5 teachers: 33.95 +- 0.18 (34.14), 5.674 +- 0.059 (5.682), 12.005 +- 0.071 (12.014), 1.038 +- 0.005 (1.051), 1.401 +- 0.013 (1.386). Without the head start randbin_303 would be 2.78. misaligned_20: 52.46 +- 0.10, below the bound 52.94 | empirical, confirming derived R6 | new | r2_random_v2_summary.csv, r2_random_v2_perseed_local.csv |
| 6 | Random init: binary4 second escape ∝ sigma^-1 (p = 1.001-1.005 over sigma 1e-4 ... 1e-3). Plateau lambda_min ∝ sigma on binary4; on mps4 it tends to the exact-saddle value, independent of sigma | empirical | new | r2_random_v2_summary.csv |
| 7 | Non-orthogonal CP teacher, MPS-4 (d = 4): c = 0.45: 5.686 +- 0.016 (R6 5.682); c = 0.2: 34.40 +- 0.20 (34.14); c = 0.05: 595 +- 11 vs 570, not yet asymptotic at sigma >= 3e-4; c = 0: sigma^-2.00 | empirical, confirming derived R6 | new | e4a_sigma_v2_summary.csv, e4a_sigma_v2_perseed_local.csv |
| 8 | Misaligned covariance, MPS-4: theta = 20°: 12.14 +- 0.03 (below 1/abs(lambda_min) = 12.31); theta = 5°: 194.9 +- 1.2, approaching the bound 193.2 from above; theta = 0: sigma^-2.00; log law wins 10/10 for theta > 0. No R6 prediction under the metric | empirical | new | e2_sigma_v2_summary.csv |
| 9 | Random MPS-4 teachers on an MPS-4 student: log law in 26/26 seeds; slopes of single-route seeds within 0.3-1.6% of R6 for that edge; first growing edge = e* = argmax s2(M_e) in every seed at sigma = 1e-3. The rank-1 plateau approaches the bond-rank-1 critical point as sigma^alpha, with alpha 0.07-0.15 above the P1 heuristic for every teacher (ordering not established). The plateau is separated from every SVD-type truncation for teachers 102, 104, 105 | empirical (P1 is a heuristic) | new | e4b_ttn_v2_summary.csv, e4b_ttn_v2_perseed_local.csv, e4b_p1_exponents.csv, e4b_ttn_plateaus.csv |
| 10 | Binary-8 student, MPS-8 teachers, sigma = 0.3-0.5: the root bipartition gets its second singular value only after a bottom bond on each half (P11) in 163/168 runs over the 7 teachers where a weight-order rule disagrees. That includes 117/120 for the 5 teachers whose root cut is strongest by 5.5-13%. Over all 12 teachers: 249/256. "Largest teacher Schmidt value first" fails | empirical, confirming derived P11 (theory_teachers 5.2) | new | p11_totals.csv, p11_counts.csv, r2_cross_b_v2_summary.csv, r2_cross_v2_summary.csv, e4c_cross_v2_summary.csv |
| 11 | Cross-site correlated inputs (N = 4, d = 3): the MPS-4 later escape is logarithmic (4/4 seeds, rho = 0.1-0.4). Increments per halving over the 3 smallest sigma: 2.133 +- 0.003 (pred 2.128) at rho = 0.4; 14.67 +- 0.06 (pred 14.39, still falling) at rho = 0.2. Binary-4 stays sigma^-1 (p = 1.005-1.011) | empirical; replication of theory_covariance N14 (derived two-stage law) | replication | r2_corr_fits.csv, r2_corr_increments_perseed.csv, r2_corr_runs.csv |
| 12 | Star teachers (q plus r, r' sharing a piggyback node): the Hessian at theta_q is -a^(K-2)(rho(S) - s_q) to 1e-15 on mps4, binary4, ht4 and binary8; strict iff rho(S) > s_q | proven (theory_saddles 4.1) + computed | replication (C11); binary4 known (deep linear network) | r2_star_hessian.csv |
| 13 | Star threshold from a saddle start at eps = 1e-6: bracketed to +-3e-4 in w (rho/s_q 0.9998 vs 1.0003) on mps4 and binary4; on ht4 between w = 1.40 and 1.42 | empirical, confirming the proven criterion | new (dynamics) | r2_star_threshold.csv, r2_star_saddle.csv |
| 14 | Escape from theta_q: slope 1/abs(lambda_q) within 0.21% where one stage is timed (w >= 1.48; binary4 at every w). Near the threshold on mps4 the event times theta_q -> blended rank-1 point -> onward, with slope 1/abs(lambda_q) + 1/abs(lambda_b) within 0.2% (w = 1.42, 1.44). On ht4 the blended point has no negative eigenvalue and is left in ∝ 1/eps time | derived (two-stage reading from computed Hessians) + empirical | new | r2_star_rotation_law.csv, r2_star_blended.csv, r2_star_saddle.csv |
| 15 | Stars from random init (w = 1.6): the flow never sits at theta_q (<= 0.13 time units; one seed 1.7). The init direction sets the class (q last / near q / blended). After its peak, a_q decays steeply but not as a power law. Sub-critical stars (w = 1.2) keep q (min a_q >= 0.992) | empirical | new | r2_star_random_classes.csv, r2_star_depth_exponents.csv, r2_star_depth.csv |
| 16 | Float64 floor: no non-odeco run behind these claims is flagged (smallest dormant signals: 1.4e-13 against the 1e-14 TTN threshold, 2.4e-12 in the star runs, 6.7e-10 in the correlated runs). Only the odeco baselines at sigma <= 3e-3 are flagged, and their exponents are unchanged | check | — | float_floor_check.csv |
| 17 | Escape degree from the origin: approaches 2 - 2/K as sigma falls but is not reached at the tested sigma (binary4 1.32-1.33 vs 1.333; MPS4 1.46-1.47 vs 1.5). Cite as a trend, not as agreement | empirical (prediction from theory_saddles R5) | known prediction | e1_kexp.csv |

Step-1 claims (aligned covariance: race by Lambda s, landscape by sqrt(Lambda) s, exact aligned clock, swap Hessian)
did not change in this round. Cite them from the "solid claims" list of notes/reviews/data_teacher_round1.md together
with E10.

## 4. Files

- Scripts: `verify_dt_r3/v_random.py`, `v_saddle.py`, `v_p11_teachers.py`, `v_hess.py`.
- Windows rerun: `verify_dt_r3/rerun/` (config, Hessian and run CSVs, raw npz).
- Both are in the session scratchpad
  (C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/).
