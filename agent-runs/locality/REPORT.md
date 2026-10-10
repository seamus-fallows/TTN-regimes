# locality: tree locality of saddle-to-saddle escapes (Q3, H2/H3)

Status: working (round 3: fixes after notes/reviews/locality_round2_verification.md) - Branch: agent/locality - Updated: 2026-10-10 00:12 UTC

## Summary
Asked: test H2 (the escape after a learned component needs the k cores of the Steiner subtree of the sites
where the next component differs: no plateau for k = 1, log(1/σ) for k = 2, σ^-(k-2) for k >= 3) and H3
(strict saddle iff k = 2) across MPS / binary / HT / custom trees, and show one teacher learned through
different saddle sequences on different trees. Scope of all claims: orthogonal pattern teachers, whitened
inputs, gradient flow, d = χ = 2 unless stated. **H2 is confirmed for k = 1-5 on all 9 trees.** Rows that
deviate are carried by seeds that stall at some σ (saddle-start k = 4, ht8) or come back to k - 2 at smaller σ
(MPS k = 5), except one: mps8 {1,5} from the saddle start stays at 3.67 ± 0.20 (4 seeds, 0.4 decades, ~40% of runs
stalled). For k = 6-7 the exponent equals k - 2 on binary8 and ht4 **with gauge-balanced init only**, over one
decade (fan-in k = 6 rows: 3.68, 4.23; MPS: 4.51); k = 8 is not measured. The k = 2 log slope carries a head start
((1 - s2/s1)/λ2 from small init, 1/λ2 from the saddle), and λ2 = s2 s1^((K-2)/K) holds including its
K-dependence. **H3 holds for greedy-order saddles**: λ_min = -λ2 exactly for k = 2; for k >= 3 λ_min ∝ σ^x -> 0
with x ≈ k - 2 (within 0.2 for k = 3-5). **With several learned components the literal H2 fails and the route
rule at rank-constrained optima holds where tested** (TD and TC separate it from the alternatives). Dead routes
are frequent for k >= 4 under fan-in init (never for k <= 3); balanced init reduces them 10x; χ = 4 removes them
on routes with an uncapped end but not on routes with both ends capped (106 / 1,005 stalls vs 169 / 1,008 at χ = 2).

## Results
Dynamics for every number: gradient flow, scipy LSODA with exact-Hessian Jacobian (rtol 1e-9, atol 1e-14), event
times by root-finding, library v0 model, float64, teacher seed 1000, init seeds 0-7. Tables are restricted to the
runs of the committed configs; rotation runs are excluded from fits; per-seed fits, mean ± SE (median).
Regenerate everything with `experiments/locality/finalize.sh <dir>`; details in `notes/locality.md` sections 7-10.

### 1. Exponent table (two components, weights 1 / 0.5; `results/twocomp_table.md`, `extensions_table.md`)
| k | trees | predicted | measured |
|---|---|---|---|
| 1 | binary4 ×2, binary8, star3 | no plateau | t2 - t1 <= 8e-4, ρ <= 4e-4 (small init) |
| 2 | mps4/6/8, midroot6 | log; slope 1.000 (small init), 2.000 (saddle start) | 0.980-1.009, 2.002-2.071; log law beats power law in 14/14 rows (fair comparison) |
| 3 | all 9 trees (42 rows) | 1 | 0.98-1.04 |
| 4 | mps4/6/8, star3, midroot6 | 2 | 1.95-2.05 (small init), 1.94-2.02 (balanced), 1.99-2.03 (χ = 4); saddle start 1.48-2.39 (1.81-2.19 on never-stalling seeds) |
| 5 | binary8 ×3, ht4 ×3 | 3 | 2.89-3.13 (all protocols); ht8 {1,2}: 2.74 ± 0.37 (median 2.96) |
| 5 | mps6 {1,3,5}, mps8 {1,5} | 3 | χ = 2, σ in [0.03, 0.3]: 2.86-3.76; σ <= 0.03: mps6 3.05 ± 0.06 (saddle), 3.15 ± 0.23 (balanced), mps8 3.13 ± 0.25 (balanced, median 2.94), **3.67 ± 0.20 (saddle; 4 seeds, 0.4 decades, 10/24 stalled)**; mps8 small init on never-stalling seeds 2.97 ± 0.05; χ = 4: 3.00 ± 0.04, 3.19 ± 0.15 |
| 6 | binary8 {0,2,4}, ht4 {0,1,2} (**balanced only**, 1 decade, 8 / 7 seeds) | 4 | 4.007 ± 0.021, 4.061 ± 0.048 (fan-in, 0.8 decades: 3.68 ± 0.20, 4.23 ± 0.10) |
| 6 | mps8 {1,6} (balanced, 1 decade) | 4 | 4.51 ± 0.19 |
| 7 | ht4 all sites (**balanced only**, 1 decade, 8 seeds) | 5 | 4.957 ± 0.061 (median 5.01) |
| 7 | ht8 {3,4} (saddle, 0.6 decades) | 5 | 5.74 ± 1.06, median 5.01 (one pinned seed) |
| 8 | mps8 {0,7} | 6 | not measured |

λ2's K-dependence (weights (2, 1), saddle start; `configs/lambda2_K.yaml`): λ_min = -1.4142 / -1.5874 / -1.6818 for
K = 4 / 6 / 8 (predicted 1.4142 / 1.5874 / 1.6818); log slopes 0.712 / 0.649 / 0.595 (predicted 0.707 / 0.630 / 0.595).
One of 168 runs (mps8, seed 7, σ = 1e-4) was lost to the runner's start-up consistency check and not rerun.

### 2. Saddle type (H3; `results/twocomp_hessian.png`, `hessians.csv`)
Gauge-projected Hessian recomputed with the fixed projection (main 6c42d85) and raw, at the stored plateau
midpoints; classification by the σ-scaling of λ_min. k = 2: λ_min = -0.5000 from the saddle start (projected and
raw; -0.2500 / -0.7500 for s2 = 0.25 / 0.75), σ-independent, one negative eigenvalue per active route; small init
(raw): -0.498 to -0.499 at σ <= 1e-3, -0.437 / -0.581 on short plateaus at σ >= 1e-3, where the projection is still
biased (-0.411 / -0.502), so raw is quoted. k >= 3: λ_min ∝ σ^x, full ranges: k = 3 0.81-1.06, k = 4 1.64-2.16,
k = 5 2.82-3.76, k = 6 4.00 / 4.04 (balanced binary8 / ht4) and 4.27 (mps8 {1,6}), k = 7 5.01 (balanced) up to 6.5
(round-1 rows, 0.6 decades). Raw and projected exponents differ by up to 0.11.

### 3. Same teacher, different trees (step 3; `results/trees_TA.png`, `trees_B4*.png`, `trees_E8.png`, `levels_summary.csv`)
Every predicted k* >= 3 plateau is detected (`find_plateaus`, 88-100% of runs) with exponent within 0.2 of
k* - 2, except B4's second k* = 5 level on ht4 with fan-in init (4.73 ± 0.90, 4 seeds, 23/48 runs dead; balanced
3.17 ± 0.10). TA: mps4 two short plateaus (second log-time; first power-favoured, 0.22 vs 0.53 RMS, law
undetermined), binary4 one σ^-1 plateau (1.02), ht4 σ^-1 then σ^-2 (1.09, 2.08).
B4 on mps4 learns its 0.3 component before its 0.6 one (log, then 2.02). Route rule vs alternatives: TD last
plateau 1.012 ± 0.010 (8 seeds; detected 1.00), TC k* = 4 plateau 2.018 ± 0.017. k* = 2 levels are log-time where
measured over 3 decades from the saddle start; elsewhere undetermined.

### 4. Bond spectra (step 4; `results/bond_events2_twocomp.csv`, `bond_profiles_multicomp.csv`, `bond_spectra.png`)
**Leakage bound**: through every two-component event (5,130) the second singular value of every non-Steiner edge
stays at the init-noise level, <= 4.3 σ² × s2 (4.0 on the 4,447 resolved events), scaling as σ^2.07. **Single-route
consistency**: every Steiner edge gains rank between the plateau midpoint and a = 0.9 (99.75% of resolved events;
this check cannot register a non-Steiner rise, so it is not a prediction success rate), and the Steiner edges cross
together (median spread 5e-5 event widths, 95th percentile 0.25; the spread grows as σ^1.94 and tests "one route",
not timing).
Several components: the rank profile at every predicted saddle equals the route rule's prediction in 93.9% of
3,114 (run, level) pairs (99.6-100% for k* = 3).

### 5. Dead routes (`results/run_classes_by_sweep.csv`, `stall_dropout_*.csv`, `stall_capped_ends.csv`, `r12.csv`)
Stalled runs (gradient at roundoff after a component is learned): fan-in χ = 2: **398 / 5,682** (all with k >= 4;
none of 5,112 runs with k <= 3); balanced init 16 / 1,008 (10x fewer than fan-in on the same cases, 169 / 1,008; all
on MPS routes); χ = 4 **106 / 1,005** on the same 24 cases: 0 on routes with an uncapped end (79 at χ = 2), 106 on
routes with both ends capped at d (90 at χ = 2; rises in 7 of 8 cases), against theory_regimes R10. Stalls are
σ-dependent in 20 of 33 fan-in rows; restricting fits to never-stalling seeds moves 9 fan-in rows by 0.3-1.4 (it
explains the off-target saddle-start k = 4 and ht8 rows). Longer runs (60x patience, 170 runs): every fan-in /
saddle-start stall in the sample is dead (route ends at roundoff, 1e-17-1e-13, no escape by t ~ 1e10); with
balanced init 6 of 10 flagged runs escaped at ~1e5 x the normal plateau, 3 are pinned (ends 3e-10, 6e-9, 4e-8)
and one was mid-escape (ends 0.72) when its budget ended. R12's criterion (both ends c = Q_end + |y_route|^2 < 0)
is associated with death (30 of 55 such runs died vs 7 of 67 others) but is neither necessary nor sufficient;
**7 dead runs had a live route end** at the plateau start (theory_saddles P14(v)); balanced-init end shrinkage
and the HT-vs-MPS stall rates (ht4 k = 5: 0/144 vs mps4 {0,3}: 30/48) also conflict with theory_saddles.md
(notes 7.5, 7.10).

### 6. Robustness (`results/robust_table.md`)
χ = 4, d = 4 (χ = 2, 4, 8), s2/s1 = 0.25 / 0.75: k = 2 slopes match (1.00-1.01; 3.02 vs 3.00; 0.37 vs 0.33; saddle
start 4.04 / 1.345 vs 4.00 / 1.333); k = 3 0.99-1.05; k = 4 1.87-2.01.

### 7. Integrator accuracy
LSODA (rtol 1e-9 / 1e-10), Radau (1e-9 / 1e-10) and RK45 agree on a k = 6 gap to 3e-8; the reviewer's independent
dense/Radau implementation agrees on gaps to 1e-7-1e-9. Conservation-law drift <= 1e-6 absolute in every run
(Gram entries O(1)); relative to the initial imbalance |Q(0)| ~ σ² it is <= 1.6e-4 in every k >= 4 run, but up to 0.5
for k <= 3 at σ <= 1e-3 (σ² below the integrator's absolute error; irrelevant there: no stalls, gaps verified).

## Changes after the round-2 verification (full table: notes/locality.md section 10)
- (a) MPS k = 5 restated from the committed tables: only mps8 {1,5} from the saddle start stays steep (3.67 ± 0.20,
  4 seeds, 0.4 decades, ~40% stalled); the other MPS k = 5 rows are 2.97-3.15 at σ <= 0.03 or on never-stalling
  seeds. P14 is cited from theory_saddles.md (it was attributed to theory_regimes).
- (b) "Spare bond channels remove most dead routes" dropped: χ = 4 gives 106 / 1,005 vs 169 / 1,008; stalls rise on
  routes with both ends capped.
- (c) Bond spectra presented as a leakage bound plus single-route consistency, not as "99.75% prediction success".
- (d) Stall counts regenerated (398 / 5,682; 106 / 1,005); dropout list 9 fan-in rows (up to 1.38) plus 3 others;
  pinned-run amplitudes corrected (3 pinned, 1 mid-escape); 7 dead runs with a live route end flagged.
- (e) k = 6-7 labelled balanced-init only; full Hessian exponent ranges; raw quoted where the projection is biased
  ("raw and projected agree to 0.01" was false: up to 0.11).
- (f) TA mps4 first level: power favoured, law undetermined; B4 ht4 fan-in exception (4.73 ± 0.90); ht4 {0,1,2}
  7 fitted seeds.
- (g) `stall_tables.py` writes the run classes, dropout and capped-end tables; `finalize.sh` also writes the
  multi-component events CSV; `r12_partial.csv` removed (unused intermediate); extensions-table ghost row fixed.

## Changes after review round 1 (full table: notes/locality.md section 9)
- H2 scope: k = 1..8 -> k = 1-5 with named exceptions; ±0.05 only for small-init / balanced / χ = 4 k = 4 rows;
  k = 6-7 pinned only with the new balanced runs on binary8 / ht4; k = 8 not measured. ht8 k = 5 corrected to
  2.74 ± 0.37 (median 2.96); ht8 {3,4} reported with its median 5.01 and the pinned seed.
- Bond statistic: the near-tautological 98.7% (midpoint vs end of run) replaced by the before/after-event test,
  crossing-time spread, leakage and per-saddle profiles for several components.
- λ_min: k = 2 saddle-start values recomputed with the fixed projection (-0.5000, was -0.44..-0.46); raw reported.
- Stall counts corrected (balanced 16/1,008 not 2; χ = 4 103/1,002 not 75; fan-in 397/5,682); "stalls at every σ"
  withdrawn; dropout bias quantified.
- "Steep MPS exponents come from shrinking route ends" -> "associated with"; the log-correction form was tested
  and is rejected by the data; P14(ii) is met at χ = 2.
- TD 1.012 ± 0.010 (8 seeds); "visited fraction" replaced by detected plateaus; "log" labels only from the fair
  comparison; growth-estimate validity range stated; rotation runs excluded; k = 1 ρ computed; λ2(K) tested.
- Portability: `LOCALITY_OUT` / `LOCALITY_EAGER`, UTF-8 output, `finalize.sh <dir>`, zero-escape cases kept, the
  wall-clock check no longer applies during post-processing, `make_configs.py` writes every config.

## What was done
Round 3 (verification fixes, no new physics runs):
1. Merged main (verification note); `analysis.py` takes model fields of error records from the run key (ghost
   row); `run.py` start-up check gets an absolute floor; `stall_tables.py` new; `finalize.sh` writes every table.
2. Regenerated all tables and figures with `finalize.sh experiments/locality/results 16`.
3. Notes sections 7.1-7.10 and 8 corrected, section 10 (round-3 change list) added; this report updated.
Round 1 (see git history up to 562d92e): theory note, runner, sweeps, analysis (b0f9ad7 … 562d92e).
Round 2 (fix round after review 1):
1. Merged main (review, library fixes); analysis rework (`analysis.py`: config filter, run classes, rotation
   exclusion, ρ, per-seed dormant fits) (e14b6a6).
2. Hessians recomputed at all 11,780 plateau midpoints with the fixed projection and raw (`hess_recompute.py`).
3. Bond events redone (`bond_events2.py`), per-saddle rank profiles; levels with detected plateaus and fair labels.
4. New runs: k >= 6 over one decade (256), λ2(K) (168), the 11 dropped slow runs, stall diagnosis (170).
5. Log-correction test (`logcorr.py`); R12 evaluation (`r12.py`).
6. Notes sections 7-9 rewritten, README, finalize.sh.

## Decisions
- **Round 3: `r12_partial.csv` removed rather than generated**: it was an `r12.py` run on the stall diagnosis
  before it had finished, superseded by `r12.csv` and cited nowhere. (Reversible: it is in git history, e14b6a6.)
- **Round 3: the lambda2K run lost to the start-up check is not rerun**: the lead asked for no new physics runs and
  for `/root/outputs` to stay untouched; the row keeps 55 of 56 runs (8 seeds fitted).
- **Round 3: superseded round-1 files removed from `results/`** (`bond_events*.csv/txt` of round 1, `pilot_n4_*`):
  no script in `finalize.sh` makes them and the notes no longer cite them; they remain in git history.
- **Round-2 extension rows use gauge-balanced init** (not saddle start): it removes most dead routes, so ≥ 5 seeds
  survive at every σ over a full decade; saddle start at k >= 6 stalls in most runs.
- **Tables restricted to the committed configs** (provenance): the extra σ points of three k = 7-8 rows from an
  earlier config are dropped rather than added to a config.
- **Rotation runs excluded from fits** (analysis standard 3), counted per row.
- **Detected plateaus** (`ttn.diagnostics.find_plateaus`) replace the band-occupation exponent for multi-component
  levels; the occupation time is kept for the fair log/power label of short levels.
- **Stall diagnosis** uses a 60x longer stall patience and saves partial trajectories rather than disabling the
  detector (dead runs would otherwise crawl to t_max for hours).
- Same runner and integrator as round 1 (LSODA via `solve_ivp`), for comparability with the reviewed results.

## Files left on the machine
- `/root/outputs/locality/<prefix>.part*`: run records (twocomp, twocomp_balanced, twocomp_chi4, robust, k5ext,
  khigh, lambda2K, multicomp, multicomp_wide, stalldiag); needed to regenerate every table.
- `/root/outputs/locality/traj/*.npz`: every accepted integrator step of every run (several GB).
- `/root/outputs/locality/obsolete_prefix/`: pre-fix pilot outputs and the invalid first stall-diagnosis attempt
  (t_max too small), not used.

## Caveats and next steps
- k = 6-7 rest on four extension rows (binary8, ht4 ×2, mps8); k = 8 is not measured; coverage per k is uneven
  (k = 2, 4 only on chains and star3; k = 7 only on HT).
- The steep MPS exponents (k >= 5 at χ = 2) are associated with route ends that shrink during the plateau; the
  mechanism is not identified (not R12's gauge trap alone: it also occurs with balanced init). The one row that
  stays steep at small σ (mps8 {1,5}, saddle start) rests on 4 seeds over 0.4 decades.
- R12's c is evaluated once, at a1 = 0.99; the 7 dead runs with a live end may have c changing sign later in the
  plateau. Evaluating c along the plateau would decide whether P14(v) really refutes the picture.
- The rise of χ = 4 stalls on routes with both ends capped has no mechanism yet.
- The midpoint Hessian is not representative for balanced-init mps8 {2,5}, {1,5} (route in a sign-correction
  phase at the minimum-speed point).
- Wrong-order and star saddles (H3's other strict cases) were not tested.
- Of the 11 slow runs dropped in round 1, 5 finished in round 2 (2-hour limit per run); 6 remain unfinished
  (ht8 all-sites saddle start, seed 3, σ = 0.26 / 0.20 / 0.15; mps8 {0,7} χ = 4, seeds 0 / 4, σ = 0.25 / 0.20).
  They belong to rows already reported as not measured (k = 8) or as spanning 0.6 decades (ht8 k = 7); rerunning
  `run.py configs/finish_twocomp*.yaml` resumes them.

## Log
- 18:20-21:10 round 1 (see git history and the round-1 report in commit 562d92e)
- 21:42 fix round started: merged main, read review locality_round1
- 21:43 launched k >= 6 extension, dropped runs, λ2(K), stall diagnosis
- 21:50 analysis rework; Hessians recomputed (projection bug confirmed: -0.4997 raw vs -0.478 stored)
- 21:55 bond events v2 (route rule at plateau states needed a subspace tolerance and the actual teacher basis)
- 22:00 λ2(K) confirmed exactly; log-correction test rejects the log-correction form in the steep MPS rows
- 22:02 first stall diagnosis invalid (t_max too small); redone with long patience and partial trajectories
- 22:05 library gauge fix merged; Hessians recomputed with it
- 22:10 corrected tables, level analysis with detected plateaus, notes sections 7-9, README
- 22:45 stall diagnosis complete (170 runs): fan-in stalls dead, balanced stalls pinned; R12 criterion
  associated but not deterministic; conservation-drift statement corrected
- 23:20 stopped the 6 remaining slow runs (documented); final tables, report
- 00:05 round 3: merged main (verification note); provenance fixes (stall_tables.py, finalize.sh, ghost row)
- 00:07 finalize.sh regenerating all tables; notes 7.1-7.10, 8, 10 and report corrected (items a-f)
