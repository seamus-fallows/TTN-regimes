# Part 3: independent checks

## Steiner counts (steiner_check.py; output data/review_locality/steiner_check.txt)
Own implementation (prune non-terminal leaves of the undirected tree graph) vs ttn.tree.Tree.steiner_nodes vs
rules.rank_rule (the column the tables label "k (Steiner)") vs k in results/twocomp_table.csv:
- all 60 (tree, S) pairs of configs/twocomp.yaml: 0 mismatches;
- exhaustive, every nonempty S on all 9 trees (999 pairs): 0 mismatches between own / library / rank rule.

## Tolerance statement in REPORT summary ("k - 2 to +-0.05 for k <= 4")
Committed twocomp_table.csv (d = 2, chi = 2, w = 1/0.5): k = 3: 42/42 rows within 0.035. k = 4: 5 of 16 rows
outside +-0.05: saddle midroot6 S=03 1.485, mps4 S=0123 1.925, mps6 S=03 2.390, mps6 S=14 2.192; small mps6 S=14
2.063. k = 5: 4 of 17 rows outside +-0.15 (mps6 S=135 small 3.761, mps8 S=15 small 3.278 / saddle 3.390, ht8 S=12
saddle 2.737) -> "most" is fair. The REPORT's detailed table does list 1.49-2.39 for saddle-start k = 4.

## Dead-route counts in the committed tables (claim 7)
- results/twocomp_balanced_table.csv: 16 stalled of 1008 balanced runs (mps8 S=25 2, mps6 S=135 1, mps8 S=15 4,
  mps8 S=16 5, mps8 S=07 4), NOT "2 of 1,008" (REPORT sec. 6, CLAUDE-level claim) nor "2 stalls in 580 runs"
  (notes 7.5). Same cases with fan-in init: 169 / 1042 stalled. So balanced init cuts stalls ~10x, not ~85x,
  and does not remove all k = 4 stalls (mps8 S=25, k = 4: 2/48).
- k <= 3: 0 stalls in 3560 runs of the main table (claim "none in ~3000 runs" holds).
- main table: 397 stalled + 3 timeouts of 5761 runs (REPORT: 376/5732); chi = 4: 73/972 (REPORT: 75/1002). Small
  bookkeeping differences (unfinished runs at the end).

## Growth exponent (claim 5) in the committed growth_summary.csv
k_est = 2/(2-gamma) medians: k = 2 saddle 1.993-1.998 (7 rows); k = 2 SMALL init 1.994-3.325 (mps6 S=23 2.36,
mps8 S=01 2.89, mps8 S=34 3.33): the single-run estimate fails for small-init k = 2 on N >= 6 MPS (the window
a in [1e-4, 1e-2] is not on the plateau). k = 3: 2.99-3.03, 4: 3.99-4.09, 5: 5.06-5.32, 6: 5.96-6.20,
7: 6.93-7.73, 8: 7.66-7.89. The notes restrict k = 2 to the saddle start; the REPORT summary does not.

## TD alternatives (rules.py, recomputed)
After U1+U2 learned, U3 = 11000000: literal H2 k = 2 (log law), exact rank rule k = 4 (edges 0,1,2 rise; cores
0-3; Furman N = 4); route rule (route_sequences.json): partial k* = 2 step to L = 0.06125 -> 0.015185, then k* = 3
(route [1,2,3]). Committed levels_summary.csv: level 2 (k* = 3) p = 1.0121 +- 0.0096, 8 seeds (notes say
"1.02 +- 0.01 (7)": rounding / seed count differ from the CSV).

## Agent's pipeline on the rerun (make_figures.py -> figs/main_table.csv vs committed; figs/compare_committed.txt)
All event-time quantities identical to 5 significant digits (p_fit, p_se, p_local, logslope, gap medians, run
classes) for all 10 rows; Hessian quantities agree to 0.1-0.5% (midpoint = min-speed STORED step, which moves
slightly with eager vs compiled Jacobian roundoff). ht8 S=34 saddle: 27/30 runs (seed 4 sigma <= 0.26 still
running/killed): p 5.68 +- 1.00 (median 5.01) vs committed 5.74 +- 1.06 (median 5.01).
make_figures.py crashes on Windows after writing the CSV: Path.write_text without encoding -> UnicodeEncodeError
on 'sigma' (cp1252); rerun with PYTHONUTF8=1.

## Dead routes, independent (stall_check.py: own model, LSODA, no stall detector)
mps4 S=03 sigma=0.2 seed 1 (runner: stalled): after U1 is learned (t ~ 1e2) s_2 on the three route edges decays
4e-3 -> 1e-5 (t=1e3) -> 1e-14 (1e4) -> 1e-16 (1e5..1e10); |grad| ~ 1e-16; a_2 = 0 up to t = 1e10. seed 5 sigma=0.1099:
same, s_2 at roundoff from t = 1e6 to 1e10. Control seed 0 sigma=0.2: both components learned by t = 1e3,
s_2 = 0.5 on all three edges. -> the runner's 'stalled' label is right for these runs (route decays to roundoff,
not merely slow). Beyond t = 1e10 LSODA crawls on roundoff (stopped there).

## Bond spectra, independent (my_bonds.py: own dense W + SVD on the saved trajectories)
binary8 S=34 small (48 runs), ht8 S=34 saddle (27), mps8 S=34 saddle (56): in 131/131 runs the non-Steiner edges
never reach rank 2 during the whole run (max s_2 / (0.1 x teacher value) = 0.39, 0.88, 0.12), all Steiner edges
reach s_2 = s2/2, and they do so within 0.0024 / 0.0099 / 0 event widths of each other and of t2. This is the
real content of claim 4 and it holds; the committed 98.7% statistic itself does not test it (see part 2).

## Gauge-projected Hessian, independent (indep.py hess: own contraction, own gauge generators, own projection)
At the runner's recorded plateau midpoints (state read from the saved trajectory):
- own projected lambda_min = runner's hess_min to 1e-15 (k = 2) .. 7e-11 (k = 5) relative, all 53 states;
  own gauge directions leave W unchanged to first order (|J u| <= 1e-16).
- k = 2 saddle start, sigma = 1e-4: runner -0.444 +- 0.024 (mps4 S=12, 8 seeds), -0.460 +- 0.013 (mps8 S=34);
  keeping only gauge generators with singular value > 1e-3 x max (drops the O(sigma) generators on the unused
  bond index): -0.50000 (range -0.50004..-0.49995) and -0.50001; unprojected Hessian: -0.5000. So the theory
  value -lambda2 = -0.5 is exact; the reported -0.44..-0.46 is a projection artifact (the near-degenerate
  generator rescales the route's new-index entries and, normalized, overlaps the escape direction).
- k = 2 small init sigma = 1e-4: -0.4987 (runner = own = cut3; route already balanced).
- k >= 3, seed 0, sigma-exponent of -lambda_min: binary4 S=12 1.035 (k-2 = 1), mps4 S=0123 2.003 (2), binary8
  S=34 3.064 (3); identical with the unprojected Hessian; moving the state to the balanced gauge representative
  changes the exponent by <= 0.016 but the prefactor by 3-12% (the projected value is gauge dependent).
- run.balance_gauge returns non-finite cores on the small-init k = 2 midpoint states (ill-conditioned bond
  Grams; eigenvalue clamp 1e-300 -> overflow). Harmless for its use at t = 0.

## TD on mps8, saddle start (claim 6), own level-occupation analysis (my_levels.py; own loss on saved trajectories)
55/56 runs order 0>1>2 (1 run 0>2>1, as committed). Per-seed power fits of the time within +-2% of each level:
level 0 (L = 0.24125, k* = 2): 0.267 +- 0.016; level 1 (0.06125, k* = 2): 0.649 +- 0.050; level 2 (0.015185, k* = 3):
1.012 +- 0.010 (8 seeds) -- committed 0.2666 +- 0.016, 0.6487 +- 0.050, 1.0121 +- 0.0096. Level 2 supports k* = 3
(route rule) over k = 2 (literal H2, log) and k = 4 (rank rule, p = 2).
BUT level 1 is labelled "log" in notes/REPORT while the committed fair comparison (levels_summary.csv) prefers a
power law: RMS log-residual 0.389 (log law) vs 0.161 (power, p = 0.65), with 12/55 durations unresolved. The
"log" label for TD level 1 is not supported by the committed analysis.

## Growth exponent on the rerun (agent's growth.py)
gamma medians identical to growth_summary.csv for all 8 rows (k_est 2.991, 3.985, 3.985, 1.993, 2.024, 1.998,
5.119 (5.120 committed), 7.724 (7.733)). k_est is biased high for k >= 5 (5.12 for k = 5, 7.72 for k = 7).

## Independent gradient flow (indep.py flow: own dense contraction + autograd, scipy Radau rtol 1e-10, own
## event rule; init regenerated and asserted equal to TTNSpec.init; same teacher vectors)
binary4 S=12 small seed 0: t2-t1 = 524.01734 (sigma 0.01) and 5253.0933 (0.001) vs runner 524.01714 / 5253.0718
(rel. diff 4e-7 / 4e-6, within LSODA rtol 1e-9 accumulated over the plateau). mps4 S=12 saddle seed 0, sigma
0.1 / 0.01 / 0.001: gaps 5.2972549 / 9.2699118 / 13.772187, identical to the runner to 1e-9.
Further independent flows (all seed 0): binary8 S=34 small sigma 0.1 (k = 5): gap 151429.92 vs runner 151429.91
(3e-8); mps4 S=0123 small 0.01 (k = 4): 21534.462 vs 21534.462 (2e-8); mps4 S=12 small 1e-2/1e-3/1e-4: gaps
2.3294874 / 4.6640791 / 6.969701, equal to 1e-8; own seed-0 log slope 1.0076 (pred. 1.000); saddle start
1e-4..0.1: 1.898 (4 points; committed per-seed mean over 7 points 2.018 +- 0.024). 11/11 independent runs agree.
