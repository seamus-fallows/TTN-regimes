# route-nonstar: training on pattern targets that break the star condition

Status: done - Branch: agent/route-nonstar - Updated: 2026-10-13 21:05 UTC

**Bugs in `ttn/routes.py`: none found.** Two things to know (not bugs, worked around in
`experiments/route_nonstar/`, `ttn/` untouched): (1) `effective_depth` raises `IndexError` (from
`np.gradient`) on a window with fewer than 2 samples instead of returning None (`run.eff_depth`
guards it); (2) `AlignedManifold.align` / `route_entry_norms` use the manifold's own bond dimensions
(uniform max(chi, blocks)); on a network with other bond dimensions they work if `.spec` is replaced
by the network's spec (`entries.py`). For non-star targets `RouteTarget.n_blocks` counts connected
pieces of the edge support graph, which is not the bond rank (by design); bonds here are rank + 2.

## Summary

Asked: what gradient-flow training of a tree tensor network from small random init does on pattern
targets that break the paper's star condition, and which of the paper's predictions survive. I ran
17 non-star targets (recombinant family on 3- and 4-core chains, the nested chain on three trees,
two-edit targets, six random targets) and 3 star controls, 40 seeds on a quarter-decade sigma grid
(9,520 runs). **Answer:** training still goes in steps with plateaus, and the paper's predictions
hold wherever a plateau is an ordinary partial target (some patterns learned, the rest absent).
But on non-star targets most later plateaus are not: training stops at the best approximation the
network's current bond ranks allow. That point contains a *recombinant*, a string that is not in
the target, made of one pattern's left part and another pattern's right part, e.g. 101 from 100
and 001. There the paper's route depth, applied anyway, says "no plateau" (every pattern is
already learned, or the missing ones cost <= 1 core), yet a plateau of length sigma^-1 to sigma^-3
follows. The pairwise rule fails the same way. **What replaces them is the repo's numerical route
rule evaluated at the actual plateau point**: it predicts the plateau-length exponent (23 of 25
fitted plateau types, every seed within 0.3), the escape's effective depth (95 % of escapes), the
escape type (strict iff k* = 2, 42 of 42) and what is removed or learned next (89-99 %). Its
failures are spectral shortcuts (order below the route) and one plateau with a non-integer exponent.

## Results

Glossary: *plateau* = a long stretch of nearly constant loss; *escape* = the loss drop that ends it;
*partial target* = W equals the sum of the learned patterns at full weight; *route depth* c = the
number of cores that must change to create the next pattern (the paper's prediction: plateau
length ~ sigma^-(c-2), log(1/sigma) for c = 2, no plateau for c <= 1); *k** = the same count from
the numerical route rule at the measured plateau point; *exponent* = slope of -log(plateau length)
vs log(sigma), fitted per seed over log10 sigma in [-4 or -3.5, -1.5]; *strict saddle* = the Hessian
has a negative eigenvalue that does not shrink with sigma. Numbers: `experiments/route_nonstar/results/headline.md`
(from `analyze.py` + `headline.py`); details: `notes/route_nonstar.md` Part II.

**Which predictions survive outside the star condition**

| prediction | verdict on non-star targets | evidence |
|---|---|---|
| plateaus are partial-target points | **fails** for most later plateaus: 68 % of plateau points after the first escape are > 0.03 (relative) from every partial target, 59 % carry a recombinant (star controls: 0.4 %, 0.9 %); they are rank-constrained optima (18 of 19 non-canonical plateau types at a predicted rank-constrained level within 5 %) | `transitions.csv.gz`, `fig2` |
| plateau ~ sigma^-(c-2) with the paper's route depth c | **holds at partial-target plateaus** (8 of 10 fitted types), **fails at all rank-constrained optima** (13 types where it predicts no plateau but one exists, 3 more with a wrong exponent) | `fit_summary.csv`, `fig3` |
| pairwise (Prim) rule | same as the paper's depth on these targets (9 / 25 hold) | `fit_summary.csv` |
| numerical route rule k* at the plateau point | **holds**: 23 / 25 fitted non-star types (all fitted seeds within 0.3, quartile spread <= 0.03); fails at 2 (below) | `fit_summary.csv`, `fig3` |
| effective depth = depth | route rule: within 0.5 in 94.7 % of 8,184 non-star escapes (median miss 0.03); paper's depth: 30.9 % | `transitions.csv.gz`, `fig4` |
| strict saddle iff order 2, degenerate otherwise, lambda_min ~ sigma^(p-2) | **holds with p = k***: all 10 k* = 2 types strict (lambda_min -0.06 to -0.60, sigma-independent), all 32 k* >= 3 types degenerate (slope 1.00-1.07 for k* = 3, 1.85-1.99 for k* = 4, 2.69-3.04 for k* = 5; exceptions: the 2 spectral shortcuts have 1.99-2.00, RND3_mps6 0.51) | `escape_type.csv`, `fig5` |
| greedy next pattern (paper) | right on 99.4 % of the 4,806 non-star escapes that learn a pattern; but 45 % of all non-star escapes (3,953) learn nothing ("clean-ups": a recombinant is removed), which the paper cannot predict; overall 54.5 % vs route rule 89.4 % (98.8 % counting the string it names, target or not) | `sequence.csv`, `fig6` |
| Theorem 3.2 order min(k*, 2 l*) at a one-pattern plateau | **holds** on every canonical one-pattern plateau, also on non-star targets; it also predicts the spectral shortcut on the star control CTL1_mps6 (order 4 < k* = 5, exponent 2.00) | `fit_summary.csv` (`exp_hand`) |
| by-hand order of the recombinant plateau (registered: order K, the whole chain) | **holds**: R3, R3d 1.00 [1.00, 1.00] (40 seeds); R4, R4d 2.00 [2.00, 2.00] (31 seeds) | note section 2, 11 |
| min(direct, 2 x spectral) at multi-pattern plateaus (App. B) | not testable by hand; at RND1_mps6's canonical three-pattern plateau the order is 4 (exponent 2.02 [2.01, 2.05], effective depth 3.91) against route depth 5; the parent-wise formula of Thm 9.2 gives 2 (wrong) | note section 11 |

**Per target** (plateau types with >= 20 window points; prediction "e: decision (fraction of fitted
seeds within 0.3)"; "secondary" = depth-5 plateaus whose registered window is cut by the float64
floor, fitted with floor-flagged sigmas included; escape type with the slope x of
log|lambda_min| vs log sigma; full tables in `results/summary_by_target.md`):

| target | plateau: learned set J (loss) | what it is | exponent: median [quartiles] (fitted seeds) | paper's depth | route rule | pairwise | escape type (x) |
|---|---|---|---|---|---|---|---|
| R3 | {000,100,001} (0.01944) | rank-constrained optimum + 101 (recombinant) | 1.00 [1.00, 1.00] (40) | 0: FAILS: predicts no plateau (0.00) | 1: holds (1.00) | 0: FAILS: predicts no plateau (0.00) | degenerate (1.00) |
| R3d | {000,100,001,101} (0.01953) | rank-constrained optimum + wrong amplitudes, no off-target string | 1.00 [1.00, 1.00] (40) | 0: FAILS: predicts no plateau (0.00) | 1: holds (1.00) | 0: FAILS: predicts no plateau (0.00) | degenerate (1.00) |
| R4 | {0000,1000,0001} (0.01944) | rank-constrained optimum + 1001 (recombinant) | 2.00 [2.00, 2.00] (31) | 0: FAILS: predicts no plateau (0.00) | 2: holds (1.00) | 0: FAILS: predicts no plateau (0.00) | degenerate (1.85) |
| R4d | {0000,1000,0001,1001} (0.01953) | rank-constrained optimum + wrong amplitudes, no off-target string | 2.00 [2.00, 2.00] (31) | 0: FAILS: predicts no plateau (0.00) | 2: holds (1.00) | 0: FAILS: predicts no plateau (0.00) | degenerate (1.86) |
| R3p | none after the first escape | W* is a product tensor: one drop to zero loss | - | holds (depths 1, 1, 0) | holds | holds | - |
| R4p | none after the first escape | W* is a product tensor: one drop to zero loss | - | holds (depths 1, 1, 0) | holds | holds | - |
| NC_mps4 | {0000} (0.2613) | partial target | 0.07 [0.06, 0.07] (6) | 0: holds (1.00) | 0: holds (1.00) | 0: holds (1.00) | strict (-0.00) |
| NC_mps4 | {0000,1100} (0.06375) | rank-constrained optimum + 0010 (recombinant) | 0.14 [0.13, 0.15] (17) | 0: FAILS: predicts no plateau (1.00) | 0: holds (1.00) | 0: FAILS: predicts no plateau (1.00) | strict (-0.00) |
| NC_mps4 | {0000,1100,1110} (0.01831) | rank-constrained optimum + 0001 (recombinant) | not fitted (1 seeds) | 0: not fitted | 0: not fitted | 0: not fitted | strict (0.00) |
| NC_binary4 | {0000,1100} (0.05775) | rank-constrained optimum + 0010 (recombinant) | 1.00 [1.00, 1.01] (40) | 0: FAILS: predicts no plateau (0.00) | 1: holds (1.00) | 0: FAILS: predicts no plateau (0.00) | degenerate (1.01) |
| NC_ht4 | {0000} (0.2612) | partial target | 1.00 [1.00, 1.01] (40) | 1: holds (1.00) | 1: holds (1.00) | 1: holds (1.00) | degenerate (1.02) |
| NC_ht4 | {0000,1100} (0.06376) | rank-constrained optimum + 0010 (recombinant) | 1.01 [1.00, 1.01] (40) | 0: FAILS: predicts no plateau (0.00) | 1: holds (1.00) | 0: FAILS: predicts no plateau (0.00) | degenerate (1.00) |
| NC_ht4 | {0000,1100} (0.05775) | rank-constrained optimum + 0010 (recombinant) | 1.03 [1.03, 1.04] (40) | 0: FAILS: predicts no plateau (0.00) | 1: holds (1.00) | 0: FAILS: predicts no plateau (0.00) | degenerate (1.02) |
| TD_mps6 | {000000,000110} (0.06124) | partial target | 0.05 [0.04, 0.06] (5) | 0: holds (1.00) | 0: holds (1.00) | 0: holds (1.00) | strict (-0.00) |
| TD_mps6 | {000000,000110,110000} (0.01518) | rank-constrained optimum + 110110 (recombinant) | 1.01 [1.00, 1.01] (40) | 0: FAILS: predicts no plateau (0.00) | 1: holds (1.00) | 0: FAILS: predicts no plateau (0.00) | degenerate (1.00) |
| TD_binary8 | {00000000,11000000,00001100} (0.01518) | rank-constrained optimum + 11001100 (recombinant) | 3.00 [2.99, 3.00] (40, secondary: floor-flagged sigmas included) | 0: FAILS: predicts no plateau (secondary) | 3: holds (secondary fit) | 0: FAILS: predicts no plateau (secondary) | degenerate (2.81) |
| RND1_mps6 | {010110,010100} (0.6012) | partial target | 1.01 [1.00, 1.01] (20) | 1: holds (1.00) | 1: holds (1.00) | 1: holds (1.00) | degenerate (1.00) |
| RND1_mps6 | {010110,010100,101110} (0.5259) | rank-constrained optimum + 101100 (recombinant) | 1.01 [1.00, 1.01] (20) | 3: FAILS (0.00) | 1: holds (1.00) | 3: FAILS (0.00) | degenerate (1.00) |
| RND1_mps6 | {110101} (0.5151) | partial target | 3.15 [3.04, 3.53] (20, secondary: floor-flagged sigmas included) | 3: holds (secondary fit) | 3: holds (secondary fit) | 3: holds (secondary fit) | degenerate (3.04) |
| RND1_mps6 | {010110,010100,101110} (0.5) | partial target | 2.02 [2.01, 2.05] (19) | 3: FAILS (0.00) | 3: FAILS (0.00) | 3: FAILS (0.00) | degenerate (2.00) |
| RND1_mps6 | {110101,101110} (0.4138) | partial target | 1.02 [1.01, 1.04] (20) | 1: holds (1.00) | 1: holds (1.00) | 1: holds (1.00) | degenerate (1.01) |
| RND1_mps6 | {110101,010110,010100,101110} (0.02587) | rank-constrained optimum + 101100 (recombinant) | 1.01 [1.00, 1.02] (40) | 0: FAILS: predicts no plateau (0.00) | 1: holds (1.00) | 0: FAILS: predicts no plateau (0.00) | degenerate (1.00) |
| RND2_mps6 | {000101} (0.1525) | partial target | not fitted (0 seeds) | 0: not fitted | 0: not fitted | 0: not fitted | strict (-0.00) |
| RND2_mps6 | {000101,011101} (0.05752) | rank-constrained optimum + 000100 (recombinant) | 2.00 [2.00, 2.00] (38) | 0: FAILS: predicts no plateau (0.00) | 2: holds (1.00) | 0: FAILS: predicts no plateau (0.00) | degenerate (1.94) |
| RND3_mps6 | {000010} (0.8808) | rank-constrained optimum + 100010 (recombinant) | not fitted (2 seeds) | 1: not fitted | 0: not fitted | 1: not fitted | strict (0.04) |
| RND3_mps6 | {110100,101100} (0.156) | partial target | not fitted (2 seeds) | 0: not fitted | 0: not fitted | 0: not fitted | strict (-0.00) |
| RND3_mps6 | {110100,101100} (0.1255) | rank-constrained optimum + 110010 (recombinant) | 0.17 [0.16, 0.17] (34) | 0: holds (1.00) | 0: holds (1.00) | 0: holds (1.00) | strict (-0.00) |
| RND3_mps6 | {110100,101100,101010} (0.08) | partial target | 0.61 [0.57, 0.70] (40) | 1: FAILS (0.25) | 1: FAILS (0.25) | 1: FAILS (0.25) | degenerate (0.51) |
| RND1_binary8 | {01010101,01000101} (0.252) | partial target | 3.00 [3.00, 3.00] (40, secondary: floor-flagged sigmas included) | 3: holds (secondary fit) | 3: holds (secondary fit) | 3: holds (secondary fit) | degenerate (2.92) |
| RND1_binary8 | {01010101,00010110,01000101} (0.05444) | rank-constrained optimum + 00000110 (recombinant) | not fitted (0 seeds) | 0: not fitted | 0: not fitted | 0: not fitted | strict (-0.00) |
| RND2_binary8 | {00111001,00111101} (0.1522) | partial target | 1.01 [1.01, 1.01] (40) | 1: holds (1.00) | 1: holds (1.00) | 1: holds (1.00) | degenerate (1.00) |
| RND2_binary8 | {00111001,00111101,10001001} (0.1002) | rank-constrained optimum + 10001101 (recombinant) | 2.00 [2.00, 2.00] (40) | 0: FAILS: predicts no plateau (0.00) | 2: holds (1.00) | 0: FAILS: predicts no plateau (0.00) | degenerate (1.96) |
| RND3_binary8 | {01111001} (0.5393) | partial target | 1.01 [1.00, 1.01] (16) | 1: holds (1.00) | 1: holds (1.00) | 1: holds (1.00) | degenerate (1.00) |
| RND3_binary8 | {11001001,11001000} (0.5) | partial target | 1.01 [1.00, 1.01] (22) | 1: holds (1.00) | 1: holds (1.00) | 1: holds (1.00) | degenerate (1.07) |
| RND3_binary8 | {01111001,11001001,11001000} (0.1142) | rank-constrained optimum + 01111000 (recombinant) | 2.00 [2.00, 2.00] (40) | 0: FAILS: predicts no plateau (0.00) | 2: holds (1.00) | 0: FAILS: predicts no plateau (0.00) | degenerate (1.99) |
| B4_mps4 | {0000} (0.225) | partial target | not fitted (0 seeds) | 0: not fitted | 0: not fitted | 0: not fitted | strict (-0.00) |
| B4_mps4 | {0000,0110} (0.18) | partial target | 2.00 [2.00, 2.00] (35) | 2: holds (1.00) | 2: holds (1.00) | 2: holds (1.00) | degenerate (1.87) |
| CTL1_mps6 | {100100} (0.8612) | partial target | 1.01 [1.01, 1.01] (7) | 1: holds (1.00) | 1: holds (1.00) | 1: holds (1.00) | degenerate (1.01) |
| CTL1_mps6 | {100011} (0.7888) | partial target | 1.00 [1.00, 1.00] (8) | 1: holds (1.00) | 1: holds (1.00) | 1: holds (1.00) | degenerate (1.02) |
| CTL1_mps6 | {010101} (0.65) | partial target | 2.00 [2.00, 2.01] (14) | 3: FAILS (0.00) | 3: FAILS (0.00) | 3: FAILS (0.00) | degenerate (1.99) |
| CTL1_mps6 | {100011,100100} (0.5) | partial target | 3.13 [3.00, 3.49] (31, secondary: floor-flagged sigmas included) | 3: holds (secondary fit) | 3: holds (secondary fit) | 3: holds (secondary fit) | degenerate (2.72) |
| CTL1_binary8 | {00011010} (0.4098) | partial target | 3.00 [3.00, 3.00] (5) | 3: holds (1.00) | 3: holds (1.00) | 3: holds (1.00) | degenerate (2.69) |
| CTL1_binary8 | {00011010,10011000} (0.1853) | partial target | 1.01 [1.01, 1.02] (40) | 1: holds (1.00) | 1: holds (1.00) | 1: holds (1.00) | degenerate (1.01) |
| CTL1_binary8 | {00011010,10011000,11001000} (0.1013) | partial target | 3.00 [2.99, 3.00] (40, secondary: floor-flagged sigmas included) | 3: holds (secondary fit) | 3: holds (secondary fit) | 3: holds (secondary fit) | degenerate (2.73) |

**Runs that did something else** (all counted; none dropped):

| what | runs | where | what happened |
|---|---|---|---|
| trapped (stopped: gradient < 1e-14 for 2 decades after a pattern was learned) | 211 | R4 81, R4d 65, B4_mps4 65 | dead routes: the same 6 (R4), 5 (R4d), 5 (B4) seeds at all 13 sigmas; the depth-4 route through both chain ends dies (all non-learned entries of W ~ 1e-15). R4, R4d stay at the rank-1 optimum with recombinant 1001 / wrong amplitudes; B4 at {0000, 0110} |
| unfinished (1800 s wall cap) | 79 | CTL1_mps6 58 (sigma <= 10^-2.25), TD_binary8 21 (sigma = 10^-3.5) | a depth-5 plateau at the float64 floor (gradient 3e-16 to 6e-14); all floor-flagged |
| floor-flagged at some plateau | 1,185 | depth-5 plateaus at sigma <= 1e-3, depth-4 at <= 10^-3.75 | excluded from registered fits |
| forgetting (a pattern falls below 1/4 of its weight after passing 1/2) | 452 | mostly inside the first escape; later only in CTL1_mps6 (star control: 21 of 40 seeds lose the heaviest pattern 010101 in its spectral escape and relearn it after a depth-5 plateau), RND1_mps6, RND3_mps6 | |
| errors | 0 | | |

Nothing was cut from the grid: all 20 targets, all sigmas (K <= 4: 10^-1 ... 10^-4; K = 6, 7:
10^-1 ... 10^-3.5) and 40 seeds ran to an outcome.

## What was done

1. Read the brief, the paper (abstract, Secs 1-3, 6.3, 7, 8, 10, App. B, C), `notes/locality.md`
   5, 7-11, `experiments/locality/README.md`, `notes/saddle_dyn.md` 8; checked files and
   `uv run pytest tests/test_routes.py` (13 passed). 7261bb8
2. Driver `nsdyn.py` (dense contraction, exact Hessian, step-driven LSODA with local clock as in
   `sdyn.py`) and per-run analysis `run.py`; pilots (2-25 s per run); targets `make_targets.py`.
3. Pre-registration `notes/route_nonstar.md` Part I, pushed before the sweep. 504dd37
4. Sweep: 9,520 runs, 120 workers, 68 min.
5. Analysis `analyze.py` (CSVs, fits, decisions), deviations 1-3 (note section 8). 636b8d6, 7222ab1
6. First results tables pushed. 7e1fb6b
7. Figures `figures.py` (fig1-fig6), headline numbers `headline.py`, loader `load.py`,
   post-hoc route-entry norms `entries.py` (deviation 4), note Part II, README. cf5bc20 ... HEAD

## Decisions

- **Plateau = loss plateau, not only learned-set change.** On these targets several patterns are
  learned in one drop and later drops learn nothing (clean-ups), so plateaus are found on the loss
  (`find_plateaus`, min_drop 2 %) and each is characterised at its minimum-gradient point; the
  brief's event table is kept as well (`events.csv`). Pre-registered.
- **Bond dimension = teacher rank across the edge + 2** (the brief's "blocks + 2"; for non-star
  targets the number of support-graph pieces is not the rank).
- **Dense contraction instead of the library's `Problem`** (at most 256 entries; exact Hessian by
  torch.func; simpler and fast enough: 2-60 s per run).
- **Transient segments merged into escapes** (deviation 1): a two-stage loss drop was being split
  into two "plateaus"; plateau = segment resolved by `find_plateaus` (the repo's definition).
- **Secondary fits for depth-5 plateaus** (deviation 2): the registered floor flag leaves only 1.25
  decades; reported separately and never used for registered decisions.
- **"No plateau predicted" scored by existence** (deviation 3): an exponent near 0 cannot tell a
  log-time plateau from no plateau.
- **Route-entry norms post hoc** (deviation 4): the in-run computation was silently skipped
  (bond-dimension mismatch); recomputed from saved states. They turned out not to be informative at
  the minimum-gradient point (note section 13).
- Pattern columns are text: read the CSVs with `load.read` (or `dtype=str`).

## Files left on the machine

| path | size | what | needed to reproduce |
|---|---|---|---|
| `/root/outputs/route-nonstar/raw/<target>/s<log10 sigma>_seed<k>.json` + `.npz` | 2.5 GB (19,040 files) | per-run record and trajectory (every solver step: t, L, grad, amplitudes, off-target mass; W at <= 400 steps; theta at every plateau point) | yes, for `analyze.py`, `entries.py`, `figures.py` fig1 (or rerun the sweep: 68 min on 120 cores) |
| `/root/outputs/route-nonstar/logs/sweep.log` | 0.5 MB | sweep log | no |

## Caveats and next steps

- The order of the rank-constrained plateaus is inferred from dynamics (exponent, effective depth,
  Hessian scaling), not measured statically; a sphere-minimum order measurement (as in
  `route-escape`) at those points would close this.
- Depth-5 plateaus have only secondary fits (float64 floor below sigma = 1e-3; plain loss, no
  cancellation-free reference point).
- The trap rule spaces its checks in absolute time, so it missed the 79 floor-limited runs (they hit
  the wall cap instead; same classification in the end).
- Spectral shortcuts at multi-pattern saddles (RND1_mps6) and the non-integer exponent of RND3_mps6
  are single cases; their mechanisms are open (note sections 11, 13).
- d = 2 only; K <= 7; one weight vector per hand-picked target.
- The lead should check `results/fit_summary.csv` decisions against `fits.csv` for any target they
  quote, and read the CSVs with `load.read`.

## Log
- 19:20 Started. Brief read; files present (notes/briefs/route_theory.md, ttn/routes.py, tests/test_routes.py, paper PDF); `uv run pytest tests/test_routes.py`: 13 passed. Machine: 128 cores, cgroup quota 122.9 CPUs, 251 GB RAM.
- 19:28 Read brief sections 1-3, 4.1 (runs/record), 4.3; paper abstract, Secs 1-3, 6.3, 7, 8, 10, App. B, C; notes/locality.md 5, 7.4-7.5, 8-11; experiments/locality/README.md; notes/saddle_dyn.md 8, 9.6.
- 19:35 Driver (experiments/route_nonstar/nsdyn.py: dense contraction, step-driven LSODA as in sdyn.py) and per-run analysis (run.py). Pilots: 2-25 s per run; non-star targets learn several patterns in one drop and plateau at rank-constrained optima carrying recombinant strings.
- 19:43 Pre-registration notes/route_nonstar.md Part I committed and pushed (504dd37). Sweep started: 9,520 runs, 120 workers (PID 4211, log /root/outputs/route-nonstar/logs/sweep.log).
- 19:55 Deviation 1 (transient segments merged into escapes), pushed.
- 20:10 Deviations 2 (secondary fits at the float64 floor) and 3 (existence check for "no plateau" predictions), pushed.
- 20:51 Sweep finished: 9,520 runs in 68 min; 9,230 converged, 211 trapped (dead routes on mps4 chains: R4, R4d, B4_mps4), 79 wall-capped (CTL1_mps6, TD_binary8 at the float64 floor); no errors.
- 20:53 First results tables pushed (7e1fb6b).
- 20:56 Draft figures (cf5bc20), note Part II, README results; found and fixed: pattern strings read as numbers by pandas (load.py), event-based "no plateau" decisions (same-drop rule), route-entry norms skipped in the runs (entries.py, deviation 4).
- 21:05 Report written.

Status: done
