# route_smallinit: the route theory against gradient-flow training from small random init (star targets)

Agent `route-smallinit`, 2026-10-13. Task: `notes/briefs/route_theory.md` sections 1-3 and 4.1. Paper
under test: `report/ttn_saddle_order_and_aligned_training_dynamics.pdf` ("the paper"). Code and CSVs:
`experiments/route_smallinit/`. Report: `agent-runs/route-smallinit/REPORT.md`.

Part I (sections 1-7) is the pre-registration, committed and pushed before the sweep. Part II (results)
and Part III (deviations) are added afterwards; Part I is not edited after the sweep starts.

# Part I: pre-registration

## 1. Question and predictions under test

A TTN is trained by gradient flow (GF) on the exact population loss L = 1/2 ||W - W*||^2 with a pattern
target W* = sum_a s_a U_a (U_a one-hot product tensors, "patterns") that satisfies the edgewise star
condition (paper Assumption 7.1), starting from a fan-in Gaussian init of scale sigma on all
parameters. Plain-word glossary:

- **amplitude** P_a = <U_a, W>: how much of pattern a the network currently outputs.
- **learned set** J(t) = {a : P_a > s_a/2}. **Learning event** of a: its last upward crossing of s_a/2.
  **Forgotten**: P_a falls below s_a/4 after an upward s_a/2 crossing.
- **plateau**: the interval between two consecutive learning events (k and k+1); duration = difference
  of the event times. Plateau 0 runs from t = 0 (nothing learned) to the first event.
- **canonical partial-target point** theta_J: a network with W = sum over a in J of s_a U_a exactly.
- **route depth** c_J(r) = |R_J(r)|, R_J(r) = intersection over a in J of St(V_ra) (St = smallest
  subtree containing the nodes where r and a differ): the number of cores that must change from zero to
  create pattern r when the patterns in J are already present (paper eq. 8.4). For J = {} it is K
  (every core).
- **pairwise cost** min over a in J of |St(V_ar)|: the cost when r must be grown from a single learned
  pattern (no splicing; the "Prim rule" of App. C, eq. C.8).
- **A_r(J)**: product of the active (already nonzero) entries that pattern r reuses, at the balanced
  theta_J (`AlignedManifold.partial_target(J, relax_from=1e-2)`), eq. C.3.
- **aligned manifold** M*: networks whose cores are nonzero only on the "block-aligned" entries
  (Sec. 7.2); on it W is exactly a sum of the target patterns, and the dynamics reduce to scalar
  equations (Theorem 7.7).

Predictions (with the paper's status): P1 (proven, Thm 8.4) theta_J is a critical point iff every
c_J(r) >= 2, its order inside M* is c*_J = min_r c_J(r), in full space at most c*_J. P2 (formal, App. C,
eq. C.6, needs Assumption C.1) a plateau whose next pattern has route depth c lasts ~ sigma^-(c-2)
(log(1/sigma)/(s_r A_r(J)) for c = 2). P3 (formal, App. C) greedy rule: next = smallest c_J(r), ties
to larger s_r A_r(J). P4 (proven, Thm 7.7 / eq. 6.15) on M* an isolated balanced route of depth c
obeys dP/dt = c A^(2/c) P^(2-2/c) (s - P) ("effective depth" c). P5 (proven, Thm 3.2) for |J| = 1 the
full-space order is min(k*, 2 l*), which is below the route depth when a spectral shortcut exists.

Questions (brief 4.1): (a) are the plateaus canonical partial-target points; (b) is the next pattern
the greedy one (vs the pairwise rule and weight order), scored conditional on the J actually reached;
(c) do plateau durations scale as sigma^-(c-2); (d) does each event show effective depth c; (e) does
training stay close to M*; (f) do the four items of Assumption C.1 hold; plus the one-step-ahead test
of the scalar route equation and a census of runs that do something else.

## 2. Targets (`experiments/route_smallinit/configs/targets.yaml`)

All satisfy the star condition (`RouteTarget.is_star`); K <= 7 cores; c*_J <= 4 for every nonempty
proper J except PS (5, the designed shortcut case; see below). Bond dimension of edge e: blocks(e) + 2,
capped at the largest rank of the cut (brief: "blocks + 2"; the cap only removes redundant channels on
edges next to a single site). Predictions for every J: `results/predictions.csv` (from `targets.py`).

| name | kind | tree, d | patterns (weights) | greedy sequence from the heaviest | what it tests |
|---|---|---|---|---|---|
| D4 | (i) | mps4, 3 | 0000 (1), 1111 (.7), 2222 (.45) | 0000 1111 2222, every depth 4 | decoupled: Cor. 7.8, every plateau ~ sigma^-2; ties at depth 4 broken by weight |
| DB4 | (i) | binary4 (K=3), 3 | 0000 (1), 1111 (.7), 2222 (.45) | depth 3 each | decoupled on a branched tree, exponent 1 |
| B4 | (ii) | mps4, 2 | 0000 (1), 0110 (.3), 1001 (.6) | 0000 0110 (c=2) 1001 (c=4) | smaller weight has the shorter route: greedy 0110, weight order 1001 |
| S5A | (iii, iv) | mps5, 3 | 00000 (1), 00201 (.4), 10101 (.75), 11200 (.55) | 00000 11200 (c=3, tie with 00201) 00201 (c=3) 10101 (c=2, pairwise 3) | tie broken by weight; splice at the last step; greedy != weight twice |
| S5B | (iii) | mps5, 3 | 00000 (1), 02201 (.75), 20111 (.55), 20200 (.4) | 00000 20200 (3) 20111 (3) 02201 (c=3, pairwise 4) | splice of depth 3 vs pairwise 4; greedy != weight at every step |
| S7 | (iii, iv) | mps7, 2 | 0000000 (1), 0001011 (.4), 0100011 (.75), 0110100 (.55) | 0000000 0110100 (4, tie) 0001011 (4) 0100011 (c=2, pairwise 3) | brief's mps7 splice; tie; greedy != weight |
| H4 | (ii, iii) | ht4 (K=7), 3 | 0000 (1), 0201 (.55), 1001 (.75), 2200 (.4) | 0000 2200 (3) 0201 (c=4, pairwise 5) 1001 (c=2, pairwise 3) | splice changes the predicted pattern (greedy 0201, pairwise and weight 1001) |
| B8 | (iii) | binary8 (K=7), 2 | 00000000 (1), 00001101 (.75), 10000001 (.4), 10000100 (.55) | 00000000 00001101 (3) 10000001 (c=4, pairwise 5) 10000100 (c=2, pairwise 3) | splice changes the predicted pattern (greedy 10000001, pairwise and weight 10000100) |
| TIE | (iv) | ht4 (K=7), 3 | 0000 (1), 1100 (.6), 2200 (.45) | 0000 1100 (3, tie) 2200 (3) | tie in depth broken by weight on a branched tree |
| PS | (v) | mps6, 2 | 000000 (1), 110001 (.8), 110010 (.8) | 000000 110010 (c=5) 110001 (c=2) | pair-star: at J = {000000} Thm 3.2 gives order 4 < route depth 5 (spectral shortcut, `notes/saddle_dyn.md` 9.6) |

Notes fixed before the sweep. (1) The brief's ht4 {0000, 0122, 1100, 2022} and binary8 {00000000,
00001111, 01011100, 10000011} have later route depths 5 and 6 on their greedy paths; they are replaced by
searched targets (`targets.py` conditions: star, every c*_J <= 4, a splice on the greedy path, greedy !=
pairwise at some step). (2) The brief's decoupled mps5 has depth 5 > 4; D4 (mps4) and DB4 (binary4) are
used instead. (3) For |J| = 1, Theorem 3.2 (`one_component_order`) gives order 2 at several
one-pattern saddles where the learned pattern is lighter than a missing one (DB4, B4, S5A, S7, H4, B8,
TIE; column `one_component_order` in `predictions.csv`): such plateaus are strict saddles (log law)
whatever the route depth. They are reached only when a lighter pattern is learned first.

## 3. Model and integration

- Network: `ttn.network.TTNSpec` with the bonds above; init `spec.init(sigma, seed)` (fan-in Gaussian,
  scale sigma, all parameters); the same seed gives the same Gaussians at every sigma.
- Loss and gradient: dense W (at most 3^5 = 243 or 2^8 = 256 entries), E = W - W* formed entrywise, so
  entries off the target support carry no cancellation; Hessian by `torch.func.hessian` (exact) as the
  LSODA Jacobian. Checked against `ttn.dynamics.Problem` (velocity 1e-20, Hessian 1e-18 absolute).
- Integrator (`driver.py`): scipy `LSODA` stepped one step at a time, rtol 1e-10, atol 1e-15; every
  recorded state is an accepted step (no dense output); crossing times are linear interpolations
  between the two bracketing steps. Local clock re-zeroed when the step falls below 1e-11 of the local
  time (exact fractions for the segment offsets, so durations of O(10) after a first event at 1e20 are
  exact). Restarts begin with a step of 1e-2. A solver that takes 3000 consecutive nonstiff steps
  with step < 1e-6 of the local time ("pinned", `notes/saddle_dyn.md` deviation 7) continues with VODE
  BDF; failed steps are retried with VODE from the last good step (at most 8).
- Stop: "done" when L < 1e-8; "trapped" when every unlearned pattern has |P_r|/s_r < 1e-35 for a decade
  of time (dead route, cf. saddle_dyn deviation 1); "unfinished_wall" after 3600 s; "failed".
- Threads: `torch.set_num_threads(1)`, BLAS pinned to one thread (`threadpool_limits(1)` and
  `OPENBLAS_NUM_THREADS=1`), one process per core (pool of 300 on the 307-CPU quota).
- Reference run per (target, sigma): the exact reduced flow on M* (`AlignedManifold.velocity/jacobian`)
  from uniform entries sigma, same driver and events (atol 1e-30).

## 4. Grid

- sigma = 10^-1.5, 10^-1.75, ..., 10^-5 (15 values, quarter decades, 3.5 decades). Pilot (seed 0,
  sigma = 1e-4, every target): 5-25 s per run, all finished.
- Seeds 0-39 for every (target, sigma): 10 x 15 x 40 = 6000 runs; 150 reference runs.
- Floor check: every seed at the 3 smallest sigma (1e-5, 10^-4.75, 10^-4.5) is run a second time with
  the library's arithmetic (gradient from `ttn.dynamics.Problem.velocity`, the norm-form loss whose
  gradient is a difference of O(1) vectors, i.e. worse roundoff): 1200 runs.
- If the budget is exceeded, the grid is cut from the small-sigma end and the cut is reported; the
  windows below are then not moved (a cell whose window cannot be completed is INCONCLUSIVE).

## 5. Per-run measurements (operational definitions beyond the brief)

- **Plateau entry** (plateau before event k >= 1): the first step after event k-1 at which every
  learned pattern (J before event k) is within 1% of its weight. None if never ("unresolved plateau").
- **Minimum-gradient point**: the step of smallest ||grad L|| between the entry and event k (between
  the two events if there is no entry).
- **Escape time**: first time after event k-1 with L < L_J - 1e-3, L_J = 1/2 sum over missing s_r^2.
- **Canonical distance**: ||W - sum_J s_a U_a|| / ||W*|| at the minimum-gradient point; also L - L_J.
- **Effective depth** of event k: `ttn.routes.effective_depth` (slope of log[(dP/dt)/(s - P)] on log P
  for P/s in [1e-4, 1e-2], depth = 2/(2 - slope)) on the amplitude of the pattern learned at event k,
  over the steps from the plateau entry to the event (primary: the plateau proper); also over all
  steps since event k-1 (`eff_depth_full`, the brief's literal wording; differs only when the learned
  pattern's shared factors are still growing after event k-1).
- **Off-target mass**: `off_target_mass` per step; its peak over the transition = max between the
  time midpoints of the plateaus before and after event k.
- **Route-entry norms** at the entry and minimum-gradient points: `AlignedManifold.align(cores, J)`
  then `route_entry_norms(aligned, J, r)` for every missing r; min, max, geometric mean over the nodes
  of R_J(r), each / sigma; alignment quality = min over edges of `align`'s quality.
- **Observed active factor** gamma_obs(r) = product of the aligned core entries at r's local types
  outside R_J(r); compared with A_r(J).
- **Separation** (C.1 items 3-4): when the event's pattern first passes 0.9 s_r, the largest P_q/s_q
  over the other unlearned patterns; "unseparated" if > 0.1.
- **One-step-ahead prediction** from the entry point: (A, the brief's) b = (P_r(entry)/A_r)^(1/c) and
  t_pred = time for eq. 6.15 to carry P from P_r(entry) to s_r/2 (exact quadrature; for small P this
  is 1/((c-2) A_r s_r b^(c-2)), resp. log(b*/b)/(A_r s_r) for c = 2); infinite if P_r(entry) <= 0
  (reported). (B, added here) the same with b = geometric mean of the route-entry norms at entry
  (P replaced by A_r b^c): tests whether the size of the dormant entries, rather than the current
  amplitude, sets the escape time. Observed = event time - entry time. One-step winner = missing
  pattern with the smallest t_pred.
- **Floor**: (i) informational bound: escape signal |dP_r/dt| over a rigorous worst-case roundoff
  bound on it (10 u x absolute-value contraction), at the entry and minimum-gradient points
  (`floor_ratio_*`). The pilot showed this bound is far too pessimistic (ratio 0.03 in a run whose
  durations agree to 5e-10 between two contraction orders and to 3e-7 with the library arithmetic), so
  it does not flag runs. (ii) **flag used**: a run at the 3 smallest sigma is floor-flagged if its
  library-arithmetic duplicate gives a different learning order, a different status, or any plateau
  duration differing by more than 1e-3 relative. Flagged runs are excluded from fits and counted.

## 6. Estimators and decision rules

Transitions are grouped into cells (target, J before, pattern learned) and scored conditional on the
J actually reached. Plateau 0 (from the origin; J = {}) is reported separately (theory: depth K, the
first pattern is not predicted).

**(a) Canonical plateaus.** A plateau (k >= 1) with an entry point is "canonical" if its canonical
distance is < 0.01. Per target: fraction canonical among plateaus with c*_J >= 2, at sigma <= 1e-3.
SUPPORTED if >= 90%; REFUTED if < 50%; else partial. Unresolved plateaus (no entry) are counted
separately.

**(b) Sequence.** For every transition with J nonempty: does the learned pattern equal the greedy /
pairwise / weight prediction? Agreement fractions per target, per cell and per sigma; the rules are
compared on the transitions where they disagree. Greedy rule SUPPORTED on a target if >= 90% of its
transitions at sigma <= 1e-3 agree; REFUTED if < 50%. Every disagreement is listed by cell.

**(c) Durations.** Per seed and cell, OLS of log(duration) on log(sigma) over the window
sigma in [1e-5, 1e-3] (9 points), using only sigmas at which that seed went through the same cell and
was not floor-flagged or trapped; a seed's exponent is fitted if it has >= 5 points spanning >= 1.5
decades. Prediction for the cell: gamma = c - 2 (route depth c of the learned pattern; theory P2);
alternatives: pairwise cost - 2; for |J| = 1, Theorem 3.2 order - 2. For c = 2 the prediction is the
log law: OLS of duration on ln(1/sigma), slope compared with 1/(s_r A_r(J)); the power-law exponent is
also reported (expected near 0). Report per cell: number of seeds, median, quartiles, fraction within
0.3 of the prediction. Decision per cell (c >= 3): SUPPORTED if |median - (c - 2)| <= 0.15 and >= 80% of
the fitted seeds are within 0.3; REFUTED if |median - (c - 2)| > 0.5 with >= 5 seeds; else
INCONCLUSIVE. For c = 2: SUPPORTED if the median log-law slope is within 25% of 1/(s_r A_r(J)) and the
median power exponent is < 0.3. Same rules for the reference runs (one "seed").

**(d) Effective depth.** Per event (k >= 1, sigma <= 1e-3): |eff_depth - c| <= 0.3 counts as a match.
SUPPORTED per cell if >= 80% match; REFUTED if the median is off by > 0.5.

**(e) Closeness to M*.** Per event: peak off-target mass over the transition (close if < 1e-2), and
canonical distance; distributions per target and sigma.

**(f) Assumption C.1.** (1) Per cell and missing pattern: slope of log(route-entry geometric mean)
against log sigma at the plateau entry (pooled OLS over seeds, sigma <= 1e-2): holds if within 0.1 of 1;
spread across nodes = ren_max/ren_min at entry (median). (2) gamma_obs/A_r(J): holds if within 5% at
sigma <= 1e-3 for >= 90% of plateau rows. (3)-(4) Fraction of events with separation > 0.1 (per target
and sigma); holds if < 10% at sigma <= 1e-3.

**One-step-ahead.** Ratio observed/predicted (A and B) per event; distribution per cell and sigma;
the scalar equation is SUPPORTED as a one-step predictor in a cell if the median ratio is within
[0.5, 2] and the predicted winner (A) equals the observed pattern in >= 90% of the events. Negative
P_r at entry is counted and reported.

**Census.** Every run is counted by status (done / trapped / unfinished / failed / error) and by
behaviour: two patterns at once (an unseparated event or an unresolved plateau between events),
forgetting, non-canonical plateau, floor-flagged; examples are described.

## 7. Status tags

Results carry: proven (statements of the paper), derived (consequences under stated assumptions),
empirical (everything measured here), conjecture.

# Part II: results

All numbers: `experiments/route_smallinit/results/` (`headline.md` is generated by `headline.py` and
lists every number quoted here; CSV columns in the experiment README). "sigma <= 1e-3" pools the 9
grid points 1e-5 ... 1e-3. Status tags: **[proven]** statement of the paper, **[empirical]** measured
here, **[conjecture]** my interpretation. Sweep: 6000 training runs (10 targets x 15 sigma x 40
seeds; 20 PS runs unfinished, section 9), 150 reference runs on M*, 1300 floor-check duplicates.

## 1. One-paragraph answer

On star-satisfying pattern targets the route theory describes training from small random init very
well along the main path, and fails in four specific, reproducible ways. **Holds [empirical]:** plateaus
sit at the canonical partial targets (98.7% within 1% of ||W*||); after the first pattern, the next
pattern is the greedy one in every transition without a depth tie (6882 of 6882 transitions at
sigma <= 1e-3), including all 747 transitions where the greedy and pairwise rules name different
patterns (the splice is real; the pairwise rule is right in none of them); per-seed duration exponents
equal c - 2 to within 0.3 in 98% of the 648 clean fits with c >= 3 (median deviation 1e-4; the
exceptions are all PS), and the log law of c = 2 holds with the predicted slope 1/(s_r A_r(J)) to 1% in
6 of 7 cells; effective depths
equal the route depth (medians 2.02, 3.02, 4.09, 5.16 for c = 2, 3, 4, 5); training stays on the
aligned manifold (peak off-target mass over a transition: median 8e-9); the scalar route equation
predicts the remaining plateau from the size of the dormant route entries (observed/predicted median
1.04). **Fails [empirical]:** (1) ties in route depth are decided by the random init, not by the weight
(the heavier pattern wins in 63% of D4 ties, 90-92% in S5A and S7, 100% in DB4 and TIE); (2) when a
lighter pattern is learned before a heavier one that can use its cores, that plateau is a strict saddle
(log-law escape, exponent ~0.1 instead of c - 2 = 1) and the heavier pattern takes over by *erasing*
learned patterns, which are relearned later ("forgetting": 20-42% of the runs of S5A, S5B, S7, PS); for
|J| = 1 this is Theorem 3.2's order 2 (and for PS its order 4: plateau ~ sigma^-2, not sigma^-3), for
|J| = 2 it is a full-space shortcut below c*_J that the paper allows (Remark 8.5) but does not predict;
(3) Assumption C.1 item 1 (dormant entries of order sigma at plateau entry) fails right after the first
escape when the missing route shares cores with the first pattern: in B4 they scale as sigma^0.70, and
the log-law slope is 0.70 x the prediction (same in the exact reduced flow, so this is a defect of the
formal estimate C.6, not of training); (4) dead routes: in D4 13/40 and in B4 5/40 seeds (the same seeds
at every sigma) the last pattern is never learned: one dormant entry of its route decays to ~1e-14 sigma
and the run ends "trapped". The brief's one-step-ahead predictor (from the amplitude P_r at plateau
entry) fails (median ratio 2.7, quartiles 0.9 and 579) because P_r at entry still carries transients
that decay by up to 10 orders of magnitude before the escape.

## 2. (a) Are the plateaus canonical partial-target points?

**Yes [empirical].** At the minimum-gradient point of every resolved plateau (k >= 1), the distance
||W - sum_J s_a U_a|| / ||W*|| has median 8e-9 (quartiles 9e-11, 5e-6); 98.7% of 8382 plateaus at
sigma <= 1e-3 are within 0.01 (registered rule: SUPPORTED, >= 90%). The fraction falls at large sigma
(0.93 at 1e-2, 0.80 at 10^-1.5) because short plateaus no longer settle. The exceptions are three kinds:
- **S5A after a mixed first escape** (29 plateaus, distance 0.21, L - L_J = -0.07): in 10-25% of the S5A
  seeds the escape from the origin produces a mixture of patterns (a rank-one "best fit" of W*), and the
  network passes 11200 almost immediately (unresolved plateau, exponent ~0.17 instead of 1).
- **PS pair** (62 plateaus, 0.08-0.14): 110001 and 110010 differ only in the last two sites and are learned
  together (two at once); the plateau between them does not exist (duration exponent 0).
- **B4 {0000} -> 0110** (14, distance 0.0125): a c = 2 plateau so short that the learned pattern has not
  fully converged when 0110 starts to rise (borderline).
Unresolved plateaus (never reach the entry criterion): PS 293, S5A 89, S5B 27 (all two-at-once).

## 3. (b) Is the next pattern the one the greedy rule predicts?

Scored per transition with J = the learned set actually reached (J nonempty), sigma <= 1e-3:

| rule | all transitions (n = 8791) | where the rules disagree (n = 3668) | greedy vs pairwise disagree (n = 747: B8, H4, S5A) |
|---|---|---|---|
| greedy (route depth, then s_r A_r(J)) | 0.960 | 0.985 | 1.000 |
| pairwise (Prim rule, eq. C.8) | 0.875 | 0.782 | 0.000 |
| weight order | 0.549 | 0.000 | - |

- **Without a tie in route depth the greedy rule is never wrong [empirical]:** 6882/6882 transitions
  at sigma <= 1e-3 (6007/6007 clean ones), and 0.996-1.000 at every sigma up to 10^-1.5; this includes
  the "contaminated" events (intervals with forgetting). The splice cases where the greedy and pairwise
  rules name different patterns (B8 at {00000000, 00001101}: greedy 10000001 (depth 4, splice), pairwise
  10000100 (5); H4 at {0000, 2200}: greedy 0201 (4), pairwise 1001 (5); S5A at {00000, 10101}) are all
  decided for the greedy rule. Registered decision: SUPPORTED (>= 90%).
- **Ties are decided by the init, not by the weight [empirical; failure of the tie-break of App. C]:**
  1909 tied transitions, greedy (= heavier) 0.82. Per cell: D4 after 0000: 0.63 (n = 216), after 1111:
  0.64, after 2222: 0.00 (18); S5A after 00000: 0.92; S7 after 0000000: 0.91; S5B after 02201: 0.67;
  DB4 and TIE: 1.00. The winner of a tie is a property of the seed: for 99.5% of (target, J, seed) the
  same pattern wins at every sigma. [conjecture] With equal depth c the escape time is
  ~1/((c-2) s_r A_r b_r^(c-2)), and b_r (the size of r's dormant entries, a random O(sigma) number
  per seed) varies by more than the weight ratio; in DB4 and TIE the dormant entries of the two tied
  patterns are coupled (they share nodes and bond channels), which makes the heavier one win always.
- The tie-break coefficient matters once: at S5A {00000, 00201} the tied 10101 (weight 0.75) and 11200
  (0.55) have s_r A_r(J) = 0.75 x A and 0.55 x A' with A' > A at the balanced partial target, so the
  App. C rule picks 11200 while `RouteTarget.greedy_next` (weight only) picks 10101; training picked
  11200 in 27/27 transitions [empirical].
- Weight order is wrong wherever it differs from the greedy rule: the route depth, not the weight, sets
  the order (B4: 0110 with weight 0.3 is learned before 1001 with weight 0.6 in 100%).
- The first pattern learned (first upward crossing; not predicted by the theory) is the heaviest in
  95-100% of the runs of B4, B8, DB4, H4, TIE, but only in 60-65% for D4, S5A, S5B, S7 and 40% for PS
  (the pair 110001 + 110010 escapes first in 24/40 seeds). The path is a property of the seed: 359 of
  the 400 (target, seed) pairs have the same crossing history at all 15 sigma.

## 4. (c) Do plateau durations scale as sigma^-(c-2)?

Per-seed OLS of log duration on log sigma, sigma in [1e-5, 1e-3] (9 points), floor-flagged and
trapped runs excluded. Three bases (`exponents.csv`): `event` (registered: every learning event),
`event_clean` (events whose interval since the previous event has no other crossing and no forgetting),
`pass` (every upward crossing; this exposes the transient plateaus of forgetting runs).

| route depth c | clean later-plateau fits | exponent median [quartiles] | within 0.3 of c - 2 |
|---|---|---|---|
| 2 (log law) | 174 | 0.116 [0.108, 0.122] | log slope / 1/(s A) = 1.00 in 6 cells; B4: 0.70 |
| 3 | 399 | 1.000 [1.000, 1.000] | 1.00 |
| 4 | 227 | 2.000 [2.000, 2.000] | 1.00 |
| 5 (PS only) | 22 | 3.30 [3.00, 3.97] | 0.50 |
| first escape, c = K | 338 | K - 2 exactly (median deviation 0) | 1.00 |

- **Registered decisions** (`cells.csv`, cells with fitted seeds, J nonempty): basis `event` (as
  registered) 37 SUPPORTED, 7 INCONCLUSIVE, 3 NOT SUPPORTED, 3 REFUTED; basis `event_clean` 42
  SUPPORTED, 3 NOT SUPPORTED (B4 {0000}->0110 log slope; PS pair, no plateau), 1 INCONCLUSIVE (PS
  {pair}->000000). The 3 REFUTED and most INCONCLUSIVE cells of the registered basis are artifacts of
  the learning-event definition in runs with forgetting: the "plateau" of a pattern that is learned,
  erased and relearned is measured from an event whose interval contains other crossings (e.g. S5A
  "{00201} -> 00000" with exponent 3 = the first-escape exponent K - 2, measured from t = 0).
- **The route-depth law holds [empirical]:** 98% of the 648 clean fits with c >= 3 are within 0.3 of
  c - 2. In every splice cell (route depth below the pairwise cost: B8 {0,1}->2, H4 {0,3}->1, S5B
  {0,2,3}->1, S7 {0,1,2}->3; 125 fits) the exponent is c - 2 (2.000, 2.000, 1.003, 1.001) and none is at
  pairwise - 2. The c = 2 splices (B8, H4, S5A, S5B, S7, last step) follow the log law with exactly the
  predicted slope. Registered per-cell decisions (`cells.csv`, basis event_clean): every cell with
  c >= 3 SUPPORTED except PS {110001,110010}->000000 (INCONCLUSIVE, below). The reference runs on M*
  have exponents c - 2 to 1e-4 in every cell.
- **The log law (c = 2) has the right form; its slope fails once [empirical]:** B4 {0000}->0110: slope
  2.33 per e-fold of 1/sigma against 1/(s A) = 3.33 (ratio 0.70, every seed and the reference run). Cause
  (item 6, C.1 item 1): the dormant entries of 0110 scale as sigma^0.70 at plateau entry, because they
  grow together with the shared cores during the first escape; 0.70 = 1 - s_0110/s_0000 [conjecture: the
  growth factor of a dormant entry during a Saxe escape of the shared factors].
- **Lighter-first plateaus are strict saddles [empirical; Theorem 3.2 confirmed where it applies]:**
  pass basis. S5A history "+10101 +00201 -10101 +00000 ..." (13/40 seeds): the plateau at
  {10101, 00201} ends after a log time (exponent 0.105, 13 seeds) although c*_J = 3; S5B "+20111
  +20200 -20111 +00000 ..." (6 seeds): exponent 0.110; in both the escaping 00000 erases a learned
  pattern. For |J| = 1 this is Theorem 3.2: the one-component order is 2 whenever the learned pattern is
  lighter than a missing one it can turn into (`one_component_order` = 2 in `predictions.csv`); here the
  same happens at |J| = 2, below the route depth: a full-space shortcut of the kind Remark 8.5 allows.
  Pass basis, all one-pattern plateaus where Theorem 3.2 gives order 2 < route depth 3 (S5A {00201},
  S5B {20200}): 17 seed fits, exponent 0.107 [0.096, 0.109], all at the log law (order 2) and none at
  c - 2 = 1. **Theorem 3.2 is confirmed; the route depth is refuted there.**
- **The spectral shortcut of the pair-star is confirmed [empirical; Theorem 3.2]:** PS seeds that learn
  000000 first (16/40, the same 16 at every sigma; saddle_dyn had 16/40 too) stay on {000000} for
  ~sigma^-2: per-seed exponent 2.000 [2.000, 2.000] (15 fits), the order-4 law of Theorem 3.2, not the
  route depth 5 (sigma^-3). The escape brings both 110001 and 110010 at once and erases 000000, which is
  relearned after a depth-5 plateau: exponent 3.00 [3.00, 3.50]. Figure `fig_shortcut_PS`.
- **PS 000000 after the pair (INCONCLUSIVE cell) [empirical]:** in the seeds that learn the pair first,
  the depth-5 plateau has exponent 3.0 (+110010 first, 12 seeds) or 3.9 (+110001 first, 11 seeds,
  quartiles 3.3-4.0) although its growth shows effective depth 5.16 like the others. The dormant
  route of 000000 is very unbalanced at entry (max/min entry ratio 125, against 2-4 elsewhere).
  [conjecture] The unbalanced route behaves as if its smallest entries were of order sigma^(4/3); the
  asymmetry between the two pair members is not understood.

## 5. (d) Effective depth

`ttn.routes.effective_depth` on the learned pattern's amplitude, from the plateau entry to the event,
sigma <= 1e-3: c = 2: 2.02 [2.02, 2.02], 90% within 0.3 (10% of events have too few points in the
window); c = 3: 3.02 [3.02, 3.04], 96%; c = 4: 4.09 [4.00, 4.15], 99.5%; c = 5: 5.16 [5.16, 5.16],
100%; first escape (c = K = 3 ... 7): K to within 0.03 (median deviation -0.001). **SUPPORTED
[empirical]** in every cell (>= 80% within 0.3). The few mismatches are relearned patterns after
forgetting (S5B 20111: median 2.8-2.9 in the forgetting histories, 3.06 otherwise) and S7
{0001011, 0100011} -> 0000000 (4.26). The estimator's systematic +0.02/+0.09/+0.16 at c = 3/4/5 is the
finite window (the reference flow on M* gives the same values).

## 6. (e) Does training stay close to the aligned manifold?

**Yes, except during the escapes that the theory does not cover [empirical].** Peak off-target fraction
of ||W||^2 over a transition (k >= 1, sigma <= 1e-3): median 8e-9 (quartiles 2e-10, 4e-7); 93.8%
below 1e-2. It scales as ~sigma^2 (median 3e-11 at 1e-5, 1.6e-4 at 10^-1.5). The large values (up to
0.5) come from PS (the pair escape), S5A/S5B/S7 takeovers (0.14-0.38) and the mixed first escapes.
Alignment quality of `align` at the plateau minimum-gradient points: 1.000 (all quartiles). The
observed active factors equal A_r(J) at the balanced theta_J (ratio 1.000; within 5% in 95% of the
plateau rows). [empirical] Caveat: W can be canonical while the network is not on M* in parameter
space (e.g. a block represented by two redundant channels); `align`'s quality does not see that.
The lighter-first takeovers suggest that this happens (section 4) [conjecture].

## 7. (f) Assumption C.1

1. **Dormant entries of order sigma at plateau entry: holds in 36 of 39 cells [empirical].** Per-seed
   slope of log(route-entry geometric mean) on log sigma (sigma <= 1e-2; cells with >= 3 seed fits):
   median within 0.1 of 1 in 36 cells (in most exactly 1.00). Exceptions: B4 {0000} -> 0110 (0.70; the
   entries grow with the shared cores during the first escape), PS {pair} -> 000000 (0.88, entries
   spread 125x across the route's nodes), PS pair internal (0: two at once). Geometric mean / sigma:
   0.78 [0.57, 1.34]; spread max/min over the route's nodes: 2.3 [1.4, 4.0] (the entries are of common
   order but not balanced).
2. **Active factors independent of sigma: holds [empirical]:** gamma_obs / A_r(J) = 1.000 (within 5%
   in 95% of rows).
3.-4. **Separated events: holds except PS [empirical]:** at sigma <= 1e-3, 1.7% of the events have
   another unlearned pattern above 10% of its weight when the learned one passes 90% (PS 21%, TIE
   1.7%, all others <= 0.3%); the fraction grows slowly with sigma (1.2% at 1e-5, 4.1% at 10^-1.5).
   Item 4 fails in the takeover transitions (a learned pattern is lost when the next arrives), which the
   separation test does not see because the erased pattern was already learned.
Signs: P_r <= 0 at plateau entry for 5.4% of the learned patterns (5.8% of all missing patterns). Such
routes still escape (the spare channels let them change sign) but later: at c >= 3 their
observed/predicted time (predictor B) is 2.1 [1.7, 2.9] (n = 473) against 1.1 [1.0, 1.4] for P_r > 0
[empirical].

## 8. One-step-ahead test of the scalar route equation

From the plateau entry (learned amplitudes within 1%), with c and A_r(J): **(A, brief)** b = (P_r /
A_r)^(1/c): observed/predicted time 2.66 [0.92, 583] (n = 7910), within [0.5, 2] in 44%, predicted
winner right in 82%: **fails [empirical]**. At entry, P_r still contains the remnants of the previous
escape (for example in B8 the other patterns sit at 1-10% of their weight right after the first escape)
which then decay by up to 10 orders of magnitude before the growing mode takes over
(`fig_example_run`). **(B, added)** b = geometric mean of the dormant route entries at entry: ratio
1.04 [0.93, 1.39] (n = 8382), within [0.5, 2] in 88%, winner right in 98%: **SUPPORTED for c <= 4**
(c = 2: 0.90, c = 3: 1.04, c = 4: 1.49); fails at c = 5 (PS, 350) where the route is unbalanced.
[conjecture] What decides the escape is the size of the dormant entries, as in eq. C.4-C.6, not the
current amplitude.

## 9. Census of runs that do something else

Training runs: 6000. Status: done 5710, trapped 270 (B4 75 = 5 seeds x 15 sigma; D4 195 = 13 seeds x
15 sigma), unfinished 20 (PS, 1-hour cap; deviation 7), failed 0. Behaviour tags (a run can have
several; `runs.csv` `behaviour`, `history`):
- **clean** 3691 (62%);
- **first escape mixed** (another pattern above 10% of its weight when the first passes 90%; the theory
  says nothing about the first escape) 1412: PS 555 (the pair), H4 374, B8 174, S5B 145, S5A 107;
- **two patterns at once later** 813: PS pair 497, S5A 172, TIE 69, S5B 51;
- **forgetting** 816: PS 251 (the pair erases 000000), S5A 195, S5B 175, S7 120 (lighter-first runs:
  the heavy all-zero pattern erases learned light patterns), H4 30, TIE 22, DB4 11, B8 10;
- **non-canonical plateau** 556 (mostly at large sigma; at sigma <= 1e-3 the 110 listed in section 2);
- **trapped** 270: after learning two patterns, the third route dies: one of its dormant entries (the
  smallest at entry: 0.03-0.12 sigma, against 0.2-0.6 sigma in the seeds that escape) decays to ~1e-14
  sigma and the amplitude to ~1e-55. The routes that die (D4: 1111/2222/0000 as the last pattern; B4:
  1001) span the whole tree, including the end cores whose bond is capped at d (no spare channel).
  [conjecture] Without a spare channel at the route's end the dormant entry cannot rotate away from the
  direction in which it is suppressed; it is a seed property (same seeds at all sigma);
- **unfinished** 20 (PS, sigma <= 10^-3.5): all had learned the pair and were on the depth-5 plateau
  before 000000 after > 1 million solver steps; in 9 of them the 000000 amplitude had fallen to
  ~1e-33 (a dying route at the float64 floor, as in the trapped runs), in the others the solver crawled;
- **floor-flagged** 11 (section 10).

## 10. Float64 floor and accuracy

Duplicates with another contraction order (all seeds at the 3 smallest sigma: 1200 runs, 1171 where
both finished) differ in event durations by 4e-8 (median; max 2.2e-2); library-arithmetic duplicates
(seeds 0-9 at 1e-5: 100 runs, 43 finished within their 30-minute cap) by 1.3e-7 (max 1.2e-3). 11 runs
exceed 1e-3 (PS 5, S7 5, B4 1; all at sigma <= 10^-4.75), never with a different order or outcome; they
are excluded from the fits. The worst-case roundoff bound (`floor_ratio_*`) is far too pessimistic to
use (it flags plateaus whose durations agree to 1e-9 between arithmetics). LSODA fell back to VODE in
16 runs (17 failed steps).

# Part III: deviations from the pre-registration (added after the sweep started)

1. **Trap rule (19:50, before any result was looked at).** The first sweep launch was stopped after
   one minute: the trap rule ("every unlearned |P_r|/s_r < 1e-35 for a decade") fired on the very
   first plateau of the K = 7 targets at sigma = 1e-5, where every amplitude starts at ~sigma^7 ~
   1e-38 (nothing is dead there). The rule now applies only once at least one pattern is learned.
2. **Orphaned workers; full restart (20:09).** Killing that first launch left its 300 worker processes
   alive (orphaned); they kept running the old code (with the trap-rule bug) for 20 minutes, competed
   with the second launch for the CPUs (load ~600 on a 307-CPU quota) and wrote results into the same
   output folder. Everything from both launches was set aside (`/root/outputs/route-smallinit/
   runs_try1_trapbug/`, `runs_try2_contaminated/`) and the whole grid was restarted from scratch with
   the code of item 1. Not outcome-selective: nothing from the set-aside runs is used.
3. **Floor check (20:09).** The registered check duplicates every run at the 3 smallest sigma with the
   library's norm-form gradient. For the K = 7 targets at sigma = 1e-5 these duplicates take 30-60
   minutes each (slower gradient and ~3x more solver steps because of the larger roundoff), i.e. about
   two machine-hours for the check alone. Replaced by: (a) every seed at the 3 smallest sigma is
   duplicated with a different contraction order of the same dense model (`arith="reorder"`: operands
   reversed, greedy path; a different roundoff realization at the same cost); (b) the library-arithmetic
   duplicate is kept for seeds 0-9 at sigma = 1e-5 only (wall cap 1800 s). Flag rule unchanged (either
   duplicate gives a different order or status, or a duration > 1 that differs by more than 1e-3
   relative). Main runs are scheduled before all floor checks.
4. **Final-plateau diagnostics (added during the first launches).** Runs that end with patterns never
   learned (trapped, unfinished) get `plateaus.csv` rows for their last plateau (`final = True`,
   `P_end`), so that the trapped routes can be described. Additive; the restarted sweep has it for
   every run.
5. **Run behaviour tags (analysis only).** "two_at_once" is split into `first_escape_mixed` (at the
   first escape from the origin another pattern is above 10% of its weight when the first passes 90%;
   the theory makes no prediction there) and `two_at_once` (the same at a later transition, or a plateau
   that never reaches its entry point).
6. **Exponents and cells also for event 0** (the escape from the origin, depth K, Saxe exponent K - 2):
   reported alongside, not used for the route-theory decisions.
7. **PS runs unfinished after 1 hour (21:13).** 20 PS runs (sigma <= 10^-3.5; all had learned the
   pair 110001 + 110010 and were still on the depth-5 plateau before 000000) hit the 3600 s wall. They
   were rerun with a 10800 s wall (`configs/sweep_ps_long.yaml`, same code and seeds); the 1-hour
   records are kept in `/root/outputs/route-smallinit/runs_unfinished_1h/`. Runs that still do not
   finish are reported as unfinished with their 1-hour record. Their reorder floor-check duplicates
   (sigma <= 10^-4.5) are not rerun.
