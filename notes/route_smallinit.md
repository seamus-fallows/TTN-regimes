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
