# route_nonstar: training on pattern targets that break the star condition

Agent `route-nonstar`, branch `agent/route-nonstar`. Experiment folder: `experiments/route_nonstar/`.
Raw outputs: `/root/outputs/route-nonstar/` (not in git). Brief: `notes/briefs/route_theory.md`
sections 1-3 and 4.3. Paper under test: `report/ttn_saddle_order_and_aligned_training_dynamics.pdf`.

**Part I (sections 1-7) is the pre-registration, written and pushed before any sweep.** It is not
edited after the sweep starts; deviations go to section 8 with the reason. Results are in Part II.

## 1. Question and what is tested

A pattern target is W* = sum_a s_a U_a, each U_a a one-hot "pattern" (a string of site values).
The paper proves its later-saddle results only under the **edgewise star condition** (Assumption
7.1): cut the tree at any edge; the target strings restricted to the two sides form a bipartite
graph; every connected piece of that graph must be a star. A target fails it when three patterns
a, b, c satisfy "b agrees with a on one side of a cut and with c on the other side". Then a network
that represents a and c with one bond channel per block also produces the **recombinant** string
(a's side + c's side), which is not in the target (e.g. 000, 100, 001 on a 3-node chain: 100 and
001 recombine into 101).

Question: what does gradient flow from small random init do on such targets, and which predictions
survive? Predictions under test (applied although the theory does not cover these targets):

- **P1 (plateaus are canonical partial-target points).** Each plateau is W = sum over J of s_a U_a
  for the learned set J. Alternative from `notes/locality.md` section 5: plateaus are
  rank-constrained optima (best approximation with the current bond ranks), which contain
  recombinants.
- **P2 (sequence).** The next pattern learned from J is predicted by (A) the paper's greedy rule
  with route depth c_J(r) = |intersection over a in J of St(V_ra)| (applied anyway; the
  intersection can be empty: depth 0 = "produced for free as a recombinant"), (B) the repo's
  numerical route rule (`experiments/locality/rules.route_rule` at the measured plateau point: the
  smallest connected core set C whose local residual R_C is non-zero; the predicted next string is
  the largest entry of R_C), (C) the pairwise rule (smallest min over a in J of |St(V_ar)|, Prim,
  eq. C.8). Plain weight order is reported as a baseline.
- **P3 (durations).** A plateau whose escape has order p lasts ~ sigma^-(p - 2) (log(1/sigma) for
  p = 2, none for p <= 1). Predicted p: (A) c*_J = min over missing r of c_J(r); (B) the route
  rule's k*; (C) the minimal pairwise cost. Per-seed exponents of duration against sigma.
- **P4 (effective depth).** The escape follows the scalar route equation with depth d = p:
  `ttn.routes.effective_depth`.
- **P5 (escape type).** Order 2 = strict saddle (negative Hessian eigenvalue independent of sigma,
  log law); order > 2 = degenerate (smallest eigenvalue -> 0 as sigma^(p-2)). Where it can be
  worked out by hand, p = min(direct degree, 2 x spectral level) (Theorem 3.2 for one learned
  pattern, proven without the star condition; Appendix B conjecture otherwise).

`experiments/locality` already trained the nested teachers TA, TB, N4, TC, TD (8 seeds, chi = 2,
random site bases) and found rank-constrained optima and the route rule; it did not measure what
the plateaus are (distance to partial targets, off-target strings), did not score the paper's
route depth or the pairwise rule, and had no spare bond channels. Its TB = our nested chain.

## 2. Cells (targets)

`experiments/route_nonstar/make_targets.py` writes `configs/targets.yaml` (20 targets, d = 2):

- **recombinant family** (path of 3 and 4 nodes): q = 0..0 plus the two single-end edits 10..0 and
  0..01 (weights 1, 0.6, 0.4), without the recombinant 10..01 (R3, R4), with it at the product
  weight 0.6 x 0.4 = 0.24 (R3p, R4p: W* is then a product tensor, rank 1 across every cut) and at a
  different weight 0.5 (R3d, R4d);
- **nested chain** 0000, 1100, 1110, 1111 (1, 0.6, 0.35, 0.2; = locality's TB) on mps4, binary4, ht4;
- **two separate local edits** (TD-like): mps6 {000000, 000110, 110000} and binary8
  {00000000, 11000000, 00001100} (1, 0.6, 0.35);
- **random** d = 2 targets: the first three non-star targets on mps6 and on binary8 from a seeded
  generator (3-5 patterns, largest weight 1, others uniform in [0.3, 0.9]) with paper greedy depths
  <= 5, numerical route-rule k* <= 5 and teacher bond ranks <= 3;
- **star-satisfying controls**, same code: the first star target of the same generator on mps6 and
  on binary8 (CTL1_*), plus B4 on mps4 (the locality control).

Every non-star target lists its violations (`RouteTarget.star_violations`: edge, patterns of the
non-star block). Predictions computed before the runs:

| target | family | tree (K) | patterns (weights) | star violations (edge: patterns) | paper greedy sequence from the largest weight: next (depth c_J / pairwise) | numerical route-rule sequence (plateau loss: k*) |
|---|---|---|---|---|---|---|
| R3 | recombinant | mps3 (3) | 000 (1), 100 (0.6), 001 (0.4) | e0: 000/100/001; e1: 000/100/001 | 100 (1/1) -> 001 (1/1) | 0.01945: 3 |
| R3p | recombinant | mps3 (3) | 000 (1), 100 (0.6), 001 (0.4), 101 (0.24) | e0: 000/100/001/101; e1: 000/100/001/101 | 100 (1/1) -> 001 (1/1) -> 101 (0/1) | one step (W* rank 1) |
| R3d | recombinant | mps3 (3) | 000 (1), 100 (0.6), 001 (0.4), 101 (0.5) | e0: 000/100/001/101; e1: 000/100/001/101 | 100 (1/1) -> 101 (1/1) -> 001 (0/1) | 0.01953: 3 |
| R4 | recombinant | mps4 (4) | 0000 (1), 1000 (0.6), 0001 (0.4) | e0, e1, e2: 0000/1000/0001 | 1000 (1/1) -> 0001 (1/1) | 0.01945: 4 |
| R4p | recombinant | mps4 (4) | 0000 (1), 1000 (0.6), 0001 (0.4), 1001 (0.24) | e0, e1, e2: all four | 1000 (1/1) -> 0001 (1/1) -> 1001 (0/1) | one step (W* rank 1) |
| R4d | recombinant | mps4 (4) | 0000 (1), 1000 (0.6), 0001 (0.4), 1001 (0.5) | e0, e1, e2: all four | 1000 (1/1) -> 1001 (1/1) -> 0001 (0/1) | 0.01953: 4 |
| NC_mps4 | nested | mps4 (4) | 0000 (1), 1100 (0.6), 1110 (0.35), 1111 (0.2) | e1, e2: all four | 1100 (2/2) -> 1110 (1/1) -> 1111 (1/1) | 0.2612: 2 -> 0.06376: 2 -> 0.01831: 2 |
| NC_binary4 | nested | binary4 (3) | as NC_mps4 | e0, e1: all four | 1100 (1/1) -> 1110 (1/1) -> 1111 (1/1) | 0.05775: 3 |
| NC_ht4 | nested | ht4 (7) | as NC_mps4 | e2, e3, e4, e5: all four | 1100 (3/3) -> 1110 (1/1) -> 1111 (1/1) | 0.2612: 3 -> 0.06376: 3 -> 0.05775: 3 |
| TD_mps6 | two_edits | mps6 (6) | 000000 (1), 000110 (0.6), 110000 (0.35) | e1, e2: all three | 000110 (2/2) -> 110000 (2/2) | 0.2412: 2 -> 0.06125: 2 -> 0.01518: 3 |
| TD_binary8 | two_edits | binary8 (7) | 00000000 (1), 11000000 (0.6), 00001100 (0.35) | e0, e2, e3, e5: all three | 11000000 (1/1) -> 00001100 (1/1) | 0.01518: 5 |
| B4_mps4 | control | mps4 (4) | 0000 (1), 0110 (0.3), 1001 (0.6) | none (star) | 0110 (2/2) -> 1001 (4/4) | 0.225: 2 -> 0.18: 4 |
| RND1_mps6 | random | mps6 (6) | 110101 (1), 010110 (0.76), 010100 (0.5), 101110 (0.45) | e2, e3: 010110/010100/101110 | 101110 (5/5) -> 010110 (3/3) -> 010100 (1/1) | 0.5151: 5 -> 0.4138: 3 -> 0.02587: 3 |
| RND2_mps6 | random | mps6 (6) | 000101 (1), 011101 (0.41), 011100 (0.37) | e2, e3, e4: all three | 011101 (2/2) -> 011100 (1/1) | 0.1525: 2 -> 0.05752: 4 |
| RND3_mps6 | random | mps6 (6) | 110100 (1), 101100 (0.78), 000010 (0.4), 101010 (0.39) | e2: all four | 101100 (2/2) -> 101010 (2/2) -> 000010 (3/3) | 0.4602: 2 -> 0.1561: 2 -> 0.1256: 2 -> 0.08: 3 |
| CTL1_mps6 | control | mps6 (6) | 010101 (1), 100011 (0.85), 100100 (0.76) | none (star) | 100011 (5/5) -> 100100 (3/3) | 0.65: 5 -> 0.2888: 3 |
| RND1_binary8 | random | binary8 (7) | 01010101 (1), 00010110 (0.71), 01000101 (0.62) | e1: all three | 01000101 (1/1) -> 00010110 (5/5) | 0.2521: 5 -> 0.05444: 2 |
| RND2_binary8 | random | binary8 (7) | 00111001 (1), 00111101 (0.86), 10001001 (0.4), 10000001 (0.38) | e2, e3, e5: all four | 00111101 (1/1) -> 10001001 (3/3) -> 10000001 (1/1) | 0.1522: 3 -> 0.1002: 4 |
| RND3_binary8 | random | binary8 (7) | 01111001 (1), 11001001 (0.81), 11001000 (0.65) | e2, e4, e5: all three | 11001001 (3/3) -> 11001000 (1/1) | 0.5: 3 -> 0.1142: 4 |
| CTL1_binary8 | control | binary8 (7) | 00011010 (1), 10011000 (0.67), 10111001 (0.45), 11001000 (0.41) | none (star) | 10011000 (5/5) -> 11001000 (3/3) -> 10111001 (5/5) | 0.4098: 5 -> 0.1853: 3 -> 0.1012: 5 |

(The route-rule sequence is `rules.route_sequence`: rank-constrained optimum after the first
escape, route rule there, raise the ranks on the minimal route, repeat; its first level is the
optimum with every bond rank 1. Trees: mps = chain with one site per core; binary4 = two leaves
with two sites each under a site-less root; binary8 = four two-site leaves, two internal nodes, a
root; ht4 = four one-site leaves, two internal nodes, a root.)

**Registered by-hand orders (P5).** (i) Recombinant family without the product weight (R3, R4, R3d,
R4d): pilots show the first escape learns a product (rank-1) tensor that already passes s_a/2 on
q and both edits and carries the recombinant. At that rank-1 optimum W1 = c u (x) e_0 ... (x) w the
residual has support only on strings whose interior sites are 0, and stationarity of W1 makes every
contraction of the residual with the active vectors at the end sites vanish, so a connected route
must contain both end nodes: direct degree d* = K (the whole chain). Pairs of single-node atoms
(rotations of u or w inside the active channel) are tangent to the rank-1 manifold, where W1 is a
minimum, so they give no instability at level 1; atoms of cost 2 to K - 1 that touch the interior
use a non-zero interior site value and cannot pair into the residual, and an end-site atom paired
with a cost-(K-1) atom has level K - 1 >= K/2. Prediction: order p = K, exponent K - 2 (R3, R3d: 1;
R4, R4d: 2), effective depth K, degenerate (not strict) saddle. (ii) R3p, R4p: W* is a product
tensor: no plateau after the first escape. (iii) A plateau that is a canonical one-pattern point
s_j U_j (only j learned, every other k_r >= 2) has the proven order min(k*, 2 l*) of Theorem 3.2
(`RouteTarget.one_component_order(j)`): exponent order - 2. Other plateaus: no hand computation;
predictors (A)-(C) only.

## 3. Runs and per-run measurements

- Model (brief section 3): whitened inputs, exact population loss, standard-basis pattern target,
  fan-in Gaussian init `spec.init(sigma, seed)` on all parameters, bond dimension on edge e = (rank
  of W* across e) + 2, capped at the largest rank the cut allows. float64, dense contraction
  (`ttn.network.to_dense`; at most 2^8 entries), gradient and exact Hessian by torch.func, one
  thread per process (torch, OpenMP, OpenBLAS, MKL, threadpoolctl).
- Integration (`nsdyn.integrate`): scipy `LSODA` class stepped directly, exact Hessian as Jacobian,
  rtol 1e-10, atol 1e-14, t_max 1e22, wall cap 1800 s per run; every recorded state is a solver step
  (no dense output); local clock re-zeroed when the step falls below 1e-11 of the local time;
  restarts with first step 0.01; failed steps retried (at most 5). Times are kept as offset + local
  time and analysed in long double relative to T0 = the first step where some P_a > 0.05 s_a
  (absolute times reach 1e17 on K = 7 trees, where float64 spacing is 16). Plain loss (no reference
  point); floor checked per plateau (below).
- Stop rules: **converged** L < 1e-10 x 1/2 ||W*||^2; **trapped** once some pattern is learned, the
  largest gradient entry < 1e-14 at 20 consecutive checks log-spaced by 10^0.1 in t with t > 1e3
  (two decades: the remaining routes sit at roundoff); **wall** (1800 s) and **t_max** =
  unfinished; **failed** after 5 failed steps.
- Grids: log10 sigma = -1, -1.25, ..., -4 (13 values) for K <= 4 (mps3, mps4, binary4) and -1, ...,
  -3.5 (11 values) for K = 6, 7 (mps6, binary8, ht4); seeds 0-39 at every (target, sigma) (the same
  seed gives the same Gaussians up to the factor sigma). 9,520 runs. Pilot timing: 2-25 s per run.
- **Fit window**: log10 sigma <= -1.5 (K <= 4: 11 values, 2.5 decades; K = 6, 7: 9 values, 2
  decades); runs that are floor-flagged at the plateau in question, trapped, unfinished or failed
  are excluded from fits and counted.

Per run (`run.py`, all on solver steps):

- amplitudes P_a = <U_a, W>, learned set J(t) = {a : P_a > s_a/2}; every crossing of s_a/2 is
  recorded; **learning event** of a = its last upward crossing (if P_a ends above s_a/2);
  **forgotten** = falls below s_a/4 after an upward crossing;
- **loss drops and plateaus**: `ttn.diagnostics.find_plateaus(t, L, min_drop=0.02)` (linear time;
  a drop must lower L by >= 2 %); segment k = the interval between drop k and drop k + 1 (segment 0
  = from init to the first drop; the last segment after the final drop of a converged run is not
  a plateau); "resolved" if `find_plateaus` reports a plateau inside it (separation ratio > 3);
  **duration** of segment k = t_half(drop k+1) - t_half(drop k);
- at each segment's **minimum-gradient point** (smallest |grad| inside the segment; for later
  segments of every run): loss, P, J_amp (= J(t) there), **nearest partial target** J_near =
  argmin over all subsets J of ||W - sum_J s_a U_a|| / ||W*|| and that distance d_near, the
  distance d_amp to the point of J_amp, **canonical** iff d_near < 0.03, off-target mass
  (`off_target_mass` definition, computed on the dense W), the **off-target strings** with
  |W[x]| >= 0.01 (up to 8, largest first) each tagged as a recombinant "e<edge>:a|b" when x equals
  target a below edge e and target b above it; Hessian eigenvalues (lambda_min, lambda_2, number
  below -1e-9); |grad| and its roundoff bound 10 u x grad of (|E| + |W| + |W*|) . W(|theta|)
  (**floor flag** iff |grad| < 100 x bound); the route rule `rules.route_rule(tree, W, W*,
  tol=1e-2, sub_tol=1e-2)` (k*, minimal routes, the string with the largest |R_C| entry on the
  route of largest ||R_C||); escape time = first t after the point with L < L_point - 1e-3,
  measured from the segment start; for the star controls also `align` + `route_entry_norms`;
- per drop: patterns entering / leaving J between the minimum-gradient points around it, the
  string whose W entry changes most, peak off-target mass, **effective depth** from the loss
  excess (`effective_depth` on L_point - L(t) with s = the drop's size; registered for every drop)
  and from the amplitude of each pattern learned in the drop (window P/s in [1e-4, 1e-2]);
- per learning event (brief's events.csv): J before it, duration since the previous learning
  event, drop index, amplitude effective depth over [previous event, event].

## 4. Plateau identity across sigma and seeds

A plateau's **key** = J_amp + "*" if it is not canonical, + the level: the index of the
route-rule-sequence level within 5 % (relative) of the plateau loss if there is one ("RS<j>"),
otherwise the loss rounded to 2 significant digits. Per-seed exponent fits pool the same seed's
durations of the same key over sigma.

## 5. Estimators and decision rules

- **P1 per plateau key**: fraction of plateau points (over seeds and window sigmas) with d_near <
  0.03 (canonical); median d_near and off-target mass; the off-target strings with the recombinant
  tag. Decision: "canonical" if >= 90 % canonical; "rank-constrained optimum" if <= 10 % canonical
  and the level matches a route-rule-sequence level within 5 % in >= 90 % of points; else
  "mixed / other".
- **P3 per-seed exponents**: least squares of log10(duration) on log10(sigma) (slope negated) over
  the window, for each (target, key, seed) with the key present at >= 5 window sigmas spanning >=
  1.5 decades. Reported: median, quartiles, number of seeds, fraction within 0.3 of each
  predictor's exponent: (A) max(c*_J - 2, 0) with J = J_amp of the plateau (complete J: no
  plateau predicted, exponent 0); (B) k* - 2 with k* the median route-rule k* of the key;
  (C) max(min pairwise - 2, 0); and, where registered, the by-hand order - 2. **Decision per
  (key, predictor): "holds"** if the median is within 0.3 of the prediction and >= 70 % of seeds
  are within 0.3; **"fails"** if the median is off by more than 0.5 or < 30 % are within 0.3;
  otherwise "partial". For predicted exponent 0 (order 2, log law) the power-law slope is compared
  with 0 and, in addition, log-law and power-law fits of the median durations are compared by RSS
  of log10(duration).
  The same fits are made on the brief's event-to-event durations, grouped by (target, J before,
  pattern), for comparability with `route-smallinit`.
- **P2 sequence** per drop out of a plateau (conditional on the J_amp actually reached): a
  predictor is right if its predicted next pattern is among those entering J in that drop; drops
  in which no pattern enters J ("clean-up drops") count as a miss for (A), (C) and weight order,
  and as a hit for (B) if its predicted string is the off-target string whose entry changes most.
  Reported per target and sigma: fraction right.
- **P4**: per drop, |effective depth - predicted order| for each predictor; summary: median and
  fraction within 0.5.
- **P5**: per key, the median lambda_min over seeds at each window sigma; slope x of
  log10|lambda_min| on log10 sigma. "strict" if the median lambda_min < -1e-4 at every window sigma
  and x < 0.25; "degenerate" if x >= 0.5 (x estimates p - 2); else "unclear". Compared with the
  duration exponent of the same key.
- Every run that is trapped, unfinished (wall, t_max), failed, or floor-flagged at some plateau is
  counted per target and sigma; none is dropped silently. Seeds that learn in different orders
  give different keys, which are fitted separately.

## 6. Status tags

Results are **empirical** unless stated; hand orders are **derived**; Theorem 3.2 statements are
**proven** (paper); anything beyond is **conjecture**.

## 7. Budget

9,520 runs at 2-25 s on 120 single-threaded workers: about 30 min. If the large trees at the
smallest sigma are too slow, log10 sigma = -3.5 (then -3.25) is cut for K = 6, 7 first.

## 8. Deviations from the pre-registration

(none yet)
