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

1. **Transient segments merged into the escape (analysis only, 19:55, before any fit was looked
   at; seen on the first ~800 runs, K = 7 trees at the smallest sigmas).** `find_plateaus` sometimes
   splits one escape into two drops when its rate has two stages (e.g. TD_binary8: 0.0152 -> 0.0105
   -> 0; CTL1_binary8: 0.1013 -> 0.0724 -> 0 while 10111001 is being learned). The segment between
   such drops is not a plateau: its minimum-gradient point has |grad| ~ 0.01-0.07 and is not near a
   critical point (route rule k* = 1). As registered, it would get its own plateau key and would
   split the transition, e.g. the star control CTL1_binary8 would show a "clean-up" drop followed
   by learning, and every predictor would be scored on a non-plateau. Rule now: a segment is a
   **plateau only if `find_plateaus` resolves it** (separation ratio > 3, the repo's plateau
   definition; registered as the "resolved" flag); an unresolved segment is a transient and the
   drops around it form one escape (learned set = difference between the plateau points before and
   after; effective depth from the first sub-drop, whose window holds the onset of the escape;
   string changing most = that of the largest sub-drop). Durations are unchanged (they end at the
   first sub-drop). Check that the rule separates the two kinds: over those runs the ratio
   |grad|^2 x width / (loss drop) at the segment's minimum-gradient point is < 1e-3 for every
   resolved segment (1,650) and >= 1e-3 for every unresolved one (77). Transient counts are kept
   per run (`runs.csv: n_transients`).
2. **Secondary duration fits for plateaus at the float64 floor (analysis only, 20:10, after
   seeing that the depth-5 plateaus of the K = 7 targets had no registered fit).** The registered
   floor flag (|grad| < 100 x roundoff bound at the minimum-gradient point) removes sigma <= 1e-3
   for depth-5 plateaus (the gradient on such a plateau is ~ sigma^4), leaving 1.25 decades, below
   the registered 1.5. Their primary fits stay "not fitted". Added, labelled as secondary and never
   used for the registered decisions: (a) unflagged points with a span of >= 1 decade, (b) all
   points including floor-flagged ones (span 1.5 decades). Both are in `fits_secondary.csv` and in
   `fit_summary.csv` (columns `*_unflagged_span1`, `*_with_flagged`).
3. **Existence check for "no plateau" predictions (analysis only, 20:10).** When a rule predicts no
   plateau (paper or pairwise: every missing pattern has depth <= 1, or nothing is missing; route
   rule: k* <= 1), the registered exponent test compares the measured exponent with 0, which a
   log-law plateau (k* = 2, exponent ~0.15 over two decades) also passes. Every plateau key is by
   construction a resolved plateau (deviation 1), so for these keys the rule is scored "fails (no
   plateau predicted)"; the registered exponent comparison is kept in the CSV
   (`frac_within_*`, `exp_*`).
4. **Route-entry norms computed post hoc (21:05).** The registered per-run measurement of the
   route-entry norms for the star controls was silently skipped: `run.py` built its
   `AlignedManifold` with the manifold's own bond dimensions, which differ from the network's
   (rank + 2 per edge), and then set it to None. `entries.py` recomputes them from the saved
   plateau states with the manifold's `align` / `route_entry_norms` on the network's own spec
   (they only read bond dimensions and the target), for the three controls and for RND3_mps6
   (non-star, but its plateaus are aligned with quality > 0.9999), into `results/route_entries.csv`.
   The `entry_*` columns of `plateaus.csv` stay empty. Event-based identities: deviation 3's
   existence rule is replaced there by the same-drop rule (README).

# Part II: results

Everything below is from the full sweep (9,520 runs, 20 targets, all 40 seeds at every grid
sigma) as analysed by `experiments/route_nonstar/analyze.py` and `headline.py`. Tables:
`experiments/route_nonstar/results/` (`headline.md` = the numbers quoted here,
`summary_by_target.md` = per-target tables, CSVs documented in the README). Figures `fig1`-`fig6`
in the same folder. "Window" = log10 sigma <= -1.5. Status tags: section 6.

## 9. Answer in short

On pattern targets that break the star condition, gradient flow from small init still learns in
steps separated by plateaus, but **the plateaus are often not partial-target points**: after the
first escape, 68 % of the plateau points of non-star targets are not within 0.03 of any partial
target (star controls: 0.4 %), and 59 % carry a *recombinant* string (a string made of the lower
side of one target pattern and the upper side of another across one edge). These plateaus are
the **rank-constrained optima** of `notes/locality.md` section 5 (18 of 19 such plateau types sit
at a level of the numerical route-rule sequence within 5 %).

- The paper's route depth, applied anyway, **fails at every such plateau**: with all patterns
  already above half weight (or the missing ones at depth <= 1) it predicts that training is over
  or proceeds without a plateau, while a plateau of length sigma^-1 to sigma^-3 follows
  (13 of 25 fitted non-star plateau types: "no plateau predicted" but one exists; 3 more with the
  wrong exponent). It **holds at the canonical plateaus** (8 of 10 fitted canonical non-star
  types; the 2 misses are the spectral shortcut and the head start below, where the route rule
  fails too), it names the next pattern correctly on 99.4 % of the escapes that learn a pattern,
  and depth 0 ("produced for free as a recombinant")
  is right in the narrow sense that such a pattern crosses half weight in the same drop as its
  parents.
- The pairwise (Prim) rule fails in exactly the same places (it agrees with the paper's depth on
  all these targets).
- **What replaces them: the numerical route rule at the actual plateau point** (smallest connected
  core set whose local residual is non-zero, k*). It predicts the plateau-length exponent k* - 2
  for 23 of 25 fitted non-star plateau types (every seed within 0.3 in 23 of them), the effective
  depth of the escape (within 0.5 of k* in 94.7 % of 8,184 escapes), the escape type (strict saddle
  iff k* = 2: 10 of 10 types; degenerate with smallest Hessian eigenvalue ~ sigma^(k*-2) for
  k* >= 3: 32 of 32), and what is learned or removed next (89 % of 8,759 escapes; 99 % if a
  "clean-up" escape counts as predicted when the route rule names the string that changes most,
  target or not; the paper's greedy rule: 54.5 %, all of its misses being clean-ups). **empirical**
- Where it fails: (i) a **spectral shortcut**: the order is below k* (exponent 2 where k* - 2 = 3)
  at the one-pattern plateau of the star control CTL1_mps6, where Theorem 3.2 predicts it exactly
  (order min(k*, 2 l*) = min(5, 4) = 4; **proven** order, **empirical** exponent), and at a
  canonical three-pattern plateau of RND1_mps6 (order 4, effective depth 3.9, route depth 5), where
  no theorem applies and the parent-wise formula of Theorem 9.2 predicts 2 (wrong); (ii) a
  **non-integer exponent** 0.61 [0.57, 0.70] at a canonical depth-3 plateau of RND3_mps6 whose
  escape still has effective depth 3.03: the route enters the plateau larger than sigma
  (Assumption C.1(1) fails); **empirical**, mechanism **conjecture** (section 13).
- Learning on these targets also produces **dead routes** (training stuck for good at a
  rank-constrained optimum with a recombinant, 146 runs on R4 / R4d, always the same 5-6 seeds at
  every sigma) and **forgetting** at spectral escapes (section 14).

## 10. P1: what the plateaus are (empirical)

Plateau points after the first escape, window sigmas, every seed (`headline.md`):

| targets | plateau points | canonical (d_near < 0.03) | carrying a recombinant string |
|---|---|---|---|
| non-star (17 targets) | 8,902 | 31.6 % | 59 % |
| star controls (3) | 2,388 | 99.6 % | 0.9 % |

By plateau type (target, key; >= 20 window points): non-star 14 canonical types and 19
non-canonical ones (17 with a recombinant, 18 at a route-sequence level within 5 %); star
controls 9 canonical types. The registered decision (>= 90 % canonical = "canonical", <= 10 % =
"rank-constrained optimum") classifies every type; no type is mixed. Typical non-canonical
plateaus (`summary_by_target.md`):

| target | plateau (learned set J, loss) | off-target strings (recombinant across edge e: lower side a, upper side b) | distance to the nearest partial target |
|---|---|---|---|
| R3 | {000,100,001}, L = 0.01944 | 101 = e0: 100 \| 001, amplitude 0.166 | 0.16 |
| R4 | {0000,1000,0001}, L = 0.01944 | 1001 = e0: 1000 \| 0001 | 0.16 |
| R3d, R4d | all four patterns, L = 0.01953 | none; the amplitudes are those of the best product tensor | 0.15 |
| NC_ht4 | {0000,1100}, L = 0.06376 then 0.05775 | 0010 = e2: 0000 \| 1110, 0001; then 0010, 0011 = e2: 0000 \| 1111 | 0.16, 0.19 |
| NC_binary4 | {0000,1100}, L = 0.05775 | 0010, 0011 (edge e0) | 0.19 |
| TD_mps6 | all three, L = 0.01518 | 110110 = e1: 110000 \| 000110 | 0.14 |
| TD_binary8 | all three, L = 0.01518 | 11001100 = e0: 11000000 \| 00001100 | 0.14 |
| RND2_binary8 | three of four, L = 0.1002 | 10001101, 00110001 (edge e2) | 0.19 |
| RND3_binary8 | all three, L = 0.1142 | 01111000 = e2: 01111001 \| 11001000 | 0.33 |

The recombinant family makes the mechanism explicit (**derived**, checked numerically): for
q = 0..0 with the two end edits, W* restricted to the end sites is the 2 x 2 matrix
[[1, 0.4], [0.6, 0]] (R3, R4) or [[1, 0.4], [0.6, 0.5]] (R3d, R4d). The first escape learns its best
rank-1 (product) approximation: loss = (second singular value)^2 / 2 = 0.019445 (0.019527),
exactly the measured levels, and the product puts amplitude 0.166 on the recombinant 10..01 (R3,
R4). All three patterns pass half their weight in that one escape, so the learned set is complete
while the loss is not zero. With 10..01 at the product weight 0.24 (R3p, R4p) W* is itself a
product tensor and there is a single drop to zero loss, no later plateau (520 + 520 runs).

Star controls behave as the paper says: their plateaus are canonical (median distance 1e-8 to
4e-3), no off-target strings beyond 0.01 except transiently.

## 11. P3 / P4 / P5: plateau lengths, effective depths, escape type (empirical)

Registered per-seed fits (window, >= 5 sigmas over >= 1.5 decades, unflagged), plateau types with
>= 5 fitted seeds (`headline.md`, `fit_summary.csv`):

| predictor | non-star types (25): holds / fails | of which "no plateau predicted" | star types (6): holds / fails |
|---|---|---|---|
| paper's route depth (applied anyway) | 9 / 16 | 13 | 5 / 1 |
| numerical route rule (k* at the plateau point) | **23 / 2** | 0 | 5 / 1 |
| pairwise rule | 9 / 16 | 13 | 5 / 1 |
| registered by-hand order (where defined) | 7 / 0 | - | 4 / 0 |

"Holds" = median within 0.3 and >= 70 % of seeds within 0.3; in every "holds" row of the route
rule the fraction within 0.3 is 1.00 and the quartiles span <= 0.05 (exponents 0.04-0.17 for k* = 2
log-time plateaus, 1.00-1.03 for k* = 3, 2.00 for k* = 4). Depth-5 plateaus are floor-flagged at
sigma <= 1e-3 and have no registered fit; the secondary fits (deviation 2; flagged points
included) give 3.00 [2.99, 3.00] (TD_binary8 recombinant plateau; paper predicts no plateau),
3.00 (RND1_binary8, CTL1_binary8 RS2), 3.13-3.15 (CTL1_mps6, RND1_mps6) against k* - 2 = 3.

The by-hand orders hold everywhere they are registered: the rank-1 optima of R3, R3d (order 3,
exponent 1.00 [1.00, 1.00], 40 seeds each) and R4, R4d (order 4, exponent 2.00 [2.00, 2.00], 31
seeds; 9 seeds trapped), and every canonical one-pattern plateau via Theorem 3.2 (NC_ht4 {0000},
NC_mps4 {0000}, RND3_binary8 {01111001}, CTL1_mps6 {100100}, {100011}, {010101}).

The two route-rule failures (non-star) and the star failure:

| target, plateau | measured exponent | route rule k* - 2 | paper / pairwise | what it is |
|---|---|---|---|---|
| CTL1_mps6 (star), J = {010101} | 2.00 [2.00, 2.01] (14 seeds), effective depth 4.00, lambda_min ~ sigma^1.99 | 3 | 3 / 3 | spectral shortcut, predicted exactly by Theorem 3.2: k* = 5, l* = 2, order 4 (**proven** order) |
| RND1_mps6, J = {010110, 010100, 101110} (canonical), learning 110101 (weight 1, the largest) | 2.02 [2.01, 2.05] (19), effective depth 3.91, lambda_min ~ sigma^2.00 | 3 | 3 / 3 | order 4 < route depth 5 at a three-pattern saddle; parent-wise min(k*_a, 2 l*_a) (Theorem 9.2 outside its assumption) gives 4, 2, 4 per parent, so the minimum (2) is wrong |
| RND3_mps6, J = {110100, 101100, 101010} (canonical), learning 000010 | 0.61 [0.57, 0.70] (40), effective depth 3.03, lambda_min ~ sigma^0.51 | 1 | 1 / 1 | depth-3 escape from a route that enters the plateau larger than sigma (section 13) |

Effective depth of every escape (loss estimator, unflagged, window): within 0.5 of the route
rule's k* in 94.7 % of 8,184 non-star escapes (median |difference| 0.03) and 88 % of 1,833 star
escapes (the misses are the log-time k* = 2 escapes, whose estimator gives 1.7-1.9, and the
spectral shortcut); within 0.5 of the paper's depth in 30.9 % (non-star) and 87 % (star).

Escape type (`escape_type.csv`, 42 plateau types with >= 20 points): every k* = 2 type is strict
(median lambda_min -0.06 to -0.60, independent of sigma: slope |x| < 0.01; plateau length grows
like log(1/sigma), power-law slope 0.04-0.17), every k* >= 3 type is degenerate with slope
x = 1.00-1.07 (k* = 3; except RND3_mps6 RS3, 0.51), 1.85-1.99 (k* = 4) and 2.69-3.04 (k* = 5;
lower values come from the floor at the smallest sigma; the two spectral-shortcut plateaus with
k* = 5 have x = 1.99 and 2.00, i.e. order 4), i.e. lambda_min ~ sigma^(k* - 2), the
same exponent as the plateau length. Recombinant plateaus are degenerate saddles of order k* =
the size of the connected region that must open a new bond channel: 3 on R3 and NC_binary4, 3 on
NC_ht4 and TD_mps6, 4 on R4, RND2/RND3_binary8, RND2_mps6, 5 on TD_binary8. **empirical**; for
the recombinant family the order K is **derived** in section 2 and matches.

## 12. P2: what is learned next (empirical)

Per escape from a plateau, conditional on the learned set reached (window; `sequence.csv`):

| targets | escapes | paper's greedy rule | route rule | route rule (any string) | pairwise | weight order |
|---|---|---|---|---|---|---|
| non-star | 8,759 | 0.545 | 0.894 | 0.988 | 0.545 | 0.474 |
| star controls | 2,275 | 0.997 | 0.991 | 0.991 | 0.997 | 0.702 |

45 % of the non-star escapes (3,953 of 8,759) are **clean-ups**: no pattern enters the learned set;
the loss drops because an off-target string is removed and the amplitudes are corrected (R3, R4,
R3d, R4d, TD_*, NC_ht4 RS1, RND1_*, RND3_binary8). The paper, the pairwise rule and weight order
cannot predict these by construction. On the 4,806 escapes that do learn a pattern, the paper's
greedy next pattern is right in 99.4 % (pairwise 99.4 %, route rule 97.9 %, weight order 86.4 %;
the lowest per target is RND3_mps6, 0.96 for the paper and 0.90 for the route rule). So the
sequence failure of the paper on non-star targets is entirely the clean-up escapes. The route rule's string
is the off-target string that is removed in 89 % of all escapes; the registered rule counts R3d /
R4d clean-ups as misses because their plateau carries no off-target string (the string that changes
most is the target string 10..01, which the route rule names: "any string" column).

## 13. Head start: the non-integer exponent of RND3_mps6 (empirical; mechanism conjecture)

At the canonical plateau J = {110100, 101100, 101010} of RND3_mps6 (learning 000010, route depth 3)
the plateau length grows from 241 (sigma = 10^-1.5) to 5,293 (10^-3.5): local exponent 0.84 at the
large end, 0.58 at the small end, median per-seed exponent 0.61; the escape itself has effective
depth 3.03 and lambda_min ~ sigma^0.51. The amplitude of 000010 at the plateau point scales as
sigma^2.33 (from 5.5e-6 to 1.2e-10), not sigma^3: the dormant route enters the plateau at about
sigma^0.6-0.8, not sigma. The route-entry norms read off the aligned network
(`results/route_entries.csv`, deviation 4; alignment quality 0.99999) confirm it: their geometric
mean over the 3 route nodes scales as sigma^0.78 at this plateau (and as sigma^0.85 already at the
strict plateau before it), where Assumption C.1(1) wants sigma^1; on the star controls the same
slope is 0.80-1.11 for the depth-3 to depth-5 routes. This plateau follows two strict (log-time) plateaus; a candidate
mechanism (**conjecture**) is that during a strict plateau of length ~ log(1/sigma)/lambda the
dormant entries of the next route grow (or shrink) at a rate lambda', which turns sigma into
sigma^(1 - lambda'/lambda) at entry and gives the exponent (c - 2)(1 - lambda'/lambda). This is a
failure of Assumption C.1(1) (dormant entries of order sigma on entry) on a canonical plateau; it
was not tested further.

## 14. Runs that do something else (empirical)

| what | runs | where | what happens |
|---|---|---|---|
| trapped (dead route) | 211 | R4 81, R4d 65, B4_mps4 65 (star control), at every sigma | the same 6 (R4), 5 (R4d), 5 (B4) seeds at all 13 sigmas (2 more R4 seeds at one or two sigmas); the depth-4 route through both chain ends dies: gradient 1e-16, all non-learned entries of W ~ 1e-15; R4 / R4d stay at the rank-1 optimum with the recombinant 1001 (or wrong amplitudes) for good; B4 stays at {0000, 0110} (locality's mps4 {0,3} dead routes) |
| wall-clock cap (1800 s) | 79 | CTL1_mps6 58 (sigma <= 10^-2.25), TD_binary8 21 (sigma = 10^-3.5) | a depth-5 plateau at the float64 floor (gradient ~ 3e-16 to 6e-14), LSODA crawls; the trap rule did not fire because its checks are spaced in absolute time, which is ~1e17 on these runs; all are floor-flagged at that plateau |
| floor-flagged plateau | 1,185 | depth-5 plateaus at sigma <= 1e-3, depth-4 at sigma <= 10^-3.75 | excluded from the registered fits; secondary fits in section 11 |
| forgetting (a pattern falls below s/4 after passing s/2) | 452 | mostly inside the first escape (NC_ht4, RND2_binary8, CTL1_binary8, RND3_mps6, ...: the first escape passes a mixed state before settling); later only in CTL1_mps6 (213 runs: 21 of 40 seeds at every sigma <= 10^-1.75), RND1_mps6, RND3_mps6 | CTL1_mps6 (a star control): the spectral escape from J = {010101} learns 100011 and 100100 and destroys 010101 (the heaviest pattern), which is relearned only after a depth-5 plateau; 35 of the 58 CTL1_mps6 wall runs are of this kind |
| transient segments merged (deviation 1) | 2,312 runs | everywhere | an escape whose loss drop has two stages |

## 15. Claims and status

1. On non-star targets, plateaus after the first escape are often rank-constrained optima carrying
   recombinant strings, not partial-target points (68 % of plateau points; 17 of 19 non-canonical
   types carry a recombinant). **empirical**
2. For the recombinant family, the plateau is the best rank-1 approximation of W*, its order is K
   (direct degree = the whole chain; no spectral shortcut), and its length grows as sigma^-(K-2).
   **derived** (order) and **empirical** (exponent 1.00 and 2.00, every seed).
3. The paper's route depth and the pairwise rule predict no plateau at these points; the plateaus
   exist and last sigma^-1 to sigma^-3. **empirical**
4. The numerical route rule at the measured plateau point predicts exponent, effective depth,
   escape type and the next string for every plateau type except spectral shortcuts and one
   head-start case. **empirical**
5. At canonical one-pattern plateaus Theorem 3.2 gives the order also on non-star targets,
   including a spectral shortcut (order 4 < k* = 5) on a star control. **proven** (order),
   **empirical** (dynamics)
6. At a canonical three-pattern plateau of RND1_mps6 the order is 4, below both the route depth
   (5) and the route rule (5); a parent-wise min(k*, 2 l*) gives 2. The conjectured
   min(direct, 2 x spectral) of Appendix B is consistent with 4 only if the spectral level is
   computed with multi-pattern atoms, which was not done. **empirical**; formula **conjecture**.
7. Dead routes make 15 % (R4) and 12.5 % (R4d, B4) of the seeds stop for good; whether a route
   dies is a property of the seed, not of sigma. **empirical**

## 16. Open and next

- A static order measurement (sphere minimum, as in `route-escape`) at the rank-constrained
  optima would turn claim 4 from "the escape behaves as order k*" into a measured order.
- The spectral level at multi-pattern saddles (RND1_mps6) needs atoms built on the learned
  channels; the parent-wise formula is not it.
- The head start of section 13: record the route entries along the preceding strict plateaus.
- Trap rule in relative time (it missed the 79 wall runs).
