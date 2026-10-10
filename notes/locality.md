# Locality of saddle-to-saddle escapes in TTNs (Q3: H2, H3)

Agent `locality`, branch `agent/locality`. Experiments in `experiments/locality/`. Status tags:
**proven**, **derived** (under stated assumptions), **empirical**, **conjecture**.

Model and diagnostics as in `notes/model.md` and `notes/diagnostics.md`: whitened inputs, exact
population loss L = 1/2 ||W - W*||^2, gradient flow, fan-in init with per-core scale sigma.

## 1. Setting: two-component pattern teachers

W* = s1 U1 + s2 U2 with U_r = (x)_i u_r^(i), u_r^(i) taken from an orthonormal basis of site i
(`pattern_teacher`). U2 differs from U1 exactly on the site set S (u_2^(i) = u_1^(i) for i not in S,
u_2^(i) orthogonal to u_1^(i) for i in S). s1 > s2 > 0 unless stated. K = number of cores.

Notation: S-nodes = cores carrying a site of S (m of them); T_S = Steiner subtree of the S-nodes
(smallest connected set of cores containing them); k = |T_S|; E(T_S) = edges inside T_S.

**Teacher bond ranks (proven).** Across edge e (bipartition A | B), W*_e = s1 a1 b1^T + s2 a2 b2^T
with a_r, b_r the restrictions of U_r. If S lies on one side, say A, then b1 = b2 and W*_e =
(s1 a1 + s2 a2) b1^T has rank 1 with singular value sqrt(s1^2 + s2^2). If S meets both sides,
a1 ⟂ a2 and b1 ⟂ b2, so the rank is 2 with singular values s1, s2. An edge separates S iff it lies
on a path between two S-nodes iff e ∈ E(T_S). So **exactly the edges of T_S must reach rank 2.**

## 2. The saddle after the first component and the escape degree (H2)

Take the student at W = s1 U1, represented by a rank-1 TTN (every learned bond uses one index,
"index 0"). Under gradient flow from small init the conservation laws make the cores balanced,
so every core has norm q in the learned direction with q^K = s1.

**Critical point iff k >= 2 (proven).** The residual is R = W* - W = s2 U2. The gradient with
respect to core v is minus the contraction of R with all other cores. Every other core w carries
u_1^(i) at its sites, and <u_1^(i), u_2^(i)> = 0 for i in S, so the gradient vanishes unless all of
S sits on v itself, i.e. unless k = 1. For k = 1 the point is not critical: the single core can add
U2 linearly (amplitude grows at rate s2 q^(K-1)) and **there is no second plateau**.

**Escape requires all k cores of T_S (proven for the necessity part).** By Eckart–Young, s1 U1 is the
unique best approximation of W* among tensors of rank <= 1 across any edge e ∈ E(T_S) (singular
values s1 > s2 there). Hence any W with L(W) < L(s1 U1) has rank >= 2 across every edge of T_S, and
a bond rank can only grow if both cores at its ends acquire components outside the learned
index. So every core of T_S must move away from its learned configuration.

**Lowest-order escape route (derived).** Expand the cores as learned part + perturbation. Two kinds
of perturbation matter:

1. *New-index route.* Each core v ∈ T_S gets an entry of size eps_v that uses a new bond index
   ("index 1") on its edges inside T_S, index 0 on edges leaving T_S, and u_2^(i) at its S-sites.
   A single such entry contracts with zero new-index entries of its neighbours, so it does not change
   W at first order and pays no quadratic penalty. The first non-zero contribution is the product
   over all of T_S:
   delta W = (prod_{v in T_S} eps_v) q^(K-k) U2, so
   L = L_saddle - lambda_k prod_{v in T_S} eps_v + O(eps^(2k)),  lambda_k = s2 q^(K-k).
   Any connected proper subset of T_S using new indices produces tensors orthogonal to U2 (some
   S-site keeps u_1) and only adds a non-negative term ||delta W||^2 / 2.
2. *Index-0 ("piggyback") route.* The S-nodes rotate their physical vectors toward u_2 while keeping
   index 0. This changes W at first order (positive curvature q^(2K-2) per core) and couples to U2 only
   at order m (number of S-nodes). For m = 2 the 2x2 Hessian block is
   [[q^(2K-2), -s2 q^(K-2)], [-s2 q^(K-2), q^(2K-2)]] with eigenvalues q^(K-2) (s1 ± s2) > 0. For
   m >= 3 the coupling is cubic or higher and the quadratic part is positive definite. So for
   s2 < s1 this route is never an escape direction. (If the *smaller* component was learned first,
   q^K = s2 and the m = 2 block has eigenvalue q^(K-2)(s2 - s1) < 0: the H1b piggyback saddle.)

So, for s2 < s1 learned in greedy order, the escape is a **degree-k monomial mode**, exactly like a
balanced depth-k linear-network mode with target lambda_k:
  eps' = lambda_k eps^(k-1)   (balanced eps_v = eps, before saturation).

**Escape-time law (derived; balanced, aligned reduction).** With eps(t1) the size of the
new-index entries when the first component has been learned:
- k = 2: t2 - t1 = (1/lambda_2) log(eps_sat / eps(t1)) + O(1);
- k >= 3: t2 - t1 = eps(t1)^-(k-2) / ((k-2) lambda_k) + O(1).

**What eps(t1) is: carry-over from the first plateau (derived).** During the first plateau W ≈ 0
and the residual is W* itself. The T_S cores contain two independent blocks, the one that will carry
component 1 (amplitude a per core) and the new-index block for component 2 (amplitude eps), and both
are driven by the same product C(t) of the shared cores outside T_S:
  a' = s1 C a^(k-1),   eps' = s2 C eps^(k-1).
Eliminating C:
- k = 2: log(eps/eps0) = (s2/s1) log(a/a0). Component 1 is learned when a reaches O(q), so
  eps(t1) ∝ sigma^(1 - s2/s1) and
  **t2 - t1 = (1 - s2/s1)/lambda_2 * log(1/sigma) + O(1),  lambda_2 = s2 s1^((K-2)/K).**
  The log-law slope is (1 - s2/s1)/lambda_2, *not* 1/lambda_2: the second component grows during
  the whole first plateau. (For s1 = 1, s2 = 1/2: slope 1, against 2 for the naive 1/lambda.)
- k >= 3: eps(t1)^-(k-2) = eps0^-(k-2) - (s2/s1) a0^-(k-2). With eps0, a0 ∝ sigma (same random
  init direction rescaled by sigma),
  **t2 - t1 = A sigma^-(k-2) + O(1)**, A = c_init (1/s2 - 1/s1) / ((k-2) q^(K-k)),
  c_init a seed-dependent O(1) factor (it can be small when the two components nearly tie, and when
  eps0^-(k-2) < (s2/s1) a0^-(k-2) the order is reversed: component 2 is learned first).
- k = 1: t2 - t1 = O(1), independent of sigma (in the aligned reduction both overlaps obey the same
  ODE, so the two events coincide).

Exponent predictions: p = k - 2 for k >= 3; log law for k = 2; none for k = 1. Because each seed
fixes the random init *direction* and sigma only rescales it, the law holds seed by seed, so the
exponent is fitted per seed and averaged.

## 3. Saddle type (H3)

**Derived.** At the exact saddle W = s1 U1 (greedy order):
- k = 2: the Hessian restricted to each pair (eps_u, eps_v) of new-index entries is
  [[0, -lambda_2], [-lambda_2, 0]]: eigenvalue **-lambda_2 = -s2 q^(K-2)**, multiplicity chi_e - 1
  (one pair per unused bond direction on the T_S edge). Strict saddle.
- k >= 3: no negative eigenvalue (the new-index directions are null to second order; the index-0
  directions are positive). Higher-order saddle.
- At the plateau *midpoint* (where the Hessian is measured) eps is small but non-zero, so for
  k >= 3 the Hessian of -lambda_k prod eps_v gives lambda_min ≈ -(k-1) lambda_k eps^(k-2): a
  negative eigenvalue that **vanishes like sigma^(k-2)** as sigma -> 0, while for k = 2 it tends to
  the constant -lambda_2. This scaling is the sharp, threshold-free test of H3.

## 4. Bond spectra at the second event

**Derived (from section 1).** The second singular value appears on exactly the edges of T_S, all
at the same time (they are all driven by the same product prod eps_v); all other edges stay rank 1,
and their top singular value moves from s1 to sqrt(s1^2 + s2^2) at the same event.

## 5. Several learned components: the route rule at rank-constrained optima

"The sites where the new component differs from what is learned" is ambiguous once two or more
components are learned. Three candidate rules (all in `experiments/locality/rules.py`):

1. *Literal H2* (`naive_k`): Steiner size of the sites where the new component differs from the closest
   learned component.
2. *Exact rank rule* (`rank_rule`, my first generalization): k_ex = number of cores incident to the edges
   whose bond rank must rise to represent the partial sum plus the new component. Correct when the
   saddles are partial sums of components.
3. **Route rule** (`route_rule`; = theory_saddles.md R1, which I implement numerically): at the actual
   saddle W, for every connected core set C compute the local residual R_C = the residual W* - W with
   every subtree hanging off C projected onto the bond subspace that the outside cores already carry;
   k* = min |C| with R_C != 0 (k* = 1 means W is not critical). **Derived**: a route on C changes W
   only inside that projected space and is first-order invisible, so its leading term is
   -<R_C, .> at degree |C|; any descent needs some R_C != 0.

The saddles are not always partial sums. If two learned components share their restriction on one
side of an edge, that bond carries a superposition (e.g. s1 e00 + s2 e11 on sites {0,1} for
U1 = 0000, U2 = 1100), and a third component whose restriction overlaps the superposition without being
in it gets a non-zero gradient: for TA = {0000, 1100, 1110} the partial sum s1 U1 + s2 U2 has k* = 1 on
every tree. The flow then slides, without a plateau, to the rank-constrained optimum of the current
bond-rank profile (here the best approximation with rank 1 across {01|23}, partly fitting U3), and the
next escape starts from there. **Sequence algorithm** (theory_saddles.md 5.1, implemented in
`rules.route_sequence`): bond ranks 1 -> rank-constrained optimum (L-BFGS on a TTN with exactly those
bond dimensions, several restarts) -> route rule there -> raise the rank on the internal edges of the
minimal route -> repeat. It predicts the plateau levels L_j and degrees k*_j.

For two components, rules 1-3 coincide (checked for every (tree, S) used; proven in section 2). For
several components:

| teacher | tree | literal H2 | exact rank rule | route sequence (L_j: k*_j) |
|---|---|---|---|---|
| TA {0000,1100,1110} | mps4 | 2, 1 | 2, 2 | 0.241: 2 -> 0.044: 2 |
| | binary4 | 1 | 3 | 0.044: 3 (first escape already fits U1, U2 and part of U3) |
| | ht4 | 3, 1 | 3, 4 | 0.241: 3 -> 0.044: 4 |
| N4 {0000,1100,1111} | mps4 | | | 0.260: 2 -> 0.080: 2 -> 0.057: 2 (partial step) |
| | binary4 | | | 0.057: 3 |
| | ht4 | | | 0.260: 3 -> 0.080: 3 -> 0.057: 3 |
| B4 {0000,0110,1001} | mps4 | | | 0.225: 2 (U2, s = 0.3) -> 0.180: 4 (U3, s = 0.6) |
| | binary4 | | | 0.225: 3 -> 0.045: 3 |
| | ht4 | | | 0.225: 5 -> 0.045: 5 |

(N = 8 teachers: `experiments/locality/results/route_sequences.json`.) On B4 locality beats weight:
the MPS learns the 0.3 component before the 0.6 one.

## 6. Relation to the literature (after `notes/literature.md` v1)

- **Landscape upper bound.** [Furman2026] Lemmas D.8-D.9 (same model and loss): at a min-norm
  non-global critical point there is a descent curve of order N, N = number of nodes touching a
  connected component of rank-deficient edges (student rank < teacher rank). For two-component
  teachers at the saddle s1 U1 the deficient edges are exactly E(T_S), which is connected, so
  **N = k**: the Steiner count is Furman's N, and our derivation adds that it is also a *lower* bound
  for the descent order at this saddle (section 2: all of T_S must move, and no lower-degree term has
  a non-zero residual correlation), plus the dynamics (escape-time law with the carry-over factor).
- For several learned components N and the rank-rule k differ: the deficient set is the union of what
  *all* remaining components need, while the escape takes the cheapest single component. Then
  k = min_r k(L, r) <= N (`rules.furman_N`). Examples (section 5 teachers): TA on `ht4`, second escape
  k = 3 vs N = 6; TC on `mps8`, exact-rank-rule degrees k_ex = 2, 3, 5 for U2, U3, U4 vs N = 8, 7, 5
  (the route-rule sequence, which includes partial steps, has k* = 2, 2, 2, 4, 2). This is the lead's refinement (ii)
  ("lowest degree with non-zero residual correlation"), computed per case by the rank rule.
- **Dynamics analogues.** [RawalDeWeese2026] Thm 4.2 (DLNs): escape Theta(1/Gamma),
  Theta(log(1/eps)/Gamma), Theta(eps^-(r-2)/Gamma) for r = 1, 2, >= 3 small layers: the same
  trichotomy, here with r = number of cores of the escape route and eps the size of their dormant
  entries. [KumarHaupt2025c] Thm 8: plateau >= ~delta^-(L-2), L the lowest homogeneity degree of the
  dormant block; our k is that L for TTN saddles. [Razin2022] Thm 1 (per-node exponent 2 - 2/L_nu) is
  for CP-structured nodes and does not count Steiner cores.
- **Dormant entries are rescaled by earlier phases** (refinement (iii)). Section 2 computes this for
  the two-component case: eps(t1) ∝ sigma^(1 - s2/s1) for k = 2 (hence the log-law slope
  (1 - s2/s1)/lambda_2) and eps(t1)^-(k-2) = eps0^-(k-2) - (s2/s1) a0^-(k-2) for k >= 3. The
  gauge-invariant handle on the dormant block is the overlap itself: a_r ≈ prod_{v in route} eps_v
  q^(K-k) / s_r on the plateau, so the law predicts, run by run, for the remaining time
  t_b - t_mid from the plateau midpoint to the next event:
  t_b - t_mid ≈ (1/(2 lambda)) log(1/a_r,mid) + c for k = 2 and ∝ a_r,mid^(-(k-2)/k) for k >= 3,
  with a_r,mid the overlap at the midpoint. This is tested directly (section 7).
- **H3 refined.** Single-core Hessian blocks are PSD (literature 1.3 item 6), so negative curvature
  needs a pair of cores. At a critical point the quadratic form is ||J delta||^2 - 2 sum_{u<v}
  <R, W[delta_u, delta_v]>. Negative curvature arises (a) from a pair of adjacent cores whose
  new-index directions are first-order invisible (J delta = 0) and see the residual at degree 2:
  this is the k = 2 route; or (b) from first-order-visible pairs whose residual coupling beats the
  curvature: the index-0 / piggyback route, with eigenvalue q^(K-2)(s_learned - s_new) for m = 2,
  negative only when the learned component is the smaller one (wrong order, H1b). For pattern
  teachers this gives the tree analogue of [Achour2024]'s "tightened" (derived): **a saddle visited in
  greedy order and without stars is non-strict iff no unlearned component can be added by raising the
  rank of a single bond** (all k(L, r) >= 3); it is strict if some k(L, r) = 2, if a wrong-order
  component can be corrected through two S-nodes, or if the piggyback graph has spectral radius
  ρ(S) > s_q (stars: theory_saddles.md v2 section 4.6; not tested here). Measured in section 7 via the
  sigma-scaling of lambda_min at plateau midpoints.

## 7. Results (round 2, after review notes/reviews/locality_round1.md)

Scope: orthogonal pattern teachers, d = 2, chi = 2 unless stated, whitened Gaussian inputs, exact population
loss. Dynamics: gradient flow, scipy LSODA with the exact-Hessian Jacobian (rtol 1e-9, atol 1e-14), event
times by root-finding (Radau fallback < 1% of runs), library v0 model, float64, teacher seed 1000, init seeds
0-7 (0-4 for the round-1 k >= 6 rows). Every table is restricted to the runs of its committed config
(`analysis.load_records(config=...)`), rotation runs are excluded from fits (counted separately), exponents
are per-seed least-squares fits with mean ± SE and the median over seeds. Conservation-law drift
max_e |Q_e(T) - Q_e(0)|: <= 1e-6 in every run (99.9% < 3.5e-8, median 1.2e-9), against Gram entries of order 1
after learning. Relative to the initial imbalance |Q(0)| ~ sigma^2 (fan-in small init) the drift is <= 1.6e-4 in
every k >= 4 run (where the imbalance matters: dead routes, 7.5), but up to 0.5 at k <= 3 and sigma <= 1e-3, where
sigma^2 is below the integrator's absolute error (those runs never stall, and their gaps agree with the reviewer's
independent dense/Radau integration to 1e-7-1e-9). With balanced init Q(0) = 0 to roundoff. Regenerate with
`finalize.sh <dir>`.

### 7.1 Two-component exponent table (step 1) — **empirical**

`results/twocomp_table.md` (115 rows), `twocomp_balanced_table.md`, `twocomp_chi4_table.md`,
`extensions_table.md` (k >= 6 over one decade, lambda2(K), k = 5 extension); figure
`results/twocomp_exponents.png`. Ranges over rows (per-seed mean; medians over seeds in brackets):

| k | rows (trees) | predicted | measured |
|---|---|---|---|
| 1 | binary4 ×2, binary8, star3 | no plateau | small init: t2 - t1 <= 8e-4, ρ <= 4e-4; saddle start: 0.63-0.65, ρ = 0.34 |
| 2 | mps4/6/8, midroot6 (7) | log, slope (1 - s2/s1)/λ2 (small init) or 1/λ2 (saddle start) | slopes 0.980-1.009 (pred. 1.000) and 2.002-2.071 (pred. 2.000); the fair comparison picks the log law in 14/14 rows (mps8 {3,4} small init nearly tied, 0.004 vs 0.005) |
| 3 | 22 + 20 rows on all 9 trees | 1 | 0.98-1.04 (medians 0.99-1.03); power law picked in 42/42 |
| 4 | mps4/6/8, star3, midroot6 (8) | 2 | small init 1.95-2.05; saddle start 1.48-2.39 (medians 1.86-2.06); balanced 1.94-2.02; chi = 4 1.99-2.03 |
| 5 | binary8 ×3, ht4 ×3 | 3 | 2.89-3.13, all protocols |
| 5 | ht8 {1,2} (saddle) | 3 | 2.74 ± 0.37 (median 2.96; 3.10 ± 0.11 without the seed that stalls at some sigma) |
| 5 | mps6 {1,3,5}, mps8 {1,5} | 3 | chi = 2, sigma in [0.03, 0.3]: 2.86-3.76 (fits include seeds that stall at some sigma; mps8 {1,5} small init on never-stalling seeds: 2.97 ± 0.05); sigma in [0.012, 0.03] (k = 5 extension): mps6 {1,3,5} 3.05 ± 0.06 (saddle start), 3.15 ± 0.23 (balanced); mps8 {1,5} 3.13 ± 0.25 (balanced, median 2.94) and **3.67 ± 0.20 (saddle start; 4 seeds, 0.4 decades, 10 of 24 runs stalled)**; chi = 4: 3.00 ± 0.04, 3.19 ± 0.15 (medians 2.99, 3.01) |
| 6 | binary8 {0,2,4}, ht4 {0,1,2} (**balanced init only**, 1.0 decade; 8 and 7 fitted seeds) | 4 | 4.007 ± 0.021, 4.061 ± 0.048 (local 4.00, 4.01) |
| 6 | mps8 {1,6} (balanced, 1.0 decade) | 4 | 4.51 ± 0.19 (local 4.26) |
| 6 | round-1 rows (0.8 decades, 2-5 seeds) | 4 | 3.4-5.1 (fan-in; e.g. binary8 {0,2,4} 3.68 ± 0.20, ht4 {0,1,2} 4.23 ± 0.10), not a measurement of the exponent |
| 7 | ht4 all sites (**balanced init only**, 1.0 decade, 8 seeds) | 5 | 4.957 ± 0.061 (median 5.01) |
| 7 | ht4/ht8 round-1 rows (0.6 decades, 3-5 seeds) | 5 | 5.3-6.3 (medians 5.0-5.7); ht8 {3,4}: 5.74 ± 1.06, median 5.01, one pinned seed (4.74 ± 0.44 without it); the sigma >= 0.45 points start below a1 = 1/2 |
| 8 | mps8 {0,7} only, sigma in [0.2, 0.6] | 6 | not measured (fan-in 2 seeds; balanced 5.54 ± 0.19, 2 seeds; chi = 4 7.29, 2 seeds; saddle start: no escaping run) |

- **Status (empirical)**: the exponent equals k - 2 for k = 1-5 on all 9 trees. Rows that deviate in
  [0.03, 0.3] are carried by seeds that stall at some sigma (saddle-start k = 4; ht8 {1,2}; mps8 {1,5} small init,
  7.5) or approach k - 2 at smaller sigma (mps6 {1,3,5}); the only k = 5 row that stays steep is **mps8 {1,5}
  from the saddle start** (3.67 ± 0.20 at sigma <= 0.03, 4 seeds, 0.4 decades, about 40% of runs stalled; 7.9).
  For k = 6-7 the exponent equals k - 2 on binary8 and ht4 **with gauge-balanced init only** over one decade
  (the fan-in rows give 3.68 and 4.23 for k = 6 over 0.8 decades), but not on the MPS (mps8 {1,6}: 4.5);
  k = 8 is not measured.
- The k = 2 law is distinguished from a power law by a fair comparison (same space and weights, 2-3 decades;
  analysis standard 5). Its slope carries the head start: small init (1 - s2/s1)/λ2, saddle start 1/λ2.
- **K-dependence of λ2** (s1 != 1; `configs/lambda2_K.yaml`, weights (2, 1), saddle start, 8 seeds):
  λ_min = -1.4142, -1.5874, -1.6818 for K = 4, 6, 8 (predicted s2 s1^((K-2)/K) = 1.4142, 1.5874, 1.6818);
  log slopes 0.712 ± 0.007, 0.649 ± 0.016, 0.595 ± 0.003 (predicted 1/λ2 = 0.707, 0.630, 0.595). One of
  the 168 runs (mps8, seed 7, sigma = 1e-4) was lost to the runner's start-up consistency check (compiled vs
  library velocity at the exact saddle, where the field is roundoff); it is counted as an error and not rerun
  (the check now has an absolute floor).
- The same-teacher contrast on N = 8, S = {3,4} (log / k = 5 / k = 7 on mps8 / binary8 / ht8) mixes protocols
  (binary8 small init, mps8 and ht8 saddle start); the ht8 value (k = 7) is consistent with 5 but does not
  discriminate k = 6, 7, 8.

### 7.2 Saddle type (H3) — **empirical**

`results/twocomp_hessian.png`; Hessians recomputed at the stored plateau midpoints (`hess_recompute.py`,
`results/hessians.csv`) with the fixed `ttn.diagnostics.project_out_gauge` (rtol 1e-3, commit 6c42d85) and
raw (unprojected) eigenvalues side by side. Classification by the primary rule of notes/diagnostics.md
(sweep-level sigma-scaling of λ_min); the counts of negative eigenvalues (fixed 1e-3 threshold) are a
single-sigma fallback only.
- k = 2: saddle start λ_min = -0.5000 (projected and raw, every row; predicted -λ2 = -0.5), independent of
  sigma; s2 = 0.25 / 0.75: -0.2500 / -0.7500. Small init (raw values): -0.498 to -0.499 at sigma <= 1e-3;
  -0.437 for mps8 {3,4} at sigma = 0.008 and -0.581 for s2 = 0.75, where the plateau is too short to reach the
  saddle (the fixed projection is still biased at these sigma >= 1e-3 states: -0.411, -0.502; raw is quoted).
  One negative eigenvalue per active route, also at chi = 4, 8 (theory R10). Round-1 values -0.44..-0.46
  (saddle start) were biased by the old projection (review issue 3).
- k >= 3: λ_min ∝ sigma^x, full ranges over all rows (raw where it differs): k = 3: 0.81-1.06 (raw 0.92-1.06);
  k = 4: 1.64-2.16; k = 5: 2.82-3.76; k = 6: 4.00, 4.04 (balanced, binary8 / ht4, one decade), 4.27 (mps8 {1,6},
  balanced), 3.19-5.23 (round-1 rows over 0.8 decades); k = 7: 5.01 (balanced ht4) up to 6.5 (round-1 rows over
  0.6 decades). So x grows with k and λ_min -> 0: no sigma-independent negative curvature; x ≈ k - 2 within
  0.2 for k = 3-5 and the balanced one-decade k = 6-7 rows. Raw and projected exponents differ by up to 0.11
  (mps6 {2,3,4} small init: 0.81 projected, 0.92 raw). Exception: balanced-init mps8 {2,5} and
  {1,5}, whose midpoint λ_min is ~1e-13-1e-10 (exponents 6-9): there the minimum-speed point falls where the
  route amplitudes are tiny (a sign-correction phase, theory_saddles.md 8), so the midpoint Hessian is not
  representative.

### 7.3 Dormant amplitudes and single-run k — **empirical**

- Dormant-amplitude test (per-seed fits): remaining time from the plateau midpoint vs the overlap a_mid
  there: k = 2 slope vs ln(1/a_mid) 0.999-1.004 (saddle start) and 1.010-1.043 (small init), predicted
  1/(2λ2) = 1; k = 3, 4, 5, 7 exponents -0.29..-0.41, -0.39..-0.52, -0.51..-0.61, -0.69..-0.79, predicted
  -(k-2)/k = -0.33, -0.50, -0.60, -0.71. This is partly circular: it follows from the growth law of the
  route mode alone, so it confirms that plateau lengths are set by the dormant amplitude, not the degree.
- Growth exponent γ = d log ȧ / d log a in a in [1e-4, 1e-2] (theory P4): medians per row give
  k = 2/(2 - γ) = 2.00 (k = 2, saddle start), 3.0, 4.0, 5.1-5.3, 6.0, 7.0-7.7, 7.7-7.9. **Validity range**:
  saddle start, or small init when the plateau is long enough that the window lies after the escape
  transient; it fails for small-init k = 2 on mps6/mps8 (k estimate 2.4-3.3, the plateau is shorter than the
  transient), is biased high for k >= 5 (ht8: 5.3 for k = 5, 7.7 for k = 7), and cannot separate k = 7 from 8
  (dk/dγ = k^2/2; per-run 5-95% ranges ±1 at k = 7).

### 7.4 Several components: same teacher, different saddle sequences (step 3) — **empirical**

Predicted sequences: route rule at rank-constrained optima (section 5; `results/route_sequences.json`).
Measured (`levels.py`, `results/levels_summary.csv`; figures `results/trees_TA.png`, `trees_B4.png`,
`trees_B4_ht_balanced.png`, `trees_E8.png`), runs pooled by event order (standard 3), per-seed fits:
- **detected plateaus**: `ttn.diagnostics.find_plateaus` on L(t) (linear time, separation ratio > 3),
  matched to a predicted level when their levels agree within 5% (`detected_frac` = fraction of runs with a
  matching plateau; replaces round 1's meaningless "visited" fraction); `p_det` = exponent of the matched
  plateau's duration vs 1/sigma;
- **fair law label** from the occupation time within 2% of the level (log law vs power law, same space;
  "log" only where the log law has the smaller residual over >= 4 sigmas).

| teacher | tree (protocol, sigma) | predicted k* per level | p_det (detected fraction) | law (occupation, fair) |
|---|---|---|---|---|
| TA {0000,1100,1110} | mps4 (small, 0.2-1e-5) | 2, 2 | too short to detect (0.06), 0.14 (0.49) | power (0.22 vs 0.53; law undetermined) / log |
| | binary4 (small) | 3 | 1.02 (1.00) | power |
| | ht4 (small; balanced) | 3, 4 | 1.09, 2.08 (0.93, 1.00); 1.13, 2.04 | power |
| B4 {0000,0110,1001} | mps4 (small, 0.2-1e-3) | 2 (0.3 component first), 4 | not detected (0.00), 2.02 (1.00) | log (0.22 vs 0.25) / power |
| | binary4 (small) | 3, 3 | 1.05, 1.03 (0.92, 1.00) | power |
| | ht4 (small) | 5, 5 | 3.09, **4.73 ± 0.90** (0.96, 0.96; 4 fitted seeds; 23 of 48 runs stall before the first event) | power |
| | ht4 (balanced) | 5, 5 | 3.00, 3.17 (0.88, 1.00) | power |
| E8 (N = 8) | mps8 (small) | 2, 6 | not detected (0.00), 4.05 (1.00) | power (0.024 vs 0.112), power |
| | binary8 (small) | 5, 5 | 3.06, 2.97 (1.00) | power |
| TC (4 components) | mps8 (saddle, 0.1-1e-4) | 2, 2, 2, 4, 2 | 0.17, 0.16, 0.18, 2.02, – (0.68, 0.57, 0.81, 1.00, 0.00) | log, log, log, power, log |
| | binary8 (small) | 3, 3, 3 | 1.12, 1.10, 1.08 (0.96, 0.35, 0.65) | power |
| TD (N = 8) | mps8 (saddle, 0.1-1e-4) | 2, 2, 3 | 0.15, 0.10, **1.00** (0.51, 0.06, 1.00) | log, **power** (0.16 vs 0.39), power |
| | mps8 (small) | 2, 2, 3 | –, –, 1.05 (0.06, 0.03, 1.00) | power, power, power |
| N4 {0000,1100,1111} | mps4 (small, 0.2-1e-5) | 2, 2, 2 | 0.09, 0.15, 0.16 (0.09, 0.46, 0.74) | power, log, log |
| | binary4 (small) | 3 | 1.04 (0.96) | power |
| | ht4 (small; balanced) | 3, 3, 3 | 1.12, 1.15, 1.11 (0.89-1.00); 1.12, 1.15, 1.09 | power |

- **Every k* >= 3 level but one** is detected as a plateau in 88-100% of runs (35-65% for some k* = 3 levels whose
  level differs from the next by < 10%, e.g. TB ht4 levels 1-2, TC binary8 level 1) with exponent
  k* - 2 within 0.2 (1.02-1.19 for k* = 3; 2.02-2.08 for k* = 4; 2.97-3.17 for k* = 5; 4.05 for k* = 6).
  The exception is B4 on ht4 with fan-in init: its second k* = 5 level gives 4.73 ± 0.90 (4 fitted seeds,
  23 of 48 runs stall before the first event), against 3.17 ± 0.10 with balanced init (8 seeds); like the
  saddle-start k = 4 rows of 7.5, this fan-in row is carried by few seeds in a case with many dead routes.
  Round 1's band-occupation exponents for k* = 3 (1.2-1.6) were biased by the approach to the level.
- **k* = 2 levels** are short; where measured over 3 decades from the saddle start (TC levels 0-2 and 4,
  TD level 0) the fair comparison picks the log law; TD level 1 (a partial step, 12/55 durations
  unresolved) and the small-init MPS8 first levels prefer a power law < 1, i.e. their law is not
  determined; round 1's "log" labels there are withdrawn.
- **Route rule vs alternatives**: TD's last level (1.012 ± 0.010, 8 seeds, saddle start; detected-plateau
  exponent 1.00) and TC's k* = 4 level (2.018 ± 0.017) separate the route rule from the literal H2 rule and
  from the exact bond-rank rule; TA separates the literal rule only. B4 on mps4: the 0.3 component is
  learned before the 0.6 one in every escaping run.

### 7.5 Stalled runs: counts, dead vs slow, theory R12 — **empirical**

A run is "stalled" when, after a component has been learned, the gradient stays below 1e-12 for 5,000
consecutive evaluations (the loss sits at the partial fit, e.g. L = 0.125 = s2^2/2).

- **Counts** (`results/run_classes_by_sweep.csv`, written by `stall_tables.py`; runs of the committed configs):
  fan-in chi = 2: **398 of 5,682** two-component runs, all with k >= 4 (none of 5,112 runs with k <= 3 in the
  two-component and robustness sweeps); gauge-balanced init (same k >= 4 cases): 16 of 1,008 (fan-in on the same
  cases and sigma grids: 169 of 1,008, a 10x reduction), all on MPS routes (mps8 {2,5}, {1,5}, {1,6}, {0,7};
  mps6 {1,3,5}); chi = 4: **106 of 1,005** (7.10); k = 5 extension (sigma <= 0.03): 34 of 96; k >= 6 extension
  (balanced): 10 of 256 (mps8 {1,6}); lambda2(K): 0 of 168 (one error, 7.1).
- **Sigma-dependent dropout** (`results/stall_dropout_{twocomp,twocomp_balanced,twocomp_chi4}.csv`): a stalling
  seed does not stall at every sigma in 20 of 33 fan-in rows with stalls (49 seeds stall at every sigma, 34 at
  some sigmas only); all 7 stalling balanced-init seeds are of the second kind; at chi = 4, 3 of 8 rows (17 seeds
  at every sigma, 3 at some). Fits use the seeds that escape at >= 3 sigmas, so partially stalling seeds enter
  with fewer sigmas. Restricting each row to seeds that never stall (`results/stall_dropout_bias.csv`, 46 rows
  with stalls) leaves the exponent unchanged in 27 rows; it moves it by >= 0.1 in **9 fan-in rows** (p_all ->
  p_never-stalled): mps6 {0,3} saddle 2.39 -> 1.92, midroot6 {0,3} saddle 1.49 -> 1.81, ht8 {1,2} saddle
  2.74 -> 3.10, ht8 {3,4} saddle 5.74 -> 4.74, mps6 {0,5} small 4.54 -> 5.01, mps6 {0,5} saddle 3.39 -> 2.62,
  mps8 {1,5} small 3.27 -> 2.98, mps6 {0,...,5} saddle 3.49 -> 3.06, midroot6 {0,5} saddle 5.12 -> 3.74 (shift
  up to 1.38; the last four keep 1-3 seeds), and in 3 other rows (balanced mps8 {1,5} 3.60 -> 3.34; chi = 4
  mps8 {0,7} 7.29 -> 6.61 and ht4 {0,1,2} 4.05 -> 4.19). The off-target saddle-start k = 4 and ht8 rows are
  carried by seeds that stall at some sigma.
- **Dead vs slow** (`configs/stall_diagnosis.yaml`: 22 (case, sigma) pairs covering stalled runs and their
  escaped controls, 170 runs, stall patience 300,000 evaluations (60x), trajectories saved; `r12.py`,
  `results/r12.csv`): every fan-in and saddle-start run that the 5,000-evaluation detector had flagged (37) is
  dead: no escape within the longer budget (up to t ~ 1e10) and route-end amplitudes at roundoff (1e-17 to
  1e-13). With balanced init the flagged runs are not dead: 6 of 10 escaped very late (gaps 2e9-5e10, about 1e5 x
  the normal plateau) and 4 did not escape within the budget: three mps8 {1,5} runs are pinned (route ends
  3e-10, 6e-9 and 4e-8 at t = 1e11-5e11) and one mps8 {2,5} run (seed 0, sigma = 0.01) has route ends of 0.72 at
  t = 5.5e12, i.e. it was in the middle of a very late escape when the budget ended (`r12.py` labels it
  "pinned" only because it is neither escaped nor at roundoff). So "stalled" = dead under fan-in init and
  saddle start, but = pinned or extremely slow under balanced init.
- **R12's criterion** (theory_saddles.md 8: a route end decays iff c = Q_end + |y_route|^2 < 0, evaluated here
  at the plateau start a1 = 0.99; the route dies iff every end has c < 0): over the 122 fan-in and saddle-start
  runs, 30 of 55 runs with both ends unfavourable died, against 7 of 67 with a favourable end; 30 of the 37 dead
  runs had both ends unfavourable. The criterion is strongly associated with death (55% vs 10%) but is neither
  sufficient (25 predicted-dead runs escaped, e.g. 7 of 9 on ht4 {1,2} from the saddle start) nor necessary.
  **Seven dead runs had a live route end** (c > 0 at one end, at the plateau start): mps6 {0,3} small chi = 2
  (seed 5, sigma = 0.11, 0.2), mps8 {1,5} small chi = 2 (seed 1, sigma = 0.19, 0.3), mps6 {0,5} small chi = 4
  (seed 2, sigma = 0.35, 0.5) and mps6 {1,3,5} saddle chi = 2 (seed 4, sigma = 0.19; both ends live, c/sigma^2 =
  0.86, 0.64). By theory_saddles.md P14(v) ("a dead stall with a live route end") these refute the picture
  as stated, unless c evaluated once at a1 = 0.99 is not the quantity that decides (c can change sign later in
  the plateau, since y_route is not conserved); that check (c along the plateau) is not done. For mps4 {0,3} at
  chi = 2 the criterion classifies 14 of 16 runs correctly (all 10 dead runs predicted; 2 false alarms with
  weakly negative c).
- **Conflicts with theory_saddles.md** (to be resolved by the theory side):
  (i) R12 says exactly balanced init removes the trap and that with Q = 0 the leaf column does not shrink. We find
  no dead route with balanced init (consistent), but route ends do shrink to 1e-10-1e-8 on MPS8 routes and the
  escape is delayed by ~1e5 or beyond the budget (inconsistent with "no shrink").
  (ii) P12's heuristics (about 0.5 dead per two-ended HT route under fan-in init; MPS with chi = d stalling less
  often) are contradicted: ht4 k = 5 routes: 0 of 144 fan-in small-init runs stalled, against 30 of 48 on mps4
  {0,3}, 17 of 48 on mps6 {0,3}; stalls concentrate on MPS routes, especially those ending at the chain ends.
  (iii) The joint criterion is a strong but not a deterministic predictor (above), and P14(v) is met by 7 dead
  runs with a live end at the plateau start.

### 7.6 Bond spectra at events (step 4) — **empirical**

Redone (review issue 2; `bond_events2.py`): ranks right after each event (at the a = 0.9 crossing), not at the
end of the run; effective rank = #{s_k^e > f x teacher s_k^{e,*}}, f = 0.05 (notes/diagnostics.md), with
f = 0.03 and 0.1 as sensitivity.
- **Two components** (`results/bond_events2_twocomp.csv`, 5,130 events; 4,447 with a resolved preceding plateau,
  ρ >= 3; the other 683 are mostly small-init k = 2 runs whose "before" state is already part of the escape).
  What the data show is a **leakage bound** plus **single-route consistency**, not a test that could have
  picked out a wrong set of edges:
  - *Leakage bound*: on every non-Steiner edge the second singular value stays at the init-noise level from the
    plateau midpoint to the end of the run: at most **4.3 sigma^2 x s2** over all 5,130 events (4.0 over the
    resolved ones), scaling as sigma^2.07; < 3e-3 x s2 for sigma <= 0.03. This is the substantive result: the
    new component enters through the Steiner edges only, to O(sigma^2).
  - *Single-route consistency*: the edges whose effective rank (f = 0.05) is higher at a = 0.9 than at the
    plateau midpoint equal the Steiner edges in 99.75% of resolved events (99.6% / 99.9% for f = 0.03 / 0.1;
    86.8% of the unresolved ones). With a threshold at f x teacher value and non-Steiner edges bounded as
    above, a non-Steiner rise cannot register, so this number only checks that every Steiner edge rises; it is
    not a prediction success rate.
  - *Crossing-time spread* (spread of the Steiner edges' crossing times / event width, k >= 3): median 5e-5,
    95th percentile 0.25. It tests that the Steiner edges rise as one route (one event), not the timing of the
    event, and it grows with sigma (∝ sigma^1.94; median 0 below sigma = 0.01, where 63% of the events cross
    within one saved time step, 0.13 at sigma >= 0.2), as expected when the init noise is a larger part of each edge's spectrum.
  (The reviewer's dense-SVD check: same conclusion.)
- **Several components** (`results/bond_profiles_multicomp.csv`): at every predicted saddle the observed rank
  profile of all bonds (middle of the time spent within 2% of the predicted level) equals the route rule's
  predicted profile in 93.9% of 3,114 resolved (run, level) pairs: k* = 3: 99.6-100%; k* = 4: 92% (fan-in),
  100% (balanced, saddle start); k* = 5: 95% / 100%; k* = 6: 100%; k* = 2: 97% from the saddle start, 86% with
  small init (short log levels on mps8, where the level band is crossed during the next escape).
  Per-event comparisons for several components (`bond_events2_multicomp_events.csv`) are less clean because one
  component event can span a partial step and its completion; the per-saddle profile is the right test.

### 7.7 Robustness (step 5) — **empirical**

N = 4, 8 seeds, small init unless stated (`results/robust_table.md`; rotation runs excluded).
- Weight ratio: k = 2 log slope 3.023 ± 0.010 (s2 = 0.25; predicted 3.000; saddle start 4.04 ± 0.05 vs 4.00)
  and 0.367 ± 0.025 (s2 = 0.75; predicted 0.333; saddle start 1.345 ± 0.016 vs 1.333). λ_min from the saddle
  start: -0.2500 and -0.7500 exactly (= -s2). k = 3 exponents 0.99-1.05, k = 4 1.87-1.99.
- Spare dimensions (chi = 4 at d = 2; d = 4 with chi = 2, 4, 8): k = 2 slopes 0.999-1.009; k = 3 0.99-1.05;
  k = 4 1.90-2.01. One negative eigenvalue per active route at k = 2 plateau points (also at chi = 8).

### 7.8 Gauge-balanced init — **empirical**

`results/twocomp_balanced_table.md` (k >= 4 cases, Q_e = 0 at t = 0, same W_0): exponents k = 4: 1.94-2.02;
k = 5: 2.95-3.06 on binary8 and ht4, 3.33 and 3.60 on mps6 {1,3,5} and mps8 {1,5} (sigma in [0.03, 0.3]; at
sigma <= 0.03: 3.15 ± 0.23 and 3.13 ± 0.25, medians 2.98 and 2.94); k = 6: 3.49-4.54 over
0.8 decades (with one decade, 7.1: 4.01, 4.06 on binary8, ht4; 4.51 on mps8); k = 7 (ht4): 4.98 ± 0.08.
Stalls: **16 of 1,008** runs (fan-in, same cases and sigma grids: 169 of 1,008), about a 10x reduction, not removal; all 16
on MPS8 and MPS6 routes (mps8 {2,5} 2/48, {1,5} 4/48, {1,6} 5/30, {0,7} 4/30; mps6 {1,3,5} 1/48), and all
sigma-dependent (a stalling seed stalls at some sigmas only). k = 5 extension (sigma <= 0.03): 34 of 96
(balanced and saddle start) stall on the same two MPS routes.

### 7.9 MPS routes with steep exponents — **empirical; mechanism open**

At chi = 2 and sigma in [0.03, 0.3] the MPS k = 5 rows exceed 3 (mps6 {1,3,5} small 3.76 ± 0.27, balanced
3.33 ± 0.20; mps8 {1,5} small 3.27 ± 0.30, saddle start 3.39 ± 0.12, balanced 3.60 ± 0.26). Most of this
does not persist (`results/extensions_table.md`, k = 5 extension, sigma = 0.012-0.03, 0.4 decades;
`results/stall_dropout_bias.csv`):
- mps6 {1,3,5}: 3.05 ± 0.06 (saddle start, 5 seeds) and 3.15 ± 0.23 (balanced, median 2.98) at sigma <= 0.03;
- mps8 {1,5} balanced: 3.13 ± 0.25 (median 2.94) at sigma <= 0.03;
- mps8 {1,5} small init: 2.97 ± 0.05 on the 5 seeds that never stall (3.27 with all seeds);
- **only mps8 {1,5} from the saddle start stays steep**: 3.67 ± 0.20 (median 3.83) at sigma <= 0.03, from 4
  fitted seeds over 0.4 decades, with 10 of 24 runs stalled (about 40%).
The fitted seeds of that row escape at every sigma, so it meets theory_saddles.md P14(ii) (a per-seed exponent
off by more than 0.2 at sigma <= 0.05 in runs that did not stall), but on thin evidence (4 seeds, 0.4 decades,
in a case where 40% of the runs die). Its Hessian exponent agrees (3.74 projected, 3.75 raw); the balanced
mps8 {1,5} Hessian fit (x = -5.4) is meaningless because its midpoint states are pinned states at t = 1e9 to
5e12, not plateau midpoints. At chi = 4 both mps rows give 3 (3.00 ± 0.04, 3.19 ± 0.15, medians 2.99, 3.01);
the mps6 whole-chain k = 6 routes stay steep at chi = 4 (5.0-5.1).
- **Route-end amplitudes** (`route_amps.py`, `results/route_amps*.csv`): at the plateau midpoint the route
  ends scale as sigma^α with α = 1.24-1.91 in the steep MPS rows (interior entries ∝ sigma), and they shrink
  during the plateau (median ratio midpoint / first event 0.53-0.78, single runs to 1e-4-1e-7), against
  0.85-1.03 in rows with on-target exponents. But α > 1 also occurs in rows with on-target exponents
  (binary8 {0,7} balanced: α = 1.34, p = 2.99; binary8 saddle 1.28; ht4 {1,2} saddle 1.31; mps4 k = 3 saddle
  1.29), so the steep exponents are **associated with** shrinking route ends, not shown to be caused by them.
- **Log-correction test** (review issue 5; `logcorr.py`, `results/logcorr.csv`): if small ends catch up
  through the conservation laws, T ~ sigma^-(k-2) [(α - 1) ln(1/sigma) + 1/(k-2)] / λ (a log correction, not a
  power shift). Per seed, over the measured sigma ranges, this form fits the steep rows much worse than a free
  power law (median RMS log residual 0.25-0.62 vs 0.05-0.12, both with a = α - 1 fixed and with a free a, which
  runs off to ~1e3), and it is indistinguishable from the power law in the on-target rows (a ≈ 0). So the
  data prefer a steeper power over this log correction; the end amplitudes do not catch up in these runs,
  which is what one expects if the ends are pinned (decaying) rather than merely small.
- The pilot's conjecture sigma^-(2k-6) (α = 2 at both ends) is not what is measured: α is seed-dependent and
  between 1 and 2.

### 7.10 Spare bond channels (chi = 4) — **empirical** (theory_regimes R10)

`results/twocomp_chi4_table.md` (24 cases with k >= 4, small init, d = 2, chi = 4): exponents k = 4: 1.99-2.03;
k = 5: 2.96-3.19 on every tree; k = 6: 3.91-4.17 (binary8, ht4, mps8) but 5.0-5.1 for the mps6 whole-chain
routes; k = 7 (ht4) 5.07 ± 0.39; k = 8 (mps8 {0,7}) 7.29 ± 0.68 (2 seeds, 0.5 decades).

Stalls (`results/stall_capped_ends.csv`, written by `stall_tables.py`): **106 of 1,005** runs at chi = 4,
against **169 of 1,008** at chi = 2 on the same 24 cases and sigma grids (round 1 said 75; round 2 said
103 of 1,002 from a run-class file written by hand before three slow runs, all stalled, had finished; the
exponent tables already included them). So chi = 4
does **not** remove most dead routes; it moves them:
- routes with at least one end on an uncapped bond: 79 stalls at chi = 2 (7 cases: midroot6 {0,3}, mps6
  {0,3}, {1,3,5}, {1,4}, mps8 {1,5}, {1,6}, {2,5}) -> 0 at chi = 4 (every one of the 13 such cases escapes);
- routes whose two ends both sit on bonds capped at d (MPS chain ends, HT leaves): 90 -> **106 stalls**; the
  stall count rises in 7 of the 8 such cases with stalls (ht4 {0,1,2} 0 -> 1/30, ht4 all 8 -> 12/30, midroot6
  {0,5} 22 -> 30/30, mps4 all 0 -> 12/48, mps6 all 7 -> 10/30, mps6 {0,5} 7 -> 18/30, mps8 {0,7} 16/30 ->
  17/27) and falls only for mps4 {0,3} (30 -> 6/48).
This conflicts with theory_regimes.md R10 ("spare channels (chi >= 2R) do [fix the freeze] in the runs so
far") for routes with both ends capped; it agrees with it for routes with an uncapped end. A mechanism for
the rise is not established (extra bond directions change the fan-in statistics of Q at the capped ends).
midroot6 and mps6 (the same graph) share exponents but not trap rates (midroot6 {0,5} 30/30 vs mps6 {0,5}
18/30 at chi = 4): rooting changes the fan-in init statistics, hence Q.

## 8. Verdict (round 3)

Scope of every statement: orthogonal pattern teachers, whitened inputs, exact population loss, gradient
flow, d = 2 and chi = 2 unless stated.

**H2, two-component teachers: confirmed for k = 1-5** (empirical; derived in section 2; the Steiner
identity is proven, theory_saddles.md): no plateau for k = 1; a log law for k = 2 with the head-start slope
(1 - s2/s1)/λ2 from small init and 1/λ2 from the saddle (fair comparison over 2-3 decades); exponent k - 2
for k = 3 (42 rows, 0.98-1.04), k = 4 (1.95-2.05 with small init, balanced init and chi = 4) and k = 5 (2.9-3.1
on binary8 and ht4). Exceptions: (i) the MPS k = 5 rows at chi = 2 exceed 3 for sigma in [0.03, 0.3]
(3.3-3.8), but at sigma <= 0.03 or on never-stalling seeds they are 2.97-3.15, except **mps8 {1,5} from the
saddle start, 3.67 ± 0.20** (4 seeds, 0.4 decades, about 40% of runs stalled; theory_saddles.md P14(ii) met on
thin evidence); at chi = 4 both MPS rows give 3; (ii) saddle-start k = 4 rows (1.48-2.39), where the fits
include seeds that stall at some sigma; restricted to never-stalling seeds they are 1.81-2.19. **For k = 6-7**
the exponent equals k - 2 on binary8 and ht4 **with gauge-balanced init only**, over one decade (4.01, 4.06,
4.96; 7-8 seeds); the fan-in k = 6 rows give 3.68 and 4.23 (0.8 decades), the MPS gives 4.51 ± 0.19 (mps8
{1,6}, balanced), and the round-1 k >= 6 rows (0.5-0.8 decades, 2-5 seeds) only show that the exponent grows
with k. **k = 8** is not measured. λ2 = s2 s1^((K-2)/K) is confirmed including its
K-dependence (s1 = 2, K = 4, 6, 8: λ_min = -1.4142, -1.5874, -1.6818 exactly; log slopes match). Refinement
of H2: plateau lengths are set by the dormant route amplitude entering the plateau, which equals sigma times
an O(1) factor only when nothing rescales the route (head start for k = 2; shrinking route ends on some MPS
routes).

**H2 with several learned components: the literal statement fails; the route rule at rank-constrained
optima holds where tested** (derived at leading order; empirical): every k* >= 3 plateau of TA, TB, N4, B4,
E8, TC, TD is detected with exponent k* - 2 (within 0.2), except B4's second k* = 5 level on ht4 with fan-in
init (4.73 ± 0.90, 4 seeds, half the runs dead; 3.17 with balanced init), and the bond-rank profile at each predicted saddle
matches the prediction (99.6-100% for k* = 3, 92-100% for k* = 4-6). TD (last plateau 1.012 ± 0.010, 8 seeds)
and TC (k* = 4 plateau 2.018 ± 0.017) separate the route rule from both the literal rule and the exact
bond-rank rule; TA separates it from the literal rule only. The k* = 2 levels are log-time where measured
over 3 decades from the saddle start; elsewhere their law is not determined.

**Same teacher, different sequences: demonstrated** (`results/trees_TA.png`, `trees_B4*.png`,
`trees_E8.png`): TA is learned by mps4 through two short plateaus (the second log-time; for the first the fair
comparison favours a power < 1, 0.22 vs 0.53 RMS, so its law is undetermined), by binary4 through one sigma^-1
plateau, by ht4 through sigma^-1 and sigma^-2 plateaus; on mps4 the 0.3 component of B4 is learned before its
0.6 component.

**Bond spectra**: a leakage bound: non-Steiner edges stay at the init-noise level, second singular value
<= 4.3 sigma^2 x s2 (slope 2.07 in sigma) through every two-component event; plus single-route consistency: all
Steiner edges gain rank at the event (99.75% of resolved events, a check that cannot register non-Steiner rises)
and they cross together (median spread 5e-5 event widths, growing as sigma^1.94). At the predicted saddles of
multi-component runs the rank profile matches in 94% of (run, level) pairs (99-100% for k* = 3).

**H3 (strict iff k = 2) for saddles visited in greedy order: supported**, by the primary rule (sigma-scaling
of λ_min, raw quoted where the projection is biased): k = 2: λ_min = -λ2 exactly (-0.5000; -0.2500 / -0.7500
for s2 = 0.25 / 0.75), sigma-independent; k >= 3: λ_min ∝ sigma^x -> 0 with x growing with k (full ranges:
k = 3 0.81-1.06, k = 4 1.64-2.16, k = 5 2.82-3.76, k = 6 4.00-4.27 over one decade, k = 7 5.01 to 6.5); x is
within 0.2 of k - 2 for k = 3-5 and the balanced one-decade k = 6-7 rows. Not tested: wrong-order and star
saddles, which theory_saddles.md shows can be strict at k* >= 3.

**Dead routes** (empirical; 7.5, 7.10): with fan-in init at chi = 2, 398 of 5,682 two-component runs stall, all
with k >= 4 (none of 5,112 runs with k <= 3); the stalls checked with a 60x longer budget are dead (route ends at
roundoff). Gauge-balanced init leaves no dead route in the checked sample but 16 of 1,008 runs stall at the
default patience (pinned or extremely slow escapes on MPS routes). chi = 4 does not remove most stalls: 106 of
1,005 against 169 of 1,008 at chi = 2 on the same cases; it removes them on routes with an uncapped end (79 -> 0)
and they rise on routes with both ends capped at d (90 -> 106), against theory_regimes.md R10. Stalls are partly
sigma-dependent, which moves 9 fan-in rows by 0.3-1.4. Theory R12's joint criterion is associated with death but
is not a deterministic predictor, and 7 dead runs had a live route end at the plateau start (theory_saddles.md
P14(v)); balanced-init end shrinkage and the HT-vs-MPS stall rates also conflict with theory_saddles.md (7.5).

## 9. Changes after review round 1 (notes/reviews/locality_round1.md)

Round-2 values as of that round; section 10 lists what round 3 changed.

| claim (round 1) | round 2 |
|---|---|
| H2 confirmed for k = 1..8, ±0.05 for k <= 4 | confirmed for k = 1-5 except MPS k = 5 at chi = 2 and saddle-start k = 4 rows; ±0.05 holds for small-init, balanced and chi = 4 k = 4 rows only; k = 6-7 pinned to k - 2 only on binary8/ht4 with balanced init over one decade (new runs); k = 8 not measured |
| ht8 k = 5: 3.1-3.8 | 2.74 ± 0.37 (median 2.96; 3.10 ± 0.11 without the seed that stalls at some sigma) |
| ht8 {3,4}: p = 5.7 | 5.74 ± 1.06, median 5.01, one pinned seed (4.74 ± 0.44 without it); large-sigma points start below a1 = 1/2; does not discriminate k = 6-8 |
| rank rises on the Steiner edges in 98.7% of events | the old statistic compared midpoint with end-of-run ranks (near-tautological); redone: ranks right after each event, 99.75% of resolved events, crossing-time spread, leakage, and per-saddle profiles for several components (7.6) |
| k = 2 λ_min = -0.49..-0.50 (projected) | saddle start was biased (-0.44..-0.46) by the old gauge projection; recomputed with the fixed projection and raw: -0.5000 exactly (7.2) |
| balanced init: 2/1,008 stalls | 16/1,008 (10x fewer than fan-in, not 85x); all on MPS routes |
| chi = 4: 75/1,002 stalls | 103/1,002 (a zero-escape case had dropped out of the table; `make_figures` now keeps such cases) |
| fan-in: 376/5,732 stalls | 397/5,682 (runs restricted to the committed config; the extra sigma points of three k = 7-8 rows were dropped) |
| a stalling seed stalls at every sigma | false in 20 of 33 fan-in rows (34 seeds stall at some sigma only); effect on fits quantified (7.5, `results/stall_dropout_bias.csv`) |
| "stalled" = dead | true for fan-in and saddle-start stalls (60x longer runs: route ends at roundoff); balanced-init stalls are pinned slow escapes; R12's criterion is associated with death but not deterministic; conflicts with R12 / P12 stated (7.5) |
| steep MPS exponents come from shrinking route ends | associated with, not shown to be caused by; the reviewer's log-correction form is rejected by the data; P14(ii) met at chi = 2 (7.9) |
| TD last level 1.02 (7 seeds) | 1.012 ± 0.010 (8 seeds) |
| "visited fraction" of predicted levels | replaced by detected plateaus (`find_plateaus`) matched to the predicted levels, and their duration exponents (7.4) |
| "log" labels for k* = 2 levels | only where the fair comparison picks the log law (TC, TD level 0 from the saddle start; N4, TA, TB mps4 level 1); TD level 1 and the small-init mps8 first levels prefer a power law < 1 (law undetermined) |
| single-run k estimate | validity range stated (saddle start, or small init with the window after the transient) (7.3) |
| rotation runs pooled into fits | excluded (counted per row) |
| k = 1 "ρ < 3" (hard-coded) | computed: ρ <= 4e-4 (small init), 0.34 (saddle start) |
| λ2's K-dependence untested | tested with s1 = 2 on mps4/6/8 (7.1) |

## 10. Changes after the round-2 verification (notes/reviews/locality_round2_verification.md)

| item | round 2 | round 3 |
|---|---|---|
| (a) MPS k = 5 | "MPS k = 5 rows at chi = 2: 3.3-3.8, P14(ii) met; mps8 {1,5} 3.72 / 3.68 at sigma <= 0.03" | from the committed tables: mps6 {1,3,5} 3.05 / 3.15 and mps8 {1,5} balanced 3.13 (median 2.94) at sigma <= 0.03; mps8 {1,5} small 2.97 on never-stalling seeds; **only mps8 {1,5} from the saddle start stays steep, 3.67 ± 0.20** (4 seeds, 0.4 decades, ~40% stalled); P14 is in theory_saddles.md, not theory_regimes (7.1, 7.9) |
| (b) chi = 4 | "stalls remain only for routes with both ends capped" (103/1,002) | 106/1,005 vs 169/1,008 at chi = 2 on the same cases: not "most removed"; uncapped-end routes 79 -> 0, both-capped routes 90 -> 106 (rise in 7 of 8 cases); conflicts with theory_regimes R10 for capped routes (7.10) |
| (c) bond spectra | "rank rises on exactly the predicted edges in 99.75% of events" | leakage bound (non-Steiner <= 4.3 sigma^2 s2, slope 2.07) plus single-route consistency; the 99.75% cannot register non-Steiner rises; the spread tests one route, not timing, and grows as sigma^1.94 (7.6) |
| (d) stall bookkeeping | fan-in 397/5,682; chi = 4 103/1,002; 5 dropout rows; 4 balanced "pinned" runs with ends 3e-10-4e-8 | 398/5,682 and 106/1,005 (the round-2 run-class file was written by hand before 5 of the 11 dropped slow runs finished; now generated); 9 fan-in dropout rows moved by >= 0.1 (up to 1.38) plus 3 others; 3 pinned runs (3e-10, 6e-9, 4e-8) and one balanced run mid-escape (ends 0.72); **7 dead runs with a live route end flagged (theory_saddles P14(v))** (7.5) |
| (e) k = 6-7 and Hessians | "k = 6-7 = k - 2 on binary8/ht4"; x = 0.93-1.04, 1.94-2.05, 2.87-3.44, 4.00-4.04, 5.01; "raw and projected agree to 0.01" | **balanced init only** (fan-in k = 6: 3.68, 4.23); full ranges k = 3 0.81-1.06, k = 4 1.64-2.16, k = 5 2.82-3.76, mps8 {1,6} 4.27, k = 7 up to 6.5; raw and projected differ by up to 0.11 and the projection is biased at sigma >= 1e-3 (k = 2 small init quoted raw: -0.437, -0.581) (7.1, 7.2) |
| (f) levels | TA mps4 "two short (log-time) plateaus"; "every k* >= 3 level within 0.2"; ht4 {0,1,2} 8 seeds | TA mps4 first level: power favoured (0.22 vs 0.53), law undetermined; B4 ht4 fan-in second level 4.73 ± 0.90 is an exception; ht4 {0,1,2} has 7 fitted seeds (7.1, 7.4, 8) |
| (g) provenance | run classes, dropout tables, multi-component event CSV made by hand; ghost row in the extensions table | `stall_tables.py` writes `run_classes_by_sweep.csv`, `stall_dropout_*.csv`, `stall_capped_ends.csv`; `finalize.sh` also writes `bond_events2_multicomp_events.csv`; `r12_partial.csv` (an intermediate run of `r12.py` on an unfinished stall diagnosis, used nowhere) removed; ghost row (a lambda2K error record without model fields) fixed in `analysis.py`; the lost lambda2K run (start-up check tolerance) not rerun |
