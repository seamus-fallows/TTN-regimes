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
  k = 3 vs N = 6; TC on `mps8`, k = 2, 3, 5 vs N = 8, 7, 5. This is the lead's refinement (ii)
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
  greedy order is non-strict iff no unlearned component can be added by raising the rank of a single
  bond** (all k(L, r) >= 3); it is strict if some k(L, r) = 2, or if a wrong-order component can be
  corrected through two S-nodes. Measured in section 7 via lambda_min vs sigma at plateau midpoints.

## 7. Results

All runs: gradient flow (scipy LSODA with the exact-Hessian Jacobian, rtol 1e-9, atol 1e-14, event
root-finding; Radau fallback in < 1% of runs), library v0 model, float64, teacher seed 1000, init seeds
0-7 (0-4 for k >= 6). Integrator check on one k = 6 run: LSODA (rtol 1e-9, 1e-10), Radau (1e-9, 1e-10)
and RK45 agree on t2 - t1 to 3e-8 relative. Conservation-law drift < 7e-8 (max over all runs) against
|Q| ~ 1e-2..1. Tables and figures: `experiments/locality/results/` (regenerate with `finalize.sh`).

### 7.1 Two-component teachers: the exponent table (steps 1, 2) — **empirical, matches derived**

`results/twocomp_table.md` (one row per tree, S, protocol). Summary (fitted exponent of t2 - t1 vs
sigma, per seed, mean ± SE; "local" = three smallest sigma):

| k (Steiner) | rows (trees) | predicted | measured |
|---|---|---|---|
| 1 | binary4 {0,1},{2,3}; binary8 {0,1}; star3 {0,1} | no plateau | t2 - t1 = 4e-4..1e-3 (small init), 0.6 (saddle start); separation ratio << 1 |
| 2 | mps4/6/8, midroot6 adjacent pairs | log law, slope (1 - s2/s1)/λ2 = 1.00 (small init), 1/λ2 = 2.00 (saddle start) | slopes 0.98-1.01 (small init), 2.00-2.07 (saddle start) |
| 3 | 26 rows: mps4/6/8, binary4/8, ht4/8, star3, midroot6 | 1 | 0.98-1.03 (local 0.97-1.01) |
| 4 | mps4/6/8, star3, midroot6 | 2 | 1.81-2.39 (local 1.94-2.47; all but mps6 within 0.1 of 2) |
| 5 | ht4 {0,2},{0,3},{1,2}; mps6 {1,3,5}; binary8, ht8, mps8 rows (pending) | 3 | ht4: 2.88-3.13; mps6 {1,3,5}: 2.86 (saddle), 3.76 ± 0.27 (small; see 7.5) |
| 6-8 | (pending; sigma range < 1 decade) | 4-6 | ht4/ht8 all-sites (k = 7): 5.8 ± 0.6, 6.7 ± 0.8 over sigma in [0.08, 0.6] |

- The Steiner count k decides the exponent on every tree, whatever the number of sites in S
  (e.g. binary4: S = {1,2}, {0,3}, {0,2}, {0,1,2}, all four sites all give k = 3 and p = 1.00), and
  regardless of where the root is (midroot6 = mps6 rerooted: same exponents).
- The k = 2 slope distinguishes the head start: small init 1.00, saddle start 2.00, exactly as
  derived in section 2 (= theory_saddles.md P2). The naive H2 slope 1/λ is refuted for small init.

### 7.2 Saddle type (H3) — **empirical, matches derived**

`results/twocomp_hessian.png`: gauge-projected Hessian at the plateau midpoint.
- k = 2: λ_min = -0.49..-0.50 at the smallest sigma (predicted -λ2 = -s2 s1^((K-2)/K) = -0.5), flat in
  sigma (fitted exponent 0.00 ± 0.00 from the saddle start); one negative eigenvalue (chi - 1 = 1).
- k >= 3: λ_min ∝ sigma^x with x = 1.00-1.04 (k = 3), 1.8-2.2 (k = 4), 2.9-3.1 (k = 5), 6.0-6.8 (k = 7),
  i.e. x ≈ k - 2: the negative curvature vanishes as sigma -> 0 (higher-order saddle).
- So "strict iff k = 2" holds for these saddles, classified by the sigma-scaling of λ_min (the fixed
  1e-3 relative threshold would call k = 3 plateaus strict at sigma >~ 0.03).

### 7.3 Single-run escape degree and dormant amplitudes — **empirical**

- Growth exponent γ = d log(da/dt)/d log a on the plateau, fitted in the window a in [1e-4, 1e-2]
  (theory P4: 2 - 2/k; `results/growth_summary.csv`, median per (tree, S, protocol), 1-22 rows per k):
  k = 2: 0.996-0.999 (saddle start); 3: 1.331-1.340; 4: 1.498-1.511; 5: 1.605-1.624; 6: 1.664-1.677;
  7: 1.712-1.741; 8: 1.739-1.747, i.e. k = 2/(2 - γ) = 2.0, 3.0, 4.0, 5.1, 6.0, 7.0-7.7, 7.7-7.9. The escape
  degree is readable from a single run and equals the Steiner count for every k = 2..8 tested, including
  the two MPS k = 5 rows whose sigma-exponent is too high (7.9): there the route degree is 5, and the
  excess sigma-exponent comes from the dormant amplitude entering the plateau, not from the degree.
- Dormant-norm test (lead's refinement iii): the remaining plateau time from the midpoint scales with
  the overlap of the escaping component there, a_mid: slope 0.98-1.01 vs ln(1/a_mid) for k = 2
  (predicted 1/(2λ) = 1), exponent -0.25..-0.31 for k = 3 (predicted -(k-2)/k = -0.33), -0.67..-0.77
  for k = 7 (predicted -0.71). Plateau lengths are set by the dormant route amplitude entering the
  plateau; sigma enters only through it.



### 7.4 Several components: same teacher, different saddle sequences (step 3) — **empirical, matches the route rule**

Predicted sequences: route rule at rank-constrained optima (section 5; `results/route_sequences.json`).
Measured: time spent within 2% of each predicted plateau level L_j, fitted against sigma per seed
(`results/levels_summary.csv`; figures `results/trees_TA.png`, `trees_B4.png`, `trees_B4_ht_balanced.png`,
`trees_E8.png`). Exponents p (mean ± SE over 5-8 seeds; "log" = k* = 2, durations grow additively):

Runs are pooled by event order (analysis standard 3); durations resting on < 3 stored integrator steps
are counted as unresolved and not fitted (only the first, very short log-time levels on mps4/mps8 have
unresolved cases: up to 24 of 45 runs). "log" = k* = 2: durations grow additively (fitted power < 1 and a
positive slope against ln(1/sigma)).

| teacher | tree (protocol, sigma range) | predicted k* per plateau | measured p ± SE (seeds) |
|---|---|---|---|
| TA {0000,1100,1110} (1, .6, .35) | mps4 (0.2-1e-5) | 2, 2 | log, log |
| | binary4 (0.1-0.002) | 3 (first escape already at L = 0.044) | 1.05 ± 0.01 (8) |
| | ht4 (0.25-0.02) | 3, 4 | 1.26 ± 0.11, 2.11 ± 0.04 (7); balanced init 1.51 ± 0.21, 2.05 ± 0.04 (8) |
| B4 {0000,0110,1001} (1, .3, .6) | mps4 (0.2-1e-3) | 2 (the 0.3 component first), 4 | log, 2.06 ± 0.03 (5) |
| | binary4 (0.1-0.002) | 3, 3 | 1.24 ± 0.12, 1.06 ± 0.01 (8) |
| | ht4, balanced init (0.3-0.05) | 5, 5 | 3.02 ± 0.02, 3.17 ± 0.10 (8) |
| E8 (N = 8; 1, .4, .7) | mps8 (0.3-0.06) | 2, 6 | log, 4.05 ± 0.10 (7) |
| | binary8 (0.3-0.04) | 5, 5 | 3.09 ± 0.04, 2.97 ± 0.08 (6, order 0>2>1) |
| TC (N = 8, 4 components) | mps8, saddle start (0.1-1e-4) | 2, 2, 2, 4, 2 | log, log, log, 2.02 ± 0.02, log (7) |
| | binary8 (0.25-0.03) | 3, 3, 3 | 1.33 ± 0.11, 1.26 ± 0.14, 1.19 ± 0.08 (5) |
| TD (N = 8; 1, .6, .35) | mps8, saddle start (0.1-1e-4) | 2, 2, 3 | log, log, **1.02 ± 0.01** (7) |
| | mps8 (0.2-0.02) | 2, 2, 3 | log, log, 1.12 ± 0.08 (6) |
| N4 {0000,1100,1111} (1, .6, .4) | mps4 (0.2-1e-5) | 2, 2, 2 (one partial step) | log, log, log |
| | binary4 (0.1-0.002) | 3 | 1.12 ± 0.03 (8) |
| | ht4 (0.25-0.02) | 3, 3, 3 | 1.42 ± 0.14, 1.32 ± 0.13, 1.17 ± 0.04 (8) |

- Locality beats weight (B4 on mps4): the 0.3 component is learned (log time) before the 0.6 one
  (sigma^-2), in every run that escapes.
- TD separates the candidate rules: literal H2 and the exact rank rule predict k = 2 and 4 for the last
  plateau; the route rule predicts 3 (a partial step first); measured 1.02 ± 0.01 over 3 decades of sigma
  (saddle start) and 1.12 ± 0.08 with small init: k = 3.
- TA, N4, TB: the first escape on binary4 already fits U1, U2 and part of U3 (a superposition in one
  core); the MPS learns the same teacher in short log-time steps; HT needs sigma^-1 and sigma^-2 plateaus.
- Plateau levels: every predicted level is visited (visited fraction 0.89-1.00; misses are seeds that
  took the other branch of a k* tie, e.g. E8 on binary8).
- Exponents of k* = 3 levels are biased upward (1.1-1.4) at the larger sigma, where the 2% band also
  catches the approach to the level; the k* >= 4 levels and the wide-sigma rows are on target.

### 7.5 Dead routes under fan-in init (theory_saddles.md R12) — **empirical**

A run is "stalled" when, after a component has been learned, the gradient stays at roundoff (|v| < 1e-12
for 5000 consecutive evaluations): the route of the next component is dead and gradient flow never
leaves the plateau (L stays at the partial fit, e.g. 0.125 = s2^2/2).

- Fan-in init: stalls occur **only for k >= 4**: k = 4: 6-30 of 48 runs per (tree, S) (mps4 S = {0,3}: 30/48,
  5 of 8 seeds); k = 5: 6-17/48; k = 6-8: 7-23 of 24-30. **No stall in any k <= 3 run** (about 3000 runs),
  and none on star3 (k = 4). A stalling seed stalls at every sigma (the imbalance pattern Q/sigma^2 is
  fixed by the init direction).
- Gauge-balanced init of the same W_0 (all Q_e = 0, `run.balance_gauge`): 2 stalls in 580 runs on the
  same k >= 4 cases (mps4 S = {0,3}: 0/48). B4 on ht4: fan-in 23/48 stalled, balanced 0/48.
- The saddle-start protocol also stalls (its noise has Q ~ sigma^2 of random sign).
- Consequence: for k >= 4 the exponent table uses the runs that escape; gauge-balanced runs give the same
  exponents (`results/twocomp_balanced_table.md`).

### 7.6 Bond spectra at events (step 4) — **empirical, matches derived**

`results/bond_spectra.png` (mps8 S = {2,5}, binary8 S = {3,4}, ht8 S = {1,2}) and
`results/bond_events.csv`: on the plateau the second singular value of every Steiner edge stays at the
dormant level (1e-8..1e-3, set by sigma); at the second event all Steiner edges rise together to s2,
the others stay at the noise level. Effective-rank changes per event agree with the predicted edge set in
the fraction of runs listed in `results/bond_events_summary.txt`.

### 7.7 Robustness (step 5) — **empirical**

N = 4, 8 seeds, small init unless stated (`results/robust_table.md`):
- Weight ratio: k = 2 log slope 3.02 ± 0.01 for s2 = 0.25 (predicted (1 - s2/s1)/λ2 = 3.00; saddle start
  4.04 ± 0.05 vs 1/λ2 = 4.00) and 0.40 ± 0.05 for s2 = 0.75 (predicted 0.33; saddle start 1.345 ± 0.016 vs
  1.333). λ_min = -0.25, -0.50, -0.64..-0.69 for s2 = 0.25, 0.5, 0.75 (predicted -s2). k = 3, 4 exponents
  unchanged (0.99-1.06, 1.87-2.05).
- Overparameterization (chi = 4 at d = 2; d = 4 with chi = 2, 4, 8): k = 2 slopes 1.00-1.01, k = 3
  exponents 0.99-1.05, k = 4 exponents 1.90-2.01. At plateau points the k = 2 Hessian has one eigenvalue
  near -λ2 and the next ones O(1e-4) (one negative mode per active route, theory R10), also at chi = 8.
- Dead routes become rarer with spare dimensions: mps4 S = {0,3}: 30/48 stalled at (d, chi) = (2, 2),
  6/48 at (2, 4), 0/48 at d = 4.

### 7.8 Gauge-balanced init (R12 check) — **empirical**

`results/twocomp_balanced_table.md`: the same k >= 4 cases with Q_e = 0 at t = 0 give tighter exponents
(k = 4: 1.94-2.01; k = 5: 2.96-2.99 on binary8 and ht4; k = 6: 3.5-4.1, medians 3.9-4.4; k = 7 on ht4:
4.98 ± 0.08 instead of 5.8 ± 0.4 with fan-in) and almost no stalls. Two MPS k = 5 rows stay above 3
with either init over sigma in [0.03, 0.3] (mps8 S = {1,5}: 3.9 ± 0.3, mps6 S = {1,3,5}: 3.3 ± 0.2);
see 7.9 for a smaller-sigma check.

### 7.9 Open deviation: two MPS k = 5 rows — **empirical, unexplained**

Smaller-sigma check (`configs/k5_extension.yaml`, sigma = 0.012-0.03, 8 seeds; local exponent over the
three smallest sigma): mps6 S = {1,3,5}: 3.05 ± 0.06 (saddle start), 3.43 ± 0.19 (balanced init);
mps8 S = {1,5}: 3.72 ± 0.16 (saddle start), 3.68 ± 0.31 (balanced init). Predicted 3. The binary8, ht4 and
ht8 k = 5 rows give 2.9-3.1. Candidate mechanism (conjecture, untested): noise entries in flat directions
next to the route (e.g. a new-index-to-old-index entry on a core adjacent to a route end) form degree-2
sub-routes orthogonal to the residual whose quartic penalty shrinks the route amplitudes algebraically
during a long plateau; the effect would grow with the plateau length (k) and depend on how many such
entries the tree offers. Testing it needs the route amplitudes measured along the plateau. Supporting
observations: the growth exponent on these plateaus is that of a degree-5 route (γ = 1.61, 7.3), and at
sigma <= 0.03 these two cases stall even with gauge-balanced init (mps6 S = {1,3,5}: 5/24, mps8 S = {1,5}:
10/24 runs; saddle start 9/24 and 7/21), so a route-killing mechanism other than the R12 gauge trap acts
here.

## 8. Verdict

**H2 (exponent set by locality), two-component teachers: supported** — derived (section 2; = Steiner
identity proven in theory_saddles.md) and empirical on 9 tree types (mps 4/6/8, binary 4/8, ht 4/8, star3,
midroot6), k = 1..8, two protocols, 5-8 seeds, sigma over 1-3 decades: no plateau for k = 1; log law for
k = 2 (log law beats a power law by a factor ~5-20 in RMS log residual); exponent k - 2 for k = 3 (26 rows,
0.98-1.03), k = 4 (1.9-2.1), k = 5 (2.9-3.1, two MPS rows higher, 7.8), k = 6-8 (within the larger errors of
< 1 decade of sigma, best with balanced init). **Refinement**: the k = 2 slope is (1 - s2/s1)/λ2, not 1/λ2
(the dormant route grows during the previous plateau; the saddle start, which has no previous plateau,
gives 1/λ2), and more generally plateau lengths are set by the dormant route amplitude entering the
plateau (remaining time ∝ a_mid^-(k-2)/k for k >= 3, ∝ ln(1/a_mid)/(2λ) for k = 2), which equals sigma times
a factor only when the previous phases do not rescale the route.

**H2 with several learned components: the literal statement fails; the supported rule is the route rule
at rank-constrained optima** (section 5; derived at leading order, empirical on B4, E8, N4, TA, TB, TC,
TD): the next plateau's exponent is k* - 2 with k* the smallest connected set of cores whose projected
local residual is non-zero, evaluated at the actual saddle, which is the best fit for the current bond-rank
profile and need not be a partial sum of components. Counter-examples to the literal rule: TA on all
three trees and TD on mps8 (literal H2 / exact rank rule: k = 2 / 4; measured 3). Furman's N (nodes of
the rank-deficient component) is an upper bound and is often much larger than k* (e.g. TC on mps8:
N = 8, 7, 5 vs k* = 2, 2, 4).

**Same teacher, different sequences: demonstrated** (`results/trees_TA.png`, `trees_B4*.png`,
`trees_E8.png`): e.g. TA is learned by mps4 in two log-time steps, by binary4 in one first escape plus a
sigma^-1 plateau, by ht4 through sigma^-1 and sigma^-2 plateaus; B4's 0.3 component comes before its
0.6 component on the MPS only.

**H3 (strict iff k = 2): supported for saddles visited in greedy order** (Hessians at 6000+ plateau
midpoints): λ_min -> -λ2 = -s2 s1^((K-2)/K) independent of sigma for k = 2; λ_min ∝ sigma^(k-2) -> 0 for
k = 3..8. Not tested here: wrong-order and star saddles, which theory_saddles.md shows can be strict
with k* >= 3 (so "only if" needs "visited in greedy order, no stars").

**New (empirical)**: under the default fan-in init, a large fraction of k >= 4 escapes never happen
(dead routes; 7.5), never for k <= 3; gauge-balanced init removes them. Bond spectra: the second singular
value rises at the event on exactly the Steiner edges in 98.7% of 5132 events (7.6).
