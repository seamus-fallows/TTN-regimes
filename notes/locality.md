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

## 5. Several learned components: the rank rule (conjecture, derived for two components)

"The sites where the new component differs from what is learned" is ambiguous once two or more
components are learned. The derivation above suggests the general rule:

  k(L, r) = number of cores incident to the edges whose bond rank increases when s_r U_r is added
  to the learned W_L = sum_{j in L} s_j U_j (k = 1 if no bond rank increases),

computed exactly for pattern teachers from the small matrices of section 1
(`experiments/locality/rules.py`). For two components it reduces to the Steiner rule (checked for
every (tree, S) used here). For more components it can differ: a learned component that already
"uses" a bond index on part of the path can force the new component to take a longer new-index
route. Example (MPS, N = 8, U1 = 0^8, U2 differs on {3,4}, U3 differs on {0,1}): relative to U1
alone both have k = 2, but after U1 and U2 are learned, adding U3 raises the rank on edges
0|1, 1|2, 2|3, so k = 4 (sigma^-2 instead of log). The order of learning therefore changes the
saddle types. To be tested in step 3 of the brief.

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

(filled in as experiments finish; see `experiments/locality/README.md`)
