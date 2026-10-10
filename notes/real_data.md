# Real data: do the teacher-student TTN phenomena survive on images? (B5)

Agent `real-data`, branch `agent/real-data`. Experiments in `experiments/real_data/`. Status tags:
**proven**, **derived** (under stated assumptions), **empirical**, **conjecture**.

## 1. Setting

- Data: MNIST, classes 3 (y = +1) vs 5 (y = -1), balanced subsample n_train = 4000, n_test = 1600
  (data seed 0). Images cropped to the central 24 x 24 pixels and block-averaged to 4 x 4 (N = 16 sites)
  or 8 x 8 (N = 64), intensities in [0, 1].
- Site features phi(x) = (cos(pi x / 2), sin(pi x / 2)) (d = 2), "raw"; or "white": per-site
  A_i phi with A_i E[phi_i phi_i^T] A_i^T = I (uncentred second moment, so each site's function space,
  span{cos, sin}, is unchanged and the network's function class is the same; only the gradient-flow
  metric and the init distribution in function space change, as in `theory_covariance.md` C1).
  Centring (subtracting the site mean) would change the function class (the span of the centred
  features does not contain the uncentred ones), so it is not used as a separate variant.
- Model: f(x) = <W, phi_1 (x) ... (x) phi_N>, empirical square loss L = 1/(2n) sum (f - y)^2, fan-in
  init with per-core scale sigma (`TTNSpec.init`), float64.
- Trees: (a) `binary16` with sites in Morton (quadtree) order: every bottom core carries two
  horizontally adjacent pixels, every level-2 subtree a 2 x 2 block, level 3 a 2 x 4 half-block;
  (b) `mps16` along a boustrophedon (snake) ordering; (c) `binary16` with a random pixel permutation.
  K = 15 (binary), 16 (MPS).

## 2. What the theory predicts on real data

The teacher-student theory assumes a product input measure. Real images have a non-product law and
no teacher tensor; the loss is L = 1/2 <W, C W> - <T, W> + const with C = E[Phi Phi^T] (non-product)
and T = E[y Phi]. What carries over:

**R1 (first plateau; derived, `theory_covariance.md` C4 and 7.4.1).** Near the origin
L = L0 - <T, W(theta)> + O(|W|^2) for any input law, so the first escape is the whitened-teacher escape
with W* -> T: t1 ∝ sigma^-(K-2) per seed (init direction fixed, sigma rescales it), exponent 13 on
binary16 and 14 on mps16. After the escape the network sits near a bond-rank-1 critical point (all
edges acquire their first singular value together).

**R2 (second escape: route degree from dimension counting; derived).** At a bond-rank-1 critical
point, a connected core set C can start an escape at degree |C| only if its residual R_C is nonzero on
the free modes of its terminal cores (`theory_saddles.md` 5.6). On a binary tree an internal core at
rank 1 has a single mode (its child and parent bonds are rank 1), so no pair of cores and no route with
an internal terminal has free modes: k* >= 3, and the minimal routes are **sibling pairs of bottom
cores through their common parent (k = 3)**. Prediction: the second plateau scales as sigma^-1 per seed
on binary16 for any pixel order and any data, and the rank rises together on the two sibling bottom
edges. On the MPS every core carries a physical leg, so every adjacent pair has free modes:
k* = 2, a strict saddle, a log(1/sigma) gap, rank rising on one edge. These counts do not depend on
the input law (only on bond ranks and site dimensions); whether R_C is nonzero does, and for generic
data it is.

**R3 (bottom-up growth; derived by the same count).** An internal edge e = (v -> parent) can join a
2-core route only once v has a free mode, i.e. once the product of the ranks of v's child bonds
exceeds r_e. So ranks grow from the bottom of a binary tree upwards; an edge at level l gains rank
either through a long route (k >= 2l + 1 at rank 1) or after the edges below it have grown.

**R4 (locality on real data; derived consequence + conjecture).** The degrees k depend on the tree
shape, not on which pixels sit where, so quadtree and scrambled binary trees have the same predicted
sigma-exponents. Locality can only enter through amplitudes (how much residual label correlation
lives between the two pixel pairs of a sibling route) and plateau levels (how much a rank budget
captures). Conjecture: adjacent pixels carry larger residual correlations, so the quadtree tree has
shorter later plateaus (smaller prefactors) and lower losses at equal rank.

**R5 (non-product metric; derived in `theory_covariance.md` C10 for MPS).** Cross-site correlations
open additional 2-core routes, so later plateaus on the MPS should be log-time or invisible.

## 3. Methods (adapted diagnostics)

- **Dynamics.** Gradient flow on the empirical loss, float64, rtol 1e-8, atol 1e-14. Phase 1 (first
  plateau): DP5, clock re-anchored (t -> 0, offset kept) whenever an accepted step is shorter than 1e-9 t,
  so that the escape at t ~ 1e18-1e20 (sigma = 0.05) is resolved; stops when L has dropped by 1%.
  Phase 2: LSODA on the clock tau = t - t_a for tau <= 1e4 (Newton matrix = Gauss-Newton part of the
  Hessian, per-sample Jacobians; the error control is LSODA's). Gauge conservation drift (cumulative,
  relative to max core norm^2) <= ~2e-9 in every checked run.
- **Data-weighted bond spectra.** Across edge e, f(x) = sum_a up_a(x_below) down_a(x_above) with
  per-sample messages; the Schmidt values w.r.t. the empirical marginals of the two sides are
  sqrt(eig(G_up^1/2 G_down G_up^1/2)), G = (1/n) M^T M. Gauge invariant; equal to the Frobenius spectra
  for a product law with identity second moments (checked to 2e-12 by exact enumeration). Needed because
  after the escape |W|_F grows to 1e6-1e7 along directions the data barely sees (raw features), so
  Frobenius spectra measure invisible weights.
- **Learning events.** (a) Loss drops: peaks of r = -dL/dt standing out by a factor 3 against the minima
  separating them from higher peaks, lowering L by >= 1% of L(0); time = rate peak, width = FWHM of the
  peak. Separation ratio and S2S rule (i)-(ii) as in `notes/diagnostics.md`. Deviation from the notes:
  the notes' width (10% -> 90% of the drop, ends at the flattest points) is not used because on real data
  the loss keeps creeping down between events (it never saturates: ill-conditioned directions within the
  current rank profile), which makes progress-based widths measure the tail (e.g. 909 vs a rate-peak FWHM
  of 17). (b) Rank events: s_k^e / s_1^e crossing 0.05 (log-interpolated), used to say which edges gain a
  singular value inside each loss drop (window: the drop's FWHM interval widened by one width on each side).
  Final-value references (notes' f x s_k^{e,*}) are not used for widths for the same creep reason.
- **Times.** t1 (first escape) = t_a + tau of the first loss drop (absolute clock); later gaps on the tau
  clock. Merged log + fine rows (`ttn.dynamics.merged_rows`) everywhere.
- **Route amplitudes** (`route_amplitudes.py`, derived from `theory_saddles.md` R1 / `locality.md` 2): at the
  bond-rank-1 plateau of a binary run f ~ c prod_v q_v(x) over bottom cores; for a pair (u, w) of bottom
  cores M_uw = (1/n) sum_j r_j h_uw(x_j) Phi_u(x_j) Phi_w(x_j)^T with h_uw = c prod_{v != u,w} q_v,
  lambda_uw = |P_u M_uw P_w|_2 (complements of the learned directions). The escape along the route
  (u, ..., w) has degree k = |Steiner(u, w)| and its leading term is -lambda_uw prod(eps); the route
  with the smallest k escapes first, ties (equal k) are decided by lambda and by the dormant entries.

## 4. Results

Setting unless stated: MNIST 3 vs 5, 4x4, raw cos/sin features, chi = 4, n_train = 4000, gradient flow
(section 3), sigma in {0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.5}, init seeds 0-4, horizon tau = 1e4 after
the escape. Numbers marked (interim) come from the partial sweep and are updated in section 6.

### 4.1 First plateau (R1): exact sigma^-(K-2) [empirical, derived]
t1 ranges from ~1e6 (sigma = 0.5) to ~1e19-1e20 (sigma = 0.05). The per-seed exponent of t1 vs 1/sigma is
13.000 +- 0.000 (binary16, K = 15) and 14.000 +- 0.000 (mps16, K = 16) to the digits shown. This is
guaranteed: near the origin L = L0 - <T, W(theta)> + O(|W|^2), the first term is homogeneous of degree K,
and the quadratic term is O(sigma^(2K)) against O(sigma^K), so the flow from sigma Z is the flow from Z
with time rescaled by sigma^-(K-2) up to O(sigma^K) corrections. On real data this holds for the raw
features at every sigma used. With whitened features the function-space scale at equal sigma is ~256x
larger (E|Phi|^2 = 2^16 instead of 1), and the asymptote is reached only at smaller sigma (pilot: 12.6 on
mps16 over sigma = 0.2-0.5).

### 4.2 Stepwise bond-rank growth and the first route (R2) [empirical]
After the escape all edges carry one data-weighted singular value. Then the next singular values escape
one route at a time, from s_k/s_1 ~ 1e-5-1e-3 to O(0.1-1) (figure `bond_spectra_raw_s*`).
- binary16 (quadtree): the first rank increase is on exactly two sibling bottom edges, a 3-core route
  through their parent, in 35/35 runs (interim). For seeds 0, 2, 3 it is the pair under the top-right 2x2
  pixel block (edges 3, 4), for seed 1 the top-left one (0, 1); seed 4 switches from (3, 4) at sigma <= 0.14
  to (0, 1) at sigma >= 0.2.
- mps16 (snake): the first rank increase is on a single edge (edge 5) in 35/35 runs (interim). It happens
  during or right after the first escape: a strict 2-core route with a large rate.

### 4.3 Which route: degree first, then amplitude (R2 + route amplitudes) [empirical]
At the bond-rank-1 plateau of each quadtree run the route amplitudes lambda_uw (section 3) were computed
for all 28 pairs of bottom cores. The sibling pair (k = 3) with the largest lambda is the pair that gains
rank first in 34/35 runs; the largest sibling amplitude exceeds the second by 2-70x. The largest amplitudes
overall belong to non-sibling pairs (k = 5, 7; e.g. lambda = 0.35 for k = 5 vs 0.05 for the best k = 3
pair, sigma = 0.5, seed 0). These routes escape later: route degree beats amplitude, as on teachers
(theory_saddles.md R8, "locality beats weight"). The plateau state is the same rank-1 critical point
for all sigma at a given seed (relative distance from an exact rank-1 product <= 0.03 for sigma <= 0.2).
There are at least two such points (seeds land on the (3,4) or (0,1) one), so the first escape is a
seed-dependent choice among KKT points of <T, W>.
