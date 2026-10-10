# Real data: do the teacher-student TTN phenomena survive on images? (B5)

Agent `real-data`, branch `agent/real-data`. Experiments in `experiments/real_data/`. Status tags:
**proven**, **derived** (under stated assumptions), **empirical**, **conjecture**. Status: complete
(2026-10-10); prediction table in section 5 (updated in fix round 1), robustness in section 7.
**Fix round 1 (local, 2026-10-10): section 8 supersedes sections 4-7 where they differ; superseded
statements are marked "[fix 1: ...]".**

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
shorter later plateaus (smaller prefactors) and lower losses at equal rank. [fix 1: adjacency alone does not
decide it: the transposed quadtree, equally local, is among the worst orders (8.4)]

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

### 4.1 First plateau (R1): exact sigma^-(K-2) [proven; fix 1: an identity of small-init flow, not evidence that predictions survive, 8.10]
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
- binary16: the first two edges to gain a second singular value are two sibling bottom edges, a 3-core
  route through their parent, in 35/35 quadtree and 89/89 scrambled runs. In 35/35 and 86/89 they cross the
  threshold together (within 5% in time). [fix 1: these count the same critical point once per sigma; all 20
  (order, seed) pairs, 96/96 rank-1 states, 8.3] For seeds 0, 2, 3 it is the pair under the top-right 2x2
  pixel block (edges 3, 4), for seed 1 the top-left one (0, 1); seed 4 switches from (3, 4) at sigma <= 0.14
  to (0, 1) at sigma >= 0.2.
- mps16 (snake): the first rank increase is on a single edge (edge 5) in 34/35 runs. It happens
  during or right after the first escape: a strict 2-core route with a large rate. [fix 1: 10/35 of these
  "increases" are the threshold already met at tau = 0; the resolved ones precede the first drop's peak,
  i.e. the first MPS route is merged with the escape; verdict "partly", 8.6]

### 4.3 Which route: degree first, then amplitude (R2 + route amplitudes) [empirical]
At the bond-rank-1 plateau of each quadtree run the route amplitudes lambda_uw (section 3) were computed
for all 28 pairs of bottom cores. The sibling pair (k = 3) with the largest lambda is the pair that gains
rank first in 34/35 quadtree runs (best/second sibling amplitude: median ratio 46) and in 78/89 scrambled
runs (30/35, 32/34, 16/20 for the three orders; median ratio 5.6, closer ties). [fix 1: 28 of these states
are not near rank 1 and runs across sigma repeat one critical point; per unit 22/26 hit in all states, 2
mixed, 2 miss, chance 1/4; 11/11 out of sample, 8.3] The largest amplitudes
overall belong to non-sibling pairs (k = 5, 7; e.g. lambda = 0.35 for k = 5 vs 0.05 for the best k = 3
pair, sigma = 0.5, seed 0). These routes escape later: route degree beats amplitude, as on teachers
(theory_saddles.md R8, "locality beats weight"). The plateau state is the same rank-1 critical point
for all sigma at a given seed (relative distance from an exact rank-1 product <= 0.03 for sigma <= 0.2).
There are at least two such points (seeds land on the (3,4) or (0,1) one), so the first escape is a
seed-dependent choice among KKT points of <T, W>.

### 4.4 Plateau durations scale with the route degree [empirical; prediction derived]
Gaps are measured event to event: the loss drop that carries a route's rise minus the previous drop
(times = rate peaks, merged rows). Runs are classified before pooling: a seed's series keeps only the sigma
values at which the same route (same edges) rises first. Fits are per seed (sigma rescales the same init
direction). Power and log laws are fitted in linear-gap space with relative residuals (`scaling.py`).
[fix 1: the code fitted the log law unweighted; all laws now use relative residuals, 8.2]
- **k = 3 (first route on binary trees): gap ∝ sigma^-1.** [fix 1: an attribution bug assigned three gaps to
  a later route; clean series: A/sigma + B fits all 11 (B < 0), free exponent 1.22 ± 0.03, log law rejected
  (rms 0.11-0.30); "1.01 ± 0.14" withdrawn, 8.1] Per-seed exponents (mean +- SE), raw features,
  sigma = 0.05-0.5 (6-7 values per seed): quadtree 1.01 +- 0.14 (4 seeds), scrambled0 0.97 +- 0.11 (4),
  scrambled1 1.27 +- 0.02 (4), scrambled2 0.91 +- 0.23 (2, 4 sigma values). Relative rms residual: power law
  0.03-0.14, log law 0.56-0.91. Excluded (seed, sigma) where another route went first: quadtree seed 4 at
  sigma >= 0.2 and seed 2 at 0.5; listed in `results/scaling.json` [fix 1: not "scaling_raw.json"]. Example (quadtree seed 0): gaps 298,
  209, 143, 100, 69, 45, 29 for sigma = 0.05 ... 0.5, exponent 1.01.
- **k = 2 (MPS, edges next to the first one): gap linear in log(1/sigma).** [fix 1: refitted with relative
  residuals for both laws in 8.5; these are later routes, the first MPS route (edge 5) has no measured gap, 8.6] Time of each edge's first rank
  increase after the first drop, per seed (5 seeds, sigma = 0.05-0.5): edge 4 log-law rms 0.09 vs power-law
  rms 0.35, slope 15.3 +- 1.5 per unit ln(1/sigma); edge 3 rms 0.14 vs 0.25, slope 31.4 +- 1.1. On the teacher
  the slope is (1 - s_r/s_q)/lambda (head start). Here the measured |lambda_min| at that plateau
  (0.056-0.077, section 4.5) gives 1/lambda ~ 15, so the edge-4 slope is consistent with a small head start.
- **Later routes have higher degree on binary trees.** [fix 1: withdrawn, not established; binary routes
  between bottom cores have odd k, 8.10] Quadtree seed 0, 4th drop: 5637, 1818, 626, 144, 44 for
  sigma = 0.07 ... 0.3, exponent 3.3, close to a k = 5 route (3). At sigma = 0.05 this event lies beyond
  tau = 1e4, consistent with the extrapolation (~1.5e4). Per-edge fits on binary trees for edges that rise
  later give exponents 2.1-2.4 (k = 4-5). These are not classified by route and are only indicative.
- **Not explained:** MPS edges far from the first one (0-2, 6-9) rise with power-law-like sigma dependence
  (exponent 0.9-1.3, 4-5 seeds). This is not what rank-1 counting predicts (every MPS pair has free modes,
  so k = 2). Possible causes are vanishing degree-2 residuals on those pairs (escape through 3-core routes)
  or dormant entries rescaled by earlier events. Open.

### 4.5 Saddle type at the first plateau (H3) [empirical]
Exact Hessian (dense residual tensor) at the plateau midpoint, raw and gauge-projected (fixed projection,
rtol 1e-3); the two agree within ~10%. Runs at the common plateau level only (same saddle).
- MPS (plateau L = 0.230, 26 runs, sigma = 0.05-0.3): lambda_min = -0.056 ... -0.077 with exponent
  0.04 +- 0.03 in sigma (`saddle_scaling`: strict). Per seed 0.038 +- 0.025 (3 seeds with >= 4 sigma).
- Binary quadtree (plateau L = 0.270, 20 runs, sigma = 0.05-0.3): lambda_min ∝ sigma^0.877+-0.005 per seed
  (3 seeds; pooled 0.73 +- 0.16): higher order. The prediction for k = 3 is exponent 1 (teachers: 0.98-1.03);
  measured 12% lower. [fix 1: with seed 4 (1.15, 3 sigma values) 0.95 ± 0.07 over 4 seeds, each < 1 decade;
  consistent with 1; MPS state is after edge 5 rose, 8.7] At sigma <= 0.14 there is exactly one eigenvalue below -1e-3 lambda_max (one active
  route); more appear at larger sigma.

### 4.6 The S2S signature on real data [empirical]
Plateaus are not flat: between events the loss keeps decreasing slowly (fitting within the current rank
profile). [fix 1: not slowly: between drops it falls by a median 0.76 of the next drop's size (8.10)] With rate-peak drops (section 3), quadtree runs are S2S (>= 2 drops, every rho > 3) in 100% of
runs at sigma <= 0.07 and 40% at sigma = 0.5. The median smallest rho grows from 2.8 (sigma = 0.5) to 24
(sigma = 0.05), so plateaus grow without bound as sigma -> 0 (condition iii). Scrambled trees behave the
same way (median min rho 37 at sigma = 0.05). Threshold sensitivity: with rho > 5 the quadtree S2S fraction rises
monotonically from 0.0 (sigma = 0.5) to 1.0 (sigma = 0.05); scrambled from 0.27 to 0.93; with rho > 2 all
binary fractions are 0.7-1.0. The MPS is mostly not S2S (fraction 0-0.4 at rho > 3, 0-0.2 at rho > 5, median
min rho 1.9-3.2). Its escapes are strict saddles with short log-time gaps, and drops overlap.

### 4.7 Locality: quadtree vs scrambled vs MPS [empirical; fix 1: narrowed to a pixel-order effect (8.4) and a tau-clock, snake-order MPS statement (8.5)]
Same data, chi = 4, raw features, 5 seeds (scrambled: 3 orders x 5 seeds at sigma in {0.05, 0.1, 0.2, 0.5},
order 0 at all 7). Train loss and test accuracy at tau = 1e4 (mean +- SE):

| sigma | quadtree L / acc | scrambled L / acc | MPS L / acc |
|---|---|---|---|
| 0.05 | 0.167 ± 0.019 / 0.903 | 0.241 ± 0.008 / 0.841 | 0.090 ± 0.0002 / 0.956 |
| 0.1 | 0.098 ± 0.001 / 0.954 | 0.207 ± 0.008 / 0.872 | 0.089 ± 0.0001 / 0.957 |
| 0.2 | 0.0954 ± 0.00002 / 0.950 | 0.154 ± 0.011 / 0.908 | 0.089 ± 0.0001 / 0.960 |
| 0.5 | 0.090 ± 0.003 / 0.955 | 0.078 ± 0.001 / 0.962 | 0.086 ± 0.0005 / 0.959 |

Time to reach L <= 0.2 (median; runs reaching it): sigma = 0.1: quadtree 333 (5/5), scrambled 6730 (7/15),
MPS 56 (5/5); sigma = 0.5: 85, 374, 28.
- **Where the residual interaction sits (route amplitudes at the rank-1 plateau).** Share of sum_uw lambda_uw^2
  on routes of degree k = 3 / 5 / 7 (4 / 8 / 16 of the 28 pairs): quadtree 0.15 ± 0.02 / 0.73 ± 0.03 /
  0.12 ± 0.01 (35 runs); scrambled 0.026 ± 0.005 / 0.19 ± 0.03 / 0.79 ± 0.03 (89 runs, 3 orders). The
  strongest single route is equally strong in both (median max lambda 0.30 vs 0.33), but the best k = 3
  route is 2.2x stronger in the quadtree (0.11 vs 0.049). The quadtree puts 88% of the interaction on
  routes with k <= 5. The scrambled tree puts 79% on k = 7 routes (sigma^-5 plateaus), more than their 57%
  share of pairs. [fix 1: SEs pooled one critical point over sigma; one value per (order, seed): 0.89 ±
  0.015 vs 0.17 ± 0.07, and the three scrambled orders are the low tail of random orders (chance 0.43), 8.4]
- **Quadtree vs scrambled (the clean locality test, same tree shape): the local tree learns much faster
  at small init.** [fix 1: an equally local transposed quadtree is among the worst orders, early learning is
  not faster, and at sigma = 0.5 the advantage does not vanish; narrowed to an order/orientation effect, 8.4] It reaches lower loss and higher test accuracy at every sigma <= 0.2, and the gap grows
  as sigma decreases. Both have the same route degrees (same shape), so the difference sits in where the
  strong label-relevant interactions lie. In the quadtree they lie on short routes (inside bottom cores,
  k = 1, and between nearby blocks, k = 3-5). In the scrambled tree they mostly sit on long routes
  (k = 5-7, sigma^-3 ... sigma^-5 plateaus): its first k = 3 route captures little (loss 0.33 -> 0.29-0.30),
  and the next escapes take thousands of time units. At sigma = 0.5 the escapes are fast for both trees and
  the advantage disappears (scrambled even ends lower at tau = 1e4). **Locality is a small-init dynamical
  effect here, not an expressivity limit at chi = 4.**
- **MPS vs binary: the brief's ordering fails.** [fix 1: only on the tau = t - t_a clock, on MNIST, with the
  snake order; slower on the absolute clock, slowest on Fashion, shape and order confounded; mechanism
  conjecture, 8.5] The snake MPS reaches every loss level first at every sigma
  and its learning times are nearly sigma-independent (at sigma = 0.5 the scrambled tree ends slightly lower
  at tau = 1e4: 0.078 vs 0.086). Every MPS core carries a pixel, so its routes are 2-core
  strict saddles (log-time). Binary trees have pixel-free internal cores and need k >= 3 routes
  (power-law plateaus). Route degree (tree shape) dominates locality at small sigma.
- **Bottom-up order:** Kendall tau between an edge's level and the time of its first rank increase is
  positive on every tree (quadtree 0.31 +- 0.02, scrambled 0.17 +- 0.02, MPS 0.27 +- 0.01, raw): ranks grow
  from the leaves towards the root on average (R3), with exceptions. [fix 1: binary trees only; on the MPS
  "level" is the chain position, 8.10]

### 4.8 Regimes (step 5): S2S -> rich -> lazy with the centred model [empirical; theory comparison derived; fix 1: narrowed, 8.8]
Binary quadtree, chi = 4, raw features, 5 seeds per sigma. Centred model (train f - f_0) at sigma = 0.5-4
(horizon min(3000, 1e4 kernel units), kernel on 128 test images). At sigma <= 0.3 the plain-model runs of
the main sweep are used (centring is immaterial there: f_0 ~ sigma^15; sigma = 0.5 overlaps and agrees);
their delta_K is computed post hoc from theta_0 and theta(T) on the same probe images. Labels: lazy if
delta_K(T) < 0.1, else S2S if >= 2 loss drops with all rho > 3, else rich. The notes' convergence rule
(L < 1e-3 L0) is dropped: no chi = 4 TTN can reach it on 4000 labels (`regime_analysis.py`).

| sigma | labels (5 runs) | delta_K(T) | train L(T) | linearized L_lin(T) | test acc |
|---|---|---|---|---|---|
| 0.05-0.3 (plain) | S2S 27/30, rich 3/30 | 1e13-1e41 | 0.095-0.17 | 0.500 | 0.90-0.96 |
| 0.5 (centred) | S2S 2, rich 3 | 1e13-1e15 | 0.095 | 0.500 | 0.954 |
| 0.7 | S2S 3, rich 2 | 1e9-1e11 | 0.085 | 0.500 | 0.961 |
| 1.0 | S2S 3, rich 2 | 1e5-1e7 | 0.080 | 0.465 | 0.961 |
| 1.4 | rich 5 | 13-1100 | 0.083 | 0.166 | 0.962 |
| 2.0 | lazy 4, rich 1 (delta_K = 0.104) | 0.006-0.10 | 0.150 | 0.151 | 0.931 |
| 2.8, 4.0 | lazy 5 + 5 | 2e-7-6e-4 | 0.151 | 0.151 | 0.931 |

- **Lazy regime exists and is exactly the linearized model** (sigma >= 2): L(T) equals the closed-form
  L_lin(T) of the model linearized at theta_0 to three digits, and runs at sigma = 2.8 and 4.0 give
  identical losses per seed (kernel units). The tangent kernel's rank is 304 of 528 parameters, exactly
  sum_v F_v chi_v - sum_e chi_e^2 (`theory_regimes.md` R2). The lazy floor L_lin(inf) = 0.105 lies well
  above what feature learning reaches (deficient bonds: chi = 4 << 256, R7). [fix 1: withdrawn per seed
  (floors 0.090-0.128; small-init runs end below it for 4 of 5 seeds); the lazy runs are not converged at
  their horizon (L(T) 0.037-0.064 above their floors), 8.8]
- **Boundaries.** S2S at small sigma (plateaus growing as sigma -> 0, section 4.6); a mixed S2S / rich zone at
  sigma = 0.5-1.0; rich, non-incremental at sigma = 1.4; lazy from sigma = 2. [fix 1: "rich" rests on sigma =
  1.4 alone (2/5 runs without any detected drop); the S2S / rich boundary moves over sigma = 0.07-1.0 with the
  separation threshold, 8.8] In function-space terms the
  kernel scale at init, tr Theta_0 / 128, is ~3e-5 at sigma = 1 and ~1e4 at sigma = 2: raw cos/sin features
  have |Phi| = 1 spread over 2^16 dimensions, so per-core sigma = 1 is still a small init in function space.
  Threshold sensitivity: delta_K threshold 0.05 / 0.2 gives 2/5 / 5/5 lazy at sigma = 2 (others unchanged);
  separation threshold 2 / 5 moves S2S <-> rich labels at sigma = 0.07-1.0 (`results/regime_analysis.json`).
- **Generalization:** rich runs (sigma = 1-1.4) have the best test accuracy (0.961-0.962), S2S runs 0.95
  (lower at sigma < 0.1, where the horizon cuts the later escapes), lazy runs 0.931 (they fit only the
  304-dimensional tangent space).

### 4.9 Whitened site features [empirical]
Per-site whitening (uncentred second moment) raises the function-space init scale ~256x at equal sigma.
The first escape is asymptotic only at smaller sigma: per-seed t1 exponents 12.91 ± 0.04 (binary quadtree),
12.83 ± 0.05 (scrambled), 13.83 ± 0.09 (MPS) over sigma = 0.05-0.5, against 13 / 13 / 14. At sigma = 0.5
the run is rich (one drop to L ~ 0.07, all ranks at once). At sigma <= 0.1 binary runs are S2S (80% of runs).
[fix 1: 17/30 = 57% (quadtree 11/15, scrambled 6/15)]
The first route is again a sibling pair (k = 3) in most runs, but the classified gap exponents are weaker
and noisier (0.6 ± 0.3, 2 seeds with a consistent route). Whitening brings the site marginals closer to the
theory's product of identities, but it does not make the dynamics cleaner: it amplifies near-constant
pixels and moves the small-init asymptote to smaller sigma.

## 5. Teacher-student prediction -> real data

Updated in fix round 1 (section 8; the round-0 table is in the git history). Counts are per (order, seed) or
per critical-point unit, not per run.

| prediction (source) | observed on MNIST 3 vs 5, 4x4, chi = 4 | verdict | evidence |
|---|---|---|---|
| first plateau t1 ∝ sigma^-(K-2) for any input law (theory_covariance C4) | t_a sigma^(K-2) constant per seed (exponents 13.000 / 14.000); an identity of small-init flow, so an integrator check, not evidence | identity (proven) | `scaling.json`, `scaling_fix1`, 8.10 |
| bond ranks grow stepwise, one route at a time (Q1) | discrete escapes of s_k/s_1 from 1e-5-1e-3 to O(1); one negative Hessian direction at most binary small-sigma plateaus | yes | `bond_spectra_raw_*`, `plateau_hessians` |
| flat plateaus at rank-constrained critical points | plateaus creep: between drops the loss falls by a median 0.76 of the next drop; S2S signature holds with rate-peak drops, rho grows as sigma -> 0 | partly | `tables.md`, 4.6, 8.11 |
| binary trees at rank 1: next route = two sibling bottom cores, k = 3 (R2) | all 20 (order, seed) pairs; siblings cross together in 96/96 rank-1 states | yes | `fix1_analysis.md`, 8.3 |
| MPS: next route on one edge, k = 2 | edge 5 alone in 25/25 resolved runs, but before the first drop's peak (merged with the escape); 10/35 recorded "increases" are the threshold at tau = 0 | partly | `fix1_analysis.md`, 8.6 |
| k = 3 gap ∝ sigma^-1 (H2) | A/sigma + B fits all 11 clean series (rms <= 0.06, B < 0); free exponent 1.22 ± 0.03 (offset-dominated); log law rms 0.11-0.30 | yes, as A/sigma + B | `first_route_gaps`, 8.1 |
| k = 2 gap ∝ log(1/sigma) | later MPS edges, relative-residual fits: edge 4 slope 14.7 ± 1.5 (log rms 0.027 vs power 0.36; 4/5 seeds), edge 3 30.3 ± 0.8 (0.029 vs 0.25; 5/5); edge 2 ambiguous, edge 1 power-like; edges 0, 6-9 pending | yes for edges 4, 3 (later routes) | `scaling_fix1`, 8.6 |
| k >= 4 later routes: sigma^-(k-2) | route identification of late binary events not clean | not established | 8.10 |
| strict saddle iff k = 2 (H3), via lambda_min scaling | MPS (after the first route): exponent 0.047 ± 0.016 (strict); binary rank-1 plateau 0.95 ± 0.07 (higher order, predicted 1) | yes | `plateau_hessians`, 8.7 |
| next route = lowest degree, then largest amplitude (R8) | 22/26 units hit in all states, 2 mixed, 2 miss (chance 1/4); 11/11 review runs; larger-amplitude k = 5, 7 routes escape later | yes | `route_amplitudes`, 8.3 |
| bottom-up rank growth (R3) | Kendall tau(level, time) > 0 on binary trees (0.17-0.31) | partly (a tendency) | `tables.md`, 8.10 |
| locality: local pixel orders learn faster | quadtree lowest final loss at every seed (sigma = 0.1), share of amplitude on k <= 5 routes predicts the final loss (rho = -0.72, n = 30); an equally local transposed quadtree is among the worst; early learning not faster | no as locality; an order effect measured by route amplitudes | `locality_orders`, 8.4 |
| MPS between quadtree and scrambled (brief) | tree shape and pixel order confounded; snake MPS fastest on the tau clock (MNIST), later on the absolute clock, slowest on Fashion | undetermined | 8.5 |
| regimes: S2S -> rich -> lazy with sigma (H4) | S2S at small sigma; rich at one sigma (1.4); lazy (>= 2) = linearized model, kernel rank = R2 formula; boundaries threshold-dependent; lazy runs unconverged | partly | `regime_sweep`, 8.8 |

## 6. Caveats and open points
- One dataset pair (MNIST 3 vs 5) at 4x4 for the scaling claims; chi = 8, Fashion-MNIST (sneaker vs boot) and
  8x8 checks in section 7 support the structural results but have too few sigma values for exponent fits.
- Events are judged from loss-rate peaks and threshold crossings of data-weighted spectra (s_k/s_1 = 0.05), not
  from teacher overlaps; later events can merge and their routes are only partly identified.
- The horizon tau = 1e4 cuts the later escapes at sigma < 0.1 (k >= 5 routes need ~1e4-1e5), which lowers the
  small-sigma test accuracy in the locality and regime tables: those numbers are at fixed training time.
- The binary first-plateau lambda_min exponent (0.88) is below the k = 3 value (1); the MPS far-edge rises scale
  like power laws (exponent ~1) where rank-1 counting predicts k = 2; both are open. [fix 1: with seed 4 the
  binary exponent is 0.95 ± 0.07 (consistent with 1); with fair fits MPS edge 2 is a tie and edge 1 still
  power-like, edges 0, 6-9 pending (8.6, 8.7)]
- [fix 1] Pixel order and tree shape are confounded for the MPS (snake only), and the locality comparison
  rests on four committed orders plus review runs at one sigma (8.4, 8.5).
- Gradient flow only (no GD/SGD); empirical loss on 4000 fixed samples.

## 7. Robustness checks

- **Integrator accuracy** [empirical]. Gauge conservation drift (cumulative, relative to max core norm^2):
  max 4.6e-8, median 1.8e-9 over the 234 main runs. Two cases rerun at rtol 1e-10, atol 1e-16 (quadtree and
  MPS, sigma = 0.1, seed 0): t1 agrees to 7 digits, the final train loss to 6 digits, later drop times to
  < 0.1% (2157.79 vs 2157.85, 4743.3 vs 4740.8); the first drop (tau ~ 10-15) moves by 0.2-0.8 time units,
  well below every gap reported (`/root/outputs/real_data/tol_check`).
- **Second dataset: Fashion-MNIST sneaker (7) vs ankle boot (9), 4x4** [empirical; 45 runs: 3 trees x
  sigma in {0.05, 0.14, 0.5} x 5 seeds; `results/fashion_tables.md`]. Structure as on MNIST: t1 exponents
  13.000 / 13.000 / 14.000 ± 0.000; the first two edges to gain rank are siblings in 15/15 quadtree and 14/15
  scrambled runs; the time of that first rank increase scales per seed as sigma^-0.92 ... -0.97 between
  sigma = 0.05 and 0.14 (k = 3; only two sigma values share the route, so this is not a scaling claim).
  [fix 1: the siblings cross together in only 3/15 and 3/15 runs (a single edge first in 12/15); the slopes
  are in no committed output and are withdrawn (8.9)] New: at
  small sigma the loss drops twice before any rank grows (0.50 -> 0.33 -> 0.12, both drops at bond rank 1).
  The flow moves between two rank-1 critical points, the analogue of a wrong first escape corrected at rank 1
  (theory_saddles.md 4.4). The Hessian at that plateau (L = 0.31-0.345, 27 runs, sigma = 0.05-0.5) has a
  sigma-independent negative eigenvalue (lambda_min = -0.04 ... -0.25; `saddle_scaling` exponent
  0.03 ± 0.09 projected, 0.02 ± 0.09 raw: strict). So the plateau is a strict saddle left by a rank-preserving rotation [empirical;
  `results/fashion_hessians.json`]. Locality: **no advantage** for the quadtree;
  the task is almost solved by a rank-1 (product) function (L = 0.12, acc 0.92). Scrambled is as fast or
  faster (sigma = 0.05: L(1e4) 0.120 vs 0.123), and the MPS is slower at first but reaches the lowest loss
  (0.089, acc 0.945). Locality matters only as much as the label depends on pixel interactions that the
  pixel order can place on short routes.
- **Bond dimension chi = 8** [empirical; 45 runs: quadtree, snake MPS, scrambled0 x sigma in {0.05, 0.14, 0.5}
  x 5 seeds; 23 runs stopped at the 2 h wall limit, horizon tau = 4700-1e4 (median 9800); `results/chi8_tables.md`].
  Same picture as chi = 4. t1 exponents are 13.000 / 13.000 / 14.000. The first two rising edges are siblings in
  14/15 quadtree and 14/15 scrambled runs, with the same sibling pair per seed as at chi = 4. The MPS's first
  rise is a single edge in 15/15. First-route gap between sigma = 0.05 and 0.14, per seed: quadtree
  1.17 ± 0.06 (5 seeds), scrambled 0.97 ± 0.11 (k = 3: 1); at sigma = 0.5 the events merge (exponent
  0.05-0.5: 0.79 ± 0.28). [fix 1: these numbers are in no committed output and are withdrawn; committed
  rank-clock two-point slopes 1.27 ± 0.08 / 1.20 ± 0.06; 4 of the 15 MPS first rises are at tau = 0; all 23
  wall-limited runs are quadtree or MPS, so the sigma = 0.5 comparison below is withdrawn (8.9)] Locality as at chi = 4: quadtree vs scrambled L(tau_end) = 0.158 vs 0.220
  (sigma = 0.05), 0.096 vs 0.176 (0.14). At sigma = 0.5 scrambled fits the training set better (0.055 vs
  0.079) but generalizes worse (test loss 0.111 vs 0.088). The MPS is fastest at small sigma (0.088 / 0.080).
- **8x8 images (N = 64; binary64 K = 63, mps64 K = 64)** [empirical; 12 runs: 3 trees x sigma in {0.1, 0.3} x
  seeds {0, 1}, tau <= 3000; `results/img8_tables.md`]. t1 exponents 61.0000 (binary, 4 seed-tree pairs) and
  62.0000 (MPS), with t1 up to 1e73 (re-anchored clock). The first two rising edges are siblings in 4/4
  quadtree and 4/4 scrambled runs; the MPS starts on a single edge (edge 20) in 4/4. [fix 1: one scrambled
  permutation, two seeds; the table's "L(1e4)" is L(3000) (8.9)] Locality at tau = 3000:
  quadtree L = 0.16-0.25 (test acc 0.84-0.91) vs scrambled 0.25-0.27 (0.83-0.86); the MPS is again best
  (0.08-0.12, acc 0.95-0.98). Too few runs for scaling fits; the structure is unchanged at N = 64.

## 8. Fix round 1 (local, 2026-10-10)

Response to the round-1 review (`notes/reviews/real_data_round1.md`, MUST/SHOULD FIX lists). This section
supersedes sections 4-7 where they differ; superseded statements there are marked "[fix 1: ...]".

**How the numbers were obtained.** The remote raw outputs (`/root/outputs/real_data`) were not available, so:
(i) everything that the committed per-run or per-series files allow was recomputed from them by
`fix1_analysis.py` (`results/fix1_analysis.{json,md}`; inputs `events_table.json`, `routes_raw.json`,
`hessians_raw.json`, `regime_analysis.json`, `scaling.json`, `chi8_scaling.json`, `tables.json`, the remote run
logs); (ii) two light subsets of `main.yaml` were rerun locally with the committed code, settings and seeds
(`configs/fix1_local.yaml`: the four binary orders at sigma = 0.1, 5 seeds, tau = 1e4, plus the MPS first
escape at sigma = 0.1; `configs/fix1_mps.yaml`: the 35 MPS cases with tau <= 1000); (iii) the round-1
reviewers' runs with this code (`data/review_real_data/audit`: six more random binary orders, row-major and
transposed-quadtree binary trees, MPS chains in other pixel orders; sigma = 0.1, seeds 0-1) enter where
stated, as "review runs". The reruns reproduce the committed values (section 8.9). Code fixes: first-route
attribution (`scaling.first_route_points`), relative-residual fits for all three laws (`scaling.fit_seed`,
`events_table.fit`), rank events already met at tau = 0 flagged (`summarize.py`), sibling parent check,
route-amplitude units (`analysis.route_units`), honest horizon labels (`tables.py`), `REAL_DATA_DIR`.

### 8.1 The k = 3 first-route gap (section 4.4) [fix: attribution bug, fit weighting]
- **Old:** "gap ∝ sigma^-1: quadtree 1.01 ± 0.14 (4 seeds), scrambled0 0.97 ± 0.11, scrambled1 1.27 ± 0.02,
  scrambled2 0.91 ± 0.23 (2 seeds); relative rms power 0.03-0.14 vs log law 0.56-0.91".
- **Bug.** Round-0 `first_route` took the first drop after drop 0 whose window *ends* after the route's rank
  crossing, without checking that the crossing lies *inside* that window. Where the route rose during the
  creep between drops or merged with drop 0, the gap of a later, different route was used: quadtree seed 3,
  sigma = 0.3 (route rose 12.4 after drop 0 inside no window; 128.9 assigned), sigma = 0.5 (merged with drop 0;
  34.4 assigned); scrambled2 seed 0, sigma = 0.5 (rose at 141.5; 560 assigned). These pulled seed 3's
  exponent from 1.24 to 0.65 and scrambled2's mean to 0.74 ± 0.21 (3 seeds; the "0.91 ± 0.23 (2 seeds)" of
  section 4.4 was stale even then). The fixed rule (route's drop = the drop whose widened window contains the
  crossing; "merged" and "between drops" excluded with their reason) gives the right answer on the reviewers'
  reruns of all three cases (checked).
- **New (clean series).** One series per (order, seed) whose first route (one sibling pair) rises inside
  drop 1 at every sigma of the series (events_table signatures; >= 4 sigma over >= 0.5 decade): 11 series.
  Free power-law exponent 1.22 ± 0.03 (mean ± SE over the 11 series; range 1.06-1.36): quadtree 1.18 ± 0.07
  (4 seeds), scrambled0 1.23 ± 0.07 (2), scrambled1 1.27 ± 0.02 (4), scrambled2 1.13 (1). The k = 3 form
  A/sigma + B (exponent fixed at 1) fits every series with relative rms <= 0.060, with B < 0 in 11/11
  (B = -5 to -41 time units). The k = 2 log law is rejected: relative rms 0.11-0.30 with the same (relative)
  weighting, against 0.002-0.058 for the power law (round 0 fitted it unweighted: 0.15-1.20). **[empirical;
  the form A/sigma + B is the derived k = 3 law with its O(1) correction, theory_saddles.md 3.3; the sign and
  size of B are empirical]**
- **The free exponent is not a robust statistic here.** With B < 0 the local slope of A/sigma + B is
  1/(1 + B sigma / A) > 1 and grows with sigma; at sigma = 0.3-0.5 the offset is 30-60% of A/sigma. The same
  runs give different free exponents for a shift of the clock origin by O(10) time units: quadtree seed 0
  (reviewers' reruns, 5 sigma values 0.05-0.5) 1.02 on the drop-to-drop clock and 1.26 on the rank-crossing
  clock (crossing - drop 0; the crossing precedes the drop's rate peak by ~8.5). Hence "gap ∝ sigma^-1.0"
  is withdrawn as a statement about a fitted exponent and replaced by "A/sigma + B fits, log law does not".
- **scrambled1 (1.27 ± 0.02, 13 SE above 1).** Explained by the offset: its four series have B/A = -0.83 ...
  -1.29 (in units of 1/sigma), and three of them reach sigma = 0.5, where B sigma/A ~ -0.5. Restricted to the sigma grid that
  main.yaml generates (0.05, 0.1, 0.2, 0.5) the exponents are 1.34, 1.28, 1.22, 1.23, so the 14 runs main.yaml
  does not generate (sigma = 0.07, 0.14, 0.3, kept from before the grid trim) do not drive the value; they are
  now reproducible with `configs/main_extra.yaml`. **[empirical]**
- Section 4.4's other gap statements: the per-series refit with fair weights is in `results/events_table.md`
  (all 36 series) and `results/fix1_analysis.md`. Two corrections: the third singular value on the first
  sibling pair ((3,3),(4,3), drop 2, quadtree) is better fitted by the log law in seeds 2, 3 (rms 0.019, 0.037
  vs power 0.12, 0.12), equally in seed 0 (0.007 vs 0.008), by neither in seed 4; the table's "route k"
  (edges + 1) is a naive count for such later events, not a route identification. **[empirical; open]**

### 8.2 Fit weighting (sections 3, 4.4, 5) [fix]
- **Old:** "Power and log laws are fitted in linear-gap space with relative residuals" (not what the code did:
  power law in log space, log law unweighted in linear space, both scored by relative residuals).
- **New:** all laws are fitted with relative residuals (power law: least squares in log g; log law and
  A/sigma + B: weighted least squares, weights 1/g). Log-law rms on the clean k = 3 series 0.11-0.30 (not
  0.56-0.91); the verdicts (power / A/sigma + B over log) hold. The MPS per-edge fits are redone in 8.6 for the
  edges reached within tau = 1000; the rest of `scaling.json` (edge_scaling, g2, grank) is still the round-0
  fit (pending, 8.12). **[fix of method]**

### 8.3 First route and route-amplitude rule, de-duplicated (sections 4.2, 4.3) [fix: pseudo-replication]
- **Old:** "first two rising edges are siblings in 35/35 quadtree and 89/89 scrambled runs; cross together in
  35/35 and 86/89"; "the largest sibling amplitude picks the first route in 34/35 (quadtree) and 78/89
  (scrambled) runs".
- **Why it changed.** Runs of one (order, seed) at different sigma start from the same init direction and sit
  at the same rank-1 critical point unless something switches, so they are one trial, not seven. Of the 124
  route-amplitude states, 28 are farther than 0.05 (relative) from a rank-1 product (up to 0.995, mostly
  sigma >= 0.3); the amplitude formula assumes rank 1, and two of them (quadtree seed 3, sigma = 0.3, 0.5)
  were taken after the route had already risen but were counted as hits.
- **New.** Units = (order, seed, critical point): rank-1 states (error <= 0.05) of the same (order, seed)
  linked when their normalised lambda^2 patterns differ by < 0.2 (total variation) and their plateau losses
  by < 0.01 (`analysis.route_units`). 96 states form 26 units in 20 (order, seed) pairs. The sibling pair with
  the largest amplitude is the first pair to gain rank in all states of 22 units, in some states of 2 units
  (quadtree seed 4 at sigma 0.14-0.3; scrambled1 seed 4 at 0.14-0.2) and in none of 2 units (scrambled0 seed 3
  at 0.14-0.2; scrambled2 seed 2 at 0.1); chance is 1/4 per unit (4 sibling pairs). Per state: 27/28
  (quadtree), 64/68 (scrambled). Out of sample (the 11 binary review runs: quadtree, six more random orders,
  row-major and transposed quadtree at seed 0, the last two also at seed 1): 11/11. **[empirical]** (The review counted 28/30 units with its own
  grouping; the conclusion is the same.)
- **First route = sibling pair.** The first two rising edges are siblings in all 124 runs, i.e. in all 20
  (order, seed) pairs; in the 96 rank-1 states the two sibling edges cross 0.05 together (within 5%) in 96/96.
  The 3 runs where they do not are at sigma >= 0.3 with a non-rank-1 plateau state (the route merged with the
  escape). The parent check added to `tables.sibling_pair` cannot change these MNIST counts (all first pairs
  are siblings); for Fashion and chi = 8 it can lower the "rise together" counts by at most one run per tree
  (pending recount). **[empirical]**

### 8.4 Locality (section 4.7) [narrowed: an order/orientation effect measured by route amplitudes]
- **Old:** "Quadtree vs scrambled (the clean locality test, same tree shape): the local tree learns much
  faster at small init ... quadtree puts 88% of the residual interaction on k <= 5 routes, scrambled 21%
  (± 0.02-0.03) ... gone at sigma = 0.5. Locality is a small-init dynamical effect."
- **What failed.** (a) An equally local order is among the worst: the transposed quadtree (same 2 x 2 blocks,
  vertical instead of horizontal pairs) ends at L(1e4) = 0.226 / 0.194 (seeds 0 / 1, sigma = 0.1) against
  0.096 / 0.099 for the quadtree; row-major gives 0.136 / 0.099 (review runs). (b) The "21%" baseline came from
  3 random permutations in the low tail: per (order, seed) at sigma = 0.05 the k <= 5 share is 0.89 ± 0.015
  (quadtree, 5 seeds) and 0.17 ± 0.07 (15 = 3 orders x 5 seeds; per order 0.34 / 0.11 / 0.06), while six more
  random orders give 0.02-0.88 (mean 0.42, seed 0, sigma = 0.1), the pair-count chance level being 12/28 = 0.43.
  (c) The quoted SEs pooled the same critical point over sigma (one value per (order, seed) is used now).
  (d) Early learning is not faster: at seed 0, 6 of the 8 other binary orders reach L <= 0.25 before the
  quadtree. (e) At sigma = 0.5 the effect does not vanish: the quadtree still reaches each loss level 3-5x
  sooner than the committed scrambled orders (L <= 0.2 at tau 85 vs 374, L <= 0.15 at 134 vs 642; medians),
  but ends higher at tau = 1e4 (0.090 ± 0.003 vs 0.078 ± 0.001).
- **New.** Per (order, seed) at sigma = 0.1, tau = 1e4: 30 runs = the four committed orders x 5 seeds
  (main.yaml cases rerun locally) + 10 review runs (`results/fix1_analysis.md`, figure `locality_orders`). The
  init cores of a seed are the same for every binary order (same tree shape), so orders compare in pairs.
  (1) The final loss depends strongly on the pixel order, and the quadtree order ends lowest at every seed:
  0.098 ± 0.001 (5 seeds; 0.096-0.103) against 0.188 ± 0.006, 0.207 ± 0.021 and 0.226 ± 0.004 for the
  committed random orders (5 seeds each), 0.119-0.224 for six more random orders (seed 0), 0.136 / 0.099 for
  row-major and 0.226 / 0.194 for the transposed quadtree (seeds 0 / 1). No other order ends below the quadtree
  of the same seed (0 of 25 runs). **[empirical]**
  (2) The share of sum lambda^2 on k <= 5 routes at the first plateau predicts the final loss across orders:
  Spearman rho = -0.72 (p = 8e-6, n = 30); without the quadtree -0.56 (p = 0.003, n = 25); within random orders
  -0.44 (p = 0.045, n = 21); within the three committed random orders alone -0.30 (p = 0.28, n = 15). Units of
  the same seed share the init draw, so the p-values are indicative. **[empirical]**
  (3) Speed: other orders reach L <= 0.25 before the quadtree in 9 of 25 same-seed comparisons (6 of 11 at
  seed 0); the quadtree's advantage is in the later escapes (median tau to L <= 0.2: 333 vs 4857 and 7316 for
  scrambled0 and scrambled1; scrambled2 never reaches it). **[empirical]**
  (4) Mechanism **[conjecture]**: the pixel order decides which label-relevant pixel interactions lie on short
  routes (k <= 5, escaping within the horizon) and which on k = 7 routes (sigma^-5 plateaus, not reached);
  whether the order is local in the image does not decide it (transposed quadtree). The claim is renamed
  "pixel-order (orientation) effect, measured by route amplitudes"; untested: more local orders and more seeds
  (the review's optional runs were not made in this round).
- Withdrawn: "locality is a small-init dynamical effect" as a statement about locality. Kept: the gap between
  orders at small sigma is not an expressivity limit at chi = 4, since at sigma = 0.5 the committed scrambled
  orders end lower than the quadtree **[empirical]**.

### 8.5 MPS vs binary trees ("the MPS is fastest at every sigma", sections 4.7, 5; REPORT) [narrowed]
- **Old:** "MPS vs binary: the brief's ordering fails. The snake MPS reaches every loss level first at every
  sigma ... Every MPS core carries a pixel, so its routes are 2-core strict saddles (log-time) ... Route degree
  (tree shape) dominates locality at small sigma"; REPORT: "fails on every task and size".
- **New.** (1) On the tau = t - t_a clock (time after the first escape), MNIST 3 vs 5, snake order, the MPS
  reaches L <= 0.25 / 0.2 / 0.15 / 0.12 first at every sigma (medians over 5 seeds; sigma = 0.1: 16 / 56 / 290 /
  377 vs quadtree 122 / 333 / 3730 / 3810; `tables.md`). **[empirical]**
  (2) On the absolute clock t = t_a + tau the first escape dominates (t_a >= 1e5 >> tau <= 1e4 at every sigma <= 0.5).
  t_a ∝ sigma^-(K-2) per init with K = 16 (MPS) vs 15 (binary), so t_a(MPS) / t_a(binary) ∝ 1/sigma with
  init-dependent prefactors **[proven scaling]**: at sigma = 0.1 the ratio to the quadtree is 7.6 / 0.82 / 131 /
  2.0 / 6.4 for seeds 0-4 (fix1 reruns), so the MPS escapes later in 4 of 5 seeds at sigma = 0.1 and in all 5 at
  sigma <= 0.07; its lead on the tau clock (~1e2) is negligible against these differences (1e14-1e16).
  **[empirical prefactors]**
  (3) Fashion-MNIST: the MPS is the slowest tree to L <= 0.25 / 0.2 / 0.15 at sigma = 0.05 (117 / 124 / 374 vs
  quadtree 35 / 37 / 44 and scrambled 21 / 22 / 26; `fashion_tables.md`), although it ends lowest. **[empirical]**
  (4) Tree shape and pixel order are confounded (only snake chains in the sweeps). Review runs (sigma = 0.1,
  seed 0) with other chain orders: row-major ends at 0.083; scrambled0 at 0.084 but reaches L <= 0.25 only at
  tau = 299 (quadtree 156), with a 3-core first route and staircase-like plateaus (rho = 8.5-30); scrambled3 ends
  at 0.128, above the quadtree's 0.096. **[empirical, one seed]**
  (5) The mechanism "every MPS route is a 2-core strict saddle (log time)" is not established: the first MPS
  route is merged with the escape and the log law holds for some later edges only (8.6). **[conjecture]**
  Withdrawn: "the MPS reaches every loss level first at every sigma" (clock-dependent), "route degree (tree
  shape) beats locality", and the REPORT's "fails on every task and size". The brief's prediction "MPS between
  quadtree and scrambled" is undetermined (shape and order confounded), not refuted.

### 8.6 MPS first route (section 4.2, table row "MPS: next route on one edge, k = 2") [yes -> partly]
- **Old:** "the first rank increase is on a single edge (edge 5) in 34/35 runs. It happens during or right
  after the first escape: a strict 2-core route with a large rate"; verdict "yes", with the k = 2 log law and
  the strict Hessian as its evidence.
- **New.** 35 reruns of the main.yaml MPS cases with tau <= 1000 (`configs/fix1_mps.yaml`; they reproduce
  the committed full-horizon timing of the first rise relative to the first drop within 0.07 time units,
  median 0.01). (1) In 10/35 runs (seed 4 at every sigma, seed 0 at sigma >= 0.2) the data-weighted s2/s1 of
  edge 5 is >= 0.05 already at the start of phase 2: the recorded "first increase" is the threshold at
  tau = 0, not a resolved event (round 0 counted these as "single edge first"; `summarize.py` now flags
  them). (2) In all 25 resolved runs the first increase is edge 5 alone, 0.05-1.96 time units *before* the
  first drop's rate peak (24/25 inside its window): the first MPS route is merged with the escape from the
  origin, so no gap is measured for it and the k = 2 log law is not tested on it. **[empirical]** (3) The log
  law is measured on the next edges. With relative-residual fits for both laws (5 seeds, sigma = 0.05-0.5):
  edge 4 rises at 14.7 ± 1.5 per unit ln(1/sigma) after the first drop (log rms 0.027 vs power 0.355; 4/5
  seeds prefer the log law, the fifth has power exponent 0.35 over one decade, nearly logarithmic) and edge 3
  at 30.3 ± 0.8 (0.029 vs 0.254; 5/5); edge 2 is ambiguous (0.141 vs 0.131; 1/5 prefer log) and edge 1
  power-like (p = 1.25 ± 0.10; 0.113 vs 0.272; 0/4). **[empirical]** Edges 0 and 6-9 rise mostly after
  tau = 1000 and are not refitted (pending, 8.12; with the round-0 unweighted fits only edge 6 favoured the
  log law). The review's "only 2 of 9 MPS edges prefer the log law" therefore stands for the refitted edges
  (3, 4 yes; 2 tie; 1 no). (4) The strict Hessian of 4.5 is measured on the plateau after edge 5 has risen
  (8.7), so it belongs to the later routes. Verdict: **partly** (a single-edge first increase, but merged with
  the escape; log time holds for edges 4, 3).

### 8.7 Saddle type at the first plateau (section 4.5) [binary exponent corrected; MPS state re-described]
- **Binary (old: "lambda_min ∝ sigma^0.877 ± 0.005 per seed (3 seeds) ... measured 12% lower").** Seed 4 was
  left out because it has only 3 sigma values at the common plateau (L = 0.27). Per seed: 0.871 (seed 0,
  sigma 0.05-0.3), 0.888 (seed 2, 0.05-0.3), 0.873 (seed 3, 0.05-0.2), 1.149 (seed 4, 0.05-0.1): 0.95 ± 0.07
  (4 seeds), each over less than one decade (0.3-0.78 decades). Consistent with the k = 3 prediction 1; the
  "12% low" statement is withdrawn. Higher order (exponent far from 0) stands. **[empirical]**
- **MPS (strict, sigma^0.04 ± 0.03).** Per seed 0.045, 0.044, -0.009, 0.078, 0.078: 0.047 ± 0.016 (5 seeds).
  Strict stands, but the state is the plateau *after* edge 5 has risen (the first MPS route is merged with
  the escape, 8.6), and it has 2-3 negative directions at sigma <= 0.1 (several active 2-core routes): it is
  the saddle of the later MPS routes (edges 4, 6, ...), not of the first one. **[empirical]**

### 8.8 Regimes (section 4.8) [narrowed]
- **"Rich" at sigma = 1.4.** All five runs are labelled rich, but with 1, 1, 0, 1, 0 detected drops: two runs
  have no drop and are "rich" because delta_K >= 0.1 and fewer than 2 drops (and their horizons are 405-417, not
  3000, kernel-time capped). All five end below their own lazy floor (0.078-0.090 vs 0.090-0.128), so they are
  not lazy; that is the evidence for "rich", which rests on this one sigma value. **[empirical]**
- **S2S / rich boundary.** Not sharp: with the separation threshold rho > 2 nearly all runs up to sigma = 1 are
  S2S; with rho > 5 the S2S fraction is 1/5 already at sigma = 0.2-0.3 and 0/5 at 0.5. The boundary moves over
  sigma = 0.07-1.0 with the threshold (`results/fix1_analysis.md`). **[empirical]**
- **Lazy runs are not converged.** At their horizon (1e4 kernel units) L(T) = L_lin(T) = 0.130-0.192, 0.037-0.064
  above their own floors L_lin(inf) (0.090-0.128). The lazy test accuracy 0.931 and "they fit only the
  304-dimensional tangent space" are statements at that fixed kernel time, not about the lazy limit. **[empirical]**
- **"The lazy floor (0.105) lies well above what feature learning reaches": withdrawn.** The floor is per seed
  0.128 / 0.112 / 0.096 / 0.090 / 0.099 (seeds 0-4). The small-init (plain) runs at sigma = 0.1-0.3 end at
  0.0953-0.103 at tau = 1e4: below the floor for seeds 0, 1 (4/4 each) and seeds 2, 4 (3/4 each), never for
  seed 3 (0.0954-0.0981 vs 0.0902). The centred rich runs at sigma = 1-1.4 end below every seed's floor
  (0.078-0.090 at T = 3000). **[empirical]**
- Unchanged: the lazy regime is the linearized model (L(T) = L_lin(T) to 5-6 digits at sigma = 2.8, 4; 1-1.5%
  at sigma = 2, seeds 0-1) and the tangent kernel has rank 304 = R2 formula **[empirical; formula derived]**.
  The generalization ranking (rich > S2S > lazy) holds at the horizons used only. **[empirical, horizon-dependent]**

### 8.9 Robustness checks (section 7) [narrowed]
- **Fashion-MNIST.** The first two edges to gain rank are siblings in 15/15 (quadtree) and 14/15 (scrambled)
  runs, but under the MNIST criterion (the two siblings cross together) only 3/15 and 3/15: in 12/15 runs per
  tree a single bottom edge rises first and its sibling later. The "sigma^-0.92 ... -0.97 (k = 3)" per-seed
  slopes are in no committed output and are withdrawn; the committed rank-clock gaps give two-point slopes
  (sigma = 0.05 -> 0.14) of 0.95 ± 0.01 (quadtree) and 0.88 ± 0.01 (scrambled0; `results/fix1_analysis.md`), but
  the first-rising edges are not the same at all three sigma values in any seed, so they are not k = 3 route times. Kept: the rank-1
  strict saddle (lambda_min ∝ sigma^0.03 ± 0.09) and no locality advantage. **[empirical]**
- **chi = 8.** 23 of 45 runs stopped at the 2 h wall limit, all quadtree (14/15) or MPS (9/15); every scrambled
  run reached tau = 1e4 (run logs). The table column "train L(1e4)" is L at tau_end (4700-1e4) and is relabelled
  in `tables.py`. The sigma = 0.5 comparison "scrambled fits the training set better (0.055 vs 0.079)" is
  confounded (all five quadtree runs stopped early) and is withdrawn; at sigma = 0.05, 0.14 the quadtree's lower
  loss stands (its runs were the ones cut short). The first-route exponents "1.17 ± 0.06 / 0.97 ± 0.11 / 0.79 ±
  0.28" are in no committed output and are withdrawn; the committed rank-clock gaps (chi8_scaling.json) give
  two-point slopes (0.05 -> 0.14) of 1.27 ± 0.08 (quadtree, 5 seeds; 1.31 ± 0.10 for the 4 seeds with a single
  route) and 1.20 ± 0.06 (scrambled0; 1.21 ± 0.08), descriptive only (two sigma values; `results/fix1_analysis.md`). The MPS "single edge
  15/15" includes 4 runs whose first rise is the threshold at tau = 0 (seed 3 at 0.5, seed 4 at all three).
  **[empirical]**
- **8x8 images.** One scrambled permutation, two seeds, two sigma values, horizon tau = 3000 (the table's
  "L(1e4)" is L(3000); relabelled). "t1 exponent 61.0000" is a two-point identity check. **[empirical, small]**
- **Integrator / reproduction.** Unchanged (drift <= 4.6e-8, 100x tighter tolerances agree). The fix-round
  reruns reproduce the committed main-sweep values: quadtree at sigma = 0.1 (5 seeds): t_a to <= 7e-15 (relative) and L(1e4) to 5 digits
  (0.09572 / 0.09853 / 0.09677 / 0.09812 / 0.10307); the 15 scrambled reruns average 0.207 (committed pooled
  0.207 ± 0.008); the 35 MPS reruns reproduce the committed first-rise timing (median |difference| 0.01, max
  0.07 time units). **[empirical]**

### 8.10 Smaller items
- **t1 ∝ sigma^-(K-2) (4.1, table row 1) [empirical -> proven].** Near the origin L = L0 - <T, W> + O(|W|^2) and
  W is homogeneous of degree K, so the flow from sigma Z is the flow from Z with time rescaled by
  sigma^-(K-2) up to O(sigma^K) relative corrections: t_a sigma^(K-2) is constant per seed (to 4+ digits here).
  It checks the integrator and the clock, and is no longer counted as evidence that teacher-student
  predictions survive on real data. **[proven]**
- **"Later routes have higher degree on binary trees" (4.4) [withdrawn -> open].** It rested on one series
  grouped by drop index (quadtree seed 0, 4th drop, exponent 3.3) and selected per-edge exponents (2.1-2.4,
  labelled "k = 4-5"); a route between two bottom cores of a binary tree has an odd number of cores (3, 5, 7),
  so "k = 4-5" is not a route identification.
- **Bottom-up order (4.7, Kendall tau) [narrowed].** On the MPS the "level" is the position along the chain,
  so its tau (0.27 on MNIST, -0.24 on Fashion) says nothing about bottom-up growth; the claim is restricted to
  binary trees (quadtree 0.31 ± 0.02, scrambled 0.17 ± 0.02, round-0 values pooled over sigma). **[empirical, a tendency]**
- **Whitened features (4.9).** "At sigma <= 0.1 binary runs are S2S (80% of runs)" -> 17/30 = 57% (quadtree
  11/15, scrambled 6/15). **[empirical]**
- **Plateaus creep (4.6) [understated].** Between consecutive drops (from the end of one widened drop window
  to the start of the next) the loss falls by a median 0.76 of the next drop's size (range 0.05-9.6; 56
  intervals in the 20 binary reruns at sigma = 0.1), so much of the decrease happens outside detected drops
  and a route can rise without a drop of its own (8.1). **[empirical]**
- **Bookkeeping.** `results/scaling_raw.json` (cited in 4.4) does not exist: the file is `results/scaling.json`.
  main.yaml generates scrambled1 at 4 sigma values but the analysed sweep has 7 (14 runs kept from before the
  grid trim): `configs/main_extra.yaml` now lists them. `tables.sibling_pair` checks the common parent.
  `data.py` reads `REAL_DATA_DIR`. README lists chi8b.yaml, tol_check.yaml and the Fashion / chi = 8 / 8x8
  commands.
  Phase 1 stops at the first accepted DP5 step below the 1% loss floor, so tau = 0 lies ~0.9 time units after
  the exact crossing (MPS, sigma = 0.1; review); t1 and all gaps are differences on one clock and unaffected.

### 8.11 Unchanged (judged solid by the review, re-checked here)
Binary first route = sibling bottom pair (k = 3) **[empirical]**; largest sibling amplitude goes first (8.3)
and the first-route time scales as 1/amplitude across orders (review: r = -0.95 over 9 orders; growth exponent
of the quadtree's first route 1.35 vs 1.33 predicted for k = 3) **[empirical]**; A/sigma + B beats the log law
for k = 3 (8.1) **[empirical]**; the binary rank-1 plateau is a higher-order saddle, the MPS post-escape plateau
strict (8.7) **[empirical]**; MPS edges 4, 3 rise in log time, edge 4 with slope 14.7 ≈ 1/|lambda_min| (13-18), edge 3 with about twice that (8.6) **[empirical]**;
the lazy regime is the linearized model, kernel rank 304 = R2 formula **[empirical; formula derived]**; plateaus
creep (stronger than stated: between drops the loss falls by a median 0.76 of the next drop, range 0.05-9.6,
56 intervals in the 20 binary reruns) **[empirical]**; Fashion: no locality advantage and a strict saddle
between two rank-1 states **[empirical]**; the pipeline is reproducible (8.9) **[empirical]**.

### 8.12 Pending (needs the remote raw outputs, `/root/outputs/real_data`)
Rerun with the fixed code, unchanged, once the raw folder is back (README "Fix round 1"): `summarize.py` on
main / regime / fashion / chi8 / img8 (rank events at tau = 0 flagged), then `scaling.py` (window-checked
first-route attribution on all 124 binary runs, including the sigma values the clean series leave out, and
relative-residual fits for `edge_scaling`, i.e. the MPS far edges 0 and 6-9 and the binary per-edge exponents),
`tables.py` (parent check, unresolved first rises, L(tau_end) labels, per-unit route table), `events_table.py`
and `make_figures.py` (regenerates `scaling_raw.*` with the fixed attribution; the round-0 file was removed
because its middle panel plotted the misattributed gaps, and `first_route_gaps` and `scaling_fix1` replace it;
`bond_spectra`, `growth_order`,
`loss_curves`, `locality_raw_*` are unaffected by the fixes). Until then `scaling.json`, `tables.*`,
`chi8_*`, `fashion_*` and `img8_*` are round-0 outputs, superseded where this section says so.

### 8.13 Summary of changed claims
| claim (section) | old | new | status |
|---|---|---|---|
| t1 ∝ sigma^-(K-2) (4.1) | evidence that predictions survive, [empirical] | identity of small-init flow (integrator check) | proven |
| k = 3 first-route gap (4.4) | sigma^-1.0, 1.01 ± 0.14 (quadtree) | A/sigma + B fits 11/11 clean series (B < 0); free exponent 1.22 ± 0.03 (not robust) | empirical (form derived) |
| log-law rejection (4.4) | rms 0.56-0.91 | rms 0.11-0.30 with the same weighting; still rejected | empirical |
| scrambled1 / scrambled2 exponents (4.4) | 1.27 ± 0.02 "yes"; 0.91 ± 0.23 | offset effect (B/A ~ -1); scrambled2 one clean series, 1.13 | empirical |
| MPS first route (4.2) | single edge, k = 2 strict, 34/35: yes | merged with the escape, 10/35 at tau = 0: partly | empirical |
| route-amplitude rule (4.3) | 34/35, 78/89 runs | 22/26 units hit, 2 mixed, 2 miss (chance 1/4); 11/11 out of sample | empirical |
| locality (4.7) | quadtree >> scrambled, 88% vs 21%, gone at sigma = 0.5 | pixel-order effect: quadtree lowest final loss at every seed, share predicts loss (rho -0.72, n = 30), not faster early, transposed quadtree among the worst | empirical; mechanism conjecture |
| MPS fastest (4.7, 5, REPORT) | every loss level first at every sigma; fails on every task | tau clock, MNIST, snake only; slower on the absolute clock (4/5 seeds at 0.1), slowest on Fashion; shape/order confounded | empirical; mechanism conjecture |
| later routes higher degree (4.4) | k = 4-5, exponent 3.3 | not established | withdrawn |
| binary lambda_min exponent (4.5) | 0.877 ± 0.005, 12% low | 0.95 ± 0.07 (4 seeds, < 1 decade), consistent with 1 | empirical |
| MPS Hessian (4.5) | strict saddle of the first (k = 2) route | strict, but the state is after edge 5 rose (later routes) | empirical |
| regimes (4.8) | S2S -> rich (1.4) -> lazy; floor well above feature learning | rich on one sigma; boundary threshold-dependent; lazy unconverged; floor claim false per seed | empirical |
| whitened S2S (4.9) | 80% | 57% (17/30) | empirical |
| bottom-up order (4.7) | all trees incl. MPS | binary trees only | empirical (tendency) |
| Fashion (7) | siblings first 15/15, 14/15; sigma^-0.92...-0.97 | siblings rise together 3/15; exponents withdrawn | empirical |
| chi = 8 (7) | L(1e4); 1.17 ± 0.06; scrambled fits better at 0.5 | L(tau_end), 23 runs wall-limited; two-point slopes 1.27 / 1.20; sigma = 0.5 comparison withdrawn | empirical |
| 8x8 (7) | L(1e4) | L(3000); one permutation, two seeds | empirical |
