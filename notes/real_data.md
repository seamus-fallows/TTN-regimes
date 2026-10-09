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

## 3. Methods

(see `experiments/real_data/README.md`)

## 4. Results

(pending)
