# Input covariance and teacher structure (Q6, Q7, H5) — agent `data-teacher`

Status tags: **proven**, **derived** (under stated assumptions), **empirical**, **conjecture**.
Experiments: `experiments/data_teacher/`. Raw outputs: `/root/outputs/data_teacher/`.

## 0. Notation and two structural facts used throughout

- Teacher components U_r = (x)_i u_r^(i) (unit vectors), W* = sum_r s_r U_r.
- Product covariance Sigma = (x)_i Sigma_i; loss L = 1/2 ||W - W*||_Sigma^2.
- "Aligned": every u_r^(i) is an eigenvector of Sigma_i, Sigma_i u_r^(i) = lambda_{r,i} u_r^(i);
  Lambda_r = prod_i lambda_{r,i} = ||U_r||_Sigma^2.
- g(theta) = <W(theta), Sigma W*> = E[f_theta f*], the correlation that drives the dynamics
  near the origin (the quadratic term 1/2 ||W||_Sigma^2 is smaller by a factor ||W|| / ||W*||).

**Fact A (binary-4 is a depth-3 linear network)** — *proven* (by inspection of the contraction).
`binary_tree(4)` has two bottom cores carrying sites (0,1) and (2,3) and a root with no sites. Both
edges induce the same bipartition (01|23). Writing G_1 as a (d^2 x chi) matrix, the root as a
(chi x chi) matrix R and G_2 as (d^2 x chi):
W_(01|23) = G_1 R G_2^T. So binary-4 is exactly a depth-3 deep linear network acting on the
d^2 x d^2 matricization, and with a product covariance its loss is
1/2 tr[(W - W*)^T Sigma_01 (W - W*) Sigma_23] (DLN with input covariance Sigma_23 and output
metric Sigma_01). Consequences: binary-4 is "matrix-like" (only one cut), all DLN theory applies, and
the rank-constrained critical points are SVD truncations of the (whitened) matricization. Genuinely
tensorial small trees are MPS-3 (K = 3, two distinct cuts) and MPS-4 (K = 4, three cuts).

**Fact B (whitened frame)** — *proven*. Substituting x~_i = Sigma_i^{-1/2} x_i (whitened) gives
f* = <Sigma^{1/2} W*, (x)_i x~_i>, i.e. a whitened problem with teacher W~* = Sigma^{1/2} W*, and the
student's cores carrying site i are multiplied by Sigma_i^{1/2} on that physical leg (still a TTN on
the same tree). Gradient flow in the new cores is preconditioned by Sigma_i on those physical legs
and starts from an anisotropic init. Hence:
- aligned covariance = whitened odeco teacher with weights sqrt(Lambda_r) s_r + preconditioning;
- misaligned covariance = whitened **non-orthogonal** CP teacher with factors Sigma_i^{1/2} u_r^(i)
  (per-site overlap u_r^T Sigma_i u_q) + preconditioning. Step 2 therefore connects to step 4(a).

## 1. Aligned anisotropic product covariance (H5)

### 1.1 Exact solution for aligned (decoupled) init — *derived* (exact, no approximation)

With `aligned_init` (cores diagonal in the teacher basis, amplitudes a_{v,r}) and an aligned
covariance, Sigma (W - W*) = sum_r Lambda_r (Pi_r - s_r) U_r (cross terms vanish because
u_r^(i) T Sigma_i u_q^(i) = lambda_{r,i} delta_rq), so the diagonal family is invariant and
L = 1/2 sum_r Lambda_r (Pi_r - s_r)^2, da_{v,r}/dt = Lambda_r (s_r - Pi_r) Pi_r / a_{v,r}.
For balanced amplitudes (a_{v,r} = a_r, Pi_r = a_r^K):

    dPi_r/dt = K Lambda_r (s_r - Pi_r) Pi_r^{2 - 2/K},   i.e.  Pi_r(t; Sigma) = Pi_r^white(Lambda_r t).

Each component follows the whitened trajectory with its own clock t -> Lambda_r t. Consequences:
- half-time t_r = T(s_r, a_0) / Lambda_r with T(s, a_0) = (1/K) int_{a_0^K}^{s/2} dPi / ((s - Pi) Pi^{2-2/K})
  = a_0^{-(K-2)} / ((K-2) s) + O(s^{-(2-2/K)}) for K >= 3. **To leading order the escape time depends on
  Lambda_r s_r only** (H5's substitution s -> Lambda s);
- widths w_r = w^white(s_r) / Lambda_r ~ 1 / (K Lambda_r s_r^{2-2/K}), **not** the
  1/(K (Lambda_r s_r)^{2-2/K}) that the substitution s -> Lambda s would give;
- final amplitudes are s_r (the network learns W*, not Sigma W*); plateau loss levels are
  L_S = 1/2 sum_{r not in S} Lambda_r s_r^2 = 1/2 sum ||s_r U_r||_Sigma^2 (function-space norms).

So H5 is right for escape times but "behaves as if its weight were Lambda_r s_r" is not literally
true for the whole trajectory; the exact statement is the per-component time rescaling.

### 1.2 Random init: first escape — *derived* (leading order as sigma -> 0)

L = 1/2 ||W||_Sigma^2 - <W, Sigma W*> + const, and Sigma W* = sum_r (Lambda_r s_r) U_r is again
odeco, with weights Lambda_r s_r. Near the origin the quadratic term is O(||W||^2) against
O(||W|| ||W*||) for the linear one, so **up to the first escape the flow is the same as the whitened
flow with teacher weights Lambda_r s_r, started from the same init** (relative error O(||W|| / s)).
Predictions: (i) paired runs (Sigma, s) and (I, Lambda s) from the same init pick the same first
component and have the same first-escape time up to O(1) (vs. t_1 ~ sigma^{-(K-2)});
(ii) the probability that component r is learned first is P_white(r | weights Lambda s): the order
statistics of H1 apply with Lambda s in place of s.

### 1.3 After the first escape: which saddle is stable — *derived* (second-order expansion)

Function-space norms sqrt(Lambda_r) s_r, not Lambda_r s_r, decide the rank-constrained optimum
(Eckart–Young in the whitened frame, Fact B): the best bond-rank-1 approximation in the Sigma-norm is
the component with the largest Lambda_r s_r^2. When the race (1.2) picks a component a with
Lambda_a s_a > Lambda_b s_b but Lambda_a s_a^2 < Lambda_b s_b^2, the plateau after the first escape
sits at a non-optimal rank-1 critical point. Whether the network can leave it quickly depends on
locality (H1b): on binary-4 the "piggyback" move (rotate the learned component's two 2-site blocks
toward component b; two cores, k = 2) has, at the saddle Pi = s_a, the Hessian in the rotation
angles (phi_1, phi_2) of the two bottom cores

    H = [[s_a^2 lambda_{b,1} lambda_{a,2},  -s_a s_b Lambda_b],
         [-s_a s_b Lambda_b,               s_a^2 lambda_{a,1} lambda_{b,2}]],
    det H = s_a^2 Lambda_b (Lambda_a s_a^2 - Lambda_b s_b^2),

(lambda_{r,v} = product of the eigenvalues of component r on the two sites of core v; Pi decouples at
second order). So the saddle with a learned is **strict iff Lambda_b s_b^2 > Lambda_a s_a^2**: the
correction criterion is the loss-greedy one (function-space norm), while the race criterion is
Lambda s. For Sigma = I this reduces to H1b (strict iff s_b > s_a). For equal per-core splits the
negative eigenvalue (in angle units) is s_a sqrt(Lambda_b) (sqrt(Lambda_a) s_a - sqrt(Lambda_b) s_b).
On MPS-4 the same move needs all 4 cores (the cross term is quartic in the angles), so no strict
saddle: a wrong first choice is not corrected quickly.

Three candidate orderings, which the experiments separate: by s_r (teacher weights), by
Lambda_r s_r (race / escape rate, H5), by Lambda_r s_r^2 (loss-greedy / stable saddles).

### 1.4 Results (step 1) — gradient flow, scipy DOP853, rtol 1e-9, atol 1e-14, float64

Setup: N = 4, d = 4, chi = 4, site covariance `power_law_cov(4, alpha)` at every site, odeco teacher
with component r on eigenvector eig_index[r] at every site. Teachers (alpha = 0.25, so
Lambda(e1)/Lambda(e2) = 2): **T1** s = (1, 0.8) with the 0.8 component on the high-variance e1
(Lambda s and Lambda s^2 both favour it: a clean reversal of the s-order); **T2** s = (1, 0.6)
(Lambda s favours the 0.6 component, Lambda s^2 the 1.0 one: race and landscape disagree).
40 seeds per setting; four variants from the same init: (Sigma, s); (I, Lambda s); (I, sqrt(Lambda) s)
(the absorbed white problem = same landscape); the same from the mapped init G x Sigma^{1/2} (same
landscape AND same start, so only the gradient-flow metric differs). Files:
`results/e1_order_table.csv`, `e1_paired_agg.csv`, `e1_plateaus.csv`, figures `e1_order_probs.png`,
`e1_paired_times.png`, `e1_examples.png`; exact check `e1_exact_aligned.{csv,png}`.

1. **Exact aligned-init solution** (1.1) — *empirical confirmation of a derived result*. TTN GF vs the
   scalar ODE: half-times agree to 1e-9 relative, widths to 1e-8 (both trees, all 3 components;
   max |a_r - ODE| <= 1.2e-5 at the steepest point). The naive substitution s -> Lambda s gets the
   half-times right to 0.1-2% but the 10-90% widths wrong by up to 45% (e.g. 0.73 vs 1.19 for
   Lambda = 4.26 on binary-4), as predicted (widths ~ 1/(Lambda s^(2-2/K))).
2. **First escape = whitened problem with weights Lambda s** (1.2) — *empirical*, all 960 paired runs:
   the same component is learned first in 100% of MPS-4 pairs and 95-100% of binary-4 pairs; the
   first-escape times differ by 0.15% / 0.34% / 0.68% (binary, sigma = 0.02 / 0.05 / 0.1, max 1.3%)
   and 0.02% / 0.04% / 0.1% (MPS, sigma = 0.05 / 0.1 / 0.2): the residual shrinks like 1/t_1, i.e. it is
   the O(||W||/s) quadratic-term correction. Against the absorbed white problem (same landscape) the
   first component agrees in only 70-80% (MPS) / 40-80% (binary) of seeds and times differ by 20-36%.
3. **Covariance reverses the order** — *empirical*. T1: the 0.8 component (high variance) is learned
   first in 90-98% of binary-4 runs and 78% of MPS-4 runs (whitened (I, s) baseline: the 1.0 component
   is favoured). P(first) under (Sigma, s) equals P(first) under (I, Lambda s) in every setting
   (identical seeds pick identical components).
4. **Landscape decides the swaps** (1.3) — *empirical confirmation of the derived criterion*. On binary-4
   a wrongly-learned first component is later *unlearned* and replaced (a strict-saddle rotation) iff
   the other component has the larger Lambda s^2: T2 under (Sigma, s): P(swap | 0.6 component first)
   = 0.33 / 0.68 / 0.94 at sigma = 0.1 / 0.05 / 0.02 and P(swap | 1.0 component first) = 0 (0 of 30
   runs); T1: P(swap | 1.0 first) = 0.75 / 0.67 / 1.0, P(swap | 0.8 first) = 0. The same rule holds in
   all four variants with their own weights (e.g. (I, Lambda s) on T2 swaps only after learning the
   1.0 component). The swap probability rises toward 1 as sigma -> 0 because the swap is a strict
   saddle (log(1/sigma) time) racing the polynomial (sigma^-1) learning of the second component.
   On MPS-4 no swap occurs in any of the 960 runs (the swap route needs all 4 cores: higher order).
5. **Race is a property of the flow metric, not the landscape** — *empirical*. The mapped-init white
   runs share landscape and starting point with (Sigma, s), yet pick a different first component in
   10-15% (MPS) and 12-53% (binary) of seeds and escape 11-23% later: preconditioning by Sigma_i on the
   physical legs speeds up the high-variance directions. So H5's "order follows Lambda s" is a
   statement about the flow, while "plateau levels and stable saddles follow Lambda s^2" is a
   statement about the landscape.
6. **Plateau levels** — *empirical*: 1/2 sum_{unlearned} Lambda_r s_r^2 to relative 1e-7 (MPS,
   sigma = 0.05) ... 1e-3 (MPS, sigma = 0.2) and 1e-4 (binary); the transient "wrong" plateau before a
   swap is not reached (the slowest point lies 7-11% below that level), as expected for a strict saddle.
7. **MPS-4 vs binary-4 order statistics vs sigma**: on MPS-4, P(first) is identical at sigma = 0.05,
   0.1, 0.2 (the init direction alone decides; tensorial teacher, several basins, H1); on binary-4 it
   drifts toward the larger Lambda s as sigma decreases (matrix-like: unique maximum of g, greedy only
   logarithmically slowly) — e.g. T1: 0.90 / 0.925 / 0.975.

8. **Escape-time scaling** — *empirical* (`results/e1_scaling_fits.csv`, `e1_scaling.png`; 10 seeds per
   point). t_1 vs sigma (6 values over one decade, teacher T1): binary-4 slopes -1.010 +- 0.002
   (Sigma), -1.012 +- 0.002 (I, s), -1.008 +- 0.001 (I, Lambda s); MPS-4 -1.999 +- 0.001, -2.001 +- 0.001,
   -1.999 +- 0.001; prediction -(K-2) = -1, -2: **the covariance does not change the exponent**, only the
   prefactor. Paired ratio t_1(Sigma, s) / t_1(I, Lambda s) = 0.994-0.9999 at every sigma (vs 0.65-0.70
   for t_1(Sigma, s) / t_1(I, s)). One-component teachers with Lambda s from 0.008 to 33 (3.6 decades):
   t_1 ~ (Lambda s)^p with p = -0.960 +- 0.007 (binary-4) and -1.009 +- 0.001 (MPS-4); prediction -1 at
   leading order (the binary deviation is the O(1) saturation time, relatively larger at K = 3; the
   paired (I, Lambda s) runs give the same -0.952 +- 0.007). The second-learning time follows the
   same exponents (binary -1.05 +- 0.01, MPS -2.002 +- 0.001).
9. **Escape degree** (theory_saddles R5) — *empirical*: the local growth exponent
   d log(da/dt)/d log a of the first-learned component is the same for all four variants
   (binary 1.26-1.28, MPS 1.32-1.35 averaged over the window a in [1e-4, 1e-2]; up to 1.30 and 1.44 in
   the cleanest windows, vs 2 - 2/K = 1.33 and 1.5; limited by the output grid on the steep escape).
   `results/e1_kexp.csv`.

**Step 1 verdict.** Aligned covariance changes the picture *quantitatively* only, and in a way the
baseline theory predicts after two different substitutions: the *race* (which component escapes
first, and when) is the whitened problem with weights Lambda_r s_r (paired agreement to < 1%), while
the *landscape* (plateau levels, which rank-1 saddles are stable, whether a wrongly learned component
is swapped out) is the whitened problem with weights sqrt(Lambda_r) s_r (Fact B), with plateau levels
1/2 sum Lambda_r s_r^2. When the two orderings disagree, binary-4 (where a 2-core correction route
exists) shows a qualitatively visible extra event: learn the fast component, unlearn it, learn the
other, relearn the first. The exponents of plateau durations in sigma are unchanged.

## 2. Misaligned covariance — theory

Fact B maps a misaligned product covariance to a whitened problem with the **non-orthogonal** CP
teacher W~* = sum_r s_r (x)_i Sigma_i^{1/2} u_r^(i) (per-site overlaps u_r^T Sigma_i u_q), with
the physical legs preconditioned by Sigma_i. Reparametrization maps critical points to critical points,
so **the set of plateau states (saddles) is exactly that of the whitened non-orthogonal problem**
(*proven*); only the timing (preconditioning, anisotropic init) differs. Two different tensors
matter:
- the race near the origin is driven by Sigma W* = sum_r s_r (x)_i Sigma_i u_r^(i) (factors
  Sigma_i u_r, effective weights s_r prod_i ||Sigma_i u_r^(i)||; per-site overlap of the factors
  grows like (lambda_1^2 - lambda_2^2)/(lambda_1^2 + lambda_2^2) at 45 degrees);
- the plateau states are rank-constrained critical points of 1/2 ||W - W*||_Sigma^2, i.e. of the
  whitened frame with factors Sigma_i^{1/2} u_r (overlap (lambda_1 - lambda_2)/(lambda_1 + lambda_2)
  at 45 degrees).
So the escape direction (top of Sigma W*) and the saddle it must reach (critical point in the
Sigma^{1/2} frame) are different tensors: after the escape the state must rotate. *Derived*.

### 2.1 Results (step 2) — gradient flow as in step 1; 20 seeds per point

Setup (`configs/e2_misaligned.yaml`): odeco teacher on e1, e2 (weights 1, 0.8), site covariance
R(theta) diag(power law alpha) R(theta)^T, theta in {0, 10, 20, 30, 45} degrees, alpha in {0.5, 1};
binary-4 at sigma = 0.05, MPS-4 at sigma = 0.1; variants (Sigma, s) and the absorbed white problem
from the mapped init. Predictions: `predict_rank1.py` (rank-1 critical values and states of the
Sigma-loss; "race" tensor = best bond-rank-1 approximation of Sigma W*). Results
`results/e2_misaligned_plateaus.csv`, figure `results/e2_misaligned_summary.png`.

1. **Plateau = rank-1 critical point of the Sigma-loss** — *empirical*: the first-plateau level matches
   the predicted critical value with median relative error 1e-6 to 1e-3 at every theta, in both
   variants (same landscape, Fact B), and the plateau state has |cos| >= 0.998 with the predicted
   critical tensor (theta <= 30). Exception: MPS-4, theta = 45: the "plateau" lasts < 1 time unit and is
   not resolved (errors up to 6%).
2. **Early alignment with Sigma W*, then realignment** — *empirical; TTN analogue of Atanasov et al.
   2022*: when W has grown to 20% of its plateau norm, it is aligned with the race tensor (top
   bond-rank-1 approximation of Sigma W*; median |cos| 0.90-0.997) more than with the saddle it
   reaches (|cos| down to 0.44-0.88 at theta = 45); on the plateau |cos| with the saddle is 0.998-1.
   At theta = 0 the two tensors coincide.
3. **Misalignment shortens the rank-1 plateau of MPS-4 by orders of magnitude** — *empirical*: median
   t_2 - t_1 (t_2 = first time max_e s_2^e > 1e-2) is 539 / 38 / 4.2 / 1.2 / 0.3 at theta = 0 / 10 / 20 /
   30 / 45 (alpha = 1; similar for alpha = 0.5: 327 / 168 / 34 / 3.3 / 0.9). On binary-4 it shrinks
   only by a factor ~7 (75 -> 10). Interpretation (*derived*, theory_teachers.md R4/R7 via Fact B):
   in the whitened frame the teacher components become non-orthogonal, which opens a 2-core
   strict-saddle route on every MPS edge (log(1/sigma) escape) where the aligned case needs all 4 cores
   (sigma^-2); binary-4 is a depth-3 DLN whose second escape is always a 3-core route. The sigma-scaling
   check is `configs/e2_sigma.yaml` (queued).
4. **Gradient-flow metric**: the absorbed white problem from the mapped init has the same plateau
   levels and durations within ~20% (`white_mapped` rows), so the landscape transfer is exact and
   the metric changes only timing.

## 3. Cross-site correlated inputs

### 3.0 Model and theory

Input model (one simple choice): x^(i) = sqrt(1 - rho) z^(i) + sqrt(rho) xi, with xi in R^d shared by all
sites; marginals stay whitened, Cov(x^(i), x^(j)) = rho I for i != j; rho = 0 is the baseline. The
feature metric M = E[Phi Phi^T] is then **not a product**: by Isserlis' theorem it is a sum over the
pairings of the 2N Gaussian factors, including pairings *inside* Phi (x^(i) with x^(j), i != j). Two
consequences (*derived*):
- f* has a nonzero mean and the teacher components are no longer orthogonal in function space,
  <U_1, M U_2> = (E[prod_i x_1^(i)])^2 = 9 rho^4 (N = 4);
- M W* contains "pairing" terms such as delta_{a_0 a_1} (x) e_r (x) e_r (the feature (x^0 . x^1) x^2_r x^3_r),
  which have bond rank 1 on binary-4 (the identity on sites 0, 1 lives inside one core) but bond
  rank d on MPS-4. So a binary student can use non-teacher features from its first escape on;
  an MPS student cannot at rank 1.
Whitening does not help: M^{1/2} is not a product operator, so there is no substitution that maps the
problem to a whitened TTN problem on the same tree. What survives is the general statement that
plateaus are critical points of the rank-constrained M-loss; we compute them numerically
(`predict_rank1.py`, `constrained.py` with the dense metric).

### 3.1 Results (step 3) — exact population gradient flow (dense M, N = 4, d = 4), as in step 1

`configs/e3_correlated.yaml`: odeco teacher on e1, e2 (weights 1, 0.7), rho in {0, 0.1, ..., 0.5},
binary-4 at sigma = 0.05, MPS-4 at sigma = 0.1, 20 seeds. Results `results/e3_correlated_plateaus.csv`,
figure `results/e3_correlated_summary.png`; predictions `results/rank1_predictions_e3_correlated.csv`.

1. **S2S survives** — *empirical*: every run has the plateau-escape-plateau structure and converges
   (L < 1e-6 L(0)); the first-plateau level matches the predicted rank-1 critical value of the M-loss
   with median relative error <= 5e-4 (binary) and <= 5e-3 (MPS; the plateau gets short, see 3).
2. **binary-4: the first saddle changes content and depth** — *empirical, explained*: the rank-1
   plateau falls from 0.33 L(0) (rho = 0, = the second component's share) to 0.20 / 0.14 / 0.10 L(0) at
   rho = 0.3 / 0.4 / 0.5. The predicted (and reached, |cos| = 1.000) rank-1 critical tensor is a
   product of two 2-site blocks that are mixtures of e1 e1^T and e2 e2^T (|cos| 0.80 and 0.60 with
   them at rho = 0.5), i.e. it blends both components and contains the non-teacher cross terms
   e1 e1 (x) e2 e2 that the shared factor makes correlated with f*. The first saddle is no longer a
   partial fit of the teacher.
3. **MPS-4: same saddle, much shorter second plateau** — *empirical*: the rank-1 saddle stays the pure
   component (|cos| 1.000 with e1^(x)4; level 0.329 -> 0.312 L(0)), but the rank-1 plateau duration
   drops from 137 (rho = 0) to 46 / 1.4 / 0.6 / 0.3 / 0.14 at rho = 0.1 / ... / 0.5 (medians): with
   non-orthogonal components in function space a cheap (2-core) escape route opens, as in steps 2 and
   4(a) (theory_teachers.md R4).
4. **Faster first escape** — *empirical*: median t_1 falls from 56 to 3.9 (binary) and from 186 to 14
   (MPS) between rho = 0 and 0.5: the correlation E[f_theta f*] that drives the escape grows with rho.
5. Order: the larger component is still learned first in most runs (binary 80-85%, MPS 65-75%).
6. Sampled data: `e3_sgd_check.py` (online SGD, fresh batches) — see 3.2.

## 4(a). Non-orthogonal CP teacher — theory

W* = s_1 U_1 + s_2 U_2, per-site overlap c = <u_1^(i), u_2^(i)>, whitened.
- Rank-1 critical points: for X = rho (x)_i x_i, L = 1/2 rho^2 - rho <(x)x_i, W*> + 1/2||W*||^2 is
  minimized at rho = <(x)x_i, W*>, so the rank-1 critical points of L are the critical points of
  g(x) = <(x)_i x_i, W*> on unit vectors, with plateau level 1/2 (||W*||^2 - g^2). *Proven*.
- MPS-4 (tensorial): with equal weights, restricted to x_i = x in span(u_1, u_2) at angle psi,
  g = s [cos^N(psi) + cos^N(psi - beta)], beta = arccos c. The symmetric point psi = beta/2 is a
  local max iff tan^2(beta/2) < 1/(N-1), i.e. **c > (N-2)/N = 1/2 for N = 4**. *Derived*. Below it
  there are two rank-1 local maxima (one near each component; the first plateau is at one of them,
  chosen by the init basin); above it a single "blended" maximum, so the first saddle is a blend of
  the two components and the order question disappears.
- Binary-4 (matrix-like, Fact A): the rank-1 critical points are the singular modes of the
  matricization W*_(01|23) = s_1 a_1 b_1^T + s_2 a_2 b_2^T with a_r = u_r (x) u_r,
  <a_1, a_2> = <b_1, b_2> = c^2. The top mode is a blend for every c > 0 (no bifurcation), the plateau
  level is sigma_2^2 / 2 (sigma_k = singular values of the 2x2 problem), and the order is greedy
  as sigma -> 0 (unique local max of g, DLN). *Derived*.

### 4(a).1 Results — gradient flow as in step 1; 20 seeds per point

`configs/e4a_nonorth.yaml`: whitened inputs, u_1 = e1, u_2 = c e1 + sqrt(1 - c^2) e2 at every site,
weights (1, s2), s2 in {0.7, 1}, c in {0, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8}; binary-4 at sigma = 0.05,
MPS-4 at sigma = 0.1. Results `results/e4a_nonorth_plateaus.csv`, figure `e4a_nonorth_summary.png`,
predictions `rank1_predictions_e2_misaligned_e4a_nonorth.csv`.

1. **Number of rank-1 saddles** — *derived and numerically confirmed*: on MPS-4 the constrained
   minimization finds two rank-1 local minima for c <= 0.45 and one for c >= 0.55 at s2 = 1
   (threshold c* = (N-2)/N = 1/2), and two for c <= 0.35, one for c >= 0.45 at s2 = 0.7 (fold between
   0.35 and 0.45). On binary-4 there is a single rank-1 minimum for every c > 0 (matrix-like), except
   numerically near-degenerate duplicates.
2. **Plateaus sit at these critical points, never at a component** — *empirical*: binary-4 first
   plateau = predicted value to median 5e-5 - 2e-3 relative, |cos| with the predicted tensor >= 0.97
   (>= 0.9999 for c >= 0.35); MPS-4 to 2e-6 (c = 0) ... 3% (c = 0.8; plateau barely resolved, below).
   The plateau level falls with c (more of the second component is captured by the blended rank-1 fit),
   e.g. binary s2 = 0.7: 0.33 / 0.25 / 0.04 L(0) at c = 0 / 0.45 / 0.8.
3. **The second plateau changes character on MPS-4, not on binary-4** — *empirical*: median duration
   of the rank-1 plateau (sigma = 0.1 / 0.05): MPS-4 137 -> 8.2 -> 2.2 -> ~1.7 for c = 0 -> 0.2 -> 0.35 ->
   0.8 (s2 = 0.7; same for s2 = 1); binary-4 35 -> 32 -> 33 -> 152 (it *grows* with c, because the
   second singular value of the (01|23) matricization shrinks and binary-4's next escape is always a
   3-core DLN route). On MPS-4 any overlap opens a 2-core strict-saddle route (theory_teachers.md R4,
   R7), so at sigma = 0.1 the two learning events nearly merge (separation ratio < 3: by the rules of
   diagnostics.md these runs are "rich, non-incremental" at this sigma). Scaling with sigma: section
   4(a).2 (`configs/e4a_sigma.yaml`).
4. Order: binary-4 learns the larger component first in 75-90% of runs (s2 = 0.7); with s2 = 1 the
   choice is 50/50, as expected from symmetry.

## Results

(filled in as experiments finish)
