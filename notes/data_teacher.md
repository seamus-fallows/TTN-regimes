# Input covariance and teacher structure (Q6, Q7, H5) — agent `data-teacher`

Status tags: **proven**, **derived** (under stated assumptions), **empirical**, **conjecture**.
Experiments: `experiments/data_teacher/`. Raw outputs: `/root/outputs/data_teacher/`.

## Summary table (deliverable)

Baseline: whitened inputs, odeco teacher, fan-in init, gradient flow; N = 4 (binary-4 = depth-3 DLN,
MPS-4 = genuinely tensorial), 20-40 seeds per setting. Details and numbers in the sections below.
Overview figure: `experiments/data_teacher/results/summary_plateau_law.png` (later-plateau duration
vs sigma, baseline sigma^-2 vs logarithmic for every complication that breaks orthogonality).

**One-line answer.** Covariance aligned with an odeco teacher only reweights (race by Lambda s,
landscape by sqrt(Lambda) s). Everything that makes the teacher's components non-orthogonal in the
metric that defines the loss changes the picture *qualitatively* on students where a 2-core
correction route exists (MPS): misaligned covariance, overlapping CP components, cross-site
correlations, generic TTN teachers. The later plateaus become strict saddles with
log(1/sigma) durations instead of higher-order saddles with sigma^-(K-2) durations, and the
plateaus are blended rank-constrained critical points instead of partial sums of components. The
first escape always keeps its sigma^-(K-2) law. Binary-4 (a depth-3 linear network on one
matricization) is insensitive to this and stays polynomial.

| complication | why checked | what changed | baseline theory (with substitution) predicts it? |
|---|---|---|---|
| 1. Aligned anisotropic product covariance (power-law site spectra, teacher on eigenvectors) | H5; the simplest non-white data; Saxe-type reweighting expected | **Quantitative only.** Order and escape times follow Lambda_r s_r (reversals of the s-order observed: 78-98% of runs); plateau levels 1/2 sum Lambda_r s_r^2; exponents in sigma unchanged (-1.01, -2.00). New event when the orderings by Lambda s and Lambda s^2 disagree: on binary-4 the first-learned component is unlearned and replaced (strict-saddle swap, P -> 1 as sigma -> 0); never on MPS-4 | **Yes, with two substitutions**: race = whitened with weights Lambda s (paired runs agree to < 1%); landscape (levels, stable saddles, swaps) = whitened with sqrt(Lambda) s (absorb Sigma^{1/2} into the cores; exact). Aligned init solved exactly (per-component clock Lambda_r t); "weight Lambda s" gets widths wrong by up to 45% |
| 2. Misaligned product covariance (rotation theta between eigenvectors and teacher factors) | non-commuting covariance is where linear-net theory breaks (Atanasov 2022) | Escape aligns first with the top rank-1 part of Sigma W*, then rotates to the saddle (realignment). Saddles are rank-1 critical points of the Sigma-loss (blends, not components). **Qualitative on MPS-4**: the later plateau becomes logarithmic in sigma (theta = 20: Dt_2 = 15.0 log(1/sigma) + B) instead of sigma^-2; it shrinks from ~540 to < 1 time unit as theta goes 0 -> 45 deg. Binary-4: same law, shorter plateau | **Yes, through the absorbed white problem** (exact landscape map to a whitened non-orthogonal CP teacher, Fact B) plus the non-orthogonal-teacher theory (2-core strict-saddle routes, theory_teachers.md). Levels match to 1e-5 - 1e-3. Timing differs from the white problem by the GF metric (preconditioning), ~20% |
| 3. Cross-site correlated inputs (shared latent factor, rho) | breaks the product structure of the loss (no whitening substitution exists) | S2S survives (exact population GF; online SGD with sampled data agrees). Plateaus = rank-1 critical points of the non-product loss (levels to <= 0.5%). **binary-4: the first saddle changes content** (blends both components plus non-teacher pairing features; level 0.33 -> 0.10 L(0) at rho = 0.5). MPS-4: first saddle stays the component, later plateau collapses (137 -> 0.14). First escape faster (t_1 / 14) | **Partly**: no substitution maps it to the baseline; the general rule "plateaus are critical points of the bond-rank-constrained loss" holds, with the non-product metric, and the collapse on MPS-4 follows the non-orthogonality mechanism |
| 4a. Non-orthogonal CP components (overlap c) | generic teachers are not odeco; theory_teachers.md predictions | First plateau is a blend (never a component); number of rank-1 saddles on MPS-4 drops from 2 to 1 above c* = 1/2 (equal weights) / fold ~0.4 (s2 = 0.7); on binary-4 always 1. **Qualitative on MPS-4**: later plateau logarithmic, Dt_2 = A log(1/sigma) with A = 31.9 +- 1.6 (pred 34.1) at c = 0.2, 4.7-5.6 (pred 5.68) at c = 0.45; sigma^-1.97 at c = 0. Binary-4 plateau gets longer with c (DLN) | **Yes** (theory_teachers.md R4, R6, R7), quantitatively |
| 4b. Random TTN teacher on the student's tree (MPS-4) | are saddles HSVD truncations? order of bond-rank growth | Saddles are critical points of the bond-rank-constrained loss, **not** HSVD truncations (rank-1 plateau 3x closer to the critical point, distance -> 0 with sigma; they differ only at rank 1 here). **Qualitative**: ranks grow one bond at a time and every later plateau is a strict saddle of duration A log(1/sigma) (odeco baseline: sigma^-(K-2)) | **Yes** (theory_teachers.md R1-R6): slopes within 3-25% of A_e*, first bond = predicted e* in 43-100% of runs (lower when two rates are within 12%) |
| 4c. Teacher on a different tree (N = 8) | which student bonds grow first; locality | Binary student, MPS teacher: bottom-up in 24/24 runs, sibling pairs first, the root bipartition (teacher's middle bond) last, blocked bonds need r^2 increments. MPS student, binary teacher: fast cascade of rank increments, first bond random at sigma = 0.45 | **Partly**: route/locality rules (P11) yes; "largest teacher Schmidt value first" fails (0-25%); the asymptotic e* rule is not reached at K = 8 accessible sigma |

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
9. **Swap criterion checked on the exact saddle** (`e1_hessian_check.py`, `results/e1_hessian_check.csv`)
   — *empirical confirmation of the derived Hessian (1.3)*: at the exact critical point "component a
   learned, b not" (aligned init), the library's gauge-projected Hessian has lambda_min equal to the
   formula s_a sqrt(Lambda_b)(s_a sqrt(Lambda_a) - s_b sqrt(Lambda_b))/q^2 to 1e-15 in all 6 strict
   binary-4 cases (e.g. -0.198631 both), strictness is predicted correctly in 12/12 binary-4 cases
   (strict iff Lambda_b s_b^2 > Lambda_a s_a^2, for every variant), and MPS-4 has no negative
   eigenvalue in 12/12 (higher-order saddle: the swap needs 4 cores).
10. **Escape degree** (theory_saddles R5) — *empirical*: the local growth exponent
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
3. **Misalignment shortens the rank-1 plateau of MPS-4 by orders of magnitude** — *empirical* (single
   sigma, pilot grid: values below ~1 time unit are resolution-limited; the sigma law is item 4): median
   t_2 - t_1 (t_2 = first time max_e s_2^e > 1e-2) is 539 / 38 / 4.2 / 1.2 / 0.3 at theta = 0 / 10 / 20 /
   30 / 45 (alpha = 1; similar for alpha = 0.5: 327 / 168 / 34 / 3.3 / 0.9). On binary-4 it shrinks
   only by a factor ~7 (75 -> 10). Interpretation (*derived*, theory_teachers.md R4/R7 via Fact B):
   in the whitened frame the teacher components become non-orthogonal, which opens a 2-core
   strict-saddle route on every MPS edge (log(1/sigma) escape) where the aligned case needs all 4 cores
   (sigma^-2); binary-4 is a depth-3 DLN whose second escape is always a 3-core route. The sigma-scaling
   check is `configs/e2_sigma.yaml` (queued).
4. **Plateau law vs sigma** (`configs/e2_sigma.yaml`, dense runs with the library integrator,
   MPS-4, alpha = 1, 6 sigma over a decade, 10 seeds; `results/e2_sigma_law{,_fits}.csv`,
   `e2_sigma_law.png`) — *empirical*: theta = 0: Dt_2 ~ sigma^-p with p = 1.98 +- 0.005 (prediction 2);
   theta = 20 deg: logarithmic, Dt_2 = A log(1/sigma) + B with A = 15.0 +- 0.4 (relative rms 1.6%,
   vs 7.5% for the best power law, p = 0.46); theta = 5 deg: crossover (A = 264 +- 15, rms 7%).
   Misalignment changes the law of the later plateau, exactly as non-orthogonal components do (4a),
   which is Fact B at work. (The slope is not predicted here: theory_teachers.md R6 is for whitened
   flow, and the Sigma problem is the whitened one with preconditioned physical legs.)
5. **Gradient-flow metric**: the absorbed white problem from the mapped init has the same plateau
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
3. **MPS-4: same saddle, much shorter second plateau** — *empirical* (durations below ~1 time unit are
   at the resolution of the 2000-point output grid of these pilot-integrator runs; the law in sigma was
   not measured for step 3 because the library's Problem has no non-product metric — that it is
   logarithmic, as in 2 and 4a, is a *conjecture* based on the same mechanism): the rank-1 saddle stays the pure
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

### 4(a).2 Plateau law of the second escape — library integrator (LSODA, rtol 1e-9, atol 1e-14), dense grid

`configs/e4a_sigma.yaml`, `run_dense.py`: MPS-4, s2 = 0.7, c in {0, 0.05, 0.2, 0.45}, sigma = 0.03 ... 0.3
(6 values, one decade), 10 seeds; two-stage integration with a dense grid after the first escape.
t_1 = first time L < 0.9 L(0); t_2 = first time the loss falls below half of the rank-1 plateau level.
Results `results/e4a_sigma_law{,_fits}.csv`, figure `e4a_sigma_law.png`; predicted slopes
`results/e4a_theory_slopes.csv` (`theory_log_slope.py`, from the computed rank-1 saddle).
Conservation drift <= 2.6e-10 in all runs.

| c | measured law of Dt_2 = t_2 - t_1 | prediction (theory_teachers.md) |
|---|---|---|
| 0 (odeco) | power law, exponent 1.97 +- 0.01 (per-sigma medians; 1.73 +- 0.04 from the mean of log, pulled down by seeds whose two components escape almost together) | sigma^-(K-2) = sigma^-2 (4-core route) |
| 0.05 | effective exponent 1.66 +- 0.05: crossover | crossover at c_x ~ sigma [log(1/sigma)/(s2 a_N)]^(1/2) ~ 0.02-0.1 here |
| 0.2 | logarithmic: A = 31.9 +- 1.6 (rel. rms of the fit 5%) | A = 34.1 (2-core strict saddle, R6) |
| 0.45 | logarithmic: A = 4.7 +- 0.3 (means; ~5.6 from medians), sigma = 0.3 excluded (drops merged) | A = 5.68 |

**Verdict** (*empirical, confirming the derived R6*): a small overlap between teacher components
changes the *law* of the later plateau on an MPS student from polynomial (sigma^-2) to logarithmic,
with the slope the 2-core strict-saddle theory predicts to within the statistical error (c = 0.2) or
~20% (c = 0.45). The first plateau keeps its sigma^-(K-2) law (it is the escape from the origin).

## 4(b). Random TTN teacher on the student's tree — results

Library integrator (LSODA, rtol 1e-9, atol 1e-14), two-stage dense grid (`run_dense.py`).
`configs/e4b_ttn.yaml`: MPS-4 student (d = 3, chi = 4), random MPS-4 teachers with bond ranks (2, 3, 2)
and spread spectra (5 teacher seeds, `teacher_generator`), sigma = 0.02 ... 0.2 (5 values), 6 seeds.
Results `results/e4b_ttn_plateaus.csv` (states vs HSVD vs critical points), `e4b_order.csv`,
`e4b_order_fits.csv`, `e4b_theory.csv`, figure `e4b_law.png`.

1. **Visited saddles are rank-constrained critical points, not HSVD truncations** — *empirical,
   confirming theory_teachers.md R1-R3/P1*: at the rank-1 plateau (speed minimum), the state is closer
   to the local minimum of the rank-(1,1,1)-constrained loss than to the HSVD truncation in 19/20
   detected plateaus; median distances (relative to ||W*||) 0.005 vs 0.016, with HSVD and the
   critical point 0.012 apart; the distance to the critical point shrinks with sigma (median 0.042 /
   0.005 / 0.004 at sigma = 0.063 / 0.036 / 0.02), the distance to HSVD does not. The critical loss is
   below the HSVD loss (e.g. 0.1588 vs 0.1637 for teacher 103). At higher ranks ((2,1,1), (2,2,2),
   (2,3,1)) HSVD and the critical point coincide to 1e-9 for these teachers, so the question has no
   content there. (Plateau detection: 42 plateaus in 150 runs; many later plateaus are too short at
   these sigma to give a clean speed minimum.)
2. **Bond ranks grow one bond at a time, through logarithmic plateaus** — *empirical*: e.g. teacher
   103: (1,1,1) -> (2,1,1) -> (2,2,1) -> (2,3,1) -> (2,3,2) with plateaus at 5e-2, 6e-3, 6e-4 L(0). The
   time from the first escape to the first rank increase is linear in log(1/sigma) for every teacher,
   with slopes 2.76 +- 0.07, 2.91 +- 0.15, 0.65 +- 0.004, 5.95 +- 0.15, 3.32 +- 0.09 vs the predicted
   A_e* = 2.49, 3.01, 0.87, 5.17, 3.66 (R6; 3-25% agreement, leading-order theory).
3. **Which bond first**: the predicted e* = argmax_e s2(M_e) (and, for these teachers, also the naive
   argmax_e s_2^e(W*)) is the first to grow in 93%, 67%, 100%, 43%, 97% of runs; the 43% case has two
   rates within 12% (0.143 vs 0.163).
So, compared with the odeco baseline, a generic TTN teacher keeps S2S but makes **every plateau after
the first a strict saddle with a log(1/sigma) duration** (MPS student), and the saddles are critical
points of the bond-rank-constrained problem.

## 4(c). Teacher on a different tree — results

`configs/e4c_cross.yaml` (N = 8, d = 2, ||W*|| = 1, 3 teacher seeds x 8 init seeds, library integrator,
dense grid): (i) random MPS-8 teacher (ranks 2) on a binary-8 student (K = 7, sigma = 0.4);
(ii) random binary-8 teacher (ranks 2) on an MPS-8 student (K = 8, sigma = 0.45). Runs stop at
L < 1e-4 L(0) (all reached it). Results `results/e4c_cross_events.csv`, `e4c_cross_summary.csv`,
`e4c_theory.csv`, figure `e4c_cross.png`.

1. **Binary student, MPS teacher: bottom-up, sibling pairs first, root bipartition last** —
   *empirical, confirming theory_teachers.md P11*: in 24/24 runs the root bipartition {0123}|{4567}
   (= the teacher's middle bond) gets its second singular value only after bottom bonds on both halves
   have grown; the first two rank increments are always a sibling pair under one parent (e.g. bonds
   {01} and {23} together: the 3-core sibling route of theory_teachers.md 5.2), and the blocked bonds
   {23}, {45} need up to min(d^2, r^2) = 4 increments (3rd/4th singular values appear last).
2. **Naive rule fails**: "the bond with the largest teacher Schmidt value on its cut grows first" is
   right in 0/24 runs (binary student) and 6/24 (MPS student).
3. **MPS student, binary teacher: first bond random at accessible sigma** — *empirical*: the first
   rank increase is on the predicted e* = argmax s2(M_e) in only 13-38% of runs. At sigma = 0.45,
   log(1/sigma) = 0.8, so the O(1) differences between the initial route amplitudes outweigh the
   log-time rate differences; the asymptotic rule needs much smaller sigma, which costs ~sigma^-6 at
   K = 8 (not run). The cuts that cross a teacher core ({012}, {01234}) need rank 4 and get their 3rd
   and 4th singular values last.

## Round 2

### R2.0 Response to the review of round 1 (notes/reviews/data_teacher_round1.md): claims changed

Status of every round-1 claim the review questioned, after re-analysis. "Re-analysis" = per-seed fits
(the same seeds are reused at every sigma, so errors are standard errors over seeds), the log and the
power law fitted and scored in the same (log) space (`lawfit.py`), runs classified by the rank-one
saddle they visit and flagged when no plateau is resolved, events by the onset rule of
`notes/diagnostics.md` (s_2^e >= f sigma_2^e(T), f = 0.05, sensitivity 0.03 / 0.1), and the
gauge-projected Hessian lambda_min (fixed library projection, raw value alongside) at the plateau
point vs sigma (`analyze_law_v2.py`).

| round-1 claim | status now | evidence |
|---|---|---|
| Single-sigma plateau durations in steps 2, 3, 4a ("540 -> < 1", "137 -> 0.14", "137 -> 8.2 -> 2.2 -> 1.7") | **Withdrawn**: they were mostly one output-grid step or false events of the fixed s_2 > 1e-2 rule | replaced by dense sigma sweeps below (R2.1, R2.3, and the 4a / 2 re-analyses) |
| 4a log law, "A = 31.9 +- 1.6 (pred 34.1) at c = 0.2, 4.7-5.6 (pred 5.68) at c = 0.45, within the statistical error" | **Corrected**: major-saddle runs only, per seed, f = 0.05: c = 0.2: A = 36.6 +- 1.2 (pred 34.1, +7%); c = 0.45: A = 5.98 +- 0.07 (pred 5.68, +5%); log law beats the power law in 6/6 seeds for both; c = 0: power law p = 2.02 +- 0.01 (log wins 0/6). Plateau lambda_min sigma-independent for c > 0 (-0.029 / -0.039 and -0.153 / -0.170 at the smallest / largest sigma), -> 0 for c = 0 (-0.0009 / -0.023): strict vs higher-order by the sweep-level rule. sigma 0.03-0.3; the extension to 3e-4 is in the queue | `results/e4a_sigma_v2_summary.csv` [to be regenerated with the extension] |
| Step 2 log law at theta = 20 | **Confirmed with per-seed errors**: A = 13.5-14.6 (f = 0.03 / 0.1), log wins 10/10; plateau lambda_min ~ -0.08, sigma-independent; theta = 0: p = 2.00, lambda_min -> 0 | `results/e2_sigma_v2_summary.csv` [extension queued] |
| "Binary-4 stays polynomial" (never measured) | **Now measured** (R2.1): sigma^-1 escape from the rank-one saddle on binary-4, binary-8, ht4, ht8 for 12 generic teachers; also under correlated inputs (R2.3) | `results/r2_saddle_fits.csv`, `r2_corr_fits.csv` |
| "Later plateaus are strict saddles" (no Hessian computed) | **Now computed**: exact-saddle Hessians (R2.1: strict on MPS, non-strict on binary / HT) and plateau-point lambda_min vs sigma (4a, 2 re-analyses) | `results/r2_saddle_hessian.csv`, `*_v2_summary.csv` |
| Correlated-input log law (a conjecture in round 1) | **Now measured** (R2.3): MPS-4 increments per halving 14.89 +- 0.22 (pred 14.39) at rho = 0.2 and 2.131 +- 0.013 (pred 2.128) at rho = 0.4; binary-4 stays sigma^-1 | `results/r2_corr_fits.csv` |
| "S2S survives" correlated inputs (MPS-4 unresolved in 55-90% of runs at rho >= 0.2) | **Restated**: on a dense grid the separation ratio of MPS-4 grows like ln(1/sigma) (rho = 0.4: 2.6 at sigma = 0.05 -> 11.4 at 0.0016); binary-4 separates like 1/sigma (13-19 -> 135-185) | `results/r2_corr_runs.csv` |
| 4b "critical points, not HSVD truncations" | **Narrowed**: the other TT-SVD sweep direction lies within 0.0014-0.005 of the rank-one critical point for teachers 101-104; only teacher 105 separates the critical point (0.028 from every SVD-type truncation; plateau 0.005 from the critical point) | `analyze_ttn.py` (TT-SVD both directions + HSVD) [regenerated with the sigma extension] |
| 4b slopes "within 3-25% of A_e*" (residual SEs) | [to be redone per seed with the f-rule after the sigma extension] | |
| 4c "first two increments always a sibling pair", "naive rule 0/24", "root last, confirming P11" | **Narrowed**: with the f-rule (f = 0.1 / 0.03): sibling pair first in 20/24 and 19/24 runs; P11 discriminates only for teacher 203 (8/8 bottom-up); five new teachers whose root cut is NOT the weakest, at three sigma, are in the queue | `results/e4c_cross_v2_summary.csv`, `configs/r2_cross.yaml` |
| Early alignment with Sigma W* then realignment | **Narrowed to binary-4** (MPS-4 only at theta = 45) | review M5 |
| Escape degree "consistent with 2 - 2/K" (1.32 vs 1.5) | **Fixed**: window starting at 10 abs(a(0)) per sigma: binary-4 1.23 / 1.28 / 1.32 at sigma = 0.1 / 0.05 / 0.02 (pred 1.333), MPS-4 1.38 / 1.43 / 1.465 at 0.2 / 0.1 / 0.05 (pred 1.5), same for all four variants | `results/e1_kexp.csv` |
| Lambda-scaling slope -0.96 "saturation" | **Corrected explanation**: within each teacher direction the slope is -1.000 (1.0001-1.0016 per seed); the pooled -0.96 is a prefactor jump between the e1 and e4 sub-series | review L1, rechecked |
| Swap "iff Lambda_b s_b^2 > Lambda_a s_a^2"; "0 of 30" | **Wording**: "only if" (necessity: 0 swaps after a landscape-correct first component in all 24 binary settings; sufficiency with probability 0.33-0.94); 0 of 27 | |
| "P(first) identical under (Sigma, s) and (I, Lambda s) in every setting" | **Except** binary T2 at sigma = 0.02 (0.800 vs 0.825) | |
| Float-floor scan | **Re-run on the dense runs** behind the claims [after the queue] | `check_float_floor.py` |
| "20-40 seeds per setting" | sigma sweeps use 5-10 seeds (stated per table) | |
| Overview figure (log y axis cannot show a log law) | **Replaced** by `r2_saddle_law.png` (per tree) and per-case linear-y panels | |
| Reproducibility | `DATA_TEACHER_RESULTS` / `DATA_TEACHER_OUT` honoured by every script (default raw path: /root/outputs on the remote, <repo>/data/data_teacher elsewhere); `PROCS` for the queue scripts; run order in the README | |

### R2.1 Tree-shape contrast for generic teachers (round-2 item 1)

*Question* (theory_teachers.md R9/P9, theory_saddles.md R13): for the same generic teacher, is the
escape from the rank-one saddle a strict saddle with a log(1/sigma) time on MPS students (k* = 2), and
a higher-order saddle with a sigma^-1 time on binary and HT students (k* = 3 through an internal core)?

*Setup.* d = 2, chi = 4. Twelve generic teachers: non-orthogonal CP (c = 0.2, 0.45 at N = 4; 0.45,
0.7 at N = 8, since the MPS 2-core rate scales like c^(N-2)), misaligned covariance (theta = 20 deg;
N = 8 also theta = 45 deg, alpha = 2, see `configs/r2_saddle_b.yaml`), random binary-tree teachers and
a random MPS teacher. Students mps4, binary4, ht4 (K = 4, 3, 7) and mps8, binary8, ht8 (K = 8, 7, 15).
**Saddle-start protocol**: the exact balanced bond-rank-one critical point (lowest loss), zero-padded,
plus fan-in noise of scale eps on every core; library LSODA (rtol 1e-9, atol 1e-14); escape event =
first time some edge has s_2^e >= 0.05 s_2^{e,*} (and >= 10x its start value). eps = 1e-3 ... 5e-2
(6 values, 1.7 decades), 5 seeds; the extension to eps = 1e-8 (MPS) and 1e-5 (binary, HT) is queued.
Per-seed fits as in R2.0. Predicted MPS slope: 1/abs(lambda_min) of the exact-saddle Hessian (fixed
gauge projection; raw = projected for every MPS saddle).

**Hessian at the exact rank-one saddle** (`results/r2_saddle_hessian.csv`) — *derived and computed*:
strict on every MPS student (lambda_min < 0, with (N-1)(d-1)(chi-1)-type multiplicity: 5 on mps4,
13-17 on mps8), no negative eigenvalue on every binary and HT student (raw lambda_min ~ -1e-9 = the
residual gradient of the numerical critical point), for all 12 teachers.

**Escape law** (`results/r2_saddle_fits.csv`, figure `results/r2_saddle_law.png`; f = 0.05):

| teacher (N) | MPS: lambda_min (exact saddle) | MPS: A fit (pred 1/abs(lambda_min)) | MPS log wins | binary: p (pred 1) | binary log wins | HT: p (pred 1) | HT log wins |
|---|---|---|---|---|---|---|---|
| nonorth_c0.2 (4) | -0.0285 | 36.11 +- 0.85 (35.14) | 5/5 | 1.019 +- 0.002 | 0/5 | 1.034 +- 0.015 | 0/5 |
| nonorth_c0.45 (4) | -0.1509 | 6.66 +- 0.03 (6.63) | 5/5 | 1.016 +- 0.002 | 0/5 | 1.032 +- 0.005 | 0/5 |
| misaligned_20 (4) | -0.0189 | 55.50 +- 1.07 (52.94) | 5/5 | 1.019 +- 0.002 | 0/5 | 1.008 +- 0.027 | 0/5 |
| randbin_302 (4) | -0.0764 | 12.85 +- 0.34 (13.09) | 4/5 | 1.026 +- 0.002 | 0/5 | 1.022 +- 0.008 | 0/5 |
| randbin_303 (4) | -0.3598 | 2.79 +- 0.04 (2.78) | 5/5 | 1.023 +- 0.004 | 0/5 | 1.013 +- 0.004 | 0/5 |
| randmps_301 (4) | -0.3762 | 2.62 +- 0.04 (2.66) | 5/5 | 1.023 +- 0.003 | 0/5 | 1.025 +- 0.008 | 0/5 |
| nonorth8_c0.45 (8) | -0.0048 | 318.31 +- 11.00 (209.70) | 5/5 | 1.019 +- 0.003 | 0/5 | 1.025 +- 0.006 | 0/5 |
| nonorth8_c0.7 (8) | -0.0574 | 17.65 +- 0.17 (17.41) | 5/5 | 1.026 +- 0.002 | 0/5 | 1.020 +- 0.006 | 0/5 |
| misaligned8_20 (8) | -0.0005 | 47433.72 +- 4068.77 (1974.82) | 5/5 | 1.079 +- 0.006 | 0/2 | n/a | n/a |
| randbin8_302 (8) | -0.3706 | 2.74 +- 0.03 (2.70) | 5/5 | 1.023 +- 0.008 | 0/5 | 1.014 +- 0.010 | 0/5 |
| randbin8_303 (8) | -0.1897 | 5.50 +- 0.12 (5.27) | 5/5 | 1.019 +- 0.005 | 0/5 | 1.024 +- 0.009 | 0/5 |
| randmps8_303 (8) | -0.1642 | 6.46 +- 0.19 (6.09) | 5/5 | 1.026 +- 0.006 | 0/5 | 1.013 +- 0.006 | 0/5 |

*Verdict* — *empirical, confirming theory_teachers R9 / theory_saddles R13*: for every teacher the
binary and HT escapes follow sigma^-p with p = 1.01-1.03 (prediction 1; log law never preferred,
0/5), while the MPS escapes follow the log law (preferred in 59/60 seeds), with slopes within 0.5-6%
of 1/abs(lambda_min) for 10 teachers. Two N = 8 exceptions: nonorth8_c0.45 (A = 318 vs 210): its
2-core rate is small (lambda_min = -0.0048) so at eps >= 1e-3 the polynomial routes still compete
(the eps extension tests this); misaligned8_20 is degenerate (the product covariance over 8 sites
leaves the second component a function-space weight of ~0.002 L(0); no escape law; replaced by the
balanced theta = 45 deg teacher). Random-init sweeps on mps4 and binary4 (the R6 slope with head start)
are running (`configs/r2_random.yaml`).

### R2.3 Sigma law under cross-site correlated inputs (round-2 item 3)

Setting of theory_covariance.md section 7: N = 4, d = 3, site-aligned teacher s = (1, 0.6), shared
latent factor rho; exact population gradient flow with the dense moment matrix, scipy LSODA with the
exact Hessian, two-stage dense grid (`r2_corr.py`); 6 sigma (factor 2 apart, 0.0016-0.05), 5 seeds;
spectral events on the 01|23 cut with final-rise widths. Runs are classified by the first saddle:
1 of 5 seeds (seed 2) visits the minor saddle on MPS-4 at every rho and is excluded from the fits.
Results `results/r2_corr_fits.csv`, `r2_corr_runs.csv`, figure `r2_corr_law.png`.

| tree | rho | law (per seed, log space) | increment per halving of sigma | prediction | separation ratio (median, sigma = 0.05 -> 0.0016) |
|---|---|---|---|---|---|
| mps4 | 0 | power, p = 2.000 +- 0.000 | - | sigma^-2 | 103 -> 530 |
| mps4 | 0.1 | log (4/4 seeds) | 116.9 +- 2.3 | - | 64 -> 94 |
| mps4 | 0.2 | log (4/4) | 14.89 +- 0.22 | 14.39 | 20 -> 45 |
| mps4 | 0.3 | log (4/4) | 4.642 +- 0.035 | - | 3.9 -> 19 |
| mps4 | 0.4 | log (4/4) | 2.131 +- 0.013 | 2.128 | 2.6 -> 11 |
| binary4 | 0 - 0.4 | power, p = 1.005-1.011 (log wins 0/5) | - | sigma^-1 (DLN) | 14-19 -> 139-185 |

*Verdict* — *empirical, confirming theory_covariance.md*: cross-site correlations turn the MPS-4
later plateau from sigma^-2 into a log law with the predicted increments (0.1% at rho = 0.4, 3.5% at
0.2); the plateaus stay separated, but only logarithmically (separation ratio < 3 at sigma >= 0.05,
rho >= 0.4). Binary-4 keeps its sigma^-1 law at every rho.

### R2.2 Star teachers (round-2 item 2)

Exact Hessian at "q learned" (`r2_star.py hessian`, `results/r2_star_hessian.csv`; fixed gauge
projection, raw equal): lambda_min = -0.371573 (mps4), -0.331034 (binary4), -0.431073 (ht4 and the
binary8 analogue) for (2; 1.6, 1.6), equal to -a^(K-2)(rho(S) - s_q) to 1e-15; at w = 1.45
(rho/s_q = 1.025) -0.071573 / -0.063764 / -0.083034; no negative eigenvalue for w <= 1.40
(rho < s_q). Dynamics (forgetting depth vs sigma, threshold scan from theta_q + noise): queued.
