# Map of regimes (Q5, H4): notes of agent `regime-map`

Status tags: **proven**, **derived** (assumptions stated), **empirical**, **conjecture**.
Experiments: `experiments/regime_map/`. Raw outputs: `/root/outputs/regime_map/`.

Notation: K cores, per-core init scale sigma (fan-in init), function-space scale alpha = sigma^K,
W_0 = W(theta_0), J = dW/dtheta at theta_0 (a D x P matrix, D = prod_i d_i), tangent operator
T = J J^T, centred model: L = 1/2 ||W(theta) - W_0 - W*||^2. Defaults: d = 3, odeco teacher with
weights [3, 2, 1] (||W*||^2 = 14).

## 1. Theory (derived here; to be compared with `notes/theory_regimes.md` when it lands)

*Written before `notes/theory_regimes.md` (draft) landed. That note agrees with every statement
below (its R2/R6 = 1.1, R4 = 1.2, R7-R8 = 1.3, R11 = 1.4, R10 = 1.5, R5 = 1.6) and goes further
(exact aligned S2S boundary, muP-TTN). The experiments test its predictions P1-P13; this section is
kept as the record of the independent derivation.*

### 1.1 Lazy limit of the centred model and its residual

Linearize: W(theta_0 + u) - W_0 = J u + O(||u||^2). Gradient flow of the linearized loss from u = 0
converges to u_lin = J^+ W* (minimum norm, it stays in the row space of J), and the function
converges to P_T W*, the orthogonal projection of W* onto the tangent space T_0 = range(J) of the
TTN manifold at W_0. **Derived** (standard linear least squares): the lazy limit fits W* only if
W* is in T_0; otherwise the loss converges to

    L_inf / L(0) = ||P_perp W*||^2 / ||W*||^2          (lazy residual)

**Representability (derived, generic init).** W(theta) has bond rank <= chi_e on every edge e, and
W_0 + W* generically has bond rank min(chi_e + r_e*, cap_e) on edge e (r_e* = teacher's rank on e,
cap_e = min(d^{|below e|}, d^{|above e|})). So the centred target W* + W_0 is exactly representable
only when chi_e = cap_e on every edge where the teacher has rank > 0, i.e. at maximal ("capped")
bond dimension. Below the cap, (i) the lazy limit leaves the residual above; (ii) at any sigma the
best attainable loss is the best tree-rank-chi approximation error of W_0 + W*, which goes to 0 as
sigma -> 0 (it is O(||W_0||^2) ~ alpha^2) and to the lazy residual as sigma -> infinity. Consequence
for the regime rules: below the cap, large-sigma runs are "stalled" by the convergence rule even
though the dynamics are kernel-like. This is a TTN-specific feature: the tangent kernel of a TTN
has rank at most dim T_0 < D unless the bonds are maximal, unlike the (infinite-dimensional,
positive definite) NTK of wide networks. Example: binary N = 4, d = 3: dim T_0 = 45, 56, 72, 80, 81
for chi = 3, 4, 6, 8, 9 (D = 81); for the N = 4 binary tree this is the rank-chi matrix manifold
(W = B_1 M B_2^T is a 3-factor matrix product across the bipartition 01|23), whose normal space at
a rank-chi point has dimension (9 - chi)^2.

### 1.2 Lazy boundary: kernel change ~ sigma^{-K}

The probe kernel Theta(theta) = J_p J_p^T is a homogeneous polynomial of degree 2(K-1) in theta.
Rescaling theta_0 -> c theta_0 rescales J by c^{K-1}, u_lin by c^{-(K-1)} and the directional
derivative D Theta[u_lin] by c^{K-2} c^{-(K-1)} c^{K-1} = c^{K-2}, while Theta_0 scales by
c^{2(K-1)}. Hence (**derived**, first order in u_lin, i.e. for large sigma):

    delta_K(T) ~ kappa / sigma^K = kappa / alpha,
    kappa := ||D Theta(theta_0^{(1)})[J^+ W*]||_F / ||Theta(theta_0^{(1)})||_F,

where theta_0^{(1)} is the init drawn at sigma = 1 with the same seed. kappa is computed at
initialization (per seed). Predicted lazy boundary for kernel threshold delta_thr:

    sigma_lazy = (kappa / delta_thr)^{1/K},     alpha_lazy = kappa / delta_thr.

Threshold sensitivity: changing delta_thr from 0.05 to 0.2 moves sigma_lazy by a factor 4^{1/K}
(1.59 for K = 3, 1.22 for K = 7).

### 1.3 Saddle-to-saddle boundary: separation ratio ~ sigma^{-(K-2)}

At small sigma, component r (weight s_r) escapes like a depth-K balanced mode
a' = a^{K-1}(s_r - a^K) from an effective amplitude a_0 = c_eff sigma (c_eff depends on chi, d,
tree and seed; the alignment phase is absorbed into it). For this ODE (u = a^K / s = learned
fraction):

    t_r(u) = (1/K) s_r^{-(2-2/K)} \int u^{-(2-2/K)} (1-u)^{-1} du,  t_r(1/2) ~ a_0^{-(K-2)} / ((K-2) s_r)
    width w_r = w_K s_r^{-(2-2/K)},   w_K = (1/K) \int_{0.1}^{0.9} u^{-(2-2/K)} (1-u)^{-1} du.

Event times differ by a_0^{-(K-2)} (1/s_{r+1} - 1/s_r)/(K-2) while widths are sigma-independent, so
(**derived**, aligned-mode approximation, K >= 3)

    rho_i ~ C_i sigma^{-(K-2)},   sigma_c = (C / rho_thr)^{1/(K-2)},   alpha_c = sigma_c^K,

with C = min_i C_i. For K = 2 the escape time is log(1/a_0)/s and rho ~ log(1/sigma). Threshold
sensitivity: rho_thr 2 -> 5 moves sigma_c by (5/2)^{1/(K-2)} (2.5x for K = 3, 1.20x for K = 7).

**Large-K limit (derived, if C and kappa grow at most exponentially in K):** sigma_c -> 1 and
sigma_lazy -> 1 as K -> infinity (H4), with log sigma_c = O(1/K); in the function-space scale
alpha the boundaries alpha_c ~ (C/rho_thr)^{K/(K-2)} and alpha_lazy ~ kappa/delta_thr have finite
limits. So the map in alpha is the K-stable description, and the window of rich non-incremental
dynamics shrinks to a band of width O(1/K) in log sigma.

### 1.4 Imbalance at fixed alpha: the small cores learn, the large cores are features

Per-core scales sigma_v with prod_v sigma_v = alpha. The tangent operator is a sum of per-core terms
T = sum_v J_v J_v^T with ||J_v||^2 ~ alpha^2 / sigma_v^2 (J_v is a contraction of all other cores),
so the kernel is dominated by the cores with the smallest sigma_v. Large cores barely move
relative to their size and act as fixed features; small cores carry the learning. If m cores are
small, the effective model is degree-m homogeneous (**derived**, scale separation): m = 1 gives a
linear (convex) model, i.e. lazy dynamics (with the lazy residual of 1.1 if the features are
incomplete); m = 2 gives depth-2 incremental dynamics with log(1/sigma) plateaus; m >= 3 gives
polynomial plateaus. Root vs rest, ratio lambda = sigma_root / sigma_rest at fixed alpha:
- lambda << 1 (small root, m = 1): lazy.
- lambda >> 1 (large root, m = K - 1 small cores): rich / saddle-to-saddle with effective degree
  K - 1 (for binary N = 4: m = 2, a 2-factor matrix factorization).
This is asymmetric, unlike two-layer linear networks where imbalance either way is lazy (Kunin et
al. 2024: rapid feature learning in linear nets only from balanced inits).

### 1.5 Large chi with fan-in init and unit learning rates ("standard parameterization")

For binary N = 4 (W = B_1 M B_2^T, B_i in R^{9 x chi}, M in R^{chi x chi}) with fan-in init:
B_i B_i^T ~ sigma^2 chi/9 I, M^T M ~ sigma^2/chi I. The root term of the kernel,
B_1 B_1^T (x) B_2 B_2^T ~ sigma^4 chi^2/81, dominates the leaf terms (~ sigma^4/9) for chi >> 3, so
the model becomes linear in the root core (fixed random features B_1 (x) B_2) and lazy as chi grows,
at any fixed sigma (**derived**; compare SP + MSE in wide nets where the readout dominates).

**Mechanism of the kernel change (derived).** For the N = 4 binary tree the leaf Gram changes by
B_1 dB_1^T = eta_B W_0 R^T (R = time-integrated residual): the gradient of a core contracted with that
core gives back the function W_0, not the core. Gram changes are therefore set by ||W_0|| ~ alpha
while the Grams themselves grow with chi (B_1 B_1^T ~ sigma^2 chi / 9 under fan-in init). This is why
the kernel change falls faster with chi than the naive "relative change of a core" estimate.

**Lazy coefficients vs chi (empirical, at init, binary N = 4, d = 3, 5 seeds, `pilots/pilot_kappa_chi.py`,
`pilots/pilot_kappa_plain.py`).** Two learning-rate rules: SP (eta_v = 1, the project default) and
NTK-type (eta_v = 1/F_v, which makes every core's expected kernel term equal,
E[J_v J_v^T] = sigma^{2(K-1)} F_v / d^N I with fan-in init).

| chi | kappa (centred), SP | kappa (centred), NTK | kappa_plain (large sigma), SP | kappa_plain, NTK |
|---|---|---|---|---|
| 12 | 1.39 | 2.48 | 0.24 | 0.74 |
| 24 | 0.113 | 0.89 | 0.054 | 0.53 |
| 48 | 0.0094 | 0.44 | 0.0058 | 0.28 |
| 96 | 0.00085 | 0.18 | 0.00071 | 0.15 |

kappa_plain is the sigma-independent large-sigma kernel change of the *plain* (uncentred) model,
whose lazy fit must remove W_0 (u = J^+ (W* - W_0) with ||W_0|| >> ||W*||). Readings: (i) under SP
both coefficients fall like ~chi^{-3}: the root (fan-in chi^2) dominates the kernel and the model
becomes linear in the root, so wide SP TTNs are lazy even without centring (chi >~ 20 at N = 4);
(ii) under NTK-type rates both fall like ~chi^{-1}, so laziness of the plain model needs
chi >~ 100 at N = 4; (iii) at large chi the centred and plain models have comparable kernel changes
at large sigma, consistent with the literature correction that centring matters only at finite chi
(`notes/literature.md` section 3, H4). The relative parameter change of the plain model matches the
isotropic-kernel estimate sqrt(d^N / (sum_v F_v sum_e chi_e)) under SP for chi >= 24.

### 1.6 Two-copy model: same lazy residual, kernel change second order

Two-copy model W = W(theta+) - W(theta-), theta+ = theta- = theta_0 at t = 0 (masked doubled TTN).
Its Jacobian is [J, -J], so the tangent space (hence the lazy residual of 1.1) is the same as the
centred model's, and the kernel is 2 J J^T. The linearized solution is theta+- = theta_0 +- u/2, and
the kernel Theta = J(theta+) J(theta+)^T + J(theta-) J(theta-)^T changes only at second order: the
first-order terms D Theta[u/2] + D Theta[-u/2] cancel (**derived**). Hence

    delta_K(two-copy) ~ kappa_2 / sigma^{2K}   versus   delta_K(centred) ~ kappa / sigma^K,

so at equal sigma the two-copy model is much lazier, and its lazy boundary sits at
sigma_lazy = (kappa_2/delta_thr)^{1/(2K)}. The S2S side is expected to agree qualitatively (both copies
start at the same small scale). Pilot check (binary N = 4, chi = 24, 2 seeds): delta_K ratio between
sigma = 3 and 10 is 1362 (two-copy) vs predicted (10/3)^6 = 1372, and 40.6 (centred) vs (10/3)^3 = 37.
**Empirical (pilot).**

## 2. Results

All runs: whitened inputs, exact population loss, d = 3, odeco teacher [3, 2, 1] (teacher seed
1000 + init seed), fan-in init, gradient flow. Labels: `notes/diagnostics.md` (converged = L(T) < 1e-3
L(0)), at all 9 threshold pairs. Every number below states tree, chi, sigma range and seed count.

### 2.1 sigma x chi, binary N = 4 (K = 3)

*Interim data (scipy RK45, rtol 1e-9, 6 seeds x 8 chi x 19 sigma in [0.01, 40]; to be replaced by the
library rerun with linear-time event refinement, `sigma_chi_b4`).*

- **Three regions plus a stall region** (`results/phase_b4_sigma_chi_centred.png`): S2S for
  sigma < sigma_c(chi); rich non-incremental in between; lazy for sigma > sigma_lazy(chi) only if
  chi >= 9 (= the cut dimension d^2); for chi < 9 the runs stall above sigma ~ 0.8-1.3. **Empirical.**
- **S2S boundary.** sigma_c (S2S fraction 1/2) = 0.25, 0.13, 0.14, 0.13, 0.10, 0.071, 0.076, 0.045 for
  chi = 3, 4, 6, 9, 12, 18, 24, 36. Predicted per seed from its deepest run only (aligned-mode
  extrapolation of the three event times, `analysis.aligned_sigma_c`): medians 0.34, 0.13, 0.13, 0.14,
  0.10, 0.075, 0.074, 0.043 - within one grid step (x1.585) everywhere. **Empirical** test of the
  **derived** small-sigma law rho_i = R_i sigma^-(K-2) + O(1) with aligned-mode O(1) terms.
- **Stall = realizability.** For chi < 9 the final loss equals the best rank-chi fit of W_0 + W* across
  the 01|23 cut (exact SVD) in every run, and the stall onset is where that best fit reaches 1e-3 L(0).
  **Empirical**, confirming theory R6.
- **Lazy boundary** (chi >= 9): sigma_lazy = 3.0, 2.1, 1.59, 1.26, 1.13 for chi = 9, 12, 18, 24, 36; first-
  order prediction (kappa/0.1)^(1/3) with kappa from linear response at init: 3.8, 2.4, 1.45, 0.98, 0.59.
  It holds where x = ||W*|| / sigma^3 <~ 1 (chi <= 18) and underestimates sigma_lazy when the predicted
  boundary has x > 1 (chi >= 24): there delta_K is dominated by higher orders in x. At sigma >= 3 the
  measured delta_K equals kappa / sigma^3 within 1% (chi = 24). **Empirical.**
- **Threshold sensitivity.** S2S boundary ratio between rho thresholds 2 and 5: 2.2-2.55 for every chi
  (predicted (5/2)^(1/(K-2)) = 2.5). Lazy boundary ratio between delta_K thresholds 0.05 and 0.2:
  2.4, 1.7, 1.6, 1.1, 1.5 for chi = 9..36 (first-order prediction 4^(1/3) = 1.59; grid step 1.585).


### 2.x Two-copy cross-check (binary N = 4)

*Interim data (scipy RK45), 6 seeds x 8 chi x 19 sigma.*

- **Lazy side agrees in mechanism, not in position.** Same lazy residual as the centred model at
  deficient bonds (chi = 3: 0.49 / 0.25 for seeds 0 / 1 in both models); the kernel change is second
  order, delta_K ~ sigma^-2K: ratio 1362 between sigma = 3 and 10 at chi = 24 (predicted (10/3)^6 =
  1372; centred: 40.6 vs (10/3)^3 = 37). Hence the two-copy lazy boundary is lower (0.80 vs 1.13 at
  chi = 36). **Empirical**, confirming theory_regimes R5.
- **S2S side differs (new, empirical).** At small sigma the two copies start identical (theta+ =
  theta-), so their near-origin escapes happen together: each copy learns *half* of components 1 and 2
  at the same time (binary N = 4, chi = 36, sigma = 0.01, seed 0: a+ = (0.502, 0.498, 0), a- = (-0.498,
  -0.502, 0) at t = 32; `pilots/twocopy_mechanism.py`). Events 1 and 2 then merge: t_2 - t_1 stays
  ~2 for sigma from 0.01 to 0.04 (instead of growing like 1/sigma), rho_1 ~ 2, and the run is labelled
  rich at every sigma in the grid for chi >= 18 (S2S boundary 0.015 vs 0.075 centred at chi = 24).
  So the two-copy model is a faithful cross-check of the lazy boundary (up to the predicted shift) but
  **not of the S2S boundary at large chi**: its incremental structure is changed by the mirror copy.

### 2.y Integrator and diagnostic checks

- **Two independent integrators agree** (binary N = 4, 1937 matched runs of the sigma x chi grid,
  centred / plain / two-copy): scipy RK45 (rtol 1e-9, atol 1e-14, log-time recording) vs the library's
  DP5 (rtol 1e-8, atol 1e-12) with linear-time event refinement. Labels agree in 98.1% of runs; all 37
  differences are rich (scipy) -> S2S (library) or the reverse near rho = 3, explained by the
  grid-limited widths of the scipy runs (which overestimate widths and so underestimate rho); delta_K
  agrees to 0.08% (median relative difference over lazy runs). Conservation-law drift (library
  Recorder): max 1e-9 over all runs.
- **Event widths** in deep S2S match the aligned-mode theory: 0.466, 0.798, 2.010 for s = 3, 2, 1
  (theory 0.46, 0.80, 2.01; K = 3, chi = 36, sigma = 0.01).
