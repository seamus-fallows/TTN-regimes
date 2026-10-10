# Map of regimes (Q5, H4): notes of agent `regime-map`

Status tags: **proven**, **derived** (assumptions stated), **empirical**, **conjecture**.
Experiments: `experiments/regime_map/`. Raw outputs: `/root/outputs/regime_map/` (copied to
`data/regime_map/` of the main checkout on 2026-10-10; fix-round reruns and the tensor-kernel cache in
`data/regime_map_local/`). **Fix round 1 (section 3) corrects several claims below; sections 0-2 are
updated in place and every change is listed in section 3.**

> **2026-10-10, verification of round 2:** errata E1-E19 of `notes/reviews/regime_map_round3_verification.md`
> are applied in place in sections 0, 4 and 5, each marked "(E#)"; the list is in section 4.9.

Notation: K cores, per-core init scale sigma (fan-in init), function-space scale alpha = sigma^K,
W_0 = W(theta_0), J = dW/dtheta at theta_0 (a D x P matrix, D = prod_i d_i), tangent operator
T = J J^T, centred model: L = 1/2 ||W(theta) - W_0 - W*||^2. Defaults: d = 3, odeco teacher with
weights [3, 2, 1] (||W*||^2 = 14).

## 0. Summary: which parameter moves which boundary

Status tags as in the project. Setting: whitened inputs, exact population loss, odeco teachers,
fan-in init, gradient flow; centred model (target W* + W_0) unless stated; 6 seeds per point.

| parameter | S2S boundary sigma_c | lazy boundary sigma_lazy | other effects |
|---|---|---|---|
| bond dimension chi (binary N = 4, full bonds, chi = 16-128, < 1 decade) | ~ chi^(-0.53 +- 0.05); measured sigma_c sqrt(chi) = 0.26-0.33 (**empirical**; not derived) | ~ chi^(-0.50 +- 0.01) (tensor kernel, chi 32-128); inside P5's bracket at chi = 24-96 and 0.07% below its lower edge at chi = 128 (sigma_lazy sqrt(chi) = 6.40-6.56 at chi >= 32; asymptote 6.50); probe kernel ~5% lower (**empirical**, confirms **derived** P5) | both boundaries ~ chi^-1/2 at N = 4: the map collapses in sigma sqrt(chi), the TTN analogue of known width scales (Tu et al. 2024, Geiger et al. 2020); sigma^K chi^(K-1) disfavoured at 2.7-2.9 SE (per-seed fits; 3.7 SE pooled) (2.1b, 3.2 item 2) |
| chi below the cut dimension | moves down with chi (0.24 at chi = 3 -> 0.14 at chi = 9, N = 4) | **no lazy regime**: centred runs stall at the Eckart-Young floor of W_0 + W* (**derived** R7, **empirical** 44/44 runs) | the stall onset (sigma = 1.0-1.6) is 1.6-2.2x above P4's small-sigma estimate; plain model converges instead (2.1) |
| chi = teacher rank, K >= 7 | masked by stalls (dead routes, fan-in) | - | balanced init removes dead routes, not the stall (2.2) |
| number of cores K (centred, d = 2, chi = 4) | label rule: 0.17, 0.49, 0.61, 0.68, 0.75 for K = 3-7 (resolved seeds); K >= 8 only brackets (5/6, 5/6, 6/6, 5/6 seeds unresolved at K = 8, 10, 12, 15). Floor-stalled runs classified by rho: K = 8: 1.01 (5/6 seeds, rho crossings; deficient bonds, floor-stalled runs); at K = 10 and 12 the brackets mostly end at runs that learned nothing (no S2S -> rich crossing at K = 12, 2/6 seeds at K = 10, both 0.90; verification E1). **Round 2 (section 4.7-4.8, realizable, d = 2 and 3):** plain model with spare channels: at K >= 8, 85 of 96 seeds are censored from above (76 S2S-like at every sampled sigma up to the event rule's domain limit, 9 still open when the budget ended) and 11 cross, all below sqrt(d) (0.86-1.37 at d = 2; at d = 3 one seed of 48, at 1.08) (E1, E2, E5); the domain limit has median ~sqrt(d) at K = 8-15 but tends to ~1.15 sqrt(d) (E4); the lower bounds (1.07-2.03 at K >= 8, E12) all exceed 1 and both extrapolated laws, so sigma_c does not tend to 1 (H4 rejected for the plain model at K <= 15, **empirical**); mechanism H_W (the plain model unlearns W_0 on the plateau): consistent with the data, not tested causally, **conjecture** (E3); centred full bonds (K <= 8): beta = -0.07 +- 0.51, unresolved; sigma_c -> sqrt(d) remains a **conjecture** | not mapped beyond K = 7 (K = 7: lazy only with 2x-overparameterized full bonds, kernel boundary 1.45) | sigma^K = const and sigma^K / sqrt(D) = const fit equally on K = 4-7 and, with crossing-terminated brackets only, on K = 4-10 (chi^2 7.7 vs 6.9, 5 dof; verification E2); stalls take over the S2S side for MPS at K >= 8 (2.3, 3.2 item 4) |
| root imbalance lambda at fixed alpha (N = 4, chi = 16-24) | S2S majority up to alpha = 1e-4 for 0.1 <= lambda / sqrt(chi) <= 10, lower at strong imbalance (coarse grid) | light root moves the lazy boundary down by ~100x in alpha; a heavy root is lazy from alpha = 10, as at balance (probe kernel; with the tensor-kernel labels of section 5: from alpha = 100 at balance and for heavy roots, from 1 for a light root) | delta_K ~ lambda^2 (local slope 1.85 +- 0.06 vs P20's 1.8-1.9), saturating at 0.59-0.61 (P20: 0.6-0.7; tensor kernel) (2.4) |
| parameterization at large chi (N = 4, tensor kernel, chi = 16-128) | - | SP: delta_K ~ chi^(-3.02 +- 0.06); NTK: chi^(-0.79 +- 0.04), -1.01 between chi = 64 and 128; muP-TTN: delta_K -> 6.2 (slope +0.09 +- 0.02) | matches theory P16 (2.5) |
| construction (two-copy vs centred) | two-copy: events 1-2 merge at large chi (not comparable) | two-copy lower by 0.79 (tensor kernel, chi 32-128; P6: 2^(-1/3) = 0.79); two-copy boundary within 2% of P6 at chi >= 32 | two-copy escapes from the lazy residual at deficient bonds (2.x) |
| thresholds | rho 2 -> 5 moves sigma_c by x2.13-2.34 (asymptotic prediction 2.5) | delta_K 0.05 -> 0.2 moves it by x1.97 (chi = 9) -> 1.30-1.35 (chi >= 32, both kernels); P7: 1.3-1.36 | labels robust; boundaries shift as predicted (2.1) |

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

**Large-K limit (derived, if C and kappa grow sub-exponentially in K, log C = o(K); fix round 1:
"at most exponentially" was wrong, an exponential C gives a limit other than 1):** sigma_c -> 1 and
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

Library gradient flow (DP5, rtol 1e-8, atol 1e-12; events re-integrated on a dense linear grid),
6 seeds x 8 chi (3, 4, 6, 9, 12, 18, 24, 36; uncapped) x 19 sigma in [0.01, 40], centred model;
`results/phase_sigma_chi_b4_centred.png`, per-seed tables `results/sensitivity_per_seed_sigma_chi_b4_centred.csv`
and `results/fix1/s2s_per_seed_sigma_chi_b4.csv`, table `results/fix1/table_2_1_s2s.csv`.
Boundaries are per seed (geometric mean x/ standard-error factor over 6 seeds); "S2S boundary" =
sigma where the seed's min_i rho_i drops below 3 (log-log interpolation), "kernel boundary" = sigma
where delta_K(T) drops below 0.1 (converged runs). *Fix round 1:* last-crossing event times (two seeds
change, section 3.2), n per cell, and the kernel boundary with both kernels.

| chi | S2S boundary (n) | deepest-run extrapolation | extrapolation / measured, mean +- SE (per-seed range) | rho exponent on the S2S side (mean +- SE) | kernel boundary: probe (label rule) / tensor-space | ratio of S2S boundaries rho 2 / rho 5 | ratio of kernel boundaries dK 0.05 / 0.2 (probe) |
|---|---|---|---|---|---|---|---|
| 3 | 0.237 x/ 1.32 (6) | 0.283 x/ 1.35 | 1.21 +- 0.08 (1.02-1.53) | -1.058 +- 0.022 | stalled (floor) | 2.18 +- 0.05 | - |
| 4 | 0.099 x/ 1.51 (6) | 0.118 x/ 1.50 | 1.25 +- 0.20 (1.00-2.24) | -1.075 +- 0.045 | stalled | 2.13 +- 0.08 | - |
| 6 | 0.137 x/ 1.19 (6) | 0.143 x/ 1.19 | 1.05 +- 0.02 (1.00-1.12) | -1.030 +- 0.004 | stalled | 2.22 +- 0.04 | - |
| 9 | 0.136 x/ 1.11 (6) | 0.141 x/ 1.11 | 1.04 +- 0.01 (1.01-1.07) | -1.030 +- 0.004 | 3.06 x/ 1.07 / 3.40 x/ 1.09 | 2.27 +- 0.02 | 1.97 +- 0.08 |
| 12 | 0.075 x/ 1.33 (6) | 0.088 x/ 1.27 | 1.25 +- 0.22 (1.00-2.37) | -1.070 +- 0.041 | 2.20 x/ 1.05 / 2.37 x/ 1.03 | 2.23 +- 0.07 | 1.62 +- 0.03 |
| 18 | 0.072 x/ 1.07 (6) | 0.073 x/ 1.08 | 1.02 +- 0.01 (1.00-1.05) | -1.031 +- 0.000 | 1.66 x/ 1.05 / 1.75 x/ 1.03 | 2.30 +- 0.01 | 1.46 +- 0.03 |
| 24 | 0.067 x/ 1.12 (6) | 0.068 x/ 1.12 | 1.01 +- 0.00 (1.00-1.02) | -1.030 +- 0.002 | 1.31 x/ 1.04 / 1.40 x/ 1.03 | 2.33 +- 0.01 | 1.38 +- 0.02 |
| 36 | 0.043 x/ 1.05 (6) | 0.044 x/ 1.06 | 1.01 +- 0.00 (0.99-1.01) | -1.034 +- 0.003 | 1.04 x/ 1.04 / 1.09 x/ 1.02 | 2.34 +- 0.01 | 1.33 +- 0.01 |

- **Map** (`phase_sigma_chi_b4_centred.png`): S2S for sigma below the S2S boundary, rich in between,
  lazy above the kernel boundary only for chi >= 9 (full bonds); for chi < 9 the runs stall above
  sigma ~ 0.6-1.0 at the best fit of W_0 + W* (unrealizable centred target). **Empirical.**
- **S2S side.** Per seed, rho scales as sigma^-(1.02-1.05) (6 seeds per chi), except four seeds with
  transients above 1/2 before the component is learned (chi = 3 seeds 1, 5; chi = 4 seed 3; chi = 12
  seed 4: -1.09 to -1.27), which approach rho ~ 1/sigma slowly; predicted sigma^-(K-2) (theory_regimes
  R9/P12; the few-% excess is the O(1) term of rho = R sigma^-1 + c). The deepest-run extrapolation
  (`analysis.aligned_sigma_c`) calibrates each component on the deepest run of the same seed, so the
  extrapolation / measured ratio (0.99-1.12 per seed at chi >= 6 apart from chi = 12 seed 4, 2.37) is a
  consistency check of rho ~ 1/sigma, not an a-priori prediction; the chi-dependence of the boundary is
  not derived. At chi = 3, 4 the ratio (1.21, 1.25) is pulled up by the transient seeds (the other
  seeds: 1.02-1.16, 1.00-1.08), not by the realizability floor, which lies 4-10x higher in sigma. The
  S2S boundary falls with chi (0.24 at chi = 3, 0.043 at chi = 36). **Empirical.** At chi = 3 it agrees
  with theory P11 (plain model, 30 seeds: per-seed median 0.27-0.31).
- **Lazy side.** Kernel boundary (probe kernel, the label rule) 3.06, 2.20, 1.66, 1.31, 1.04 at chi = 9,
  12, 18, 24, 36; with the tensor-space kernel of the theory 3.40, 2.37, 1.75, 1.40, 1.09, against
  theory_regimes P5's bracket [1.38, 1.69], [1.24, 1.42], [1.04, 1.12] at chi = 18, 24, 36 (inside at
  chi = 24 and 36, 3% above at chi = 18 where the isotropic constants are not expected to hold). The
  first-order law (kappa/0.1)^(1/3) is right only at chi <= 12 (2.4 at chi = 12; 0.59 at chi = 36).
  **Empirical**, confirming the theory's correction that the coherent second-order kernel change sets
  the lazy boundary at large chi (section 2.1b for chi up to 128).
- **Threshold sensitivity.** rho threshold 2 -> 5 moves the S2S boundary by 2.13-2.34 (asymptotic
  prediction (5/2)^(1/(K-2)) = 2.5; the O(1) term in rho lowers it at finite sigma). delta_K threshold
  0.05 -> 0.2 moves the kernel boundary by 1.97 (chi = 9) down to 1.33 (chi = 36): the local slope of
  delta_K vs x = ||W*|| / sigma^3 grows from ~1 to ~2 with chi, so the ratio goes from 4^(1/3) = 1.59
  toward 4^(1/6) = 1.26 (theory P7: 1.3-1.36 at chi >= 32). **Empirical.**
- **Realizability.** For chi < 9 every stalled run (44) ends at the Eckart-Young minimum of W_0 + W*
  across the 01|23 cut (`l_end / l_best` = 1 to 1e-6). Stall onset (all 6 seeds stalled) at sigma = 1.0,
  1.0, 1.59 for chi = 3, 4, 6 (first stalled seed at 0.63 for chi = 3), 1.6-2.2x above theory P4's
  small-sigma estimate sigma_floor = 0.56, 0.60, 0.71, increasingly so at chi = 6 where the floor is
  reached only at sigma ~ 1, outside the small-sigma expansion. A run that ends on a floor below 1e-3
  counts as converged (chi = 6, sigma = 1.0: floors 1.6e-4 to 8.8e-4, labelled rich). **Empirical**,
  confirming theory R7 qualitatively.

### 2.1b Boundaries vs bond dimension and the collapse variable (binary N = 4)

`configs/chi_b4.yaml`: chi = 9..128 (uncapped; full / overparameterized bonds), 6 seeds, centred model
over sigma in [0.004, 4] (22 values) and two-copy over [0.25, 4]; library GF with event refinement;
figure `results/boundaries_vs_chi_chi_b4.png`, per-seed tables `results/boundaries_per_seed_chi_b4.csv`,
`results/fix1/s2s_per_seed_chi_b4.csv`, `results/fix1/lazy_per_seed_chi_b4.csv`. Boundaries per seed as
in 2.1 (S2S: label rule, last-crossing events; lazy: delta_K = 0.1 crossing); slopes fitted to all
per-seed values (each (chi, seed) is an independent init, but a seed shares its teacher across chi, so
per-seed fits are given too), errors from the residual scatter (`results/fix1/slopes_vs_chi_chi_b4.csv`,
`lazy_slopes_chi_b4.csv`). Corrected in fix round 1 (section 3.2 items 1-2).

| | slope vs chi | chi range (decades) | per-seed values | per-seed fits (mean +- SE) | prediction |
|---|---|---|---|---|---|
| S2S boundary (centred) | -0.528 +- 0.053 | 16-128 (0.9) | 42 | -0.528 +- 0.073 | (none derived) |
| S2S boundary (centred) | -0.548 +- 0.078 | 32-128 (0.6) | 30 | -0.548 +- 0.074 | |
| S2S boundary (centred) | -0.552 +- 0.049 | 9-128 (1.15) | 54 | -0.552 +- 0.036 | |
| lazy boundary, probe kernel (centred) | -0.543 +- 0.015 | 16-128 | 42 | -0.543 +- 0.012 | -1/2 (theory P5, second order) |
| lazy boundary, probe kernel (centred) | -0.522 +- 0.020 | 32-128 | 30 | -0.522 +- 0.013 | -1/2 |
| lazy boundary, tensor kernel (centred) | -0.553 +- 0.013 | 16-128 | 42 | -0.553 +- 0.014 | -1/2 |
| lazy boundary, tensor kernel (centred) | -0.504 +- 0.011 | 32-128 | 30 | -0.504 +- 0.011 | -1/2 |
| lazy boundary, tensor kernel (two-copy) | -0.499 +- 0.014 | 32-128 | 18 | -0.499 +- 0.010 | -1/2 (P6) |

- **Lazy boundary vs theory P5/P6 (empirical, quantitative, tensor-space kernel).** Per-seed medians
  1.13, 0.94, 0.82, 0.66, 0.571 at chi = 32, 48, 64, 96, 128 lie inside P5's bracket [1.10, 1.20],
  [0.92, 0.96], [0.80, 0.82], [0.66, 0.67] and 0.07% below the lower edge of [0.5712, 0.5773] at
  chi = 128 (sigma_lazy sqrt(chi) = 6.40, 6.49, 6.56, 6.48, 6.46 against the asymptote 6.50); at
  chi = 9-16 they are 1.5-6.3% above the bracket (+6.3, +2.9, +1.5%; chi = 9 has 3 seeds with a tensor boundary). Two-copy: 0.90,
  0.65, 0.45 at chi = 32, 64, 128 against P6's 0.91, 0.65, 0.46 (sigma sqrt(chi) = 5.06, 5.21, 5.13 vs
  5.16), and two-copy / centred = 0.791, 0.794, 0.794 (P6: 2^(-1/3) = 0.79). delta_K(x) near the boundary
  (0.03 < delta_K < 0.3, chi >= 32, 43 runs) is 0.96 (median; 0.69-1.82) of R4's
  sqrt((c_1 x)^2 + (c_2 x^2)^2), i.e. delta_K ~ c_2 x^2 there (`lazy_r4_check_chi_b4.csv`). The probe
  kernel (label rule) gives 0.54-1.11 (median 0.78) of the tensor-space delta_K there, hence probe-kernel
  boundaries 5-6% lower (sigma_lazy sqrt(chi) = 6.10-6.29); the old "5-8% low, as in the theory note's
  GF checks" compared kernels of two kinds and is withdrawn. Threshold ratio (delta_K 0.05 vs 0.2)
  1.30-1.35 at chi >= 32 with either kernel (P7: 1.3-1.36).
- **The S2S boundary scales the same way (empirical; N = 4, fan-in, chi = 16-128).** Measured
  sigma_c sqrt(chi) = 0.27, 0.33, 0.27, 0.33, 0.28 at chi = 32..128 (medians, quantized by the sigma
  grid; the deepest-run extrapolations give 0.29, 0.29, 0.31, 0.30, 0.27). So at large chi both
  boundaries depend on sigma and chi mainly through sigma sqrt(chi), i.e. through the per-bond Gram
  scale p = chi sigma^2 / d^2 (the eigenvalue scale of the subtree Gram matrices B B^T): S2S below
  p ~ 0.01, lazy above p ~ 4.3, and the rich window is a fixed factor ~18-23 in sigma (median per chi,
  chi >= 32; ~10^4 in alpha = sigma^3). **Collapse test** of the candidate variables suggested in
  `notes/literature.md` (H4, correction 2): sigma^K (slope 0) is rejected; the per-path amplitude times
  the number of paths (sigma^K chi^(K-1) in iid normalization = sigma^3 chi / d^2 in fan-in variables,
  slope -1/3) is disfavoured at 3.7 standard errors for chi 16-128 and 2.8 for chi 32-128 (S2S; the
  lazy slope is far from -1/3); the per-bond Gram scale (slope -1/2) is consistent with both
  boundaries. The fits span 0.6-0.9 decade (1.15 with chi = 9). **Empirical** for binary N = 4 (a
  3-factor matrix product) with fan-in init only; deeper trees untested at large chi. Not a new scaling
  idea: it is the TTN analogue of the width-dependent scales sigma^2 w of Tu, Aranguri & Jacot
  (NeurIPS 2024) and alpha sqrt(h) of Geiger et al. (J. Stat. Mech. 2020).
- Why the S2S boundary moves like the Gram scale is not derived here. A heuristic: the near-origin
  escape is set by the initial amplitude of the best rank-one direction of each core; the leaf cores'
  Gram B B^T ~ p grows with chi, while the root is light by sqrt(chi) (E Q = sigma^2 (1 - 1/chi) on the root
  edges), so the effective per-core amplitude grows like sigma sqrt(chi). **Conjecture.**

### 2.2 Stalls at chi = teacher rank (K >= 7)

Stall test (`configs/stall_test*.yaml`; binary N = 8 (K = 7) and MPS N = 8 (K = 8), d = 3, chi = 3 =
teacher rank, sigma in {0.3, 0.45, 0.6}, 6 seeds; LSODA, budget T = 50 t_esc + 1e4 (1.5e5-1.7e6
time units at these sigma)). Stall type from the bond spectra of W at the end (3rd singular value,
minimum over edges): dead route < 1e-12, decaying 1e-12..1e-4, long plateau >= 1e-4.

| tree | model | init | converged (S2S) | stalled: dead route | stalled: decaying |
|---|---|---|---|---|---|
| binary 8 | plain | fan-in | 4/18 | 12/18 | 2/18 |
| binary 8 | plain | gauge-balanced | 6/18 | 0 | 12/18 |
| MPS 8 | plain | fan-in | 0/18 | 16/18 | 2/18 |
| binary 8 | centred | fan-in | 4/18 | (spectra of W - W_0 recorded) | 14/18 stalled |
| binary 8 | centred | gauge-balanced | 8/18 | | 10/18 stalled |

- With chi equal to the teacher rank, most K >= 7 runs never learn the last component within the
  budget, in both the plain and the centred model, so the stall is not caused by W_0 in the centred
  target. With fan-in init the stalled route is dead (3rd bond singular value at roundoff: the
  recorded 1e-16 to 1e-18 are below the SVD's resolution, read them as "at roundoff"); gauge-balanced init (same W_0, every Q_e = 0, `balance.py`) removes the dead routes but the
  route still decays (s_3 ~ 3e-5) and most runs do not escape. **Empirical** (36 + 18 runs), confirming
  theory_regimes R10 / P14 and the conservation-law trap of theory_saddles R12.
- Consequence for the map: at chi = teacher rank and K >= 7, "stalled" occupies the whole S2S side
  at practical budgets; the S2S boundary is measured with spare channels or full bonds (2.3).

### 2.3 sigma x K: the S2S boundary and H4

Sweeps at d = 2, odeco teacher [3, 2], 6 seeds: centred model (`sigma_K_centred_d2`; chi = 4 = 2 x
teacher rank on every tree, plus full bonds for MPS 6 / binary 8 / MPS 8), plain model
(`sigma_K_d2`, chi = 2 and 4; `sigma_K_full_d2`, full bonds), trees binary N = 4, 8, 16 and MPS N = 4, 5,
6, 8, 10, 12 (K = 3..15); LSODA; per-seed boundaries (`seed_s2s_boundary`). Figures
`phase_sigma_K_*.png`, `sigma_c_vs_K.png`, `stalls_vs_K.png`; table `sigma_c_vs_K.csv`.

*Corrected in fix round 1 (section 3, item 4); the committed bullets claimed a resolved boundary up to
K = 12 and a "derived" sqrt(d) limit.*

- **Centred model, chi = 4, label rule** (`seed_s2s_boundary`: the first block of S2S runs, skipping
  runs that stalled below it; boundary = geometric mean of the block's last sigma and the next one;
  'unresolved' when that next run stalled). Resolved seeds / 6 and medians: K = 3: 6/6, 0.17
  [0.14, 0.20]; 4: 6/6, 0.49; 5: 6/6, 0.61; 6: 4/6, 0.68 (resolved seeds; 0.80 with the two unresolved
  brackets); 7: 6/6, 0.75. For K = 8, 10, 12, 15 only 1, 1, 0, 0 of 6 seeds are resolved (5, 5, 6, 5
  unresolved: the next run up is floor-stalled; K = 15 also 1 'all_s2s'); the old medians 1.01, 0.90,
  0.92 and [0.63, 1.36] are brackets [last S2S, first floor-stalled run], i.e. lower bounds of the
  label boundary, not crossings (`sigma_c_vs_K.csv` had shown 0, 2, 0, 0 unresolved: counter bug).
- **Floor-stalled runs classified by min_i rho_i** (the review's rule; `results/fix1/k_sweep_*.csv`,
  from the copied raw outputs; rule in section 3.2 item 4): the first floor-stalled run above the label
  bracket is S2S-like (rho > 3) in 13 of the 18 unresolved MPS seeds. Resolved seeds and medians then:
  K = 6: 6/6, 0.80; K = 8: 5/6, 1.01 (rho crossings); K = 15: none (missing grid points). *Erratum E1
  (verification of fix round 1):* the old "K = 10: 1.16; K = 12: 1.15" are not S2S -> rich crossings: at
  K = 12 all 5 brackets end at a run that learned nothing within its budget (0/6 seeds reach rho <= 3);
  at K = 10 only 2 seeds end at a crossing (both 0.90). MPS 10/12 S2S-like floor-stalled runs persist up
  to sigma ~ 1.0 (5/6 seeds) or 1.3 (1 seed) at K = 12, and the next runs learn nothing. These runs end on the floor of the unrealizable centred target (chi = 4 is deficient for
  MPS 6-12), so they do not measure the H4 boundary of a realizable problem. **Empirical.**
- **Scaling laws** (`results/fix1/k_sweep_laws.csv`, per-K medians with bootstrap errors over seeds):
  on K = 4-7 both u_0 = sigma_c^K = const and u_0 = sigma_c^K / sqrt(D) = const fit (chi^2 0.79 and
  0.59 for 3 dof; spreads 4.6x and 3.7x), and with only label- or rho-terminated brackets also on
  K = 4-10 (chi^2 7.7 vs 6.9, 5 dof; erratum E2: the old "sigma^K = const fails (chi^2 22.5)" came from
  stall-terminated brackets and is withdrawn). The old "within 20% in sigma" test was weak
  (sigma_c = u^(1/K) compresses a factor 2-9 in u into 20% in sigma). **Empirical**; neither law is
  established.
- **H4 and sqrt(d): conjecture.** "sigma_c(K) -> sqrt(d), not 1" was tagged "derived from R9"; R9's
  effective-amplitude law is heuristic, only d = 2 was run, and the K >= 8 values are brackets or
  floor-limited, so sqrt(d) and 1 cannot be distinguished. Round 2 (section 4) tests realizable versions. **Conjecture** (as R9 / P15 in `notes/theory_regimes.md`); a test needs
  full bonds (or spare channels with the plain model), longer budgets at K >= 8, and d = 3.
- **Stalls take over at large K.** Even with spare channels, MPS trees stall on the S2S side (MPS 8:
  5/6 seeds stall at the smallest sigma; MPS 12: no seed reaches a resolved label boundary; centred floor
  above sigma ~ 1). Binary trees stall less (binary 16: S2S up to sigma = 0.63-1.0). See
  `stalls_vs_K.png`; at chi = teacher rank the stalled fraction is higher at every K >= 6 (2.2).
- **Plain model.** Up to K = 8 the plain and centred boundaries agree within a grid step (0.17 vs 0.17
  at K = 3 (fix round 1: the old "0.20" did not match `sigma_c_vs_K.csv`), 0.61 vs 0.61 at K = 5, 0.75
  vs 0.75 at K = 7; at K = 8 both are brackets, see section 3: the initial function is negligible there). At K >= 10 the plain boundary exceeds the centred one (binary
  16: 1.50 vs 1.02): at sigma > 1 the plain run first collapses W_0 (loss / (1/2 ||W*||^2) from ~190 to
  ~1 by t ~ 0.1) and then learns saddle-to-saddle from the small residual scale. The plain model is
  therefore not a test of H4 at large K.

### 2.4 Imbalance: root scale at fixed function-space scale (binary N = 4)

`configs/imbalance_b4.yaml`: root init lambda sigma, other cores sigma, alpha = lambda sigma^3 fixed;
centred model; chi = 24 (full, overparameterized) and chi = 3 (deficient): 9 alpha (1e-6..100) x 10 lambda
(1e-2..1e2 x the expected-balance point lambda_bal = sqrt(chi)) x 6 seeds; plus theory P20's check point
chi = 16, alpha = 3.375, lambda = 0.01..100; `results/phase_imbalance_b4.png`.

- **Check point P20 (chi = 16, alpha = 3.375)** (corrected in fix round 1, section 3 item 6; the panel
  of the committed figure was empty, the numbers come from the reviewers' reruns of the committed code,
  `results/fix1/imbalance_P20.csv`). Probe-kernel median delta_K = 8.3e-5, 7.4e-4, 0.0069, 0.036, 0.167,
  0.45, 0.53, 0.50, 0.50 for lambda = 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100; with the tensor-space kernel
  of the theory 1.1e-4, 9.9e-4, 0.0096, 0.052, 0.22, 0.51, 0.61, 0.60, 0.59. Per-seed local slopes
  d ln delta_K / d ln lambda (tensor kernel, mean +- SE, 6 seeds) 1.97 +- 0.01, 1.85 +- 0.06, 1.55 +- 0.11,
  1.22 +- 0.09 from the smallest lambda up: delta_K ~ lambda^2 as lambda -> 0 (P20: 1.8-1.9 between 0.03
  and 0.1), saturating at 0.59-0.61 for lambda >= 10 (at the lower edge of P20's 0.6-0.7). Labels: lazy for lambda <= 0.3 (6/6),
  rich for lambda >= 1 (5/6 at lambda = 1). **Empirical**, confirming theory R13 / P20 (slope and
  saturation; the old "~20% below the prediction" was the probe kernel).
- **Map, chi = 24 (majority labels, 6 seeds; one-decade alpha grid).** The lazy boundary moves to
  smaller alpha as the root shrinks: lazy for alpha >= 0.1 at lambda / sqrt(chi) = 0.01 (4/6 at 0.1) and
  for alpha >= 1 at 0.03 (at 0.1 and alpha = 1 a 3/3 tie), against alpha >= 10 at balance (a factor ~100 in alpha). A heavy root
  (lambda >= sqrt(chi)) is lazy from alpha = 10 (3/6-4/6; 6/6 at 100), as at balance: it neither makes
  training lazier nor prevents laziness. S2S majority up to alpha = 1e-4 for 0.1 <= lambda / sqrt(chi)
  <= 10, up to 1e-5 at 0.01, 0.03 and 32, only at 1e-6 at 100. *Withdrawn in fix round 1:* "the S2S
  region is largest near lambda = sqrt(chi)" (one cell, alpha = 1e-3, with 3/6 S2S seeds).
  **Empirical.** So imbalance is a second knob in the sense of Kunin et al. 2024 on the light-root side;
  the heavy-root side at fixed alpha = 3.375 (P20 point) is rich, consistent with R13's degree-(K - 1)
  model, but at alpha >= 10 it is lazy like the balanced init.
- **chi = 3 (deficient).** Same S2S/rich structure at small alpha; stalled at the centred floor for
  alpha >= 1 (all lambda), with delta_K < 0.1 (lazy residual) at alpha = 100 for lambda >= sqrt(chi).
- The fan-in default (lambda = 1) is root-light by sqrt(chi) (E Q_v = sigma^2 (1 - 1/chi) on the root
  edges): at chi = 24 it sits at lambda / sqrt(chi) = 0.2, on the light side of balance.

### 2.5 Large bond dimension: SP, NTK and muP-TTN (binary N = 4)

`configs/large_chi_b4.yaml`: chi = 4..128 (uncapped), parameterizations of theory_regimes section 4.3
(SP: fan-in, eta = 1; NTK: eta_v = 1/F_v; muP-TTN: root init x (2 chi)^-1/2, eta_v = chi_v / F_v),
centred and plain, sigma = 1.0 and 1.5, 6 seeds; `results/large_chi_large_chi_b4_*.png`,
`large_chi_exponents_*.csv`, `results/fix1/large_chi_slopes_tensor.csv` and `large_chi_levels.csv`.
Slopes of log(quantity) vs log chi over chi = 16..128 (4 values, 0.9 decade; 24 runs each; error from
the residual scatter). sigma = 1.5 is the theory's check point (x = 1.11). Fix round 1: delta_K with
the tensor-space kernel of the theory (the 128-probe kernel gives 0.68-0.83 of it at chi >= 32 as per-(lr, chi) medians; single runs span 0.50-1.05).

| centred, sigma = 1.5 | delta_K (tensor) | delta_K (probe) | delta_theta | root (rel. Frobenius change) | leaf cores |
|---|---|---|---|---|---|
| SP, measured | -3.02 +- 0.06 | -3.09 +- 0.09 | -1.63 +- 0.04 | -1.02 +- 0.03 | -2.96 +- 0.07 |
| SP, theory P16 | -3 | | -3/2 | -1 | -5/2 |
| NTK, measured | -0.79 +- 0.04 (64-128: -1.01) | -0.84 +- 0.06 | -0.52 +- 0.03 | -0.93 +- 0.02 | -0.53 +- 0.03 |
| NTK, theory | -1 | | -1/2 | -1 | -1/2 |
| muP-TTN, measured | +0.085 +- 0.024 | +0.03 +- 0.06 | -0.06 +- 0.02 | -0.50 +- 0.01 | -0.07 +- 0.02 |
| muP-TTN, theory | 0 | | 0 | -1/2 (Frobenius; spectral O(1)) | 0 |

- **Feature learning survives under muP-TTN**: delta_K (tensor) = 5.1, 5.7, 6.1, 6.2 at chi = 16, 32, 64,
  128, approaching P16's 6.2-6.3 (local slope +0.04 between 64 and 128; the old "delta_K ~ 4-5" was the
  probe kernel). It is lost under SP (delta_K ~ chi^-3: the root dominates the kernel, the model becomes
  linear in the root) and NTK (levels 0.250 and 0.124 at chi = 64, 128, R4's 0.240 and 0.120; the
  chi = 16-128 fit, -0.79, is pulled by the pre-asymptotic chi = 16-32 where delta_K = 0.4-0.6; in R12
  both orders scale as chi^-1, so the old "approaching -1 as c_2 x^2 fades" is withdrawn). **Empirical**,
  confirming theory P16 for delta_K, delta_theta and the root; the SP leaf-core exponent is steeper than
  predicted at sigma = 1.5 (-2.96 vs -5/2, 6.5 SE) and closer at sigma = 1.0 (-2.58 +- 0.07); SP
  delta_theta is 3.3 SE from -3/2.
- At sigma = 1.0 (x = 3.74, further from the lazy limit) the slopes are shallower (tensor kernel: SP
  -2.58 +- 0.06, NTK -0.62 +- 0.02, muP-TTN +0.089 +- 0.024): the exponents are asymptotic in x -> 0.
- Plain and centred models give the same exponents within errors at chi >= 16 (tensor kernel, sigma =
  1.5: SP -3.06 +- 0.07, NTK -0.83 +- 0.04, muP-TTN +0.095 +- 0.024 plain): with full,
  overparameterized bonds centring does not change the scaling (literature correction 1).
- Scope: binary N = 4 only, four chi values. K = 7 (binary N = 8, d = 2, plain, sigma = 1, chi = 4..32;
  `large_chi_b8`): incomplete, not interpreted.

### 2.6 sigma x chi at K = 7 (binary N = 8)

Two maps (6 seeds; figures `phase_sigma_chi_b8.png`, `phase_sigma_chi_b8_d2.png`; per-seed tables
`boundaries_per_seed_sigma_chi_b8*.csv`). 13 of 240 and 15 of 390 runs were not finished at the cut-off
(listed in the report); the statements below use the finished ones.

**d = 3, teacher [3, 2, 1], deficient bonds (cut dimension of the top bonds 81), chi = 3, 6, 24:**
- No lazy regime: every centred run above sigma ~ 0.9-1.3 stalls at the floor of the unrealizable
  centred target (theory P4: sigma_floor ~ 0.75), and the plain model at chi = 24 stays rich up to
  sigma = 1.5. **Empirical.**
- S2S boundary (per-seed median, bracket = [last S2S, next sigma]): centred chi = 24: 0.54 [0.45,
  0.65], rich at 0.65-0.93; plain chi = 24: 0.52; plain chi = 6: 0.93 [0.79, 1.09]; centred chi = 6:
  S2S up to 0.43, then (missing points) floor-stalled from 0.91. *Fix round 1:* at chi = 24 the S2S
  block is a single run (sigma = 0.45, the lowest grid point) for 5 of 6 seeds (centred and plain), so
  0.54 is the bracket [0.45, 0.65], i.e. sigma_c <= 0.65, not a measured crossing.
- chi = 3 (= teacher rank): stalls dominate already at sigma = 0.3 (centred 4/6, plain 4/6 stalled),
  as in the stall test (2.2).

**d = 2, teacher [3, 2] (cut dimensions 4 and 16), centred, chi = 2, 4, 8, 16 (full, square) and
8/32 (full, 2x):**
- S2S boundary 0.83 [0.72, 0.97] for chi = 4, 8, 16 and 0.53 [0.46, 0.61] for the 2x-overparameterized
  full bonds (more channels move the boundary down, as at N = 4). Theory P15 (plain model, chi = 16,
  12 seeds) gives a median 0.71.
- **Lazy regime at K = 7 exists only with overparameterized full bonds:** 8/32: lazy from sigma ~ 1.4
  (2/5 seeds at 1.42, majority from 2.5), kernel boundary (delta_K = 0.1 crossing) 1.45 (median of
  seeds); the remaining large-sigma runs are "stalled" because the lazy fit had not finished within the
  step budget (delta_K < 0.01, loss still falling). Square full bonds (chi = 16): every run above
  sigma = 1.7 stops at the step budget with delta_K = 1e-6..0.17 and l = 0.05-0.3 still decreasing - the
  ill-conditioned lazy fit of theory P9 (lambda_min / lambda_bar ~ 1e-4 at square cores), not a
  floor. Deficient bonds (chi = 2, 4, 8): floor-stalled from sigma ~ 1.07. **Empirical.**

### 2.x Two-copy cross-check (binary N = 4)

*Interim data (scipy RK45), 6 seeds x 8 chi x 19 sigma, except where marked "library" (fix round 1:
the library numbers come from the committed CSVs of the `sigma_chi_b4` library sweep).*

- **Lazy side agrees in mechanism, not in position.** Same lazy residual as the centred model at
  deficient bonds (chi = 3: 0.49 / 0.25 for seeds 0 / 1 in both models); the kernel change is second
  order, delta_K ~ sigma^-2K: ratio 1362 between sigma = 3 and 10 at chi = 24 (predicted (10/3)^6 =
  1372; centred: 40.6 vs (10/3)^3 = 37). Hence the two-copy lazy boundary is lower: library, probe
  kernel, 0.82 vs 1.04 at chi = 36 (geometric means of 6 per-seed boundaries; the interim 0.80 vs 1.13
  is superseded). **Empirical**, confirming theory_regimes R5.
- **S2S side differs (new, empirical).** At small sigma the two copies start identical (theta+ =
  theta-), so their near-origin escapes happen together: each copy learns *half* of components 1 and 2
  at the same time (binary N = 4, chi = 36, sigma = 0.01, seed 0: a+ = (0.502, 0.498, 0), a- = (-0.498,
  -0.502, 0) at t = 32; `pilots/twocopy_mechanism.py`). Events 1 and 2 then merge: t_2 - t_1 stays
  ~2 for sigma from 0.01 to 0.04 (instead of growing like 1/sigma) and rho_1 ~ 2 (interim runs). In the
  library sweep (fix round 1, committed CSVs) the two smallest sigma (0.010, 0.016) are labelled S2S in
  5, 4, 5 seeds at sigma = 0.010 and 4, 3, 2 seeds at 0.016 (chi = 18, 24, 36; majority-label boundary 0.018, 0.016, 0.014; the
  interim "rich at every sigma" is superseded), but min_i rho_i stays small there (largest value at sigma <= 0.016: 7.1, 14.4, 4.8 at chi = 18, 24, 36; erratum E4) (the rho = 5
  crossing exists for 2, 3, 0 of 6 seeds at chi = 18, 24, 36), so the scaling check (iii) of
  `notes/diagnostics.md` fails. So the two-copy model is a faithful cross-check of the lazy boundary (up
  to the predicted shift) but **not of the S2S boundary at large chi**: its incremental structure is
  changed by the mirror copy. (Two-copy labels use the pre-fix event rule; not rerun.)
- **Deficient bonds, large sigma: lazy, then rich (library runs, chi = 3, 6 seeds).** The two-copy model
  is realizable at chi = 3 (bond 2 chi = 6 >= teacher rank), so instead of stalling at the floor it
  escapes from the lazy residual (theory_regimes R7 / P8): converged ("rich", delta_K = 0.3-1.3) at
  sigma = 1.6, 2.5 and 4.0 (at 4.0 in 1/6 runs before the time budget, the others at l < 1e-3 at the
  budget end), partially at sigma = 6.3 (l = 0.01-0.04 at the budget) and not at sigma = 10 (plateau at
  l = 0.44, the lazy residual; delta_K = 0.05). The escape is pushed beyond any fixed budget as sigma
  grows (tau_esc ~ sigma^6 in kernel time, P8). **Empirical.**

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
- **Width resolution check** (`analysis.add_resolution_flags`): the refined window spans two log-grid
  points, so its resolution is ~0.06 t / 200. Every N = 4 run has all widths resolved (>= 5 samples per
  width; 2204/2204). At K >= 7 deep in the S2S region (events at t ~ 1e5-1e7) the widths are not
  resolved (45% of the full-bond K = 7, 8 runs), so their rho values are lower bounds (e.g. rho ~ 5e3
  where the aligned theory gives ~1e6); no label depends on them (0 runs 'undetermined': every
  unresolved run has rho > 3 already). No rho magnitudes or rho exponents are reported for K >= 7.

## 3. Fix round 1 (local, 2026-10-10)

Review: `notes/reviews/regime_map_round1.md`. Done locally by agent `regime-map-fix`. The remote raw
outputs were copied to `data/regime_map/` of the main checkout during the round (11721 files, 1.24 GB,
verified); every number below comes from them through the fixed analysis, unless marked "local reruns".
Before the copy arrived, 1,323 runs had been regenerated locally with the fixed driver from the
committed configs (`fix_round1.py run`: the N = 4 S2S side and lazy windows, large chi at sigma = 1.5,
the K-sweep runs above the unresolved brackets; `data/regime_map_local/`); they now serve as a
cross-check. Code: `experiments/regime_map/fix_round1.py` (`tensor`, `analyse`), `make_figures.py
--raw`; tables and fix-round figures in `experiments/regime_map/results/fix1/`; every standard figure
and CSV in `results/` regenerated from the copied raw outputs with the fixed code.

**Checks.** (i) The pre-fix rule applied to the copied rows reproduces the committed per-seed CSVs to
<= 5e-15 (chi_b4: sigma_c, prediction, kernel boundary, rho slope, status of all 54 seeds;
sigma_chi_b4: the rho = 2/3/5 boundaries). (ii) Local reruns vs copied runs (1,323 runs in common):
labels identical in all; min_i rho_i agree to 1e-6 (median relative difference; larger differences
only in lazy runs whose events were not refined locally), probe delta_K to 2e-4, tensor-space
delta_K to 7e-7 (`validation_local_vs_remote.csv`).

### 3.1 Code changes

- Event time (`run_regime.final_rise`, `refine_events`): the **last** upward crossing of 1/2 before
  the component first reaches 0.9, i.e. inside the refined rise window (`notes/diagnostics.md`).
  Before, a transient above 1/2 that fell back before the component was learned was kept as the event
  whenever the coarse log grid caught it, so the label depended on log-grid sampling. (A first version
  of the fix took the last crossing over the whole run; that moved the event of a component that was
  learned, dipped below 1/2 while another component escaped and recovered, and turned 6 K = 7 runs
  from S2S to rich, e.g. binary 8, seed 2, sigma = 0.16; corrected before any number below.) Rows carry
  `n_up` (upward crossings of 1/2 before the first 0.9; > 1 = transient), the pre-fix
  `event_times_old`, `rhos_old`, `label_old`. `analysis.relabel_last_crossing` relabels pre-fix rows
  from their recorded log + fine rows (used for the copied raw outputs).
- Tensor-space kernel `dK_tensor` (J D J^T, D = learning rates x mask; the kernel of
  `notes/theory_regimes.md`): computed in-run for D x P <= 2e7 and, for the copied runs, from theta_end
  (`fix_round1.py tensor`, cache `data/regime_map_local/tensor_kernel/`). Implemented with D batched
  reverse-mode passes (`run_regime.tensor_kernel`, identical to `ttn.diagnostics.tensor_tangent_kernel`
  to 2e-16 on test cases) because the library's forward construction allocates (core size)^2 per core
  (34 GB for the two-copy root at chi = 128). The label rule keeps the 128-probe kernel `dK`.
- K sweep: `sigma_c_vs_K.csv` counted only the exact status 'unresolved' (missing
  'unresolved+stall_below') and `summary.png` printed the number of finite sigma_c as "resolved"; both
  now count status prefixes and draw unresolved boundaries as brackets. Floor-stalled runs are
  classified by min_i rho_i (`fix_round1.k_sweep_analysis`; section 3.2 item 4).
- `plots.fig_imbalance`: alpha was rounded to 3 digits (3.375 -> 3.38), so the P20 panel was empty;
  the panel now also shows the tensor-space kernel. Invalid mathtext (`\sqrt\chi`) fixed.
- Spawned worker pools (fork does not exist on Windows) with the raw root from `--out` / `--raw` /
  `$REGIME_MAP_OUT`; `make_figures.py --raw`; `pilots/lazy_pred_b8.py` takes an output directory and no
  longer overwrites the committed CSV (one thread per process); README step 2 documents the
  output-directory argument; run ids ignore keys that are missing for a row (`analysis.rid_of`).
- Speed option `refine` (local reruns only): the large-chi reruns and the lazy-window runs that end
  lazy skip the event refinement; no number uses their events.

### 3.2 Claims: old -> new

Numbering as in the review's fix list; status tags as in the project.

**Relabel (review fix 2; all copied runs, `results/fix1/relabel_runs.csv` for N = 4).** 116 of the 9,094 runs (erratum E5; 14 more have repeated crossings only in a component that is never learned, all stalled)
have a transient above 1/2 before the component is learned; 15 labels change, all rich -> S2S: chi_b4 3
(chi = 12 seed 4 at sigma = 0.015-0.029), sigma_chi_b4 8 (the same chi = 12 seed 4 at 0.010-0.025 and
chi = 4 seed 3 at 0.040, centred and plain), sigma_K_d2 1 (binary 4, chi = 2, seed 4, sigma = 0.24),
imbalance_b4 3 (chi = 3 map, alpha = 1e-4 to 1e-3). Per-seed boundaries that move: chi_b4 chi = 12
seed 4 0.0126 -> 0.0339 [0.029, 0.040] (it was the 5.9x outlier against its deepest-run extrapolation
0.075, now 2.2x); sigma_chi_b4 (rho = 3 crossing) chi = 4 seed 3 0.029 -> 0.055 and chi = 12 seed 4
NaN -> 0.032 (it had been dropped silently, so table 2.1 at chi = 12 averaged 5 seeds: 0.088 x/ 1.33,
now 0.075 x/ 1.33 over 6; chi = 4: 0.089 x/ 1.58 -> 0.099 x/ 1.51); sigma_K_d2 binary 4, chi = 2, seed
4: 0.20 -> 0.29. The majority-label tables (`boundaries_sigma_chi_b4_*.csv`) and every K >= 4 per-seed
table are unchanged. As the review found, the effect is rare and confined to a few seeds.

1. **Claim 1 (binary N = 4: "every boundary is predicted quantitatively")** -> narrowed and reworded.
   - Old: "the S2S boundary of each seed is predicted from that seed's deepest run alone (0.97-1.08x for
     chi >= 6)". The aligned-mode extrapolation calibrates one amplitude per component on the deepest
     run of the same seed, so it tests rho ~ 1/sigma along that seed's sweep; it is not an a-priori
     prediction, and the chi-dependence of the boundary is not derived. 0.97-1.08 were means over
     seeds. New (last-crossing events): extrapolation / measured = 0.86-1.23 per seed for 53 of the 54
     chi_b4 seeds (label rule, grid factor 1.39) and 2.22 for chi = 12, seed 4; on sigma_chi_b4 (rho = 3
     crossing) 0.99-1.12 per seed for chi >= 6 apart from chi = 12 seed 4 (2.37), 1.02-1.53 at chi = 3,
     and 1.00-1.08 at chi = 4 apart from seed 3 (2.24). **Empirical** (a consistency check of R9's
     rho ~ sigma^-(K-2), K = 3).
   - Per-seed rho exponent on the S2S side: -1.015 to -1.031 for 53 of 54 chi_b4 seeds and -1.02 to
     -1.05 on sigma_chi_b4, except four seeds with transients above 1/2 (chi = 3 seeds 1 and 5, chi = 4
     seed 3, chi = 12 seed 4: -1.09 to -1.27), which approach rho ~ 1/sigma slowly, as R9 says for mixed
     first escapes. Predicted -(K - 2) = -1 (R9: derived for the first gap, heuristic for later gaps).
     **Empirical.**
   - pred / measured at chi = 3, 4 (1.21 +- 0.08 and 1.25 +- 0.20, mean +- SE over 6 seeds) is pulled up
     mainly by those transient seeds (a steeper rho exponent makes the 1/sigma extrapolation overshoot);
     the other seeds give 1.02-1.16 and 1.00-1.08. The old explanation by the realizability floor is
     withdrawn: the floor-stall onset (all seeds stalled from sigma = 1.0, 1.0, 1.59 at chi = 3, 4, 6;
     first stalled seed at 0.63 for chi = 3) lies 4-10x above these S2S boundaries (2.7x for that first
     seed), and is itself 1.6-2.2x above P4's small-sigma estimate 0.56, 0.60, 0.71. **Empirical.**
   - Lazy boundary: old "follows P5 to 4-8%" with the probe kernel, and "5-8% low, as in the theory
     note's GF checks" (those used the tensor kernel). New: with the tensor-space kernel the per-seed
     medians are 1.13, 0.94, 0.82, 0.66, 0.571 at chi = 32, 48, 64, 96, 128, inside P5's bracket
     [1.10, 1.20], [0.92, 0.96], [0.80, 0.82], [0.66, 0.67] and 0.07% below the lower edge of
     [0.5712, 0.5773] at chi = 128 (sigma_lazy sqrt(chi) = 6.40-6.56 against the asymptote 6.50); also
     inside at chi = 24; at chi = 9-16, where the isotropic constants are not expected to hold, 2-6%
     above the bracket. The probe kernel gives 0.54-1.11 (median 0.78) of
     the tensor-space delta_K near the boundary (chi >= 32), hence its 5-6% lower boundary.
     **Empirical**, confirms **derived** P5 (N = 4, d = 3, SP, 6 seeds).
2. **Claim 2 (collapse in sigma sqrt(chi), "new")** -> scope, statistics and provenance corrected.
   - The quoted "sigma_c sqrt(chi) = 0.29, 0.29, 0.31, 0.30, 0.27 at chi = 32..128" were medians of the
     deepest-run predictions; measured medians are 0.27, 0.33, 0.27, 0.33, 0.28 (quantized by the
     sigma grid; last-crossing events, unchanged at chi >= 16).
   - S2S slope vs chi (per-seed label boundaries, pooled fit, SE from the scatter): -0.528 +- 0.053 for
     chi 16-128 (42 values, 0.9 decade; per-seed fits -0.528 +- 0.073), -0.548 +- 0.078 for chi 32-128
     (0.6 decade), -0.552 +- 0.049 for chi 9-128 (1.15 decades; was -0.528 +- 0.061 before the
     relabel). -1/3 (per-path amplitude x number of paths) differs by 2.7 SE (chi >= 16) and 2.9 SE (chi >= 32) in the per-seed fits (erratum E3; the pooled fit, which treats seeds sharing teachers across chi as independent, gives 3.7 SE and 2.8 SE
     (chi >= 32), not "4.2 SE at chi >= 32"; -1/2 is within 0.6 SE.
   - Lazy slope: probe kernel -0.543 +- 0.015 (chi 16-128; per-seed fits -0.543 +- 0.012, 3.6 SE from
     -1/2) and -0.522 +- 0.020 (chi 32-128); tensor-space kernel -0.553 +- 0.013 (chi 16-128) and
     -0.504 +- 0.011 (chi 32-128, 0.6 decade), i.e. -1/2 once chi = 16-24 (pre-asymptotic) are left out.
     Two-copy (tensor) -0.499 +- 0.014 (chi 32-128).
   - Rich window (lazy / S2S boundary per seed, probe kernel): median 18-23 at chi >= 32.
   - Status: **empirical** for binary N = 4 (a 3-factor matrix product), fan-in init, chi = 16-128 (seven
     chi values, five for chi >= 32; < 1 decade). Not new as a scaling idea: sigma^2 chi is the TTN analogue of the
     width-dependent scales sigma^2 w of Tu, Aranguri & Jacot (NeurIPS 2024) and alpha sqrt(h) of Geiger
     et al. (J. Stat. Mech. 2020) (`notes/literature.md`, H4 correction 2); what is measured here is
     that both TTN boundaries follow it at N = 4.
3. **Claim 3 (no lazy regime below the cut dimension; runs stall at the Eckart-Young floor)**:
   unchanged (**derived** R7 + **empirical**, N = 4). Added: a run that ends on the floor counts as
   converged when the floor is below 1e-3 (chi = 6, sigma = 1.0: floors 1.6e-4 to 8.8e-4, labelled
   rich).
4. **Claim 4 (sigma_c -> sqrt(d), "H4 refined", "derived from R9")** -> **conjecture**; the K >= 8
   label values were brackets.
   - Counter bug: `sigma_c_vs_K.csv` counted only the status 'unresolved' and `summary.png` printed the
     number of finite sigma_c as "resolved". True counts (centred, chi = 4): unresolved 2, 5, 5, 6, 5 of 6
     seeds at K = 6, 8, 10, 12, 15 (CSV: 0, 0, 2, 0, 0); K = 15 has one more 'all_s2s' seed. Both files
     are regenerated with the fixed counter; brackets are drawn as brackets.
   - With the label rule the K >= 8 values were brackets [last S2S run, first floor-stalled run]:
     K = 8 "1.01" = [0.88, 1.16] (5/6 seeds), 10 "0.90" = [0.80, 1.00], 12 "0.92" = [0.83, 1.03],
     15 [0.63, 1.36]; lower bounds of the S2S block, not crossings. K = 6 is 4/6 resolved (median 0.68
     over them; the old 0.80 mixed in two brackets).
   - Floor-stalled runs classified by min_i rho_i (`results/fix1/k_sweep_per_seed.csv`,
     `fix1_k_sweep.png`; walk upward from each unresolved seed's last S2S run: rho > 3 extends the
     S2S-like block, rho <= 3 ends it, a run without a ratio ends it if it learned nothing or stayed
     kernel-like (delta_K < 0.1), and is skipped if it learned one component; brackets wider than one
     grid step are not counted as resolved). The first floor-stalled run above the label bracket is
     S2S-like in 13 of the 18 unresolved MPS seeds (K = 6-12; rho = 4-642, L/L_0 = 0.001-0.074).
     Resolved seeds and medians: K = 6: 6/6, 0.80 (the two reclassified seeds at [1.47, 2.0] and
     [1.08, 1.47]); K = 8: 5/6, 1.01 (seed 1: S2S-like at 1.16, open); K = 10: 4/6, "1.16" (two seeds at
     [0.80, 1.00], two at [1.26, 1.59]; two brackets [1.00, 1.59] too wide; erratum E1: only the two 0.90 seeds end at a rho crossing); K = 12: 5/6, "1.15" (erratum E1: all five end at no-learning stalls, not crossings; four at
     [1.03, 1.29], one at [1.29, 1.60]); K = 15: 0/6 (the points 0.76 and 0.92 were not finished at the
     cut-off and the next runs learned nothing, L/L_0 = 1, so the brackets span 3-5 grid steps).
   - Caveat (why this does not settle H4): at chi = 4 the bonds of MPS 6-12 are deficient, the centred
     target is unrealizable and these runs end on its floor. Their later event comes at t = 8-1000 and in
     9 of the 13 S2S-like runs the smaller component is learned first, so the separation may come from
     the capacity conflict with W_0 rather than from a small-init saddle. **Empirical**; the K >= 8
     boundary of a realizable problem (full bonds, or the plain model with spare channels) is still
     unmeasured.
   - Laws (`results/fix1/k_sweep_laws.csv`; per-K medians of resolved seeds, bootstrap errors over
     seeds, crude with 4-6 seeds): on K = 4-7 sigma_c^K = const has chi^2 = 0.79 and
     sigma_c^K / sqrt(D) = const 0.59 (3 dof; spreads 4.6x and 3.7x): indistinguishable. With the
     rho-classified K = 8-12 [withdrawn, erratum E2: stall-terminated brackets; with crossing-terminated brackets only the laws are indistinguishable on K = 4-10, chi^2 7.7 vs 6.9, 5 dof]: chi^2 = 22.5 (law A) and 6.3 (law B) for 6 dof, spreads 98x and 12x:
     sigma^K = const fails, sigma^K / sqrt(D) is acceptable but its constant drifts from ~0.01 (K <= 7)
     to 0.07-0.10 (K = 8-12). The old "within ~20% in sigma" test was weak (sigma_c = u^(1/K): 20% in
     sigma is a factor 1.2^K = 2-9 in u).
   - Status: "sigma_c -> sqrt(d), not 1" was tagged "derived from R9"; R9's effective-amplitude law is
     heuristic, only d = 2 was run, and the K >= 8 values rest on floor-stalled runs at deficient bonds:
     **conjecture**. [Withdrawn, erratum E1: "the measured sigma_c exceeds 1 at K = 10-12 (medians 1.15-1.16)"; those brackets end at no-learning stalls.] In this
     deficient-bond setting, below sqrt(2) = 1.41. A test needs full bonds or spare channels with the
     plain model, longer budgets at K >= 8, and d = 3.
5. **Claim 5 (large chi: delta_K ~ chi^-3.09 (SP), chi^-0.84 (NTK, "approaching -1"), chi^+0.03
   (muP-TTN), "as P16 predicts")** -> tensor-space kernel; scope stated.
   - Tensor-space kernel, centred, sigma = 1.5, chi = 16-128 (4 values, 0.9 decade, 24 runs per fit):
     slopes SP -3.02 +- 0.06 (P16: -3), NTK -0.79 +- 0.04, muP-TTN +0.085 +- 0.024. The NTK fit is
     pulled by chi = 16-32, where delta_K = 0.4-0.6 (pre-asymptotic): between chi = 64 and 128 the local
     slope is -1.01 and the levels 0.250 and 0.124 match R4's 0.240 and 0.120. muP-TTN levels 5.1, 5.7,
     6.1, 6.2 at chi = 16, 32, 64, 128 (P16: -> 6.2-6.3; local slope +0.04 between 64 and 128); the old
     "delta_K ~ 4-5" was the probe kernel, which gives 0.68-0.83 of the tensor value at chi >= 32 (per-(lr, chi) medians; single runs 0.50-1.05).
     **Empirical**, confirms P16 for SP, for the muP-TTN level and for NTK at chi >= 64.
   - Withdrawn: "NTK slope approaching -1 as the second-order term c_2 x^2 fades": in R12 both orders
     scale as chi^-1; the shallow fit is the pre-asymptotic chi = 16-32.
   - Unchanged (kernel-independent): SP delta_theta -1.63 +- 0.04 (3.3 SE from -3/2), leaf cores -2.96 +-
     0.07 (6.5 SE from -5/2), root -1.02 +- 0.03; muP-TTN delta_theta -0.064 +- 0.018. At sigma = 1.0
     (x = 3.74) the tensor-kernel slopes are shallower (SP -2.58 +- 0.06, NTK -0.62 +- 0.02, muP-TTN
     +0.089 +- 0.024): the exponents are asymptotic in x -> 0. Plain model at sigma = 1.5: SP -3.06 +-
     0.07, NTK -0.83 +- 0.04, muP-TTN +0.095 +- 0.024 (same within errors). Binary N = 4 only.
6. **Claim 6 (imbalance).** Old: "delta_K ~ lambda^2 (local slope 2.18 -> 1.27; theory 1.8-1.9), saturating
   at 0.50-0.53 (theory 0.6-0.7, ~20% below); heavy root keeps delta_K ~ 0.5 (not lazy); S2S widest near
   lambda = sqrt(chi)". The quoted delta_K values did not come from committed code (the P20 panel of
   `phase_imbalance_b4.png` was empty, alpha rounding bug); they are those of the copied raw runs (54 runs,
   chi = 16, alpha = 3.375, 6 seeds; the reviewers' reruns of the committed code give the same values),
   which `results/fix1/imbalance_P20.csv` and the regenerated figure now show with both kernels. New:
   - delta_K ~ lambda^2 as lambda -> 0: per-seed local slopes (mean +- SE, 6 seeds) 1.97 +- 0.01,
     1.85 +- 0.06, 1.55 +- 0.11, 1.22 +- 0.09 between lambda = 0.01, 0.03, 0.1, 0.3, 1 (tensor kernel;
     probe kernel 1.98 +- 0.01, 1.86 +- 0.06, 1.57 +- 0.10, 1.25 +- 0.08). The first slope of the old
     text, 2.18, was the log-ratio without the division by ln 3; it is 1.98 from the medians. P20's
     1.8-1.9 between 0.03 and 0.1: 1.85 +- 0.06. **Empirical**, confirms P20.
   - Saturation for lambda >= 10: median delta_K = 0.61, 0.60, 0.59 at lambda = 10, 30, 100 with the
     tensor kernel (range over seeds 0.51-0.75), at the lower edge of P20's 0.6-0.7: **confirmed**. The "20% below the
     prediction" of the old text (0.53, 0.50, 0.50) is the probe kernel. **Empirical.**
   - Labels at the check point: lazy 6/6 for lambda <= 0.3, lambda = 1: 5 rich / 1 lazy, rich 6/6 for
     lambda >= 3.
   - Map at chi = 24 (`phase_imbalance_b4.png`, regenerated; the relabel changes three chi = 3 runs and no
     chi = 24 run; majority labels, one-decade alpha grid): a
     light root lowers the lazy boundary (lazy from alpha = 0.1 at lambda / sqrt(chi) = 0.01, 4/6 seeds,
     and from alpha = 1 at 0.03; at 0.1 and alpha = 1 a tie, 3 lazy / 3 rich, erratum E6), i.e. by up to ~100x in alpha against alpha = 10 at balance. A heavy
     root does **not** make training lazier, and it does not prevent laziness either: for
     lambda / sqrt(chi) >= 1 the runs are lazy from alpha = 10 (majority 3/6-4/6; 6/6 at 100) as at
     balance; at the P20 point (alpha = 3.375) a heavy root is rich (delta_K ~ 0.6). **Empirical.**
   - Withdrawn: "the S2S region is largest near lambda = sqrt(chi)". It rested on one cell (alpha = 1e-3,
     lambda / sqrt(chi) = 1) with 3/6 S2S seeds on a grid with one-decade steps in alpha. What the map
     supports: S2S majority up to alpha = 1e-4 for 0.1 <= lambda / sqrt(chi) <= 10, up to 1e-5 at 0.01,
     0.03 and 32, and only at 1e-6 at 100 (several cells 5/6 or 3/6). **Empirical**, coarse.
   - Local slopes now come with per-seed errors (above); the map numbers are majority labels of 6 seeds.

7. **Two-copy cross-check (section 2.x)**: the bullets quoted the superseded interim scipy runs. From
   the library CSVs (regenerated, unchanged by the relabel; `sensitivity_per_seed_sigma_chi_b4_twocopy.csv`,
   `boundaries_sigma_chi_b4_twocopy.csv`; `results/fix1/csv_checks_sigma_chi_b4.csv`):
   - Old: "labelled rich at every sigma in the grid for chi >= 18 (S2S boundary 0.015 vs 0.075 centred
     at chi = 24)". New: the two-copy runs are labelled S2S at the two smallest sigma (0.010, 0.016) in
     5, 4, 5 seeds at sigma = 0.010 and 4, 3, 2 seeds at 0.016 (chi = 18, 24, 36; majority-label boundary 0.018, 0.016, 0.014), but
     min_i rho_i stays small there (largest 7.1, 14.4, 4.8 at chi = 18, 24, 36; erratum E4) (rho = 5 crossing found for 2, 3, 0 of 6 seeds at chi = 18, 24, 36),
     so the sweep-level scaling check (iii) of `notes/diagnostics.md` fails: these are not S2S runs in the
     sense of the diagnostics, and events 1-2 merge as described. **Empirical** (the relabel changes no
     two-copy label: one run has a transient above 1/2).
   - Old: "two-copy lazy boundary lower (0.80 vs 1.13 at chi = 36)" (scipy). New: 0.82 (two-copy) vs
     1.04 (centred), geometric means of the per-seed probe-kernel boundaries at chi = 36 (6 seeds each;
     ratio 0.79, P6: 2^(-1/3) = 0.79). **Empirical.**
8. **Numbers that did not match the committed CSVs** (corrected in place in sections 0-2):
   imbalance slope 2.18 -> 1.98; S2S slope vs chi (chi 16-128) -0.524 +- 0.045 -> -0.528 +- 0.053 (pre-fix
   labels, unchanged by the relabel; item 2); sigma_chi_b4 kernel boundary at chi = 9: 3.01 -> 3.06
   (geometric mean, x/ 1.07, 6 seeds); kernel-threshold ratio (delta_K 0.05 / 0.2) at chi = 9: 1.89 ->
   1.97 +- 0.08; plain-model K = 3 boundary (section 2.3): 0.20 -> 0.17 (`sigma_c_vs_K.csv`, plain,
   chi = 4, binary 4: median 0.170).
9. **Smaller items.**
   - K = 7, d = 3, centred chi = 24 (section 2.6): "S2S boundary 0.54 [0.45, 0.65]" rests on a single S2S
     run per seed (the lowest grid point, sigma = 0.45) for 5 of 6 seeds; it is the bracket
     [0.45, 0.65], i.e. sigma_c <= 0.65, not a measured crossing.
   - A run that ends on the floor of an unrealizable centred target counts as converged when the floor
     is below 1e-3 (chi = 6, sigma = 1.0: floors 1.6e-4 to 8.8e-4, labelled rich; review A).
   - "Stalled" also contains runs cut off at max_steps before the time budget (MPS 8, sigma = 0.1:
     2e4 steps at t = 1.4e9 < T = 5.7e9; the second route at 1.6e-11). Listed as a third stall cause next
     to floors and long or dead routes.
   - Section 1.3: the large-K limit needs C and kappa to grow sub-exponentially in K (log C = o(K)), not
     "at most exponentially": an exponentially growing C (which section 2.3's sqrt(d) conjecture would
     need) gives a limit different from 1, so the old wording contradicted section 2.3.
   - Dead-route singular values of 1e-16 to 1e-18 (section 2.2) are below the SVD's resolution
     (~1e-16 relative): read them as "at roundoff", not as measured values.


### 3.3 Pending and open decisions

- **K >= 8 S2S boundary of a realizable problem** (claim 4): needs new runs: full bonds, or the plain
  model with spare channels, longer budgets at K >= 8 (MPS 8 seed 1 is open above sigma = 1.16; binary
  16 lacks the points 0.76 and 0.92, unfinished at the remote cut-off), and d = 3 to separate sqrt(d)
  from 1. Not run here (minutes to hours per run at K >= 10).
- **Kernel of the label rule.** Labels use the 128-probe kernel (`notes/diagnostics.md`); the theory's
  constants refer to the tensor-space kernel, which is ~1.3x larger (median) near the lazy boundary at
  chi >= 32 and moves every lazy boundary up by ~5% (table 2.1: 3.40, 2.37, 1.75, 1.40, 1.09 at chi = 9,
  12, 18, 24, 36 against 3.06, 2.20, 1.66, 1.31, 1.04 with the probe kernel). Decision for the lead:
  keep the probe kernel for labels (current) or switch to the tensor kernel for N <= 8.
- Not re-analysed: the interim scipy sweep (`b4_sigma_chi`, superseded), `large_chi_b8` (K = 7, not
  interpreted), the stall test (no label changes).

### 3.4 Errata from the verification of fix round 1 (`notes/reviews/regime_map_round2_verification.md`)

Applied in place in sections 0-3, marked "erratum E1-E6":
- **E1.** The K = 10 / 12 medians (1.16, 1.15) are not S2S -> rich crossings. At K = 12 all 5 brackets
  end at no-learning stalls; at K = 10, 2 seeds end at a crossing (0.90). "sigma_c > 1 at K = 10-12" is
  withdrawn.
- **E2.** "sigma^K = const fails (chi^2 22.5)" is withdrawn. With crossing-terminated brackets only, the
  two laws are indistinguishable on K = 4-10 (chi^2 7.7 vs 6.9, 5 dof).
- **E3.** sigma^K chi^(K-1) is disfavoured at 2.7-2.9 SE in the per-seed fits (3.7 SE pooled).
- **E4.** Two-copy runs at small sigma:
  - the largest min_i rho_i at sigma <= 0.016 is 7.1, 14.4 and 4.8 at chi = 18, 24, 36;
  - S2S-labelled seeds are 5, 4, 5 at sigma = 0.010 and 4, 3, 2 at 0.016.
- **E5.** 116 runs have a transient above 1/2 before the component is learned. 14 more have repeated
  crossings only in a never-learned component.
- **E6.** Minor corrections:
  - two-copy / centred ratio 0.791 / 0.794 / 0.794;
  - 1.5-6.3% above P5's bracket at chi = 9-16;
  - probe / tensor kernel 0.68-0.83 as medians (single runs 0.50-1.05);
  - P20 saturation at the lower edge of 0.6-0.7;
  - imbalance map tie at lambda / sqrt(chi) = 0.1, alpha = 1.


## 4. Round 2: sigma_c(K) at large depth in realizable problems (pre-registered 2026-10-10 12:26 UTC)

Written before any round-2 run. Changes made after the runs start are listed as amendments in 4.6.
(E7: the pre-registration commit f57a28c is from 12:26:25 UTC; the first round-2 raw file, a pilot, is
from 12:27:57 and the first arm-P file from 12:32:29. The heading said 12:50, which is the arm-C launch.)
Raw outputs: `/root/outputs/regime_map_r2/`. Code: `experiments/regime_map/round2.py`.

### 4.1 Question and setup

Question: does the S2S/rich boundary sigma_c(K) tend to a constant as K grows, and is it 1 or sqrt(d)?
Round 1 could not answer this. At K >= 8 the centred chi = 4 bonds are deficient, so the values were
brackets or came from floor-stalled runs, and only d = 2 was run (section 3.2 item 4).

- **Arm P (primary): plain model, spare channels.** The target is W*, which is realizable because
  rank W* = R = 2 <= chi on every cut.
  - chi = 4 = 2R on every bond, capped at the cut dimension.
  - Odeco teacher with weights (3, 2) at both **d = 2 and d = 3**. The teacher spectrum, rank and chi
    are identical, so only D = d^N changes between the two values of d.
  - Trees: `binary_tree(N)`, N = 4..16, i.e. K = N - 1 = 3..15. This is one tree family, and
    D = d^(K+1).
  - Fan-in init, 6 seeds (init seeds 0-5, teacher seeds 1000 + seed).
- **Arm C (secondary, if the budget allows): centred model with full bonds.** Every chi_e equals the
  cut dimension, so the target W* + W_0 is realizable. This arm has the same teacher and trees, at the
  K reachable in the budget (P grows like d^(N/2) chi).
  - Arm C removes the init function from the target, so the random part of W_0 does not have to be
    unlearned (see H_W).
- **Rules and code.** The merged round-1 fix code: last-crossing events, refined on a linear grid, and
  the labels of `notes/diagnostics.md`. Float64, `torch.set_num_threads(1)` per process, LSODA for
  long stiff plateaus.
- **Per-seed boundary.** With R = 2 there is a single ratio rho = rho_1. sigma_c is the sigma where rho
  first drops through rho* = 3 coming from small sigma, interpolated log-log between the two runs that
  bracket the crossing.
  - A run that learned component 1 but not component 2 by the end of its budget counts as S2S-like when
    (t_end - t_1) / w_1 > 3. This is a lower bound on rho.
  - A run without any event carries no information and is skipped; a skipped run inside a bracket is
    flagged.
  - The sigma grid is adaptive per seed: a coarse ladder, then bisection, until the bracket is at most
    a factor 2^(1/(K-2)) in sigma, i.e. about a factor 2 in rho.

### 4.2 Hypotheses (all **conjecture** before this round)

- **H_A (law A; theory_regimes P15, H4 "kappa_K -> 1").** sigma_c^K / u_0c(K) is independent of K and
  of d.
  - Consequences: sigma_c -> 1, and the d-ratio r_K := sigma_c(d=3) / sigma_c(d=2) = 1 at every K.
- **H_B (law B; R9 effective amplitude u_eff = c sigma^K / sqrt(D) with c independent of K and d; the
  round-1 conjecture).** sigma_c^K / (sqrt(D) u_0c(K)) is independent of K and of d.
  - Consequences: sigma_c -> sqrt(d), and r_K = 1.5^(N/(2K)), which is 1.31 at K = 3, 1.26 at K = 7 and
    1.24 at K = 15.
  - Calibration from theory R9 (bin8, d = 2, full bonds, plain: sigma_c = 0.71) gives sigma_c(K = 15) =
    1.07 (d = 2) and 1.32 (d = 3) under H_B, and 0.89 under H_A.
- **H_W (plain model only; heuristic, this round).** In the plain model every core norm obeys
  d||G_v||^2/dt = -2(||W||^2 - <W, W*>).
  - At init, ||W||^2 ~ sigma^(2K) and <W, W*> ~ s sigma^K / sqrt(D).
  - So before the first escape the cores shrink by a relative amount ~ sigma^K D^(1/2 - 1/K) / s: the
    plain model unlearns the random part of W_0 while it waits on the plateau.
  - Where sigma^K sqrt(D) / (s d) >~ 1 this shrinkage lowers the effective amplitude, so plain-model
    plateaus are longer than H_A or H_B predict and the plain boundary lies above both.
  - At the H_B boundary this happens for K >~ 7 (d = 2) and K >~ 5 (d = 3).
  - Round-1 indication (old event rule, d = 2, chi = 4): plain binary 16 had rho ~ 4300 at sigma =
    0.92, where H_B predicts ~ 20, and rho ~ 2 only at sigma = 2.
  - Prediction: arm P gives beta > 1 (below) at large K, with sigma_c possibly above sqrt(d); arm C does
    not.

### 4.3 Statistic and fits

- **Per-seed statistic.** y = ln(sigma_c^K / u_0c(K)). Here u_0c(K) = a_0c(K)^K is the exact
  aligned-init boundary amplitude for weights (3, 2) at the same rho*, computed by
  `analysis.aligned_boundary_a0`; it runs from 2.8e-4 at K = 3 to 1.19e-2 at K = 15.
  - Dividing by u_0c removes the K-dependence that the threshold itself induces: rho ~ u^(-(K-2)/K),
    so a fixed rho* corresponds to a K-dependent amplitude.
- **Model.** y = -gamma + beta ln sqrt(D). H_A is beta = 0; H_B is beta = 1. Equivalently the
  large-K limit is sigma_c -> d^(beta/2).
- **Fits per arm**, each with standard errors:
  1. Joint fit over both d with common gamma and beta (primary). Weighted least squares on the cell
     means of y, with weights from a bootstrap over seeds (10^4 resamples).
  2. Per-d fits (gamma_d, beta_d). These test the K-trend alone.
  3. The d-contrast at fixed K: Delta y_K = y(d=3) - y(d=2) = beta (N/2) ln 1.5. This is a
     one-parameter fit for beta_Delta, which needs no gamma if gamma does not depend on d. It is the
     cleanest test because tree, chi, teacher spectrum and K are all identical.
  4. The one-parameter laws (beta fixed at 0 or at 1, gamma free), per d and jointly: chi^2 / dof and
     p-value.
- **Reported per cell (K, d):** n resolved / 6; median sigma_c with a bootstrap 95% CI; the brackets of
  unresolved seeds. Also the d-ratio r_K with a bootstrap CI.

### 4.4 Decision rule

- **A law is "favoured"** when all three hold:
  - the primary beta is within 2 SE of the law's value;
  - the other law's value is more than 3 SE away;
  - the law's one-parameter fit has chi^2 / dof < 2.
- **"Neither"** when beta is more than 3 SE from both 0 and 1.
- **"Unresolved"** in every other case.
- The rule is applied to arm P and arm C separately. For arm P it is applied twice: to all cells, and
  to the cells without random-part shrinkage, sigma_c^K sqrt(D) / (s_2 d) < 1 (H_W).
- **Data that would favour each hypothesis:**

  | hypothesis | r_K | y vs K | other |
  |---|---|---|---|
  | H_A | ~1 at every K | flat in K at both d | |
  | H_B | 1.24-1.31 at every K | grows like ln 2^(N/2) at d = 2 and ln 3^(N/2) at d = 3 | |
  | H_W | | arm P: grows faster than ln sqrt(D) at large K (beta > 1) | sigma_c(K = 13-15) above sqrt(d), while arm C stays near beta = 1 |

- **Censoring.** Seeds without a crossing inside the explored range are reported as one-sided bounds
  and left out of the fits, with their count reported. A cell with more than 2 of 6 seeds censored is
  left out of the fits and reported separately.
- **Threshold sensitivity.** Repeat the fits at rho* = 2 and 5, recomputing u_0c for that rho*.
- **Large-K value.** sigma_c at K = 13-15 with its CI, compared with 1 and sqrt(d).

### 4.5 Budget

About 3 h of wall-clock on this machine (384 cores). Arm P has priority over arm C. If the large-K
runs do not finish, the report gives the largest K reached and how the run time scales with K.

### 4.6 Amendments (made after the timing pilots and after the first arm-P rows, which were discarded and re-run)

(E10: the header said "none uses arm results". Amendments 1-3 were committed 21 s after the last
discarded arm-P trajectory (12:34:06), and two of the discarded rows show the t = 0 event bug. The
domain rule depends only on a(0), so it cannot select on outcomes.)

1. **Time budget.** run_regime's default T = 50 t_esc + 1e4 / max(lambda_bar, 1) is short at large
   sigma: the plain model first shrinks W_0 and then escapes from a small residual scale. The pilot
   (binary 16, d = 3, sigma = 2) stopped at T = 1.5e3 with nothing learned. Round 2 therefore uses
   T = max(50 t_esc + 1e4, 1e6), and max_steps (3e4 in arm P) bounds the cost.
2. **Domain of the event rule.** The final rise starts at a_r = 0.1. In the plain model at large sigma,
   W_0 itself projects on component r with amplitude a_r(0) ~ N(0,1) sigma^K / (sqrt(D) s_r), so a
   component can start above 0.1 or even above 0.9.
   - Pilot: binary 16, d = 2, sigma = 2 has a_1(0) = 8.85 (E8; first written as ~40), which the rule
     recorded as an "event" at t = 0. Arm P later reached max_r |a_r(0)| = 50 (binary 16, d = 2,
     sigma = 2.25).
   - Runs with max_r |a_r(0)| >= 0.1 are therefore outside the rule's domain. They are classified 'D'
     and carry no information; rows now store a0 = a(t_0).
   - A seed that is S2S-like up to its first out-of-domain sigma is reported as censored, with status
     'domain' (sigma_c > the last S2S-like sigma).
   - The domain limit is sigma_dom ~ (0.1 s_2 sqrt(D) / |z|)^(1/K). (E4: it does not tend to sqrt(d).
     z = sqrt(D) <W(xi), u> is a product of K per-core factors with E log < 0: log|z| falls by about
     0.14 per core (data -0.143 at d = 2, -0.105 at d = 3; the verifier's simulation -0.14 at
     K = 31-127, chi = 4), so sigma_dom -> ~1.15 sqrt(d), about 1.63 (d = 2) and 1.99 (d = 3). The
     K = 15 medians are 1.48 and 1.86, not ~1.3 and ~1.6 as first written.)
   - H_B's boundary sigma_c^K / sqrt(D) ~ 0.86 u_0c ~ 0.01 lies inside the domain, so H_B stays
     testable.
   - Arm C (centred) is not affected: there the overlaps of W - W_0 start at 0. Its direct loss
     records the overlaps of W, which are offset by a_r(0) ~ 0.005 near the boundary; the effect on
     the event times is negligible.
3. **Process pool.** Workers are forked on Linux. A spawned worker holds its own copy of torch
   (~440 MB RSS), so 330 spawned workers ran out of the 125 GB of memory: the first launch died with
   BrokenProcessPool. The first 22 rows of arm P were discarded because they used the pre-amendment
   event handling (`/root/outputs/regime_map_r2/r2_plain_discarded_eventbug/`). (E9: 22 rows plus 112
   trajectories; 90 trajectories have no row because the pool died before they were recorded. They lack
   the a0 field and were excluded wholesale, not by outcome. All 112 configurations were re-run
   bit-identically, so the results are unaffected.)
4. **Arm C scope.** The full-bond dopri5 pilots (binary 8 and 10 at d = 2, binary 6 and 7 at d = 3;
   P = 830-3900) had not finished after 20 min under full load. Arm C therefore uses LSODA and only
   the trees with P <= 1152: d = 2 for K = 3-8, and d = 3 for K = 3-4. Arm C's d-contrast thus rests on
   K = 3-4 only.
   - Full bonds also differ between the two d (chi_e = 4 vs 9 at N = 4), so the arm C d-contrast mixes
     d with chi. Arm P keeps chi = 4 at both d.
5. **Out-of-memory crashes in arm P.** The arm P driver died twice (12:54 and 13:41 UTC) with
   BrokenProcessPool.
   - Cause: `run_regime.tensor_kernel` builds a D x D identity for its batched reverse passes. The
     guard only bounded D x P, so trees with D = 2^14-2^15 (d = 2) or 3^9 (d = 3) allocated 2-9 GB per
     run, and the kernel killed workers.
   - Fix: the guard now also requires D^2 <= 2e7, and arm P skips the tensor kernel, which only the
     lazy label needs, not sigma_c.
   - Every completed row is kept (each relaunch resumes from the rows). Some walkers lost up to an hour
     of wall-clock.
6. **Arm C extension** (E11; added to this list after the verification). At 14:22 UTC (commit 32f24f7)
   arm C was extended to (K, d) = (9, 2) and (5, 3) on cores left idle by arm P.
   - (5, 3) ran one sigma per seed (rho 2.7-6.6). It enters the arm-C Tobit fit as 5 lower bounds and 1
     upper bound.
   - (9, 2) produced no rows.

## 5. Round-1 N = 4 maps relabelled with the tensor-space kernel (lead's request, 2026-10-10)

**Change.** Labels of `sigma_chi_b4`, `chi_b4` and `imbalance_b4` use the tensor-space kernel J D J^T of
`notes/theory_regimes.md` instead of the 128-probe kernel. The rules are unchanged: delta_K < 0.1 is
lazy, rho > 3 is S2S, and convergence is measured against 1/2 ||W*||^2.
- dK_tensor was computed for all 5,298 runs from theta_end and the rebuilt initial point
  (`relabel_tensor.py compute`; cache `/root/outputs/regime_map/tensor_kernel/`, none missing).
- `analysis.load` selects the label kernel with `REGIME_MAP_LABEL_KERNEL=tensor`.
- The report tables (`report_tables.py --only sigma_chi_b4 imbalance`), the numbers
  (`report_numbers.py`), the report figures (`plot_report.py`; copied to `report/figures/regime_map/`)
  and the standard figures of these sweeps (`make_figures.py --only sigma_chi_b4 chi_b4 imbalance_b4
  summary`) were regenerated with it.
- The K sweeps, the N = 8 maps and the large-chi tables keep the probe kernel. Large chi and P20 already
  quoted the tensor kernel.

**Label changes** (`results/tensor_relabel/label_changes.csv`): 65 of 5,298 runs (1.2%).

| sweep | model | runs | changed | transitions |
|---|---|---|---|---|
| sigma_chi_b4 | centred | 912 | 9 | lazy -> rich |
| sigma_chi_b4 | plain | 912 | 9 | 3 lazy -> rich, 3 rich -> lazy, 3 s2s -> lazy |
| sigma_chi_b4 | two-copy | 912 | 6 | lazy -> rich |
| chi_b4 | centred | 1188 | 15 | lazy -> rich |
| chi_b4 | two-copy | 240 | 4 | lazy -> rich |
| imbalance_b4 | all | 1134 | 22 | lazy -> rich |

Near the boundary the probe kernel is ~0.8x the tensor value (section 3.2 item 1), so almost every
change is lazy -> rich at the lazy boundary.

**Lazy boundary** (per-seed sigma where delta_K drops below 0.1, coming from large sigma;
`results/tensor_relabel/lazy_boundary_*.csv`). The tensor / probe ratio of per-seed boundaries, median
per chi:

| sweep | ratio |
|---|---|
| centred, chi_b4 | 1.04-1.09 at chi = 12-128 (0.99 at chi = 9, where only 3 seeds have a tensor boundary) |
| centred, sigma_chi_b4 | 1.04-1.08 at chi = 12-36; 1.14 at chi = 9 |
| two-copy | 1.01-1.07 at chi >= 9; 1.15 at chi = 6 (2 seeds) |
| plain | 1.04-1.07 |

The lazy boundary therefore moves up by 1-9% (centred 4-9%), and by 14-15% at chi = 6-9 (E18; first
written as "about 4-9%", which is the centred range). The S2S boundary and the rho statistics do not
change: they do not depend on the kernel.

**Imbalance map at chi = 24** (`report/imbalance_map_b4.csv`, panel (c) of `report_map`). 7 of 189
majority cells change, all lazy -> rich and all at chi = 24:
- **Balance and heavy roots** (lambda / sqrt(chi) >= 1) are now lazy from alpha = 100. At alpha = 10
  balance is rich (4/6), and heavy roots are rich (6/6).
- **Light root at lambda / sqrt(chi) = 0.01** is lazy from alpha = 1 (it was 0.1 with 4/6 seeds),
  and so is 0.03.
- The claims become "a light root lowers the lazy boundary by ~100x in alpha (alpha = 1 vs 100)" and
  "a heavy root is lazy from alpha = 100, as at balance". These replace "alpha = 10" in sections 0,
  2.4 and 3.2 item 6 (probe-kernel statements).
- **Status:** empirical, majority labels of 6 seeds, one-decade alpha grid.

### 4.7 Round 2 results (runs stopped at 15:34 UTC, about 3 h after the first round-2 run)

Tables in `experiments/regime_map/results/round2/`: `per_seed_*.csv`, `cells_*_rho{2,3,5}.csv`,
`fits.csv`, `ratio_*.csv`, `censored_*.csv`, `shrink_*.csv`, `timing_*.csv`. Figures:
`round2_sigma_c_P.png`, `round2_sigma_c_C.png`, `round2_shrink.png`. Script: `round2_analysis.py`.
Status tag for every number below: **empirical**.

**Arm P (plain, chi = 4, teacher (3, 2), binary N = 4-16, d = 2 and 3; 1,070 runs, 156 seeds).**

| K | d = 2: resolved / censored | median sigma_c [95% CI] | d = 3: resolved / censored | median sigma_c [95% CI] |
|---|---|---|---|---|
| 3 | 6 / 0 | 0.16 [0.08, 0.35] | 6 / 0 | 0.20 [0.03, 0.31] |
| 4 | 6 / 0 | 0.39 [0.30, 0.49] | 4 / 2 | 0.44 [0.37, 0.74] |
| 5 | 6 / 0 | 0.59 [0.45, 0.81] | 6 / 0 | 0.92 [0.40, 1.30] |
| 6 | 5 / 1 | 0.85 [0.62, 1.18] | 3 / 3 | 0.75 [0.75, 0.83] |
| 7 | 5 / 1 | 0.78 [0.50, 1.01] | 1 / 5 | (0.86) |
| 8 | 2 / 4 | 1.12 [1.07, 1.16] | 0 / 6 | - |
| 9 | 2 / 4 | 0.95 [0.86, 1.05] | 0 / 6 | - |
| 10 | 2 / 4 | 1.25 [1.12, 1.37] | 1 / 5 | (1.08) |
| 11 | 0 / 6 | - | 0 / 6 | - |
| 12 | 4 / 2 | 1.19 [0.96, 1.22] | 0 / 6 | - |
| 13 | 0 / 6 | - | 0 / 6 (3 open) | - |
| 14 | 0 / 6 | - | 0 / 6 (1 open) | - |
| 15 | 0 / 6 (2 open) | - | 0 / 6 (3 open) | - |

- "Censored" = status 'domain' or 'open'. 'domain': S2S-like (rho > 3) at every sampled sigma up to
  the first sigma whose initial overlap max_r |a_r(0)| reaches 0.1, where the event rule stops
  applying (amendment 4.6 item 2). "open" = still S2S-like at the last run when the budget ended
  (status 'all_s2s'), at 0.79-0.995 of the seed's own domain limit. The "(n open)" marks follow
  per_seed_P.csv: 9 open seeds (E16; the table first showed 12, apparently walker states from the run
  log).
- **Censoring dominates at large K.**
  - 97 of 156 seeds are censored from above (88 'domain', 9 open).
  - At K >= 8, 85 of 96 seeds are censored (76 'domain', 9 open) and 11 cross. All 11 crossings lie
    below sqrt(d): sigma_c = 0.86-1.37 at d = 2, K = 8-12 (4 of 6 seeds at K = 12), and 1.08 at d = 3
    (E1, E2, E5).
  - d = 3, K >= 8: 40 'domain', 7 open and one crossing (E1; first written as "no seed crosses").
    K = 10, seed 0 has rho = 504, 213, 35.6, 2.86, 0.33 at sigma = 0.78-1.13, so sigma_c = 1.08, well
    below its domain limit of 1.82.
  - Lower bounds at K >= 11: median 1.45 (range 1.18-1.75) at d = 2 and 1.69 (1.41-2.03) at d = 3.
    At K >= 8 the range is 1.07-2.03; all 97 bounds span 0.97-2.34 (E12).
  - Per-seed domain limits sigma_dom (exact, since a(0) ~ sigma^K): medians 1.50 (d = 2) and 1.80
    (d = 3) at K >= 11, i.e. 1.04-1.06 sqrt(d), and still rising with K. They do not approach sqrt(d):
    the large-K limit is about 1.15 sqrt(d) (E4, amendment 4.6 item 2).
  - The censored bounds at K >= 8 sit a median 3% below each seed's own domain limit. Only 38 of these
    85 bounds exceed sqrt(d): 20 of 38 at d = 2 and 18 of 47 at d = 3 (E4).
  - 95 of the 97 lower bounds exceed 1, and all 97 exceed both the H_A and the H_B prediction (each law
    with gamma fitted on the resolved cells; smallest margins 3% over H_B and 27% over H_A).
  - The bounds hold at the sampled sigmas only (E13). The 85 censored seeds at K >= 8 have a median of
    4 in-domain runs (range 1-7), with gaps up to a factor 1.86 in sigma (17 seeds have a gap > 1.3).
    Most top-end rho values are far above 3: the median smallest S2S-like rho is 117 at d = 2 and 2,630
    at d = 3. But rho is not monotone in sigma (K = 11, d = 2, seed 4: rho = 3.94 at sigma = 1.29 and
    12.1 at 1.32), and 7 seeds have an S2S-like rho < 6. For these 7 a rich window in an unsampled gap
    cannot be excluded.
  - 14 of the 97 bounds rest on the lower-bound rule of 4.1 (E14). All are at d = 3, K = 12-15, where
    the second component was not learned by t_end (up to 1e6), so a stall would be classified the same
    way. Also, (t_end - t_1) / w_1 bounds rho = gap / max(w_1, w_2) from below only if w_2 <= w_1; this
    caveat applies to the registered rule itself.
- **Pre-registered fits** (rule 4.4; cells with at most 2 censored seeds, i.e. at rho* = 3: d = 2,
  K = 3-7 and 12; d = 3, K = 3-5; E15).

  | rho* | cells | beta | rule |
  |---|---|---|---|
  | 3 | 9 | 1.21 +- 0.22 (H_A chi^2 35/8, H_B 7/8) | H_B favoured |
  | 5 | | 1.555 +- 0.294 with Birge-scaled SEs; 1.555 +- 0.263 without | H_B favoured with Birge-scaled SEs; unresolved without (z_B = 2.11) (E6) |
  | 2 | | 0.72 +- 0.56 | unresolved |
  | 3, cells without shrinkage (x_W < 1, K = 3-7) | | 0.61 +- 0.51 | unresolved |
  | 3, median-based variant | | 1.38 +- 0.30 | not pre-registered (E19) |

  - E6: the fitting code (`wls`) multiplies the SEs by sqrt(chi^2 / dof) when that exceeds 1 (Birge
    scaling). The pre-registration does not include this, and it decides rho* = 5. The primary
    rho* = 3 fit (Birge factor 1) and arm C are unaffected.
  - d-contrast (cells with both d resolved: K = 3-5, dof 2; d = 3, K = 6 has only 3/6 resolved seeds;
    E15): beta_Delta = 0.81 +- 0.51.
  - d-ratios r_K = 1.27 [0.22, 3.08], 1.13 [0.86, 1.92], 1.57 [0.63, 2.45], 0.89 [0.64, 1.22] at
    K = 3-6. The CIs at K = 3-5 contain both 1 and H_B's 1.27-1.31. At K = 6 the CI excludes H_B's
    1.27, but censoring removes 3 of the 6 d = 3 seeds there (E15; first written as "each CI contains
    both").
  - The cells left in these fits are a selection: censoring removes the large-K cells, and with them
    the seeds with the largest sigma_c.
- **Censored fit over all seeds** (Tobit, not pre-registered; 59 exact values, 97 lower bounds):
  beta = 3.15 +- 0.23 (bootstrap over seeds). Likelihood-ratio tests reject H_A (p = 1e-36) and H_B
  (p = 4e-22); rho* = 2 and 5 give 3.11 and 3.16. This model extrapolates a linear law into the region
  where the event-defined boundary does not exist, so read it only as "at large K the plain boundary,
  if any, lies above both laws".
- **Mechanism H_W** (`shrink_P.csv`, `round2_shrink.png`). The pre-escape loss of squared core norm,
  1 - min_v ||G_v(t)||^2 / ||G_v(0)||^2 before any overlap reaches 0.1, against
  x_W = sigma^K sqrt(D) / (s_2 d):

  | x_W | <= 0.01 | 0.1-1 | 1-10 | 10-100 | 100-1e3 | 1e3-1e4 | >= 1e4 |
  |---|---|---|---|---|---|---|---|
  | median shrinkage | 0.1% | 0.5% | 10% | 34% | 52% | 72% | 92% |

  - In the centred model (arm C, W_0 part of the target) the median is 0 at every x_W up to 1e2.
  - Near the top of the S2S block the per-seed local exponent of rho vs sigma is far from -(K - 2)
    once x_W >~ 1: medians -3.7, -4.9, -6.4 at K = 8-10 (d = 2; predicted -6, -7, -8), and -2.9 to
    +1.8 at d = 3, K = 8-13. rho stops falling with sigma: the effective amplitude saturates.
  - (E3; first written as "This confirms H_W's mechanism and direction", tagged "heuristic", which is
    not a project tag.) The data are **consistent** with H_W; they do not confirm it.
    - The shrinkage itself follows from the exact identity d||G_v||^2/dt = -2(||W||^2 - <W, W*>)
      whenever ||W_0||^2 >> <W_0, W*>. Its absence in the centred model follows from that model's loss.
    - What the data add: the size of the shrinkage, which grows with x_W from about x_W ~ 1 as the
      heuristic says, and the flattening of rho vs sigma near the top of the S2S block.
    - Nothing tests the causal step from shrinkage to longer plateaus to a boundary above both laws.
      That needs an intervention, e.g. the plain model started from the shrunk state, or a plateau
      length predicted from the shrunk amplitude.
    - H_W's registered signature (4.4) also required "arm C stays near beta = 1". Arm C gave
      -0.07 +- 0.51, 2.1 SE from 1.
    - Status: H_W **conjecture**; shrinkage numbers **empirical**; the norm identity is standard.

**Arm C (centred, full bonds; 360 runs).**

| K (d = 2) | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| median sigma_c [95% CI] | 0.16 [0.08, 0.37] | 0.39 [0.30, 0.49] | 0.50 [0.23, 0.82] | 0.65 [0.54, 0.75] | 0.81 [0.46, 0.91] | 0.83 [0.47, 1.15] |

- d = 3: K = 3 gives 0.15 [0.13, 0.19] and K = 4 gives 0.28 [0.11, 0.39], all 6/6 seeds resolved.
- K = 5 at d = 3 (P = 2349) and K = 9 at d = 2 (P = 3264) were started at 14:22 but did not finish:
  6 runs of about 1 h each, no bracket. (E11: (5, 3) has one run per seed, rho 2.7-6.6, which enter the
  Tobit fit below as 5 lower bounds and 1 upper bound; (9, 2) produced no rows. Amendment 6.)
- Per-seed rho exponents: -1.05, -2.13, -3.21, -4.93, -5.18, -6.93 at K = 3-8, against -(K - 2).
- Fits:
  - beta = -0.07 +- 0.51 (H_A 0.1 SE, H_B 2.1 SE; rule: unresolved);
  - median variant 0.21 +- 0.63;
  - Tobit -0.17 +- 0.55 (LR vs H_B p = 0.08).
- d-ratio 0.93 [0.41, 2.08] at K = 3 and 0.73 [0.28, 1.08] at K = 4. The K = 4 CI excludes H_B's
  1.29, but here d also changes the full-bond chi (4 vs 9).
- The plain arm agrees with arm C up to K = 4 (0.16, 0.39 at both). It lies above arm C from K = 5,
  where x_W at the boundary approaches 1 (K = 5: 0.59 vs 0.50; K = 6: 0.85 vs 0.65).

**Run time** (`timing_*.csv`; wall time per run under full load, about 330 concurrent runs).
- Arm P (P = 48-688): median 100-250 s at K = 3-5, 300-650 s at K = 6-10 and 700-1,650 s at
  K = 13-15 (timing_P.csv; first written as 800-1,600 s). The maximum is about 1.3 h; run time grows
  roughly linearly with K.
- Each seed needs 5-7 sequential walker steps, so at K >= 13 some walkers had not finished when the
  budget ended. By the analysis statuses 0-3 seeds per cell are open (9 in all: d = 2, K = 15: 2;
  d = 3, K = 13: 3, K = 14: 1, K = 15: 3), not "3-6 seeds per cell" as first written (E16). The
  binding constraint at large K is the domain limit, not the budget.
- Arm C: P = 832-1,152 (K = 7-8, d = 2) takes 500-700 s per run; P = 2,349 takes about 1 h per run
  with LSODA's exact Hessian. Full bonds are therefore out of reach beyond K = 8 (d = 2) and K = 4
  (d = 3) on this machine.

### 4.8 Answer and status

- **Plain model with spare channels** (the realizable version of H4 as originally stated):
  - sigma_c does **not** tend to 1. At K >= 8, 85 of 96 seeds are censored from above: 76 are S2S-like
    at every sampled sigma up to the event rule's domain limit sigma_dom = (0.1 s_2 sqrt(D) / |z|)^(1/K),
    and 9 were still S2S-like (open) when the budget ended (E2). At d = 3 this holds for 47 of 48
    seeds; seed 0 at K = 10 crosses at 1.08 (E1). The censored bounds at K >= 8 (1.07-2.03, E12) all
    exceed 1, and all 97 bounds exceed the H_A and H_B extrapolations.
  - For most seeds (85/96 at K >= 8) no crossing is found below the domain limit, i.e. below the scale
    where the initial function itself overlaps a teacher component at 0.1. 11 of the 96 do cross, all
    below sqrt(d) (E5; first written as "not a crossing below sqrt(d) at all").
  - Mechanism: H_W, the plain model unlearning W_0 on the plateau. Consistent with the data, not
    confirmed: the core shrinkage is required by an exact norm identity, the causal step to the
    boundary is untested, and H_W's registered signature "arm C near beta = 1" failed. Status
    **conjecture** (E3; first written as "confirmed by the core-shrinkage diagnostic").
  - Status: **empirical** (binary trees, chi = 4, teacher (3, 2), d = 2 and 3, K = 3-15). The rejection
    of H4 uses the domain censoring (amendment 4.6) and a comparison of bounds with extrapolated laws
    that was not pre-registered. The bounds hold at the sampled sigmas only, and 14 of them rest on the
    lower-bound rule (E13, E14).
  - On the cells with at most 2 censored seeds (d = 2: K = 3-7 and 12; d = 3: K = 3-5), the
    pre-registered rule favours H_B at rho* = 3 (beta = 1.21 +- 0.22). At rho* = 5 it favours H_B only
    with Birge-scaled SEs, which were not pre-registered; without them it is unresolved (E6). It is
    unresolved at rho* = 2 and in the no-shrinkage subset. This is a censored selection and does not
    answer the large-K question.
- **Centred model with full bonds** (no W_0 to unlearn; K <= 8 at d = 2, K <= 4 at d = 3): beta =
  -0.07 +- 0.51, **unresolved** (leans H_A; H_B 2.1 SE away).
  - sigma_c rises 0.16 -> 0.83 over K = 3-8 and is still below 1.
- **H4 (sigma_c -> 1) for the plain model is rejected (empirical).**
- **"sigma_c -> sqrt(d)" stays a conjecture.** In the plain model most seeds' S2S region reaches the
  domain limit. The bounds sit a median 3% below each seed's own domain limit, whose median is about
  sqrt(d) at K = 8-15 (1.04-1.06 sqrt(d) at K >= 11) and which tends to about 1.15 sqrt(d), not
  sqrt(d). Only 38 of the 85 censored bounds at K >= 8 exceed sqrt(d), so sqrt(d) is not shown to be a
  lower bound (E4; first written as "tends to sqrt(d), so sqrt(d) is a lower bound there"). In the
  centred model the data do not reach the large-K regime.
- **Not resolved:**
  - the centred boundary beyond K = 8 (cost);
  - the d-dependence at fixed K: both arms' d-ratios have CIs containing 1 and 1.25-1.3, except arm P
    at K = 6 (CI [0.64, 1.22], with 3 of 6 d = 3 seeds censored; E15) and arm C at K = 4, which has
    the chi confound;
  - what happens above the domain limit, where W_0 contains each teacher component at amplitude >= 0.1
    and "learning from scratch" no longer applies.

### 4.9 Errata from the verification of round 2 (`notes/reviews/regime_map_round3_verification.md`, 2026-10-10)

Applied in place in sections 0, 4.6-4.8 and 5, each marked "(E#)". Major (the claim changes):
- **E1.** d = 3, K >= 8 is not "every seed": 40 'domain', 7 open, and seed 0 at K = 10 crosses at 1.08.
- **E2.** 85/96 seeds at K >= 8 are censored (76 at the domain limit, 9 open), not "85 stay S2S up to the
  domain limit".
- **E3.** H_W is consistent with the data, not confirmed: conjecture; shrinkage numbers empirical.
- **E4.** sigma_dom tends to ~1.15 sqrt(d), not sqrt(d); medians 1.04-1.06 sqrt(d) at K >= 11; only 38 of
  85 bounds at K >= 8 exceed sqrt(d).
- **E5.** 11 of 96 seeds at K >= 8 cross below sqrt(d).

Minor (wording, numbers, provenance):
- **E6.** rho* = 5 is "H_B favoured" only with Birge-scaled SEs, which were not pre-registered.
- **E7.** Pre-registration at 12:26 UTC (f57a28c), not 12:50.
- **E8.** Pilot a_1(0) = 8.85, not ~40.
- **E9.** Discarded: 22 rows plus 112 trajectories, all re-run bit-identically.
- **E10.** Amendments were made after the first (discarded) arm-P rows.
- **E11.** Arm-C extension to (9, 2) and (5, 3) added as amendment 6.
- **E12.** Lower-bound range 1.07-2.03 at K >= 8 (1.18-2.03 at K >= 11; 0.97-2.34 over all 97).
- **E13.** "S2S-like up to the domain limit" holds at the sampled sigmas only.
- **E14.** 14 bounds rest on the lower-bound rule; that rule bounds rho only if w_2 <= w_1.
- **E15.** d-contrast on K = 3-5; the K = 6 d-ratio CI excludes H_B's 1.27; fit cells per d spelled out.
- **E16.** 9 open seeds (CSV), not 12; "3-6 open per cell" corrected.
- **E17.** (REPORT only) at d = 3 the largest K with a resolved seed is 10 (1/6); the largest fit-eligible
  cell is K = 5.
- **E18.** Relabel: lazy boundary up 1-9% (centred 4-9%), 14-15% at chi = 6-9.
- **E19.** The median-based variant is marked "not pre-registered".

Also corrected: arm-P run time at K = 13-15 is 700-1,650 s (timing_P.csv). Not re-checked by the
verifier: the imbalance-map statement "7 of 189 majority cells change" and the regenerated report
figures.
