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

Library gradient flow (DP5, rtol 1e-8, atol 1e-12; events re-integrated on a dense linear grid),
6 seeds x 8 chi (3, 4, 6, 9, 12, 18, 24, 36; uncapped) x 19 sigma in [0.01, 40], centred model;
`results/phase_sigma_chi_b4_centred.png`, per-seed tables `results/sensitivity_per_seed_sigma_chi_b4_centred.csv`.
Boundaries are per seed (geometric mean x/ standard-error factor over 6 seeds); "S2S boundary" =
sigma where the seed's min_i rho_i drops below 3 (log-log interpolation), "kernel boundary" = sigma
where delta_K(T) drops below 0.1 (converged runs).

| chi | S2S boundary | predicted from deepest run (aligned extrapolation) | pred / measured | rho exponent on the S2S side (mean +- SE) | kernel boundary | ratio of S2S boundaries rho 2 / rho 5 | ratio of kernel boundaries dK 0.05 / 0.2 |
|---|---|---|---|---|---|---|---|
| 3 | 0.236 x/ 1.32 | 0.283 x/ 1.35 | 1.23 +- 0.10 | -1.056 +- 0.022 | stalled (floor) | 2.15 +- 0.05 | - |
| 4 | 0.089 x/ 1.58 | 0.118 x/ 1.50 | 1.50 +- 0.47 | -1.070 +- 0.040 | stalled | 2.03 +- 0.18 | - |
| 6 | 0.137 x/ 1.19 | 0.143 x/ 1.19 | 1.06 +- 0.05 | -1.030 +- 0.004 | stalled | 2.22 +- 0.04 | - |
| 9 | 0.136 x/ 1.11 | 0.141 x/ 1.11 | 0.97 +- 0.07 | -1.030 +- 0.004 | 3.01 x/ 1.10 | 2.27 +- 0.02 | 1.89 +- 0.11 |
| 12 | 0.088 x/ 1.33 | 0.091 x/ 1.34 | 1.04 +- 0.05 | -1.029 +- 0.002 | 2.20 x/ 1.05 | 2.29 +- 0.02 | 1.62 +- 0.03 |
| 18 | 0.072 x/ 1.07 | 0.073 x/ 1.08 | 1.08 +- 0.05 | -1.031 +- 0.000 | 1.66 x/ 1.05 | 2.30 +- 0.01 | 1.46 +- 0.03 |
| 24 | 0.067 x/ 1.12 | 0.068 x/ 1.12 | 1.01 +- 0.05 | -1.030 +- 0.002 | 1.31 x/ 1.04 | 2.33 +- 0.01 | 1.38 +- 0.02 |
| 36 | 0.043 x/ 1.05 | 0.044 x/ 1.06 | 1.03 +- 0.07 | -1.034 +- 0.003 | 1.04 x/ 1.04 | 2.34 +- 0.01 | 1.33 +- 0.01 |

- **Map** (`phase_sigma_chi_b4_centred.png`): S2S for sigma below the S2S boundary, rich in between,
  lazy above the kernel boundary only for chi >= 9 (full bonds); for chi < 9 the runs stall above
  sigma ~ 0.6-1.0 at the best fit of W_0 + W* (unrealizable centred target). **Empirical.**
- **S2S side.** Per seed, rho scales as sigma^-1.03 (6 seeds per chi, SE <= 0.04): the predicted
  sigma^-(K-2) (theory_regimes R9/P12; the 3% excess is the O(1) term of rho = R sigma^-1 + c).
  The boundary of each seed is predicted from that seed's deepest run alone by the aligned-mode
  extrapolation (`analysis.aligned_sigma_c`): pred / measured = 0.97-1.08 for chi >= 6 (1.23 and 1.50 at
  chi = 3, 4, where the boundary is pushed toward the realizability floor). The S2S boundary falls
  with chi (0.24 at chi = 3, 0.043 at chi = 36). **Empirical.** At chi = 3 it agrees with theory P11
  (plain model, 30 seeds: per-seed median 0.27-0.31).
- **Lazy side.** Kernel boundary 3.0, 2.2, 1.66, 1.31, 1.04 at chi = 9, 12, 18, 24, 36: theory_regimes P5
  (second-order law, sigma_lazy -> 6.50 chi^-1/2) gives 1.53, 1.33, 1.08 at chi = 18, 24, 36. The
  first-order law (kappa/0.1)^(1/3) is right only at chi <= 12 (2.4 at chi = 12; 0.59 at chi = 36).
  **Empirical**, confirming the theory's correction that the coherent second-order kernel change sets
  the lazy boundary at large chi.
- **Threshold sensitivity.** rho threshold 2 -> 5 moves the S2S boundary by 2.15-2.34 (asymptotic
  prediction (5/2)^(1/(K-2)) = 2.5; the O(1) term in rho lowers it at finite sigma). delta_K threshold
  0.05 -> 0.2 moves the kernel boundary by 1.89 (chi = 9) down to 1.33 (chi = 36): the local slope of
  delta_K vs x = ||W*|| / sigma^3 grows from ~1 to ~2 with chi, so the ratio goes from 4^(1/3) = 1.59
  toward 4^(1/6) = 1.26 (theory P7: 1.3-1.36 at chi >= 32). **Empirical.**
- **Realizability.** For chi < 9 every stalled run (44) ends at the Eckart-Young minimum of W_0 + W*
  across the 01|23 cut (`l_end / l_best` = 1 to 1e-6). Stall onset (all 6 seeds stalled) at sigma = 1.0,
  1.0, 1.59 for chi = 3, 4, 6 (first stalled seed at 0.63 for chi = 3); theory P4's small-sigma estimate
  sigma_floor = 0.56, 0.60, 0.71 is lower, increasingly so at chi = 6 where the floor is reached only at
  sigma ~ 1, outside the small-sigma expansion. **Empirical**, confirming theory R7 qualitatively.

### 2.1b Boundaries vs bond dimension and the collapse variable (binary N = 4)

`configs/chi_b4.yaml`: chi = 9..128 (uncapped; full / overparameterized bonds), 6 seeds, centred model
over sigma in [0.004, 4] (22 values) and two-copy over [0.25, 4]; library GF with event refinement;
figure `results/boundaries_vs_chi_chi_b4.png`, per-seed table `results/boundaries_per_seed_chi_b4.csv`.
Boundaries per seed as in 2.1 (rho = 3 crossing; delta_K = 0.1 crossing); slopes fitted to all
per-seed values (each (chi, seed) is an independent init), errors from the residual scatter.

| | slope vs chi | chi range | per-seed values | prediction |
|---|---|---|---|---|
| S2S boundary (centred) | -0.524 +- 0.045 | 16-128 | 42 | (none derived) |
| S2S boundary (centred) | -0.535 +- 0.060 | 9-128 | 54 | |
| lazy boundary (centred) | -0.543 +- 0.015 | 16-128 | 42 | -1/2 (theory P5, second order) |
| lazy boundary (centred) | -0.522 +- 0.020 | 32-128 | 30 | -1/2 |
| lazy boundary (two-copy) | -0.514 +- 0.023 | 32-128 | 18 | -1/2 (P6) |

- **Lazy boundary vs theory P5/P6 (empirical, quantitative).** sigma_lazy sqrt(chi) = 6.29, 6.14, 6.18,
  6.12, 6.10 at chi = 32, 48, 64, 96, 128 (median of seeds), against the predicted asymptote 6.50; the
  two-copy model gives 4.96, 4.91, 4.83 at chi = 32, 64, 128 against 5.16. Measured boundaries are
  5-8% below the asymptotic formulas, as in the theory note's own GF checks (0.82 / 0.79 vs
  [0.81, 0.83] at chi = 64). Threshold ratio (delta_K 0.05 vs 0.2) 1.30-1.33 at chi >= 48 (P7: 1.3-1.36).
- **The S2S boundary scales the same way (new, empirical).** sigma_c sqrt(chi) = 0.29, 0.29, 0.31,
  0.30, 0.27 at chi = 32..128. So at large chi both boundaries depend on sigma and chi only through
  sigma sqrt(chi), i.e. through the per-bond Gram scale p = chi sigma^2 / d^2 (the eigenvalue scale of
  the subtree Gram matrices B B^T): S2S below p ~ 0.01, lazy above p ~ 4.3, and the rich window is a
  fixed factor ~21 in sigma (~10^4 in alpha = sigma^3). **Collapse test** of the candidate variables
  suggested in `notes/literature.md` (H4, correction 2): sigma^K (slope 0) and the per-path amplitude
  times the number of paths (sigma^K chi^(K-1) in iid normalization = sigma^3 chi / d^2 in fan-in
  variables, slope -1/3) are rejected (the S2S slope differs from -1/3 by 4.2 standard errors, the lazy
  slope by 9.4, chi >= 32); the per-bond Gram scale (slope -1/2) is consistent with both boundaries.
  **Empirical** for binary N = 4 (a 3-factor matrix product) only; deeper trees untested at large chi.
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
  target. With fan-in init the stalled route is dead (3rd bond singular value at roundoff, 1e-16 to
  1e-18); gauge-balanced init (same W_0, every Q_e = 0, `balance.py`) removes the dead routes but the
  route still decays (s_3 ~ 3e-5) and most runs do not escape. **Empirical** (36 + 18 runs), confirming
  theory_regimes R10 / P14 and the conservation-law trap of theory_saddles R12.
- Consequence for the map: at chi = teacher rank and K >= 7, "stalled" occupies the whole S2S side
  at practical budgets; the S2S boundary is measured with spare channels or full bonds (2.3).

### 2.4 Imbalance: root scale at fixed function-space scale (binary N = 4)

`configs/imbalance_b4.yaml`: root init lambda sigma, other cores sigma, alpha = lambda sigma^3 fixed;
centred model; chi = 24 (full, overparameterized) and chi = 3 (deficient): 9 alpha (1e-6..100) x 10 lambda
(1e-2..1e2 x the expected-balance point lambda_bal = sqrt(chi)) x 6 seeds; plus theory P20's check point
chi = 16, alpha = 3.375, lambda = 0.01..100; `results/phase_imbalance_b4.png`.

- **Check point P20 (chi = 16, alpha = 3.375).** Median delta_K = 1e-4, 7e-4, 0.0069, 0.036, 0.167, 0.45,
  0.53, 0.50, 0.50 for lambda = 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100. Local slope d ln delta_K / d ln lambda
  = 2.18, 1.86, 1.51, 1.27 from the smallest lambda up: delta_K ~ lambda^2 as lambda -> 0 (theory: 1.8-1.9
  between 0.03 and 0.1; measured 1.86), saturating at 0.50-0.53 for lambda >= 3 (theory: 0.6-0.7). Labels:
  lazy for lambda <= 0.3 (all finished runs), rich for lambda >= 1 (5/6 at lambda = 1).
  **Empirical**, confirming theory R13 (a small root makes training lazy; a large root freezes the root
  and leaves a degree-(K-1) model that is not lazy); the saturation level is ~20% below the prediction.
- **Map, chi = 24 (majority labels, 6 seeds).** The lazy boundary moves to smaller alpha as the root
  shrinks: lazy for alpha >= 0.1 at lambda / sqrt(chi) = 0.01 and for alpha >= 1 at <= 0.1, but only for
  alpha >= 10 at lambda >= sqrt(chi) (a factor ~100 in alpha). The S2S region is largest near the
  expected-balance point lambda = sqrt(chi) (S2S up to alpha = 1e-3 there, 1e-4 for 0.1 <= lambda/sqrt(chi)
  <= 10) and shrinks for strong imbalance in either direction (S2S only up to alpha = 1e-5 at
  lambda / sqrt(chi) = 0.01 or 100). **Empirical.** So imbalance is a second knob in the sense of
  Kunin et al. 2024, with the TTN-specific asymmetry of theory R13: a light root -> lazy, a heavy root ->
  rich but non-incremental (its effective model has degree K - 1 = 2, whose plateaus are only
  logarithmic, so rho grows only like log(1/sigma)).
- **chi = 3 (deficient).** Same S2S/rich structure at small alpha; stalled at the centred floor for
  alpha >= 1 (all lambda), with delta_K < 0.1 (lazy residual) at alpha = 100 for lambda >= sqrt(chi).
- The fan-in default (lambda = 1) is root-light by sqrt(chi) (E Q_v = sigma^2 (1 - 1/chi) on the root
  edges): at chi = 24 it sits at lambda / sqrt(chi) = 0.2, on the light side of balance.

### 2.5 Large bond dimension: SP, NTK and muP-TTN (binary N = 4)

`configs/large_chi_b4.yaml`: chi = 4..128 (uncapped), parameterizations of theory_regimes section 4.3
(SP: fan-in, eta = 1; NTK: eta_v = 1/F_v; muP-TTN: root init x (2 chi)^-1/2, eta_v = chi_v / F_v),
centred and plain, sigma = 1.0 and 1.5, 6 seeds; `results/large_chi_large_chi_b4_*.png` and
`large_chi_exponents_*.csv`. Slopes of log(quantity) vs log chi over chi = 16..128 (24 independent
runs each; error from the residual scatter). sigma = 1.5 is the theory's check point (x = 1.11):

| | delta_K | delta_theta | root (rel. Frobenius change) | leaf cores |
|---|---|---|---|---|
| SP, measured | -3.09 +- 0.09 | -1.63 +- 0.04 | -1.02 +- 0.03 | -2.96 +- 0.07 |
| SP, theory P16 | -3 | -3/2 | -1 | -5/2 |
| NTK, measured | -0.84 +- 0.06 | -0.52 +- 0.03 | -0.93 +- 0.02 | -0.53 +- 0.03 |
| NTK, theory | -1 | -1/2 | -1 | -1/2 |
| muP-TTN, measured | +0.03 +- 0.06 | -0.06 +- 0.02 | -0.50 +- 0.01 | -0.07 +- 0.02 |
| muP-TTN, theory | 0 | 0 | -1/2 (Frobenius; spectral O(1)) | 0 |

- **Feature learning survives under muP-TTN** (delta_K ~ 4-5 at every chi, slope 0.03 +- 0.06) and is
  lost under SP (delta_K ~ chi^-3: the root dominates the kernel, the model becomes linear in the root)
  and NTK (delta_K ~ chi^-0.84 here, approaching -1 as the second-order term c_2 x^2 fades). **Empirical**,
  confirming theory P16 for delta_K, delta_theta and the root; the SP leaf-core exponent is steeper than
  predicted at sigma = 1.5 (-2.96 vs -5/2) and closer at sigma = 1.0 (-2.58 +- 0.07).
- At sigma = 1.0 (x = 3.74, further from the lazy limit) the SP and NTK delta_K slopes are shallower
  (-2.66 +- 0.09, -0.68 +- 0.06): the exponents are asymptotic in x -> 0. muP-TTN is unchanged
  (+0.03 +- 0.06).
- Plain and centred models give the same exponents within errors at chi >= 16 (e.g. SP delta_K -3.16 +-
  0.10 plain vs -3.09 +- 0.09 centred at sigma = 1.5): with full, overparameterized bonds centring does not
  change the scaling (literature correction 1).
- K = 7 (binary N = 8, d = 2, plain, sigma = 1, chi = 4..32; `large_chi_b8`): incomplete (see report).

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
