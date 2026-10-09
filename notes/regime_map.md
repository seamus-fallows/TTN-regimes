# Map of regimes (Q5, H4): notes of agent `regime-map`

Status tags: **proven**, **derived** (assumptions stated), **empirical**, **conjecture**.
Experiments: `experiments/regime_map/`. Raw outputs: `/root/outputs/regime_map/`.

Notation: K cores, per-core init scale sigma (fan-in init), function-space scale alpha = sigma^K,
W_0 = W(theta_0), J = dW/dtheta at theta_0 (a D x P matrix, D = prod_i d_i), tangent operator
T = J J^T, centred model: L = 1/2 ||W(theta) - W_0 - W*||^2. Defaults: d = 3, odeco teacher with
weights [3, 2, 1] (||W*||^2 = 14).

## 1. Theory (derived here; to be compared with `notes/theory_regimes.md` when it lands)

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

## 2. Results

(in progress)
