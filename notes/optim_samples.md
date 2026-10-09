# Q8: learning rate, SGD noise and finite samples (agent `optim-samples`)

Status tags: **proven**, **derived** (assumptions stated), **empirical**, **conjecture**.
Setting unless stated: whitened Gaussian inputs, exact population loss, odeco teacher
W* = sum_r s_r U_r with U_r = u_r^(1) (x) ... (x) u_r^(N), s_1 > s_2 > ..., K cores, fan-in init with
per-core scale sigma. Experiments: `experiments/optim_samples/`.

## 1. Sharpness at the global minima and the GD stability limit

**Proposition 1 (proven).** Let theta be a global minimum (W(theta) = W*) of the population loss
L = 1/2 ||W - W*||_F^2. Then H = J^T J with J = dW/dtheta, so lambda_max(H) = lambda_max(Theta), where
Theta = J J^T = sum_v J_v J_v^T is the tangent kernel acting on tensors. If the minimum is the
*balanced aligned* one (cores diagonal in the teacher basis, every core amplitude a_{v,r} =
s_r^{1/K}), or any orthogonal gauge transform of it (these are exactly the minima with Q_e = 0 when
the bond dimension equals the teacher rank), then

    lambda_max(H) = K s_1^{2 - 2/K},  top eigenvector dW = U_1.

*Proof.* For core v, write W as the contraction of G_v with the subtree tensors T_c of its children c
(matricized: sites below c x chi_c) and the environment A_v above it (sites outside the subtree of v
x chi_v). Then J_v J_v^T = (x)_c (T_c T_c^T) (x) I_{phys(v)} (x) (A_v A_v^T), whose eigenvalues are
products of squared singular values of the T_c and of A_v. At the balanced aligned minimum the
singular values of T_c are s_r^{n_c/K} (n_c = number of cores in the subtree of c) and those of A_v
are s_r^{(K - n_v)/K}; since sum_c n_c + (K - n_v) = K - 1, every J_v J_v^T has top eigenvalue
s_1^{2(K-1)/K}, attained by U_1 for every v simultaneously. Hence Theta U_1 = K s_1^{2-2/K} U_1, and
lambda_max(sum_v J_v J_v^T) <= sum_v lambda_max(J_v J_v^T) = K s_1^{2-2/K} is attained. Orthogonal
gauges leave Theta invariant. []

**Proposition 2 (derived; flattest minimum).** Along the aligned minima with per-core amplitudes
a_{v,1} for component 1 (prod_v a_{v,1} = s_1), Theta U_1 = (sum_v s_1^2 / a_{v,1}^2) U_1, so
lambda_max >= s_1^2 sum_v a_{v,1}^{-2} >= K s_1^{2-2/K} (AM-GM), with equality iff balanced. The
balanced minimum is therefore the flattest aligned minimum (the TTN analogue of the deep-linear
result of Mulayoff & Michaeli 2020). *Assumption*: we have checked only aligned/diagonal-gauge
families and orthogonal gauges, not every point of the minimum manifold (with chi > R the manifold is
larger).

**Numerical check (empirical, machine precision).** `experiments/optim_samples/check_sharpness.py`
(full Hessian): binary N=4, 8; MPS N=4, 8; HT N=4; d=3, teacher [3, 2, 1], chi = 3 and 5. The
balanced and orthogonally gauged minima give lambda_max = K s_1^{2-2/K} to relative error < 3e-15;
random diagonal imbalances give exactly s_1^2 sum_v a_v^{-2}. Results: `results/sharpness_check.csv`.

**Consequences (derived).**
- GD is linearly stable at the balanced minimum iff eta < eta_max := 2 / (K s_1^{2-2/K}).
  The PLAN.md formula is correct. For [3, 2, 1]: eta_max = 0.1541 (binary N=4, K=3), 0.04345
  (binary N=8, K=7), 0.03656 (MPS N=8, K=8).
- Because the balanced minimum is the flattest, for eta > eta_max **no aligned global minimum is
  linearly stable**, so GD cannot converge to zero loss. Small init gives Q_e = O(sigma^2), so the GF
  solution is (nearly) the flattest minimum: there is no edge-of-stability "self-stabilization"
  route (Damian et al. 2022) to a flatter minimum. The threshold is sharp.
- Component-wise version: a saddle or minimum where component r is learned (balanced) has local
  curvature K s_r^{2-2/K} in the U_r direction. With eta given, components with
  s_r > s_c(eta) := (2 / (K eta))^{K/(2K-2)} cannot be learned stably, the others can.

## 2. Balanced 1D map above eta_max (derived + numerics)

For the aligned balanced mode, per-core amplitude x = a / s^{1/K}, GD is the map
x -> x - h x^{K-1} (x^K - 1), h = eta s^{2-2/K}, with h_c = 2/K. Numerically (start x = 0.3,
2e5 iterations, K = 3, 7, 8): stable fixed point for h < h_c; stable 2-cycle straddling x = 1 for
1 < h/h_c < ~1.22 (supercritical flip; mean loss grows continuously from 0, e.g. 0.002 s^2 at
h/h_c = 1.01, 0.03 s^2 at 1.2); period 4 at ~1.25; chaos at ~1.35-1.4 with periodic windows
(period 3 at 1.5); for h/h_c >~ 2 (K = 3) or ~2.5 (K = 7, 8) an overshoot throws x back near the
origin, where the gradient ~ x^{K-1} is tiny, so the mode **re-enters the plateau at the saddle**;
divergence for larger h (K = 3: h/h_c >= 2.5).

## 3. GD versus gradient flow below eta_max (derived)

Backward error analysis: GD follows to O(eta^2) the flow of the modified loss
L~ = L + (eta/4) ||grad L||^2 (Barrett & Dherin 2021). Critical points of L are critical points of
L~ (grad L~ = grad L + (eta/2) H grad L), so **the saddles and their order are unchanged at O(eta)**.
Writing the first-order correction as a time shift along the GF path (tangential part), the GD event
times in units t = eta * steps are

    t_e^GD - t_e^GF = (eta/2) ln( ||grad L(theta_GF(t_e))|| / ||grad L(theta_0)|| ) + O(eta^2)
                      + (path-curvature term).

The path-curvature term vanishes for straight paths (aligned init: exact at O(eta)). For a balanced
mode escaping from amplitude a_0, ||grad L|| ~ a^{K-1}, so the shift is ~ (eta/2)(K-1) ln(a_e/a_0)
~ (eta/2)(K-1) ln(1/sigma): an absolute O(eta log(1/sigma)) delay, while t_e ~ sigma^{-(K-2)}. Hence
the relative error vanishes as sigma -> 0 for every stable eta, and GD is slower than GF.

Pilot (empirical, binary N=4, d=3, chi=3, [3, 2, 1], sigma = 0.1, seed 0, GF = DOP853 rtol 1e-10):
relative event-time error is linear in eta, 5.8e-4 at eta = 0.01 eta_max and 5.1e-2 at 0.9 eta_max.
Shift/eta = 2.94 (event 1), 3.16 (event 2) vs the tangential prediction 2.71, 2.42 (to be redone
with dense output and the full variational correction).

## 4. SGD noise on plateaus (derived)

The per-sample gradient is J(theta)^T Phi(x) (f(x) - y), so **SGD noise is multiplicative**: its
covariance carries the factor J, which vanishes on the unexcited directions as fast as the signal
does (J ~ sigma^{K-1} near the origin; on a new bond channel J is proportional to the other new-channel
entries). Noise of this form cannot push the system off a higher-order saddle faster than the drift.

1D aligned balanced reduction: online SGD with batch B gives
a_{k+1} = a_k + eta a_k^{K-1} xi_k, xi_k = (1/B) sum_b Z_1 Y (before the nonlinear phase), with
Z_r = prod_i <u_r^(i), x_i>, Y = sum_r s_r Z_r, E xi = s_1, Var xi = V / B,

    V = s_1^2 (3^N - 1) + sum_{r>=2} s_r^2    (E Z^4 = 3^N: product features are heavy-tailed).

With u = a^{-(K-2)}/(K-2) (Ito), du = -s_1 dt - sqrt(eta V/B) dW + [(K-1) eta V / (2 B (K-2) u)] dt.
Predictions for the first escape time t_1 (in t = eta * steps):
- mean: E t_1^SGD - t_1^GF = (eta/2)(K-1)(1 + V/(B s_1^2)) ln(a_e/a_0) > 0, i.e. the GD delay times
  E[xi^2]/E[xi]^2. **Noise lengthens, not shortens, the plateau** (volatility drag on log-growth).
- spread: std(t_1) = sqrt(t_1^GF eta V / B) / s_1, i.e. relative spread ~ sqrt(eta V/(B t_1 s_1^2))
  -> 0 as sigma -> 0.
- both depend on eta and B only through eta/B (for B << V/s_1^2): exponents +1 (mean shift) and
  +1/2 (std) in eta/B.

## 5. SGD stability limit (derived; in progress)

At an interpolating minimum (realizable teacher) the noise vanishes, and linear mean-square stability
of e = W - W* under e <- (I - eta Theta Phi_B Phi_B^T) e requires the spectral radius of
M -> M - eta(Theta M + M Theta) + eta^2[(1 - 1/B) Theta M Theta + Theta F(M) Theta / B] to be < 1,
with F(M) = E[Phi Phi^T M Phi Phi^T] = (x)_i F_i, F_i(A) = A + A^T + tr(A) I (Gaussian fourth
moments). Restricted to U_1: eta < 2 / (lambda (1 + (3^N - 1)/B)). For B << 3^N this is ~ 3^{-N} B
times the GD limit.

## 6. Finite samples (to do)
