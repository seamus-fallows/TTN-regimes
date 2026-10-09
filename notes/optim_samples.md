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
x -> x - h x^{K-1} (x^K - 1), h = eta s^{2-2/K}, with h_c = 2/K (`map1d.py`,
`results/map1d_attractor.csv`, start x = 0.3, 6e4 iterations). Stable fixed point for h < h_c; stable
2-cycle straddling x = 1 for 1 < h/h_c < 1.245 (supercritical flip; loss on the cycle grows
continuously from 0); period 4 at 1.245-1.25, period 8 at ~1.31, chaos from ~1.33 with periodic
windows (period 3 at ~1.49); the orbit falls back to the plateau near x = 0 ("re-entry") from
h/h_c = 1.84 / 1.99 / 2.23 / 2.29 and diverges from 2.16 / 2.41 / 2.72 / 2.86 for K = 3 / 4 / 7 / 8.
The flip and cascade thresholds are K-independent in units of h_c; re-entry and divergence move
up with K.

## 3. GD versus gradient flow (step 1 results)

Setup: odeco teacher [3, 2, 1], d = 3, chi = 3, teacher seed 101; fan-in init seeds 0-4; GF reference
DOP853 rtol 1e-10, atol 1e-14 (conservation drift <= 1e-13); GD at eta = ratio x eta_max,
t = eta x steps. Trees: binary N=4 (K=3) at sigma = 0.003, 0.01, 0.03, 0.1, 0.3; binary N=8 (K=7) and
MPS N=8 (K=8) at sigma = 0.9; 18 ratios from 0.01 to 3. Imbalanced variant: binary N=4, sigma = 0.1,
root core at scale 2, 4, 6. 750 runs (`gd_sweep.py`, `configs/gd_sweep.yaml`; table
`results/gd_sweep.csv`, `results/gd_summary.csv`; figures `results/fig_*.png`).

### 3.1 Event-time shift below eta_max: exact first-order theory (derived, verified)

Backward error analysis: GD follows to O(eta^2) the flow of L~ = L + (eta/4)||grad L||^2
(Barrett & Dherin 2021). Critical points of L are critical points of L~, so **the saddles are the
same**. Writing theta_GD(t) = theta_GF(t) + eta delta(t), delta solves the variational equation
delta' = -H (delta + g/2), delta(0) = 0 along the GF path (g = grad L), and the event a_r = 1/2 moves by

    t_e^GD - t_e^GF = eta c_e + O(eta^2),   c_e = -grad a_r . delta(t_e) / (d a_r/dt)(t_e).

`predict_shift.py` integrates this for every GF reference (no free parameter). Tangential
approximation (path curvature ignored): c_e ~ (1/2) ln(|g(t_e)| / |g(0)|) ~ (1/2)(K-1) ln(1/sigma) for the
first event; good for the first event at small sigma (6.13 vs exact 6.42 at sigma = 0.003), wrong
for later events (it misses negative shifts).

**Empirical check (all trees, 5 seeds, every event):** at eta <= 0.1 eta_max the measured shift/eta
equals c_e to 3 significant digits per seed, including negative shifts (e.g. binary N=4, sigma = 0.003,
event 1: measured 6.183-6.495 vs predicted 6.196-6.504; MPS N=8 seed 2: -32.40 vs -32.48);
measured/predicted stays within 2% up to 0.9 eta_max (`fig_gd_vs_gf.png` b, c).
**Scaling (derived + empirical):** relative error of t_1 = eta c_1 / t_1 with c_1 ~ ln(1/sigma) and
t_1 ~ sigma^{-(K-2)}: binary N=4 at 0.9 eta_max gives 0.111, 0.056, 0.023, 0.0099, 0.0036 for
sigma = 0.3, 0.1, 0.03, 0.01, 0.003 (fig a). So **every stable step size reproduces the GF event
times to within a few percent at small init, and the agreement improves as sigma -> 0.**

### 3.2 Saddle sequence below eta_max (empirical)

Same learning order as GF in 225/235 runs below eta_max. The 10 exceptions are two seeds of binary
N=4 at sigma = 0.1, where GF has a near-tie between components 1 and 2: GF learns component 1 at
t = 7.8 and component 2 only after a second plateau (t = 23.2 and 16.6); with eta >= 0.2 eta_max
(seed 1) or >= 0.5 eta_max (seed 3) both are learned together at t ~ 7.8-7.9 and **the second
plateau disappears**. An O(eta) perturbation can flip a near-tied race, after which the loser of
the race is no longer suppressed. No flips at sigma <= 0.03, at sigma = 0.3, or on the N = 8 trees
(5 seeds each). **Conjecture:** the probability of a flip vanishes as eta -> 0 and as sigma -> 0 (the
relative perturbation eta c / t ~ eta sigma^{K-2} ln(1/sigma)); to be quantified with more seeds.
This partly confirms the literature correction to H6 (sequence changes are possible below the
stability limit), but only through near-ties.

### 3.3 The stability limit (empirical, confirms section 1)

| tree, sigma | converges like GF | oscillates (bounded) | thrown back to origin | diverges |
|---|---|---|---|---|
| binary N=4, sigma <= 0.3 (25 runs/ratio) | all at <= 0.95 eta_max | all at 1.05-1.3 | all at 2.0 | 1.5-1.75 and >= 2.5 |
| binary N=8, sigma = 0.9 | <= 0.7 (all), 0.9 (4/5), 0.95 (2/5) | 1.05-1.5 | 1.75-2.5 (mostly) | 3.0 (4/5) |
| MPS N=8, sigma = 0.9 | <= 0.7 (all), 0.9 (2/5) | 0.95-1.5 (4/5) | 1.75-2.5 (3/5) | 3.0 (3/5) |

- For small init the boundary is exactly eta_max = 2/(K s_1^{2-2/K}) (0/170 failures at
  <= 0.95 eta_max, 25/25 oscillating at 1.05). **Empirical, 5 sigma x 5 seeds.**
- At sigma = 0.9 the N = 8 inits are imbalanced (fan-in init gives Q_e != 0 at O(sigma^2)), the GF
  state is 8-12% sharper than balanced, and the edge moves to ~2/lambda_GF ~ 0.9 eta_max, with slow
  EoS transients (runs at 0.95 still oscillating at T).
- Above eta_max the outcome follows the 1D map qualitatively (2-cycle, then period doubling, then
  re-entry to the origin, then divergence) but **the bounded chaotic window of the 1D map
  (h/h_c ~ 1.35-1.8) is not robust in the full network**: binary N=4 diverges at 1.5-1.75, while at
  2.0 every run is thrown back to the origin saddle (as the 1D map predicts) and repeatedly
  re-learns and loses components (`fig_eos_example.png`). Conjecture: transverse (imbalance)
  directions are unstable on the chaotic set (cf. Hofmann 2026, imbalance factor 1 - a^2 r^2).
- **Oscillation loss (P5 of the theory note; empirical, exact 1D comparison).** The time-averaged
  excess loss of oscillating runs, L_tail - L_GF(T), equals s_1^2 x (mean loss of the 1D map attractor,
  `map1d.py`) to 3-4 digits on binary N=4: 0.1306 vs 0.1306 (1.05 eta_max, 25 runs), 0.2472 vs 0.2479
  (1.1), 0.4496 vs 0.4496 (1.2), 0.553 vs 0.557 (1.3); MPS N=8 and binary N=8 agree to a few % up to
  1.1-1.2 (binary N=8 at 1.2-1.3: 0.26 vs 0.30-0.39, some seeds leave the 2-cycle). The theory note's
  linear form 3K^2 s_1^2/(4(K-1)(5K-4)) (eta/eta_max - 1) is its small-excess limit (5% high at 1.05,
  23% high at 1.2). So the full TTN above eta_max is the 1D balanced map of component 1 plus GF-like
  learning of the rest.
- Oscillation stays in the learned top component(s): at 1.05-1.3 eta_max component 1 oscillates
  (2- or 4-cycle) while components 2 and 3 are learned at GF-like times (fig_eos_example, middle).
  Component-wise threshold s_c(eta) (section 1) explains which components oscillate.

### 3.4 Which minimum GD selects, and balance (empirical; cf. Ghosh 2025, Kreisler 2023)

- Small (balanced) init: the GF minimum is the flattest, final sharpness = lambda_bal for every
  stable eta. GD *creates* a small imbalance that grows linearly with eta (|tr Q_e| from 0.016 (GF)
  to ~0.08 at 0.95 eta_max, sigma = 0.1), because Q_e changes by O(eta^2) per step.
- Imbalanced init (root core scale 2, 4, 6; GF minimum 1.13x, 1.68x, 2.37x sharper than balanced):
  GD converges to sharpness min(lambda_GF, 2/eta) (`fig_min_selection.png` a), i.e. it selects a
  flatter, more balanced minimum (|tr Q_e| down to ~0.6x GF) as soon as 2/eta < lambda_GF: classic
  EoS selection. Some imbalanced runs with eta > 0.75 eta_max are thrown back to the origin or
  diverge instead (root x6: 1-4 of 5 seeds).
- At the stability edge the imbalance is removed fastest: imbalanced inits that still converge at
  eta = eta_max end with |tr Q_e| ~0.35-0.4x the GF value; for small init, oscillating runs at
  1.05-1.3 eta_max end below the GF value (0.0139 vs 0.0162 at sigma = 0.1), as for DLNs (Ghosh 2025;
  Wang et al. 2022).

## 4. SGD noise on plateaus (derived)

**Noise is multiplicative.** The per-sample gradient is J(theta)^T Phi(x) (f(x) - y), so the SGD noise
covariance carries J twice. Near the origin J ~ sigma^{K-1}; for a dormant bond channel, J is
proportional to the other dormant entries. The noise therefore vanishes on every unexcited direction
as fast as the drift does (the TTN origin and dormant-channel saddles are "Type-II" saddles in the
sense of Ziyin 2023), and it cannot push the system off a higher-order saddle faster than gradient
flow. Plateaus are not shortened (H6, SGD part), unlike isotropic-noise escape (Jin et al. 2017).

**Aligned 1D reduction (derived; assumptions: aligned balanced mode, odeco teacher, other components
unlearned, eta small).** Online SGD with batch B moves the per-core amplitude of component r as
a_{k+1} = a_k + eta a_k^{K-1} xi_k with xi = (1/B) sum_b Z_r(x_b) (y_b - f(x_b)),
Z_r = prod_i <u_r^(i), x_i>, so E xi = m(a) = s_r - a^K and
Var xi = V(a)/B, V(a) = m^2 (E Z^4 - 1) + R^2 = m^2 (3^N - 1) + sum_{q != r} s_q^2
(E Z^4 = 3^N: products of Gaussians are heavy-tailed). With f = a^{K-1} m and phi(a) = int da/f (the GF
time), Ito's formula for phi gives for the event a^K = s_r/2:

    E t^SGD - t^GF = (eta/2) int_{a0}^{ae} (f'/f) (1 + V/(B m^2)) da,
    Var t^SGD      = (eta/B) int_{a0}^{ae} V / (m^2 f) da.

The first term of the mean (the "1") is GD's own backward-error shift (section 3.1); noise multiplies
it by E xi^2 / (E xi)^2 = 1 + V/(B m^2), so **SGD delays escapes** (a volatility drag on the
log-growth). Small-a form: shift ~ (eta/2)(K-1)(1 + (3^N - 1)/B) ln(a_e/a0), relative spread
std/t ~ sqrt(eta V / (B t s_r^2)) -> 0 as sigma -> 0 (t ~ sigma^{-(K-2)}). Scaling predictions:
std(t) ~ (eta/B)^{1/2} t^{1/2}, mean shift ~ eta/B (when V/B >> s^2), both in units t = eta x steps; in
steps and samples the plateau costs t/eta steps and B t/eta samples.

**Relation to "SGD lowers the effective init" (Pesme et al. 2021; derived).** The drag can be read as
a smaller effective initialization. For a k >= 3 (higher-order) escape the extra time is additive,
O(eta (V/B) ln(1/a0)), so a0_eff^{-(K-2)} = a0^{-(K-2)} + O(eta (V/B) ln(1/a0)): a negligible change at small
sigma. For a k = 2 (strict-saddle, exponential) escape with rate lambda the log-growth rate drops to
lambda - eta lambda^2 (1 + V/B)/2, i.e. a0_eff = a0^{1 + eta lambda (1 + V/B)/2}: a power-law reduction, the
form found for diagonal networks. So noise matters more for the k = 2 (log-time) plateaus than for
the long k >= 3 plateaus.

**Conservation laws under SGD (derived).** Q_e changes per step by eta^2 (g_v^T g_v - g_p g_p^T); the
expectation of this includes the noise covariance, Cov/B, which differs between the cores on the two
sides of a bond. Q therefore drifts systematically (not as a random walk) by ~ eta t Cov/B: the
noise-driven flow along gauge orbits of Ziyin 2024. Prediction: |Delta tr Q_e| ~ eta/B.

**Stochastic collapse.** *Correction (after `notes/theory_optimizer.md` v2, section 5.4):* my first
version argued that k >= 3 routes cannot collapse because their noise is proportional to other
dormant entries. That is wrong at later saddles: once a component is learned, a new-channel entry
is coupled to a mixed bond entry by a **zero-mean per-sample coefficient of size O(1)** (built from
the learned cores and the residual), so the pair is driven by multiplicative noise that is linear
in the route, and its log-amplitude decays at a constant rate gamma_c ~ c_gamma eta/B while the GF
growth rate is only s_new b. Hence db/dt = s_new b^2 - gamma_c b for a k = 3 route, plateaus are
lengthened by -ln(1-x)/x and the route is trapped for x = c_gamma (eta/B) T_GF >= 1 (theory note P9).
The first plateau is exempt (no learned cores, so every coupling is small). My estimate for
strict (k = 2) escapes stands: the dormant multiplier 1 + eta q xi has Lyapunov exponent
eta q s_new - eta^2 q^2 E xi^2/2, so collapse needs s_new < eta q (V + zeta^2)/(2B); without label noise
(V ~ 3^n s_new^2) that requires eta q 3^n ~ 2B, beyond the SGD stability limit, so k = 2 escapes need
label noise to collapse (components weaker than ~eta q zeta^2/(2B)). Tests: `p9_sgd.py` (k = 3,
section 6), `collapse_test.py` (k = 2 with label noise).

### 4.1 Test of the theory note's P9/P10 (empirical; `p9_sgd.py`, `plot_p9.py`, `results/p9_fit.csv`)

Binary N=4, d = chi = 3, teacher [3, 2, 1]; start on the aligned route (component 1 learned,
components 2 and 3 at per-core amplitude b0), so the GF second plateau is exact: T_GF = 49.7, 166.3,
499.6 for b0 = 0.01, 0.003, 0.001 (exact 1D integral = `integrate_gf` lsoda to 1e-8). Online SGD
(`ttn.dynamics.run_sgd`), 4 noise seeds per point, eta/B = 6.25e-5, 1.25e-4, 2.5e-4 at B = 64
(eta = 0.004-0.016 = 0.08-0.3 eta*_ms) and B = 16 (b0 = 0.01). Event: a_2 = 1/2.

| b0 (T_GF) | eta/B = 6.25e-5 | 1.25e-4 | 2.5e-4 |
|---|---|---|---|
| 0.01 (50), B=64 / B=16 | 1.056 / 1.056 (c 33.6 / 33.5) | 1.130 / 1.129 (35.6 / 35.3) | 1.386 / 1.380 (40.1 / 39.7) |
| 0.003 (166), B=64 | 1.267 (37.5) | 3.06 +- 0.26 (45.1) | trapped (4/4) |
| 0.001 (500), B=64 | trapped (4/4) | trapped (4/4) | trapped (2/2) |

(T_SGD/T_GF, mean over seeds; c = c_gamma from inverting -ln(1-x)/x.) Findings:
- **Lengthening and trapping confirmed** (P9 (i)): no SGD run escaped faster than GF (P9 (iv)); the
  route amplitude sigma_2^{1/3} of trapped runs decays (0.001 -> 5e-4, 3e-5, 1e-6 by t = 1600 at the
  three eta/B).
- **Only eta/B matters** (P9 (ii)): B = 16 and B = 64 at equal eta/B agree to 0.5%.
- **c_gamma = 34-45** for this teacher (heuristic of the note: 26), increasing with x: the law's
  shape -ln(1-x)/x underestimates the lengthening near the trap; the trapping threshold
  (x = c (eta/B) T_GF ~ 1) is consistent with all trapped runs (c >= 32).
- **P10 safety rule fails for long plateaus with this teacher:** at eta/B = 2.5e-4 (0.25-0.3 eta*_ms)
  the learned component dips below 1/2 within T ~ 70 (min a_1 = 0.44-0.46); over 1600 time units it is
  erased even at 0.15 eta*_ms (min a_1 = 0.04, -0.07). Excursions accumulate with run length, and the
  residual noise grows like s^2 while eta*_ms shrinks only like s^{-4/3}, so larger teachers need a
  smaller fraction of eta*_ms (conjecture).

## 5. SGD stability limit (derived)

At an interpolating minimum (realizable teacher, no label noise) the noise vanishes; linear
mean-square stability of e = W - W* under e <- (I - eta Theta Phi_B Phi_B^T) e requires the spectral
radius of M -> M - eta(Theta M + M Theta) + eta^2[(1 - 1/B) Theta M Theta + Theta F(M) Theta / B] to
be < 1, with F(M) = E[Phi Phi^T M Phi Phi^T] = (x)_i F_i, F_i(A) = A + A^T + tr(A) I (Gaussian fourth
moments; F checked by Monte Carlo). Restricted to U_1 this gives eta < 2 / (lambda (1 + (3^N - 1)/B)).
Exact values at the balanced minimum, binary N=4, teacher [3, 2, 1] (`sgd_stability.py`,
`results/sgd_stability.csv`):

| B | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 1024 | inf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eta_ms | 0.00107 | 0.00213 | 0.00423 | 0.00831 | 0.0161 | 0.0300 | 0.0528 | 0.0831 | 0.112 | 0.142 | 0.154 |
| 1D (U_1) estimate | 0.0019 | 0.0038 | 0.0073 | 0.0140 | 0.0257 | 0.0440 | 0.0685 | 0.0948 | 0.117 | 0.143 | 0.154 |

eta_ms = 0.00107 B for B <= 16: **small-batch SGD on product features is ~150x less stable than GD
per sample** (effective kurtosis ~145 = 1.8 x 3^4: other directions feed the noise). For B >~ 3^N
the GD limit returns. Mean-square stability is conservative (heavy tails: almost-sure stability can
hold beyond it); the empirical SGD limit is not yet measured.

## 6. Finite samples and implicit bias (step 3)

Setup (`samples_sweep.py`, `configs/samples_sweep.yaml`): binary N=8, d = 2 (W in R^256), odeco
teacher [3, 2] (rank 2 on every bond; the teacher-rank TTN has 52 parameters and its manifold has
dimension 28 = 52 - 6 x 4 gauge), teacher seed 101; n Gaussian training inputs (data seed 500 + seed),
noise-free labels; 5 seeds per point (seed = data and init). Students: chi = 4 (208 parameters;
tangent space at a random point has dimension 112, measured rank of J) and chi = 16 (full bond
dimension, 832 parameters, tangent space = R^256). Training: GD on the empirical loss with
eta = 1/lambda_max (power iteration every 200 steps; relative step capped at 1% near the origin),
60 000 steps (train loss then ~1e-6 to 1e-3 of its initial value; no run reached the 1e-8 target).
Small init: plain model, sigma = 0.3, 0.5. Large init: centred model (W - W_0 fits W*), sigma = 1.
**Lazy limit:** the exact minimum-norm solution of the model linearized at theta_0 (kernel regression
with the initial tangent kernel, `lazy_limit`), which the centred model approaches as sigma -> inf.
Test error = exact population loss, reported as ||W - W*||^2 / ||W*||^2 (1 = zero predictor).
Effective bond rank = number of bond singular values > 0.1 x 2 = 0.2, averaged over the 6 edges.
Results: `results/samples_summary.csv`, `fig_samples.png`.

| n | small init sigma=0.3, chi=4 | small sigma=0.5, chi=4 | centred sigma=1, chi=4 | lazy limit chi=4 | small sigma=0.5, chi=16 | centred sigma=1, chi=16 | lazy limit chi=16 |
|---|---|---|---|---|---|---|---|
| 128 | 0.79 (median 0.014) | 0.81 (0.17) | 1.25 | 18.5 | 0.57 | 0.99 | 2.35 |
| 192 | 0.13 (0.0063) | 0.35 (0.0040) | 0.098 | 3.4 | 0.55 (0.017) | 0.36 | 1.86 |
| 256 | 1.9e-3 | 1.5e-3 | 0.20 | 3.05 | 0.089 (0.0016) | 0.13 | < 1e-20 (exact) |
| 384 | 1.8e-4 | 1.6e-4 | 0.021 | 2.25 | 1.3e-4 | 0.020 | < 1e-22 |
| 512 | 4.5e-5 | 2.5e-5 | 0.010 | 1.74 | 6.8e-5 | 0.037 | < 1e-22 |

(mean over 5 seeds; standard errors in the CSV.) Findings (all **empirical**, 5 seeds):
- **Small init generalizes from the fewest samples**: test error < 2e-3 at n = 256 and ~1e-4 at
  n = 384 for chi = 4, with effective bond rank exactly 2 (the teacher's) on every edge from n = 256 on;
  the remaining error is the unfinished training (train loss ~1e-6). Both sigma = 0.3 and 0.5 behave
  the same. Below n ~ 128-192 small-init GD interpolates with the wrong low-rank tensor (rank 1.3-3,
  test error ~1-4).
- **The lazy limit cannot fit W* at finite chi** (chi = 4: tangent dimension 112 < 256): test error
  > 1 (worse than predicting zero) for every n up to 512, with a double-descent peak of ~30 near
  n ~ 112, and effective rank 4-5.3 (high-rank, not the teacher's). At full chi = 16 it needs
  n = d^N = 256 samples and then interpolates W* exactly (unique interpolant); below that it is again
  worse than zero (1.9-2.9) because a single draw of the tangent kernel is strongly anisotropic (cf.
  CLAUDE.md, lazy-regime finding), with effective rank up to 6.9.
- **Centred sigma = 1 is not lazy** (parameters move 30-70% relative): it is in between, error
  ~0.01-0.04 at n = 384-512, rank 2.4-2.9: feature learning with a weaker low-rank bias.
- **Sample complexity of the low-rank bias (conjecture):** small-init GD needs n ~ 200-256, about
  7-9x the 28-dimensional teacher manifold and close to d^N = 256 here. Proposed reason: the first escape
  follows the empirical correlation tensor (1/n) sum_j y_j Phi(x_j), whose noise per direction is set
  by the heavy tails of product features, E[y^2 |Phi|^2] / ||W*||^2 = (1 + 2/d)^N = 256 for d = 2,
  N = 8, so n must exceed ~(1 + 2/d)^N before the escape direction is right. For d = 2 this equals
  d^N, so the advantage over the lazy regime is only a constant here; for d >= 3 it should grow
  exponentially in N ((5/3)^N vs 3^N). Not yet tested.
- **Effective ranks (the implicit bias, directly):** at every n the learned tensor's bond rank is
  ordered small init (1.3-3.9) < centred sigma = 1 (2.4-5) < lazy limit (4-6.9); only small init lands
  on rank 2 once the data determine W*.

## 7. Synthesis: the learning-rate axis and when gradient-flow results apply (draft)

For small (saddle-to-saddle) init, with s_1 the largest teacher weight and K the number of cores:

| step size (GD) | what happens | status |
|---|---|---|
| eta << eta_max = 2/(K s_1^{2-2/K}) | GF: same saddles, same order, event times shifted by eta c_e (exact first-order formula), relative shift ~ eta ln(1/sigma) / t ~ eta sigma^{K-2} ln(1/sigma) | proven (sharpness), derived + empirical (shift) |
| up to 0.95 eta_max | still GF-like; the only discrete change is a rare flip of near-tied races (2-9% of inits at 0.9 eta_max, 0% at 0.1 eta_max), which can remove a whole plateau | empirical |
| eta_max < eta < ~1.3 eta_max | the top component oscillates (2-cycle, then period 4 at 1.245); the others are learned at GF-like times; the loss floors at s_1^2 L_map(eta/eta_max) | derived (1D map) + empirical (to 3 digits) |
| ~1.5-1.75 eta_max (K = 3) | diverges | empirical |
| ~2 eta_max (K = 3; 1.75-2.5 for K = 7, 8) | learned components are thrown back to the origin saddle and re-learned repeatedly | empirical (1D map: re-entry) |
| >= 2.5-3 eta_max | diverges | empirical |

For imbalanced (or large, non-small) inits the GF minimum is sharper than the balanced one, and GD
with 2/lambda_GF < eta < eta_max converges to a flatter, more balanced minimum at sharpness ~2/eta (EoS
selection).

**Across the lazy transition (centred model, binary N=4, sigma = 0.3-2, 5 seeds x 11 step sizes,
`configs/gd_centred.yaml`, `fig_lr_regime_map.png`; empirical).** The GD limit is 2/lambda_GF(sigma),
lambda_GF = sharpness of the GF state reached (for sigma >= 1 the state at the end of the run, T = 2-5,
not yet converged): predicted eta_max(sigma)/eta_max(balanced) = 1.00, 0.99, 0.97, 0.88, 0.57, 0.19 for
sigma = 0.3, 0.5, 0.7, 1.0, 1.4, 2.0; observed converge/fail boundaries 0.9-1.05, 0.9-1.05, 0.9-1.05,
~0.9 (3/5 converge), 0.5-0.7, ~0.2 (3/5 at 0.2). The GF sharpness grows like sigma^{2(K-1)} = sigma^4 in the
lazy regime (init sharpness 15.5 -> 61.6 from sigma = 1.4 to 2, ratio 4.0 vs 1.43^4 = 4.2). Above the
limit, small-init runs oscillate boundedly, while lazy-side runs (sigma >= 1.4) are thrown back or
diverge: there is no bounded EoS regime in the lazy corner. The learned bond rank of the GF state is
the teacher's (3) for sigma <= 0.5 and 4-6 for sigma >= 1. So **the stability limit is set by the
teacher in the saddle-to-saddle regime and by the init scale (sigma^{-2(K-1)}) in the lazy regime**,
and a fixed eta that is safe at small init becomes unstable as sigma grows past ~s_1^{1/K}.

Online SGD with batch B at small step: the first plateau keeps its GF law (P7; mean delay
~ (eta/2)(K-1)(1 + kappa/B) ln(a_e/a_0), spread ~ sqrt(kappa (eta/B) t)); later degenerate plateaus are
lengthened by -ln(1-x)/x and trapped for x = c_gamma (eta/B) T_GF >= 1 (P9; c_gamma ~ 33-37 measured for
binary N=4, teacher [3, 2, 1]); the step must also stay below the SGD stability limit
eta_ms(B) ~ 0.001 B (binary N=4, B <= 16), ~150x below GD per sample, because product features have
kurtosis 3^N.

**Practical rule (derived from the above).** GF results describe GD training whenever
eta < eta_max (event times to within eta c_e). They describe SGD training only when, in addition,
eta/B << 1/(c_gamma T_plateau) for every later plateau one wants to cross (otherwise SGD stalls on it),
and eta << eta_ms(B). Because plateau lengths grow like sigma^{-(k-2)}, at small init and fixed eta/B
SGD will eventually stall where GF would only be slow.
