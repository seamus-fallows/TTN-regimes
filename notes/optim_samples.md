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
h/h_c = 1.84 / 1.99 / 2.23 / 2.29 and diverges from 2.16 / 2.41 / 2.72 / 2.86 for K = 3 / 4 / 7 / 8
(first occurrence on a grid of step 0.005; the regimes interleave: above the first divergence, 30
grid points (K = 3) and 45 (K = 7) still end in re-entry rather than divergence).
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

**Empirical check (fix round 1; persistent events, `analyze_shift.py`, `results/shift_accuracy*.csv`,
`fig_shift_accuracy.png`).** Over all runs at eta <= 0.1 eta_max (binary N=4 with 5-45 seeds per
point, binary N=8 and MPS N=8 with 5 seeds; 526 event comparisons), the measured shift/eta agrees with
c_e within 0.3% in 66%, within 1% in 90% and within 5% in 97% of the comparisons.
- First events: all 202 within 1% (max 0.69%); median relative deviation 3e-4 at 0.01 eta_max, 3e-3
  at 0.1 eta_max, 1-3% at 0.9 eta_max (max 9%). The deviation is linear in eta, as an O(eta^2)
  correction must be.
- Later events: mostly as accurate, but the relative deviation is large where c_e ~ 0 (the O(eta^2)
  term then dominates): e.g. binary N=4, sigma = 0.03, seed 1, event 2: -0.050 measured at 0.1 eta_max
  vs c_e = -0.147 (-0.119 at 0.03 eta_max); 14 of 526 comparisons deviate by more than 5%.
- **c_e is the exact eta -> 0 limit:** a Richardson extrapolation from the 0.01 and 0.03 eta_max runs
  agrees with c_e to median 1.2e-5 and max 1.8e-4 relative (57 events; the review found median 4e-5
  with independent code).
- The round-1 statement "3 digits per seed for every event at eta <= 0.1 eta_max, within 2% up to
  0.9 eta_max" was too strong; the numbers above replace it. (The round-1 figure plotted medians with
  a clipped y-range; `fig_shift_accuracy.png` shows every run.)
**Scaling (derived + empirical):** relative error of t_1 = eta c_1 / t_1 with c_1 ~ ln(1/sigma) and
t_1 ~ sigma^{-(K-2)}: binary N=4 at 0.9 eta_max gives 0.111, 0.056, 0.023, 0.0099, 0.0036 for
sigma = 0.3, 0.1, 0.03, 0.01, 0.003 (`fig_gd_vs_gf.png` a). So **every stable step size reproduces the
GF event times to within a few percent at small init, and the agreement improves as sigma -> 0.**

### 3.2 Saddle sequence below eta_max (empirical)

**GD below eta_max kept the GF saddle sequence in every checked case.** With persistent events
(last upward crossing of a_r = 1/2, notes/diagnostics.md), the learning order of GD equals GF's in
765/765 runs below eta_max: binary N=4 at sigma = 0.003-0.3 (675 runs: 5 seeds x 18 steps, plus 40
more seeds at 0.1, 0.3, 0.5, 0.9 eta_max for sigma = 0.03, 0.1, 0.3), binary N=8 and MPS N=8 at
sigma = 0.9 (45 runs each). What GD does change is the size of **transient excursions**: in a near-
tied race the eventually-second component rises during the first event and falls back (GF already
shows a transient crossing of 1/2 in 0/5, 0/5, 1/45, 4/45, 7/45 inits at sigma = 0.003, 0.01, 0.03,
0.1, 0.3). With GD the fraction of inits with a transient crossing grows with eta: at sigma = 0.1 from
0.09 +- 0.04 (0.1 eta_max) to 0.18 +- 0.06 (0.9 eta_max); at sigma = 0.03 from 0.02 to 0.07 (45 inits
each); the component is then learned at its GF time (e.g. sigma = 0.1 seed 1 at 0.9 eta_max: transient
crossing at t = 7.97, persistent event at 23.53 vs GF 23.24). `analyze_order.py`,
`results/order_flips.csv`, `fig_order_flips.png`.

*Withdrawn (fix round 1):* the round-1 claim that "an O(eta) step flips near-tied races and can remove
a whole plateau", and the conjecture that the flip probability vanishes as sigma -> 0. Both came from
first-crossing events that counted these transients as events (the review's per-step reruns and
this round's persistent events show no plateau is ever removed).

### 3.3 The stability limit (empirical, confirms section 1)

Classes (`analyze_gd.py`, fix round 1): "GF's level" = final loss <= 1.05 L_GF(T) + 1e-3 L(0)
(L_GF(T) = the GF reference at the same time). GF-like = GF's level with a loss that never rises by
more than that tolerance; catapult = GF's level after a loss rise; thrown back to origin (all
components lost after being learned); bounded oscillation (tail max above GF's level); diverged;
other (any other final loss). Runs: 5 seeds per point (seeds 0-4); events persistent.

| tree, sigma | GF-like | catapult | oscillates (bounded) | thrown back to origin | diverges |
|---|---|---|---|---|---|
| binary N=4, sigma 0.003-0.3 (25 runs per ratio; 15 at 0.01/0.03/0.2/1.1/1.3/1.75/2.5/3 eta_max) | all 195 at <= 0.95 eta_max (675/675 incl. the 40 extra seeds) | none | all at 1.05-1.3 | 2/25 at 1.5, 25/25 at 2.0 | 23/25 at 1.5, all at 1.75, 2.5, 3.0 |
| binary N=8, sigma = 0.9 | all at <= 0.7, 3/5 at 0.9, 0/5 at 0.95 | 1/5 at 0.9, 2/5 at 0.95, 1/5 at 1.05 | 1/5 at 0.9, 3/5 at 0.95, 4-5/5 at 1.05-1.5 | 4/5 at 1.75-2.0, 3/5 at 2.5 | 1/5 at 2.5, 4/5 at 3.0 |
| MPS N=8, sigma = 0.9 | all at <= 0.7, 2/5 at 0.9, 1/5 at 0.95 (+ seed 3 at 1.05-1.75, see below) | 3/5 at 0.9-0.95, 1/5 at 1.05 | 1/5 at 0.95, 3-4/5 at 1.05-1.5 | 3/5 at 1.75-2.5 | 3/5 at 3.0 |

- For small init the boundary is exactly eta_max = 2/(K s_1^{2-2/K}): 195/195 runs (seeds 0-4; 675/675
  with the order seeds) at <= 0.95 eta_max are GF-like, 25/25 at 1.05 oscillate. **Empirical, 5 sigma
  x 5 seeds**; the independent review's own code finds the edge between 0.99 and 1.01 eta_max. Note:
  "GF-like" includes runs that end on GF's last plateau at T (39/195 at L/L_0 = 0.071, component 3
  not yet learned in either GF or GD).
- N = 8 trees at sigma = 0.9: this is not small init. The GF state is 10-16% (binary N=8) and 12-35%
  (MPS N=8) sharper than balanced (fan-in imbalance), so the edge moves to ~2/lambda_GF ~ 0.75-0.9
  eta_max, with EoS transients (catapults and slow oscillations at 0.9-0.95 eta_max). Most N = 8 GF
  references learn only 1-2 of the 3 components by T. MPS N=8 seed 3 never learns component 1 within
  T = 60 (GF loss 5.0 of 7.06), so its "GF-like" runs above eta_max (0.95-1.75) never meet the
  instability; it is excluded from the oscillation-loss comparison.
- Above eta_max the outcome follows the 1D map qualitatively (2-cycle, then period doubling, then
  re-entry to the origin, then divergence) but **the bounded chaotic window of the 1D map
  (h/h_c ~ 1.35-1.8) is not robust in the full network**: binary N=4 diverges at 1.5-1.75, while at
  2.0 every run is thrown back to the origin saddle (as the 1D map predicts) and repeatedly
  re-learns and loses components (`fig_eos_example.png`). Conjecture: transverse (imbalance)
  directions are unstable on the chaotic set (cf. Hofmann 2026, imbalance factor 1 - a^2 r^2).
- **Oscillation loss (P5 of the theory note; empirical, exact 1D comparison; `analyze_p5.py`,
  `results/p5_excess_loss.csv`).** The time-averaged excess loss of oscillating runs, L_tail - L_GF(T),
  equals s_1^2 x (mean loss of the 1D map attractor, `map1d.py`) on binary N=4 to within 0.9%:
  0.1306 vs 0.1306 (1.05 eta_max, 25 runs), 0.2472 vs 0.2479 (1.1, 15), 0.4496 vs 0.4496 (1.2, 25),
  0.553 vs 0.557 (1.3, 15). Binary N=8 (5 runs per point): within 1% at 1.05-1.1 eta_max but 14% and
  32% low at 1.2 and 1.3 (some seeds leave the 2-cycle). MPS N=8 (4 runs, seed 3 excluded): 17% high
  at 1.05, within 3% at 1.1-1.3. The theory note's linear form 3K^2 s_1^2/(4(K-1)(5K-4))
  (eta/eta_max - 1) is the small-excess limit of the map (5% high at 1.05, 23% at 1.2, 49% at 1.3).
- Oscillation stays in the learned top component(s): at 1.05-1.3 eta_max component 1 oscillates
  (2- or 4-cycle) while components 2 and 3 are learned at GF-like times (fig_eos_example, middle).
  Component-wise threshold s_c(eta) (section 1) explains which components oscillate.

### 3.4 Which minimum GD selects, and balance (empirical; cf. Ghosh 2025, Kreisler 2023)

- Small (balanced) init: the GF minimum is the flattest, final sharpness = lambda_bal for every
  stable eta. GD *creates* a small imbalance that grows linearly with eta (|tr Q_e| from 0.016 (GF)
  to ~0.08 at 0.95 eta_max, sigma = 0.1), because Q_e changes by O(eta^2) per step.
- Imbalanced init (root core scale 2, 4, 6; GF minimum 1.13x, 1.68x, 2.37x sharper than balanced):
  GD converges to sharpness min(lambda_GF, 2/eta) (`fig_min_selection.png` a; to within 1-2% for the
  converged runs), i.e. it selects a flatter, more balanced minimum (|tr Q_e| down to ~0.6x GF) as soon
  as 2/eta < lambda_GF: classic EoS selection. These runs are classified "catapult" (the loss rises
  before converging): root x2: 2-3/5 at 0.9-0.95 eta_max; root x4: 1-4/5 from 0.5 eta_max; root x6:
  1-3/5 from 0.5 eta_max. Divergence starts at 0.6 eta_max for root x6 (1/5; 2-5/5 from 0.7) and at
  0.95 eta_max for root x4 (1/5).
- At the stability edge the imbalance is removed fastest: imbalanced inits that still converge at
  eta = eta_max end with |tr Q_e| ~0.35-0.4x the GF value; for small init, oscillating runs at
  1.05-1.3 eta_max end below the GF value (0.0139 vs 0.0162 at sigma = 0.1), as for DLNs (Ghosh 2025;
  Wang et al. 2022).

### 3.5 Across the lazy transition: 2/lambda_GF and the catapult window (centred model, chi = 9)

*Withdrawn (fix round 1):* the round-1 statement that the GD limit is 2/lambda_GF(sigma) across the
lazy transition, falling like sigma^{-2(K-1)} (0.19 eta_max at sigma = 2), and that on the lazy side
steps above it never oscillate but re-enter or diverge. Those runs used chi = 3, while the centred
target W* + W_0 has bond rank up to 6 (3 from W_0, 3 from W*) > chi: their GF references stopped at L/L_0 = 0.06-0.55
(sigma = 1.4-2) and were **stalled** (diagnostics.md), not lazy, and "GF-like" ignored the final loss.
They remain in `gd_sweep.csv` (centred, chi = 3) but are not used.

Setup (`configs/gd_centred_chi9.yaml`, `analyze_centred.py`, `results/centred_chi9_*.csv`,
`fig_centred_chi9.png`): binary N=4, d = 3, teacher [3, 2, 1], **chi = 9 = the cut dimension d^2**, so
every target is representable. Centred model (W(theta) - W(theta_0) fits W*), fan-in init, seeds 0-4.
GF references (LSODA, stopped at loss 1e-8): sigma = 1, 1.41, ..., 16 (factor sqrt 2); **all 45
converge** (loss <= 1e-8; GF time 2900 at sigma = 1 down to 2.5 at sigma = 16). lambda_GF = top
Hessian eigenvalue at the GF endpoint. GD at eta = ratio x 2/lambda_GF of the same init (ratios
0.5-4), run as long as GF took to converge, at centred sigma = 1, 2, 4, 8 and, for comparison, the
plain model at small init sigma = 0.1, 0.3 with chi = 9 (GF references there end at loss 1.6e-8 and
3e-8 at the 10^4 cap: the surplus channels decay slowly; lambda_GF = 1.0001 and 1.004 lambda_bal).
Classes as in section 3.3 (GF's level = 1.05 L_GF + 1e-3 L_0).

- **Sharpness of the converged GF minimum (derived scaling + empirical).** lambda_GF / lambda_bal =
  1.53 (sigma = 1), 14.4 (2), 228 (4), 3650 (8), 58 000 (16). Per-seed power-law fits over
  sigma = 1.41-16 (1.05 decades): exponent **3.965 +- 0.014** (5 seeds), local slopes 3.76 (1.41-2)
  then 3.97-4.00; the lazy prediction is 2(K-1) = 4 (tangent kernel at init ~ sigma^{2(K-1)}, and on
  the lazy side the minimum is next to theta_0). So 2/lambda_GF = eta_max at small init (sigma <= 0.3),
  0.65 eta_max at sigma = 1, and falls like sigma^{-4} beyond: 0.069, 0.0044, 2.7e-4 eta_max at
  sigma = 2, 4, 8.
- **Below 2/lambda_GF: monotone, GF-like GD at every sigma** (all runs at 0.5-0.95 x 2/lambda_GF, all
  six init scales, final sharpness = lambda_GF).
- **Small init (plain sigma = 0.1, 0.3): no catapult.** Bounded oscillation at 1.05-1.2 (10/10 each),
  then divergence (and re-entry at 2.0 for sigma = 0.1), exactly as for chi = 3 (section 3.3). The GF
  minimum is already the flattest one (section 1), so there is nothing flatter to catapult to.
- **Larger init: a catapult window above 2/lambda_GF** (cf. Lewkowycz et al. 2020). The loss rises
  (at sigma = 8 by up to 1e5 L_0) and GD then converges to **GF's loss** at a flatter minimum, final
  sharpness x eta/2 = 0.75-0.97 (i.e. sharpness 1.5-1.94/eta; `centred_chi9_selection.csv`):
  sigma = 1: 1/5 at 1.05 (4/5 reach a flatter minimum without a visible rise), 5/5 at 1.2, 2/5 at 1.5
  (3/5 oscillate), then divergence (3-5/5 from 2.0; 1/5 re-entry at 2.0 and 2.5);
  sigma = 2: 3/5 at 1.05 (2/5 GF-like), 5/5 at 1.2-2.0, 4/5 at 2.5-3.0, all diverge at 4;
  sigma = 4: 5/5 at 1.05-2.5, 2/5 at 3.0 (3/5 diverge), all diverge at 4;
  sigma = 8: 5/5 at 1.05-2.5, 4/5 at 3.0, all diverge at 4.
  So the window widens from ~1.5x (sigma = 1) to ~3x 2/lambda_GF (sigma >= 2), and divergence becomes
  general between 3 and 4 x 2/lambda_GF on the lazy side (Lewkowycz et al. find 4/lambda_0 for their
  two-layer model). **Empirical** (4 init scales, 5 seeds, 10 ratios).
- **Reframed claim:** 2/lambda_GF(sigma) is the end of *monotone, GF-like* GD at every init scale. On
  the small-init (saddle-to-saddle) side it is also the end of convergence (bounded oscillation, then
  divergence); on the lazy side GD still reaches GF's loss up to ~3 x 2/lambda_GF, by catapulting to a
  flatter minimum, so its trajectory (and its minimum) differs from GF while the final loss does not.

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
log-growth). Small-a form: shift ~ (eta/2)(K-1)(1 + kappa/B) ln(a_e/a0) with kappa = 3^N - 1 +
sum_{r>=2} s_r^2/s_1^2 (= V/s_1^2 at a = 0; fix round 1: the R^2 term was dropped), relative spread
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
(V ~ 3^n s_new^2) that requires eta q 3^n s_new ~ 2B (fix round 1: s_new was missing), beyond the
SGD stability limit, so k = 2 escapes need
label noise to collapse (components weaker than ~eta q zeta^2/(2B)). Tests: `p9_sgd.py` (k = 3,
section 6), `collapse_test.py` (k = 2 with label noise).

### 4.0 Step-2 sweep results (empirical; `sgd_sweep.py`, `analyze_sgd.py`, `fig_sgd.png`)

Setup (fix round 1 rerun): binary N=4, d = chi = 3, odeco [3, 2, 1] (teacher seed 101), online SGD
(fresh Gaussian inputs, no label noise), overlaps checked every 4 steps (resolution 0.002 time
units), **persistent events** and every run continued to 1.3 x (later of the first two GF events) + 2
time units; **independent noise generators** for every (init, noise seed, B, eta). Blocks: B in
{1, 2, ..., 256} at eta = 5e-4 (= 0.47 eta*_ms at B = 1, decreasing fractions at larger B) for fan-in
sigma = 0.1, 0.3 (5 inits x 10 noise seeds) and sigma = 0.03 (B in {1, 4, 16, 64, 256}); eta sweeps at
B = 4 (eta = 1e-4 ... 2e-3) and B = 64 (eta = 5e-4 ... 2e-2), sigma = 0.1; aligned balanced init
(per-core amplitude 0.05, 20 noise seeds). 1780 runs. GF references: the step-1 DOP853 runs.
Per run, before pooling: t_esc = first time any component reaches 1/2 (escape from the origin);
"forgot" = the first-learned component later falls below 1/2 (P10 erasure); persistent order vs GF.
Pooled numbers are means over the 5 inits with the standard error ACROSS inits.

- **First escape, mean (P7):** no significant shortening at any setting. Delays at B = 1:
  +0.24 +- 0.10 (sigma = 0.03, t_esc = 24.2), -0.01 +- 0.12 (sigma = 0.1, 7.2), +0.13 +- 0.07
  (sigma = 0.3, 2.3) vs P7's (eta/2)(K-1)(1 + kappa/B) ln(a_e/a0) (evaluated with each init's GD
  coefficient c_1) +0.17, +0.12, +0.08: consistent within 1.3 s.e., but only resolved as positive at
  sigma = 0.03 (2.5 s.e.). For B >= 2 all delays are below ~0.1 and mostly within 2 s.e. of zero.
  Aligned init (exact 1D theory): +0.07 vs +0.11 at B = 1. *Correction:* the round-1 report quoted a
  single init's s.e. ("+-0.41/0.24/0.21") and first-crossing component events; the review then read
  the B = 1 delays as 1.6-3.4x P7. With the escape-time definition and s.e. across inits, P7's
  magnitude is consistent but only weakly tested (the delays are ~1-3% of t_esc).
- **First escape, spread (P7):** std(t_esc) ~ (eta/B)^p with per-init fits p = 0.524 +- 0.024
  (sigma = 0.03), 0.434 +- 0.020 (0.1), 0.491 +- 0.007 (0.3) over the B sweep (2.4 decades in eta/B),
  0.491 +- 0.042 (eta sweep, B = 4), 0.479 +- 0.046 (eta sweep, B = 64); predicted 1/2 (sigma = 0.1 is
  3 s.e. low). Magnitude: per init and setting 0.4-2.2x the 1D small-a formula sqrt(kappa (eta/B) t)
  (typically 1.0-1.3x); aligned init 0.77-1.28x the exact aligned formula.
- **Forgetting (P10 in random-init runs; new in round 2):** at eta = 0.47 eta*_ms (B = 1, and B = 4
  at eta = 2e-3) the learned component is erased and re-learned within the run in 66% (sigma = 0.03,
  longest runs), 26% (sigma = 0.1), 20% (sigma = 0.3) and 22% (B = 4) of the runs; 4% at 0.38 eta*_ms
  (B = 64, eta = 0.02); 0% at <= 0.24 eta*_ms (all other settings). Round 1 stopped right after the
  second crossing and missed this.
- **Order:** persistent order differs from GF in 4-8% of runs at B = 1 only (0 elsewhere).
- **Second plateau (P9), inits with a resolved GF plateau (rho > 3 with final-rise widths) and no
  near-tie, same-order runs that never forgot** (sigma = 0.03: 3 inits; sigma = 0.1: 3 inits; one
  sigma = 0.3 init, not used): lengthened by +19.8 +- 5.1% (sigma = 0.03, B = 1; P9 with c = 35: +19%),
  +3.6 +- 1.1% (B = 4; +4.0%), +10.8 +- 0.6% (sigma = 0.1, B = 1; +5-6%), +11 +- 3% (sigma = 0.1, B = 4,
  eta = 2e-3; +5-6%), and within ~1-2 s.e. of 0-2% at eta/B <= 1.25e-4. Consistent with P9 with c_gamma
  ~ 35 at sigma = 0.03 and ~1.5-2x larger at sigma = 0.1 near 0.5 eta*_ms; never shorter than GF beyond
  1-2 s.e. (s.e. across 3 inits).
- **Gauge balance (Ziyin 2024, derived scaling):** |Delta tr Q_e| at the end grows like (eta/B)^p with
  per-init p = 0.902 +- 0.031, 1.002 +- 0.011, 1.02 +- 0.013 (sigma = 0.03, 0.1, 0.3), 0.98 +- 0.03,
  1.01 +- 0.01 (eta sweeps); predicted 1 (`fig_sgd.png` d).
- No divergence in any run (all eta <= 0.47 eta*_ms).

### 4.1 Test of the theory note's P9/P10 (empirical; `p9_sgd.py`, `plot_p9.py`, `results/p9_*.csv`)

Binary N=4, d = chi = 3, teacher [3, 2, 1]; start on the aligned route (component 1 learned,
components 2 and 3 at per-core amplitude b0), so the GF second plateau is exact: T_GF = 49.7, 99.7,
166.3, 249.7 for b0 = 0.01, 0.005, 0.003, 0.002 (exact 1D integral = `integrate_gf` lsoda to 1e-8).
Online SGD (`ttn.dynamics.run_sgd`), **6 independent noise seeds per point** (fix round 1: round-1
runs at B = 16 and 64 shared generator seeds and were pathwise coupled; they are kept in
`results/p9_sgd_round1.csv`), eta/B = 3.125e-5 ... 2.5e-4 at B = 64 (eta = 0.002-0.016 = 0.04-0.30
eta*_ms), and at B = 16 for b0 = 0.01, 0.003. Event: persistent crossing of a_2 = 1/2; runs up to
4 T_GF (cap 1600). R = T_SGD/T_GF, mean +- s.e. over the escaped seeds; z = (eta/B) T_GF.

| b0 (T_GF) | eta/B = 3.1e-5 | 6.25e-5 | 1.25e-4 | 2.5e-4 |
|---|---|---|---|---|
| 0.01 (50), B = 64 / 16 | 1.027 +- 0.001 / 1.024 +- 0.003 | 1.048 +- 0.003 / 1.052 +- 0.004 | 1.131 +- 0.017 / 1.124 +- 0.005 | 1.45 +- 0.07 / 1.38 +- 0.03 |
| 0.005 (100) | 1.048 +- 0.002 | 1.118 +- 0.006 | 1.372 +- 0.011 | trapped 6/6 |
| 0.003 (166), B = 64 / 16 | 1.090 +- 0.001 / 1.087 +- 0.001 | 1.250 +- 0.008 / 1.237 +- 0.004 | 2.69 +- 0.15 / 2.45 +- 0.17 | trapped 12/12 |
| 0.002 (250) | 1.153 +- 0.003 | 1.547 +- 0.015 | trapped 6/6 | trapped 6/6 |

- **Lengthening and trapping (P9 (i), (iv)):** every escaped run is slower than GF; 30/30 runs at
  z >= 0.025 are trapped (no escape by 4 T_GF).
- **Only eta/B matters (P9 (ii)), now with independent noise:** B = 16 and B = 64 agree within their
  standard errors at all 8 matched points (largest gap 2.69 +- 0.15 vs 2.45 +- 0.17).
- **Shape of the law:** one c_gamma for all 19 escaped conditions, weighted least squares in R:
  P9 (-ln(1-x)/x) c_gamma = 32.6, chi^2 = 379 (18 dof); pole law 1/(1-x): c = 16.9, chi^2 = 535; linear
  1 + x/2: chi^2 = 929. The P9 shape fits best and its threshold x = 1 classifies 23/24 conditions
  correctly (all 19 escaped; 4/5 trapped; b0 = 0.005 at eta/B = 2.5e-4 traps at x = c z = 0.81), but
  chi^2/dof ~ 21: the measured lengthening rises more steeply near the trap than -ln(1-x)/x (fitted
  c_gamma grows from ~30 at small x to ~45 near x = 1). So the law is right qualitatively and to ~10%
  for R <~ 1.5, not exact. c_gamma ~ 33 vs the theory note's heuristic 26.
- **P10 (learned component under noise), counts per step-size fraction:** min a_1 < 1/2 in 0/72 runs
  at <= 0.076 eta*_ms, 1/36 at 0.12-0.15 eta*_ms (a 1000-time-unit run), 20/36 at 0.25-0.30 eta*_ms
  (6/12 at 0.25, B = 16; 14/24 at 0.30, B = 64). Excursions grow with run length; most components are
  relearned (a_1 ~ 1 at the end). So P10's "safe at <= 0.25 eta*_ms" fails for this teacher on long
  plateaus; ~0.1 eta*_ms was safe over <= 1000 time units. The trapped conditions at eta/B = 2.5e-4
  overlap with these erasures (19/24 runs dip: 6/6, 10/12, 3/6 for b0 = 0.002, 0.003, 0.005), so the
  clean trapping evidence is b0 = 0.002 at
  1.25e-4 (6/6 trapped, 1/6 dips) and b0 = 0.005, 0.003 at 2.5e-4 only partly.

### 4.2 Stochastic collapse of a k = 2 route (empirical; `collapse_test.py`, `check_k2_saddle.py`)

MPS N=4 (K = 4), d = chi = 3, pattern teacher [3, 0.3] with patterns [[0,0,0,0],[1,1,0,0]] (component 2
needs a second channel on the bond between cores 0 and 1 only). **Saddle type by the sweep-level rule
of `notes/diagnostics.md`** (recomputed in fix round 1 after the library's gauge-projection fix): Hessian
at the GF second-plateau midpoint (dense grid around the events), sigma = 0.1, 0.07, 0.05, 0.03, 0.02,
0.01, 5 seeds (2 at sigma = 0.01 unresolved: no plateau midpoint found). **Raw** lambda_min (the value to
quote for these states, diagnostics.md): -0.4871 +- 0.0089, -0.5027 +- 0.0047, -0.5105 +- 0.0026,
-0.5161 +- 0.0010, -0.5179 +- 0.0005, -0.5190 +- 0.0001, tending to the exact saddle value
-s2 s1^{(K-2)/K} = -0.5196: a sigma-independent negative constant, i.e. **a strict saddle**. The
gauge-projected value with the library default (rtol = 1e-3) stays biased here (-0.503 at sigma = 0.01:
near-null generators above the cut; with rtol >= 0.03 it equals the raw value), as the library notes
warn for saddle-start states at sigma ~ 1e-2; both are in `results/k2_saddle_sweep.csv`
(`check_k2_saddle.py`). Online SGD, B = 1, eta = 5e-4 (0.47 eta*_ms), 5 noise seeds,
label noise zeta in {0, 10, 20, 30}:
- zeta = 0: component 2 learned in 5/5 runs, second plateau 5.11 +- 0.25 vs GF 4.73 (+8 +- 5%): no
  collapse, as predicted (collapse would need s2 > ~30 here).
- zeta = 10, 20, 30: the learned component 1 is erased in 15/15 runs (min a_1 after learning <= 0.05)
  and the overlaps wander (a_2 up to 2-8): label-noise SGD at this step does not hold the learned
  block, so collapse of the route cannot be assessed. **Inconclusive** for the label-noise
  prediction (collapse for s2 < ~eta q zeta^2/2); a test would need a much smaller eta with
  correspondingly larger zeta, i.e. components far below the label-noise level.
- A first version of this test used binary N=4, where this pattern lies inside one core (k = 1;
  both components learned together at t ~ 5); kept in `/root/outputs/optim_samples/collapse_binary4_k1/`.

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

MPS N=4 (K = 4, lambda = 20.78): eta_ms = 0.000765, 0.00152, 0.00301, 0.00590, 0.0113, 0.0209, 0.0357,
0.0541, 0.0708, 0.0890 for B = 1 ... 1024 (GD 0.0962), equal to the theory note's P8 values.

eta_ms / B = 0.00107 at B = 1, falling to 0.00100 at B = 16 (and 0.00082 at B = 64): **small-batch SGD
on product features is ~150x less stable than GD per sample** (effective kurtosis ~145 = 1.8 x 3^4: other directions feed the noise). For B >~ 3^N
the GD limit returns.

**Empirical limit (`sgd_limit_test.py`, `results/sgd_limit_test.csv`; 5 seeds per point, start 1e-5
off the balanced minimum, run length 25 time units).** At B = 1 and at B = 16 alike: converged in 5/5
runs up to 1.5 eta*_ms, 2/5 at 2.0 eta*_ms (the other 3 neither converge nor diverge), diverged (9/10)
at 2.5 and 3.0 eta*_ms. Mean-square stability is conservative by a factor ~2, as the theory note's
almost-sure threshold predicts (P8: eta_as / eta*_ms = 1.93 at B = 1, 1.87 at B = 16). **Empirical.**

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

(mean over 5 seeds; standard errors in the CSV; the exact lazy limit is counted once per init, fix
round 1: it was triplicated for chi = 4.) Scope: one teacher ([3, 2], teacher seed 101), one tree,
noise-free labels, GD with capped steps (no run met the 1e-8 train-loss target). Findings (all
**empirical**, 5 seeds):
- **Small init recovers W* with the teacher's ranks from n ~ 256**: test error 1.5-1.9e-3 at n = 256
  and ~1e-4 at n = 384 for chi = 4 (sigma = 0.3 and 0.5 within a factor 2 of each other, not ordered),
  effective bond rank 2.0 (sigma = 0.5) and 2.1 (sigma = 0.3) at n = 256, exactly 2.0 from n = 384; the
  remaining error is unfinished training (train loss ~3e-6-1e-5 of initial). Below n ~ 128-192
  small-init GD interpolates with a wrong lower- or higher-rank tensor (rank 1.3-3.9, test error ~1-4).
- **The lazy failures are what dimension counting predicts.** chi = 4: the tangent space at a random
  init has dimension 112 < 256, W* is not in it, so the lazy limit cannot reach W* at any n; observed
  test error 1.7-30 (worse than the zero predictor; double-descent peak near n ~ 112), rank 4-5.3.
  chi = 16: tangent space = R^256, so the min-norm interpolant is unique, and exact, only once
  n >= d^N = 256 (observed: < 1e-20 at n >= 256); below that the anisotropic single-draw kernel gives
  test error 1.9-2.9 on average (per seed 0.64-5.8; two single seeds below 1: 0.79 at n = 128, 0.64 at
  n = 192), rank up to 6.9.
- **Centred sigma = 1 is not lazy** (parameters move 30-70% relative): error ~0.01-0.04 at n = 384-512,
  rank 2.4-2.9 (train loss ~5e-4-1e-3 of initial).
- **Centred sigma = 1.5, 2 do not converge:** after 60 000 steps the train loss is still 0.3-5%
  (sigma = 1.5) and 4-16% (sigma = 2) of its initial value at n = 512; training 5x longer (300 000 steps,
  `chi4_centred_long`) lowers it only to 1.4% and 7.4% (means) and the test error from 0.53 to 0.36
  (sigma = 1.5) and from 1.51 to 1.43 (sigma = 2). Their test error therefore mixes incomplete fitting
  with implicit bias and is not used to rank init scales.
- **Recovery fraction** (test error < 1e-2; same picture at 1e-3 and 1e-1): small init 0-1/5 seeds up
  to n = 128, 2-4/5 at n = 192, 5/5 from n = 256 (chi = 4) or n = 384 (chi = 16); lazy limit 0/5 at
  every n for chi = 4 and 5/5 from n = 256 for chi = 16; centred sigma = 1 at most 2/5 for chi = 4 up to
  n = 512.
- **Implicit-bias comparison (narrowed):** among runs that fit the data (train loss <= 1e-3), the
  saddle-to-saddle regime (small init) generalizes from the fewest samples and is the only one that
  lands on the teacher's bond ranks; the converged centred sigma = 1 runs (rich, moderately large init)
  need more data and keep rank 2.4-2.9; the lazy limit fails as dimension counting predicts. We do not
  claim a monotone dependence on the init scale: within small init it is not monotone, and the
  sigma >= 1.5 runs are not converged.
- **Second setting, d = 3, N = 6 (W in R^729, K = 5; teacher [3, 2]: 64 parameters, manifold
  dimension 48; chi = 4 student, tangent dimension <= 120), small init sigma = 0.5:** recovery 0/5 up
  to n = 128, 2/5 at 192, 4/5 at 256, 5/5 at 384 (test error 5.5e-5, bond rank 2.0); the exact lazy
  limit has test error 2.6-39 (worse than zero) at every n <= 384, rank 5.8-8. Without a bond-dimension
  cap the lazy regime would need n = d^N = 729.
- **Sample complexity of the low-rank bias (empirical; conjecture refuted).** Small-init GD needs
  n ~ 200-250 in both settings, ~5-9x the teacher manifold dimension (28, 48). I had conjectured that
  the threshold is set by the heavy tails of product features, ~(1 + 2/d)^N; that predicts 256 for
  d = 2, N = 8 but 21 for d = 3, N = 6, and the observed thresholds (~250 in both) do not follow it. The
  threshold is roughly the number of student parameters for chi = 4 (208, 184), but chi = 16 (832
  parameters) recovers by n = 384, so that is not the rule either. Open.
- **Effective ranks (the implicit bias, directly):** at every n the learned tensor's bond rank is
  ordered small init (1.3-3.9) < centred sigma = 1 (2.4-5) < lazy limit (4-6.9); only small init lands
  on rank 2 once the data determine W*.

## 7. Synthesis: the learning-rate axis and when gradient-flow results apply

(Rewritten in fix round 1; the claims that changed are listed in
`agent-runs/optim-samples/REPORT.md`.) For small (saddle-to-saddle) init, with s_1 the largest teacher
weight and K the number of cores (evidence: binary N=4 at sigma 0.003-0.3, binary N=8 and MPS N=8 at
sigma 0.9, 5 seeds; 45 seeds for the learning order; persistent events):

| GD step | what happens | status |
|---|---|---|
| eta << eta_max = 2/(K s_1^{2-2/K}) | GF: same saddles, same order, event times shifted by eta c_e (exact first-order formula; Richardson limit agrees to 1e-5); relative shift ~ eta ln(1/sigma)/t ~ eta sigma^{K-2} ln(1/sigma) | proven (sharpness), derived + empirical (shift) |
| up to 0.95 eta_max | still GF-like: the persistent saddle sequence equals GF's in 765/765 runs; event times within ~1% (first events) at 0.1 eta_max and 1-3% (max 9%) at 0.9 eta_max, worse for later events where c_e ~ 0; only transient excursions of a near-tied component grow with eta | empirical |
| eta_max < eta < ~1.3 eta_max | the top component oscillates (2-cycle, period 4 from 1.245); the others are learned at GF-like times; excess loss = s_1^2 x (1D-map mean loss), within 0.9% on binary N=4 | derived (1D map) + empirical |
| ~1.5-1.75 eta_max (K = 3) | diverges | empirical |
| ~2 eta_max (K = 3; 1.75-2.5 for K = 7, 8) | learned components are thrown back to the origin saddle and re-learned repeatedly | empirical (1D map: re-entry) |
| >= 2.5-3 eta_max | diverges | empirical |

**Across init scales** (binary N=4, chi = 9 so that GF converges; section 3.5): the edge of monotone,
GF-like GD is 2/lambda_GF(sigma) at every init scale: = eta_max for sigma <= 0.3, 0.65 eta_max at
sigma = 1, then ~ sigma^{-4} (fitted exponent 3.965 +- 0.014 over sigma = 1.41-16; predicted
2(K-1) = 4). Above it, small init oscillates and then diverges (no flatter minimum exists), while
larger init (sigma >= 1) catapults and converges to GF's loss at a flatter minimum (sharpness
1.5-1.94/eta) up to ~1.5 x (sigma = 1) or ~3 x 2/lambda_GF (sigma = 2-8); divergence is general from
3-4 x 2/lambda_GF. Imbalanced small inits behave like the larger-init side: for
2/lambda_GF < eta < eta_max they converge at sharpness ~2/eta to flatter, more balanced minima.

**Online SGD** (binary N=4, teacher [3, 2, 1]):
- The first escape keeps its GF law: no shortening; mean delays consistent with P7 within 1.3 s.e.
  but resolved only at sigma = 0.03; spread ~ (eta/B)^p with p = 0.43-0.52 (predicted 1/2).
- Later degenerate plateaus are lengthened. On the aligned route the P9 shape -ln(1-x)/x,
  x = c_gamma (eta/B) T_GF, fits better than a pole or a linear law (chi^2 379 vs 535 vs 929) with
  c_gamma = 33, and x = 1 separates escaped from trapped conditions in 23/24 cases. It is not exact
  (chi^2/dof ~ 21; the effective c_gamma rises from ~30 to ~45 near the trap). In random-init runs:
  +20 +- 5% at sigma = 0.03, B = 1 (P9 with c = 35: +19%) and +11% at sigma = 0.1 (P9: +5-6%).
- Learned components are erased and re-learned under noise: 20-66% of random-init runs at
  0.47 eta*_ms, none at <= 0.24 eta*_ms over those run lengths; on 50-1000-time-unit aligned plateaus
  0/72 at <= 0.076, 1/36 at 0.12-0.15 and 20/36 at 0.25-0.30 eta*_ms. Erasure depends on the step
  fraction and on the run length.
- The gauge balance drifts ~ (eta/B)^{0.90-1.02}.
- Stability: eta*_ms(B) = 0.00107 B at B = 1, 0.00100 B at B = 16 and 0.00082 B at B = 64 (binary N=4;
  MPS N=4: 0.000765 at B = 1); convergence observed up to 1.5 eta*_ms (5/5), 2/5 at 2 eta*_ms; ~150x
  below GD per sample (product features have kurtosis 3^N).

**Practical rule (derived from the above; tested on binary N=4 and N=8 trees, one teacher).**
- *GD.* Gradient-flow results describe GD training at small init whenever eta < eta_max: same saddle
  sequence, event times off by eta c_e. At any init they hold whenever eta < 2/lambda_GF(sigma).
  Between 2/lambda_GF and ~3 x 2/lambda_GF on the lazy side, GD reaches GF's final loss by a
  different path and to a flatter minimum.
- *SGD.* GF results describe SGD training only when, in addition:
  - eta < eta*_ms(B) (~B/(lambda 3^N) for small B);
  - (eta/B) c_gamma T_plateau << 1 for every later plateau that must be crossed;
  - eta is at most ~0.1 eta*_ms when the learned components must survive long plateaus.

  Because plateaus grow like sigma^{-(k-2)}, at fixed eta/B SGD eventually stalls (or forgets learned
  components) where GF is only slow.

**Implicit bias and generalization** (narrowed in fix round 1). Scope: one teacher per setting (binary
N=8 d=2 with [3, 2]; N=6 d=3), noise-free labels, GD with capped steps.
- Small init (the saddle-to-saddle regime) recovers the teacher with its bond ranks from n ~ 256
  (test error ~1.5e-3 at 256 and ~1e-4 at 384, rank 2.0-2.1).
- The lazy limit fails exactly as dimension counting predicts. At chi = 4 the tangent space has
  dimension 112 < 256 and cannot contain W* at any n. At full bond dimension it is exact only once
  n = d^N.
- Converged rich runs at moderately large init (centred sigma = 1) need more data and keep bond
  rank 2.4-2.9.
- Larger centred inits (sigma >= 1.5) are not converged and are not ranked.
- No monotone dependence on init scale is claimed (it is not monotone within small init).
- The saddle-to-saddle regime is the sample-efficient one here, but its threshold (5-9x the teacher
  manifold dimension) is not explained.
