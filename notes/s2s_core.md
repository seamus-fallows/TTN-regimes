# s2s-core: saddle-to-saddle phenomenology and escape-time scaling (Q1, Q2)

Agent `s2s-core`. Experiments in `experiments/s2s_core/`. Status tags: **proven**, **derived**
(under stated assumptions), **empirical**, **conjecture**. Setting: `notes/model.md` (whitened
inputs, exact population loss, odeco teacher W* = sum_r s_r U_r, fan-in init with per-core scale
sigma, gradient flow).

## 1. Theory

**Relation to the project theory notes.** Sections 1.1-1.2 were derived here before
`notes/theory_escape.md` existed; theory_escape v2 (verified) contains the same results and is the
reference for them: P1 = its E1 and recipe 2.5 (t_hat_1 per init; my c_K = its delta_K), B1 = its E2
(Kumar-Haupt lower bound), 1.2 = its E5 and 5.1 (reduced network, exact race on MPS); 1.4 = the
piggyback block of `notes/theory_saddles.md` 4.1 (m = 2 strict iff a larger component is unlearned).
What this note adds: the RMS-overlap heuristic P3 (1.5), the measured tests (section 2), and the
dormant-suppression analysis (2.2). In what follows "derived here" marks my own derivations.

Notation: K cores, g(theta) = <W(theta), W*> (degree-K homogeneous polynomial in theta),
L = 1/2 ||W||^2 - g + 1/2 ||W*||^2. With the same seed, fan-in init is linear in sigma:
theta_0(sigma) = sigma * theta_hat, theta_hat = the sigma = 1 init.

### 1.1 Correlation flow and exact scaling of the first escape

**Correlation flow.** For ||W|| << s_r the term 1/2 ||W||^2 (degree 2K) is negligible next to g
(degree K): its gradient is smaller by a factor ~||W||/s. To leading order the first plateau is the
gradient flow of g alone, theta' = grad g(theta) ("correlation flow").

**Scaling (proven).** If theta(t) solves theta' = grad g(theta) then lambda theta(lambda^{K-2} t)
solves it too (grad g is homogeneous of degree K-1). For K >= 3 the correlation flow blows up in
finite time T(theta_0) whenever g eventually becomes positive along the flow, and
T(sigma theta_hat) = sigma^{-(K-2)} T(theta_hat).

**First-escape prediction (derived; assumption: the escaping component is balanced and aligned by
the time ||W|| becomes O(s), which the conservation laws guarantee up to O(sigma^2) imbalance).**

    t_1(sigma, seed) = sigma^{-(K-2)} T(theta_hat) + c_K(s_{r1}) + o(1),          (P1)

where r1 = the component whose overlap blows up first in the correlation flow, and
c_K(s) = s^{-(2-2/K)} [ int_0^{2^{-1/K}} x dx / (1 - x^K) - 2^{1-2/K} / (K-2) ] is the difference
between the a = 1/2 crossing time of the balanced Saxe mode q' = q^{K-1}(s - q^K) and the blow-up
time of its correlation part q' = s q^{K-1}, in the limit q_0 -> 0 (derivation: subtract the two
integrals dt = dq / (q^{K-1}(s - q^K)) and dt = dq / (s q^{K-1})). Numerically the bracket is
-0.850, -0.266, -0.047, 0.072, 0.148, 0.201 for K = 3..8.

(P1) has **no free parameters**: T(theta_hat) is computed per seed by integrating the correlation
flow from the sigma = 1 init (cheap: no stiffness, self-similar blow-up), and it predicts the
exponent K - 2 exactly and the prefactor per seed. It also predicts which component is learned
first (r1).

**Spherical decomposition (derived).** Write theta = rho u, |u| = 1, d tau = rho^{K-2} dt. Then
du/dtau = grad_S g(u) (spherical gradient ascent of g, independent of rho), and
d rho / dtau = K g(u) rho. Hence

    T(theta_0) = rho_0^{-(K-2)} int_0^inf exp( -(K-2) K int_0^tau g(u(tau')) dtau' ) dtau.

If u started at a maximizer of g on the sphere, the integral would be 1 / ((K-2) K g_max).

**Lower bound (proven for the trees used here, odeco W*).** For any TTN,
|g(theta)| <= s_1 prod_v ||G_v||_F <= s_1 (||theta|| / sqrt(K))^K. (Write g = sum_r s_r a_r with
a_r the tree contraction evaluated at u_r at every site; bound each internal contraction by the
Frobenius norms; the cores carrying physical legs see disjoint slices G_v(u_r, ...) for different r,
so sum_r prod_{v in P} ||slice_{v,r}|| <= prod_{v in P} (sum_r ||slice_{v,r}||^2)^{1/2} by Hoelder
with |P| >= 2 factors, P = cores with physical legs.) Hence g_max <= s_1 K^{-K/2} on the unit
sphere, with equality at the balanced rank-one configuration aligned with component 1, and

    T(theta_hat) >= ||theta_hat||^{-(K-2)} K^{K/2-1} / ((K-2) s_1).                 (B1)

The PLAN's early hypothesis t_1 ~ rho_0^{-(K-2)} [tau_align + K^{K/2-1}/((K-2) s_1)] is (B1) plus an
alignment delay; as theory_escape 2.2 notes, there is no separate additive alignment phase (the whole
integral is alignment). Only the parameters g depends on matter: at each physical leg, directions
orthogonal to span{u_r} (and for a two-site core, the mixed slices u_r (x) u_r', r != r') have zero
gradient in the correlation flow and stay frozen at O(sigma). Replacing ||theta_hat|| by the norm of
the *active* parameters rho_act (slices along the teacher directions + cores without physical legs)
gives a tighter bound (B1').

### 1.2 Decoupling of components in the correlation flow (derived)

For an odeco teacher, g = sum_r s_r a_r(theta) with a_r = f_theta(u_r^(1), ..., u_r^(N)). On a core
with physical legs, a_r depends only on the slice G_v(u_r, ..., u_r; bonds). These slices are disjoint
for different r. Hence:
- **MPS** (every core carries a site): the correlation flow splits into R independent scalar
  "deep linear networks" (depth K, widths chi, scalar input and output), one per component, each
  with its own blow-up time T_r. Then T = min_r T_r and r1 = argmin: the first component is decided by
  a race between independent DLN escapes (`order` studies the statistics).
- **binary / HT**: the slices at cores carrying sites are disjoint, but the internal cores (no
  physical legs) are shared by all components, so the components are coupled through them already
  in the correlation flow.

### 1.3 Later escapes (derived, heuristic)

After component r1 is learned, its cores are balanced rank-one with amplitude s^{1/K} along a bond
direction b_e on every edge e. Any part of another component's slices along b_e creates mixed terms
(e.g. u_2 (x) u_1 (x) ...) that 1/2 ||W||^2 penalizes with curvature O(s^{2-2/K}); those parts relax in
O(1) time. The rest of component 2's slices keeps following its correlation flow in the
orthogonal complement of b_e. An odeco component differs from the learned one at every site, so
adding it needs all K cores (the k-rule of H2 gives k = K): the next escape is again a degree-K
escape and

    t_{i+1} - t_i ~ sigma^{-(K-2)} * C_i(seed)        (exponent K - 2 for every gap)   (P2)

with C_i from the projected correlation flow (no closed form). Widths of events are O(1) (time from
a = 0.1 to a = 0.9 of a Saxe mode, ~ s^{-(2-2/K)}), so separation ratios grow like sigma^{-(K-2)}:
**S2S for every tree with K >= 3 as sigma -> 0** (derived). Caveat: if two components blow up at
nearly the same rescaled time (close weights), the separation at a given sigma can be small.

### 1.4 Saddle type at partial fits (derived)

At the partial fit W_S (components in S learned, balanced), the second-order change of L along a
direction that adds component r' needs every core to change, so it is of order eps^K: zero Hessian
(higher-order saddle) for K >= 3. Exception ("piggyback", H1b): perturb only the cores carrying
physical legs, G_v += eps u_r'^{(x) sites(v)} (x) (bond directions of the learned component r).
Then W gains eps^m q^K U_r' (m = number of cores with physical legs) plus mixed terms of order eps.
For m = 2 (binary N = 4) the Hessian along this direction is q (s_r - s_r') with q = s_r^{1/K}:
**negative iff a larger component r' is unlearned while r is learned** (a wrong-order saddle is
strict). For m >= 3 (binary N = 8: m = 4; MPS and HT: m = N) the gain is O(eps^m) and the direction
is again higher order. Prediction for the Hessian at plateau midpoints: no eigenvalue below -epsilon
for K >= 3, except binary N = 4 plateaus where a larger component is still unlearned.

### 1.5 Typical prefactor: RMS-overlap heuristic (derived heuristic, P3)

Fan-in init makes W_0 rotation-invariant on every physical leg with E||W_0||^2 = sigma^{2K}, so the
initial overlap with any unit rank-one U_r has RMS a_rms = sigma^K d^{-N/2}, independent of chi.
If component r escaped like an aligned balanced mode started at that amplitude
(q_0 = a_rms^{1/K} = sigma d^{-N/(2K)} per core), then

    T_r ~ T_rms(s_r) = d^{N(K-2)/(2K)} / ((K-2) s_r)     [in units of sigma^{-(K-2)}]      (P3)

and t_1 ~ sigma^{-(K-2)} min_r T_r. This has no free parameters and no chi dependence. It replaces the
PLAN's rho_0-based hypothesis, whose aligned-at-start form is the lower bound (B1) and is off by
factors 4 (K = 3) to ~5000 (K = 8, chi = 6) in pilots (section 2.1).

## 2. Experiments and findings

### 2.1 Pilots (throwaway scipy code, before the library landed; to be redone with the library)

**Library fix 57d5262** (teacher and student shared random streams) changed every teacher draw. My
pilots before it used teacher seed 100 and init seeds 0..39, so they were never correlated with the
teacher, but their per-seed numbers are not comparable with later runs. Pilots marked [old lib] below
will be redone; the K sweep is already on the fixed library.

**t_1 vs K (fixed library; correlation flow, chi = 3, d = 4, weights [3, 2, 1], 30 seeds)**:
geometric-mean T vs P3 (`d^{N(K-2)/(2K)}/((K-2) s_1)`) and vs the Kumar-Haupt formula
`1/(K(K-2) s_1 K^{-K/2} ||theta_hat||^{K-2})`:

| tree | K | T (geo. mean) | T / P3 | T / KH |
|---|---|---|---|---|
| mps 3..10 | 3..10 | 0.67 .. 17.6 | 0.96 .. 1.65 | 3.0 .. 2.5e4 |
| binary 4, 6, 8, 12, 16 | 3, 5, 7, 11, 15 | 0.88 .. 814 | 0.94 .. 2.1 | 4.1 .. 3.1e7 |
| ht 3, 4, 6, 8 | 5, 7, 11, 15 | 0.41 .. 5.4 | 1.06 .. 1.72 | 16 .. 2.1e5 |

So the KH formula (init along the KKT point, delta = ||theta_0||) is a strict lower bound (B1) that
misses by up to 7 orders of magnitude at K = 15, while the RMS-overlap heuristic P3 tracks the
typical prefactor over a factor-1500 range within a factor 0.94-2.1 (the excess grows slowly with K).

All gradient flow with scipy DOP853 / LSODA, rtol 1e-9, atol 1e-14 sigma; correlation flow
DOP853 rtol 1e-10. d = 4, odeco teacher seed 100.

- **(P1) confirmed per seed** (empirical): t_1 sigma^{K-2} at sigma = 0.3..0.003 (binary N=4,
  K=3), 0.3..0.01 (binary N=8, MPS N=8) converges to the correlation-flow blow-up time T(theta_hat)
  to 4-5 digits (binary 4 seed 0: 0.76962 at sigma = 3e-3 vs T = 0.77021; the remaining offset
  t_1 - sigma^{-1} T = -0.198 matches c_3(3) = -0.196). The first component learned is the one whose
  correlation flow blows up first (also when it is the smallest, MPS 8 seed 1).
- **Race model** (MPS: proven by decoupling; others: empirical): the blow-up time with only
  component r in the teacher, T_r, gives T = min_r T_r exactly for MPS (rel. diff 1e-12) and within
  1-3% for binary/HT, with the correct first component in 93-100% of 40 seeds.
- **Typical prefactor (P3)**: median T_r / T_rms(s_r) = 1.0-1.2 (binary 4, MPS 4), 1.3-1.75 (binary 8,
  MPS 8, HT 4), chi = 3, 40 seeds. The aligned-at-start bound (B1, full norm) is 3.5-5600x too small.
- **chi dependence** is weak: median T for chi = 6 vs chi = 3 is lower by factors 1.2 (binary 4,
  MPS 4) to 1.6-1.9 (binary 8, MPS 8, HT 4), much less than the (sum chi)^{-(K-2)/2} ~ 1/8 implied
  by the rho_0 picture for K = 8.
- Seed-to-seed spread of T: coefficient of variation 0.2 (K = 3) to 0.8 (HT 4, MPS 8).

### 2.2 Step 1: aligned init reproduces the exact mode solutions (derived; known for DLNs)

Aligned balanced init (`ttn.network.aligned_init`, per-core amplitude a0 on every component) on the
odeco teacher [3, 2, 1]: the TTN gradient flow stays in the aligned family and each component follows
the balanced Saxe mode a' = a^{K-1}(s - a^K) (Saxe et al. 2014 eq. 15 for DLNs; explicit for TTNs here,
`notes/theory_escape.md` 2.3). **Empirical check** (`results/aligned.csv`, DP5 rtol 1e-10, atol 1e-14):
max |a_GF - a_theory| over t and r = 2.0e-10 (binary4), 3.1e-10 (mps4), 2.2e-9 (binary8), 2.5e-9
(ht4), 2.8e-9 (mps8); event times identical to 1e-11 relative on the same grid; Q drift <= 2e-14.

### 2.2b First escape: per-seed test of the correlation-flow prediction (empirical; P1 = theory_escape E1)

Sweep `configs/first.yaml` (+ `first_k78.yaml`): for every seed, T(theta_hat) from the correlation flow
of the sigma = 1 init, then gradient flow (LSODA rtol 1e-8, atol 1e-11 sigma) at 8 sigmas until the
first component is learned, event refined in linear time. Fan-in and gauge-balanced inits, d = 4,
chi = 3, 6, both teachers, 10 seeds (K = 7, 8: separated teacher, fan-in; `results/first_*.csv`,
figure c).
- **Exponent** (per seed, then mean +- SE): binary4 (sigma 0.1 .. 0.002) 1.0045 +- 0.0002 raw,
  0.9996 +- 0.0001 with the predicted O(1) offset c_K removed; mps4 (same sigmas) 2.0002 / 2.0000;
  binary8 (sigma 0.4 .. 0.01, 1.6 decades) 4.9984 +- 0.0003 (K - 2 = 1, 2, 5). [ht4, mps8 pending]
- **Prefactor, no free parameter**: |t_1 / (sigma^-(K-2) T(theta_hat) + c_K) - 1| at the smallest
  sigma <= 2e-3 (binary4), 6e-5 (mps4), 2e-8 (binary8, sigma = 0.01). At larger sigma the residual
  is the O(1) offset (K = 3, 4: the predicted c_K accounts for it to within ~25% (binary4) and ~10%
  (mps4)) or, at K = 7, ~37x the c_K term at sigma = 0.3 (2% at sigma = 0.4), as theory_escape 2.4
  anticipates (the escape direction has not converged at moderate sigma).
- **First component** = the one dominating the correlation flow at its blow-up: 100% of runs on mps4
  and binary8; 80-90% on binary4 with the close teacher (matrix-like pair: the wrong order at finite
  sigma, theory_escape E9).
- **Kumar-Haupt is a lower bound, far off**: median T / t_KH = 3.1-4.8 (binary4), 7.9-13.7 (mps4),
  184-489 (binary8), inside the ranges theory_escape 9.2 predicts (2.5-5.2, 5.2-18, 94-4463).
- **Fan-in imbalance and the prefactor**: gauge-balancing the same init (same W_0, Q_e = 0) changes
  T per seed by a factor 0.88-1.92 (median 0.99-1.18 per cell; usually slower), keeps the first
  component in 80-100% of seeds, and leaves the exponent unchanged.

### 2.2c Leading-order prefactor vs d, chi, K (empirical; tests of theory_escape section 9)

Correlation-flow blow-up time only (`lo_sweep.py`, the exact sigma -> 0 limit of sigma^(K-2) t_1 per
init; verified against GF above), median with bootstrap 95% CI.
- **d-dependence at full bond** (`configs/lo_d.yaml`, single component s = 1, 200 inits; theory_escape
  9.3, 9.5): binary4 median t_hat_1 = 1.544 [1.468, 1.594], 1.569 [1.525, 1.616], 1.563 [1.525, 1.586]
  for d = 2, 3, 4 (chi = d^2), i.e. pi/2 = 1.571 and flat in d as predicted, while the Kumar-Haupt
  time falls 1.9x (0.576 -> 0.301); sd ln t_hat_1 = 0.218, 0.157, 0.135 vs the predicted
  1/(2 sqrt(chi_b)) = 0.250, 0.167, 0.125. mps4: 1.39 [1.26, 1.58], 1.41 [1.30, 1.52], 1.50 [1.43, 1.60]
  (predicted 1.30, 1.35, 1.46), while t_KH falls 2.9x (predicted 2.8x). The rho_0 (whole-norm) law is
  refuted; the per-init prediction is not.
- **Balance at the escape** (lead's check of the critique claim that KKT points with g > 0 are
  balanced with core norms 1/K): at the end of the correlation flow every core's squared-norm fraction
  is 1/K to within 5e-4 .. 1.3e-2 (max over 1200 inits) (`frac_dev_max` in `results/lo_d.csv`).
  (Expected by the conservation laws: Q_e is constant while |theta|^2 grows without bound.)
- **K dependence** (`configs/lo_K.yaml`, chi = 3, d = 4, s = (3, 2, 1), 50 inits per tree,
  `results/lo_K_table.csv`, figure e2): median t_hat_1 grows from 0.68 (mps3) / 0.86 (binary4) to 52 (mps12),
  814 (binary16) and 5.6 (ht8). The zero-parameter RMS-overlap heuristic P3 = d^{N(K-2)/(2K)}/((K-2) s_1)
  (section 1.5) tracks the median within a factor 0.87-2.1 over K = 3-15 (MPS: 0.95-1.54; binary:
  0.87-2.13; HT: 1.10-1.80; the excess grows with K), while the median t_hat_1 / t_KH grows from 3 to
  2.9e7. In the brief's terms: t_1 vs K across trees is captured by P3 to a factor 2 with no fit
  (empirical; P3 is a heuristic, derived from the RMS initial overlap under an aligned-mode assumption).
- chi dependence: [lo_chi pending]

### 2.3 Saddle-to-saddle phenomenology at K = 3, 4 (empirical)

Sweep `configs/main.yaml`: fan-in init, d = 4, chi = 3, 6, teachers [3, 2, 1] and [1, 0.85, 0.7],
sigma = 0.1 .. 0.003 (8 values), 10 seeds; LSODA rtol 1e-8, atol 1e-11 sigma. Events refined in
linear time (no duration read off the log grid).
- **All runs learn all components** on binary4 (320/320). On mps4: 160/160 at chi = 6; at chi = 3,
  48/80 (separated) and 58/80 (close): the rest are trapped (2.7).
- **S2S criterion** (diagnostics.md) among complete runs: binary4 separated 158/160, close teacher
  5/20 at sigma = 0.1 rising monotonically to 20/20 at sigma = 0.003 (overlapping transitions at larger
  sigma); mps4 266/266. Criterion (iii) (plateaus grow without bound) is the exponent table of 2.4.
- **Plateau levels are the partial-fit levels** L_S = 1/2 sum_{r not learned} s_r^2 (S = the components
  learned so far, in whatever order): max relative deviation at the plateau midpoint 2e-3 (mps4),
  2e-2 (binary4, at sigma = 0.1), 3e-6 (the few complete K = 7 runs).
- **Bond spectra grow one value at a time on every edge simultaneously**: in 213/236 runs with
  event-window spectra, each event raises the number of singular values above s_r/2 by exactly one on
  every edge, and all edges step at the same output sample of the refined window (spread < width/40).
  The 23 exceptions are overlapping events (two ranks in one window: binary4, close teacher,
  sigma >= 0.05). Effective bond rank 0 -> 1 -> 2 -> 3 (figure b).
- **Order** (recorded; `order` studies it): binary4 learns the largest component first in 320/320
  runs' first event (matrix-like, greedy); the full order is greedy in 157/160 at chi = 3, 6 ... (close
  teacher: 0.9-0.96). On mps4 the first component is random (P(largest first) = 0.5-0.8 per cell) but is
  the same at every sigma for each seed (10/10 seeds in every cell), as the race picture predicts
  (theory_escape E8).

### 2.4 Escape-time exponents (empirical; prediction derived)

Per-seed fits of log t vs log sigma over the sigmas at which the seed completes in its modal order
(>= 4 sigmas), mean +- SE over seeds; predicted K - 2 for t_1 and every gap (odeco: every route uses
all K cores, theory_saddles R2):

| tree | K | chi | teacher | seeds | t_1 | t_2 - t_1 | t_3 - t_2 |
|---|---|---|---|---|---|---|---|
| binary4 | 3 | 3 | separated | 10 | 1.005 +- 0.000 | 1.003 +- 0.001 | 1.004 +- 0.003 |
| binary4 | 3 | 6 | separated | 10 | 1.006 +- 0.001 | 1.008 +- 0.002 | 1.012 +- 0.002 |
| binary4 | 3 | 3 | close | 10 | 1.000 +- 0.003 | 1.002 +- 0.042 | 0.92 +- 0.11 |
| binary4 | 3 | 6 | close | 10 | 0.999 +- 0.005 | 1.036 +- 0.036 | 1.12 +- 0.05 |
| mps4 | 4 | 3 | separated | 6 | 2.000 +- 0.000 | 1.999 +- 0.000 | 1.990 +- 0.005 |
| mps4 | 4 | 6 | separated | 10 | 2.000 +- 0.000 | 2.001 +- 0.001 | 1.998 +- 0.001 |
| mps4 | 4 | 3 | close | 7 | 2.000 +- 0.000 | 2.000 +- 0.001 | 2.004 +- 0.002 |
| mps4 | 4 | 6 | close | 10 | 2.000 +- 0.000 | 2.001 +- 0.001 | 2.001 +- 0.001 |

The residual deviations of the raw t_1 exponent from K - 2 (e.g. 1.005) are the predicted O(1) offset
c_K (section 1.1): with it removed the exponent is 0.9996 +- 0.0001 (2.5). The close-teacher binary4
gaps scatter more because transitions overlap at sigma >= 0.03. K = 7, 8: section 2.6 (pending).

### 2.5 Saddle type at plateau midpoints (empirical; theory_saddles R10)

Gauge-projected Hessian at the plateau midpoints (minimum speed between events). Primary rule
(diagnostics.md): slope of log(-lambda_min) vs log sigma per seed: 1.06 +- 0.01 / 1.02 +- 0.00 (first /
second plateau, binary4 chi = 3, separated), 1.08 / 1.04 (chi = 6); mps4 2.01 / 1.99 (chi = 3), 2.01 /
2.00 (chi = 6): lambda_min -> 0 like sigma^{K-2}, i.e. **higher-order saddles**, as predicted for odeco
teachers (no strict direction: m >= 3, or m = 2 on binary4 with the larger components learned first).
Magnitude: ratio of lambda_min to the route formula -(K-1) s_r (s_r |a_r|)^{(K-2)/K} (theory_saddles R10,
balanced route, A = 1 for k = K) evaluated with the measured dormant overlap: median 0.88-1.15 per cell.
Caveat: the library's gauge projection was found to bias projected eigenvalues by up to ~20% near
rank-deficient saddles (fix pending); slopes are unaffected; values will be recomputed, raw values are
reported next to them (`recompute_hessians.py`).

### 2.7 Dormant-component suppression (lead's question; empirical unless tagged)

Question: do components that are not yet learned shrink during earlier escapes and plateaus, by how
much (vs sigma, K, chi), does a gauge-balanced init avoid it, and are later plateau times physics or
roundoff? Data: sweeps `main` (fan-in), `large` (K = 7, 8, chi = 6), `k78` (K = 7, 8, chi = 3, fan-in
and gauge-balanced, no amplitude stop), `balanced` (K = 3, 4); `results/dormant_*.csv`,
`results/main_runs.csv` (column `class`), figure g.

**(a) Change during the first escape.** Ratio of each dormant amplitude |<W, U_r>| across the refined
rise window (0.1 -> 0.9) of the first escaping component, separated teacher: median factor x1.5-2
(binary4: dormant components GROW, the head start of theory_saddles 3.3), x0.7 (mps4), x0.4-0.6
(K = 7, 8, chi = 3), x0.6-0.8 (K = 7, 8, chi = 6), with no trend in sigma (0.003-0.4); a drop by more
than 100x occurs in 0% (K = 3, 4; K = 7, 8 at chi = 6) and 2.5-6% (K = 7, 8 at chi = 3) of cases (tail
down to 1e-6 .. 1e-10). Later escapes: same picture (`dormant_escape_drops.csv`). So escapes do not
suppress dormant components appreciably except in a chi = R tail.

**What does suppress them: the conservation-law trap on the following plateaus** (theory_saddles R12).
On a plateau the dormant component's slices on the route END cores decay exponentially while the inner
slices stay put; once an end is dead the component is never learned in float64. Run classes
(separated teacher; trapped = a never-learned component has a slice < 1e-8 sigma, i.e. 7 decades below
the init scale; class rules in `analyze_main.py`):

| tree | K | chi | complete | trapped | stalled / unresolved | n |
|---|---|---|---|---|---|---|
| binary4 | 3 | 3, 6 | 1.00 | 0 | 0 | 160 |
| mps4 | 4 | 3 | 0.60 | 0.40 | 0 | 80 |
| mps4 | 4 | 6 | 1.00 | 0 | 0 | 80 |
| ht4 | 7 | 3 | 0.10 | 0.82 | 0.08 | 60 |
| binary8 | 7 | 3 | 0.05 | 0.27 | 0.68 | 79 |
| mps8 | 8 | 3 | 0 | 0.30 | 0.70 | 70 |

(`main` sweep; the K = 7, 8 "unresolved" runs hit the budget or the amplitude stop and are re-run in
k78 / large, see below.) At K = 4, chi = 3 the SAME seeds trap at every sigma from 0.003 to 0.1
(separated: 4/10 at all 8 sigmas): whether a seed traps is fixed by its init direction.

**(c) Decay rate (derived; empirical check).** Reduced model (theory_saddles section 8): the dead end's
new column x and the parent's visible slice y_vis obey x' = -q^{2(K-2)} |y_vis|^2 x,
y_vis' = -q^{2(K-2)} |x|^2 y_vis with |x|^2 - |y_vis|^2 - |y_route|^2 = Q_nu nu conserved
(q = s_q^{1/K}); if Q_nu nu + |y_route|^2 < 0 then x ~ exp(-gamma t), gamma = q^{2(K-2)}
|Q_nu nu + |y_route|^2|. Fan-in init makes Q = O(sigma^2) and y_route = O(sigma), so gamma ~ sigma^2 per
seed. Measured (mps4, chi = 3, end-core slices of the trapped component): gamma / sigma^2 is constant
per seed within a factor 1.8 over sigma = 0.003-0.1 for 5 of 7 trapped seed/teacher pairs (values
0.0003 .. 1.0 across seeds); two pairs switch between two such levels (by 20-100x) at sigma ~ 0.02-0.03.
On two re-run trapped seeds, Q_nu nu (from the init, conserved) makes the end core the light side at
BOTH route ends, the joint condition of theory_saddles v2 for a dead route; both ends decay at the same
rate, 3-15x below the single-end estimate s_q |Q_nu nu| (which ignores |y_route|^2): the reduced model
gives the sign and the sigma^2 scaling, the magnitude to an order of magnitude.
Consequence (derived): trap time ~ sigma^-2 vs escape time ~ sigma^-(K-2). K = 3: escape first, never
trapped (0/320). K = 4: same order, a sigma-independent seed-dependent fraction traps (4/10 at chi = 3).
K >= 5: the ends' fate is settled long before the route can escape.

**(b) Gauge-balanced init removes the trap (empirical; supports theory_saddles R12).** Same seeds and
W_0, rescaled so that every Q_e = 0 (`s2s.gauge_balance`):
- K = 4 (mps4, chi = 3, sigma = 0.003 .. 0.03): every balanced run completes (fan-in: 4/10 seeds trap at
  every sigma). K = 3: nothing to remove (fan-in never traps). [`balanced` sweep]
- K = 7, 8, chi = 3, sigma = 0.15-0.3 (`k78`, 90 matched runs per init): trapped 0/90 balanced vs 47/90
  fan-in (binary8 0 vs 9/30, ht4 0 vs 24/30, mps8 0 vs 14/30). Of the 47 fan-in-trapped seeds, 17
  complete with balanced init, 29 are "stalled" (all route ends alive, budget exhausted), 1 unresolved.
  Complete: binary8 3 vs 2, ht4 17 vs 3, mps8 0 vs 0 (30/30 balanced mps8 runs learn 2 of 3
  components; with fan-in 18/30 learn only 1). Balanced init is not uniformly better: 5 runs complete
  with fan-in but stall with balanced init (the first escapes differ).
- **Stalled is not stuck (continuation test).** 8 stalled k78 runs (both inits; chosen as the ones with
  the shortest predicted remaining plateau) were continued from their final state: all 8 escape, at
  0.49-0.85x the balanced-route estimate of the remaining plateau b^-(K-2)/((K-2) s_r) with
  b = (s_r |a_r|)^(1/K) from the dormant amplitude at the end of the run; the ratio is the same per
  seed across sigma (e.g. 0.80-0.85, 0.49-0.54, 0.85). The estimate is an upper bound (derived: for a
  fixed product of route amplitudes the balanced route is the slowest, since a_v^2 - a_w^2 is
  conserved and the product's growth rate is s prod_v a_v sum_v 1/a_v^2). For all stalled runs the
  estimate is a median 10-60x the time already integrated: these runs sit on ever-longer plateaus whose
  length is set by how far the dormant amplitude was suppressed, (suppression)^-(K-2)/K.

**(d) Physics vs roundoff.** A dead end's slice decays exponentially until it reaches the integrator's
absolute tolerance / float64 roundoff (~1e-12 .. 1e-15 absolute), where it stays; any later escape of
that component would be seeded by roundoff, so trapped runs are reported as trapped and their later
"plateau" is not reported as a time. An earlier criterion (dormant amplitude |<W, U_r>| < 1e-14) was
too strict at K >= 7: the amplitude is a product of K per-core route amplitudes and can be 1e-15 while
every per-core amplitude is O(1e-3) and the escape signal ~ s b^{K-1} is far above roundoff; runs
stopped by that criterion are classed unresolved and re-run without it (k78). All reported event times
come from runs in which every dormant per-core slice stayed far above roundoff.

