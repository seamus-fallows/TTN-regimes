# Order of learning (Q4; hypotheses H1, H1b) — agent `order`

Status: **in progress** (steps 1 done; steps 2-4 running). Experiment folder `experiments/order/`
(README there: configs, commands, cost). Tags: **proven**, **derived** (assumptions stated),
**empirical**, **conjecture**. Companion theory notes: `notes/theory_escape.md` (v2; sections 3, 5 cover
the same ground from the escape side) and `notes/theory_saddles.md` (v2; section 4, H1b).

## 0. Results at a glance

| # | claim | status | evidence |
|---|---|---|---|
| O1 | Near the origin the order is decided by the leading-order (LO) flow of g = <W, W*>, whose outcome does not depend on sigma (exact scaling symmetry) | derived (corollary of Kumar & Haupt 2025a) | per-seed GF = LO on 180/180 seeds (pilot); MPS4 P(greedy) identical for sigma 0.003-0.1 |
| O2 | m >= 3 (cores carrying the differing sites): every component is a local max of g on the sphere, order random (H1) | derived | P(larger first) = 0.60-0.88 at rho = 0.5 on MPS4-8 and binary8 (400 seeds per setting) |
| O3 | Odeco teacher on an MPS: the LO flow decouples into one independent chain per component; winner = argmin tau_r/s_r | derived (exact at leading order) | 40800/40800 MPS GF runs agree with the per-seed race |
| O4 | P(greedy) falls toward 1/2 with K and rises slowly with chi; d enters only through bond caps (MPS) | empirical (+ derived for d) | sd log tau table, d-pairs chi^2 |
| O5 | Selection rule: argmax_r s_r X_r^(K-2), X_r = projection of the init on component r's KKT orbit, predicts 76-92% of winners; fitted exponent ~K-2 | empirical | `results/main_rules.csv` |
| O6 | m = 2 (binary4 = depth-3 DLN; matrix-like pattern pairs): greedy as sigma -> 0, P(smaller first) ~ sigma^(n(1-rho)/rho) | derived (heuristic for the exponent) | binary4 sigma scan (pilot): gamma within 1 SE of (1-rho)/rho for rho = 0.7-0.9 |
| O7 | H1b: a wrong order is corrected through a strict saddle (lambda_min = -(s_1-s_2) s_2^((K-2)/K)) only for m = 2; the rotation takes ln(1/delta)/|lambda_min| | derived; Hessian checked | saddle-start slope 5.39 vs 5.386 (pilot) |

## 1. Setting

Odeco teacher W* = sum_r s_r u_r^(1) (x) ... (x) u_r^(N), whitened inputs, L = 1/2 ||W - W*||^2, fan-in
Gaussian init at per-core scale sigma. Two components with weights (1, rho), rho < 1, unless stated.
"Greedy" = the larger component has the earlier learning event (first time a_r = <W, U_r>/s_r crosses
1/2, `notes/diagnostics.md`). P_g = P(greedy) over the init.

The init is invariant in law under orthogonal rotations of any physical leg, so the teacher bases can be
taken to be standard (u_r^(i) = e_r) without changing any order statistic (all runs here do this). Then
T_r(theta) = <W(theta), e_r (x) ... (x) e_r> contracts the "slices" G_v[r, ..., r, :] of the cores with
physical legs and the full cores without physical legs (shared by all components).

## 2. Leading-order dynamics and its scale invariance  [derived]

L = 1/2 ||W||^2 - g(theta) + const, g = sum_r s_r T_r, homogeneous of degree K. Near the origin
dtheta/dt = grad g + O(|theta|^(2K-1)). The LO flow F_t of grad g satisfies F_t(sigma n) = sigma
F_(sigma^(K-2) t)(n). Hence:
1. from sigma n_0 the trajectory in direction space is the same for every sigma;
2. with theta = R n, d tau = R^(K-2) dt: dn/dtau = grad_S g(n), d ln R/d tau = K g(n); blow-up happens at
   tau = infinity, so the direction converges to a critical point of g on the sphere (Kumar & Haupt
   2025a: KKT point);
3. GF leaves the LO flow at R = O(1), i.e. at tau_1 ~ ln(1/sigma)/(K g): as sigma -> 0 the first
   component is the local maximum whose basin contains n_0; at finite sigma the direction may be in
   transit.

Check (`balance_check.py`, `results/balance_check.csv`): at |theta| = 1e4 |theta_0| on the LO flow, every
core's share of ||theta||^2 equals 1/K to < 5e-5 and per-bond imbalance |Q_e|/|theta|^2 < 3e-5 (5 trees x
5 seeds): the escape direction is balanced, as the critic's derivation says (lead message, item 3).

## 3. Which components are local maxima: the count m  [derived]

m = number of cores carrying sites where the two components differ (odeco: cores with physical legs).
At B (only component 2, balanced rank-one), component 1's term s_1 T_1 is a product of m slices that
vanish at B: the gain along the perturbation is O(eps^m), the spherical constraint costs O(eps^2).
- m >= 3 (odeco on MPS N >= 3, binary N >= 8, HT): every component is a local max: random order (H1).
- m = 2 (odeco on binary4, which is exactly a depth-3 DLN A R B^T; pattern teachers differing on two
  cores): the gain is bilinear, s_1 x^T S y; the spherical Hessian at B has the eigenvalue
  (s_1 - s_2) K^(-(K-2)/2) > 0 (n of them, n = chi_e for two adjacent cores sharing bond e, else 1,
  `notes/theory_escape.md` 3.3): only the largest component is a local max (DLN: Jacot et al. 2021 Thm 2).
- m = 1: the components form one rank-one TTN component and are learned together.
Same classification: `notes/theory_saddles.md` R6 (piggyback graph; for >2 components the criterion is
rho(S) > s_q, which for odeco teachers reduces to "some larger component has m = 2").

## 4. m >= 3 on an MPS: an exact race  [derived; verified]

On an MPS every core carries one site, so T_1 and T_2 depend on disjoint parameters and the LO flow
splits into independent chain flows d theta^(r)/dt = s_r grad T(theta^(r)) (scalar-output chain DLNs
with widths = bond dimensions). Component r blows up at sigma^-(K-2) tau_r / s_r, tau_1, tau_2 iid, so

    P_g(rho) = P(tau_2 / tau_1 > rho)   (independent of sigma at leading order).

With a log-normal tau, P_g = Phi(ln(1/rho)/(sqrt 2 sd ln tau)). The same race appears as
`notes/theory_escape.md` 5.1. For binary8 the internal cores are shared; the race built from
single-component blow-up times is then an approximation.

## 5. m = 2 at finite sigma: wrong order with probability ~ sigma^(n(1-rho)/rho)  [derived, heuristic]

Near B the distance from B's stable manifold grows like exp(lambda_u tau) while ln R grows at rate
K g(B); lambda_u / (K g(B)) = (s_1 - s_2)/s_2 = (1 - rho)/rho. A trajectory entering at distance delta_0
leaves at radius R_0 delta_0^(-rho/(1-rho)); the smaller component wins iff R reaches O(1) first, i.e.
delta_0 < sigma^((1-rho)/rho) in each of the n unstable directions. With a bounded density of delta_0:
P(wrong) ~ C sigma^(n (1-rho)/rho). Corollary: conditioned on a wrong order, the residual deviation
delta_1 = delta_0 sigma^(-(1-rho)/rho) is O(1)-distributed (roughly uniform), so the subsequent correction
takes O(1/|lambda|), not ln(1/sigma) (section 6).

## 6. H1b: correction of a wrong order through a strict saddle  [derived; Hessian checked]

At B (component 2 learned, amplitude q = s_2^(1/K) per core), adding x e_1 and y e_1 to the two
component-specific slices in the learned bond index gives
L - L_B = 1/2 q^(2(K-1)) (x^2 + y^2) - s_1 q^(K-2) x y + O(4), Hessian q^(K-2) [[s_2, -s_1], [-s_1, s_2]],
lambda_min = -(s_1 - s_2) s_2^((K-2)/K) (binary4: -(1-rho) rho^(1/3)). Exact gauge-projected Hessian at
B (library `ttn.diagnostics`): binary4, s = (0.8, 1): -0.18566 (prediction -0.18566); s = (0.5, 1):
-0.39685 (-0.39685); MPS4 odeco (m = 4): no negative eigenvalue (`notes/theory_saddles.md` 4.1 has the
same block). Predicted sequence for m = 2: L_0 -> s_1^2/2 (wrong plateau) -> rotation of the learned bond
index to component 1 after ln(1/delta_1)/|lambda_min| -> plateau s_2^2/2 with a_2 ~ 0 (the smaller
component is forgotten) -> relearning of component 2 by a k-core route (binary4: k = 3, time ~ sigma^-1)
-> 0. For m >= 3 no rotation: the larger component is added in a new bond index.

## 7. Results

### 7.1 Step 1: order statistics (complete)
See `agent-runs/order/REPORT.md` (Results, step 1) for the table and numbers; data
`experiments/order/results/main_summary.csv`, figure `results/main_pg_vs_rho.png`. Full GF, sigma = 0.01,
400 seeds x 6 rho x 28 settings.

### 7.2 sigma dependence: m = 2 versus m >= 3 (complete)

Full GF (RK45 with event location, rtol 1e-9, atol 1e-14), d = 2. Commands `sweep.py configs/b4_sigma.yaml`
(binary4, chi 2 and 4, 1000 seeds, sigma 1e-4..0.1), `m4_sigma.yaml` (MPS4, chi 4, 400 seeds, sigma
0.003-0.3), `b8_sigma.yaml` (binary8, chi 4, 400 seeds, sigma 0.01-0.3); `analyze_sigma.py`;
`results/sigma_table.csv`, `results/sigma_fits.csv`, `results/sigma_scan.png`.

- **m >= 3: P(greedy) does not depend on sigma** [empirical, confirms O1]. MPS4: P(smaller first) =
  0.185 / 0.395 / 0.490 at rho = 0.5 / 0.8 / 0.95 for every sigma in 0.003-0.1; binary8: 0.240 / 0.4075 /
  0.4875 for every sigma in 0.01-0.1. Per seed the winner is the same at all sigma <= 0.1 for 99.0-100% of
  seeds (the few changes are at sigma = 0.3). Fitted gamma = 0.000 +- 0.004.
- **m = 2 (binary4): P(smaller first) = C sigma^gamma** [empirical; exponent derived in section 5].
  Fits over sigma = 1e-4..1e-2 (binomial ML, SE from a bootstrap over seeds, because the same seeds are
  used at every sigma):

  | rho | (1-rho)/rho | gamma, chi = 2 | gamma, chi = 4 |
  |---|---|---|---|
  | 0.7 | 0.429 | 0.35 +- 0.05 | 0.45 +- 0.09 |
  | 0.8 | 0.250 | 0.232 +- 0.027 | 0.232 +- 0.028 |
  | 0.9 | 0.111 | 0.110 +- 0.010 | 0.105 +- 0.010 |
  | 0.95 | 0.053 | 0.045 +- 0.005 | 0.041 +- 0.005 |

  Agreement within 1.5 SE for rho = 0.7-0.9; rho = 0.95 is 1.6-2.4 SE low (P(smaller first) >= 0.33 at
  all sigma, i.e. far from the small-P regime the derivation assumes). rho = 0.5: wrong order only at
  sigma >= 3e-3 (P = 0.033 / 0.021 at sigma = 0.1 for chi 2 / 4; 0 of 1000 at sigma <= 1e-3).
  Wrong-order seeds at small sigma are (nearly) a subset of those at large sigma, as the threshold
  picture (delta_0 < sigma^gamma) predicts. Values agree with the independent GF table of
  `notes/theory_escape.md` 5.5 (rho = 0.7, d = chi = 2: 0.137 / 0.054 / 0.026 here vs their estimate
  0.155 / 0.064 / 0.024 at sigma = 0.1 / 0.01 / 0.001).

### 7.3 H1b (complete)

Full GF with LSODA and the exact Hessian (rtol 1e-10, atol 1e-15), event times by root finding on the
dense output, gauge-projected Hessian (`ttn.diagnostics`) at the minimum-speed point between the two
events. `h1b.py configs/h1b_*.yaml`, `analyze_h1b.py`; `results/h1b_*.csv`, `results/h1b_fits.csv`,
figures `results/h1b_correction_time.png`, `results/h1b_natural.png`, `results/h1b_example.png`.

1. **Controlled start at the wrong saddle, binary4 (m = 2)**: component 2 learned exactly (balanced,
   amplitude rho^(1/3)) plus a fan-in perturbation of scale sigma; 8 seeds x 6 sigma (1e-7..1e-2) x 4 rho,
   d = 2, chi = 4. In all 192 runs the learned bond index rotates to component 1 (a_2 falls below 6e-4,
   the second singular value stays ~0), the loss then sits at s_2^2/2 = 0.125 / 0.245 / 0.32 / 0.405, and
   component 2 is relearned after ~sigma^-1. **Strict saddle by the primary rule**: lambda_min at the
   plateau midpoint is sigma-independent (per-seed exponent of |lambda_min| vs sigma: 0.000) and equals
   -0.39685 / -0.26637 / -0.18566 / -0.09655 to 5 digits for rho = 0.5 / 0.7 / 0.8 / 0.9 (prediction
   -(1-rho) rho^(1/3): identical). **Correction time = ln(1/sigma)/|lambda_min| + const**: per-seed
   slopes of t_corr vs ln(1/sigma) 2.555 +- 0.036, 3.809 +- 0.056, 5.466 +- 0.082, 10.51 +- 0.16 vs
   1/|lambda_min| = 2.520, 3.754, 5.386, 10.357 (ratio 1.014-1.015, within 1 SE). A power law
   t = c sigma^-p fits 10x worse in the same space (rms 1.5-6.0 vs 0.13-0.56; p = 0.08).
2. **Controlled start, MPS4 (m = 4)**: no rotation (a_2 >= 0.96 when a_1 crosses 1/2), component 1 is
   added in a new bond index; |lambda_min| at the midpoint scales as sigma^2.007 +- 0.016 (rho 0.5) and
   sigma^2.000 +- 0.014 (rho 0.8), i.e. a higher-order saddle (k - 2 = 2); t_1 ~ sigma^-1.97 +- 0.03. The
   fixed 1e-3 threshold would call some sigma = 0.05-0.1 midpoints strict (n_neg up to 1.6 on average),
   the scaling rule does not.
3. **Natural wrong-order runs, binary4** (the seeds whose first event was component 2 in b4_sigma, chi = 4;
   up to 40 per (rho, sigma), rho 0.7-0.9, sigma 1e-4..0.1; 421 runs): for sigma <= 0.01, 92.5-100% rotate
   (the rest add component 1 in a new index); at sigma = 0.1 only 2.5-42% rotate (the 3-core add route,
   ~sigma^-1 = 10, competes). For the rotating runs |lambda_pred| (t_1 - t_2) has mean 1.07-1.38
   (SE 0.14-0.30) at every sigma from 1e-4 to 1e-2: **no ln(1/sigma) growth**, as predicted in section 5
   (a log law would raise it by ln 100 = 4.6 over this range). The wrong "plateau" is not resolved:
   separation ratio (t_1 - t_2)/max(width) has median 0.3-0.6, > 3 in 0-7.5% of runs; lambda_min at the
   minimum-speed point lies between 0 and the saddle value. So in natural runs a wrong order shows up as a
   brief overshoot of the smaller component, then a plateau at s_2^2/2 with the smaller component
   forgotten, then its relearning.
4. **Natural wrong-order runs, MPS4 (m = 4)** (seeds whose first event was component 2 in the main sweep,
   d = 2, chi = 4, sigma = 0.01, 20 per rho = 0.5, 0.8): no rotation (a_2 >= 0.999 when a_1 crosses 1/2);
   component 1 is added in a new bond index after a median gap of 2.4e4 / 2.6e4 (~sigma^-2: a 4-core
   route); lambda_min at the plateau midpoint has median -8.7e-5 / -8.5e-5 (raw -9.0e-5 / -8.6e-5),
   |lambda_min|/lambda_max ~ 6e-5 (single-sigma fallback rule: non-strict; the sigma scaling in item 2 is
   the primary evidence). **One run stalled** (rho 0.8, seed 25): after component 2 is learned, component
   1's overlap decays (1e-12 at t = 1e5, 1e-50 at t = 1e7): its route dies (the conservation-law trap of
   `notes/theory_saddles.md` 8) and GF stays at the wrong saddle (L = 0.5) indefinitely (run stopped when
   a_1 < 1e-25). So for m >= 3 a wrong order is either kept (larger component learned later) or, rarely,
   permanent.

All lambda_min values above were computed with the fixed gauge projection (library commit 6c42d85,
rtol 1e-3) and are reported with the raw (unprojected) value in `results/h1b_*.csv`
(`lam_min_raw`); for binary4 the two agree to 6 digits at sigma <= 1e-3.

### 7.4 Control: "matrix-like" (m = 2) versus tensor-like pattern teachers on one tree (complete)

MPS6, d = 2, chi = 4 (bonds 2-4-4-4-2), pattern teachers (component 1 = e_0 at every site, component 2
differs from it on the sites listed, weight rho), full GF to the first event, 400 seeds, sigma 0.003-0.1
(`configs/pat_mps6_*.yaml`, `analyze_sigma.py`, `results/sigma_table.csv`, `results/sigma_fits.csv`).
P(smaller first) at sigma = 0.01 (Wilson 95% CI half-width <= 0.05):

| differing sites (cores) | m | rho = 0.7 | 0.8 | 0.9 | sigma dependence |
|---|---|---|---|---|---|
| 2, 3 (adjacent, share a bond of dim 4) | 2 | 0.000 | 0.045 | 0.20 | gamma = 0.67 +- 0.09 (rho 0.8), 0.22 +- 0.03 (0.9) |
| 0, 5 (the two ends) | 2 | 0.065 | 0.17 | 0.33 | gamma = 0.46 +- 0.06, 0.249 +- 0.029, 0.101 +- 0.014 |
| 0, 1, 2 | 3 | 0.22 | 0.32 | 0.45 | constant for sigma <= 0.03 (+0.01-0.015 at sigma = 0.1) |
| 0, 1, 2, 3 | 4 | 0.33 | 0.41 | 0.47 | identical at every sigma (100% of seeds) |
| all six (odeco, main sweep) | 6 | 0.40 | 0.46 | 0.49 | sigma-independent (section 7.2) |

- **The control behaves as predicted** [empirical, derived]: the non-adjacent m = 2 pair follows
  sigma^((1-rho)/rho) (0.429 / 0.25 / 0.111 predicted) in full GF on a K = 6 tree, i.e. it is greedy as
  sigma -> 0; m >= 3 pairs are random and sigma-independent.
- **P(smaller first) increases monotonically with m** at fixed tree, sigma and rho [empirical]: the more
  cores the components differ on, the more random the order (m = 3 already far from greedy).
- **Adjacent m = 2 pairs are more greedy than non-adjacent ones**: their measured exponents (0.67, 0.22)
  are 2.7x and 2.0x the non-adjacent ones, below the asymptotic prediction n (1-rho)/rho with n = chi_e
  = 4 (1.0, 0.44) of `notes/theory_escape.md` 5.3 over this sigma range (their LO emulation shows the
  same slow approach: local slope 0.24 at sigma ~ 10^-1.5 rising to ~0.45 at 1e-4 for n = 4, rho = 0.9).
  A full-GF test down to sigma = 1e-5 on MPS4 is in 7.4b.

### 7.4b MPS4 multiplicity test (pending)

### 7.5 Over-parameterized bonds restore greedy order, slowly (complete)

Leading-order race (per-component blow-up times; exact for odeco on MPS, 96-99% for binary8), d = 2,
uncapped bond dimension chi = 1..32 on every edge (`cap: false`: the end bonds exceed the rank they can
carry), 500 seeds per point (`sweep.py configs/chi_scan.yaml`, `analyze_chi.py`, `results/chi_scan.csv`,
`results/chi_scan_joint_fit.csv`, `results/chi_scan.png`); GF check at chi = 16, 32 in `chi_gf.yaml`.

- sd(log tau) falls like chi^-1/2 on every tree: per-tree exponents 0.48-0.54 (chi >= 4). Joint fit over
  all trees and chi >= 2: **sd(log tau) = 0.57 (K-2)^(0.61 +- 0.03) chi^-(0.53 +- 0.01)** (rms residual 5%
  in log) [empirical]. (`notes/theory_escape.md` 5.1 reports ~sqrt(K-2) at capped bonds.)
- Race P(larger first) at chi = 32: 1.00 / 0.86 / 0.70 (MPS4), 0.99 / 0.78 / 0.64 (MPS6), 0.97 / 0.73 / 0.61
  (MPS8) at rho = 0.5 / 0.8 / 0.9. With a log-normal tau, P = Phi(ln(1/rho)/(sqrt 2 sd)), so P >= 0.9 needs
  sd <= ln(1/rho)/1.81, i.e. chi >= [1.03 (K-2)^0.61 / ln(1/rho)]^1.89 by the fit: for rho = 0.9,
  chi ~ 160 (K = 4) and ~570 (K = 8) [extrapolation of the empirical fit].
- **GF check** (sigma = 0.01, 400 seeds, `chi_gf.yaml`, `results/chi_gf.csv`): MPS4 chi = 32: P = 0.84 /
  0.68 / 0.58 at rho = 0.8 / 0.9 / 0.95 (race curve 0.86 / 0.70 / 0.60), per-seed race agreement 100%;
  binary8 chi = 32: 0.745 / 0.62 / 0.565 (race 0.72 / 0.61 / 0.55), agreement 99.5%; chi = 16 likewise.
- So the answer to "does larger chi restore greedy order?" is yes in the limit, as for over-parameterized
  CP (Ge et al. 2021), but slowly: the spread of log escape times shrinks only like chi^-1/2.

## 8. Status of H1 and H1b

**H1 ("the first escape direction is decided by the basin of the init, so the order is random; P(largest
first) increases with s_1/s_2, decreases with K, may concentrate as chi grows")**
- **Holds for tensor-like pairs (m >= 3)** [empirical over 5 trees, 28 (d, chi) settings, 67200 GF runs;
  derived at leading order; exact race on MPS]. Quantitatively: P(larger first) = 0.60-0.88 at rho = 0.5
  and 0.50-0.65 at rho = 0.8 (sigma = 0.01); independent of sigma (7.2); the per-init winner is the
  argmin of tau_r/s_r of the leading-order race (100% on MPS).
- **K and chi** [empirical]: the spread sd(log tau) grows with K (~sqrt(K-2), as in
  `notes/theory_escape.md` 5.1) and falls with chi; with standard bond caps it saturates at fixed d, with
  uncapped (over-parameterized) bonds it falls like chi^-1/2 (7.5), so over-parameterized bonds restore
  greedy order, slowly (the analogue of over-parameterized CP, Ge et al. 2021).
- **Fails for matrix-like pairs (m = 2)**, which include binary N = 4 (a depth-3 DLN): only the largest
  component is a local maximum and the order is greedy as sigma -> 0 [derived; known for DLNs], with a
  slow power law P(smaller first) ~ sigma^((1-rho)/rho) at finite sigma [derived heuristically; confirmed
  in GF for rho = 0.7-0.9 on binary4 and MPS6].
- **"init overlap enters with power K-2"** [empirical]: with the overlap defined as the projection of the
  init onto the component's KKT orbit (nearest balanced bond-rank-one configuration), log tau_r ~ -q log
  X_r with q ~ K-2, and "argmax s_r X_r^(K-2)" predicts 76-92% of winners (the rest needs the full
  alignment dynamics, which the LO race captures exactly).

**H1b ("a wrong order is corrected through a strict saddle with lambda = q (s_learned - s_new), in time
~ log(1/sigma), before the second rank is added")**
- **Holds for m = 2, with the stated eigenvalue** [derived; checked]: from the wrong saddle the gauge-
  projected Hessian has lambda_min = -(s_1-s_2) s_2^((K-2)/K) (5-digit agreement), sigma-independent
  (strict by the primary rule); the learned bond index rotates to the larger component after
  t_corr = ln(1/sigma)/|lambda_min| + const (slopes within 1.5% / 1 SE), the smaller component is then
  forgotten (a_2 < 1e-3) and relearned in a new bond index after ~sigma^-(k-2) (k = 3 on binary4).
- **But in natural runs the "log(1/sigma)" part does not appear** [derived + empirical]: the wrong order
  itself becomes rare as sigma -> 0, and when it happens the piggyback amplitude is O(1) on arrival, so the
  correction takes O(1/|lambda_min|) (|lambda| (t_1 - t_2) = 1.1-1.4 at every sigma 1e-4..1e-2) and no
  wrong plateau is resolved (separation ratio < 3 in 92-100% of runs). The visible signature is an
  overshoot of the smaller component, then a plateau at s_2^2/2 with the smaller component forgotten.
- **Fails for m >= 3** [derived; checked]: the wrong-order state is a higher-order saddle (|lambda_min| ~
  sigma^2 on MPS4), no rotation happens, the larger component is added in a new bond index after
  ~sigma^-(k-2); the wrong order stays. Rarely (1/40 MPS4 runs at sigma = 0.01) the larger component's
  route dies and GF stalls at the wrong saddle.

## 9. Relation to the literature; what is new

Known (cited in `notes/literature.md`): directional convergence near the origin to a KKT point of g
(Kumar & Haupt 2025a); first escape of a DLN goes to the top mode (Jacot et al. 2021, Thm 2), and
non-top rank-r critical points of DLNs are strict saddles (Achour et al. 2024, Thm 7) — binary4 is a
depth-3 DLN, so the binary4 results are instances of these; odeco components are local optima of
rank-one approximation (Anandkumar et al. 2014); order set by weight x (init overlap)^(p-2) in tensor PCA
(Ben Arous et al. 2024a) and related models; over-parameterized CP learns largest first (Ge et al. 2021).
New here for TTNs (together with `notes/theory_escape.md` and `notes/theory_saddles.md`, which derived
parts of the same picture independently): the m-count classification of which components are attractors
of the first escape on a general tree; the exact, sigma-independent race on MPS and its verification on
40800 GF runs; quantitative P(larger first) tables versus rho, K, tree, chi, d; the finite-sigma
wrong-order law sigma^((1-rho)/rho) for matrix-like pairs measured in full GF; a measured selection rule
with exponent K-2 for the KKT-projection overlap; H1b dynamics measured (eigenvalue, log-law correction
from the saddle, forget-and-relearn, O(1/lambda) correction in natural runs, no correction for m >= 3).

## 10. Open issues

- Prefactor C(rho) of the m = 2 law and the pre-asymptotic regime (rho = 0.95 never reaches small P).
- Adjacent m = 2 pairs: is the asymptotic exponent n (1-rho)/rho with n = chi_e reached in GF (7.4b)?
- Why the KKT-projection rule misses 8-24% of winners: the alignment phase mixes the non-aligned part of
  the init into the escape direction; a closed form for the distribution of tau on a chain is open.
- R >= 3: stalls of later components (conservation-law trap) at chi = R; see 7.6.
