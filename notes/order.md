# Order of learning (Q4; hypotheses H1, H1b) — agent `order`

Status: **complete (round 1 + fix round after review + errata after verification, section 12)**,
2026-10-10. Experiment folder `experiments/order/`
(README there: configs, commands, cost). Tags: **proven**, **derived** (assumptions stated),
**empirical**, **conjecture**. Companion theory notes: `notes/theory_escape.md` (v2; sections 3, 5 cover
the same ground from the escape side) and `notes/theory_saddles.md` (v2; section 4, H1b).

## 0. Results at a glance

Revised in the fix round after `notes/reviews/order_round1.md`; the changed claims are listed in
section 11. The errata of the verification (`notes/reviews/order_round2_verification.md`, E1-E10) are
applied in place and listed in section 12; its "Safe to cite" table (C1-C17) is the final list of
citable claims.

| # | claim | status | evidence |
|---|---|---|---|
| O1 | Near the origin the order is decided by the leading-order (LO) flow of g = <W, W*>, whose outcome does not depend on sigma (exact scaling symmetry) | derived (corollary of Kumar & Haupt 2025a) | per-seed GF = LO race on 40800/40800 MPS runs; P identical for sigma 0.003-0.1 (MPS4), 0.01-0.1 (binary8), MPS6 m = 4 pattern |
| O2 | m >= 3 (cores carrying the differing sites): every component is a local max of g on the sphere, order random (H1) | derived | P(larger first) = 0.60-0.88 at rho = 0.5, 0.51-0.65 at 0.8 on MPS4-8 and binary8 (sigma = 0.01, 400 seeds per setting) |
| O3 | Odeco teacher on an MPS: the LO flow decouples into one independent chain per component; winner = argmin tau_r/s_r | derived (exact at leading order) | 40800/40800 MPS GF runs agree with the per-seed race; binary8 96-99% |
| O4 | P(greedy) falls toward 1/2 with K and rises slowly with chi; on an MPS d enters only through the bond caps | empirical (K, chi); derived (d, MPS); empirical at one chi (d, binary8) | `main_settings.csv`; independent-seed d test (7.1b) |
| O5 | Selection rule: argmax_r s_r X_r^q, X_r = projection of the init on component r's KKT orbit; q = (0.99-1.25)(K-2) (bootstrap SE 1-4%); predicts 76-92% of winners out of sample on m >= 3 trees, the same with q fixed to K-2 | empirical | `results/main_rules.csv` (columns q_se, order_accuracy_oos, order_accuracy_q_Km2) |
| O6 | m = 2 (binary4 = depth-3 DLN; matrix-like pattern pairs): greedy as sigma -> 0; P(smaller first) consistent with sigma^(n(1-rho)/rho), n = 1 for non-adjacent pairs (fits over the three smallest sigma agree with (1-rho)/rho within 1.1 SE in all 13 non-adjacent sets; the full-range fits are low in 12/13, pre-asymptotic); adjacent pairs (shared bond chi_e) are 2-3.4x steeper; n = chi_e is a conjecture, supported at the small-sigma end on MPS4 (rho = 0.9: 0.40 +- 0.05 for sigma = 1e-3..1e-5 vs 0.444), not reached on MPS6 down to sigma = 0.003 | heuristic derivation; empirical | 7.2, 7.4, 7.4b |
| O7 | H1b, after a wrong first escape: m = 2, k >= 3 (non-adjacent pair): strict saddle lambda_min = -(s_1-s_2) s_2^((K-2)/K), rotation to the larger component in ln(1/sigma)/|lambda_min| from the saddle (O(1/|lambda_min|) in natural runs), smaller one forgotten and relearned; m = 2, k = 2 (adjacent pair): strict saddle lambda = -s_1 s_2^((K-2)/K) (multiplicity chi_e - 1), the larger component is added, nothing forgotten; m >= 3: no strict saddle, wrong order kept | derived; Hessians and dynamics checked on binary4 and MPS4 (m = 2); MPS4, MPS6 and binary8 (m >= 3) | 7.3 |
| O8 | Over-parameterized (uncapped) bonds: sd(log tau) ~ sqrt(K-2)/sqrt(chi) at large chi; greedy order restored only slowly | empirical | 7.5 |
| O9 | R = 3 (d = chi = 4, one sigma per tree): the full order is a random permutation favouring large components, consistent with an iid race model in every m >= 3 case (\|z\| <= 1.6) and, paired against each run's own LO race on the same runs, no less greedy (41 vs 43 discordant runs; the fix round's "less greedy in 6/6, combined z = -1.9" is withdrawn, section 12); R = 4 = chi: most runs do not learn all four components by t = 1e9 (seen directly in the t_max and no-stop runs, inferred for the others; MPS6 often only 2 of 4) | empirical | 7.6 |

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

## 5. m = 2 at finite sigma: wrong order with probability ~ sigma^(n(1-rho)/rho)  [heuristic derivation; n = 1 for non-adjacent pairs, n = chi_e for adjacent pairs is a conjecture]

Near B the distance from B's stable manifold grows like exp(lambda_u tau) while ln R grows at rate
K g(B); lambda_u / (K g(B)) = (s_1 - s_2)/s_2 = (1 - rho)/rho. A trajectory entering at distance delta_0
leaves at radius R_0 delta_0^(-rho/(1-rho)); the smaller component wins iff R reaches O(1) first, i.e.
delta_0 < sigma^((1-rho)/rho) in each of the n unstable directions. With a bounded density of delta_0:
P(wrong) ~ C sigma^(n (1-rho)/rho). Corollary: conditioned on a wrong order, the residual deviation
delta_1 = delta_0 sigma^(-(1-rho)/rho) is O(1)-distributed (roughly uniform), so the subsequent correction
takes O(1/|lambda|), not ln(1/sigma) (section 6).

## 6. H1b: what happens after a wrong first escape  [derived; Hessians checked]

Revised after review (`notes/reviews/order_round1.md`, item 1). The answer depends on the pair, through
m (cores carrying differing sites) and k (cores in the Steiner subtree of those sites; for m = 2, k = 2
iff the two cores are adjacent) (`notes/theory_saddles.md` 4.3-4.4):

- **m = 2, k >= 3 (non-adjacent pair; binary4, MPS cores 0,2, MPS ends): rotation.** At B (component 2
  learned, amplitude q = s_2^(1/K) per core), adding x e_1 and y e_1 to the two component-specific slices
  in the learned bond index gives L - L_B = 1/2 q^(2(K-1)) (x^2 + y^2) - s_1 q^(K-2) x y + O(4), so the
  piggyback block has lambda_min = -(s_1 - s_2) s_2^((K-2)/K) (binary4: -(1-rho) rho^(1/3)). B is a strict
  saddle; the learned bond index rotates to component 1 after ln(1/delta_1)/|lambda_min|, component 2 is
  forgotten (a_2 -> 0), the loss sits at s_2^2/2, and component 2 is relearned by a k-core route after
  ~sigma^-(k-2).
- **m = 2, k = 2 (adjacent pair sharing a bond of dimension chi_e > 1): adding.** The 2-core route that
  puts component 1 on a spare index of the shared bond has lambda = -s_1 s_2^((K-2)/K), multiplicity
  chi_e - 1, more negative than the piggyback value (which is still present, as the chi_e-th eigenvalue).
  Gradient flow adds component 1 on the spare index after ln(1/sigma)/(s_1 s_2^((K-2)/K)); nothing rotates
  and nothing is forgotten (a_2 stays 1). This branch was missed in round 1 (the reviewers found it).
- **m >= 3: no strict saddle** toward the larger component (gain O(eps^m)); the larger component is added
  in a new bond index by a k-core route after ~sigma^-(k-2); the wrong order is kept. (Exception found
  later: when the differing sites cover two leaf pairs of a binary tree, e.g. binary8 odeco, the saddle
  order is 6 < k = 7 and below a crossover sigma the smaller component is forgotten; section 8 and
  `notes/theory_saddle_order.md` v2, section 7 item 5.)

Checks of the eigenvalues at the exact wrong saddle (library gauge projection, fixed cut 6c42d85; raw
values identical here): binary4 s = (0.8, 1): -0.18566, (0.5, 1): -0.39685 (both = prediction); MPS4
cores 1,2 (k = 2), rho = 0.8: -0.89443 three times (= -s_2^(1/2), multiplicity chi_e - 1 = 3), fourth
-0.17889 (piggyback); MPS4 cores 0,2: -0.17889 (rotation value); MPS4 odeco (m = 4): no negative
eigenvalue.

## 7. Results

### 7.1 Step 1: order statistics (complete)

Full GF, sigma = 0.01, 400 seeds x 6 rho x 28 (tree, d, chi) settings with standard bond caps
(`configs/main.yaml`; `results/main_summary.csv`, `main_settings.csv`, `main_d_pairs.csv`,
`main_rules.csv`, `main_heuristic.csv`; figures `main_pg_vs_rho.png`, `main_spread.png`, `main_rules.png`).
Accuracy (`tol_check.py`, `results/tol_check.csv`): rtol 1e-9 vs 1e-11 give the same winner on 400/400
inits (4 trees), |delta t1|/t1 < 8e-10, conservation drift at the event < 4e-11.

P(larger first), rho = 0.5 / 0.8 / 0.9 (binomial SE <= 0.025):

| tree (K, m) | (d, chi) = (2,2) | (2,4) | (4,4) | (4,8) |
|---|---|---|---|---|
| binary4 (3, 2) | 0.99 / 0.86 / 0.69 | 1.00 / 0.87 / 0.70 | 1.00 / 0.91 / 0.76 | 1.00 / 0.92 / 0.77 |
| mps4 (4, 4) | 0.79 / 0.60 / 0.54 | 0.82 / 0.61 / 0.55 | 0.87 / 0.61 / 0.52 | 0.88 / 0.65 / 0.55 |
| mps6 (6, 6) | 0.69 / 0.55 / 0.52 | 0.73 / 0.54 / 0.52 | 0.81 / 0.61 / 0.56 | 0.82 / 0.62 / 0.53 |
| binary8 (7, 4) | 0.70 / 0.58 / 0.53 | 0.76 / 0.59 / 0.54 | 0.78 / 0.59 / 0.51 | 0.86 / 0.64 / 0.57 |
| mps8 (8, 8) | 0.63 / 0.51 / 0.50 | 0.71 / 0.58 / 0.54 | 0.73 / 0.55 / 0.52 | 0.76 / 0.58 / 0.53 |

- Race model (O3) per seed: 100% of the 40800 MPS runs, 96-99% of binary8 runs; the race curve
  P(tau_2/tau_1 > rho) from the pooled blow-up times overlays the GF points (figure). The log-normal
  approximation Phi(ln(1/rho)/(sqrt 2 sd log tau)) is within the binomial error of the race curve.
- sd(log tau) (`main_settings.csv`): MPS d = 2, chi = 2: 0.59 / 0.93 / 1.20 for K = 4 / 6 / 8; MPS8 d = 4:
  1.19 / 0.76 / 0.66 for chi = 2 / 4 / 8 (agrees with `notes/theory_escape.md` table 2 within ~0.03).
- d: on an MPS, d only rescales every component slice by the same factor, so it matters only through
  the bond caps [derived]. The round-1 "chi^2 with 6 dof" comparison of d = 2 vs d = 4 settings was
  invalid (the 6 rho rows reuse the same seeds, and d = 2 and d = 4 inits share the first core's random
  draws; review item 6) and is withdrawn; an independent-seed test is in 7.1b.
- Common seeds: GF P(larger first) sits 0.018 +- 0.019 below the pooled race curve on average over the
  102 MPS (setting, rho) cells (76% negative), although the per-seed race agreement is 100%: the 400
  seeds' realized pairs fluctuate around the all-pairs curve, and the same seeds (and shared first-core
  draws) are reused across settings (`results/main_race_offset.csv`). On 1500 fresh seeds (MPS4, d = 4,
  chi = 2) the offset has the opposite sign (0.81 realized vs 0.79 curve). Ranges over the m >= 3
  settings: GF 0.60-0.88 (rho = 0.5), 0.51-0.65 (0.8); race curve 0.66-0.89 and 0.55-0.66.
- Selection rule (O5): see section 8.

### 7.1b d test with independent seeds (fix round)

d = 2 vs d = 4 at identical bond dimensions, full GF, sigma = 0.01, 1000 seeds each, disjoint seed ranges
(200000+ for d = 2, 300000+ for d = 4: no shared random draws), two-sample z test per rho
(`configs/d_test_*.yaml`, `analyze_dtest.py`, `results/d_test.csv`):

| tree, bonds | rho | P(larger first), d = 2 | d = 4 | z |
|---|---|---|---|---|
| binary8, 4-4-4-4-4-4 | 0.5 / 0.8 | 0.774 / 0.591 | 0.774 / 0.572 | 0.00 / -0.86 |
| MPS4, 2-2-2 | 0.5 / 0.8 | 0.788 / 0.606 | 0.809 / 0.621 | +1.17 / +0.69 |

No d dependence is detected at equal bonds (|difference| <= 0.021, two-sample SE 0.018-0.022);
differences larger than about 0.06 are excluded at 95% (95% CIs on the difference, in table order:
[-0.037, +0.037], [-0.062, +0.024], [-0.014, +0.056], [-0.028, +0.058]; verification E3): derived for
MPS, empirical (at this resolution, one chi) for binary8.

### 7.2 sigma dependence: m = 2 versus m >= 3 (complete)

Full GF (RK45 with event location, rtol 1e-9, atol 1e-14), d = 2. Commands `sweep.py configs/b4_sigma.yaml`
(binary4, chi 2 and 4, 1000 seeds, sigma 1e-4..0.1), `m4_sigma.yaml` (MPS4, chi 4, 400 seeds, sigma
0.003-0.3), `b8_sigma.yaml` (binary8, chi 4, 400 seeds, sigma 0.01-0.3); `analyze_sigma.py`;
`results/sigma_table.csv`, `results/sigma_fits.csv`, `results/sigma_scan.png`.

- **m >= 3: P(greedy) does not depend on sigma** [empirical, confirms O1]. MPS4: P(smaller first) =
  0.185 / 0.395 / 0.490 at rho = 0.5 / 0.8 / 0.95 for every sigma in 0.003-0.1; binary8: 0.240 / 0.4075 /
  0.4875 for every sigma in 0.01-0.1. Per seed the winner is the same at all sigma <= 0.1 for 99.0-100% of
  seeds (the few changes are at sigma = 0.3). Fitted gamma = 0.000 +- 0.004.
- **m = 2 (binary4): P(smaller first) is consistent with C sigma^gamma, gamma = (1-rho)/rho**
  [empirical; exponent from the heuristic of section 5, n = 1]. One fit range for every m = 2 set: the
  full measured sigma range (here 1e-4..0.1; binomial ML, SE from a bootstrap over seeds because the same
  seeds are used at every sigma); the fit over sigma <= 0.01 is given as a sensitivity check:

  | rho | (1-rho)/rho | gamma, chi = 2 (full; sigma <= 0.01) | gamma, chi = 4 (full; sigma <= 0.01) |
  |---|---|---|---|
  | 0.7 | 0.429 | 0.370 +- 0.034; 0.35 +- 0.06 | 0.42 +- 0.05; 0.45 +- 0.08 |
  | 0.8 | 0.250 | 0.237 +- 0.018; 0.232 +- 0.025 | 0.248 +- 0.018; 0.232 +- 0.028 |
  | 0.9 | 0.111 | 0.103 +- 0.008; 0.110 +- 0.010 | 0.102 +- 0.008; 0.105 +- 0.010 |
  | 0.95 | 0.053 | 0.042 +- 0.004; 0.045 +- 0.005 | 0.040 +- 0.004; 0.041 +- 0.005 |

  Within 1.8 SE for rho = 0.7-0.9; rho = 0.95 is 2.6-3.2 SE low (P(smaller first) >= 0.33 at all sigma,
  far from the small-P regime of the derivation). Local slopes between consecutive sigma (seed bootstrap
  SE in `results/sigma_fits.csv`) scatter around the prediction without a trend, e.g. rho = 0.8, chi = 2:
  0.28, 0.22, 0.22, 0.23, 0.23, 0.25 (+-0.04-0.07). Goodness of fit (verification E5; replaces the
  binomial deviance, which is uninformative because the seeds are shared across sigma): fits over the
  three smallest sigma agree with (1-rho)/rho within 1.1 SE in all 13 non-adjacent m = 2 sets (binary4
  here, MPS6 ends in 7.4, MPS4 cores 0,2 in 7.4b), while the full-range fits are low in 12/13, so the
  full-range deficit is pre-asymptotic. rho = 0.5: wrong order only at sigma >= 3e-3 (P = 0.033 /
  0.021 at sigma = 0.1 for chi 2 / 4; 0 of 1000 at sigma <= 1e-3). The wrong-order seeds are exactly nested: every seed
  that comes out in the wrong order at a smaller sigma does so at all larger sigma (100% in all 8 (chi,
  rho) sets; column `nested_frac_min`), as the threshold picture delta_0 < sigma^gamma predicts. Values
  agree with the independent GF table of `notes/theory_escape.md` 5.5 (rho = 0.7, d = chi = 2: 0.137 /
  0.054 / 0.026 here vs their estimate 0.155 / 0.064 / 0.024 at sigma = 0.1 / 0.01 / 0.001).

### 7.3 H1b (complete; revised after review)

Full GF with LSODA and the exact Hessian (rtol 1e-10, atol 1e-15), event times by root finding on the
dense output (full precision; a fine record window around the events), gauge-projected Hessian
(`ttn.diagnostics`, fixed cut 6c42d85) and the raw Hessian at the minimum-speed point between the two
events; runs stop as "stalled" if a component's slice of some core falls to 1e-14 x the largest core
norm. d = 2, chi = 4. `h1b.py configs/h1b_*.yaml`, `analyze_h1b.py`; `results/h1b_*.csv`,
`results/h1b_fits.csv`, `results/h1b_local_slopes.csv`; figures `results/h1b_correction_time.png`,
`results/h1b_natural.png`, `results/h1b_example.png`.

**Saddle starts** (component 2 learned exactly, balanced, plus a fan-in perturbation of scale sigma;
8 seeds x 6 sigma 1e-7..1e-2; per-seed fits, mean +- SE over seeds; slope of t1 against ln(1/sigma)):

| pair | rho | class (runs) | slope | 1/|lambda_pred| | lambda_min at the midpoint (raw) | lambda_pred |
|---|---|---|---|---|---|---|
| binary4 (m = 2, k = 3) | 0.5 / 0.7 / 0.8 / 0.9 | rotated 192/192 | 2.555 / 3.809 / 5.466 / 10.51 (+-1.5%) | 2.520 / 3.754 / 5.386 / 10.357 | -0.39665 / -0.26626 / -0.18559 / -0.09651 | -0.39685 / -0.26637 / -0.18566 / -0.09655 |
| MPS4 cores 0,2 (m = 2, k = 3) | 0.5 / 0.8 / 0.9 | rotated 144/144 | 2.8282 / 5.5903 / 10.541 (+-0.1%) | 2.8284 / 5.5902 / 10.541 | -0.35359 / -0.17890 / -0.09488 | -0.35355 / -0.17888 / -0.09487 |
| MPS4 cores 1,2 (m = 2, k = 2) | 0.5 / 0.8 / 0.9 | added 144/144 | 1.4130 / 1.1171 / 1.0533 (+-0.1%) | 1.4142 / 1.1180 / 1.0541 | -0.70704 / -0.89427 / -0.94850 (x3) | -0.70711 / -0.89443 / -0.94868 |

- **Rotation (m = 2, k >= 3; binary4 and MPS4 cores 0,2)**: the learned bond index rotates to component 1,
  a_2 falls below 6e-4, the loss sits at s_2^2/2, component 2 is relearned after ~sigma^-1. lambda_min is
  sigma-independent (per-seed exponent of |lambda_min| vs sigma: 0.0000 +- 0.0001): strict by the
  primary rule. The fitted slopes of binary4 are 1.4% above 1/|lambda| only because the fit includes
  sigma = 1e-2: the local slope x |lambda| between consecutive sigma is 1.07, 1.018, 1.002, 1.0002,
  1.0000 from sigma = 1e-2 down to 1e-7 (all rho); for MPS4 cores 0,2 it is 1.0000 +- 0.0005 throughout.
  A power law t = c sigma^-p fits 11-88x worse in the same space for the rotation sets (104-110x for
  the adjacent set; verification E9) (rms 1.5-6.0 vs 0.02-0.56).
- **Adding (m = 2, k = 2; MPS4 cores 1,2)**: lambda_min = -s_2^(1/2) = -s_1 s_2^((K-2)/K) with multiplicity
  3 = chi_e - 1 (4 negative eigenvalues: those three plus the piggyback one, e.g. -0.161 at rho = 0.8);
  component 1 is added on a spare index of the shared bond, a_2 stays 1 (minimum 1.0000 in 144/144), and
  t1 follows ln(1/sigma)/(s_1 s_2^((K-2)/K)) (local slope x |lambda| -> 1.0000 at small sigma). At sigma =
  1e-2 the projected lambda_min is biased (-0.58..-0.80; library note on saddle starts at moderate sigma);
  the raw values are quoted.
- **m >= 3 (MPS4 odeco, k = 4; MPS6 odeco, k = 6)**: no rotation (a_2 >= 0.96 resp. >= 0.9996 when a_1
  crosses 1/2); component 1 is added in a new bond index. |lambda_min| at the midpoint scales as
  sigma^2.007 +- 0.016 / sigma^2.000 +- 0.014 (MPS4, rho 0.5 / 0.8; 8 seeds x 5 sigma 0.01-0.1) and
  sigma^4.22 +- 0.09 (MPS6, rho 0.8; 5 seeds x 5 sigma 0.05-0.2): higher-order saddles, |lambda_min| ->
  0 like sigma^(k-2) (MPS6 slightly steeper than 4 over this range). t1 ~ sigma^-1.97 +- 0.03 (MPS4)
  and sigma^-4.26 +- 0.13 (MPS6). One MPS6 seed of six stalls at every sigma (its route dies).
  binary8 (m = 4 leaf cores, k = 7; 6 seeds x 5 sigma 0.15-0.35, a factor 2.3 only; the CSV was rebuilt
  from the stored trajectories because one run, sigma = 0.3 seed 0, had not finished after 80 min and was
  stopped: `rebuild_h1b_csv.py`): no rotation (a_2 >= 0.995), |lambda_min| ~ sigma^5.2 +- 0.3, t1 ~
  sigma^-5.3 +- 0.3 (k - 2 = 5). The fixed 1e-3 threshold would call some MPS4 midpoints at sigma = 0.05-0.1 strict
  (n_neg up to 1.6 on average); the scaling rule does not.

**Natural wrong-order runs** (random init; seeds whose first event was component 2 in the sigma or
pattern sweeps; up to 40 per (rho, sigma)):

| set | sigma | runs | class | |lambda_pred| (t1 - t2), mean | median t1 - t2 | resolved wrong plateau (separation > 3) |
|---|---|---|---|---|---|---|
| binary4 (k = 3), rho 0.7-0.9 | 1e-4..1e-2 | 301 | rotated 296, added 2, other 3 | 1.07-1.38 (SE 0.14-0.29), flat in sigma | 3.0-10.6 | 0-2.5% |
| binary4, rho 0.7-0.9 | 0.1 | 120 | rotated 30, added 52, other 38 | - | 2.7-6.3 | 5-7.5% |
| MPS4 cores 0,2 (k = 3), rho 0.8, 0.9 | 1e-4..1e-2 | 240 | rotated 237, added 1, other 2 | 0.98-1.24 (SE 0.10-0.17), flat | 4.7-12.4 | 0% |
| MPS4 cores 1,2 (k = 2), rho 0.8, 0.9 | 1e-4..1e-2 | 150 | added 150 | 0.19-0.28 | 0.13-0.23 | 0% |
| MPS4 odeco (m = 4), rho 0.5, 0.8 | 1e-2 | 40 | added 39, stalled 1 | - | 2.4e4-2.6e4 | 95-100% |

- **k >= 3 (rotation)**: in natural runs the wrong order is followed by rotation within O(1/|lambda|):
  |lambda| (t1 - t2) is flat in sigma (a ln(1/sigma) law would raise it by 4.6 over this range) and no
  wrong plateau is resolved; the visible signature is an overshoot of the smaller component, then a
  plateau at s_2^2/2 with it forgotten, then its relearning. All 421 + 240 runs now have a resolved
  minimum-speed point (round 1 left 43 binary4 runs at sigma = 1e-4 without one; they are included now, and
  the means barely move: 1.08-1.15 at sigma = 1e-4); the adjacent (k = 2) natural set has 57/150 without
  one (near-ties, median gap 0.06; not used; verification E6). For sigma <= 1e-2, lambda_min there lies
  at or above the saddle value to within 1.3% (26 of 296 binary4 runs and 12 of 237 MPS4 0,2 runs sit
  0-1.3% below it; verification E4; at sigma = 0.1 it is about -0.30, below the saddle value: the point
  is not near the saddle); see `results/h1b_natural.png`.
- **k = 2 (adding)**: the larger component follows within 0.1-1 time units of the smaller (a near-tie);
  a_2 never falls back below 1/2 (median minimum after t1: 0.62-0.68); nothing is forgotten.
- **m >= 3**: the wrong plateau is resolved (separation 750-1080 median); the larger component is added
  after ~sigma^-2 (k = 4); a_2 stays 1; one run of 40 stalls (route dies; component 1's overlap falls to
  1e-50 by t = 1e7). lambda_min at the midpoint: median -2.7e-4 / -1.5e-4 (raw -2.7e-4 / -1.5e-4),
  |lambda_min|/lambda_max ~ 1e-4 (single-sigma fallback; the sigma scaling of the saddle starts is the
  primary evidence).

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

- **The control is consistent with the prediction** [empirical; exponent heuristic]: the non-adjacent
  m = 2 pair (n = 1) gives gamma = 0.46 +- 0.06 / 0.249 +- 0.029 / 0.101 +- 0.014 against (1-rho)/rho =
  0.429 / 0.25 / 0.111 (full measured range 0.003-0.1, as for all m = 2 sets; local slopes in
  `results/sigma_fits.csv`), i.e. it becomes greedy as sigma -> 0; m >= 3 pairs are random and
  sigma-independent.
- **P(smaller first) increases monotonically with m** at fixed tree, sigma and rho [empirical]: the more
  cores the components differ on, the more random the order (m = 3 already far from greedy).
- **Adjacent m = 2 pairs are steeper than non-adjacent ones**: gamma = 0.67 +- 0.09 / 0.22 +- 0.03 (rho =
  0.8 / 0.9), 2.7x and 2.2x the non-adjacent ones, but below n (1-rho)/rho with n = chi_e = 4 (1.0 /
  0.444), the conjectured asymptotic multiplicity of `notes/theory_escape.md` 5.3. Local slopes at
  rho = 0.9: 0.28, 0.18, 0.20 (+-0.05) over sigma 0.1 -> 0.003, i.e. no approach to 0.444 is visible here
  (MPS6 down to sigma = 0.003; on MPS4 the small-sigma end does approach it, 7.4b).

### 7.4b Adjacent versus separated m = 2 pairs on MPS4, down to sigma = 1e-5 (complete)

MPS4, d = 2, chi = 4 (bonds 2-4-2), full GF, 1000 seeds, sigma 1e-5..0.1 (`pat_mps4_adj.yaml`: components
differ on the adjacent cores 1, 2, which share a bond of dimension chi_e = 4; `pat_mps4_sep.yaml`: cores 0,
2). P(smaller first), and gamma over the full range (seed bootstrap SE):

| pair | rho | sigma = 1e-1 | 1e-2 | 1e-3 | 1e-4 | 1e-5 | gamma | local slopes (decade by decade) | n (1-rho)/rho |
|---|---|---|---|---|---|---|---|---|---|
| separated (n = 1) | 0.8 | 0.263 | 0.150 | 0.085 | 0.051 | 0.029 | 0.240 +- 0.017 | 0.244, 0.247, 0.222, 0.245 (+-0.02-0.06) | 0.25 |
| separated (n = 1) | 0.9 | 0.382 | 0.317 | 0.261 | 0.205 | 0.153 | 0.095 +- 0.006 | 0.081, 0.084, 0.105, 0.127 (+-0.01-0.02) | 0.111 |
| adjacent (n = chi_e = 4?) | 0.8 | 0.182 | 0.031 | 0.004 | 0 | 0 | 0.82 +- 0.06 | 0.77, 0.89 (+-0.07, 0.23) | 1.0 |
| adjacent (n = chi_e = 4?) | 0.9 | 0.336 | 0.189 | 0.093 | 0.035 | 0.016 | 0.31 +- 0.02 | 0.25, 0.31, 0.42, 0.34 (+-0.02, 0.03, 0.06, 0.09) | 0.444 |

[empirical] The separated pair is consistent with (1-rho)/rho (rho = 0.9: 2.5 SE low over the full range,
but its local slopes rise from 0.08 to 0.13 across the range and the fit over the three smallest sigma
gives 0.115 +- 0.011; rho = 0.8: 0.231 +- 0.032). The adjacent pair's full-range exponent is 3.2-3.4x the
separated one. At rho = 0.9 its local exponent rises as sigma falls (curvature of log P in ln sigma
-0.013 +- 0.005, 2.4 SE), and the fit over sigma = 1e-3..1e-5 gives **0.40 +- 0.05**: consistent with
n (1-rho)/rho = 0.444 for n = chi_e = 4, and 5 SE above (1-rho)/rho. At rho = 0.8 (0.79 +- 0.07 over the
three smallest sigma with P > 0) and on MPS6 (0.19 +- 0.03 at rho = 0.9 and 0.78 +- 0.15 at 0.8, sigma
>= 0.003; 7.4) the exponent stays below the n = chi_e value. **n = chi_e remains a conjecture**: supported
at the small-sigma end on MPS4 (rho = 0.9), not reached on MPS6 down to sigma = 0.003; qualitatively,
adjacent pairs are 2-3.4x steeper on both trees, as the LO emulation of `notes/theory_escape.md` 5.6
says. (Round 1 wrote "still steepening"; the fix round wrote "not reached by sigma = 1e-5, no steady
approach", which was too pessimistic for MPS4 at rho = 0.9: both superseded, verification E2.)

### 7.5 Over-parameterized bonds restore greedy order, slowly (complete)

Leading-order race (per-component blow-up times; exact for odeco on MPS, 96-99% for binary8), d = 2,
uncapped bond dimension chi = 1..32 on every edge (`cap: false`: the end bonds exceed the rank they can
carry), 500 seeds per point (`sweep.py configs/chi_scan.yaml`, `analyze_chi.py`, `results/chi_scan.csv`,
`results/chi_scan_joint_fit.csv`, `results/chi_scan.png`); GF check at chi = 16, 32 in `chi_gf.yaml`.

- sd(log tau) falls like chi^-1/2 on every tree: per-tree exponents 0.48-0.54 (chi >= 4) [empirical].
  The dependence on K does not separate from chi: on the MPS trees the K-exponent (fit of log sd against
  log(K-2) at fixed chi) falls from 0.66 (chi = 1, 2) to 0.59 / 0.58 (chi = 4, 8) and 0.52 / 0.53
  (chi = 16, 32) (`results/chi_scan_K_exponent.csv`), and binary8 sits 6% above the MPS values. So
  **sd(log tau) ~ sqrt(K-2) / sqrt(chi) at large chi** (as `notes/theory_escape.md` 5.1 finds at capped
  bonds); a single power law 0.57 (K-2)^0.61 chi^-0.53 fits all points to 5% rms but its +-0.03 on the
  K-exponent comes from structured residuals and is not a real uncertainty.
- Race P(larger first) at chi = 32: 1.00 / 0.86 / 0.70 (MPS4), 0.99 / 0.78 / 0.64 (MPS6), 0.97 / 0.73 / 0.61
  (MPS8) at rho = 0.5 / 0.8 / 0.9. With a log-normal tau, P = Phi(ln(1/rho)/(sqrt 2 sd)), so P >= 0.9 needs
  sd <= ln(1/rho)/1.81, i.e. chi >= [1.03 (K-2)^0.61 / ln(1/rho)]^1.89 by the joint fit: for rho = 0.9,
  chi ~ 160 (K = 4) and ~570 (K = 8) [rough extrapolation of the empirical fit].
- **GF check** (sigma = 0.01, 400 seeds, `chi_gf.yaml`, `results/chi_gf.csv`): MPS4 chi = 32: P = 0.84 /
  0.68 / 0.58 at rho = 0.8 / 0.9 / 0.95 (race curve 0.86 / 0.70 / 0.60), per-seed race agreement 100%;
  binary8 chi = 32: 0.745 / 0.62 / 0.565 (race 0.72 / 0.61 / 0.55), agreement 99.5%; chi = 16 likewise.
- So the answer to "does larger chi restore greedy order?" is yes in the limit, as for over-parameterized
  CP (Ge et al. 2021), but slowly: the spread of log escape times shrinks only like chi^-1/2.

### 7.6 Step 4: three and four components (revised after review)

Scope of everything in this section: odeco teachers with weights (1, rho, rho^2[, rho^3]), rho = 0.8,
0.9, **d = chi = 4** (standard caps: every bond has dimension 4), fan-in init, 200 seeds, **one sigma per
tree**: 0.05 (binary4), 0.1 (MPS4), 0.2 (MPS6), 0.25 (binary8), chosen so that dormant components stay
above the float64 floor. The sigma-independence of the first event was checked only for sigma <= 0.1
(7.2), so MPS6 and binary8 are outside the checked range. GF: LSODA + exact Hessian (rtol 1e-9, atol
1e-14), exact event location for every first crossing a_r = 1/2, stopped at L < 1e-8 L(0), at t = 1e9
("unfinished"), or when some component's slice of some core falls to 1e-14 x the largest core norm
("stalled": its route is decaying exponentially; this is not a roundoff floor, since unlearned slices
keep falling to 1e-21-1e-35 and then underflow; the threshold is justified empirically: in the 22
no-stop runs no learned component's slice fell below 6.7e-6 before its event, while 25 of the 29
unlearned ones fell below 1e-14; verification). Per seed also the LO blow-up time tau_r of every
component. Model = "iid race" (Thurstone type): each component's blow-up time an independent draw from
the pooled tau's of the **same runs** that enter the GF statistic, order = ranking of tau_r / s_r.

Round 1 (raw `multi2`) used an overlap-based stall stop (|a_r| < 1e-25) and an overlap-based roundoff
flag (|a_r| < 1e-14), which excluded 18-19 binary8 runs per rho that did learn every component later,
and compared GF on completed runs with a model built from all runs (review item 3). R = 3 was rerun with
the slice criterion (`configs/multi3.yaml`, `analyze_multi.py multi3`, `results/multi3_summary.csv`,
`results/multi3_perms.csv`, `results/multi3_orders.png`). The rerun reproduces multi2 run for run
(same status and identical event times in all 1597 runs; the slice stop fires later, median x1.27 in
t): only the roundoff flag changed, and with it the 18-19 binary8 runs per rho are back (verification
E10). Three of the 1600 runs (MPS6 rho 0.8 seed 46,
rho 0.9 seed 94; binary8 rho 0.9 seed 130) were still integrating after 1.5 h and were stopped; they are
missing from the tables (n = 199).

**R = 3** (GF vs model on the same runs; z = (P_GF - P_model)/SE; first-event and full-order agreement
of GF with each seed's own LO race):

| tree (m), sigma | rho | used (excluded) | P(greedy order), GF vs model | z | Kendall tau with greedy, GF vs model | race = GF: first / full order |
|---|---|---|---|---|---|---|
| binary4 (2), sigma 0.05 | 0.8 | 200 of 200 (0 stalled, 0 unfinished) | 0.705 +- 0.032 vs 0.518 | +5.29 | 0.80 +- 0.02 vs 0.60 | 0.87 / 0.64 |
| binary4 (2), sigma 0.05 | 0.9 | 200 of 200 (0 stalled, 0 unfinished) | 0.490 +- 0.035 vs 0.319 | +5.20 | 0.58 +- 0.03 vs 0.32 | 0.81 / 0.53 |
| binary8 (4), sigma 0.25 | 0.8 | 198 of 200 (1 stalled, 1 unfinished) | 0.227 +- 0.030 vs 0.278 | -1.60 | 0.17 +- 0.04 vs 0.25 | 0.98 / 0.73 |
| binary8 (4), sigma 0.25 | 0.9 | 196 of 199 (1 stalled, 2 unfinished) | 0.194 +- 0.028 vs 0.214 | -0.70 | 0.05 +- 0.05 vs 0.12 | 0.97 / 0.70 |
| mps4 (4), sigma 0.1 | 0.8 | 195 of 200 (5 stalled, 0 unfinished) | 0.354 +- 0.034 vs 0.359 | -0.14 | 0.36 +- 0.04 vs 0.39 | 1.00 / 0.86 |
| mps4 (4), sigma 0.1 | 0.9 | 195 of 200 (5 stalled, 0 unfinished) | 0.226 +- 0.030 vs 0.251 | -0.81 | 0.14 +- 0.05 vs 0.19 | 1.00 / 0.89 |
| mps6 (6), sigma 0.2 | 0.8 | 91 of 199 (87 stalled, 21 unfinished) | 0.253 +- 0.046 vs 0.303 | -1.04 | 0.27 +- 0.06 vs 0.29 | 1.00 / 0.81 |
| mps6 (6), sigma 0.2 | 0.9 | 97 of 199 (87 stalled, 15 unfinished) | 0.216 +- 0.042 vs 0.228 | -0.26 | 0.17 +- 0.06 vs 0.15 | 1.00 / 0.77 |

- **binary4 (m = 2)**: far more greedy than the race model (z ~ +5), as the m = 2 picture predicts.
- **m >= 3**: GF is consistent with the iid race model in every case (|z| <= 1.6 for P(greedy order);
  Kendall tau with the greedy order within 1.7 SE) [empirical]. Paired against each run's own LO race
  on the same runs, GF is no less greedy: in 41 runs GF is greedy and the race is not, in 43 the reverse
  (per row 11/13, 10/8, 10/7, 4/4, 2/5, 4/6; McNemar p >= 0.45 in every row; verification). Each seed's
  own race predicts the first event in 97-100% of runs and the complete order in 70-89%; the
  misorderings are in the later events, where the learned components couple the remaining ones (on MPS4
  the second and third escapes come 10% to 2x later than their LO blow-up times). So: a random
  permutation favouring large components, consistent with independent clocks; the coupling after the
  first escape changes the full order in 11-30% of runs but not its greediness. **Withdrawn after
  verification (E1):** the fix round's "less greedy than the iid race in 6/6 m >= 3 cases (combined z =
  -1.9)". The 6/6 sign comes from the realized seed sets (their own race orders are already below the
  pooled model in 5/6 cases), and the combined z treated rho = 0.8 and 0.9 as independent although they
  share the 200 seeds per tree (GF-greedy indicators correlate 0.73-0.90); allowing for that gives
  z = -1.4.
- **Exclusions on MPS6** (half of the runs): the excluded runs have the smallest component first in their
  own race more often (24% vs 13% of used runs at rho = 0.8), but the race-first distributions differ
  only marginally (chi^2 p ~ 0.05 at rho = 0.8, 0.2 at 0.9) and the race-greedy fraction is the same
  (0.31 vs 0.29; 0.235 vs 0.237; verification). The unlearned component is the smallest one most often
  (rho = 0.8: largest 23, middle 37, smallest 48; rho = 0.9: 33 / 32 / 37). The MPS6 row is therefore a
  statistic conditional on completion. On MPS4 and binary8 at most 5 of 200 runs are excluded.

**R = 4 = chi (every bond must reach full rank)**. Round 1 said "GF mostly stalls before the last
component"; corrected:
- Not all four components are learned by t = 1e9 in most runs [empirical, scope above]: MPS4 289/400
  (sigma 0.1), MPS6 396/399 (0.2), binary8 389/393 (0.25); binary4 0/400 (a deep linear network).
  "Not learned by t = 1e9" is observed directly only for the runs that reached t_max and for the 22
  no-stop runs below; for the other multi2 runs (stopped by the round-1 overlap criterion) it is
  inferred from those 22 runs and from the reviewer's MPS4 no-stop check (verification E10).
- **Without any stall stop** (`configs/multi_nostop.yaml`, rho 0.8, 12 seeds per tree, to t = 1e9; one
  run per tree still integrating after 1.5 h was stopped, so 11 per tree,
  `results/multi_nostop_summary.csv`): binary8: 0 of 11 runs learn all four components; learned at t = 1e9 (count:runs) 3:11; unlearned component (rank:runs) 1:6 2:1 3:1 4:3; mps6: 0 of 11 runs learn all four components; learned at t = 1e9 (count:runs) 2:7 3:4; unlearned component (rank:runs) 1:2 2:3 3:6 4:7.
- How many components are learned at the stop (multi2 + no-stop; counts at the stop time, not at
  t = 1e9, for the multi2 runs): MPS4 3 of 4 in 288/289 incomplete runs; binary8 3 of 4 in all incomplete no-stop runs (the unlearned one is the largest in 6/11); MPS6
  only **2 of 4** in about half of the incomplete runs (multi2: 88 + 90 runs with 2, 111 + 107 with 3), so
  "before the last component" was wrong for MPS6.
- Round 1's "stalled" also included slow, pinned routes that never reach the floor: those are now
  "unfinished at t = 1e9" (e.g. 21 + 15 of the MPS6 R = 3 runs).
- This is outside Q4 and agrees with `notes/theory_regimes.md` R10 (K >= 5 with chi = R: very long last
  plateaus, fan-in init can freeze them; spare channels chi >= 2R cure it), except that **MPS4 (K = 4)
  also stalls in 72% of runs** at chi = R = 4, sigma = 0.1: R10's "K >= 5" is not a sharp bound. Only one
  sigma per tree, fan-in init and chi = R were run; gauge-balanced init and spare channels were not.

## 8. Status of H1 and H1b

**H1 ("the first escape direction is decided by the basin of the init, so the order is random; P(largest
first) increases with s_1/s_2, decreases with K, may concentrate as chi grows")**
- **Holds for tensor-like pairs (m >= 3)** [empirical over 5 trees, 28 (d, chi) settings, 67200 GF runs;
  derived at leading order; exact race on MPS]. P(larger first) = 0.60-0.88 at rho = 0.5 and 0.51-0.65
  at rho = 0.8 (sigma = 0.01, 400 seeds; race curves 0.66-0.89 and 0.55-0.66, see 7.1 on the common-seed
  offset); independent of sigma for sigma <= 0.1 (7.2); the per-init winner is the argmin of tau_r/s_r of
  the leading-order race (100% on MPS, 96-99% on binary8).
- **K and chi** [empirical]: sd(log tau) grows with K and falls with chi; with standard bond caps it
  saturates at fixed d; with uncapped (over-parameterized) bonds sd(log tau) ~ sqrt(K-2)/sqrt(chi) at
  large chi (7.5), so over-parameterized bonds restore greedy order, slowly (the analogue of
  over-parameterized CP, Ge et al. 2021).
- **Fails for matrix-like pairs (m = 2)**, which include binary N = 4 (a depth-3 DLN): only the largest
  component is a local maximum and the order is greedy as sigma -> 0 [derived; known for DLNs]. At finite
  sigma P(smaller first) is consistent with sigma^((1-rho)/rho) for non-adjacent pairs [heuristic
  derivation; GF on binary4, MPS6 ends, MPS4 cores 0,2 for rho = 0.7-0.9, sigma down to 1e-4..1e-5;
  fits over the three smallest sigma agree within 1.1 SE in all 13 sets]; adjacent pairs are 2-3.4x
  steeper, and the conjectured n = chi_e is supported at the small-sigma end on MPS4 (rho = 0.9: 0.40 +-
  0.05 vs 0.444) but not reached on MPS6 down to sigma = 0.003 (7.4b).
- **"init overlap enters with power K-2"** [empirical]: with the overlap defined as the projection of the
  init onto the component's KKT orbit, log tau_r ~ -q log X_r with q = (0.99-1.25)(K-2), above K-2 in
  most settings (bootstrap SE 1-4% of q); "argmax s_r X_r^(K-2)" predicts 76-92% of winners, the same out
  of sample (2-fold over seeds) and with q fitted; the rest needs the full alignment dynamics, which the
  LO race captures exactly.

**H1b ("a wrong order is corrected through a strict saddle with lambda = q (s_learned - s_new), in time
~ log(1/sigma), before the second rank is added")** — round 1 said "holds for m = 2"; narrowed:
- **Holds for m = 2 pairs with k >= 3 (non-adjacent cores)** [derived; checked on binary4 (K = 3) and
  MPS4 cores 0,2 (K = 4)]: from the wrong saddle the gauge-projected Hessian has lambda_min =
  -(s_1-s_2) s_2^((K-2)/K) (4-5 digits), sigma-independent (strict by the primary rule); the learned bond
  index rotates to the larger component after t = ln(1/sigma)/|lambda_min| + const (local slope x
  |lambda| -> 1.0000 at small sigma), the smaller component is forgotten (a_2 < 1e-3) and relearned in a
  new bond index after ~sigma^-(k-2).
- **Fails for adjacent m = 2 pairs (k = 2)** [derived; checked on MPS4 cores 1,2]: the wrong saddle is
  strict, but its most negative direction (-s_1 s_2^((K-2)/K), multiplicity chi_e - 1) adds the larger
  component on a spare bond index; t1 = ln(1/sigma)/(s_1 s_2^((K-2)/K)) + const; nothing rotates or is
  forgotten (case 2 of `notes/theory_saddles.md` 4.4).
- **In natural runs the "log(1/sigma)" part does not appear** [derived + empirical]: the wrong order is
  rare as sigma -> 0 and, when it happens, the escape route is O(1) on arrival, so the correction takes
  O(1/|lambda|) (k >= 3: |lambda| (t1 - t2) = 1.0-1.4 flat in sigma 1e-4..1e-2 on binary4 and MPS4;
  k = 2: 0.2-0.3) and no wrong plateau is resolved.
- **Fails for m >= 3** [derived; checked on three trees: MPS4 (k = 4: 8 seeds x 5 sigma over one decade,
  plus 40 natural runs), MPS6 (k = 6: 5 seeds x 5 sigma 0.05-0.2) and binary8 (k = 7: 6 seeds x 5 sigma
  0.15-0.35)]: the wrong-order state is a higher-order
  saddle (|lambda_min| ~ sigma^2.00 on MPS4, ~ sigma^4.2 on MPS6), no rotation happens, the larger
  component is added in a new bond index after ~sigma^-(k-2); the wrong order stays. Sometimes the larger
  component's route dies and GF stalls at the wrong saddle (1/40 natural MPS4 runs; 1/6 MPS6 seeds).
  binary8 (k = 7) behaves the same over a narrow sigma range (|lambda_min| ~ sigma^5.2 +- 0.3, no
  rotation). *Cross-reference (2026-10-10):* by the saddle-order theorem (`notes/theory_saddle_order.md`
  v2, Theorem 2, section 7 item 5, claim S12) the binary8 odeco wrong saddle, whose differing sites cover
  two leaf pairs, has order 6, not k = 7: below a crossover (aligned amplitude b ~ 0.155) the escape is
  predicted ~sigma^-4, along non-residual patterns, followed by forgetting of the smaller component
  (derived; untested). The binary8 runs here (sigma 0.15-0.35) sit at or above that crossover and follow
  the route law, as expected; so "the wrong order stays" is shown for binary8 only above the crossover.
  MPS4 and MPS6 are not affected.

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
wrong-order law for non-adjacent matrix-like pairs, consistent with sigma^((1-rho)/rho) in full GF (the
law itself is a heuristic of `notes/theory_escape.md` 5.3 and of section 5 here); a measured selection
rule with exponent ~K-2 for the KKT-projection overlap; the after-a-wrong-order dynamics measured for the
three branches (non-adjacent m = 2: eigenvalue, log-law rotation from the saddle, forget-and-relearn,
O(1/lambda) correction in natural runs; adjacent m = 2: adding through the k = 2 route; m >= 3: no
correction), which are the cases of `notes/theory_saddles.md` 4.4.

## 10. Open issues

- Prefactor C(rho) of the m = 2 law and the pre-asymptotic regime (rho = 0.95 never reaches small P).
- Adjacent m = 2 pairs: n = chi_e (conjecture) is supported at the small-sigma end on MPS4 at rho = 0.9
  (0.40 +- 0.05 over sigma = 1e-3..1e-5 vs 0.444) but not reached at rho = 0.8 or on MPS6 (down to
  sigma = 0.003); smaller sigma on MPS6 or more seeds at rho = 0.8 would decide it.
- Why the KKT-projection rule misses 8-24% of winners and why q > K-2: the alignment phase mixes the
  non-aligned part of the init into the escape direction; a closed form for the distribution of tau on a
  chain is open.
- R >= 3: whether stalls at chi = R disappear with gauge-balanced init or spare channels (theory_regimes
  R10, P14); the R = 3 statistics on MPS6 are conditional on completion (half of the runs). (The issue
  "why GF is slightly less greedy than independent clocks" is dropped: the paired test detects no such
  effect, E1.)
- binary8 below the saddle-order crossover (sigma well below 0.15): order-6 escape ~sigma^-4 with
  forgetting of the smaller component, predicted by `notes/theory_saddle_order.md` v2, not run.
- binary8 / HT trees with shared internal cores: the independent race is 96-99% exact; a coupled
  closed-form race is open.

## 11. Claims changed in the fix round (review `notes/reviews/order_round1.md`)

1. **H1b (O7)**: "holds for m = 2" -> holds for m = 2 pairs with k >= 3 (non-adjacent); adjacent pairs
   (k = 2) add the larger component on a spare bond index through a strict saddle with lambda = -s_1
   s_2^((K-2)/K) (multiplicity chi_e - 1) and forget nothing (new runs: MPS4 cores 1,2 and 0,2, saddle and
   natural). Section 6 rewritten.
2. **m = 2 law (O6)**: n carried everywhere; n = chi_e tagged conjecture; "still steepening" withdrawn;
   one fit range (full measured range) for all m = 2 sets with local slopes; "confirmed" -> "consistent
   with"; wrong-order seeds shown to be exactly nested.
3. **R = 3 (O9)**: rerun with a slice-norm stall criterion (multi3); GF and model compared on the same
   runs; excluded runs' race orders and unlearned components reported; sigma, d = chi = 4 stated; GF is
   below the model in 6/6 m >= 3 cases (combined z = -1.9), "close to an independent race" qualified.
   [Superseded: the "less greedy" reading is withdrawn after verification, section 12 E1.]
4. **R = 4 (O9)**: "mostly stalls before the last component" -> most runs do not learn all four by t =
   1e9 (no-stop check on MPS6 and binary8 added); MPS6 often learns only 2 of 4; scope (one sigma per
   tree, d = chi = 4, fan-in) stated; cross-reference to theory_regimes R10 (MPS4, K = 4, also stalls).
5. **m >= 3, no strict saddle**: scope stated; two trees added (MPS6 saddle starts, |lambda_min| ~
   sigma^4.2; binary8, ~sigma^5.2 over sigma 0.15-0.35).
6. **d (O4)**: the invalid chi^2 d-pairs test withdrawn; independent-seed test added (no effect at the
   0.04 level on binary8 and MPS4). [Corrected after verification, E3: differences larger than about
   0.06 are excluded at 95%.]
7. **Unresolved midpoint Hessians**: the 43 natural binary4 runs without one are now resolved (fine record
   window around the events); full-precision event times (t ~ 1e8 with gaps ~ 10 had lost digits in the
   CSV, affecting only the round-1 MPS4 adjacent/separated natural runs, which were rerun).
8. **Smaller corrections**: H1b slope precision (local slope x |lambda| -> 1.0000; the 1.5% was the
   sigma = 1e-2 point); "lambda_min between 0 and the saddle value" restricted to sigma <= 1e-2 (refined
   after verification, E4: at or above the saddle value to within 1.3%); race-curve
   ranges and the common-seed offset (-0.018) reported; O8 wording (sqrt(K-2) at large chi, the joint
   power law does not separate); O5 with SE, out-of-sample accuracy and accuracy at q = K-2.
9. **Code**: --raw/--res reach spawned workers; default worker count = container CPUs; run_queue.sh
   passes --raw/--res; README fixed (multi2/multi3, integrators, procs); stall criterion uses slice norms
   (the overlap criterion also fired during a normal forget phase); tol_check now includes MPS8 (400 -> 500
   inits, all winners equal at rtol 1e-11, |dt1|/t1 <= 1.0e-9); the symmetry check is complete for MPS8
   (mean log(tau_1/tau_2) = -0.02 +- 0.04).

## 12. Errata after the verification of the fix round (2026-10-10)

Source: `notes/reviews/order_round2_verification.md` (all 7 MUST and 5 SHOULD items fixed; every
recomputed number reproduces). Applied in place in the notes-cleanup pass; its "Safe to cite" table
(C1-C17) is the final list of citable claims, and its "Not safe to cite" list stands.

- **E1 (withdrawn claim; O9, 7.6, 10).** "GF is less greedy than independent clocks in 6/6 m >= 3 cases
  (combined z = -1.9)" is withdrawn with its open issue. The R = 3 data are consistent with the iid race
  model (|z| <= 1.6 per case); paired against each run's own LO race, 41 runs are greedy for GF only and
  43 for the race only (McNemar p >= 0.45 in every row); the combined z corrected for the seeds shared
  by rho = 0.8 and 0.9 is -1.4.
- **E2 (O6, 7.4, 7.4b, 8, 10).** n = chi_e stays a conjecture, but "not reached by sigma = 1e-5, no
  steady approach" was too pessimistic: on MPS4 at rho = 0.9 the small-sigma fit (sigma = 1e-3..1e-5)
  is 0.40 +- 0.05 against 0.444; MPS6 (to sigma = 0.003) and both rho = 0.8 cases stay below.
- **E3 (7.1b, 11 item 6).** d test: differences larger than about 0.06 are excluded at 95% (not "no effect
  at the 0.04 level"); two-sample SE 0.018-0.022.
- **E4 (7.3).** Natural runs: lambda_min at the minimum-speed point is at or above the saddle value to
  within 1.3% (26/296 binary4 and 12/237 MPS4 0,2 runs sit 0-1.3% below it).
- **E5 (7.2, O6, 8).** Goodness of fit of the m = 2 law: fits over the three smallest sigma agree with
  (1-rho)/rho within 1.1 SE in all 13 non-adjacent sets; the full-range fits are low in 12/13
  (pre-asymptotic). Replaces the deviance p-values.
- **E6 (7.3).** The adjacent (k = 2) natural set has 57/150 runs without a resolved minimum-speed point
  (near-ties, median gap 0.06; not used).
- **E7 (O5, 8).** Bootstrap SE of q: 1-4%, not 1-9%.
- **E8 (REPORT log).** The 21:25 ("+-1.5%") and 01:13 ("steepening toward n = chi_e = 4") entries are
  marked superseded in `agent-runs/order/REPORT.md`.
- **E9 (7.3).** The power law fits 11-88x worse than the log law for the rotation sets (104-110x for the
  adjacent set), not 10-80x.
- **E10 (7.6).** "Not learned by t = 1e9" is observed directly only in the t_max and no-stop runs and
  inferred for the other multi2 runs; the "learned at the stop" counts are at the stop time; multi3
  reproduces multi2 run for run (only the roundoff flag changed).
- **Wording (7.6, no erratum number).** The slice stall rule is justified empirically (no false
  positive in the 22 no-stop runs, margin of 8 orders of magnitude), not by a float64 roundoff floor.
- **Cross-reference (6, 8, 10; not from the verification).** The saddle-order theorem
  (`notes/theory_saddle_order.md` v2) gives order 6 < k = 7 for the binary8 odeco wrong saddle; the
  binary8 runs here lie at or above the predicted crossover, so the m >= 3 "wrong order kept" result
  holds for binary8 only there.
