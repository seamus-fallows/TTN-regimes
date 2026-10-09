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

### 7.2-7.5 (sigma scans, controls, chi scan, H1b, R >= 3): pending

## 8. Status of H1 and H1b (to be finalized)

## 9. What is new relative to the literature (to be finalized)

## 10. Open issues (to be finalized)
