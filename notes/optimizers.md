# Optimizer dependence of the saddle-to-saddle picture (backlog B1, agent `optimizers`)

Status: **v0, 2026-10-09**, in progress. Experiments: `experiments/optimizers/`. Tags: **proven**,
**derived** (assumptions stated), **heuristic**, **empirical**, **conjecture**.

Question: which features of the gradient-flow (GF) saddle-to-saddle (S2S) picture — plateaus,
incremental bond-rank growth, learning order, plateau scaling sigma^-(K-2) — survive under the
optimizers used in practice: Adam / RMSProp (default in TN-ML code), heavy-ball momentum (control),
GD with re-gauging to a canonical form, Riemannian GD on the fixed-bond-dimension manifold, per-core
Gram-preconditioned GD (block-diagonal Gauss-Newton, the ALS normal equations used simultaneously),
and the non-gradient sweep methods ALS (one-site) and DMRG (two-site + SVD truncation).

Model and loss: `notes/model.md` (whitened inputs, exact population loss, fan-in init with per-core
scale sigma). Events, widths, separation ratios rho: `notes/diagnostics.md` (via `ttn.diagnostics`).

## 1. Theory

Notation: L = 1/2 ||W(theta) - W*||^2, g(theta) = <W(theta), W*> (degree-K homogeneous), so near the
origin L = L_0 - g + O(|theta|^(2K)) and grad L ~ -grad g is homogeneous of degree K - 1.
The init is theta_0 = sigma phi with phi a fixed (seeded) direction.

### 1.1 What each method is invariant under (proven, by inspection of the update rules)

| method | equivariant under | conserves Q_e (gauge charge) | depends on init scale through |
|---|---|---|---|
| GF | all orthogonal maps of theta (incl. O(chi) gauge, site rotations) | yes, exactly | sigma^-(K-2) escape time |
| GD, small lr | same | up to O(lr^2) per step | same as GF at t = lr n |
| heavy ball (momentum beta) | same | no (velocity term), -> GF as lr -> 0 at t = lr n / (1 - beta) | same as GF |
| Adam, RMSProp | coordinate permutations and sign flips only (diagonal preconditioner) | no | only through sigma / lr and eps (1.2) |
| GD + re-gauging (root-centred canonical form) | GL gauge at re-gauge times | no (Q reset each period) | none at leading order (1.4) |
| Riemannian GD (HSVD retraction) | the full GL gauge (acts on W only), site rotations | n/a (works on W) | none as sigma -> 0 (1.5) |
| ALS / DMRG | GL gauge and per-core rescaling | n/a | none at all (1.6) |

Adam's lack of rotation equivariance means its trajectory depends on the basis in which the teacher's
factors are written relative to the coordinate axes of the cores (here: Haar-random site bases) and on
the arbitrary bond basis of the init; GF does not care.

### 1.2 Adam and RMSProp near the origin: the init scale is forgotten (derived)

Adam's step is -lr * u_n with u_n = mhat_n / (sqrt(vhat_n) + eps), mhat, vhat the bias-corrected
exponential averages of g and g^2 (per coordinate).

**(a) Homogeneity.** With eps = 0, u_n is invariant under a common positive rescaling of all past
gradients. Near the origin grad L(c theta) = c^(K-1) grad L(theta) (to leading order), so if
(theta_n) is the Adam trajectory from theta_0 with learning rate lr, then (c theta_n) is the Adam
trajectory from c theta_0 with learning rate c lr. Near the origin the dynamics depend on the init
scale only through **sigma / lr** (time in steps).

**(b) sigma / lr -> 0.** Adam's first (bias-corrected) step is exactly -lr sign(grad L(theta_0)),
which depends on phi only through the sign pattern of grad g(phi). After it |theta| ~ lr >> sigma, and
the gradients there exceed the first one by ~(lr/sigma)^(K-1), so the history weight of the init is
negligible. The trajectory converges to a sigma-independent limit: **the number of steps to the first
escape stays bounded as sigma -> 0** (GF: sigma^-(K-2)).

**(c) lr -> 0 at fixed sigma (ODE limit).** With slowly varying gradients mhat/sqrt(vhat) -> sign(g)
per coordinate, so Adam follows the sign flow d theta/d tau = -sign(grad L), tau = lr n (an
l_inf steepest descent; RMSProp: the same). The sign flow is 0-homogeneous in the gradient: near the
origin theta(tau) = sigma psi(tau / sigma) solves it whenever psi does, so leaving the sigma-ball
costs tau = O(sigma), and growing every learned core entry to its final size c_* costs tau ~ c_*.
Hence **n_1 lr -> A (a constant ~ the learned entry size, O(0.1-1)) as sigma -> 0**: the first-escape
time in steps is ~A / lr, independent of sigma. When gradients grow quickly (an escape),
|mhat / sqrt(vhat)| can exceed 1, up to (1 - beta1) / sqrt(1 - beta2) = 3.2 for default betas, so
A is only defined up to this factor.

**(d) eps restores GF below a crossover scale.** If |g_i| << eps, u ~ mhat / eps: Adam is GD
(EMA momentum) with learning rate lr / eps, i.e. GF with t = lr n / eps. The gradient at the init
has rms c_g sigma^(K-1) (measured: c_g = 0.22 binary4, 0.011 binary8, 0.031 mps8, d = 3, chi = 4,
odeco (3, 2, 1)), so for

  sigma < sigma_eps = (eps / c_g)^(1/(K-1))   (eps = 1e-8: 2e-4 binary4, 0.10 binary8, 0.12 mps8)

Adam first crawls along the GF path until the gradient reaches ~eps:

  n_1 lr ~ eps T_GF(sigma -> sigma_eps) + A,   i.e. n_1 lr ~ eps C sigma^-(K-2) for sigma << sigma_eps.

**The sigma^-(K-2) plateau law comes back under Adam, with prefactor eps / lr**, for inits below
sigma_eps. With the default eps = 1e-8 this crossover (~0.1 per core) sits inside the range of
small inits used in practice for K >= 7.

**(e) Order and incremental learning (heuristic).** Each coordinate moves at the same speed lr,
with the sign of the *total* gradient, which is dominated by the component with the largest
gradient contribution (s_r a_r^(K-1) for an odeco teacher). So components are still learned one
at a time, selected by the same quantity as in GF, but once a component dominates it is learned in
~A / lr steps whatever its gradient size: the waiting time is set by the time needed to finish the
previous components, not by sigma. Separation ratios (gap / width) are then O(1) and
sigma-independent: S2S by the rho > 3 rule is neither guaranteed nor excluded, and the scaling
criterion (iii) (plateaus grow as sigma -> 0) fails in the sign regime.

**(e') Later plateaus: Adam's memory, then the adaptive edge of stability (heuristic, from the
probes of section 2.4).** After a component is learned its gradient vanishes, so the second
moments decay: P = sqrt(vhat) + eps shrinks by beta2^(1/2) per step, from the level P_0 set by the
gradients of the escape just finished. Coordinates are shared between components (Haar-random
teacher bases), so the dormant components see GD with a growing learning rate lr / P(n). The decay
stops when the preconditioned sharpness lambda_max(P^-1 H) reaches the adaptive-edge-of-stability
threshold 2 (1 + beta1) / ((1 - beta1) lr) (= 38 / lr for beta1 = 0.9; Cohen et al. 2022, cited
from memory): oscillations along the stiff learned directions then keep v from decaying further.
From then on Adam is GD with an **lr-independent** step of ~2 (1 + beta1) / ((1 - beta1) lambda_max)
GF-time units per step (lambda_max ~ K s^(2 - 2/K) of the learned components). A plateau that
needs GF time T (from the dormant amplitudes Adam has at its start) therefore lasts

  n(T) ~ (2 / (1 - beta2)) ln(1 + T P_0 (1 - beta2) / (2 lr))           while P is still decaying,
  n(T) ~ n_AEoS + (T - T_AEoS) (1 - beta1) lambda_max / (2 (1 + beta1))  once P is held at the AEoS.

Short GF plateaus (binary N = 4) end in the first, logarithmic phase: weak dependence on sigma and
on lr, a strong one on beta2. Long ones (deep trees) end in the second, linear phase: the GF time
scale is back, converted at a fixed rate of ~lambda_max / 38 steps per GF-time unit. Whether the
sigma^-(k-2) law also comes back depends on whether the dormant amplitudes at the start of the
plateau still remember sigma: Adam's sign phase during the first escape moves every coordinate by
~lr per step, so they need not.

**(f) Balance (heuristic).** Under the sign flow every active core entry grows at the same rate, so
at the end of an escape entries have similar magnitudes in every core, and ||G_v||^2 is roughly
proportional to the number n_v of entries of core v. GF keeps Q_e = O(sigma^2) (balanced:
||G_v|| = ||G_p|| along a rank-one escape). Heuristic Adam prediction for the balance index
b_e = ||Q_e|| / (||G_v^T G_v|| + ||G_p,(v) G_p,(v)^T||): b_e ~ |n_v - n_p| / (n_v + n_p), e.g.
binary4 (d = 3, chi = 4): 36 vs 16 entries, b ~ 0.38.

### 1.3 Heavy-ball momentum (control; known)

theta_{n+1} = theta_n - lr m_n, m_n = beta m_{n-1} + grad L: as lr -> 0 at fixed beta this is GF
with t = lr n / (1 - beta) (the velocity relaxes in ~1/(1 - beta) steps, much faster than any
plateau). Prediction: same events as GF at t = lr n / (1 - beta), Q_e drift -> 0 with lr.

### 1.4 GD with re-gauging to the root-centred canonical form (derived)

Re-gauging (QR sweeps leaves -> root) leaves W unchanged, makes every non-root core an isometry
(||G_v||_F^2 = chi_v) and moves all of ||W|| ~ sigma^K into the root. In this gauge the root's
environment is an isometry E, so the root's Hessian block is the identity and its flow is linear:

  dR/dt = E^T W* - R   =>   R(t) = (1 - e^-t) E^T W* + e^-t R(0)   (frames fixed).

The root reaches the projection of W* onto the current frames in O(1) time, **independently of
sigma**: the first plateau is gone. The frames' gradients are proportional to R, so after R has
grown they rotate on an O(1 / ||R||^2) time scale; periodic re-gauging keeps them isometric.
Re-gauging at t = 0 alone is GF from a maximally imbalanced init (all scale in the root), the
"small root" case of `notes/theory_regimes.md` R11, which is lazy-like for the root. Prediction:
first event at t = O(1) (lr n), all components rising together, order set by the projections of the
components onto the random initial frames (not by weight), balance index near 1 on every edge.

**Correction for deep trees (derived after the binary8 / mps8 data, section 2.5).** The O(1) statement
assumes that the projection E^T W* of the teacher onto the random initial frames is O(1). A random
chi-dimensional frame of the d^(n_e)-dimensional space of an edge's sites keeps a fraction ~chi / d^(n_e)
of a fixed vector's squared norm, so a rank-one component keeps ~prod over the root's edges of
chi / d^(n_e): (4/9)^2 = 0.2 on binary4 (d = 3, two sites per side) but (4/81)^2 = 2.4e-3 on binary8 (four
sites per side). The root then first fits a tiny projection and the frames must rotate towards the
teacher; their gradients are proportional to the (small) root, and rotating a deep subtree's frame needs
coordinated changes of several cores. Learning becomes sequential again, with waiting times set by the
frame overlaps (chi / d^(n_e)), **not by sigma**: up to O(sigma^K) corrections (the root's initial value)
the trajectory is the same for every sigma, because the canonical form at t = 0 removes sigma from every
core but the root.

### 1.5 Riemannian GD on the fixed-bond-dimension manifold (derived)

Step: W_{n+1} = HSVD_chi(W_n - lr P_{T(W_n)} (W_n - W*)), with P_T the orthogonal projector onto
the tangent space (range of dW/dtheta; implemented with the dense Jacobian and a least-squares
solve) and HSVD the leaves-to-root hierarchical SVD (a retraction). The fixed-rank manifold is a
cone, so T(c W) = T(W): the tangent space depends only on W's frames. Since W in T(W),

  W_1 = HSVD((1 - lr) W_0 + lr P_T(W_0) W*) -> HSVD(lr P_T(W_0) W*)   as sigma^K -> 0,

a sigma-independent point. The continuous-time version (Riemannian GF, dW/dt = P_T (W* - W), which
equals the Gauss-Newton / natural-gradient flow of the parameters, d theta/dt = J^+ (W* - W)) moves
W at speed ||P_T W*|| = O(1) from the start. **No plateau at small init; escape time O(1) and
sigma-independent.** The bond ranks are fixed at chi throughout (the manifold has full rank), so
incremental rank growth can at most show up as staged growth of the singular values.

### 1.6 ALS and DMRG sweeps: exactly scale-free (proven)

One-site ALS with the orthogonality centre moved by QR: in canonical form centred at v the
environment E_v is an isometry, so the least-squares core is G_v = E_v^T W* (the gradient of
<W, W*> in G_v; implemented so). The QR factor Q of sigma Z equals that of Z, and the centre core is
overwritten by a function of the frames only. Hence **every ALS iterate W_n is independent of
sigma** (exactly, in exact arithmetic), for any sigma > 0 and any per-core scales. The same holds
for two-site DMRG (merged least-squares tensor E_{vu}^T W*, then SVD truncation). Consequences:
no sigma dependence of anything (plateaus cannot scale with sigma); after the first micro-step
W = E E^T W* is a projection of the teacher, whose bond ranks are min(chi, rank of the projected
teacher) at once: no incremental rank growth in the GF sense. For binary N = 4 (a 3-factor matrix
product A B C^T) one-site ALS is exact after one sweep whenever chi >= rank W* and the projected
teacher keeps full rank (column space of W* C fixed by the first A-update).

### 1.7 Per-core Gram preconditioning, all cores at once (derived, aligned scalar model)

Preconditioning each core's gradient by the inverse of its exact Hessian block (Kronecker product
of environment Grams, as in the ALS normal equations) and updating **all cores simultaneously** with
step lr moves each core a fraction lr toward its own least-squares solution, which near the origin
is ~sigma^-(K-1) times larger than the core. In the aligned scalar model (b_v = b for all cores,
W = b^K), b' = b + lr (s - b^K) / b^(K-1): from b = sigma the first step overshoots to
W ~ (lr s)^K sigma^(-K(K-1)), after which W decays by (1 - lr)^K per step. Steps to come back:
n ~ (K - 1) ln(1/sigma) / lr. **Plateau replaced by an overshoot of the loss followed by a
geometric return lasting ~ln(1/sigma)/lr steps**; events (a_r = 1/2) are crossed at step 1 by the
overshoot and are not meaningful. Damping (here: each Kronecker factor + damping x mean eigenvalue)
caps the overshoot. Near the global minimum (b^K = s) the update is stable for lr < 2/K.

## 2. Results

All runs: exact population loss, float64; GF = `ttn.dynamics.integrate_gf` (LSODA, rtol 1e-9, atol
1e-14, stop at L < 1e-6 L0); event = component overlap a_r reaching 1/2; widths, separation ratios and
the S2S rule (all rho > 3) as in `notes/diagnostics.md`. Exponent p: least-squares fit of
log T = c + p log(1/sigma) over all converged seeds (`ttn.diagnostics.plateau_scaling`), +- its standard
error. Tables: `experiments/optimizers/results/`; figures: `.../results/figures/`.

### 2.1 Binary N = 4 (K = 3), odeco (3, 2, 1), every optimizer (`scaling_b4`, 6 seeds x 7 sigma in [0.002, 0.128])

| method | p, first escape | p, gap 1 | p, gap 2 | weight order | S2S | balance index (end) |
|---|---|---|---|---|---|---|
| GF (prediction p = K - 2 = 1) | 1.007 +- 0.026 | 1.008 +- 0.046 | 1.011 +- 0.038 | 100% | 95% | 0.000 |
| GD lr 0.02 | 1.005 +- 0.026 | 1.008 +- 0.046 | 1.011 +- 0.038 | 100% | 95% | 0.000 |
| heavy ball lr 0.002, beta 0.9 | 0.970 +- 0.025 | 1.006 +- 0.047 | 1.010 +- 0.038 | 100% | 95% | 0.003 |
| Adam lr 1e-3 | 0.012 +- 0.006 | 0.50 +- 0.13 | 0.03 +- 0.11 | 98% | 19% | 0.38 |
| Adam lr 1e-3, beta2 0.99 | 0.011 +- 0.007 | 0.36 +- 0.10 | 0.03 +- 0.05 | 98% | 7% | 0.34 |
| Adam lr 1e-3, beta2 0.9 | 0.013 +- 0.007 | 0.21 +- 0.07 | 0.04 +- 0.02 | 93% | 0% | 0.33 |
| Adam lr 1e-2 | 0.009 +- 0.008 | 0.23 +- 0.18 | 0.12 +- 0.07 | 86% | 12% | 0.39 |
| RMSProp lr 1e-3 | -0.003 +- 0.012 | 0.14 +- 0.07 | 0.04 +- 0.03 | 98% | 0% | 0.33 |
| GD + re-gauge every step / every 50 steps, lr 0.02 | 0.000 +- 0.027 | 0.00 +- 0.09 | 0.00 +- 0.09 | 67% | 0% | (gauge-fixed) |
| Riemannian GD lr 0.1 / 1 | 0.000 +- 0.02 | 0.00 +- 0.09 | 0.00 +- 0.15 | 0% | 0% | (gauge-fixed) |
| ALS / DMRG | 0 (exact) | 0 | 0 | 0% | 0% | (gauge-fixed) |

Findings (all **empirical** unless tagged otherwise):

1. GD and heavy ball are GF at t = lr n / (1 - beta): same events, same exponent; Q_e drift 1e-5 (GD),
   3e-4 (heavy ball) relative, vs 1e-11 for the integrator (figure `balance.png`).
2. **Adam / RMSProp: the first escape no longer depends on sigma** (p = 0.01 +- 0.01; ~185 steps at
   lr 1e-3, ~30 at lr 1e-2, for sigma from 0.002 to 0.128), as derived in 1.2 (b, c). The learning is still
   sequential and in weight order (86-98% of runs), but the first escape is a slow polynomial rise
   (width ~ its own time, sign-flow growth), so separation ratios are small (median rho_1 ~ 1-3) and the
   S2S rule holds in only 0-19% of runs. Later gaps grow weakly as sigma decreases (p = 0.14-0.50) and
   shrink with the memory length of beta2 (p = 0.50, 0.36, 0.21 for beta2 = 0.999, 0.99, 0.9): the
   memory phase of 1.2 (e').
2b. **The sign-flow limit (derived 1.2 c) is reached as lr -> 0.** First escape in sign-flow time
   tau_1 = lr n_1 (median over 42 runs, all sigma): 0.296 (lr 1e-2), 0.180 (1e-3), 0.162 (3e-4),
   0.168 (1e-4), seed-to-seed s.d. 0.006-0.02; exponent in sigma 0.012 +- 0.003 at lr 1e-4. RMSProp:
   0.086 (1e-3), 0.225 (1e-4). Later gaps also scale as 1/lr at small lr (gap 1 in tau: 0.44, 0.38,
   0.37 for lr 1e-4, 3e-4, 1e-3), i.e. they are sign-flow durations, with a weak sigma dependence
   (p = 0.30 +- 0.06 at lr 1e-4).
3. **Re-gauged GD, Riemannian GD, ALS, DMRG: no plateau and no sigma dependence at all** (event times
   identical across sigma to the printed digits for re-gauged GD, proven for ALS / DMRG). All components
   rise together; the order is set by the projections of the teacher on the random initial frames, never
   (Riemannian, ALS, DMRG) or rarely (re-gauged GD, 67%) the weight order. On binary N = 4 ALS is exact
   after one sweep (1.6).
4. Adam ends unbalanced: balance index 0.38 on the root edges (entry-count heuristic 0.385, 1.2 f);
   GF keeps O(sigma^2) (0.000).

### 2.2 Adam internals (`adam_probe`, figure `adam_probe.png`)

Median preconditioner P = sqrt(vhat) and lambda_max(P^-1 H) / (38 / lr) along four runs. Binary4,
beta2 = 0.999: after the first component, P follows the pure-memory decay beta2^(n/2) and the
preconditioned sharpness stays below 0.1 of the threshold until the second component escapes (memory
phase). Binary8 and MPS8 E8 (long plateaus): P decays, then lambda_max(P^-1 H) reaches the adaptive-
edge-of-stability threshold (ratio 1.0 +- 0.1) and P stops decaying at ~2-4e-4; the effective step
lr / P is then ~2-5 GF-time units per step, close to 38 / lambda_max (lambda_max = 26 on binary8:
1.5). Binary4, beta2 = 0.9: memory too short to matter, Adam sits at the AEoS during the plateaus.
**Empirical**, supporting the heuristic 1.2 (e').

### 2.3 Gram-preconditioned GD (`precond_b4`, 6 seeds x 7 sigma)

- Scale-free damping (1e-3 x mean eigenvalue per Kronecker factor), lr 0.01 / 0.1 / 0.5: the first step
  overshoots the loss to up to 1e22-1e34 L0 (growing as sigma^-6 = sigma^-K(K-1), as derived in 1.7),
  then the loss returns; converged runs need ~60-400 steps for lr 0.5-0.1 independently of sigma, but
  4-9 of 42 runs per lr did not converge within 5e4 steps. Order: mixed (33% weight order).
- Absolute damping floor 1e-4 / 1e-2 per Kronecker factor, lr 0.1: no overshoot (max loss ~L0 for the
  larger floor), convergence in 14-36 steps for every sigma (p = 0.04 +- 0.03 and 0.11 +- 0.03 for the
  time to L < 1e-3 L0). **My prediction 1.7 that an absolute floor restores the GF plateau (as eps does for
  Adam) fails in this sigma range**: the floor enters once per bond leg of a core, so near the origin the
  method is GD with step lr / floor^(#bond legs) = 10-1e5 GF-time units per step, and the GF escape time
  (~10-1000 units at these sigma) is crossed in a few steps. The plateau would reappear only once
  T_GF(sigma) floor^(#legs) / lr >> 1, far below sigma = 0.002.

### 2.4 Consistency with `notes/theory_optimizer.md` v2 (P1, P2, P6)

`experiments/optimizers/consistency.py` -> `results/theory_optimizer_P1P2.csv`, `..._P6.csv`.

- **P1 / A1 (sharpness at the GF minimum = K s_1^(2-2/K)): confirmed.** Hessian lambda_max at the end of
  converged runs / pattern-rule value (12.98 binary4, 46.03 binary8, 54.71 mps8; chi = 4 here, the extra
  bond label has gain 0): GF, GD and heavy ball 1.000-1.002 from balanced inits (binary4 fan-in too, where
  the imbalance is O(sigma^2)); 1.04 (1.00-1.09) for fan-in binary8 GF, whose minimum keeps the fan-in
  imbalance (balance index 0.19), consistent with P1's conserved-quantity correction.
- **P2 / A2 (every other minimum is at least as sharp): confirmed, and Adam lands on those.** Adam / RMSProp
  end at imbalanced minima (balance index 0.3-0.8) with lambda_max / pattern rule = 1.06-1.37 (lr 1e-3) and
  1.3-6.1 (lr 1e-2, means 1.3-2.5 by tree). The larger Adam's step, the sharper the minimum, the opposite of
  GD's edge-of-stability selection (P3): Adam's stability constrains lambda_max(P^-1 H) < 38 / lr, not
  lambda_max(H) < 2 / lr.
- **P6 (GD event shift): consistent in magnitude only.** binary4, GD lr 0.02 vs GF, same seeds: measured
  shifts Delta t / lr = 2.7-5.4 against (1/2) ln(|grad L(t_r)| / |grad L(0)|) = 2.2-4.2 for sigma >= 0.016
  (measured 0.4-1.3 lr larger); at smaller sigma the shift (~0.1 time units) is below the resolution of my
  GF output grid. Not a precise test (that is `optim-samples`' job); nothing contradicts P6.
- P5, P7, P9 concern GD above eta_c and SGD, which this experiment does not run.

### 2.5 Deep trees: binary N = 8 (K = 7) and MPS N = 8 (K = 8) (`deep`, 864 runs; `later_b8`, 168 runs)

Problems: odeco (3, 2, 1) with d = 3, chi = 4, and the locality teacher E8 (d = 2, chi = 3); sigma = 0.3
(S2S) and 0.8 (moderate); balanced init (clean GF baseline) and fan-in init; 6 seeds each. Table:
`results/deep_table.csv`; per-seed order agreement with GF: `results/order_agreement_with_gf.csv`;
figures `deep_side_<tree>_<teacher>_sigma0.3_seed0.png`, `later_b8_scaling.png`.

1. **GD and heavy ball = GF** (sigma = 0.8, where GD fits in the budget): first escape 8.59 vs 8.58,
   gaps equal to < 1% (binary8 odeco); same per-seed order in 83% (first component 98%; the rest are
   runs cut by the step budget).
2. **Fan-in GF stalls on MPS8 odeco** (0/12 converged by t = 2e6; dormant components die, overlaps
   ~1e-40, R12 trap) and partly on binary8 (4/6 at sigma = 0.3); the balanced version of the same W_0
   converges (12/12 binary8, 10/12 MPS8 by t = 2e6). Adam does not rescue MPS8 fan-in at lr 1e-3 (0/12
   converged in 3e5 steps) but does in part at lr 1e-2 (7/12); on binary8 fan-in Adam converges 11/12.
3. **Adam's first escape is sigma-independent on deep trees too**: 107-115 steps (lr 1e-3) and 19-23
   steps (lr 1e-2) for sigma from 0.25 to 0.8 on binary8 (p = 0.04 +- 0.02 and 0.18 +- 0.05), while GF's
   grows 260-fold (p = 4.74 +- 0.18; theory K - 2 = 5).
4. **Adam's later plateaus do scale with sigma, with about half GF's exponent** (`later_b8`, 7 sigma in
   [0.25, 0.8], 6 seeds; only 0.5 decade, so indicative): p = 2.19 +- 0.32 / 2.65 +- 0.32 (lr 1e-3, gaps
   1 and 2), 2.40 +- 0.52 / 2.96 +- 0.33 (lr 1e-2), against GF 4.72 +- 0.40 / 4.64 +- 0.64. They are long
   (up to 8e4 steps at sigma = 0.25) and S2S by the rho > 3 rule in 83-100% of runs at sigma <= 0.4.
   This is the AEoS phase of 1.2 (e'): once a component is learned, Adam is GD with an lr-independent
   step, so the GF time scale of the dormant route (which depends on sigma through its amplitude) comes
   back, partly. Why the exponent is about half is open (conjecture: Adam's sign phase changes the
   dormant amplitudes in a sigma-dependent way).
5. **Order.** Adam / RMSProp pick the same first component as GF from the same init in 92-94% of deep
   runs (full order 66-74%; binary4: 86-98%): the selection rule (largest weight x overlap^(K-2), i.e.
   largest gradient) is the same because the sign of the total gradient is set by the dominant
   component. The locality order of the E8 teacher survives: MPS8 learns q0, qA (k = 2), then qB (k = 6)
   in 4-5 of 6 runs for GF and Adam alike; binary8 learns q0, qB, qA in 6/6 for GF, Adam and RMSProp at
   sigma = 0.3 (theory: tied k = 5 routes, weight favours qB).
6. **Gauge-invariant methods.** ALS and DMRG reach the teacher in less than one sweep (7-11 micro-steps,
   0.05 s) on every deep problem, with event times identical at sigma = 0.3 and 0.8 (proven, 1.6);
   Riemannian GD at lr 1 in 6-9 steps (identical at both sigma); orders at chance level (first
   component as GF's in 31-54%, chance 33%). The rank-growth pattern differs from GF in kind: the bond
   spectra jump to their final rank within a few micro-steps as the sweep passes each edge.
7. **Re-gauged GD is sequential on deep trees**, with sigma-independent plateaus (`later_b8`: event
   times identical to 4 digits for sigma = 0.3-0.5; p = -0.02 +- 0.21, 0.20 +- 0.25, -0.05 +- 0.59), set by
   the overlaps of the random initial frames with the teacher (correction to 1.4). It also stalls: on
   MPS8 odeco only 1/24 runs converged within 1e5 steps.
8. **Gram-preconditioned GD** (scale-free damping) is unreliable on deep trees: overshoot then return,
   converging in 1/6 runs on MPS8 odeco at sigma = 0.3 within 1e5 steps (4/6 binary8), 6/6 on the E8 teacher.
9. **Balance.** Adam / RMSProp end with balance index 0.37-0.82 (GF 0.000 from balanced init); the
   entry-count heuristic of 1.2 (f) captures the trend across edges (heuristic 0, 0.14, 0.28, 0.6 ->
   measured ~0.2, 0.25, 0.35, 0.45-0.5) but misses a floor of ~0.2 for equal-size cores (`balance.png`).

### 2.6 Finite samples: implicit bias (`implicit_bias_b4`, 288 runs; figure `implicit_bias_b4.png`)

binary N = 4, d = 3, chi = 4 (W has 81 entries; the rank-(3,3) manifold has dimension 45), odeco (3, 2, 1),
n = 30, 45, 60, 90 Gaussian samples, sigma = 0.01 and 0.1, 6 seeds, trained to empirical loss < 1e-8 L0
(GF with t_max = 1e6 to make sure it interpolates; RMSProp never reaches that tolerance and is reported
at its final state).

- n = 30: no method recovers the teacher (population loss 12-40 > L0 = 7). n = 90: all do.
- n = 45 (= manifold dimension): GF ends with bond ranks (3, 3) in 11/12 runs (one rank-4 edge);
  Adam lr 1e-3 in 8/12, Adam lr 1e-2 in 9/12 and RMSProp in 11/12 runs end with a rank-4 edge
  (bond dimension fully used); no method recovers the teacher.
- n = 60: teacher recovered (population loss < 0.01) by GF in 10/12, Adam lr 1e-3 in 9/12, RMSProp in 8/12,
  Adam lr 1e-2 in 6/12; the failures are rank-4 interpolants with population loss ~2.

**Empirical**: GF from small init has the strongest low-bond-rank bias; Adam (more so at larger lr and
larger sigma) and RMSProp more often use the full bond dimension and generalize worse at intermediate
sample sizes, as expected from the loss of gauge equivariance (Singh 2026, abstract only).

## 3. Synthesis (for practitioners and the report)

**Why GF has plateaus that scale with the init, and what removes them (derived).** Near the origin the
gradient is ~sigma^(K-1) while the displacement needed to learn a component is O(1) (relative to the
init: O(sigma) to leave the saddle neighbourhood, then O(1)). Any method whose step is invariant under a
rescaling of the gradient near the origin escapes in a sigma-independent number of steps:
Adam / RMSProp (per-coordinate 1/sqrt(v), 1.2), Gauss-Newton / Riemannian GD (tangent projection, 1.5),
per-core Gram preconditioning ((E^T E)^-1 ~ sigma^-2(K-1), 1.7), GD in a canonical gauge (the root holds the
whole scale and sees an isometric environment, 1.4), ALS / DMRG (exactly scale-free, 1.6). An absolute floor
in the preconditioner (Adam's eps, absolute damping) turns it back into GD with a rescaled step below the
scale where the gradient (or Gram) falls under the floor, so the GF law returns for small enough sigma
(Adam eps = 1e-8 on K = 7-8 trees: sigma < ~0.1).

**Which S2S features survive which optimizer** (binary N = 4 complete; deep trees in section 2.5):

| method | plateaus | first escape vs sigma | later plateaus | order | rank growth | balance |
|---|---|---|---|---|---|---|
| GF / GD (lr < eta_c) / heavy ball | yes (S2S) | sigma^-(K-2) | sigma^-(k-2) (k-rule) | weight x overlap | one singular value at a time, all edges together | conserved (GD: O(lr^2) drift) |
| Adam / RMSProp | first: no; later: yes | sigma-independent (~0.17 / lr steps on binary4) unless sigma < sigma_eps | memory phase (log, beta2-dependent) then AEoS phase (GF-like time, ~lambda_max/38 steps per GF unit) | mostly weight order (86-98% binary4) | sequential | drifts to O(1) imbalance, sharper minima |
| GD + re-gauging | no | O(1), sigma-free | none | projection on random frames | all at once | gauge-fixed |
| Riemannian GD | no | O(1), sigma-free | none | not weight order | all at once | n/a |
| Gram-preconditioned GD | no (overshoot instead, without absolute damping) | sigma-free | none | mixed | all at once | n/a |
| ALS / DMRG | no | exactly sigma-free (proven) | none | not weight order (sweep position) | all at once per bond as the sweep passes | n/a |

**Statement (empirical on binary N = 4, deep trees pending).** Gradient-flow results describe GD and
momentum quantitatively. They describe Adam qualitatively for the order and the sequential nature of
learning, but not for the time scales: Adam's first escape does not slow down as the init shrinks (until
eps takes over), and its later plateaus are governed by its own second-moment memory and adaptive edge
of stability. Methods that act on the tensor W rather than on the cores (Riemannian GD, ALS / DMRG) or
that fix the gauge (canonical form) do not show saddle-to-saddle dynamics at all: they fit all teacher
components together, in an order set by the random initial frames.
