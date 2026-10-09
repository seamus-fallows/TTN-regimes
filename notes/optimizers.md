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
|mhat / sqrt(vhat)| can exceed 1: a single spike gives (1 - beta1) / sqrt(1 - beta2) = 3.2 for default
betas, and the general bound (Cauchy-Schwarz over the gradient history, ignoring bias correction) is
(1 - beta1) / sqrt((1 - beta2)(1 - beta1^2 / beta2)) = 7.3; A is only defined up to such a factor.
(Corrected after review round 1: v1 quoted 3.2 as the bound.)

**(d) eps restores GF below a crossover scale.** If |g_i| << eps, u ~ mhat / eps: Adam is GD
(EMA momentum) with learning rate lr / eps, i.e. GF with t = lr n / eps. The gradient at the init
has rms c_g sigma^(K-1) (measured: c_g = 0.22 binary4, 0.011 binary8, 0.031 mps8, d = 3, chi = 4,
odeco (3, 2, 1)), so Adam enters this GD-like regime for

  sigma < sigma_g = (eps / c_g)^(1/(K-1))   (eps = 1e-8: 2e-4 binary4, 0.10 binary8, 0.12 mps8),

and then first crawls along the GF path until the gradient reaches ~eps:

  n_1 ~ (eps / lr) T_GF(sigma) + n_sign(lr),

with n_sign the sign-phase escape (n_sign lr = tau_sign, 0.17-0.30 here). **The sigma^-(K-2) plateau
law comes back under Adam, with prefactor eps / lr.** It dominates the first escape only once
eps T_GF(sigma) > lr n_sign(lr) = tau_sign, i.e. below the crossover sigma_x defined by
eps T_GF(sigma_x) = tau_sign(lr), which is much smaller than sigma_g when T_GF grows slowly: binary4
(T_GF ~ 0.67 / sigma): sigma_x ~ 4 eps / tau_sign ~ 1e-7 to 1e-8; MPS8 (T_GF = T_0 (0.3 / sigma)^6,
T_0 ~ 1e3-7e3): sigma_x ~ 0.08 (tested, 2.7). sigma_x depends on lr only through tau_sign(lr) = lr n_sign,
which is lr-independent in the sign-flow limit and grows at large lr (0.17 -> 0.30 from lr 1e-4 to 1e-2 on
binary4). Only K = 8 (MPS8) was tested. (Corrected after review round 1: v1 equated the crossover with
sigma_g and quoted "~0.1 for K = 7-8".)

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
GF-time units per step (lambda_max ~ K s^(2 - 2/K) of the learned components). Which phase a given
plateau ends in must be checked with the probe: on binary8 (seed 0, sigma = 0.3) the preconditioned
sharpness stays at 0.05-0.4 of the threshold during the second plateau and reaches it only on the last
one (2.2; review round 1). A plateau that needs GF time T (from the dormant amplitudes Adam has at its
start) therefore lasts

  n(T) ~ (2 / (1 - beta2)) ln(1 + T P_0 (1 - beta2) / (2 lr))           while P is still decaying,
  n(T) ~ n_AEoS + (T - T_AEoS) (1 - beta1) lambda_max / (2 (1 + beta1))  once P is held at the AEoS.

Short GF plateaus (binary N = 4) end in the first, logarithmic phase: weak dependence on sigma and
on lr, a strong one on beta2. Long ones may end in the second, linear phase, where the GF time scale is
back, converted at a fixed rate of ~lambda_max / 38 steps per GF-time unit. Whether the sigma^-(k-2) law
also comes back depends on whether the dormant amplitudes at the start of the plateau still remember
sigma (on MPS8 at small sigma they stay at their init level, 2.7). This model is heuristic; it was not
fitted per seed, and its sigma dependence was not resolved (2.5).

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
components onto the random initial frames, balance index near 1 on every edge. (Review round 1: on binary4
the largest component comes strictly first in 6/6 seeds and the events are spread over 38-112 steps, so the
projections follow the weights there and "together" is too strong.)

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
geometric return lasting ~ln(1/sigma)/lr steps** (W overshoots as sigma^-K(K-1), the loss as
sigma^-2K(K-1) = sigma^-12 at K = 3); events (a_r = 1/2) are crossed at step 1 by the overshoot and are not
meaningful, so events are only defined after the loss has returned below L0 (analysis). The return-time
law fails in kind (2.3). Damping (here: each Kronecker factor + damping x mean eigenvalue)
caps the overshoot. Near the global minimum (b^K = s) the update is stable for lr < 2/K.

## 2. Results

Status: **v2 (fix round after review round 1, `notes/reviews/optimizers_round1.md`)**; the claims that changed
are listed in section 5.

Conventions (all sections): exact population loss, float64. **GF** = scipy LSODA (rtol 1e-9, atol 1e-14) on the
library's velocity and exact Hessian, **with root-finding of every crossing a_r = 0.1, 0.5, 0.9** (exact event
times; round 1 interpolated on an output grid, errors up to 1.7%), stop at L < 1e-6 L0. **Discrete methods**:
events from a fine trace sampled at every step for the first 2000 steps, then every max(1, n_steps / 50000)
steps. Event = last upward crossing of a_r = 1/2 after which a_r stays >= 1/2 (transient crossings are not
events); for methods whose first steps overshoot the loss (Gram-preconditioned GD) events are only defined after
the loss has returned below L0, and a component already above 1/2 at that point has no resolved event. Width =
final rise 0.1 -> 0.9; S2S = every separation ratio rho > 3 (`notes/diagnostics.md`). A gap or first escape
shorter than two output intervals is unresolved and excluded (counted). Two events in the same step / micro-step
= no resolved order. Exponents p (T ~ sigma^-p): per-seed fits over the sigma values where the seed follows its
most common learning order (for plateau durations; not for times to reach a loss level), mean +- s.e. over seeds;
a log law T = a + b ln(1/sigma) is fitted in the same ln T space for comparison. Clocks: GF time for GF, GD
(lr n), heavy ball (lr n / (1 - beta)); steps for Adam, RMSProp, Gram-preconditioned, Riemannian and re-gauged GD;
sweeps for ALS / DMRG. Tables `experiments/optimizers/results/*.csv`, figures `.../results/figures/`, all
regenerated by `make_figures.py --results-dir experiments/optimizers/results` and `consistency.py`.

### 2.1 Binary N = 4 (K = 3), every optimizer (`scaling_b4`: d = 3, chi = 4, odeco (3, 2, 1), fan-in init, 7 sigma in [0.002, 0.128] (1.8 decades) x 6 seeds per method; `scaling_b4_exponents.csv`, `order_independent_seeds.csv`)

| method | p, first escape | p, gap 1 | p, gap 2 | order (same init) | S2S (rho > 3) |
|---|---|---|---|---|---|
| GF (prediction K - 2 = 1) | 1.007 +- 0.001 | 1.008 +- 0.001 | 1.011 +- 0.001 | weight order 100% | 95% |
| GD lr 0.02 | 1.005 +- 0.001 | 1.008 +- 0.001 | 1.011 +- 0.001 | = GF 100% | 95% |
| heavy ball lr 0.002, beta 0.9 | 0.970 +- 0.002 | 1.006 +- 0.003 | 1.010 +- 0.002 | = GF 100% | 95% |
| Adam lr 1e-4 / 3e-4 / 1e-3 / 1e-2 | 0.012 / 0.013 / 0.009 / 0.013 (+- 0.003-0.009) | 0.24 / 0.30 / 0.36 / unresolved in 7 runs | 0.07 / -0.08 / 0.01 / 0.11 (+- 0.04-0.20) | = GF 86-98% | 12-21% |
| Adam lr 1e-3, beta2 0.99 / 0.9 | 0.009 / 0.011 | 0.38 +- 0.16 / 0.14 +- 0.05 | 0.03 / 0.09 | = GF 93-98% | 7% / 0% |
| RMSProp lr 1e-3 / 1e-4 | 0.001 +- 0.019 / 0.012 +- 0.004 | 0.15 +- 0.07 / 0.07 +- 0.04 | 0.04 / 0.10 | = GF 93-98% | 0% |
| GD + re-gauge (every step / 50 steps) | 0.000 +- 0.000 | 0.00 +- 0.00 | 0.00 +- 0.00 | largest first in 6/6 seeds | 0% |
| Riemannian GD lr 1 / 0.1 | events in step 1 / step 7, all components in the same step (6/6 seeds) | no resolved order | | | 0% |
| ALS / DMRG | >= 2 components in the same micro-step in 5/6 and 6/6 seeds | no resolved order | | | 0% |
| Gram-preconditioned GD (2.3) | overshoot; events mostly at the return | | | | |

Time to L < 1e-3 L0: p = 1.001 +- 0.000 (GF), 0.07-0.23 (Adam), 0.05-0.07 (RMSProp), 0.000 (re-gauge, Riemannian,
ALS, DMRG). For GF the power law fits per seed with rms(ln T) = 0.005 against 0.62 for the log law (same space).
For Adam's later gaps the two laws are not distinguished (rms 0.1-0.5 for both).

Findings (empirical unless tagged):

1. **GD reproduces GF**: exponents equal within 0.003; the first-escape shift t_GD - t_GF is 0.053-0.136 GF time
   units and grows by 0.0199 +- 0.0000 per e-fold of 1/sigma (6 seeds), **the asymptotic slope lr (K - 1) / 2 =
   0.020 predicted by `notes/theory_optimizer.md` P6** (later events: 0.0178 +- 0.0013, 11% below the asymptote;
   `theory_optimizer_P6_slope.csv`).
2. **Heavy ball (lr 0.002, beta 0.9, i.e. 0.13 eta_c; only lr <= 0.27 eta_c tested anywhere)** follows GF with a
   delay that is not fixed: 1.00 GF units at sigma = 0.128 growing to 2.56 at 0.002, by 0.376 +- 0.001 per
   e-fold, which is 19.0 x GD's slope = (1 + beta) / (1 - beta) (conjecture: the momentum analogue of P6 carries
   this factor). Hence its first-escape exponent 0.970 instead of 1.007. Q_e drift 1e-5 (GD), 3e-4 (heavy ball)
   relative, vs 1e-11 for GF (`balance.png`).
3. **Adam / RMSProp: the first escape does not depend on sigma** (p = 0.009-0.013, 1.8 decades, 6 seeds per lr),
   as derived in 1.2 (b, c); **the sign-flow limit is reached as lr -> 0**: lr n_1 = 0.296 (lr 1e-2), 0.180
   (1e-3), 0.162 (3e-4), 0.168 (1e-4), seed-to-seed s.d. 0.006-0.02 (`adam_escape.png`, left).
4. Adam / RMSProp learn the components in the same order as GF from the same init (86-98% of the 42 inits), but
   **by the project's rule they are "rich, non-incremental" on binary N = 4**: rho > 3 fails in 79-100% of runs
   (Adam 0-21% S2S, RMSProp 0%), because the first escape is a slow polynomial rise (width comparable to its own
   time). Later gaps depend weakly on sigma (p = 0.0-0.4, smaller for shorter beta2 memory).
5. **Re-gauged GD, Riemannian GD, ALS, DMRG: no sigma dependence** (event times identical across sigma for
   re-gauged GD and Riemannian lr 1; proven for ALS / DMRG, 1.6) and no plateau. These methods give one trajectory
   per seed at every sigma, so **orders are counted over 6 independent seeds**: ALS / DMRG and Riemannian GD have
   no resolved order on binary N = 4 (components cross 1/2 in the same step); re-gauged GD learns the largest
   component strictly first in 6/6 seeds, with events spread over steps 38-112 (not "all at once"). ALS is exact
   after one sweep (1.6).
6. Adam / RMSProp end unbalanced: balance index 0.29-0.39 on the root edges (entry-count heuristic 0.385, 1.2 f);
   GF keeps O(sigma^2) (0.000).

### 2.2 Adam internals (`adam_probe`: 4 single runs, figure `adam_probe.png`)

Median preconditioner P = sqrt(vhat) and lambda_max(P^-1 H) / (38 / lr) along four runs (one seed each, so
illustrations, not statistics). Binary4, beta2 = 0.999: after the first component, P follows the pure-memory
decay beta2^(n/2) and the preconditioned sharpness stays below 0.1 of the threshold until the second component
escapes (memory phase). Binary4, beta2 = 0.9: the memory is too short to matter and Adam reaches the threshold
during the plateaus. Binary8 odeco (seed 0, sigma = 0.3, balanced) and MPS8 E8: P decays, and lambda_max(P^-1 H)
reaches the adaptive-edge-of-stability (AEoS) threshold **only on the last plateau**; on binary8's second
plateau it stays at 0.05-0.4 of the threshold (review round 1). **Empirical, one run each.** The two-phase
picture of 1.2 (e') is therefore illustrated, not established, and which phase sets a given plateau has to be
read off a probe of that plateau.

### 2.3 Gram-preconditioned GD (`precond_b4`: binary N = 4 as in 2.1, 7 sigma x 6 seeds per setting; `scaling_b4` re-run without the loss guard; deep trees in `deep`; tables `precond_b4_table.csv`, `precond_b4_exponents.csv`)

- Scale-free damping (1e-3 x mean eigenvalue per Kronecker factor): the first step overshoots the loss to
  5e2-4e34 L0, growing as sigma^-12 = sigma^-2K(K-1) (W as sigma^-K(K-1), derived 1.7), then returns. Time to
  L < 1e-3 L0, per-seed fits: p = 0.65 +- 0.07 (lr 0.01), -0.23 +- 0.22 (lr 0.1), -0.11 +- 0.13 (lr 0.5); 9, 7
  and 7 of 42 runs did not converge within 5e4 steps. **Not sigma-free at lr 0.01.** At lr 0.1 the time is flat
  (~380 steps for sigma 0.002-0.016), whereas the derived return time (K - 1) ln(1/sigma) / lr changes by ~42
  steps over that range: **the scalar-model return law fails in kind** (not only by a factor 3 as stated in v1).
- Absolute damping floor 1e-4 / 1e-2 per Kronecker factor, lr 0.1: converges in 14-36 steps for every sigma
  (6/6 seeds, no failures), p = 0.035 +- 0.020 and 0.107 +- 0.008: **weakly sigma-dependent, not exactly
  sigma-free**. The prediction of 1.7 that an absolute floor restores the GF plateau fails in this sigma range
  (the floor enters once per bond leg, so near the origin the method is GD with step lr / floor^(#bond legs) =
  10-1e5 GF units per step); not tested at smaller sigma.
- Events: components cross 1/2 during the overshoot; after the loss has returned most of them are already above
  1/2 (no resolved event) and the orders that remain agree with GF's in 0/40 binary4 inits and 9% of deep inits
  (`order_agreement_with_gf.csv`).
- Deep trees (scale-free damping, lr 0.1): the return takes 90-630 steps at sigma = 0.3 and 0-7 at 0.8, the time
  to converge is 147-632 steps at sigma = 0.3 vs 26-35 at 0.8, and MPS8 odeco converges in 9/24 runs (4/12 at
  sigma = 0.3 + 0.8 balanced, 5/12 fan-in) within 1e5 steps. **Gram-preconditioned GD is therefore removed from
  the "plateau-free, sigma-independent" group** (v1 claim); it has no long flat plateau, but its time depends on
  sigma and it is unreliable without an absolute floor.

### 2.4 Consistency with `notes/theory_optimizer.md` v2 (P1, P2, P6; `consistency.py`)

- **P1 / A1 (sharpness at the GF minimum = K s_1^(2-2/K)): confirmed** from balanced inits: Hessian lambda_max at
  the end of converged runs / pattern-rule value (12.98 binary4, 46.03 binary8, 54.71 mps8) = 1.000-1.002 for GF,
  GD and heavy ball (3 runs per tree, init, sigma and method). Fan-in binary8 (sigma 0.3-0.8): GF 1.03
  (1.00-1.07), GD 1.08, heavy ball 1.10, whose minima keep the fan-in imbalance (balance index 0.18-0.32); P1's
  conserved-quantity correction was not computed, so this part is only qualitative.
- **P2 / A2: consistent** (A2 is proven for chi_e = R; here chi = 4 > R = 3, so this is a consistency check, not a
  test): Adam / RMSProp end at imbalanced minima (balance index 0.21-0.96) with lambda_max / pattern rule =
  1.03-1.94 (Adam lr <= 1e-3), 1.07-6.1 (lr 1e-2) and 1.01-1.59 (RMSProp), never below 1. The larger Adam's step,
  the sharper the minimum, unlike GD's edge-of-stability selection (P3); Adam's stability constrains
  lambda_max(P^-1 H) < 38 / lr, not lambda_max(H) < 2 / lr. (`theory_optimizer_P1P2.csv`.)
- **P6 (GD event shift): asymptotic slope confirmed** (2.1, finding 1): 0.0199 +- 0.0000 per e-fold of 1/sigma vs
  predicted lr (K - 1) / 2 = 0.020 (binary4, GD lr 0.02 = 0.13 eta_c, 6 seeds, 1.8 decades; GF events located
  exactly). The path form (1/2) ln |grad L(t_r)| / |grad L(0)| needs the gradient at the exact event and was
  not evaluated (`theory_optimizer_P6.csv` reads it off the output grid: coarse).
- P5, P7, P9 (GD above eta_c, SGD) are outside this experiment.

### 2.5 Deep trees: binary N = 8 (K = 7) and MPS N = 8 (K = 8) (`deep`, 864 runs; `later_b8`, 168 runs)

Problems: odeco (3, 2, 1) with d = 3, chi = 4, and the locality teacher E8 (d = 2, chi = 3); sigma = 0.3 and
0.8; balanced init and fan-in init; 6 seeds each (`deep`). `later_b8`: binary8 odeco, balanced, 7 sigma in
[0.25, 0.8] (0.5 decade), 6 seeds. Tables `deep_table.csv`, `later_b8_exponents.csv`,
`order_agreement_with_gf.csv`, `order_independent_seeds.csv`, `r12_route_ends_mps8.csv`; figures
`deep_side_*.png`, `later_b8_scaling.png`.

1. **GD = GF** (sigma = 0.8, 48 inits, lr 0.01 = 0.23-0.27 eta_c): first escape 1.001-1.006 x GF's (median
   1.002), first gap 0.989-1.015 x. Heavy ball: first escape 1.01-1.11 x GF's (median 1.04), first gap
   0.79-1.71 x; same first component as GF in 100% of resolved inits for both.
2. **Fan-in GF stalls are specific to MPS8 odeco**: GF converges by t = 2e6 in 0/12 fan-in MPS8 odeco runs, but in
   10/12 fan-in binary8 odeco, 11/12 fan-in binary8 E8 and 11/12 fan-in MPS8 E8 runs. **The conservation-law trap
   (`notes/theory_saddles.md` R12) is verified at the route ends** for 11 of the 12 MPS8 stalls: on the leaf edge,
   in the bond direction nu orthogonal to the learned components, the leaf's column has died (norm^2
   2e-26-3e-32) while the parent keeps norm^2 = -nu^T Q_0(0) nu exactly (e.g. 0.035 vs -0.035; fan-in makes Q_0
   negative there), and on the root edge the root side has died (1e-30) while the child keeps nu^T Q_6(0) nu
   exactly. The 12th (sigma = 0.3, seed 5) learned only one component; its root side is dying (4e-10). The
   gauge-balanced version of the same W_0 converges in 10/12 MPS8 odeco runs; the 2 others (sigma = 0.3) are not
   trapped (Q = 0 to 1e-15, both sides equal and alive, 0.004-0.015) but still on a plateau at t = 2e6.
3. **Adam's first escape is sigma-free on deep trees too** (`later_b8`, 42 runs per lr, every event resolved with
   the step-by-step early trace): p = 0.049 +- 0.022 (lr 1e-3, 107-115 steps) and 0.145 +- 0.033 (lr 1e-2, 19-23
   steps; small but not zero), against GF's 4.74 +- 0.05 over [0.25, 0.8] and **4.99 +- 0.00 on sigma <= 0.4
   (theory K - 2 = 5)**.
4. **Adam's later plateaus grow as sigma decreases, but no law is established** (0.5 decade): per-seed local
   slopes at lr 1e-3 are 1.2-5.5 (gap 1, sigma <= 0.4), 0.8-4.3 (gap 1, sigma >= 0.4), 1.6-2.6 and 1.1-3.6
   (gap 0); at lr 1e-2 1.6-4.6, 2.4-3.9, 1.5-4.0 and -0.4-4.6; the per-seed fits over the whole range give 2.2 +- 0.2 / 2.8 +- 0.4 (lr 1e-3) and 2.4 +- 0.5 / 3.0 +- 0.2 (lr 1e-2),
   GF 4.72 +- 0.12 / 4.85 +- 0.41 (gap 0 on sigma <= 0.4: 4.93 +- 0.08). **The v1 statement "about half GF's
   exponent" is withdrawn**, and so is the attribution to the adaptive edge of stability for this plateau (2.2).
   The plateaus are long (up to 8e4 steps at sigma = 0.25) and S2S by the rho > 3 rule in 83-100% of runs at
   sigma <= 0.4.
5. **Order.** Adam / RMSProp pick the same first component as GF in 94-100% of resolved deep inits (full order
   67-83%); the E8 locality order survives (MPS8: q0, qA (k = 2), qB (k = 6) in 4-5 of 6 seeds for GF and Adam;
   binary8: q0, qB, qA in 6/6 for GF, Adam and RMSProp at sigma = 0.3). The GF gap q0 -> qA on MPS8 (a k = 2,
   log-time escape) is, with exact event times, 0.16-8.5 GF units at sigma = 0.3 (median 3.8) and 0.02-6.0 at
   0.8 (v1's 0.03-0.09 were output-grid artefacts); under Adam it is resolved in 50-92% of runs (median 9-37
   steps).
6. **ALS and DMRG** reach the teacher in less than one sweep (7-11 micro-steps, 0.05 s) on every deep problem,
   with event times identical at sigma = 0.3 and 0.8 (proven, 1.6). **They have no resolved learning order**:
   at least two components cross 1/2 in the same micro-step in 59 of the 60 (problem, method, seed)
   combinations over the five problems (`order_independent_seeds.csv`; counted over independent seeds).
7. **Riemannian GD** (lr 1): sigma-free (identical event times at sigma = 0.3 and 0.8; at lr 0.1 the
   (1 - lr) W_0 term makes them differ by 10-55% at sigma = 0.8), converges in 6-9 steps, and is **staged over
   steps 2-6 with the weakest component strictly last in 5/6 seeds on binary8 odeco and binary8 E8** (3/6 and
   2/6 on MPS8 odeco and E8; ties in 2-6 of 6 seeds per problem).
8. **Re-gauged GD** is sequential on deep trees, with sigma-independent event times (`later_b8`: per-seed
   p = -0.02 +- 0.02, 0.00 +- 0.05, 0.02 +- 0.05, 5 seeds), set by the overlaps of the random initial frames with
   the teacher (correction to 1.4); largest component strictly first in 2/6 (binary8 odeco) and 5/6 (binary8 E8)
   seeds. On MPS8 odeco 2/24 runs converged within the 1e5-step budget (1e3 GF time units): budget-limited, not
   shown to be a stall.
9. **Gram-preconditioned GD**: section 2.3 (sigma-dependent on deep trees, MPS8 odeco 9/24 converged).
10. **Balance.** Adam / RMSProp end with balance index 0.37-0.82 (GF 0.000 from balanced init); the entry-count
    heuristic of 1.2 (f) captures the trend across edges but misses a floor of ~0.2 (`balance.png`).

### 2.6 Finite samples (`implicit_bias2_b4`, 288 runs; tables `implicit_bias2_per_draw.csv`, `implicit_bias2_paired.csv`; figure `implicit_bias2_b4.png`)

binary N = 4, d = 3, chi = 4 (W has 81 entries; the rank-(3,3) manifold has dimension 45), odeco (3, 2, 1);
n = 45 and 60 Gaussian samples; **4 independent data draws** (data seeds 200-203) x 6 init seeds x sigma in
{0.01, 0.1}; GF (exact events, t_max 1e6), Adam lr 1e-3 and 1e-2 (5e5 steps); trained to empirical loss
< 1e-8 L0. Comparisons are paired (same draw and init) and use only runs that reached the training tolerance.

- n = 60: every run that reached the tolerance recovered the teacher (population loss < 0.01), for every method;
  all failures are runs that stopped at the step budget (Adam reached the tolerance in 2-6 of 6 runs per draw,
  GF in 4-6). Paired differences of the population loss are ~1e-7: no difference.
- n = 45: no method recovers the teacher (population loss 2.5-26). Adam has the lower population loss in 12/21
  and 8/23 pairs (lr 1e-3; sigma 0.01, 0.1) and 13/21 and 10/23 (lr 1e-2): **no consistent difference**; the
  per-draw medians scatter by a factor ~5 and their ranking between methods changes from draw to draw.
- n = 45, bond rank: Adam lr 1e-2 more often ends with a fourth bond singular value above the threshold
  0.1 x (smallest teacher value) (35/44 paired runs vs 5/44 for GF; Adam lr 1e-3: 16/44); the entropy rank shows
  no systematic difference (per-draw means 2.0-2.7 for every method).

**The v1 claim "GF has the strongest low-bond-rank bias; Adam generalizes worse at n = 45-60" is withdrawn**: it
came from a single data draw per init seed (review round 1: on new draws 1/6 vs 1/6 rank-4 runs), and the n = 60
failures were runs that never reached the training tolerance. What remains (empirical, binary N = 4 only): with
n at the manifold dimension, Adam at the larger lr more often uses the spare bond dimension by a fixed-threshold
count, without a measurable effect on the population loss. The sharper, imbalanced minima of Adam (2.4) are
unaffected by this withdrawal.

### 2.7 Adam's eps restores the GF law (`eps_mps8`: MPS N = 8 (K = 8), odeco, d = 3, chi = 4, balanced init, 8 sigma in [0.01, 0.3] (1.5 decades) x 6 seeds, lr 1e-3 / 1e-2, eps 1e-8 / 1e-30, budget 2e5 steps; `eps_mps8_first_escape.csv`, `eps_mps8_prediction_per_seed.csv`, figure `adam_escape.png`)

- eps = 1e-30 (no floor): first escape 121-139 steps (lr 1e-3) and 23.5-29.3 steps (lr 1e-2) over the whole
  range, all 96 runs; GF needs ~1e3-7e3 time units at sigma = 0.3 and ~sigma^-6 times more below.
- eps = 1e-8: identical to eps = 1e-30 for sigma >= 0.1; below, the escape slows down by ~sigma^-6. Test of
  the derived law 1.2 (d), n_1 = (eps / lr) T_GF(sigma) + n_sign, with T_GF(sigma) = (that seed's exact GF
  first-escape time at sigma = 0.3, `deep` sweep, same init direction) x (0.3 / sigma)^6 and n_sign the same
  seed's eps = 1e-30 escape, **no fitted parameter**: measured / predicted = **0.9989-1.0003** (lr 1e-3,
  sigma = 0.02-0.03) and **1.0009-1.0062** (lr 1e-2, sigma = 0.015-0.02), every seed; in the crossover region
  the deviations are systematic: 1.01-1.04 (lr 1e-2, sigma = 0.03), 0.95-0.99 (lr 1e-3, 0.05), 1.08-1.17 (lr 1e-2,
  0.05). The prediction also decides which runs escape within the 2e5-step budget: 60/60 (seed, sigma, lr)
  cases at sigma <= 0.05 agree. **Derived law confirmed quantitatively** (empirical scope: MPS8 odeco, balanced
  init, first escape, 6 seeds). (v1 quoted 0.97-1.05: the spread came from GF event times read off an output
  grid.)
- The crossover sits at sigma_x ~ 0.05-0.1 here, where eps T_GF(sigma) ~ tau_sign = lr n_sign (1.2 d); it depends
  on lr through tau_sign and on the tree through T_GF: on binary4 it is ~1e-8, far below the sigma_g = 2e-4 where
  the init gradient falls under eps. Only K = 8 was tested.
- Within the 2e5-step budget, no Adam run with sigma <= 0.1 learned all three components (eps 1e-30 and 1e-8
  alike): after the first, sigma-independent escape the dormant components stay at their init amplitude (overlap
  ~1e-20 at sigma = 0.01) and their escape needs far longer.

## 3. Synthesis (for practitioners and the report)

**Why GF has plateaus that scale with the init, and what removes them (derived).** Near the origin the
gradient is ~sigma^(K-1) while the displacement needed to leave the saddle and learn a component is O(1)
relative to the init. A method whose step is invariant under a rescaling of the gradient near the origin
escapes in a sigma-independent number of steps: Adam / RMSProp (per-coordinate 1/sqrt(v), 1.2), Riemannian GD
(tangent projection, 1.5), GD in a canonical gauge (1.4), ALS / DMRG (exactly scale-free, 1.6). An absolute
floor in the preconditioner (Adam's eps) turns it back into GD with a rescaled step, and the GF law returns
once that crawl costs more than the sign phase (confirmed to 0.1% per seed on MPS8, 2.7). GF, GD and heavy ball
keep the GF law. Gram-preconditioned GD (all cores at once) has no flat plateau but its overshoot-and-return
time depends on sigma (2.3).

**What survives which optimizer** (binary N = 4: 7 sigma over 1.8 decades x 6 seeds; binary N = 8 / MPS N = 8:
2-7 sigma over <= 0.5 decade, 6 seeds; exponents are per-seed fits, mean +- s.e.; "seeds" = independent inits):

| method | plateaus / regime | first escape vs sigma | later plateaus vs sigma | learning order | balance / minimum | cost to L < 1e-3 L0 (median) |
|---|---|---|---|---|---|---|
| GF (reference) | S2S | p = 1.007 +- 0.001 (K=3), 4.99 +- 0.00 on sigma <= 0.4 (K=7; theory K-2) | p = 1.01 (K=3), 4.7-4.9 (K=7) | init-selected (weight x overlap) | conserved; flattest (pattern-rule) minimum | 7e3-1.7e4 integrator evaluations |
| GD (lr <= 0.27 eta_c tested) | = GF | = GF; shift lr (K-1)/2 per e-fold (P6, confirmed) | = GF | = GF (100%) | O(lr^2) drift | ~T_GF / lr steps |
| heavy ball (beta 0.9, lr <= 0.27 eta_c) | = GF up to a delay | p = 0.970 (delay 1.0-2.6 GF units, growing as ln(1/sigma)) | = GF | = GF (100%) | small drift | ~T_GF (1 - beta) / lr steps |
| Adam / RMSProp | sequential; rich, non-incremental by rho on K = 3, S2S on K = 7 (sigma <= 0.4) | **sigma-free** (p = 0.01 K=3; 0.05 / 0.15 K=7), lr n_1 -> 0.17 as lr -> 0, until sigma < sigma_x where n_1 = (eps / lr) T_GF + n_sign | grow as sigma decreases; weak at K = 3 (p 0.0-0.4), local slopes 1-5.5 at K = 7; no law | same first component as GF 94-100% | O(1) imbalance (0.2-0.96), sharper minima (1.01-6x) | 1e3-2e5 steps |
| GD + re-gauging | K = 3: no plateau; K = 7-8: sequential, sigma-free plateaus, budget-limited on MPS8 | sigma-free (p = 0.00) | sigma-free | largest first 6/6 (K=3), 2/6-5/6 (K=7) | gauge-fixed | 2e2-2e4 steps |
| Riemannian GD (lr 1) | no plateau | sigma-free, 2-6 steps | none | binary8: staged, weakest last 5/6; binary4: all in step 1 | n/a | 4-9 steps (dense Jacobian each) |
| Gram-preconditioned GD | overshoot then return, no flat plateau | not sigma-free at lr 0.01 (p 0.65) or on deep trees; ~flat at lr 0.1-0.5 (7/42 failures); abs. damping: p 0.04-0.11, no failures | n/a | events mostly unresolved | n/a | 14-36 steps (abs. damping) to 1e5 (MPS8, no convergence) |
| ALS / DMRG | no plateau | exactly sigma-free (proven) | none | no resolved order (59/60 seed-problems tie) | n/a | < 1 sweep (6-11 micro-steps) |

**Statement for practitioners** (empirical; N <= 8, d <= 3, chi <= 4, odeco and one locality teacher, 6 seeds per
setting).
- GD and momentum at moderate steps: the gradient-flow saddle-to-saddle picture applies quantitatively (event
  times, plateau exponents, order); the step-size shifts are small and follow P6 (GD) or P6 x (1 + beta) /
  (1 - beta) (heavy ball, conjecture).
- Adam (the TN-ML default): learning stays sequential and starts with the same component as GF in 94-100% of
  runs, and the locality order of a structured teacher survives; but the first plateau no longer depends on the
  init scale, and later plateaus are set by Adam's own dynamics (second-moment memory, sometimes the adaptive edge
  of stability). They still grow as the init shrinks on a K = 7 tree, without an established law. With the
  default eps = 1e-8, at per-core init scales below sigma_x (~0.05-0.1 on the K = 8 MPS tested here) Adam is GD in
  disguise and inherits GF's sigma^-(K-2) plateau with prefactor eps / lr. Adam ends at unbalanced, sharper
  minima. No generalization difference to GF was found with few samples (binary N = 4, n = 45-60).
- Riemannian GD and ALS / DMRG show no saddle-to-saddle dynamics: no plateau, no init-scale dependence (exactly,
  for ALS / DMRG); ALS / DMRG fit all components within the same micro-steps of the first sweep, Riemannian GD
  within a few steps, staged by weight on binary N = 8. Re-gauged GD sits in between (sequential on deep trees,
  sigma-free). Gram preconditioning needs an absolute damping floor to be reliable.
- Fan-in initialization traps GF (and Adam at lr 1e-3) on MPS N = 8 with the odeco teacher (dormant components
  die by the conservation-law mechanism R12, verified at the route ends); a gauge-balanced version of the same
  initial tensor removes the trap for GF but not the long plateaus (2/6 runs unconverged at sigma = 0.3), and does
  not change Adam's failure rate at lr 1e-2 (4/12 vs 5/12).

## 4. Open problems

- Adam's later plateaus: a law needs >= 1.5 decades. Proposed saddle-start protocol: take an Adam run at the
  moment its first component is learned (parameters and Adam state m, v), rescale the dormant part of the
  parameters (projection on the not-yet-learned components' patterns) by eps over several decades, keep the
  state, and run Adam and GF from there with a probe (P, lambda_max(P^-1 H)) on every plateau. A fresh Adam
  state at a saddle would not do: it has no memory, so it escapes like a first escape.
- Why heavy ball's shift is (1 + beta) / (1 - beta) times GD's (a modified-equation derivation).
- Re-gauged GD on MPS8: stall or slow escape (runs beyond 1e3 GF time units).
- Gram preconditioning: a model of the overshoot-and-return time (the scalar model fails in kind).
- Larger chi, HT trees, non-orthogonal teachers, SGD-type Adam on finite samples (not run).

## 5. Claims changed after review round 1 (`notes/reviews/optimizers_round1.md`)

| v1 claim | v2 |
|---|---|
| Gram-preconditioned GD belongs to the "no plateau, sigma-independent" group; its order agrees with GF in 33% | **withdrawn**: binary4 runs had stopped after one step (missing `blowup`; re-run); time to converge is sigma-dependent (p = 0.65 at lr 0.01; deep trees 26-35 vs 147-632 steps); events mostly unresolved after the overshoot (2.3) |
| ALS, DMRG, Riemannian: "order unrelated to teacher weights" (0% of 42 inits) | counted over 6 independent seeds: ALS / DMRG have **no resolved order** (59/60 ties); Riemannian lr 1 is staged with the weakest component last in 5/6 binary8 seeds; re-gauged GD learns the largest first in 6/6 binary4 seeds (2.1, 2.5) |
| Adam's later plateaus grow at "about half GF's rate" on binary8, AEoS phase | **withdrawn**: 0.5 decade, local slopes 1-5.5 per seed, no law; the probe contradicts AEoS for gap 0 (2.2, 2.5) |
| eps law to 0.97-1.05; sigma_eps ~ 0.1 for K = 7-8 | to **0.1%** with exact GF times (0.9989-1.0003, 1.0009-1.0062); crossover sigma_x depends on lr and tree (~1e-8 on binary4), only K = 8 tested (1.2 d, 2.7) |
| Finite samples: GF keeps rank 3, Adam fills chi = 4 and generalizes worse at n = 45-60 | **withdrawn** (single data draw; n = 60 failures were budget stops); 4 draws: no population-loss difference; only a threshold-rank difference at n = 45 for Adam lr 1e-2 (2.6) |
| Fan-in GF stalls on MPS8 and binary8 (R12); balanced init avoids it, also for Adam | stalls specific to MPS8 odeco (binary8 10/12, E8 22/24 converge); **R12 verified at the route ends**; balanced GF still 2/6 unconverged at sigma = 0.3; Adam's failures unchanged by balancing (2.5) |
| Heavy ball: fixed delay; "lr < eta_c" | delay grows as 0.376 per e-fold = (1 + beta) / (1 - beta) x GD's P6 slope; only lr <= 0.27 eta_c tested (2.1) |
| Adam sequential with S2S on all trees | binary4: rich, non-incremental by rho (79-100% of runs fail rho > 3); S2S on binary8 at sigma <= 0.4 (2.1, 2.5) |
| P2/A2 "confirmed" | "consistent" (A2 needs chi_e = R) (2.4) |
| P6 consistent in magnitude only | asymptotic slope **confirmed** (0.0199 vs 0.020) (2.4) |
| binary8 GF exponent 4.76 | 4.74 +- 0.05 over [0.25, 0.8]; 4.99 +- 0.00 on sigma <= 0.4 |
| log law rms 370 (artefact) | fair fit in ln T: 0.62 vs 0.005 (GF, binary4) |
| loss overshoot ~ sigma^-6; Adam step bound 3.2 | sigma^-12 (W ~ sigma^-6); general bound 7.3 (3.2 = single spike) |
| re-gauged GD "stalls" on MPS8 | budget-limited at 1e3 GF time units |
| MPS8 E8 q0 -> qA gap 0.03-0.09 (unresolved) | 0.16-8.5 GF units with exact event times |

Code fixes: Gram runs without the loss guard; transient crossings are no longer events; events after a loss
overshoot only; same-step ties flag unresolved orders; GF events by root finding; fine trace every step for the
first 2000 steps; fair log-law fit; float round trip in the summary cache (the v1 `adam_escape.png` lacked the
eps = 1e-30 curves); `run.py` uses the spawn context and `--dry` creates no directory. Superseded runs were moved to
`/root/outputs/optimizers/<sweep>/superseded/` (not deleted).
