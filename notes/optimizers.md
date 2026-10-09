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
