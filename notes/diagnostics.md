# Regime diagnostics (definitions used by every experiment)

Regimes are assigned by these measurable quantities, never by eye. The thresholds are conventions:
every phase diagram reports how its boundaries move when the thresholds change (sensitivity check:
delta_K threshold in {0.05, 0.1, 0.2}, separation threshold in {2, 3, 5}). Library functions are in
`ttn/diagnostics.py`; trajectories record the raw quantities (`ttn/dynamics.py`, `Recorder`).

## Raw quantities (recorded on a log-spaced time grid plus extra points where needed)

| symbol | definition |
|---|---|
| L(t) | exact population loss 1/2 \|\|W(t) - W*\|\|_Sigma^2; normalized l(t) = L(t)/L(0) |
| s_k^e(t) | singular values of W(t) matricized across edge e (bond / Schmidt spectrum), Sigma-weighted when inputs are anisotropic |
| s_k^{e,*} | the same for the teacher W* (for an odeco teacher: the weights s_r on every edge) |
| a_r(t) | overlap with teacher component r (CP teachers): <W(t), U_r> / s_r with U_r the r-th rank-one term; a_r = 1 when learned |
| delta_theta(t) | \|\|theta(t) - theta(0)\|\| / \|\|theta(0)\|\| (also per core) |
| Theta_t | tangent kernel on a fixed probe set of P = 128 Gaussian inputs: Theta_t = J_t D J_t^T, J_t = d f(probe)/d theta at time t, D = diagonal per-parameter learning-rate multipliers times the gradient mask (D = I by default), so that df/dt = -Theta (f - f*) under gradient flow; for small N also the tensor-space operator J D J^T |
| delta_K(t) | \|\|Theta_t - Theta_0\|\|_F / \|\|Theta_0\|\|_F (name kept for brevity; Theta is the kernel, K the number of cores) |
| A(t) | kernel-target alignment y^T Theta_t y / (\|\|Theta_t\|\|_F \|\|y\|\|^2) on the probe set (y = teacher outputs) |
| Q_e(t) | gauge conserved quantities; drift max_e \|\|Q_e(t) - Q_e(0)\|\| is an integrator-accuracy check (must stay < 1e-6 relative under gradient flow) |
| H(t) | Hessian of L at theta(t), restricted to the complement of the gauge directions |

## Learning events

- For CP teachers, event r occurs at t_r = first time a_r(t) crosses 1/2. Its width is
  w_r = t(a_r = 0.9) - t(a_r = 0.1).
- For general teachers, event (e, k) occurs when s_k^e(t) first crosses half of its final value; its
  width is the time between the 10% and 90% crossings of that final value (as for components). The
  event set is the distinct times after merging events closer than their widths (a single
  component typically raises several bond spectra at once).
- The **effective bond rank** r_e(t) = #{k : s_k^e(t) > theta_r} with theta_r = 0.1 x (smallest
  nonzero teacher Schmidt value on that edge); the threshold-free alternative is the entropy rank
  exp(H(p)), p_k = s_k^2 / sum_j s_j^2.

## Plateaus and the saddle-to-saddle signature

Order events by time. The **separation ratio** between consecutive events i, i+1 is
rho_i = (t_{i+1} - t_i) / max(w_i, w_{i+1}). Between two events with rho_i > 3 the loss sits on a
plateau at the level of a partial fit; the plateau midpoint is the time of minimum parameter speed
\|\|d theta / dt\|\| between the two events, and that is where H is evaluated.

A run shows **saddle-to-saddle (S2S)** dynamics when (i) there are at least two events, (ii) every
rho_i > 3, and (iii) across a sweep, plateau durations grow without bound as the init scale goes to 0
(the scaling check: log(1/sigma) or a power of 1/sigma).

Plateaus must be judged in **linear** time between events, not only on log-time plots: an escape
needing two cores takes ~log(1/sigma), which can be invisible on a log-time axis after a long first
plateau.

## Regime assignment (one run)

Reading of the rules: "converges" means L(T) < 1e-3 L(0); labels are checked in the order diverged,
stalled, then the others; condition (iii) of the S2S signature is a sweep-level check, so a single
run is labelled S2S on (i)-(ii) and the sweep confirms (iii); a converged run with a single event
(e.g. a one-component teacher, or all components learned together) is "rich, non-incremental" if
delta_K >= 0.1.

| regime | rule |
|---|---|
| **lazy** | loss converges and delta_K(T) < 0.1 at the end of training T (delta_theta small as a corollary) |
| **saddle-to-saddle** | delta_K(T) >= 0.1 and the S2S signature above |
| **rich, non-incremental** | delta_K(T) >= 0.1, loss converges, but some rho_i <= 3 (overlapping transitions) |
| **stalled** | loss not below 1e-3 L(0) by the time budget (e.g. barren-plateau-like initial plateau) |
| **diverged** | GD/SGD only: loss non-finite or > 1e6 L(0) |

## Saddle type at a plateau

At the plateau midpoint, compute the gauge-projected Hessian spectrum. **Strict saddle**:
lambda_min < -epsilon (epsilon = 1e-3 x the largest eigenvalue's magnitude scale at that point, also
reported raw). **Higher-order (non-strict) saddle**: no eigenvalue below -epsilon although the loss
later decreases. Report the number of negative eigenvalues and lambda_min / lambda_max.

## Reporting rules

- Every scaling claim: at least 5 init scales spanning at least one decade, at least 5 seeds,
  mean +- standard error, fitted exponent with its standard error, and the theoretical prediction
  overlaid.
- Every regime boundary: the threshold sensitivity check above.
- Every number states which dynamics produced it (gradient flow with tolerances, GD with step size,
  SGD with batch size).
