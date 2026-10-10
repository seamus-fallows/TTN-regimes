# Regime diagnostics (definitions used by every experiment)

Regimes are assigned by these measurable quantities, never by eye. The thresholds are conventions:
every phase diagram reports how its boundaries move when the thresholds change (sensitivity check:
delta_K threshold in {0.05, 0.1, 0.2}, separation threshold in {2, 3, 5}). Library functions are in
`ttn/diagnostics.py`; trajectories record the raw quantities (`ttn/dynamics.py`, `Recorder`).
Report Sec. 2.7 follows this file; Appendix B states the rule each experiment used.

Tags: **impl.** = where the rule is implemented; **used by** = experiments that use it with their own
code or deviate from the library. Where several rules exist, the **recommended** one is marked.

## Raw quantities

Recorded at the log times (`ttn.dynamics.default_log_times`: t = 0 and 200 log-spaced times from
1e-3 min(tau_0, tau_L, t_max), tau_0 = |theta|/|d theta/dt|, tau_L = L/|dL/dt| at t = 0). Loss,
overlaps and spectra are also recorded on **fine rows**: after every integrator step at which theta
moved by 1% of its norm or log L by 0.01 since the last kept row (impl. `ttn.dynamics.Recorder`);
`ttn.dynamics.merged_rows` joins log and fine rows, and the library reads events from the merged rows
(`ttn.runner.learning_events`, `summary`).

| symbol | definition | impl. |
|---|---|---|
| L(t) | exact population loss 1/2 \|\|W(t) - W*\|\|_Sigma^2; normalized l(t) = L(t)/L(0) | `Recorder` (loss, pop_loss) |
| s_k^e(t) | singular values of W(t) matricized across edge e (bond / Schmidt spectrum), Sigma-weighted when inputs are anisotropic | `ttn.network.bond_spectra` via `Recorder` (spectra) |
| s_k^{e,*} | the same for the teacher W* (for an odeco teacher: the weights s_r on every edge) | `ttn.runner.target_spectra` (meta['teacher_spectra']) |
| a_r(t) | overlap with teacher component r (CP teachers): <W, Sigma U_r> / <W*, Sigma U_r>, U_r the r-th rank-one term; 0 at W = 0 and 1 at W = W* for any covariance; equals <W, U_r> / s_r for an odeco teacher with whitened inputs | `ttn.teachers.CPTeacher.component_overlaps` |
| delta_theta(t) | \|\|theta(t) - theta(0)\|\| / \|\|theta(0)\|\| | `Recorder` (rel_dist_from_init); per core: experiments/regime_map |
| Theta_t | tangent kernel on a fixed probe set of P = 128 Gaussian inputs: Theta_t = J_t D J_t^T, J_t = d f(probe)/d theta at time t, D = diagonal per-parameter learning-rate multipliers times the gradient mask (D = I by default), so that df/dt = -Theta (f - f*) under gradient flow; for small N also the tensor-space operator J D J^T | `ttn.diagnostics.tangent_kernel`, `tensor_tangent_kernel`; recorded by `ttn.runner.RunRecorder` and experiments/regime_map |
| delta_K(t) | \|\|Theta_t - Theta_0\|\|_F / \|\|Theta_0\|\|_F (name kept for brevity; Theta is the kernel, K the number of cores) | `ttn.diagnostics.kernel_change` |
| A(t) | kernel-target alignment y^T Theta_t y / (\|\|Theta_t\|\|_F \|\|y\|\|^2) on the probe set (y = teacher outputs) | `ttn.diagnostics.kernel_alignment` (`RunRecorder`: ntk_alignment) |
| drift(t) | max_e \|Q_e(t) - Q_e(0)\|_max / (max_e \|Q_e(0)\|_max + max_u \|\|G_u\|\|_F^2 / eta_u), Q_e the gauge conserved quantities; integrator check, must stay < 1e-6 under gradient flow | `ttn.network.conserved_quantities`, `Recorder` (conservation_drift) |
| H(t), H_perp(t) | Hessian of L at theta(t), and its restriction to the complement of the well-conditioned gauge directions (see saddle type) | `ttn.diagnostics.hessian`, `project_out_gauge`; `RunRecorder` (hessian_times), `ttn.runner.plateau_hessians` |

Open: the regime-map review (`notes/reviews/regime_map_round1.md`, B item 3) found delta_K computed from
the probe kernel while the theory constants refer to the tensor-space kernel; fix pending.

## Learning events

### Overlap events (CP teachers whose components are orthogonal in the input metric)

- Rule: event r at the **last upward crossing** of a_r = 1/2 (the first crossing unless a transient
  crossed 1/2 and fell back). Width = duration of the **final rise**: w_r = t(a_r first reaches 0.9) -
  t(last upward crossing of 0.1 before that). Non-learned components can rise transiently while another
  component is learned (observed: a_2 up to 0.41 during the first escape on binary N = 4, then back
  down); a first-crossing time or width would then span the whole plateau and misclassify clean S2S
  runs. Report transients (more than one upward crossing of 1/2).
- impl.: `ttn.diagnostics.crossing_times(..., last=True)`, `rise_times`, `event_widths`, used by
  `ttn.runner.learning_events` / `summary` when meta['problem']['orthogonal_components'] is true.
- used by: s2s-core (event windows re-integrated on linear grids until the rise holds >= 40 points,
  `s2s.zoom_rise`), locality and optimizers (root finding on the integrator's dense output), 00_smoke.
  Deviations: experiments/regime_map `refine_events` keeps the first crossing when a transient precedes
  the final rise (flagged by the regime-map review, fix pending); pilot_h2_locality uses first crossings.

### Spectral (bond) events (non-orthogonal CP teachers, TTN teachers)

One escape can raise several overlaps and later escapes none, so events come from the bond spectra.
Three rules are in use:

- **(a) Onset (recommended).** Event (e, k) at the first time s_k^e reaches f x s_k^{e,*} (the
  teacher's value), **f = 0.05**, sensitivity f in {0.03, 0.1}. No width. Reason: a 50%-of-final rule
  can skip intermediate (blended) saddles whose spectra stop below the final value
  (`theory_teachers.md` section 3.3; report Appendix A, corollary "Second escape").
  impl.: not in the library. used by: data-teacher v2 analyses (`analyze_law_v2.py`,
  `analyze_cross_v2.py`: first crossing after the first escape, on any edge; `analyze_r2_saddle.py`:
  also at least 10x the value at the start), locality bond events (`bond_events2.py`: crossing times
  and effective ranks at f x the teacher's value, index by index).
- **(b) Half the teacher's value.** used by: optimizers (`experiments/optimizers/analysis.py`: first
  crossing of s_k^e / s_k^{e,*} = 1/2, width = final rise 0.1 -> 0.9 (`event_widths`), per edge, no
  merging); data-teacher correlated inputs (`r2_corr.py`: last upward crossing of 1/2 of the teacher's
  value, width = last upward crossing of 0.9 minus that of 0.1); the round-1 growth table of
  data-teacher `analyze_ttn.py` (first crossing, no width).
- **(c) Library default.** impl.: `ttn.diagnostics.spectral_events`, used by
  `ttn.runner.learning_events` / `summary`: curves c(t) = s_k^e(t) / s_k^e(t_end) for the (e, k) the
  target needs (teacher spectra as reference when known); event at the last upward crossing of
  c = 1/2; width = final rise 0.1 -> 0.9 of c. Then `ttn.diagnostics.merge_events`: events sorted by
  time, consecutive events whose gap is at most the larger of their widths (rho <= 1) are merged
  (single linkage); a merged event has the mean time and the largest width of its members.

No code measures a spectral width from f to 0.9 of the teacher's value (an earlier version of this
file said so).

### Resolution

An event is resolved iff its width is at least 3 sample intervals across its rise; otherwise the width
is an overestimate and separation ratios involving it are only lower bounds. impl.:
`ttn.diagnostics.sample_spacing`, `ratio_lower_bounds`; `ttn.runner.learning_events` (resolved,
RESOLVED_FACTOR = 3) and `summary` (n_unresolved).

### Bond ranks

- **Effective bond rank** r_e(t) = #{k : s_k^e(t) > 0.1 x (smallest nonzero teacher Schmidt value on
  that edge)}. impl.: `ttn.diagnostics.effective_rank_relative` with `smallest_nonzero`. used by:
  optimizers (`analysis.effective_ranks`), optim-samples (0.1 x the smallest teacher weight),
  s2s-core plots. Deviation: locality bond events count s_k^e > f x s_k^{e,*} index by index (f = 0.05).
- **Entropy rank** (threshold-free) exp(H(p)), p_k = s_k^2 / sum_j s_j^2. impl.:
  `ttn.diagnostics.entropy_rank`. used by: optimizers, optim-samples.

## Plateaus and the saddle-to-saddle signature

Order events by time. The **separation ratio** between consecutive events i, i+1 is
rho_i = (t_{i+1} - t_i) / max(w_i, w_{i+1}) (impl. `ttn.diagnostics.separation_ratios`). Between two
events with rho_i > 3 the loss sits on a plateau (at a partial-fit level for odeco teachers; for
generic teachers near a critical point of the bond-rank-constrained loss, `notes/data_teacher.md`);
the plateau midpoint is the log time of
minimum parameter speed \|\|d theta / dt\|\| between the two events, and that is where H is evaluated
(impl. `ttn.diagnostics.plateau_midpoints`, `ttn.runner.plateau_hessians`).

A run shows **saddle-to-saddle (S2S)** dynamics when (i) there are at least two events, (ii) every
rho_i > 3, and (iii) across a sweep, plateau durations grow without bound as the init scale goes to 0
(the scaling check: log(1/sigma) or a power of 1/sigma; impl. `ttn.diagnostics.plateau_scaling`).

Plateaus must be judged in **linear** time between events, not only on log-time plots: an escape
needing two cores takes ~log(1/sigma), which can be invisible on a log-time axis after a long first
plateau. Plateaus of the loss alone (no teacher components) are found in linear time with the same
rho > 3 rule by `ttn.diagnostics.find_plateaus` (used by `ttn.runner.summary`, locality multi-component
levels, real-data).

## Regime assignment (one run)

impl.: `ttn.diagnostics.classify_regime` (thresholds `REGIME_THRESHOLDS`), called by
`ttn.runner.summary`; experiments/regime_map calls it from its own runner.

Reading of the rules (agrees with `classify_regime`): "converges" means L(T) < 1e-3 L(0); labels are
checked in the order diverged, stalled, then the others; condition (iii) of the S2S signature is a
sweep-level check, so a single run is labelled S2S on (i)-(ii) and the sweep confirms (iii); a
converged run with a single event (e.g. a one-component teacher, or all components learned together)
is "rich, non-incremental" if delta_K >= 0.1.

| regime | rule |
|---|---|
| **lazy** | loss converges and delta_K(T) < 0.1 at the end of training T (delta_theta small as a corollary) |
| **saddle-to-saddle** | delta_K(T) >= 0.1 and the S2S signature above |
| **rich, non-incremental** | delta_K(T) >= 0.1, loss converges, but some rho_i <= 3 (overlapping transitions) |
| **stalled** | loss not below 1e-3 L(0) by the time budget (e.g. barren-plateau-like initial plateau) |
| **diverged** | loss non-finite or > 1e6 L(0) (GD/SGD; under gradient flow only an integrator failure) |

Labels and references the code uses beyond the table:
- **lazy (projected fit)**: delta_K < 0.1 and final loss within 10% of the loss of the linearized fit
  (`ttn.diagnostics.projected_fit_loss`, `ttn.runner.lazy_floor`; > 0 when a bond is deficient);
  checked before "stalled". impl.: `classify_regime(..., lazy_floor=...)`, set by `ttn.runner.summary`
  when the run has a kernel probe.
- **undetermined**: every rho_i <= 3 involves an unresolved event, so it is only a lower bound.
  impl.: `classify_regime(..., lower_bounds=...)`, set by `ttn.runner.summary`.
- **diverged** is not restricted to GD/SGD: `classify_regime` labels any run with a non-finite final
  loss or one above 1e6 L(0), and `ttn.runner.summary` labels runs that stopped as diverged or
  nonfinite (for gradient flow: an integrator failure).
- **Convergence reference**: experiments/regime_map judges convergence (and its stop floor) against
  1/2 \|\|W*\|\|^2 instead of L(0). The two agree when the fitted tensor vanishes at t = 0 (centred
  and two-copy models: the residual is -W*) and differ only for the plain model at large sigma.

## Saddle type at a plateau

**Primary rule (recommended; sweep level).** Compute lambda_min of the gauge-projected Hessian at the
plateau midpoint for several sigma. **Strict saddle**: lambda_min tends to a negative sigma-independent
constant as sigma -> 0 (k = 2 routes). **Higher-order saddle**: lambda_min -> 0 like sigma^(k-2)
(k >= 3; the origin has k = K). Reason: at a plateau point (not exactly at the saddle) higher-order
saddles show small negative eigenvalues of order sigma^(k-2), which a fixed threshold misclassifies at
moderate sigma (`theory_saddles.md` R10-R11). impl.: `ttn.diagnostics.saddle_scaling`. used by (own fits
of lambda_min against sigma): locality, data-teacher v2 (`analyze_law_v2.py`), s2s-core.

Gauge projection (impl. `ttn.diagnostics.gauge_directions`, `project_out_gauge`, default rtol = 1e-3):
only the left singular vectors of the unnormalized generator matrix with singular values above rtol
times the largest are removed; renormalizing the near-null generators of nearly unused bond indices
(the old default rtol = 1e-10) biased lambda_min by up to ~20% near rank-deficient saddles. Always
report the raw lambda_min too; for saddle-start states at moderate sigma (~1e-2), where bad generators
survive any safe cut, the raw value is the one to quote.

**Fallback (single sigma only, state it).** At the plateau midpoint, compute the gauge-projected
Hessian spectrum. **Strict saddle**: lambda_min < -epsilon (epsilon = 1e-3 x the largest eigenvalue
magnitude at that point, also reported raw). **Higher-order (non-strict) saddle**: no eigenvalue below
-epsilon although the loss later decreases. Report the number of negative eigenvalues and
lambda_min / lambda_max. impl.: `ttn.diagnostics.saddle_type` (rel_eps = 1e-3); recorded by locality
`run.py`, s2s-core `s2s.py`, data-teacher `r2_saddle_runs.py`, `r2_star.py`. This threshold calls the
first plateau (an order-K saddle) strict at every sigma and k = 3 plateaus strict for
sigma >~ 0.01-0.04.

## Reporting rules (conventions; no code)

- Every scaling claim: at least 5 init scales spanning at least one decade, at least 5 seeds,
  mean +- standard error, fitted exponent with its standard error, and the theoretical prediction
  overlaid.
- Every regime boundary: the threshold sensitivity check above.
- Every number states which dynamics produced it (gradient flow with tolerances, GD with step size,
  SGD with batch size).
