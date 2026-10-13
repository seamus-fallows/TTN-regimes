# route_generic: does "depth = number of cores that change" hold outside pattern targets?

Agent `route-generic` (brief `notes/briefs/route_theory.md` section 4.4). Pre-registration, results and
deviations: `notes/route_generic.md`. Report: `agent-runs/route-generic/REPORT.md`.

## Question

The route theory (`report/ttn_saddle_order_and_aligned_training_dynamics.pdf`) is proved for pattern
targets (sums of orthogonal one-hot product tensors). Its picture: training from a small init of scale
sigma learns in steps; each step has an effective depth equal to the number of cores that must change
for it; the plateau before the step lasts ~ sigma^-(depth - 2) (log(1/sigma) for depth 2). Does this
hold, event by event, for random CP teachers with non-orthogonal components (per-site overlap 0.3 and
0.6), random TTN teachers and MNIST, and where does it stop holding? A pattern target is run through
the same pipeline as a control.

## Setup

- Trees mps6 (6 cores) and binary8 (7 cores: 4 bottom cores with 2 sites each, 2 internal, root).
- Targets (`common.py`): `pattern` (star-condition pattern targets, d = 2: mps6 000000 / 001100 /
  101000 / 111110, binary8 00000000 / 00000111 / 01000100 / 10100100, weights 1, 0.7, 0.5, 0.35),
  `cp03` / `cp06` (3 components, d = 3, per-site overlap 0.3 / 0.6 between every pair, random weights
  in [0.3, 1]), `ttn` (random TTN on the student's tree, bond dimension 2, d = 3). Student bond
  dimension = target blocks / rank + 2.
- Whitened inputs, exact population loss on the dense tensor, fan-in Gaussian init, float64, one thread
  per process. Gradient flow: scipy `LSODA` stepped directly, exact Hessian as Jacobian (library
  `Problem.hessian`), rtol 1e-10, atol 1e-15, every accepted step recorded (`run.py`).
- sigma = 10^-3.5 ... 10^-1 (quarter decades, 11 values); pattern: 30 init seeds; cp03, cp06, ttn: 5
  teachers x 6 init seeds.
- MNIST: the real_data pipeline unchanged (`experiments/real_data/run.py`), MNIST 3 vs 5, 4 x 4
  pixels, raw cos/sin features, binary16 quadtree and mps16 snake, chi = 4, sigma = 0.005 ... 0.5
  (9 values), seeds 0-9, horizon tau = 1e4 after the first escape (`configs/mnist.yaml`).
- Events (`events.py`, `mnist_events.py`): a rise is a bond singular value s_k^e growing from the
  noise floor (rise time = last upward crossing of half its target value; onset = last upward crossing
  of 5 % of it; MNIST: the ratio s_k^e / s_1^e crossing 0.05, onset 0.005); rises with overlapping
  [onset, rise] intervals form one learning event. Full definitions and decision rules:
  `notes/route_generic.md` sections 3-5.

## How to run

```
# raw outputs (not in git) under /root/outputs/route-generic/ (any directory works)
uv run python experiments/route_generic/run.py experiments/route_generic/configs/synthetic.yaml /root/outputs/route-generic/synthetic 64
uv run python experiments/real_data/download.py /root/outputs/route-generic/data
REAL_DATA_DIR=/root/outputs/route-generic/data uv run python experiments/real_data/run.py \
    experiments/route_generic/configs/mnist.yaml /root/outputs/route-generic/mnist 56
# tables and figures
uv run python experiments/route_generic/analyze.py /root/outputs/route-generic/synthetic /root/outputs/route-generic/mnist experiments/route_generic/results 32
uv run python experiments/route_generic/figures.py experiments/route_generic/results
```
Both runners are resumable (finished runs are skipped). `configs/pilot.yaml` and `configs/mnist_pilot.yaml`
are the timing pilots (not part of the results). Variants: edit a YAML (keys: target, family, tree, d,
patterns + weights | n_comp + overlap | teacher_chi, teacher_seeds, seeds, sigmas, rtol, atol, tau_max,
stop_rel, wall, max_steps).

## Files

| file | role |
|---|---|
| `common.py` | trees, targets (pattern, CP with exact per-site overlap, random TTN), cut spectra, k_region |
| `run.py` | gradient-flow driver (one .npz per run with every solver step) and process pool |
| `events.py` | rises, learning events, k_region, durations, effective depths, floor flags (synthetic) |
| `mnist_events.py` | the same for MNIST runs of the real_data pipeline (data-weighted spectra) |
| `analyze.py` | CSV tables: runs, events, rises, exponents (per-series fits), summary (decisions) |
| `figures.py` | draft figures from the CSVs |
| `tables.py` | markdown tables (`results/summary.md`) from the CSVs |
| `plateau_hessian.py` | exploratory: Hessian at the middle of the plateau before each event (two sigma), from the stored state at the first escape |

## CSV columns (`results/`)

Identifying columns in every file: `target` (pattern, cp03, cp06, ttn, mnist), `tree` (mps6, binary8,
binary16_quadtree, mps16_snake), `d` (physical dimension per site), `chi` (largest student bond
dimension), `sigma` (init scale; not in exponents/summary), `seed` (init seed), `teacher_seed` (0 for
pattern and MNIST). Times are gradient-flow times (dimensionless; the loss is 1/2 ||W - W*||^2).

**runs.csv** (one row per run): `status` (converged = L < 1e-7 L0; unfinished = tau > 1e9; wall;
stalled; failed; MNIST: completed = horizon reached), `trapped` (not converged and a target singular value
never rose, final value < 1e-10 of target), `n_steps` (solver steps), `restarts` (solver clock re-zeros),
`solver_fails`, `wall_s` (seconds), `t_a` (time of the first escape, L < L0/2; MNIST: end of phase 1),
`final_loss_rel` (final loss / initial loss), `n_rises_target` (target singular values above the
rank-one level), `n_rises_learned`, `n_events`, `n_floor_flagged_events`, `n_relearn` (extra upward
crossings), `nonstiff_at_end` (LSODA in Adams mode at the end), `tau_end` (MNIST horizon reached).

**events.csv** (one row per learning event of a run): `event` (0 = escape from the origin, then in time
order), `rises` (the rises of the event, "edge:k" with k = 0 for the largest singular value), `prior`
(rises completed before the event), `bonds` (edges that gain rank; edge e joins core e and its parent),
`n_bonds`, `k_region` (cores of the smallest subtree containing the end cores of those edges),
`n_components` (connected pieces of the bond set: > 1 means separate routes rose together),
`k_max_component` (cores of the largest piece), `t_event` (first rise time of the event on the clock
tau = t - t_a; event 0: absolute time), `duration` (plateau before the event: t_event minus the previous
event's t_event; event 0: absolute time of its first rise; MNIST event 0: t_a), `eff_depth` (median over
the event's rises of the growth-law depth, window y/s in [1e-4, 1e-2]), `eff_depth_min`, `eff_depth_max`,
`n_depth` (rises with an estimate), `eff_depth_wide` (same, window [1e-3, 1e-1], secondary), `floor_min`
(smallest s_k^e / s_1^e of the event's rises on the plateau before onset), `floor_flag` (floor_min <
1e-14: excluded from fits), `relearn`, `status` (run status). Pattern and CP rows: `component` (target
component whose amplitude <U_a, W> crosses half its target value closest to the event), `amp_dt`,
`eff_depth_amp` (growth-law depth on that amplitude), `forgotten`. Pattern rows: `J_before` (components
learned before), `c_pred` (route depth c_J(r) of the paper), `pred_bonds` (edges of R_J(r)),
`bonds_match` (measured bonds = predicted), `greedy_next` (component the greedy rule predicts from
J_before). MNIST rows: `merged_into_escape` (rises merged into event 0), `t_escape_peak` (rate peak of
the first loss drop on the tau clock), `below_meas_floor` (a rise below 1e-6 relative on the plateau:
onset not resolved).

**rises.csv** (one row per tracked singular value of a run): `edge`, `k`, `target` (teacher value;
synthetic only), `learned`, `n_up` (upward crossings of half the target), `final` (final value / target;
MNIST: final ratio s_k / s_1), `t_on`, `t_half` (onset and rise time), `event`, `depth`, `depth_points`,
`depth_wide`, `floor_min`.

**exponents.csv** (one row per series = target, tree, teacher seed, init seed and event signature
(`prior`, `rises`); the same event followed across sigma): `event` (median index), `event_min`,
`event_max`, `bonds`, `n_bonds`, `k_region`, `n_components`, `k_max_component`, `n_sigma_present`,
`n_points` (points in the fit window), `n_flagged`, `sigma_min`, `sigma_max`, `span_dec` (decades),
`fitted` (>= 5 points over >= 1.5 decades), `reason`, `p` (duration ~ sigma^-p, least squares in log-log),
`p_se`, `rms_pow` (rms of log residuals), `log_A`, `log_B` (duration = A ln(1/sigma) + B, relative
residuals), `rms_log`, `log_preferred`, `eff_depth` (median over the window), `n_eff_depth`,
`eff_depth_wide`, `eff_depth_amp`, `p_pred` = k_region - 2, `dp` = p - p_pred, `dd` = eff_depth -
k_region, `dp_vs_depth` = p - (eff_depth - 2); pattern series also `component`, `c_pred`, `J_before`,
`pred_bonds`, `bonds_match`.

**summary.csv** (one row per target x tree x event class): `events` (event0 or later), `k_class` (all or
k = value), `n_series`, `n_series_both` (series with both d_eff and p), median and quartiles of `dd`,
`dp`, `p`, `eff_depth`, `k_region`, `frac_dd_within_0.3`, `frac_dp_within_0.3`,
`frac_p_vs_depth_within_0.3`, `frac_log_preferred`, `frac_disconnected` (n_components > 1), `verdict`
(HOLDS / PARTIAL / FAILS / INCONCLUSIVE by the rules of notes/route_generic.md section 5).

**plateau_hessians.csv** (exploratory; one row per event of seeds 0-1 of every teacher at sigma = 1e-3 and
1e-2, for pattern / cp03 / cp06 / ttn on mps6 and cp06 / ttn on binary8): identifying columns, `event`,
`rises`, `prior`, `bonds`, `k_region`, `eff_depth` (from events.csv), `t_prev`, `t_event` (plateau
bounds, tau clock), `n_negative` (Hessian eigenvalues below -1e-6 max|lambda| at the plateau middle; at a
higher-order saddle these are ~sigma-small, so use the scaling file), `lambda_min`, `lambda_min_rel`,
`loss`, `cores_min_dir` / `n_cores_min_dir` (cores carrying >= 10 % of the most negative eigenvector).
**plateau_hessians_scaling.csv**: the same event at sigma = 1e-3 (`_lo`) and 1e-2 (`_hi`), and
`lambda_slope` = d log|lambda_min| / d log sigma (0: strict saddle; c - 2 at a saddle of order c).

## Results

See `notes/route_generic.md` Part II and the report. Figures in `results/`.
