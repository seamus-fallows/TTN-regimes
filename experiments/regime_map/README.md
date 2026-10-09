# Experiment `regime_map`: map of training regimes (Q5, H4)

## Question
Where do saddle-to-saddle (S2S), rich non-incremental and lazy dynamics occur as functions of the
per-core init scale sigma, the bond dimension chi, the number of cores K and the root/rest
imbalance, and do the boundaries follow the predictions of `notes/theory_regimes.md`?
Findings: `notes/regime_map.md`. Agent report: `agent-runs/regime-map/REPORT.md`.

## Setup (defaults)
- Whitened Gaussian inputs, exact population loss, d = 3, odeco teacher with weights [3, 2, 1]
  (teacher seed 1000 + init seed, drawn from the independent teacher stream).
- Fan-in init with per-core scale sigma (`TTNSpec.init`), probe set of 128 Gaussian inputs
  (generator seed 12345) for the tangent kernel.
- Models: **centred** (target W* + W_0, `centred_population_loss`; main one-knob family),
  **plain** (target W*), **two-copy** (`antisymmetric_double` + gradient mask, W_0 = 0).
- Labels: rules of `notes/diagnostics.md` (converged = L(T) < 1e-3 L(0); lazy: delta_K(T) < 0.1;
  S2S: every rho_i > 3; rich otherwise; final-rise event widths), at all 9 threshold pairs
  delta_K in {0.05, 0.1, 0.2} x rho in {2, 3, 5}.

## Dynamics used (interim, until `ttn/dynamics.py` lands)
Gradient flow, scipy `solve_ivp` RK45, rtol 1e-9, atol 1e-14, float64 (`pilots/interim.py`).
Stopped when l < 1e-12, or l < 1e-4 and t |grad L|^2 < 1e-7 L(0), or at the budget
T = 50 t_esc + 1e4 / max(lambda_bar, 1), where t_esc = (0.3 sigma)^-(K-2) / ((K-2) s_min) and
lambda_bar = tr Theta(0) / D (kernel scale). Accuracy check: drift of the gauge conserved
quantities Q_e (column `q_drift`, relative to the largest Gram).

## How to run
```
# sigma x chi, binary N = 4 (centred, plain, two-copy; 8 chi x 19 sigma x 6 seeds)
uv run python experiments/regime_map/pilots/sweep_b4_sigma_chi.py
# per-seed lazy-limit predictions (residual, kappa)
uv run python experiments/regime_map/lazy_predict.py b4 binary 4 3 3,4,6,9,12,18,24,36 6 nocap
# figures + boundary tables
uv run python experiments/regime_map/plots.py sigma_chi b4_sigma_chi centred experiments/regime_map/results/lazy_pred_b4.csv
```
Raw outputs (per-run trajectories `traj/*.npz`, summaries `results.jsonl`) go to
`/root/outputs/regime_map/<exp>/`; rerunning the sweep script regenerates them (runs already in
`results.jsonl` are skipped).

## Variants
Edit the grid lists at the top of a sweep script (CHIS, SIGMAS, SEEDS, models); `run_point` also
takes `root_scale` (imbalance), `lr` (sp | ntk), `bonds` (explicit per-edge bond dimensions by
subtree size), `tau_max`.

## Results
(in progress; see `results/` and `notes/regime_map.md`)
