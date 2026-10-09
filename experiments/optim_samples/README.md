# optim_samples: learning rate, SGD noise and finite samples (Q8)

Agent `optim-samples`. Findings and derivations: `notes/optim_samples.md`. Report:
`agent-runs/optim-samples/REPORT.md`.

## Question
How do discrete-step gradient descent, SGD noise and a finite training set modify the gradient-flow
(GF) picture of TTN training regimes? Where is the learning-rate axis on the regime map, when do GF
results apply to practical training, and what does saddle-to-saddle imply for generalization?

## Setup (common)
Whitened Gaussian inputs, odeco teacher [3, 2, 1] (d = 3) with **teacher seed 101** (independent
of init seeds 0-4), student chi = 3, fan-in init. Trees: binary N=4 (K=3), binary N=8 (K=7),
MPS N=8 (K=8). eta_max = 2 / (K s_1^{2-2/K}). Step 3 uses binary N=8, d = 2, teacher [3, 2],
student chi = 4. All float64 on CPU. The GD/SGD loops here are minimal and were written before
`ttn.dynamics` landed; GF references use `scipy` DOP853 (rtol 1e-10, atol 1e-14).

## Scripts (run from the repo root with `uv run python experiments/optim_samples/<script>`)
| script | what | output |
|---|---|---|
| `check_sharpness.py` | top Hessian eigenvalue at balanced / gauged / imbalanced minima vs K s_1^{2-2/K} | `results/sharpness_check.csv` |
| `map1d.py` | bifurcation diagram of GD on one balanced mode | `results/map1d_attractor.csv`, `map1d_points.csv` |
| `sgd_stability.py [tree N ...]` | exact linear mean-square stability limit of online SGD vs B | `results/sgd_stability.csv` |
| `gd_sweep.py configs/gd_sweep.yaml` | step 1: GF references + GD at 18 step sizes (750 runs, ~25 min on 28 workers) | raw: `/root/outputs/optim_samples/gd/` |
| `predict_shift.py` | first-order (backward-error) prediction of GD event shifts | `results/gd_shift_prediction.csv` |
| `analyze_gd.py` | step-1 table and outcome classes; exports EoS example trajectories | `results/gd_sweep.csv`, `gd_summary.csv`, `eos_example_*.csv` |
| `plot_gd.py` | step-1 figures | `results/fig_gd_vs_gf.png`, `fig_stability.png`, `fig_eos_example.png`, `fig_min_selection.png` |
| `sgd_sweep.py configs/sgd_sweep.yaml` | step 2: online SGD, B = 1..256, eta sweeps, aligned init (3960 runs, ~2.5 h) | raw: `/root/outputs/optim_samples/sgd/*.jsonl` |
| `samples_sweep.py configs/samples_sweep.yaml` | step 3: fixed training sets, small vs centred large init, lazy limit | raw: `/root/outputs/optim_samples/samples/*.jsonl` |

Order to regenerate everything: `check_sharpness.py`, `map1d.py`, `sgd_stability.py`, `gd_sweep.py`,
`predict_shift.py`, `analyze_gd.py`, `plot_gd.py`, then the step-2/3 sweeps and their analysis
scripts. Sweeps are resumable (finished runs are skipped). Variants: edit or copy a YAML config
(trees, sigmas, seeds, ratios / batch sizes / n_train); `--workers` sets the process count (one
torch thread each).

## Results summary (details, tags and numbers in `notes/optim_samples.md`)
- Sharpness at the balanced global minimum is exactly K s_1^{2-2/K} (proven; checked to 1e-15), the
  balanced minimum is the flattest aligned one, so eta_max = 2/(K s_1^{2-2/K}).
- Small init: GD converges like GF for every eta <= 0.95 eta_max and never for eta >= 1.05 eta_max
  (binary N=4, 5 init scales x 5 seeds). Event-time shifts are predicted exactly at O(eta) by the
  backward-error variational equation; relative errors vanish as sigma -> 0.
- Above eta_max: bounded oscillation of the top component(s) (1.05-1.3), divergence, and a regime
  in which learned components are thrown back to the origin saddle and re-learned (2 eta_max).
- Imbalanced init: GD selects minima with sharpness min(lambda_GF, 2/eta) and reduced imbalance.
- SGD (pilot, to be finalized): noise does not shorten plateaus; event-time spread ~ sqrt(eta/B).
