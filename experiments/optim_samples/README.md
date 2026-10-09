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
| `gd_sweep.py configs/gd_sweep.yaml` | step 1: GF references + GD at 18 step sizes (750 runs) | raw: `/root/outputs/optim_samples/gd/` |
| `gd_sweep.py configs/gd_order.yaml` | 40 more init seeds (order flips below eta_max) | same |
| `gd_sweep.py configs/gd_centred.yaml` | centred model, sigma 0.3-2: stability limit across the lazy transition | same |
| `predict_shift.py` | first-order (backward-error) prediction of GD event shifts | `results/gd_shift_prediction.csv` |
| `analyze_gd.py` | step-1 table and outcome classes; exports EoS example trajectories | `results/gd_sweep.csv`, `gd_summary.csv`, `eos_example_*.csv` |
| `analyze_order.py` | order flips vs eta, near-tie measure | `results/order_flips.csv`, `fig_order_flips.png` |
| `plot_gd.py` | step-1 figures | `fig_gd_vs_gf.png`, `fig_stability.png`, `fig_eos_example.png`, `fig_min_selection.png` |
| `plot_regime.py` | learning-rate axis of the regime map | `fig_lr_regime_map.png` |
| `sgd_sweep.py configs/sgd_sweep.yaml` | step 2: online SGD, B = 1..256, eta sweeps, aligned init | raw: `/root/outputs/optim_samples/sgd/*.jsonl` |
| `analyze_sgd.py`, `plot_sgd.py` | step-2 tables (incl. second plateau vs P9) and figures | `results/sgd_*.csv`, `fig_sgd.png`, `fig_sgd_stability.png` |
| `p9_sgd.py`, `plot_p9.py` | test of the theory note's P9/P10 (SGD on a later plateau; uses `ttn.dynamics.run_sgd`) | `results/p9_sgd.csv`, `p9_fit.csv`, `fig_p9.png` |
| `collapse_test.py` | stochastic collapse of a k = 2 route with label noise, MPS N=4 (`ttn.dynamics.run_sgd`) | `results/collapse_test.csv` |
| `check_k2_saddle.py` | gauge-projected Hessian lambda_min at the k = 2 plateau vs sigma (saddle type) | `results/k2_saddle_sweep.csv`, `k2_saddle_hessian.json` |
| `sgd_limit_test.py` | empirical SGD stability limit at the balanced minimum (B = 1, 16) | `results/sgd_limit_test.csv` |
| `samples_sweep.py configs/samples_sweep.yaml` | step 3: fixed training sets, small vs centred large init, exact lazy limit | raw: `/root/outputs/optim_samples/samples/*.jsonl` |
| `analyze_samples.py` | step-3 tables and figure | `results/samples_summary.csv`, `fig_samples.png` |
| `validate_library.py` | reruns sample GD/GF/SGD settings with `ttn.dynamics` and compares | `results/validate_library.csv` |

All scripts take `--out-dir DIR` (summaries and figures; default `/root/outputs/optim_samples/results`,
outside git, so a re-run never overwrites the committed results; pass `--out-dir
experiments/optim_samples/results` to regenerate them) and `--raw-dir DIR` (raw per-run data; default
`/root/outputs/optim_samples`); the sweeps take `--workers N`. Analysis scripts read the CSVs written by
earlier scripts from the same `--out-dir`.

Helpers: `common.py` (setup, Problem with centred variant, GF reference via scipy DOP853, events),
`plot_style.py`. Order to regenerate everything: `check_sharpness.py`, `map1d.py`,
`sgd_stability.py`, the three `gd_sweep.py` configs, `predict_shift.py`, `analyze_gd.py`,
`analyze_order.py`, `plot_gd.py`, `plot_regime.py`, `sgd_sweep.py`, `analyze_sgd.py`, `plot_sgd.py`,
`p9_sgd.py`, `plot_p9.py`, `collapse_test.py`, `check_k2_saddle.py`, `sgd_limit_test.py`,
`samples_sweep.py`, `analyze_samples.py`; `validate_library.py` any time after the GD/SGD sweeps. Sweeps are
resumable (finished runs are skipped). Variants: edit or copy a YAML config (trees, sigmas, seeds,
ratios / batch sizes / n_train); `--workers` sets the process count (one torch thread each). Cost on
30 vCPUs: step-1 sweep ~25 min, SGD sweep ~3 h, samples sweep ~45 min, P9 ~40 min.

## Results summary (details, tags and numbers in `notes/optim_samples.md`)
- Sharpness at the balanced global minimum is exactly K s_1^{2-2/K} (proven; checked to 1e-15), the
  balanced minimum is the flattest aligned one, so eta_max = 2/(K s_1^{2-2/K}).
- Small init: GD converges like GF for every eta <= 0.95 eta_max and never for eta >= 1.05 eta_max
  (binary N=4, 5 init scales x 5 seeds). Event-time shifts are predicted exactly at O(eta) by the
  backward-error variational equation; relative errors vanish as sigma -> 0.
- Above eta_max: bounded oscillation of the top component(s) (1.05-1.3), divergence, and a regime
  in which learned components are thrown back to the origin saddle and re-learned (2 eta_max).
- Imbalanced init: GD selects minima with sharpness min(lambda_GF, 2/eta) and reduced imbalance.
- Above eta_max the time-averaged excess loss equals s_1^2 x the 1D-map attractor loss (3-4 digits).
- Across the lazy transition the GD limit is 2/lambda_GF(sigma) ~ sigma^{-2(K-1)} (centred model).
- SGD: the first plateau keeps its GF law (spread ~ sqrt(eta/B)); later k = 3 plateaus are lengthened
  and trapped as in the theory note's P9 (c_gamma ~ 34-45, only eta/B matters).
- Finite samples: small init recovers W* with bond rank 2 from n ~ 256 (d = 2, N = 8); the exact lazy
  limit at chi = 4 never generalizes (tangent space too small), at full chi it needs n = d^N.
