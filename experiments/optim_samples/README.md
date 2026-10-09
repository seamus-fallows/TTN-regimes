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
`ttn.dynamics` landed; GF references use `scipy` DOP853 (rtol 1e-10, atol 1e-14); the chi = 9 runs use LSODA
(`ttn.dynamics.integrate_gf`), stopped at loss 1e-8.

## How to run

From the repository root, with config paths written out in full, for example

    uv run python experiments/optim_samples/gd_sweep.py experiments/optim_samples/configs/gd_sweep.yaml --workers 8
    uv run python experiments/optim_samples/analyze_gd.py --out-dir experiments/optim_samples/results

Every script takes `--raw-dir DIR` (raw per-run data; default `/root/outputs/optim_samples`, or the
environment variable `OPTIM_SAMPLES_RAW`) and `--out-dir DIR` (summary CSVs and figures; default
`<raw-dir>/results` or `OPTIM_SAMPLES_OUT`). The default output directory is outside git, so a re-run
never overwrites the committed results; pass `--out-dir experiments/optim_samples/results` to
regenerate them. Analysis scripts read the CSVs of earlier scripts from the same `--out-dir`. Sweeps take
`--workers N` (default: min(30, number of CPUs); one torch thread per process) and are resumable
(finished runs are skipped; move a raw directory aside to rerun it). On machines without `/root`, pass
`--raw-dir` / `--out-dir` (or set the variables). `PYTHONDONTWRITEBYTECODE=1` keeps `__pycache__` out
of the folder.

## Scripts
| script | what | output |
|---|---|---|
| `check_sharpness.py` | top Hessian eigenvalue at balanced / gauged / imbalanced minima vs K s_1^{2-2/K} | `sharpness_check.csv` |
| `map1d.py` | bifurcation diagram of GD on one balanced mode | `map1d_attractor.csv`, `map1d_points.csv` |
| `sgd_stability.py [tree N ...]` | exact linear mean-square stability limit of online SGD vs B | `sgd_stability.csv` |
| `gd_sweep.py <config>` | GF references + GD runs; configs: `gd_sweep.yaml` (step 1, 750 runs), `gd_order.yaml` (40 more seeds), `gd_centred.yaml` (round 1, chi = 3 centred; superseded), `gd_centred_chi9.yaml` (fix round 1: chi = 9 centred + small init, steps relative to 2/lambda_GF) | raw: `<raw-dir>/gd/` |
| `predict_shift.py` | first-order (backward-error) prediction of GD event shifts (step-1 references) | `gd_shift_prediction.csv` |
| `analyze_gd.py` | per-run table, outcome classes (GF-like, catapult, oscillating, thrown back, other, diverged), EoS example export | `gd_sweep.csv`, `gd_summary.csv`, `eos_example_*.csv` |
| `analyze_shift.py` | accuracy of the O(eta) shift per event and seed; Richardson check | `shift_accuracy*.csv`, `shift_richardson.csv`, `fig_shift_accuracy.png` |
| `analyze_order.py` | persistent order vs GF, transient crossings, near-tie measure | `order_flips.csv`, `fig_order_flips.png` |
| `analyze_p5.py` | P5: oscillation excess loss vs the 1D map | `p5_excess_loss.csv` |
| `analyze_centred.py` | chi = 9: lambda_GF(sigma) fit, GD classes vs eta/(2/lambda_GF) | `centred_chi9_*.csv`, `fig_centred_chi9.png` |
| `plot_gd.py` | step-1 figures | `fig_gd_vs_gf.png`, `fig_stability.png`, `fig_eos_example.png`, `fig_min_selection.png` |
| `plot_regime.py` | learning-rate axis of the regime map | `fig_lr_regime_map.png` |
| `sgd_sweep.py experiments/optim_samples/configs/sgd_sweep.yaml` | step 2: online SGD, B = 1..256, eta sweeps, aligned init; runs continue past the second GF event | raw: `<raw-dir>/sgd/*.jsonl` |
| `analyze_sgd.py`, `plot_sgd.py` | step-2 tables (per init and pooled, s.e. across inits; second plateau vs P9) and figures | `sgd_*.csv`, `fig_sgd.png`, `fig_sgd_stability.png` |
| `p9_sgd.py`, `plot_p9.py` | P9/P10 test (aligned-route start, independent noise seeds; `ttn.dynamics.run_sgd`), law-shape fits | `p9_sgd.csv`, `p9_conditions.csv`, `p9_fit.csv`, `fig_p9.png` |
| `collapse_test.py` | stochastic collapse of a k = 2 route with label noise, MPS N=4 | `collapse_test.csv` |
| `check_k2_saddle.py` | raw and gauge-projected Hessian lambda_min at the k = 2 plateau vs sigma | `k2_saddle_sweep.csv`, `k2_saddle_hessian.json` |
| `sgd_limit_test.py` | empirical SGD stability limit at the balanced minimum (B = 1, 16) | `sgd_limit_test.csv` |
| `samples_sweep.py experiments/optim_samples/configs/samples_sweep.yaml` | step 3: fixed training sets, small vs centred init, exact lazy limit | raw: `<raw-dir>/samples/*.jsonl` |
| `analyze_samples.py` | step-3 tables and figure | `samples_summary.csv`, `fig_samples.png` |
| `validate_library.py` | reruns sample GD/GF/SGD settings with `ttn.dynamics` and compares (round 1) | `validate_library.csv` |

Helpers: `common.py` (setup, Problem with centred variant, GF reference via scipy DOP853 with exact
event location, persistent events `record_events`), `plot_style.py`. Order to regenerate everything:
`check_sharpness.py`, `map1d.py`, `sgd_stability.py`, `gd_sweep.py` with `gd_sweep.yaml`,
`gd_order.yaml`, `gd_centred_chi9.yaml`, then `predict_shift.py`, `analyze_gd.py`, `analyze_shift.py`,
`analyze_order.py`, `analyze_p5.py`, `analyze_centred.py`, `plot_gd.py`, `plot_regime.py`; then
`sgd_sweep.py` (needs the GF references), `analyze_sgd.py`, `plot_sgd.py`, `p9_sgd.py`, `plot_p9.py`,
`collapse_test.py`, `check_k2_saddle.py`, `sgd_limit_test.py`, `samples_sweep.py`,
`analyze_samples.py`. Cost on 30 vCPUs (shared): GD sweeps ~40 min, chi = 9 centred ~1.5 h, SGD sweep
~3 h, samples ~1 h, P9 ~40 min.

## Results summary (details, tags and numbers in `notes/optim_samples.md`; fix round 1)
- Sharpness at the balanced global minimum is exactly K s_1^{2-2/K} (proven; checked to 1e-15), the
  balanced minimum is the flattest aligned one, so eta_max = 2/(K s_1^{2-2/K}).
- Small init: GD below eta_max keeps the GF saddle sequence in every checked case (persistent
  events: 765/765 runs); it converges like GF for every eta <= 0.95 eta_max and never for
  eta >= 1.05 eta_max (binary N=4, 5 init scales x 5 seeds). Event-time shifts are eta c_e + O(eta^2)
  with c_e from the backward-error variational equation (Richardson limit agrees to ~1e-5); measured
  shifts agree with c_e within 0.3% / 1% / 5% in 66% / 90% / 97% of 526 comparisons at
  eta <= 0.1 eta_max, worse where c_e ~ 0.
- Above eta_max: bounded oscillation of the top component(s) (1.05-1.3), divergence, and a regime
  in which learned components are thrown back to the origin saddle and re-learned (2 eta_max). The
  time-averaged excess loss equals s_1^2 x the 1D-map attractor loss (within 0.9%, binary N=4).
- Imbalanced init: GD selects minima with sharpness min(lambda_GF, 2/eta) and reduced imbalance.
- Across init scales (centred model, chi = 9, converged GF): 2/lambda_GF(sigma) (= eta_max at small
  init, ~ sigma^{3.97} beyond sigma ~ 1.4) is the end of monotone GF-like GD; above it, small init
  oscillates then diverges, larger init catapults to GF's loss at a flatter minimum up to
  ~1.5-3 x 2/lambda_GF.
- SGD: the first escape keeps its GF law (spread ~ (eta/B)^{0.43-0.52}, no shortening); later k = 3
  plateaus are lengthened and trapped; the P9 shape fits better than pole/linear laws (c_gamma ~ 33),
  with residual curvature near the trap; only eta/B matters (independent noise at B = 16 and 64);
  learned components are erased at >= ~0.25 eta*_ms on long plateaus.
- Finite samples (one teacher, capped GD steps): small init recovers W* with bond rank 2 from
  n ~ 256 (d = 2, N = 8); the exact lazy limit fails as dimension counting predicts (chi = 4: tangent
  space of dimension 112 < 256; full chi: needs n = d^N).
