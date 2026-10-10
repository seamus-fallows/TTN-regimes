# Experiment `real_data` (backlog B5): teacher-student TTN phenomena on real images

**Question.** Do the phenomena established with teachers (first plateau ~ sigma^-(K-2), stepwise
bond-rank growth, the route/locality rule, the regime map) appear when a TTN is trained by gradient
flow on a real dataset? Findings: `notes/real_data.md`. Running report: `agent-runs/real-data/REPORT.md`.

## Setup
- Data: MNIST 3 (y = +1) vs 5 (y = -1), balanced, n_train = 4000, n_test = 1600 (data seed 0), images
  cropped to 24 x 24 and block-averaged to 4 x 4 (N = 16 sites). `download.py` fetches the raw IDX
  files (MNIST from the torchvision mirror, Fashion-MNIST from zalandoresearch) into
  `/root/outputs/real_data/data/` and writes 4x4 and 8x8 versions.
- Features phi(x) = (cos(pi x/2), sin(pi x/2)) ("raw"), or per-site whitened with the uncentred second
  moment ("white"). Model: f = <W, (x)_i phi_i>, empirical square loss, fan-in init with per-core
  sigma (`TTNSpec.init`), float64, one thread per process.
- Trees (`data.pixel_order`, `data.make_tree`): `binary16` (K = 15) on the Morton/quadtree order
  ("quadtree"), `mps16` (K = 16) on the boustrophedon order ("snake"), `binary16` on a random
  permutation ("scrambled<k>", permutation seed k + 12345). Bond dimension chi (default 4).
- Dynamics: gradient flow, two phases (`run.py`): DP5 through the first plateau until the loss has
  dropped by 1%, then LSODA (Gauss-Newton Newton matrix) on the clock tau = t - t_a for a horizon T2;
  rtol 1e-8, atol 1e-14. Library `Recorder` (log + fine rows) plus test metrics and data-weighted
  bond spectra (`data.function_spectra`).
- Analysis definitions: `analysis.py` docstring (events on data-weighted spectra referenced to the
  run's final values, f = 0.05; merged log + fine rows).

## How to run (in this order)
```
uv run python experiments/real_data/download.py /root/outputs/real_data/data
# sweeps (resumable; one thread per process; ~60 processes saturate the 64-core memory bandwidth)
uv run python experiments/real_data/run.py experiments/real_data/configs/main.yaml   /root/outputs/real_data/main 48
uv run python experiments/real_data/run.py experiments/real_data/configs/regime.yaml /root/outputs/real_data/regime 12
uv run python experiments/real_data/run.py experiments/real_data/configs/chi8.yaml    /root/outputs/real_data/chi8 24
uv run python experiments/real_data/run.py experiments/real_data/configs/fashion.yaml /root/outputs/real_data/fashion 24
uv run python experiments/real_data/run.py experiments/real_data/configs/img8.yaml    /root/outputs/real_data/img8 12
# analysis (A = /root/outputs/real_data/analysis, R = experiments/real_data/results or a fresh directory)
uv run python experiments/real_data/summarize.py /root/outputs/real_data/main $A/main_summary 8
uv run python experiments/real_data/summarize.py /root/outputs/real_data/regime $A/regime_summary 8
uv run python experiments/real_data/route_amplitudes.py $A/main_summary.json /root/outputs/real_data/main $A/routes_raw.json 6 _raw_binary
uv run python experiments/real_data/plateau_hessians.py $A/main_summary.json /root/outputs/real_data/main $A/hessians_raw.json 6 _raw_binary_quadtree,_raw_mps
uv run python experiments/real_data/regime_analysis.py $A/regime_summary.json /root/outputs/real_data/regime $A/regime_analysis.json 8 $A/main_summary.json /root/outputs/real_data/main 0.3
uv run python experiments/real_data/scaling.py $A/main_summary.json $R/scaling.json
uv run python experiments/real_data/tables.py $A/main_summary.json $R/tables $A/routes_raw.json
uv run python experiments/real_data/events_table.py $A/main_summary.json $R/events_table
uv run python experiments/real_data/make_figures.py $A/main_summary.json /root/outputs/real_data/main $R raw $A/routes_raw.json $A/hessians_raw.json $A/regime_analysis.json
```
`run.py <config> <out_dir> <workers>` skips finished cases and appends a line per run to `<out_dir>/runs.jsonl`.
Config keys: `DEFAULTS` in `run.py` (dataset, size, pos/neg classes, n_train, tree, ordering, chi, features,
centred, sigmas, seeds, t2, t2_units = abs | kernel, grid density, tolerances, hessian = gn | dense,
kernel_probe, wall_limit). Scripts never write into the run directories; the analysis output paths are
arguments.

## Files
| file | role |
|---|---|
| `download.py` | data download and downsampling |
| `data.py` | tasks, features, pixel orderings, trees, `GNProblem` (closed-form per-sample Jacobian, Gauss-Newton Newton matrix), `DenseHessianProblem` (exact Hessian, N <= 16), `messages`, `function_spectra` |
| `run.py` | two-phase gradient-flow runner (re-anchored clock in phase 1, LSODA phase 2), `RealRecorder` |
| `analysis.py` | loading, rate-peak loss drops, events, time-to-loss |
| `summarize.py` | one summary row per run (drops, rho, S2S, rank events, edges per drop, loss levels) |
| `scaling.py` | per-seed fits: t1, classified first-route gaps, per-edge times; power vs log laws |
| `events_table.py` | per-event route degree vs gap exponent |
| `route_amplitudes.py` | route amplitudes lambda_uw at the rank-1 plateau (binary trees) |
| `plateau_hessians.py` | exact Hessian at the first plateau, raw and gauge-projected |
| `regime_analysis.py` | regime labels, post-hoc delta_K, linearized-model reference L_lin(t) |
| `tables.py`, `make_figures.py` | tables and figures |
| `configs/` | `main`, `regime`, `chi8`, `fashion`, `img8`; `pilot`, `smoke*` (development) |

## Results (details and evidence: `notes/real_data.md`)
- First plateau t1 ∝ sigma^-(K-2) exactly (13.000 binary16, 14.000 mps16).
- First route after the escape: a sibling pair of bottom cores on binary trees (k = 3), gap ∝ sigma^-1
  (1.01 ± 0.14 quadtree); a single edge on the MPS (k = 2), gap ∝ log(1/sigma); Hessians: MPS strict,
  binary higher order (lambda_min ∝ sigma^0.88).
- Route amplitudes at the rank-1 plateau predict the first route in 34/35 (quadtree) and 78/89 (scrambled).
- Locality: quadtree >> scrambled at small sigma (88% vs 21% of the residual interaction on k <= 5 routes);
  the MPS is fastest at every sigma (2-core routes everywhere); no locality advantage at sigma = 0.5.
- Regimes (centred): S2S (sigma <= 0.3) -> rich (1.4) -> lazy (>= 2), lazy = linearized model exactly.
