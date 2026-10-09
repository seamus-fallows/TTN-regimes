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

## How to run
```
uv run python experiments/real_data/download.py /root/outputs/real_data/data
uv run python experiments/real_data/run.py experiments/real_data/configs/<config>.yaml /root/outputs/real_data/<config> <workers>
```
`run.py` is resumable (finished cases are skipped) and appends a line per run to `<out>/runs.jsonl`.
Config keys: see the `DEFAULTS` dict in `run.py` (task, tree, ordering, chi, features, centred, sigma
lists, seeds, t2, grid density, tolerances, hessian = gn | dense, wall_limit).

## Files
| file | role |
|---|---|
| `download.py` | data download and downsampling |
| `data.py` | tasks, features, pixel orderings, trees, `GNProblem` / `DenseHessianProblem` (fast Newton matrices), `messages`, `function_spectra` |
| `run.py` | two-phase gradient-flow runner |
| `analysis.py` | events, ranks, time-to-loss |
| `configs/` | sweep definitions (`smoke`, `pilot`, ...) |

## Results
(pending)
