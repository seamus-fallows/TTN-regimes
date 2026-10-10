# Experiment `real_data` (backlog B5): teacher-student TTN phenomena on real images

**Question.** Do the phenomena established with teachers (first plateau ~ sigma^-(K-2), stepwise
bond-rank growth, the route/locality rule, the regime map) appear when a TTN is trained by gradient
flow on a real dataset? Findings: `notes/real_data.md` (section 8 = fix round 1, which supersedes
earlier statements where they differ). Running report: `agent-runs/real-data/REPORT.md`.

## Setup
- Data: MNIST 3 (y = +1) vs 5 (y = -1), balanced, n_train = 4000, n_test = 1600 (data seed 0), images
  cropped to 24 x 24 and block-averaged to 4 x 4 (N = 16 sites). `download.py` fetches the raw IDX
  files (MNIST from the torchvision mirror, Fashion-MNIST from zalandoresearch) and writes 4x4 and 8x8
  versions. The data directory is `/root/outputs/real_data/data` unless the environment variable
  `REAL_DATA_DIR` points elsewhere (read by `data.py`).
- Features phi(x) = (cos(pi x/2), sin(pi x/2)) ("raw"), or per-site whitened with the uncentred second
  moment ("white"). Model: f = <W, (x)_i phi_i>, empirical square loss, fan-in init with per-core
  sigma (`TTNSpec.init`), float64, one thread per process.
- Trees (`data.pixel_order`, `data.make_tree`): `binary16` (K = 15) on the Morton/quadtree order
  ("quadtree"), `mps16` (K = 16) on the boustrophedon order ("snake"), `binary16` on a random
  permutation ("scrambled<k>", permutation seed k + 12345), row-major ("rowmajor"). Bond dimension chi (default 4).
- Dynamics: gradient flow, two phases (`run.py`): DP5 through the first plateau until the loss has
  dropped by 1%, then LSODA (Gauss-Newton Newton matrix) on the clock tau = t - t_a for a horizon T2;
  rtol 1e-8, atol 1e-14. Library `Recorder` (log + fine rows) plus test metrics and data-weighted
  bond spectra (`data.function_spectra`).
- Analysis definitions (`notes/real_data.md` section 3): learning events = peaks of the loss rate -dL/dt
  (FWHM widths; separation ratios as in `notes/diagnostics.md`); which edges gain rank = data-weighted
  s_k/s_1 crossing 0.05; merged log + fine rows. A route's drop is the drop whose widened window contains
  its rank crossing (fix round 1); power law, log law and A/sigma + B are all fitted with relative residuals.

## How to run
Raw outputs go to `/root/outputs/real_data/<sweep>/` (remote) or any directory given as argument; they are
not in git. `D = /root/outputs/real_data`, `A = $D/analysis`, `R = experiments/real_data/results`.
```
uv run python experiments/real_data/download.py $D/data          # or set REAL_DATA_DIR to an existing copy
# sweeps (resumable; one thread per process; ~60 processes saturate a 64-core machine's memory bandwidth)
uv run python experiments/real_data/run.py experiments/real_data/configs/main.yaml       $D/main 48
uv run python experiments/real_data/run.py experiments/real_data/configs/main_extra.yaml $D/main 14  # 14 scrambled1 runs kept from before the grid trim
uv run python experiments/real_data/run.py experiments/real_data/configs/regime.yaml     $D/regime 12
uv run python experiments/real_data/run.py experiments/real_data/configs/fashion.yaml    $D/fashion 24
uv run python experiments/real_data/run.py experiments/real_data/configs/chi8.yaml       $D/chi8 24
uv run python experiments/real_data/run.py experiments/real_data/configs/chi8b.yaml      $D/chi8b 21  # second half of chi8 (same cases), run in parallel
uv run python experiments/real_data/run.py experiments/real_data/configs/img8.yaml       $D/img8 12
uv run python experiments/real_data/run.py experiments/real_data/configs/tol_check.yaml  $D/tol_check 2  # 100x tighter tolerances
# analysis of the main and regime sweeps
uv run python experiments/real_data/summarize.py $D/main $A/main_summary 8
uv run python experiments/real_data/summarize.py $D/regime $A/regime_summary 8
uv run python experiments/real_data/route_amplitudes.py $A/main_summary.json $D/main $A/routes_raw.json 6 _raw_binary
uv run python experiments/real_data/plateau_hessians.py $A/main_summary.json $D/main $A/hessians_raw.json 6 _raw_binary_quadtree,_raw_mps
uv run python experiments/real_data/regime_analysis.py $A/regime_summary.json $D/regime $A/regime_analysis.json 8 $A/main_summary.json $D/main 0.3
uv run python experiments/real_data/scaling.py $A/main_summary.json $R/scaling.json
uv run python experiments/real_data/tables.py $A/main_summary.json $R/tables $A/routes_raw.json
uv run python experiments/real_data/events_table.py $A/main_summary.json $R/events_table
uv run python experiments/real_data/make_figures.py $A/main_summary.json $D/main $R raw $A/routes_raw.json $A/hessians_raw.json $A/regime_analysis.json
# robustness sweeps (chi8_runs = one directory with links to the runs of chi8/ and chi8b/)
uv run python experiments/real_data/summarize.py $D/fashion $A/fashion_summary 8
uv run python experiments/real_data/tables.py $A/fashion_summary.json $R/fashion_tables
uv run python experiments/real_data/scaling.py $A/fashion_summary.json $R/fashion_scaling.json
uv run python experiments/real_data/plateau_hessians.py $A/fashion_summary.json $D/fashion $R/fashion_hessians.json 6 _raw_binary_quadtree,_raw_binary_scrambled0,_raw_mps
uv run python experiments/real_data/make_figures.py $A/fashion_summary.json $D/fashion $R/fashion raw
uv run python experiments/real_data/summarize.py $D/chi8_runs $A/chi8_summary 8
uv run python experiments/real_data/tables.py $A/chi8_summary.json $R/chi8_tables
uv run python experiments/real_data/scaling.py $A/chi8_summary.json $R/chi8_scaling.json
uv run python experiments/real_data/summarize.py $D/img8 $A/img8_summary 8
uv run python experiments/real_data/tables.py $A/img8_summary.json $R/img8_tables
```
The robustness commands are reconstructed from the scripts' interfaces (the remote shell history is not
kept); the committed `fashion_*`, `chi8_*`, `img8_*` files are their outputs. `run.py <config> <out_dir>
<workers>` skips finished cases and appends a line per run to `<out_dir>/runs.jsonl`. Config keys: `DEFAULTS`
in `run.py` (dataset, size, pos/neg classes, n_train, tree, ordering, chi, features, centred, sigmas, seeds,
t2, t2_units = abs | kernel, grid density, tolerances, hessian = gn | dense, kernel_probe, wall_limit, tag).
Scripts never write into the run directories; the analysis output paths are arguments.

### Fix round 1 (local, 2026-10-10)
The raw outputs above were not available (remote machine stopped). The fix round re-analyses the committed
outputs and reruns two light subsets of `main.yaml` locally (`L = .../TTN_regimes/data/real_data_local`):
```
export REAL_DATA_DIR=<directory with mnist_4x4.npz>          # e.g. data/review_real_data/data
uv run python experiments/real_data/run.py experiments/real_data/configs/fix1_local.yaml $L/fix1 12  # 20 binary (sigma = 0.1) + 5 MPS (phase 1)
uv run python experiments/real_data/run.py experiments/real_data/configs/fix1_mps.yaml   $L/fix1_mps 12  # 35 MPS, tau <= 1000
uv run python experiments/real_data/summarize.py $L/fix1 $L/fix1_summary 8
uv run python experiments/real_data/summarize.py $L/fix1_mps $L/fix1_mps_summary 8
uv run python experiments/real_data/scaling.py $L/fix1_mps_summary.json $R/fix1_mps_scaling.json   # MPS per-edge fits (fair)
uv run python experiments/real_data/events_table.py --refit <round-0 events_table.json> $R/events_table  # fair refit
uv run python experiments/real_data/fix1_analysis.py $R $R/fix1_analysis --local $L/fix1_summary.json $L/fix1_mps_summary.json \
    --audit <review>/audit/audit_table.json <review>/audit/b2/audit_table.json --logs agent-runs/real-data/logs
uv run python experiments/real_data/make_figures.py --json $R $R      # figures from committed JSON only
```
`<review>` = `data/review_real_data` (the round-1 reviewers' runs with this code: 6 more random orders, row-major
and transposed-quadtree binary trees, MPS with other pixel orders; seed 0 and 1, sigma = 0.1).
**Pending (needs the remote raw outputs):** rerun `summarize.py`, `scaling.py`, `tables.py`, `events_table.py`
and `make_figures.py` on the full sweeps with the fixed code (first-route attribution, fair fits, unresolved
rank events, parent check, per-unit route counts), which regenerates `scaling.json`, `tables.*`,
`scaling_raw.*` (removed in fix round 1: its middle panel plotted misattributed gaps; `first_route_gaps` and
`scaling_fix1` replace it) and the `chi8_*`/`fashion_*`/`img8_*` tables; until then those files are round-0 outputs and
`notes/real_data.md` section 8 lists which of their numbers are superseded.

## Files
| file | role |
|---|---|
| `download.py` | data download and downsampling |
| `data.py` | tasks, features, pixel orderings, trees, `GNProblem` (closed-form per-sample Jacobian, Gauss-Newton Newton matrix), `DenseHessianProblem` (exact Hessian, N <= 16), `messages`, `function_spectra`; `REAL_DATA_DIR` |
| `run.py` | two-phase gradient-flow runner (re-anchored clock in phase 1, LSODA phase 2), `RealRecorder` |
| `analysis.py` | loading, rate-peak loss drops, events, time-to-loss; `route_units` (rank-1 filter, critical-point units) |
| `summarize.py` | one summary row per run (drops, rho, S2S, rank events incl. those already met at tau = 0, edges per drop, loss levels) |
| `scaling.py` | per-seed fits: t1, classified first-route gaps (window-checked attribution), per-edge times; power law, log law, A/sigma + B with relative residuals |
| `events_table.py` | per-event route degree vs gap exponent; `--refit` for an existing table |
| `route_amplitudes.py` | route amplitudes lambda_uw at the rank-1 plateau (binary trees) |
| `plateau_hessians.py` | exact Hessian at the first plateau, raw and gauge-projected |
| `regime_analysis.py` | regime labels, post-hoc delta_K, linearized-model reference L_lin(t) |
| `tables.py`, `make_figures.py` | tables and figures (`make_figures.py --json`: the figures that need only committed JSON) |
| `fix1_analysis.py` | fix round 1: re-analysis from committed outputs (+ local reruns, review runs) -> `results/fix1_analysis.{json,md}` |
| `configs/` | `main` (+ `main_extra`: 14 scrambled1 runs main.yaml does not generate), `regime`, `fashion`, `chi8` + `chi8b` (same cases, two runners), `img8`, `tol_check`; `fix1_local`, `fix1_mps` (fix round 1); `pilot`, `smoke*` (development) |
| `results/` | small outputs: tables, JSON summaries, figures (`fix1_*` = fix round 1; see the pending note above for round-0 files) |

## Results (details and evidence: `notes/real_data.md`; section 8 = fix round 1)
- First plateau t1 ∝ sigma^-(K-2) (13.000 binary16, 14.000 mps16): an identity of small-init flow [proven],
  an integrator check rather than evidence.
- First route after the escape on binary trees: a sibling pair of bottom cores (k = 3) in all 20 (order, seed)
  pairs; its gap follows A/sigma + B (all 11 clean series, B < 0; free exponent 1.22 ± 0.03, not robust); the
  log law is rejected with the same relative-residual weighting (`first_route_gaps`). The round-0 "1.01 ± 0.14"
  came from an attribution bug.
- MPS: the first rank increase (edge 5) is merged with the escape (10/35 runs already above threshold at
  tau = 0); later MPS routes (edges 4, 3) rise in log time; the plateau after the escape is a strict saddle,
  the binary rank-1 plateau a higher-order one (lambda_min ∝ sigma^0.95 ± 0.07) (`scaling_fix1`, `plateau_hessians`).
- Route amplitudes at the rank-1 plateau predict the first sibling pair in 22 of 26 (order, seed, critical
  point) units (2 mixed, 2 miss; chance 1/4) and 11/11 review runs (`route_amplitudes`).
- Pixel order: at sigma = 0.1 the quadtree order ends lowest at every seed (0.098 ± 0.001 vs 0.19-0.23 for
  random orders), the k <= 5 share of route amplitude predicts the final loss (Spearman -0.72, n = 30), but an
  equally local transposed quadtree is among the worst and early learning is not faster: an order/orientation
  effect, not locality as such (`locality_orders`).
- MPS vs binary: the snake MPS is fastest only on the tau = t - t_a clock on MNIST; on the absolute clock its
  first escape is later for 4 of 5 seeds at sigma = 0.1, and on Fashion it is the slowest tree.
- Regimes (centred): S2S at small sigma, rich at sigma = 1.4 (one sigma), lazy from 2 with L(T) = L_lin(T);
  boundaries depend on the thresholds; the lazy runs are not converged at their horizon (`regime_sweep`).
- Robustness: Fashion-MNIST (rank-1 strict saddle, no order advantage), chi = 8 (23 runs wall-limited; locality
  direction as at chi = 4 at sigma <= 0.14), 8x8 images (one permutation, two seeds, tau = 3000); tolerance check
  passed; local reruns reproduce t_a to 1e-14 and L(1e4) to 5 digits.
