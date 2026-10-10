# s2s_core: saddle-to-saddle phenomenology and escape-time scaling (Q1, Q2)

Agent `s2s-core`, brief `notes/briefs/s2s-core.md`. Findings, derivations and status tags:
`notes/s2s_core.md`. Running report: `agent-runs/s2s-core/REPORT.md` (on the agent branch).

## Question

Does gradient flow on a TTN from small initialization show saddle-to-saddle (S2S) dynamics, which
saddles does it visit, and how do plateau durations scale with the init scale sigma, the tree (K
cores) and the bond dimension chi, against theory (`notes/theory_escape.md`,
`notes/theory_saddles.md`)? Added by the lead: do dormant components get suppressed during earlier
escapes and plateaus (which would turn long plateaus into stalls)?

## Setup (common to all parts)

- Model as in `notes/model.md`: whitened Gaussian inputs, exact population loss, d = 4 at every site,
  odeco teachers (`ttn.teachers.odeco_teacher`, teacher seed 1000), weights "separated" [3, 2, 1] and
  "close" [1, 0.85, 0.7].
- Trees: binary N = 4 (K = 3), MPS N = 4 (K = 4), binary N = 8 (K = 7), HT N = 4 (K = 7), MPS N = 8
  (K = 8). Bond dimension chi = 3 (= number of components) and chi = 6, capped at the largest rank of
  each bipartition (`TTNSpec.uniform`).
- Inits: fan-in (`spec.init(sigma, seed)`, seeds 0-9); gauge-balanced (same W_0, every Q_e = 0,
  `s2s.gauge_balance`); aligned balanced (Saxe-type, `ttn.network.aligned_init`).
- Dynamics: gradient flow. Random-init runs: `ttn.dynamics.integrate_gf(method="lsoda")` (exact
  Hessian Jacobian; later plateaus are stiff), rtol 1e-8, atol 1e-11 sigma. Aligned runs: DP5
  (`method="dopri5"`), rtol 1e-10, atol 1e-14, and LSODA as a cross-check.
- Events: `a_r(t) = component_overlaps`, event = final rise through 1/2 (last upward crossing),
  width = 0.1 -> 0.9 of the final rise (`notes/diagnostics.md`), both refined by re-integrating the
  event window on linear grids until the rise holds >= 40 samples (`s2s.zoom_rise`). No duration is
  read off the log output grid. Plateau midpoints = minimum parameter speed between events; the
  gauge-projected Hessian is evaluated there (`ttn.diagnostics.project_out_gauge`).
- First-escape prediction (P1 / theory_escape E1): blow-up time T of the correlation flow
  theta' = grad <W(theta), W*> from the sigma = 1 init (`s2s.correlation_flow_blowup`), so that
  t_1 = sigma^-(K-2) T + O(1) per seed with no free parameter.

## Parts, scripts and configs

| part | script | config | output (raw) | results |
|---|---|---|---|---|
| 1. aligned init vs exact solution | `aligned.py`, `plot_aligned.py` | `configs/aligned.yaml` | `/root/outputs/s2s_core/aligned/` | `results/aligned.csv`, `fig_a_aligned.*` |
| 2-4. full S2S runs (events, plateaus, spectra, Hessians, dormant components) | `run.py` (+ `s2s.py`), `analyze_main.py`, `analyze_dormant.py`, `plot_run.py` | `configs/main.yaml` | `/root/outputs/s2s_core/main/` | `results/main_*.csv`, `fig_b_run_*` |
| 3a. first escape vs sigma, per-seed prediction | `first_escape.py`, `analyze_first.py` | `configs/first.yaml` | `/root/outputs/s2s_core/first/` | `results/first_*.csv` |
| 3b. chi, d, K dependence at leading order | `lo_sweep.py` | `configs/lo_chi.yaml`, `lo_d.yaml`, `lo_K.yaml` | - | `results/lo_*.csv` |
| dormant suppression: gauge-balanced init | `run.py` | `configs/balanced.yaml` | `/root/outputs/s2s_core/balanced/` | `results/balanced_*.csv` |

## How to run

All scripts take an output directory (`--out`, default `results/`) and a process count where they
run sweeps (`--workers`, default 60; one thread per process). Order:

```
uv run python experiments/s2s_core/aligned.py experiments/s2s_core/configs/aligned.yaml
uv run python experiments/s2s_core/plot_aligned.py
uv run python experiments/s2s_core/run.py experiments/s2s_core/configs/main.yaml --workers 60      # ~2 h on 60 CPUs
uv run python experiments/s2s_core/run.py experiments/s2s_core/configs/balanced.yaml --workers 60
uv run python experiments/s2s_core/first_escape.py experiments/s2s_core/configs/first.yaml --workers 60
uv run python experiments/s2s_core/lo_sweep.py experiments/s2s_core/configs/lo_chi.yaml
uv run python experiments/s2s_core/analyze_main.py --raw /root/outputs/s2s_core/main
uv run python experiments/s2s_core/analyze_dormant.py --raw /root/outputs/s2s_core/main /root/outputs/s2s_core/balanced
uv run python experiments/s2s_core/analyze_first.py
uv run python experiments/s2s_core/plot_run.py <run name>
```

`run.py` and `first_escape.py` are resumable (finished runs are skipped). Variants: edit or copy a
config (trees, chis, teachers, inits, sigmas, seeds, integrator tolerances, wall limit). The sweep
driver is experiment code written before `ttn.runner` landed.

## Run budget and stopping (affects only stalled runs)

t_max = 1e3 x the RMS-overlap escape time of the weakest component; a run also stops when the loss
reaches 1e-9 L(0), after a wall limit (400-600 s) or 5000-20000 integrator steps per chunk, or when
every dormant amplitude |<W, U_r>| has stayed below 1e-15 for a whole output chunk (its later escape
would be seeded by float64 roundoff). The first ~310 runs of `main.yaml` used wall 600 s and 20000
steps and no roundoff stop; this changes only when stalled runs end, not their classification.

## Raw outputs

`/root/outputs/s2s_core/<config name>/<run>.npz` (log-grid trajectory: loss, overlaps, spectra, core
norms, slice norms, per-edge imbalance; refined event windows; parameters at plateau midpoints) and
`<run>.json` (summary: events, widths, order, separation ratios, plateaus with Hessian spectra,
dormant amplitudes). Not in git; regenerate with the commands above.

## Results

Full statements with status tags and numbers: `notes/s2s_core.md` section 2; summary in
`agent-runs/s2s-core/REPORT.md`. In short (odeco teachers, d = 4, gradient flow):
- Aligned init reproduces the exact mode solutions on all five trees (max |a_GF - a_theory| <= 3e-9).
- First escape: t_1 = sigma^-(K-2) T(theta_hat) + c_K per seed with no free parameter (exponent
  0.9996, 2.0000, 4.998 for K = 3, 4, 7; prefactor to <= 2e-3 at the smallest sigma); the Kumar-Haupt
  rho_0 law is a lower bound off by 3-500x (K = 3-7) and up to 3e7 at K = 15; at full bond the binary4
  prefactor is pi/2 independent of d; the RMS-overlap heuristic P3 tracks the K dependence within 2x.
- Random init: S2S with plateaus at the partial-fit levels, one rank step per event on every edge at
  once, every gap ~ sigma^-(K-2) per seed (1.00, 2.00, 5.0 for K = 3, 4, 7), plateaus are higher-order
  saddles (lambda_min ~ sigma^(K-2), magnitude within 0.8-1.7 of the route formula).
- Dormant suppression: escapes barely shrink dormant components; plateaus do, through the conservation-
  law trap of fan-in init (dead route ends, decay rate ~ sigma^2): never at K = 3, a sigma-independent
  4/10 seeds at K = 4, chi = 3, most HT4 runs at chi = 3; gauge-balanced init removes the trap (0/210
  trapped); chi = 2R avoids it at K = 3, 4, 7 (binary8, HT4). Runs that stall with live routes are on
  ever-longer plateaus (continued runs escape at 0.5-1.8x the route estimate in 10 of 11 cases).

## Figures (results/)

| figure | file |
|---|---|
| (a) aligned init vs exact solution | `fig_a_aligned.*` |
| (b) representative runs | `fig_b_run_mps4_chi3_..._s0.01_seed0.*` (K = 4), `fig_b_run_binary8_chi6_..._s0.1_seed0.*` (K = 7) |
| (c) t_1 vs sigma with per-seed prediction | `fig_c_first_escape_chi{3,6}_{fanin,balanced}.*` |
| (d) gaps vs sigma | `fig_d_gaps_chi{3,6}_{separated,close}.*` |
| (e) prefactor vs chi and vs K (leading order) | `fig_e_chi.*`, `fig_e2_K.*` |
| (f) Hessian lambda_min vs sigma at plateau midpoints | `fig_f_hessian.*` |
| (g) dormant suppression: run classes, trap decay rate | `fig_g_dormant.*` |
