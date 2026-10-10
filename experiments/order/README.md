# Experiment `order` (Q4): which teacher component is learned first, and how random is the order?

Agent `order`, branch `agent/order`. Theory and findings: `notes/order.md`. Running report:
`agent-runs/order/REPORT.md`.

## Question

From small random (fan-in) initialization, gradient flow on a TTN learns the components of an
odeco teacher one at a time. Is the order greedy (largest weight first, as in deep linear networks)
or decided by the initialization, and on what does it depend (weight ratio rho = s_2/s_1, number of
cores K, tree, bond dimension chi, physical dimension d, init scale sigma)? Is a wrong order
corrected through a strict saddle (H1b)? With R >= 3 components, what is the distribution of orders?

## Setup (common to all runs)

- Model `notes/model.md`: whitened inputs, exact population loss, fan-in Gaussian init
  `spec.init(sigma, seed)`, float64.
- Teachers: odeco (`odeco_teacher`) or pattern (`pattern_teacher`) CP teachers in the **standard
  site basis** (no teacher randomness; equivalent in law to Haar bases because the init is
  rotation invariant on every physical leg). Weights (1, rho) or (1, rho, rho^2, ...).
- "Learned first" = earliest first crossing of a_r = 1/2 (`notes/diagnostics.md`).
- Seeds 0..n-1 are init seeds; the same seeds are reused across rho and sigma (common random
  numbers), so per-seed comparisons are paired.
- Dynamics:
  - first-event sweeps (`sweep.py`): gradient flow integrated with scipy `solve_ivp` RK45
    (Dormand-Prince 5(4)), rtol 1e-9, atol 1e-14, stopped by event location at the first crossing;
  - leading-order (LO) flow d theta/dt = grad <W, W*> from the unit-scale init, to |theta| = 1e4
    |theta_0| (blow-up time extrapolated radially): `common.lo_blowup`;
  - full trajectories (`h1b.py`): LSODA with the exact Hessian as Jacobian, rtol 1e-10, atol 1e-15,
    dense output (event times interpolated);
  - R >= 3 (`multi.py` -> `common.gf_events`): LSODA with the exact Hessian (as
    `ttn.dynamics.integrate_gf(method="lsoda")`), rtol 1e-9, atol 1e-14, exact event location for every
    crossing, stop at L < 1e-8 L(0), at t_max, or when a component's slice of some core falls to 1e-14 x
    the largest core norm ("stalled"; `stall_stop: false` disables it). (The round-1 `multi` run used
    the library integrator on a log grid; it was replaced by `multi2`, then by `multi3` with the
    slice-norm stall criterion.) Full trajectories (`h1b.py` -> `gf_trajectory`) use the same stall
    criterion; the round-1 overlap criterion (|a_r| < 1e-25) also fired during a normal forget phase.
    1500 log-spaced records.

## Files

| file | what |
|---|---|
| `common.py` | teachers, first-event GF, LO blow-up, init statistics, full trajectories, projected Hessian (library `ttn.diagnostics`) |
| `sweep.py` | config-driven order-statistics sweep (parallel, resumable JSON lines) |
| `h1b.py` | full runs through the wrong-order plateau: event times, Hessian at the plateau midpoint |
| `multi.py` | R >= 3 components, full order |
| `balance_check.py` | per-core balance of the LO escape direction |
| `repair.py` | re-integrates first-event runs that failed on the float64 resolution of t (clock restart) |
| `rebuild_h1b_csv.py` | rebuilds an h1b results CSV from the stored trajectories if a run was stopped early |
| `analyze_dtest.py` | independent-seed d test (d_test_* configs) |
| `fill_tables.py` | writes the generated R = 3 / no-stop tables into notes/order.md from results/*.csv |
| `analyze_main.py`, `analyze_sigma.py`, `analyze_rule.py`, `analyze_h1b.py`, `analyze_multi.py`, `analyze_chi.py` | summaries and figures from saved results |
| `style.py` | plot style (fixed categorical colour order) |
| `run_queue.sh` | runs a list of configs in sequence |
| `configs/` | YAML configs (below) |
| `results/` | summary CSVs and figures |

## How to run

```bash
uv run python experiments/order/sweep.py experiments/order/configs/main.yaml      # step 1
uv run python experiments/order/analyze_main.py main
uv run python experiments/order/sweep.py experiments/order/configs/b4_sigma.yaml  # sigma scans
uv run python experiments/order/analyze_sigma.py b4_sigma m4_sigma b8_sigma pat_mps6_m2end ...
uv run python experiments/order/h1b.py experiments/order/configs/h1b_saddle_b4.yaml
uv run python experiments/order/analyze_h1b.py
uv run python experiments/order/multi.py experiments/order/configs/multi3.yaml   # R = 3
uv run python experiments/order/multi.py experiments/order/configs/multi.yaml    # R = 3, 4 (raw: multi2)
uv run python experiments/order/multi.py experiments/order/configs/multi_nostop.yaml
uv run python experiments/order/analyze_multi.py multi3      # or multi2 / multi_nostop
# or several in sequence: experiments/order/run_queue.sh [--raw DIR] [--res DIR] b4_sigma m4_sigma ...
```

Every script uses the CPUs of the container by default (cgroup quota; `--procs` or `PROCS=` to
change; one torch thread and one BLAS thread per
process). Raw outputs go to `/root/outputs/order/<config name>/` (JSON lines, `.npz` trajectories);
they are not in git and are regenerated by the commands above. `h1b_natural_*` configs read the
wrong-order seeds from the `b4_sigma` / `main` raw outputs, so run those first; after `main`, run
`repair.py main` (MPS8 runs that hit the float64 time resolution). Order: main, repair, b4_sigma,
m4_sigma, b8_sigma, h1b_* (saddle ones any time), pat_*, chi_scan, chi_gf, multi; then the analyze_*
scripts. Every script accepts `--raw DIR` (raw outputs, default /root/outputs/order) and `--res DIR`
(summaries/figures, default experiments/order/results) so re-runs need not overwrite committed results.

## Configs

| config | what | cost (92 CPUs) |
|---|---|---|
| `main.yaml` | step 1: 5 trees x d {2,4} x chi {2,4,8} (28 distinct specs) x 6 rho x 400 seeds, sigma 0.01; plus LO blow-up times and init statistics per seed | ~2 h |
| `b4_sigma.yaml` | binary4 (m = 2), chi {2,4}, 5 rho x 7 sigma (1e-4..0.1) x 1000 seeds | ~30 min |
| `m4_sigma.yaml`, `b8_sigma.yaml` | m >= 3 sigma-independence, 400 seeds | ~15 min |
| `pat_mps6_*.yaml` | control: MPS6 pattern teachers differing on m = 2 (ends / adjacent), 3, 4 cores; 3 rho x 4 sigma x 400 seeds | ~15 min each |
| `pat_mps4_adj.yaml`, `pat_mps4_sep.yaml` | multiplicity test: MPS4 m = 2 pairs, adjacent (n = chi_e = 4) vs separated (n = 1), sigma down to 1e-5, 1000 seeds | ~20 min each |
| `symmetry_check.yaml` | exchangeability check of the two components' blow-up times on fresh seeds | ~5 min |
| `chi_scan.yaml`, `chi_gf.yaml` | over-parameterization: uncapped chi 1..32, LO race (500 seeds) and GF check (chi 16, 32) | ~30 min |
| `h1b_*.yaml` | H1b: saddle starts (binary4, MPS4) and natural wrong-order runs | ~20 min |
| `multi.yaml` | step 4: R = 3, 4 components, 200 seeds, sigma per tree (0.05-0.25); raw folder `multi2` (overlap-based stall stop, round 1) | ~2.5 h |
| `multi3.yaml` | R = 3 rerun with the slice-norm stall criterion and slice flags (review round 1) | ~1 h |
| `multi_nostop.yaml` | R = 4 = chi on MPS6 and binary8 without any stall stop, to t = 1e9, 12 seeds | ~1 h |
| `h1b_saddle_mps4_adj.yaml`, `h1b_saddle_mps4_sep.yaml`, `h1b_natural_mps4_adj.yaml`, `h1b_natural_mps4_sep.yaml` | H1b on adjacent (k = 2) and separated (k = 3) m = 2 pairs on MPS4 | ~15 min |
| `h1b_saddle_mps6.yaml`, `h1b_saddle_b8.yaml` | m >= 3 wrong-saddle starts on MPS6 and binary8 | ~1 h (slow runs) |
| `d_test_*.yaml` | d = 2 vs d = 4 at equal bond dimensions with independent seed ranges | ~20 min |

Variants: any key of a config can be changed (trees as `"kind:N"`, `d`, `chi`, `cap`, `rho`,
`sigma`, `seeds`, `pattern`); a new `name` writes a new output folder.

## Results

See `notes/order.md` (section 7) and `agent-runs/order/REPORT.md`; summary tables are in
`results/*.csv`.
