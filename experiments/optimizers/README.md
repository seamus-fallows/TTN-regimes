# Experiment `optimizers`: does the saddle-to-saddle picture survive practical optimizers? (B1)

Agent `optimizers`; findings and theory in `notes/optimizers.md`; running report in
`agent-runs/optimizers/REPORT.md`.

## Question
Gradient flow (GF) from small init on a TTN shows saddle-to-saddle (S2S) dynamics: plateaus, one bond
singular value at a time, learning order by (weight x init overlap), plateau ~ sigma^-(K-2). Which of these
survive the optimizers used in practice: Adam / RMSProp, heavy-ball momentum, GD with re-gauging to a
canonical form, Riemannian GD on the fixed-bond-dimension manifold, per-core Gram-preconditioned GD (ALS
normal equations used simultaneously), and the sweep methods ALS (one site) and DMRG (two sites + SVD)?

## Setup
- Model: `notes/model.md` (whitened inputs, exact population loss, float64); finite samples in
  `implicit_bias_b4` only (empirical loss, `ttn.dynamics.make_dataset`).
- Trees / teachers: binary N=4 (K=3), d=3, chi=4, odeco (3, 2, 1); binary N=8 (K=7) and MPS N=8 (K=8),
  d=3, chi=4, odeco (3, 2, 1); the locality pattern teacher E8 of `notes/theory_saddles.md` 5.4
  (q0 = 0^8 weight 1, qA = 1 on sites {3,4} weight 0.4, qB = 1 on sites {1,6} weight 0.7; d=2, chi=3) on
  binary N=8 and MPS N=8. Teacher seed 100, init seeds 0-5 (independent streams).
- Inits: fan-in (`TTNSpec.init`); `init: balanced` = the same W_0 gauge-balanced (Q_e = 0, `gauge.balance`),
  used for the deep-tree GF baselines because fan-in GF stalls there (conservation-law trap).
- Clocks: GF time for gf (and lr n / (1 - beta) for GD / heavy ball in figures); steps for Adam, RMSProp,
  Gram-preconditioned GD, Riemannian GD, re-gauged GD; sweeps for ALS / DMRG.
- Diagnostics: library `Recorder` (loss, bond spectra, overlaps a_r, conservation drift) plus the balance
  index ||Q_e|| / (||G_v^T G_v|| + ||G_p G_p^T||) and ||Q_e|| per edge; discrete methods also keep a fine
  trace (loss, overlaps every max(1, n_steps / 50000) steps) for event times. Events, widths, separation
  ratios: `ttn.diagnostics` (`notes/diagnostics.md`).

## Code
| file | contents |
|---|---|
| `common.py` | problem from a config row, init (fan-in / balanced), extra recorder diagnostics |
| `gauge.py` | canonical forms (QR sweeps), centre moves, environment Grams, leaves-to-root HSVD, gauge balancing |
| `optimizers.py` | GF (library LSODA), GD / heavy ball / Adam / RMSProp (torch.optim), re-gauged GD, Gram-preconditioned GD, Riemannian GD, ALS, DMRG |
| `run.py` | sweeps from YAML (Cartesian grids, `when` filters, sigma-dependent budgets), parallel and resumable |
| `analysis.py` | per-run summary (events, gaps, separation ratios, order, rank growth, balance) -> `results/<sweep>_summary.csv` |
| `adam_probe.py` | Adam's preconditioner and preconditioned sharpness along a run |
| `plots.py`, `make_figures.py` | all figures and tables |
| `test_optimizers.py` | tests of the gauge code (`uv run pytest experiments/optimizers`) |

## How to run
```
uv run python -m experiments.optimizers.run experiments/optimizers/configs/<sweep>.yaml [--workers N]
uv run python -m experiments.optimizers.adam_probe experiments/optimizers/configs/adam_probe.yaml
uv run python -m experiments.optimizers.make_figures            # all figures and tables
```
(`PYTHONPATH=.` may be needed so that `experiments` is importable.) Raw trajectories go to
`/root/outputs/optimizers/<sweep>/` (override with `OPTIMIZERS_OUT` / `--out`), one `.npz` per run
(`Trajectory` format, cfg in meta), not in git; summaries and figures in `results/`. Runs replaced in the fix
round after review round 1 (GF without exact event location, Gram runs with the loss guard, binary8 Adam with
the coarse trace) are in `<sweep>/superseded/` and are ignored by the analysis.

| config | runs | what |
|---|---|---|
| `scaling_b4.yaml` | 798 | every optimizer, binary4 odeco, sigma 0.002-0.128, 6 seeds |
| `precond_b4.yaml` | 210 | Gram preconditioning: scale-free vs absolute damping |
| `deep.yaml` | 864 | binary8 / mps8, odeco and E8, balanced and fan-in, sigma 0.3 and 0.8 |
| `later_b8.yaml` | 168 | binary8 odeco, 7 sigma in [0.25, 0.8]: GF, Adam, re-gauged GD |
| `eps_mps8.yaml` | 192 | Adam eps crossover, mps8, sigma 0.01-0.3 |
| `implicit_bias_b4.yaml` | 288 | finite samples, one data draw per init (round 1; superseded by `implicit_bias2_b4`) |
| `implicit_bias2_b4.yaml` | 288 | finite samples n = 45, 60 with 4 independent data draws, paired GF / Adam |
| `adam_probe.yaml` | 9 | Adam internals |
| `pilot_calib.yaml`, `pilot_trap.yaml` | - | pilots (cost, fan-in trap) |

## Variants
Any key of a config row can be swept: optimizer hyperparameters (`lr`, `beta`, `beta1`, `beta2`, `eps`,
`alpha`, `period`, `damping`, `damping_abs`, `cutoff`, `n_sweeps`), problem (`tree`, `n_sites`, `d`, `chi`,
`teacher`, `weights`, `patterns`), init (`sigma`, `seed`, `init`), data (`n_samples`, `data_seed`), budgets
(`t_max`, `n_steps`, or `budget_c` x sigma^-`budget_power` GF-time units), stop tolerance `tol` (loss / L0).

## Order of the scripts
1. Sweeps: `run.py` with each config (any order; `deep` before `eps_mps8`'s figure, which uses the
   GF times of `deep` as the reference). Process count: `--workers` (one thread per process).
2. `adam_probe.py configs/adam_probe.yaml` (Adam internals).
3. `make_figures.py --results-dir experiments/optimizers/results` (summaries are built on demand from the
   raw runs, cached as `<sweep>_summary.csv`; delete them to rebuild).
4. `consistency.py` with `OPTIMIZERS_RESULTS=experiments/optimizers/results` (theory_optimizer P1/P2/P6 and the
   R12 route-end check; needs the summaries of step 3; `p6` adds the coarse path-form P6 table).
Default output of steps 3-4 is `/root/outputs/optimizers/results` (outside git), so re-running never
overwrites the committed results unless asked to.

## Results (details, numbers and caveats in `notes/optimizers.md` section 2; claims changed after review round 1 in its section 5)
| figure / table | shows |
|---|---|
| `scaling_b4_scaling.png`, `scaling_b4_exponents.csv` | binary4: event times vs sigma (per-seed exponents): GF/GD p = 1.007, heavy ball 0.970, Adam/RMSProp first escape p = 0.01, re-gauged GD p = 0 |
| `scaling_b4_side_sigma0.008_seed0.png`, `deep_side_*.png` | loss, overlaps and bond spectrum per optimizer from the same init |
| `later_b8_scaling.png`, `later_b8_exponents.csv` | binary8: GF p = 4.74 (4.99 on sigma <= 0.4; theory 5); Adam first escape sigma-free; later plateaus grow, no law; re-gauged GD sigma-free |
| `adam_escape.png`, `eps_mps8_*.csv` | Adam's first escape: sign-flow limit (binary4) and the eps crossover on mps8 with the parameter-free prediction (per seed within 0.1%) |
| `adam_probe.png` | Adam's preconditioner along four single runs (memory decay; AEoS reached on the last plateaus) |
| `balance.png` | conservation-law drift (GF 1e-11, GD 1e-5, Adam O(1)) and Adam's final balance vs the entry-count heuristic |
| `implicit_bias2_b4.png`, `implicit_bias2_*.csv` | finite samples, 4 data draws: no population-loss difference between GF and Adam |
| `order_agreement_with_gf.csv`, `order_independent_seeds.csv` | learning order vs GF (resolved orders only); sigma-free methods per independent seed (ties, weakest last, largest first) |
| `r12_route_ends_mps8.csv` | conservation-law trap verified at the MPS8 route ends |
| `theory_optimizer_P1P2.csv`, `theory_optimizer_P6_slope.csv` | sharpness at the minima (P1, P2); GD event shift slope (P6) |
| `deep_table.csv`, `cost_table.csv`, `precond_b4_*.csv` | summary tables |
