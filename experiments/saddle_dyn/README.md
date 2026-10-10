# saddle_dyn: escape times from a learned rank-one saddle vs the saddle order

**Question.** `notes/theory_saddle_order.md` proves the saddle order p at a learned rank-one saddle
theta_S (p = k* when the generalized piggyback operator S has lambda_max < s_q; p = 2 l* < k* when it
exceeds s_q at a level l* < k*/2) and derives (S9, reduced model) that the escape takes
t_esc ~ eps^-(p-2) from theta_S + eps xi and ~ sigma^-(p-2) from small init. This experiment tests the
dynamical prediction with pre-registered windows and decision rules: **`notes/saddle_dyn.md`** (Part I =
registration, written and pushed before any run; Part II = results).

## Setup

- Cells (section 2 of the note): MPS6 pair-star (0.8; order 4, k* = 5) and its control (0.6; 5);
  MPS6 110001 wrong order (1.5; 4 vs 6) and control (0.7; 6); MPS5 10101 wrong order without a low
  split (1.5; 5 = k*); HT4 1111 wrong order (1.5; 6 vs 7) and control (0.5; 7); the locality HT4
  teacher (s_q = 0.5; 6 vs 7); the s2s_core HT4 "close" odeco teacher (s_q = 0.85; 6 vs 7); the
  `order` agent's binary8 wrong-order saddle (rho = 0.8 and 0.5; 6 vs 7). Definitions:
  `configs/cells.yaml`.
- H-A: gradient flow from theta_S + eps xi (12 directions per cell, eps = 10^-1 ... 10^-5 in quarter
  decades, truncated at the float64 floor; binary8: fan-in perturbation at sigma as in the `order`
  experiment). H-B: from small fan-in init (PS08: 40 seeds, PS06: 12, binary8 rho = 0.8: 30), the
  plateau dt = t_esc - t_q of the runs that learn q first.
- Dynamics: whitened population loss, float64, one thread per process, LSODA (rtol 1e-10, atol
  1e-15, exact Hessian as Jacobian). The gradient near theta_S is **cancellation-free**
  (W - W_S by a telescoping sum over cores; `sdyn.Model.delta`). Contraction from the verifier's
  independent `experiments/theory_checks/saddle_order_verify/vttn.py`.
- The integrator is driven step by step (`sdyn.integrate`): events are detected on the step values
  and refined by bisection with exact re-integration. scipy's LSODA dense output (used by
  `solve_ivp` events) was found unreliable here (2-4 % off at step times); see deviation 3 in the note.
- Floor flag: a-posteriori bound on the accumulated roundoff in the dormant (new) entries
  (`runs.floor_index`; flagged if > 1 % of their norm). Trap rule (deviation 1): runs whose routes
  are dead (0 < L - L_S < 1e-30, or every residual amplitude < 1e-45, for a decade of time) stop as
  "trapped".

## How to run (from the repo root)

    uv run python experiments/saddle_dyn/test_sdyn.py                 # gradient/Hessian/re-gauge checks
    uv run python experiments/saddle_dyn/sweep.py ha --procs 125       # H-A grid, 1836 runs
    uv run python experiments/saddle_dyn/sweep.py hb --procs 62        # H-B grid, 842 runs
    uv run python experiments/saddle_dyn/sweep.py checks --procs 60    # tolerance + plain-loss reruns
    uv run python experiments/saddle_dyn/analyze.py                    # CSVs + decisions -> results/
    uv run python experiments/saddle_dyn/figures.py                    # figures -> results/

Sweeps are resumable (existing JSON is skipped) and write raw per-run JSON to
`/root/outputs/saddle_dyn/{HA,HB}/<cell>/` (not in git). `analyze.py --raw DIR --out DIR` reads and
writes elsewhere. Variants: `sweep.py ha --cells PS08,W6`; other seeds/grids by editing
`configs/cells.yaml` (H-A) or `sweep.HB_GRID` and `--seeds-*` (H-B).

## Files

| file | contents |
|---|---|
| `sdyn.py` | model (pattern teacher, theta_S, cancellation-free residual, gradient, Hessian, roundoff bound, conserved quantities, orthogonal re-gauge), step-wise LSODA driver with events and the trap rule |
| `runs.py` | one H-A or H-B run -> JSON record (escape times at L - L_S = -1e-4, -1e-3, -1e-2, escape amplitudes and type, floor index, trajectory, what follows the escape) |
| `sweep.py` | the registered grids, parallel and resumable |
| `analyze.py` | per-seed fits in the registered windows, competing-law RSS, local exponents, two-mechanism crossover fit, decisions; H-B classification |
| `figures.py` | `results/ha_escape`, `ha_local`, `hb_dt`, `hb_local` (.pdf/.png) |
| `results/` | `ha_runs.csv` (every run), `ha_seed_fits.csv`, `ha_local_exponents.csv`, `ha_crossover.csv`, `ha_decisions.csv`, `ha_checks.csv`, `hb_*.csv`, figures |

## Results (details, tables and status tags: `notes/saddle_dyn.md` Part II)

Registered decisions (H-A = saddle start, H-B = small init; gamma = escape exponent, prediction p - 2):

| cell | p (k*) | H-A gamma (seeds) | H-A decision | H-B |
|---|---|---|---|---|
| PS08 pair-star | 4 (5) | 2.001 ± 0.001 (12) | SUPPORTED | 2.002 ± 0.001 (16 of 40 seeds visit theta_S): SUPPORTED |
| W6 MPS6 110001 (1.5) | 4 (6) | 2.001 ± 0.001 (12) | SUPPORTED | - |
| HT4W HT4 1111 (1.5) | 6 (7) | 3.996 ± 0.003 (12) | SUPPORTED | - |
| HT4L locality HT4 | 6 (7) | 3.997 ± 0.003 (12) | SUPPORTED | - |
| B8W05 binary8 rho 0.5 | 6 (7) | 3.998 ± 0.007 (12) | SUPPORTED | - |
| B8W binary8 rho 0.8 | 6 (7) | 4.48 ± 0.15 (12; 7 at 4, 5 at 5) | INCONCLUSIVE | 4.31 ± 0.12 (13; 10 at 4.0-4.2, 3 at 5): INCONCLUSIVE; route law above sigma = 0.1: SUPPORTED |
| HT4S s2s_core close teacher | 6 (7) | 4.72 ± 0.14 (11; 3 at 4, 8 at 5) | REFUTED (window above the crossover) | - |
| controls PS06, C1, C2, C3 | = k* | route law exactly in the seeds whose route survives; others shifted by +1 (+2) or trapped | INCONCLUSIVE (< 10 complete seeds) | PS06: INCONCLUSIVE (5 of 12 seeds visit) |

Figures: `results/ha_escape.png` (every run), `results/ha_local.png` (local exponents), `results/hb_dt.png`,
`results/hb_local.png`.
