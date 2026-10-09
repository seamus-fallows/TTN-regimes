# Experiment `locality` (Q3: H2, H3)

**Question.** After a component is learned, how long does the next escape take, and how does that
depend on where the next component lives in the tree? H2: the escape needs the k cores of the Steiner
subtree of the sites where the new component differs; plateau ~ sigma^-(k-2) (k >= 3), ~ log(1/sigma)
(k = 2), none (k = 1). H3: the saddle is strict iff k = 2. Theory: `notes/locality.md` (this agent),
`notes/theory_saddles.md` (lead, draft).

## Setup
- Model as in `notes/model.md`: whitened Gaussian inputs, exact population loss, fan-in init with per-core
  scale sigma, float64.
- Teachers: pattern teachers (`ttn.teachers.pattern_teacher`), random orthonormal site bases with teacher
  seed 1000 (independent of the init seeds 0-7).
  - two components, weights (1, 0.5) (also (1, 0.25), (1, 0.75)), the second differing on a site set S;
  - multi-component teachers TA, TB (nested), B4, E8, N4 (from `notes/theory_saddles.md` section 5), TC, TD.
- Trees: `mps` N = 4, 6, 8; `binary` N = 4, 8; `ht` N = 4, 8; custom `star3` (root without sites,
  three children with two sites each) and `midroot6` (chain rooted in the middle); see `rules.py`.
- d = 2, chi = 2 (two components) or 3 (three-component theory examples); robustness chi = 4, 8, d = 4.
- Two protocols: **small** (the brief's: fan-in init at scale sigma, the first plateau included; the plateau
  length is the time between consecutive learning events) and **saddle** (exact balanced saddle s1 U1 from
  `aligned_init` plus sigma-scale fan-in noise; isolates the escape law and works for HT8, whose first
  plateau is unreachable at small sigma).
- Dynamics: gradient flow with scipy LSODA (`solve_ivp`, exact-Hessian Jacobian; the solver behind
  `ttn.dynamics.integrate_gf(method="lsoda")`), rtol 1e-9, atol 1e-14, on the flat parameter vector of the
  library model (`ttn.dynamics.Problem`), field and Jacobian compiled with torch.compile; event times by
  root-finding (Radau fallback if LSODA's dense output fails); stalled runs (dead routes) detected and
  recorded; wall-clock limit 1500 s (900 s for k >= 6). Runner: `run.py`.
- Learning event of component r: final upward crossing of a_r = 1/2 (every crossing is recorded). Plateau =
  interval between consecutive events; its midpoint = minimum gradient norm inside it, where the
  gauge-projected Hessian is evaluated.

## How to run
Order: (1) `run.py` for each config, (2) `predict_sequences.py`, (3) `finalize.sh <out_dir> [workers]`.
```
uv run python experiments/locality/run.py experiments/locality/configs/twocomp.yaml /root/outputs/locality/twocomp 36
uv run python experiments/locality/run.py experiments/locality/configs/twocomp_balanced.yaml /root/outputs/locality/twocomp_balanced 16
uv run python experiments/locality/run.py experiments/locality/configs/multicomp.yaml /root/outputs/locality/multicomp 16
uv run python experiments/locality/run.py experiments/locality/configs/multicomp_wide.yaml /root/outputs/locality/multicomp_wide 8
uv run python experiments/locality/run.py experiments/locality/configs/robust.yaml /root/outputs/locality/robust 22
uv run python experiments/locality/run.py experiments/locality/configs/k5_extension.yaml /root/outputs/locality/k5ext 16
uv run python experiments/locality/predict_sequences.py        # route-rule predictions -> results/route_sequences.json
experiments/locality/finalize.sh /tmp/locality_results 8       # all tables and figures into a fresh directory
```
`run.py <config> <out_prefix> <workers>` is resumable (finished runs are skipped, errors and timeouts are
retried) and appends one JSON line per run to `<out_prefix>.part<pid>`; every accepted integrator step is
saved to `/root/outputs/locality/traj/*.npz`. Configs are written by `make_configs.py`.

Variants: case keys `kind`, `n`, `S` (two-component teacher) or `name` + `patterns` + `weights`, `d`, `chi`,
`start` (small | saddle | small_balanced), `sigmas`, `seeds`, `teacher_seed` (default 1000), `rtol`, `atol`,
`t_max`, `wall_limit`.

## Files
| file | role |
|---|---|
| `rules.py` | trees, Steiner rule, bond-rank rule, route rule k* (projected local residual), Furman's N, literal H2 |
| `run.py` | runner (library model + LSODA, events, plateau Hessians, stall detection, balanced init) |
| `make_configs.py`, `configs/*.yaml` | sweep definitions |
| `predict_sequences.py` | route-rule predictions for the multi-component teachers |
| `levels.py` | step 3: time at each predicted plateau level |
| `make_figures.py` | exponent tables, exponents and Hessian figures |
| `plot_bonds.py` | step 4 figure |
| `finalize.sh` | regenerate all results into a given directory |
| `pilot/pilot3.py` | RK45 runner used before library v0 (superseded) |
| `analysis.py` | plateau table, per-seed exponent fits, log-law slopes, Hessian scaling, dormant-norm test |
| `growth.py` | single-run escape degree from d log(da/dt) / d log a = 2 - 2/k |
| `bond_events.py` | which bonds gain rank at each event vs prediction |
| `plot_trees.py` | step 3 figure; `plot_escape.py`: pilot figure (superseded) |
| `pilot/pilot2.py`, `fit.py` | first pilot (pre-fix library; superseded) |

## Results (details: `notes/locality.md` sections 7-8)
- Two components: the plateau after the first component scales as predicted by the Steiner count k on all
  9 trees: no plateau (k = 1), log law with slope (1 - s2/s1)/λ2 (k = 2; 1/λ2 from the saddle start),
  σ^-(k-2) for k = 3 (0.98-1.03), 4 (1.9-2.1), 5 (2.9-3.1 except two MPS rows), 6-8 (within errors).
  `results/twocomp_table.md`, `twocomp_exponents.png`.
- Hessian at plateau midpoints: λ_min = -λ2 (k = 2), ∝ σ^(k-2) (k >= 3). `results/twocomp_hessian.png`.
- Several components: route rule at rank-constrained optima predicts all sequences tested (TA, TB, N4, B4,
  E8, TC, TD); literal H2 fails. `results/trees_*.png`, `levels_summary.csv`.
- Bond spectra: rank rises on exactly the Steiner edges in 98.7% of events. `results/bond_spectra.png`.
- Fan-in init kills k >= 4 routes in a sizeable fraction of runs; gauge-balanced init does not.
- Robust to χ, d and the weight ratio. `results/robust_table.md`.
