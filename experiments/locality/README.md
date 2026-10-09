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
- Dynamics: gradient flow, scipy `solve_ivp` RK45 (Dormand-Prince 5(4)), rtol 1e-9, atol 1e-14, on the flat
  parameter vector with a `torch.compile`d autograd gradient of `ttn.losses.population_loss`
  (`pilot/pilot3.py`; to be ported to `ttn.dynamics` when it lands). Wall-clock limit 1500 s per run.
- Learning event of component r: final upward crossing of a_r = 1/2 (every crossing is recorded). Plateau =
  interval between consecutive events; its midpoint = minimum gradient norm inside it, where the
  gauge-projected Hessian is evaluated.

## How to run
```
uv run python experiments/locality/pilot/make_cases_2comp.py experiments/locality/pilot/cases_2comp.json
uv run python experiments/locality/pilot/pilot3.py experiments/locality/pilot/cases_2comp.json /root/outputs/locality/r2comp 36
uv run python experiments/locality/pilot/pilot3.py experiments/locality/pilot/cases_theory_examples.json /root/outputs/locality/rtheory 12
uv run python experiments/locality/pilot/pilot3.py experiments/locality/pilot/cases_step3_n4.json /root/outputs/locality/rstep3n4 9
uv run python experiments/locality/pilot/make_cases_extra.py   # nested N = 6, 8 and robustness case lists
uv run python experiments/locality/analysis.py /root/outputs/locality/r2comp          # exponent table
```
The runner is resumable (finished runs are skipped) and writes one JSON line per run to
`<prefix>.part<pid>`; subsampled trajectories go to `/root/outputs/locality/pilot3/*.npz`.

Variants: edit or extend the case lists (keys: `kind`, `n`, `S` or `patterns` + `weights`, `d`, `chi`,
`start` = small | saddle, `sigmas`, `seeds`, `teacher_seed`).

## Files
| file | role |
|---|---|
| `rules.py` | trees, Steiner rule, bond-rank rule, route rule k* (projected local residual), Furman's N, literal H2 |
| `pilot/pilot3.py` | runner (gradient flow, events, plateau Hessians, trajectories) |
| `analysis.py` | plateau table, per-seed exponent fits, log-law slopes, Hessian scaling, dormant-norm test |
| `growth.py` | single-run escape degree from d log(da/dt) / d log a = 2 - 2/k |
| `bond_events.py` | which bonds gain rank at each event vs prediction |
| `plot_escape.py`, `plot_trees.py` | figures |
| `pilot/pilot2.py`, `fit.py` | first pilot (pre-fix library; superseded) |

## Results
See `notes/locality.md` section 7 and `agent-runs/locality/REPORT.md` (being filled).
