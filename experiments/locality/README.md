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
X=experiments/locality; O=${LOCALITY_OUT:-/root/outputs/locality}
uv run python $X/make_configs.py                                   # writes every configs/*.yaml below
for c in twocomp twocomp_balanced twocomp_chi4 robust k5_extension khigh_extension lambda2_K \
         multicomp multicomp_wide stall_diagnosis; do
  uv run python $X/run.py $X/configs/$c.yaml $O/<prefix> <workers>   # prefixes: see finalize.sh
done
uv run python $X/run.py $X/configs/finish_twocomp.yaml $O/twocomp 6          # slow runs, longer wall limit
uv run python $X/run.py $X/configs/finish_twocomp_chi4.yaml $O/twocomp_chi4 6
uv run python $X/predict_sequences.py      # route-rule predictions -> results/route_sequences.json
$X/finalize.sh /tmp/locality_results 8     # every table and figure, into a fresh directory
LOCALITY_OUT=data/locality bash $X/finalize.sh /tmp/locality_results 8   # the same from the local copy
```
- `run.py <config> <out_prefix> <workers>` is resumable (finished runs are skipped; errors and timeouts are
  retried) and appends one JSON line per run to `<out_prefix>.part<pid>`; every accepted integrator step is saved
  to `$LOCALITY_OUT/traj/*.npz`.
- Portability: `LOCALITY_OUT` sets the raw-output directory (default `/root/outputs/locality`), for run records
  and trajectories alike. Run records store the remote trajectory path (`/root/outputs/locality/traj/<run>.npz`);
  where that file does not exist, `analysis.load_records` points the record to `$LOCALITY_OUT/traj/<run>.npz`, so
  every step that reads trajectories (`hess_recompute.py`, `growth.py`, `route_amps.py`, `levels.py`,
  `bond_events2.py`, `r12.py`, `plot_trees.py`, `plot_bonds.py`) runs on a local copy with
  `LOCALITY_OUT=data/locality` alone (no link needed; on the remote machine nothing changes).
  `LOCALITY_EAGER=1` disables `torch.compile` (identical numbers; needed without a C++ compiler, e.g. on
  Windows).
- `finalize.sh` never writes into `results/` unless that directory is passed; tables are restricted to the runs
  of the committed configs (`--config`), so they are regenerable. It stops at the first failing step
  ("finalize.sh: step failed: ..."; the step's error stays on the terminal); `hess_recompute.py` fails if no
  record or a trajectory is not found, and recomputes every row (it no longer resumes from an existing
  `hessians.csv`, which had kept Hessians of superseded runs: `notes/locality.md` section 11).
- Case keys: `kind`, `n`, `S` (two-component teacher) or `name` + `patterns` + `weights`, `d`, `chi`, `start`
  (small | saddle | small_balanced), `sigmas`, `seeds`, `teacher_seed` (default 1000), `rtol`, `atol`, `t_max`,
  `wall_limit`, `stall_detect`, `stall_patience` (evaluations at |v| < 1e-12 before a run is called stalled;
  default 5000), `save_partial` (keep the trajectory of stalled / timed-out runs).

## Files
| file | role |
|---|---|
| `rules.py` | trees, Steiner rule, bond-rank rule, route rule k* (projected local residual), route sequence at rank-constrained optima, Furman's N, literal H2 |
| `run.py` | runner (library model + LSODA, events, plateau Hessians, stall detection, balanced init) |
| `make_configs.py`, `configs/*.yaml` | sweep definitions |
| `analysis.py` | plateau table, run classes, per-seed fits, fair log-vs-power comparison, stall dropout |
| `make_figures.py` | exponent tables (`*_table.md/.csv`), exponents and Hessian figures |
| `hess_recompute.py` | Hessians at the stored plateau midpoints: raw and fixed gauge projection |
| `growth.py` | single-run escape degree from the growth exponent |
| `levels.py`, `predict_sequences.py`, `plot_trees.py` | step 3: predicted levels, detected plateaus, trees figures |
| `bond_events2.py`, `plot_bonds.py` | step 4: rank changes at events, per-saddle rank profiles, figure |
| `route_amps.py`, `logcorr.py` | route-entry amplitudes at plateau midpoints; log-correction test |
| `r12.py` | stall diagnosis: dead vs slow, theory R12's criterion at the route ends |
| `stall_tables.py` | run classes per sweep, sigma-dependent stall dropout and its effect on fits, stalls vs capped route ends |
| `finalize.sh` | regenerate all results into a given directory |
| `bond_events.py`, `pilot/`, `fit.py`, `plot_escape.py` | round-1 / pilot code (superseded) |

## Results (details: `notes/locality.md` sections 7-10; round 3 after notes/reviews/locality_round2_verification.md)
- Two components: plateau exponent k - 2 (k = Steiner subtree size) for k = 1-5 on all 9 trees (no plateau for
  k = 1; log law with head-start slope for k = 2); deviating rows are carried by partially stalling seeds or
  return to k - 2 at small sigma, except mps8 {1,5} from the saddle start (3.67 ± 0.20, 4 seeds, 0.4 decades);
  k = 6-7 = k - 2 on binary8 and ht4 with balanced init only, over one decade; k = 8 not measured. `results/twocomp_table.md`, `extensions_table.md`, `twocomp_exponents.png`.
- Hessian at plateau midpoints (fixed gauge projection and raw): λ_min = -λ2 = -s2 s1^((K-2)/K) for k = 2,
  ∝ sigma^x -> 0 for k >= 3 (x within 0.2 of k - 2 in 127 of 135 k = 3-5 rows; raw quoted where the projection is biased).
  `results/twocomp_hessian.png`, `hessians.csv` (regenerated from the final run records on 2026-10-10:
  `notes/locality.md` section 11).
- Several components: the route rule at rank-constrained optima predicts every k* >= 3 plateau (detected
  plateaus, exponents within 0.2; one exception, B4 on ht4 with fan-in init) and the rank profiles at the
  predicted saddles; TD and TC separate it from
  the literal H2 rule and the bond-rank rule. `results/trees_*.png`, `levels_summary.csv`, `bond_profiles_multicomp.csv`.
- Bond spectra: leakage bound (non-Steiner edges <= 4.3 sigma^2 s2 through every event) plus single-route
  consistency (all Steiner edges rise, together). `results/bond_events2_twocomp.csv`, `bond_spectra.png`.
- Dead routes (fan-in init, chi = 2, k >= 4 only): 398 / 5,682 runs; balanced init 16 / 1,008; chi = 4 106 / 1,005
  (vs 169 / 1,008 at chi = 2 on the same cases: removed on routes with an uncapped end, more on routes with both
  ends capped). `results/run_classes_by_sweep.csv`, `stall_dropout_*.csv`, `stall_capped_ends.csv`, `r12.csv`.
