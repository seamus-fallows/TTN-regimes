# Theory check: conservation-law trap predicted from the initial condition

> **2026-10-10:** verified in `notes/reviews/trap_verification.md` (tables reproduced, verdicts recomputed
> independently); its 12 errata are applied in `notes/theory_trap.md` v1.1. Code-level caveats are listed
> under "Known limitations" below; the code itself is unchanged.

Theory: `notes/theory_trap.md`. Question: can the init alone (plus the sigma-free correlation flow of
the first plateau, no gradient flow of the loss) predict which committed runs end with a dead route?

## Code

| file | what |
|---|---|
| `trapcrit.py` | the criterion: plateau-start state (saddle start: the init; small init: rescaled correlation flow to blow-up), route ends, the conserved GN quantity C_u of each end (two-core and three-core clusters), the Q-only and R12 comparison criteria, the reduced route ODE (k = 4, finite sigma) |
| `predict_locality.py` | rebuilds every committed locality and pilot two-component run from its config and seed (`TTNSpec.init(sigma, seed)`, `pattern_teacher(..., seed=teacher_seed)`; balanced: `s2s.gauge_balance`), predicts alive / pinned / dead, joins the observed class |
| `analyze_locality.py` | confusion tables (predicted vs observed; stall diagnosis outcomes from `experiments/locality/results/r12.csv`) |
| `predict_s2s.py` | s2s_core odeco runs: predicted learning order (correlation flows) and an approximate phase-2 criterion for the last component |
| `gf_check_s2s.py` | mechanism check (light gradient flow): s2s_core MPS4, C' from the actual state after the second escape |
| `montecarlo.py` | probability law: Monte Carlo over init draws |
| `tables.py`, `plot_trap.py` | markdown tables and figures for the note |

## How to run (from the worktree root, Git Bash)

```
PY=/c/Users/seamu/Iliad-ML/TTN_regimes/.venv/Scripts/python.exe
PYTHONPATH=. $PY experiments/theory_checks/trap/predict_locality.py --data <data/locality> --pilot <data/pilot_h2_locality> --workers 6
PYTHONPATH=. $PY experiments/theory_checks/trap/analyze_locality.py
PYTHONPATH=. $PY experiments/theory_checks/trap/predict_s2s.py --workers 6
PYTHONPATH=. $PY experiments/theory_checks/trap/gf_check_s2s.py 0.03 separated   # also: 0.03 close
PYTHONPATH=. $PY experiments/theory_checks/trap/gf_check_s2s.py tree ht 4 0.2 0,1,2,3,4,5   # and: tree binary 8 ...
PYTHONPATH=. $PY experiments/theory_checks/trap/montecarlo.py --workers 4
PYTHONPATH=. $PY experiments/theory_checks/trap/tables.py source,tree,chi,start   # (or s2s)
PYTHONPATH=. $PY experiments/theory_checks/trap/plot_trap.py
```

Raw run records are read from `data/locality/` and `data/pilot_h2_locality/` (read only; not in git);
the s2s_core classes from the committed `experiments/s2s_core/results/main_runs.csv`.

## Results (details in `notes/theory_trap.md` sections 7-8)

| file | content |
|---|---|
| `results/locality_pilot_predictions.csv.gz` | one row per committed run: observed class, predicted verdict (sigma -> 0), reduced model at the run's sigma, R12 and Q-only verdicts, per-end C_u, visible degree, protection |
| `results/locality_pilot_joined.csv.gz` | the same joined with the stall-diagnosis outcomes and the pilot's final amplitudes |
| `results/summary_by_criterion.csv` | stalled / runs per predicted class, per criterion and protocol |
| `results/confusion_by_case.csv`, `table_by_case.md`, `table_by_tree.md`, `table_by_k.md` | confusion tables per (tree, D, k, start, chi), per (tree, chi, start), per k |
| `results/stall_diagnosis_confusion.csv` | predicted vs dead / pinned / escaped after 60x patience |
| `results/per_seed.csv` | per (case, seed): prediction and number of sigmas with a stall |
| `results/wrong_order_runs.csv` | runs whose correlation flow escapes along the smaller component (excluded from the tables) |
| `results/s2s_predictions.csv`, `table_s2s.md` | s2s_core: predicted order and init-only phase-2 verdict vs class |
| `results/gf_check_s2s_*.txt` | mechanism checks from the actual phase-2 state (MPS4 both teachers, HT4, binary8) |
| `results/montecarlo_summary.csv`, `montecarlo_runs.csv.gz` | trap probability under fan-in init (Monte Carlo and closed form) |
| `results/fig_criteria.*`, `fig_probability.*`, `probability_vs_measured.csv` | figures |

Compute: per-run predictions about 6 min on 6 processes; Monte Carlo about 10 min on 4; gradient-flow
checks a few minutes each (single process).

## Known limitations (from the verification, 2026-10-10)

- **k = 4 route ODE, basis dependence** (erratum 4). `trapcrit.reduced_route` takes the first basis
  vector of the new space (`complement(w)[:, 0]`) on inner route edges with c > 1. `predict` checks only
  the end edges before using it, so the sigma -> 0 verdicts of k = 4 routes with a spare inner edge depend
  on an arbitrary basis choice. Affected: the MPS4 chi = 4 rows of `table_by_case.md` and
  `table_by_tree.md`, the pooled counts that include them (`summary_by_criterion.csv`, k = 4 in
  `table_by_k.md`), and the MPS4 {0,3} chi = 4 Monte Carlo row. Fix (not done): restrict the ODE verdict to
  routes with c = 1 on every edge, as the finite-sigma call in `predict_locality.py` already does, or use
  vector route amplitudes; then re-tabulate.
- **Pilot relabelling** (erratum 6). `analyze_locality.py` labels pilot runs that hit t_max or the wall
  clock with |a2_end| < 1e-20 as dead. This rule is new here, not the pilot's own detector. It affects 46
  of the 102 "dead" predicted-dead pilot runs (10 of the 102 end with the second singular value at
  1e-9 to 1e-4). With the pilot's own collapse detector alone: predicted dead 56 of 63 resolved runs
  collapse, predicted alive none.
- **R12 and Q-only verdicts** use "dead iff at least two ends negative" (`classify`), not "all ends"
  (erratum 5; the same for two-ended routes).
- **Correlation-flow precision at K = 8** (erratum 12). The stop at overlap M = 1e10 leaves the
  learned/small split ambiguous at about 6%, which flips ends with |C_u| < ~0.003. DOP853 fails at
  M = 1e16; a larger M needs the blow-up integrated in a log-amplitude variable.
- **Missing death channel** (erratum 2). The criterion has no inner three-core clusters, which act from
  k = 6; "pinned" and "alive" verdicts at k >= 6 can still die.
- **s2s-core checks.** `gf_check_s2s.py` results: MPS4 19/20 plus one stall with a barely-alive end;
  HT4 6/6 on seeds 0-5 (`results/gf_check_s2s_ht4.txt`) and 8/10 with the verifier's out-of-sample run
  on seeds 6-9 (`gf_check_s2s.py tree ht 4 0.2 6,7,8,9`, output not in `results/`).
