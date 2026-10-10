# Theory check: conservation-law trap predicted from the initial condition

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
