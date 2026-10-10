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

## How to run (from the worktree root, Git Bash)

```
PY=/c/Users/seamu/Iliad-ML/TTN_regimes/.venv/Scripts/python.exe
PYTHONPATH=. $PY experiments/theory_checks/trap/predict_locality.py --data <data/locality> --pilot <data/pilot_h2_locality> --workers 6
PYTHONPATH=. $PY experiments/theory_checks/trap/analyze_locality.py
PYTHONPATH=. $PY experiments/theory_checks/trap/predict_s2s.py --workers 6
PYTHONPATH=. $PY experiments/theory_checks/trap/gf_check_s2s.py 0.03 separated
PYTHONPATH=. $PY experiments/theory_checks/trap/montecarlo.py --workers 6
```

Raw run records are read from `data/locality/` and `data/pilot_h2_locality/` (read only; not in git);
the s2s_core classes from the committed `experiments/s2s_core/results/main_runs.csv`.

## Results

See `results/` and `notes/theory_trap.md` sections 7-8.
