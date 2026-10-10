# Part 0: environment caveat

The working tree has UNCOMMITTED changes by someone else in ttn/diagnostics.py, ttn/dynamics.py, ttn/plotting.py,
ttn/theory.py, pyproject.toml, uv.lock, tests/ (+ untracked ttn/config.py, ttn/runner.py, experiments/00_smoke/).
The reproduction imports the working-tree library. Diff checked: the hunks touch saddle_type's docstring/NaN
dropping, Problem.describe, Recorder, integrate_gf, event helpers; NOT Problem.velocity/hessian/loss,
population_loss, gauge_directions, project_out_gauge, pattern_teacher, TTNSpec.init/aligned_init. Rerun numbers
equal the committed ones to all printed digits, consistent with no effect.
