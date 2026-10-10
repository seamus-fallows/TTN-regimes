# Reproduction review of `optimizers` (B1) - part 1: setup and README check

Repo HEAD 81d1ab9 (merge 79575a0 of agent/optimizers). Machine: Windows 11, Intel Ultra 9 275HX (24 cores),
5 worker processes + 1 independent-check process.

## README run instructions
- `uv run pytest experiments/optimizers`: 4 tests pass.
- `uv run python -m experiments.optimizers.run <cfg> [--workers N]` FAILS on Windows:
  run.py:124 `mp.get_context("fork")` -> `ValueError: cannot find context for 'fork'`.
  Also the default `--out /root/outputs/optimizers` resolves to `C:\root\outputs\optimizers` and is
  `mkdir`-ed before the `--dry` check (would litter C:\ on Windows); I always passed `--out`.
- Workaround used: scratchpad `driver.py` imports `experiments.optimizers.run.{jobs_from_config, run_id, work}`
  unchanged and runs them in a `spawn` pool (per-run code path is exactly the agent's).
- Reduced configs are strict subsets of the committed YAMLs: all 357 run ids (sha1 of the cfg) match
  rows of the committed `*_summary.csv`, so every reproduced run is compared one-to-one with the
  committed row (`file` column).
- README says scaling_b4 = 672 runs; the YAML expands to 798 (19 settings x 42) - doc slip.

## Environment caveat discovered mid-run (not caused by me)
At 21:54-21:55 (local) someone else modified the library working tree (uncommitted): ttn/dynamics.py, ttn/diagnostics.py,
ttn/plotting.py, ttn/theory.py (+538/-105 lines; Recorder "fine rows" on by default, threadpoolctl thread limit,
Problem.describe() adds component_targets). HEAD also moved (81d1ab9 -> d689db6, report/notes commits only).
My driver started 21:48:06; workers recycle every 20 tasks, so later workers imported the modified library.
Runs are labelled by meta['problem'] containing 'component_targets' (new lib): see scratch libversion.py.
Effect seen: discrete methods give identical results under both versions; GF runs under the new library carry fine
rows, so the agent's analysis takes GF event times from them -> they equal my exact event times (e.g. MPS8 balanced
seed 0: 984.7913 vs committed grid-interpolated 968.04). The trajectory itself is unchanged (same grid values).
