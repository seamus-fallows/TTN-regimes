#!/bin/sh
# Dense (library integrator) sweeps one after the other; logs in agent-runs/data-teacher/logs/.
cd "$(dirname "$0")/../.."
for c in "$@"; do
  uv run python experiments/data_teacher/run_dense.py experiments/data_teacher/configs/$c.yaml --procs 28 > agent-runs/data-teacher/logs/$c.dense.log 2>&1
done
