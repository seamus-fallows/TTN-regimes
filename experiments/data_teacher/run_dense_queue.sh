#!/bin/sh
# Dense (library integrator) sweeps one after the other.
# Usage: [PROCS=28] experiments/data_teacher/run_dense_queue.sh <config name> ...
cd "$(dirname "$0")/../.."
. experiments/data_teacher/env1.sh
L=${DATA_TEACHER_LOGS:-agent-runs/data-teacher/logs}
mkdir -p "$L"
for c in "$@"; do
  uv run python experiments/data_teacher/run_dense.py experiments/data_teacher/configs/$c.yaml --procs ${PROCS:-28} > $L/$c.dense.log 2>&1
done
