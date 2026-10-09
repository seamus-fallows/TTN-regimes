#!/bin/sh
# Runs the pilot-integrator sweeps one after the other (each is resumable).
# Usage: [PROCS=28] experiments/data_teacher/run_queue.sh <config name> ...
# Logs go to agent-runs/data-teacher/logs/ (or $DATA_TEACHER_LOGS).
cd "$(dirname "$0")/../.."
. experiments/data_teacher/env1.sh
L=${DATA_TEACHER_LOGS:-agent-runs/data-teacher/logs}
mkdir -p "$L"
P=experiments/data_teacher
for c in "$@"; do
  case $c in
    e1_*) uv run python $P/run_aligned.py $P/configs/$c.yaml --procs ${PROCS:-28} > $L/$c.run.log 2>&1 ;;
    *) uv run python $P/run_generic.py $P/configs/$c.yaml --procs ${PROCS:-28} > $L/$c.run.log 2>&1 ;;
  esac
done
