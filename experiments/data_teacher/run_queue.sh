#!/bin/sh
# Runs the data-teacher sweeps one after the other (each is resumable). Logs in agent-runs/data-teacher/logs/.
cd "$(dirname "$0")/../.."
L=agent-runs/data-teacher/logs
P=experiments/data_teacher
for c in "$@"; do
  case $c in
    e1_*) uv run python $P/run_aligned.py $P/configs/$c.yaml --procs 28 > $L/$c.run.log 2>&1 ;;
    *) uv run python $P/run_generic.py $P/configs/$c.yaml --procs 28 > $L/$c.run.log 2>&1 ;;
  esac
done
