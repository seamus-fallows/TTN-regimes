#!/bin/sh
# Third round-2 queue: resolution check of the base saddle-start runs (after r2_chain2.sh).
cd "$(dirname "$0")/../.."
while pgrep -f "r2_chain2?[.]sh" > /dev/null; do sleep 30; done
. experiments/data_teacher/env1.sh
uv run python experiments/data_teacher/r2_saddle_runs.py experiments/data_teacher/configs/r2_saddle_fine.yaml --procs 28 > agent-runs/data-teacher/logs/r2_saddle_fine.log 2>&1
