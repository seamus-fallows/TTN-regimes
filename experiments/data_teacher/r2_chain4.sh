#!/bin/sh
# Fourth queue: fine threshold scan for the star teachers (after r2_chain.sh, which runs the star sweeps).
cd "$(dirname "$0")/../.."
while pgrep -f "r2_chain[.]sh" > /dev/null; do sleep 30; done
. experiments/data_teacher/env1.sh
uv run python experiments/data_teacher/r2_star.py threshold --procs 28 > agent-runs/data-teacher/logs/r2_star_threshold.log 2>&1
