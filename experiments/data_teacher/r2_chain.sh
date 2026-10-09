#!/bin/sh
# Round-2 queue after the random-init sweep: N = 8 misaligned supplement, saddle-start extension,
# round-1 sigma extensions (dense), star runs. Each step is resumable.
cd "$(dirname "$0")/../.."
until [ -f experiments/data_teacher/results/r2_random_dense_summary.csv ]; do sleep 30; done
. experiments/data_teacher/env1.sh
L=agent-runs/data-teacher/logs
uv run python experiments/data_teacher/r2_saddle_runs.py experiments/data_teacher/configs/r2_saddle_b.yaml --procs 28 > $L/r2_saddle_b.log 2>&1
uv run python experiments/data_teacher/r2_saddle_runs.py experiments/data_teacher/configs/r2_saddle_ext.yaml --procs 28 > $L/r2_saddle_ext.log 2>&1
experiments/data_teacher/run_dense_queue.sh e4a_sigma e2_sigma
uv run python experiments/data_teacher/r2_star.py saddle --procs 28 > $L/r2_star_saddle.log 2>&1
uv run python experiments/data_teacher/r2_star.py random --procs 28 > $L/r2_star_random.log 2>&1
