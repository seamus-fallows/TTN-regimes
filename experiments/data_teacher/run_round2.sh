#!/bin/sh
# Round 2 and its fix rounds, in order (every runner is resumable: finished runs are skipped).
# Usage: [PROCS=28] [DATA_TEACHER_OUT=<raw outputs>] [DATA_TEACHER_RESULTS=<csv/figures>] \
#        [DATA_TEACHER_LOGS=<logs>] experiments/data_teacher/run_round2.sh
# Set DATA_TEACHER_RESULTS when re-running, or the committed results are overwritten.
cd "$(dirname "$0")/../.."
. experiments/data_teacher/env1.sh
L=${DATA_TEACHER_LOGS:-agent-runs/data-teacher/logs}
P=${PROCS:-28}
mkdir -p "$L"
D=experiments/data_teacher
# item 1: saddle start (stores the exact saddles in results/r2_saddle_cores.pt and their Hessians)
for c in r2_saddle r2_saddle_b r2_saddle_ext r2_saddle_m20 r2_saddle_fine; do
  uv run python $D/r2_saddle_runs.py $D/configs/$c.yaml --procs $P > $L/$c.log 2>&1
done
uv run python $D/recompute_saddle_hessians.py --procs $P > $L/recompute_hessians.log 2>&1
# item 1 random init, round-1 sigma extensions, step 4(c) cross-tree teachers (library, dense grid)
PROCS=$P $D/run_dense_queue.sh r2_random e4a_sigma e2_sigma e4b_ttn r2_cross r2_cross_b
# item 2: star teachers
uv run python $D/r2_star.py hessian > $L/r2_star_hessian.log 2>&1
uv run python $D/r2_star.py saddle --procs $P > $L/r2_star_saddle.log 2>&1
uv run python $D/r2_star.py threshold --procs $P > $L/r2_star_threshold.log 2>&1
uv run python $D/r2_star.py complete_ht > $L/r2_star_complete_ht.log 2>&1
uv run python $D/r2_star.py random --procs $P > $L/r2_star_random.log 2>&1
# item 3: correlated inputs
uv run python $D/r2_corr.py --procs $P > $L/r2_corr.log 2>&1
