#!/bin/sh
# Second round-2 queue (review follow-ups): runs after r2_chain.sh has finished.
cd "$(dirname "$0")/../.."
while pgrep -f "r2_chain[.]sh" > /dev/null; do sleep 30; done
. experiments/data_teacher/env1.sh
experiments/data_teacher/run_dense_queue.sh e4b_ttn r2_cross
