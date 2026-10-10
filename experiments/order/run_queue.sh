#!/bin/sh
# Runs the order experiment's configs in sequence (each is resumable). Logs in agent-runs/order/logs.
# Usage: run_queue.sh [--raw DIR] [--res DIR] config1 config2 ...   (names without .yaml)
# --raw/--res (or the environment variables ORDER_RAW / ORDER_RES / PROCS) are passed to every
# script, so a re-run need not overwrite the committed experiments/order/results/.
cd "$(dirname "$0")/../.."
opts=""
while [ $# -gt 0 ]; do
  case $1 in
    --raw|--res) opts="$opts $1 $2"; shift 2 ;;
    *) break ;;
  esac
done
for c in "$@"; do
  case $c in
    h1b_*) script=experiments/order/h1b.py ;;
    multi*) script=experiments/order/multi.py ;;
    *) script=experiments/order/sweep.py ;;
  esac
  uv run python $script experiments/order/configs/$c.yaml $opts > agent-runs/order/logs/$c.log 2>&1
done
