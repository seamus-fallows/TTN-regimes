#!/bin/sh
# Runs the order experiment's sweeps in sequence (each is resumable). Logs in agent-runs/order/logs.
cd "$(dirname "$0")/../.."
for c in "$@"; do
  case $c in
    h1b_*) script=experiments/order/h1b.py ;;
    multi*) script=experiments/order/multi.py ;;
    *) script=experiments/order/sweep.py ;;
  esac
  uv run python $script experiments/order/configs/$c.yaml > agent-runs/order/logs/$c.log 2>&1
done
