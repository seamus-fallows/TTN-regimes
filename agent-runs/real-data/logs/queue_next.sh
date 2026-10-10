#!/bin/bash
# launched by the agent: start chi8 and fashion sweeps once the main sweep has exited
cd /root/work/TTN-regimes
MAIN=$(cat agent-runs/real-data/logs/main.pid)
while kill -0 $MAIN 2>/dev/null; do sleep 30; done
nohup uv run python experiments/real_data/run.py experiments/real_data/configs/chi8.yaml /root/outputs/real_data/chi8 24 > agent-runs/real-data/logs/chi8.log 2>&1 &
echo $! > agent-runs/real-data/logs/chi8.pid
nohup uv run python experiments/real_data/run.py experiments/real_data/configs/fashion.yaml /root/outputs/real_data/fashion 24 > agent-runs/real-data/logs/fashion.log 2>&1 &
echo $! > agent-runs/real-data/logs/fashion.pid
nohup uv run python experiments/real_data/run.py experiments/real_data/configs/img8.yaml /root/outputs/real_data/img8 12 > agent-runs/real-data/logs/img8.log 2>&1 &
echo $! > agent-runs/real-data/logs/img8.pid
