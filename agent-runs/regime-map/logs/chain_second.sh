#!/bin/bash
# run the second batch of sweeps after the main sweep (pid in main.pid) exits
cd /root/work/TTN-regimes
MAIN=$(cat agent-runs/regime-map/logs/main.pid)
while kill -0 "$MAIN" 2>/dev/null; do sleep 30; done
uv run python experiments/regime_map/sweep.py experiments/regime_map/configs/imbalance_b4.yaml experiments/regime_map/configs/large_chi_b4.yaml experiments/regime_map/configs/large_chi_b8.yaml 345
