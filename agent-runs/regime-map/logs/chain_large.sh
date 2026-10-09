#!/bin/bash
# when the N = 4 pool has finished: the centred sigma x K sweep (priority) and the large-tree slices
cd /root/work/TTN-regimes
while pgrep -f "configs/[c]hi_b4.yaml" > /dev/null; do sleep 30; done
C=experiments/regime_map/configs
nohup uv run python experiments/regime_map/sweep.py $C/sigma_K_centred_d2.yaml --n-proc 100 > agent-runs/regime-map/logs/sweep_centred_K.log 2>&1 &
uv run python experiments/regime_map/sweep.py $C/sigma_K_large.yaml $C/sigma_K_d2_large.yaml --n-proc 40
