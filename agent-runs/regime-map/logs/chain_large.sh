#!/bin/bash
# when the N = 4 pool has finished: the large-tree K slices (low priority)
cd /root/work/TTN-regimes
while pgrep -f "configs/[c]hi_b4.yaml" > /dev/null; do sleep 30; done
C=experiments/regime_map/configs
uv run python experiments/regime_map/sweep.py $C/sigma_K_large.yaml $C/sigma_K_d2_large.yaml --n-proc 60
