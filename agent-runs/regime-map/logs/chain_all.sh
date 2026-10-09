#!/bin/bash
# start the production batch when the running sweeps (N = 4 sigma x chi, stall tests) have finished
cd /root/work/TTN-regimes
while pgrep -f "regime_map/[s]weep.py" > /dev/null; do sleep 30; done
C=experiments/regime_map/configs
uv run python experiments/regime_map/sweep.py $C/sigma_chi_b8.yaml $C/sigma_K.yaml $C/chi_b4.yaml $C/imbalance_b4.yaml $C/large_chi_b4.yaml $C/large_chi_b8.yaml --n-proc 350
