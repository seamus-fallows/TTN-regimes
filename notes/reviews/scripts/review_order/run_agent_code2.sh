#!/bin/sh
S=/c/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_order
cd /c/Users/seamu/Iliad-ML/TTN_regimes
O="--procs 3 --raw data/review_order/raw --res data/review_order/res"
L=data/review_order/logs
until grep -q done $L/queue.txt 2>/dev/null; do sleep 5; done
uv run python experiments/order/h1b.py $S/configs/h1b_saddle_mps4adj.yaml $O > $L/h1b_saddle_mps4adj.log 2>&1; date >> $L/queue.txt
uv run python experiments/order/h1b.py $S/configs/h1b_saddle_mps4sep.yaml $O > $L/h1b_saddle_mps4sep.log 2>&1; date >> $L/queue.txt
echo done2 >> $L/queue.txt
