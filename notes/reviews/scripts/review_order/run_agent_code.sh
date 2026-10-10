#!/bin/sh
# Reviewer: agent-code reruns (3 worker processes), outputs under data/review_order
S=/c/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_order
cd /c/Users/seamu/Iliad-ML/TTN_regimes
O="--procs 3 --raw data/review_order/raw --res data/review_order/res"
L=data/review_order/logs; mkdir -p $L
date > $L/queue.txt
uv run python experiments/order/sweep.py $S/configs/b4_sigma.yaml $O > $L/b4_sigma.log 2>&1; date >> $L/queue.txt
uv run python experiments/order/h1b.py $S/configs/h1b_saddle_b4.yaml $O > $L/h1b_saddle_b4.log 2>&1; date >> $L/queue.txt
uv run python experiments/order/h1b.py $S/configs/h1b_natural_b4.yaml $O > $L/h1b_natural_b4.log 2>&1; date >> $L/queue.txt
uv run python experiments/order/h1b.py $S/configs/h1b_saddle_mps4.yaml $O > $L/h1b_saddle_mps4.log 2>&1; date >> $L/queue.txt
uv run python experiments/order/multi.py $S/configs/multi.yaml $O > $L/multi.log 2>&1; date >> $L/queue.txt
echo done >> $L/queue.txt
