#!/bin/sh
# h1b reruns with ORDER_RAW/ORDER_RES env vars (spawned workers do not see --raw/--res)
S=/c/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_order
cd /c/Users/seamu/Iliad-ML/TTN_regimes
export ORDER_RAW=data/review_order/raw ORDER_RES=data/review_order/res
L=data/review_order/logs
until grep -q done $L/queue.txt 2>/dev/null; do sleep 3; done
echo "h1b env-var queue start $(date)" > $L/queue3.txt
for c in h1b_saddle_b4 h1b_natural_b4 h1b_saddle_mps4adj h1b_saddle_mps4sep h1b_saddle_mps4; do
  uv run python experiments/order/h1b.py $S/configs/$c.yaml --procs 3 > $L/${c}_env.log 2>&1; echo "$c $(date)" >> $L/queue3.txt
done
echo done3 >> $L/queue3.txt
