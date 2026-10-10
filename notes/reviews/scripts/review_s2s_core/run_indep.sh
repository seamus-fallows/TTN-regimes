#!/bin/bash
# Reviewer's independent checks, sequential (one process).
cd /c/Users/seamu/Iliad-ML/TTN_regimes
S=/c/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_s2s_core
O=data/review_s2s_core/indep
export PYTHONDONTWRITEBYTECODE=1
run() { echo "=== $*"; uv run python $S/indep.py "$@" 2>&1 | grep -v VIRTUAL_ENV; }
run corr binary4 3 0,1,2,3,4,5,6,7,8,9 $O/corr_binary4.json
run corr mps4 3 0,1,2,3,4,5,6,7,8,9 $O/corr_mps4.json
run full binary4 3 0 0.01 $O/full_binary4_s0.01_seed0.json fanin
run full mps4 3 0 0.01 $O/full_mps4_s0.01_seed0.json fanin
run full mps4 3 3 0.01 $O/full_mps4_s0.01_seed3_fanin.json fanin 600
run full mps4 3 3 0.01 $O/full_mps4_s0.01_seed3_balanced.json balanced
run first binary4 3 0 0.01,0.003,0.002 $O/first_binary4_seed0.json
run first mps4 3 2 0.01,0.002 $O/first_mps4_seed2.json
echo ALLDONE
