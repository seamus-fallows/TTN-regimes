#!/usr/bin/env bash
# agent's analysis on the rerun + stall check + reviewer's bond check; <= 6 busy processes
cd /c/Users/seamu/Iliad-ML/TTN_regimes
SP=/c/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_locality
R=data/review_locality
export LOCALITY_TRAJ=$(pwd)/$R/traj
(uv run python $R/code/make_figures.py $R/out/main --out $R/figs/main > $R/figs/main_make_figures.log 2>&1; echo mf $?) &
(uv run python $R/code/growth.py $R/out/main --out $R/figs/growth --workers 1 > $R/figs/growth.log 2>&1; echo growth $?) &
(uv run python $SP/my_bonds.py $R/out/main $R/figs/my_bonds.json > $R/figs/my_bonds.log 2>&1; echo bonds $?) &
(uv run python $SP/stall_check.py mps 4 03 0.2 1 240 > $R/out/stallcheck_s1.log 2>&1; uv run python $SP/stall_check.py mps 4 03 0.2 0 120 > $R/out/stallcheck_s0_control.log 2>&1; echo stall1 $?) &
(uv run python $SP/stall_check.py mps 4 03 0.1099 5 240 > $R/out/stallcheck_s5.log 2>&1; echo stall5 $?) &
wait
echo batch_a done
