#!/usr/bin/env bash
cd /c/Users/seamu/Iliad-ML/TTN_regimes
SP=/c/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_locality
R=data/review_locality
date
bash $SP/batch_hess.sh
date
(uv run python $R/code/levels.py $R/out/main --out $R/figs/levels --workers 2 > $R/figs/levels.log 2>&1; echo levels $?) &
(uv run python $SP/my_levels.py $R/out/main $R/figs/my_levels.json > $R/figs/my_levels.log 2>&1; echo mylevels $?) &
wait
date
bash $SP/run_indep_flows.sh
date
echo batch_b done
