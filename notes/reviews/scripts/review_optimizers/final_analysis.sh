#!/bin/bash
cd c:/Users/seamu/Iliad-ML/TTN_regimes
OUT=data/review_optimizers
uv run python "C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_optimizers/compare.py" > $OUT/results/compare_all.txt 2>&1
uv run python "C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_optimizers/compare_eps.py" > $OUT/results/compare_eps.txt 2>&1
uv run python "C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_optimizers/compare_invariant.py" > $OUT/results/compare_invariant.txt 2>&1
uv run python "C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_optimizers/indep_hess.py" $OUT/raw/scaling_b4 $OUT/indep/hess_b4.csv > $OUT/results/indep_hess_b4.txt 2>&1
uv run python "C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_optimizers/indep_hess.py" $OUT/raw/implicit_bias_b4 $OUT/indep/hess_bias.csv > $OUT/results/indep_hess_bias.txt 2>&1
OPTIMIZERS_OUT=$OUT/raw uv run python -m experiments.optimizers.make_figures --results-dir $OUT/results eps b4 > $OUT/results/make_figures.txt 2>&1
echo finished
