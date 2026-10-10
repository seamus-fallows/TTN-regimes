#!/bin/bash
# Reviewer: independent finite-sample check (lazy limit + small-init GD), one process, sequential.
cd /c/Users/seamu/Iliad-ML/TTN_regimes
D=data/review_optim_samples/repro
S=/c/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_optim_samples
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
for chi in 4 16; do for n in 128 192 256 384; do for seed in 0 1 2; do
  uv run python $S/indep_samples.py $D lazy $chi $n $seed 2>&1 | grep -v VIRTUAL
done; done; done
for n in 256 128 384; do for seed in 0 1; do
  uv run python $S/indep_samples.py $D gd 4 $n $seed 60000 2>&1 | grep -v VIRTUAL
done; done
