#!/bin/sh
# Reviewer: independent-code queue part 2 (one process), starts after the MPS4 sigma run has 400 seeds
S=/c/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_order
cd /c/Users/seamu/Iliad-ML/TTN_regimes
I=data/review_order/indep
until [ -f $I/mps4_d2_chi4_sigma.jsonl ] && [ $(wc -l < $I/mps4_d2_chi4_sigma.jsonl) -ge 400 ]; do sleep 10; done
date > $I/queue2.txt
uv run python $S/indep_chain.py $I/binary4_d2_chi4_rho08.jsonl binary4 2 4 0 1000 0.8 0.01,0.003,0.001,0.0003,0.0001 0 > $I/log_b4a.txt 2>&1; date >> $I/queue2.txt
uv run python $S/indep_chain.py $I/binary4_d2_chi4_rho09.jsonl binary4 2 4 0 1000 0.9 0.01,0.0001 0 > $I/log_b4b.txt 2>&1; date >> $I/queue2.txt
CAP=0 uv run python $S/indep_chain.py $I/chiscan_mps4_chi32.jsonl mps4 2 32 0 500 - - 1 > $I/log_cs1.txt 2>&1; date >> $I/queue2.txt
CAP=0 uv run python $S/indep_chain.py $I/chiscan_mps8_chi4.jsonl mps8 2 4 0 150 - - 1 > $I/log_cs2.txt 2>&1; date >> $I/queue2.txt
echo done >> $I/queue2.txt
