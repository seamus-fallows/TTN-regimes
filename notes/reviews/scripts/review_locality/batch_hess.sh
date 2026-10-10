#!/usr/bin/env bash
cd /c/Users/seamu/Iliad-ML/TTN_regimes
export SP=/c/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_locality
export R=data/review_locality
cat $R/out/main.part* > $R/out/main_all.jsonl
mkdir -p $R/indep_hess
echo '{"kind":"mps","n":4,"S":[1,2],"start":"saddle","sigma":0.0001}' > $R/indep_hess/f1.json
echo '{"kind":"mps","n":4,"S":[1,2],"start":"small","sigma":0.0001}' > $R/indep_hess/f2.json
echo '{"kind":"binary","n":4,"S":[1,2],"start":"small","seed":0}' > $R/indep_hess/f3.json
echo '{"kind":"binary","n":8,"S":[3,4],"start":"small","seed":0}' > $R/indep_hess/f4.json
echo '{"kind":"mps","n":4,"S":[0,1,2,3],"start":"small","seed":0}' > $R/indep_hess/f5.json
echo '{"kind":"mps","n":8,"S":[3,4],"start":"saddle","sigma":0.0001}' > $R/indep_hess/f6.json
printf '%s\n' 1 2 3 4 5 6 | xargs -P 4 -I{} bash -c 'timeout 900 uv run python $SP/indep.py hess $R/out/main_all.jsonl @$R/indep_hess/f{}.json > $R/indep_hess/h{}.jsonl 2> $R/indep_hess/h{}.err'
echo hess done
