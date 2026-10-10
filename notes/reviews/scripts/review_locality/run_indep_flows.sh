#!/usr/bin/env bash
# independent gradient-flow runs (reviewer's own implementation), at most 6 at once (xargs -P 3)
cd /c/Users/seamu/Iliad-ML/TTN_regimes
export SP=/c/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_locality
export O=data/review_locality/indep_flow
mkdir -p $O
printf '%s\n' "binary 4 12 small 0.01 0" "binary 4 12 small 0.001 0" "mps 4 12 saddle 0.001 0" \
  "binary 8 34 small 0.1 0" "mps 4 0123 small 0.01 0" \
  "mps 4 12 saddle 0.1 0" "mps 4 12 saddle 0.01 0" "mps 4 12 saddle 0.0001 0" \
  "mps 4 12 small 0.01 0" "mps 4 12 small 0.001 0" "mps 4 12 small 0.0001 0" |
xargs -P 3 -I{} bash -c 'tag=$(echo {} | tr " " "_"); timeout 600 uv run python $SP/indep.py flow {} > $O/$tag.json 2> $O/$tag.err'
echo flows done
