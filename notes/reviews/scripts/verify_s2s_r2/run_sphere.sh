cd /c/Users/seamu/Iliad-ML/TTN_regimes
for tree in binary4 mps4; do
 for chi in 3 6; do
  for teacher in close separated; do
   for init in fanin balanced; do
     uv run python C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/verify_s2s_r2/sphere_T.py C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/verify_s2s_r2/sphere/${tree}_${chi}_${teacher}_${init}.csv $tree $chi $teacher $init 0 1 2 3 4 5 6 7 8 9 > C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/verify_s2s_r2/sphere/${tree}_${chi}_${teacher}_${init}.log 2>&1 &
   done
  done
  wait
 done
done
echo ALLDONE
