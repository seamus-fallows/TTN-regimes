# s2s-core claims audit: working notes (part 2)

4. Scope: mps8 0/220 runs learn component 3 (main 0/174 incl chi6, k78 0/60, large 0/50); 2-event mps8 runs
   do show plateau-at-partial-fit (level err <= 1.3e-6) and rho >> 3, but no gap exponent fitted.
   K=7 gaps: chi=6 only, sigma 0.07-0.3 (0.63 decade, below the >=1-decade reporting rule).
   binary8 local exponents per sigma pair stable (median 4.94-5.04, per-seed 4.67-5.14) -> solid.
   ht4 per-seed local exponents 2.3-6.6; gap2 mean 4.75+-0.14 -> weak.
5. Trap fractions: main/large classes change with sigma for the same seed (censoring by budget);
   k78 (0.15-0.3) per-seed constant except mps8 seeds 1,5,7. mps4 close: seed 1 trapped all sigma,
   seed 7 trapped sigma<=0.05 only, seed 5 stalled all sigma (14 trapped + 8 stalled / 80).
   mps8 stalled min slice median 1e-6.5 sigma (k78 fanin), 1e-6 sigma (large) vs trapped threshold 1e-8 sigma;
   balanced stalled ~1e-2 sigma.
6. Balanced 0/210: 120 runs in cells where fan-in never traps (binary4 chi3/6, mps4 chi6); informative 0/120
   (mps4 chi3: 0/30 vs fan-in 12/30 at same sigmas; k78 0/90 vs 47/90). k78 balanced: 69/90 stalled.
   chi=6 bonds capped at d=4 on MPS ends / HT leaves (TTNSpec.uniform cap), so 'chi=2R' is literal only on binary trees.
   ht4 chi6 trapped 5/50 = seed 9 at all 5 sigmas (notes say 2/50; README says chi=2R avoids it on HT4).
7. Escape drops: >100x drops at chi=6 K=7,8: 1.4% binary8, 2% ht4, 0.7% mps8 (claimed 0%); mps4 chi3 sep 0.6%.
   Later escapes K=7,8 chi=3: medians 0.07-0.30, >100x in 7.8-10.7% (claimed 'same picture').
8. Edge check: CSV gives 497/531 complete runs with window spectra (332 complete runs lack them); notes 517/551.
9. Drift > 1e-6: 11 runs (k78 4, large 6, main 1), mostly mps8. Drift is per-chunk (relative to chunk start).
10. README omissions: large/k78/first_k78/lo_d/lo_K/continue_stalled/recompute_hessians not in How-to-run;
    'results/balanced_*.csv' does not exist; budget section still describes amplitude roundoff stop.
