# SGD first plateau re-run (agent sgd_sweep.py; binary4 sigma=0.1 inits 0-2; NEW noise seeds 100-107; B = 1,4,16,64,256;
# eta = 5e-4; my config stops after the first event)
- Caveat (my config): stop_after=1 stops at the first event of ANY component; for near-tie seed 1 component 2 sometimes
  wins -> t1 missing in 2-5/8 runs; seed 1 excluded from comparisons below.
- shift1 = E t1 - t1_GF (mean +- se over 8 noise seeds), re-run vs committed (10 seeds; 20 at B=256):
  seed0: B1 +0.29+-0.15 (+0.20+-0.13); B4 +0.16+-0.07 (-0.04+-0.10); B16 +0.06+-0.04 (-0.02+-0.04); B64 +0.050+-0.019 (+0.004+-0.021); B256 +0.018+-0.010 (+0.005+-0.007)
  seed2: B1 -0.13+-0.18 (+0.20+-0.23); B4 +0.21+-0.08 (+0.04+-0.08); B16 +0.02+-0.04 (+0.07+-0.04); B64 +0.031+-0.010 (-0.010+-0.029); B256 -0.008+-0.011 (-0.007+-0.009)
  -> no significant shortening (most negative: -0.7 se); several small positive shifts.
- std(t1) seed0: 0.41/0.20/0.12/0.053/0.029 (committed 0.40/0.32/0.115/0.066/0.031); seed2: 0.51/0.23/0.12/0.029/0.032
  (0.71/0.26/0.12/0.093/0.042). analyze_sgd per-seed exponent of std(t1) vs eta/B: 0.514 +- 0.036 (2 seeds), committed 0.497 +- 0.018.
- |tr Q_e change| exponent 0.952 +- 0.035 (committed 0.971 +- 0.029; predicted 1).
# SGD m.s. limit, agent code (sgd_stability functions, B=1,16): see logs/sgd_stability_subset.log
