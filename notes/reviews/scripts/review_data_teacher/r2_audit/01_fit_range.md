# R2 audit note 1: global log-law fits vs asymptotic slopes (fit range)

All numbers recomputed from committed CSVs (results/*_runs*.csv). Local slope = per-seed least-squares
slope of gap vs ln(1/sigma) over the smallest sigma values only (theory_teachers.md section 9 protocol:
"slope from at least three values of sigma spaced by factors >= 2"); mean +- SE over seeds.

## Random init (r2_random_v2_runs.csv, mps4, f = 0.05; sigma = 3e-4, 1e-3, 3e-3)
| teacher | agent global fit (seeds) | local slope (5 seeds unless noted) | R6 | 1/|lambda_min| (no head start: upper bound) |
|---|---|---|---|---|
| misaligned_20 | 58.85 +- 0.86 (5) | 52.46 +- 0.10 | - | 52.94 |
| nonorth_c0.2 major | 36.29 +- 0.41 (4) | 33.95 +- 0.18 (4) | 34.14 | 35.14 |
| nonorth_c0.45 | 5.65 +- 0.08 (4) | 5.674 +- 0.059 | 5.682 | 6.627 |
| randbin_302 | 12.13 +- 0.28 (5) | 12.005 +- 0.071 | 12.014 | 13.094 |
| randbin_303 | 1.03 (1 seed) | 1.038 +- 0.005 | 1.051 | 2.779 |
| randmps_301 | 1.385 +- 0.002 (3) | 1.401 +- 0.013 | 1.386 | 2.658 |
Global fits of misaligned_20 and c = 0.2 exceed the bound A <= 1/gamma (R6: A = (1-kappa)/gamma): not asymptotic.

## Round-1 re-analyses (e4a_sigma_v2, e2_sigma_v2, e4b_ttn_v2; 3 smallest sigma)
- e4a c = 0.2: local 34.40 +- 0.20 (9 seeds) vs R6 34.14 (+0.8%); agent 36.53 +- 0.82 (+7%).
- e4a c = 0.45: local 5.686 +- 0.016 (10) vs 5.682; agent 5.764 +- 0.035.
- e4a c = 0.05: local 594.9 +- 11.0 (8) vs 569.9 (+4.4%); agent 527 +- 50.
- e2 theta = 20: local 12.14 +- 0.03; plateau 1/|lambda_min| = 12.31 (bound); agent 12.61 +- 0.12 (> bound).
- e2 theta = 5: local 194.9 +- 1.2; bound 193.2; agent 250.6 +- 3.9 (+30% above bound).
- e4b: 101 2.500 (2.488), 103 0.854 (0.868), 105 3.679 (3.657); 102/104 mix edges (see note 3).

## Saddle start (r2_saddle_runs_all.csv, mps, f = 0.05, per seed, eps <= 1e-4 vs 1e-3..5e-2)
Deviation of local slope from 1/|lambda_min|: <=1e-4: misaligned8_45 +0.16%, misaligned_20 +0.18%,
nonorth8_c0.45 -0.05% (agent: +11%, "slow crossover, still +11% at eps = 1e-8": wrong), nonorth8_c0.7
+0.24%, nonorth_c0.2 +0.05%, nonorth_c0.45 +0.22%, randbin8_302 +0.36%, randbin8_303 +2.98%,
randbin_302 +0.18%, randbin_303 +0.27%, randmps8_303 +1.87%, randmps_301 +0.15%.
Large-eps window (1e-3..5e-2): nonorth8_c0.45 +43%: the global fit is pulled by the crossover.
Binary/HT local p over eps 1e-5..1e-3: 0.9998-1.0009 in all 24 cases (global 1.003-1.013, significantly
> 1 per seed); over 1e-3..5e-2: 1.013-1.034. Random-init binary4 local p (sigma 1e-4..1e-3): 1.001-1.005
(global 1.025-1.062).
