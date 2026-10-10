# Real data, fix round 1: re-analysis (`fix1_analysis.py`; committed outputs, local reruns, full raw sweep)

## First route (k = 3): clean series

Series = one seed whose first route (one sibling pair) rises inside drop 1 at every sigma of the series (events_table.json). Gap = drop 1 - drop 0 (tau clock). All laws fitted with relative residuals; 'unw.' = the round-0 unweighted log-law fit.

| tree | seed | edges | sigma (n) | power p (rms) | log law rms (unw.) | A / sigma + B: A, B (rms) | B/A | p on 0.05/0.1/0.2/0.5 |
|---|---|---|---|---|---|---|---|---|
| binary:quadtree | 0 | [3, 4] | 0.05-0.3 (6) | 1.057 (0.002) | 0.159 (0.284) | 14.9, -5.4 (0.011) | -0.36 | 1.060 |
| binary:quadtree | 1 | [0, 1] | 0.1-0.5 (5) | 1.360 (0.022) | 0.186 (0.415) | 6.1, -5.9 (0.040) | -0.95 | 1.369 |
| binary:quadtree | 2 | [3, 4] | 0.05-0.2 (5) | 1.057 (0.008) | 0.109 (0.154) | 36.7, -17.0 (0.012) | -0.46 | 1.051 |
| binary:quadtree | 3 | [3, 4] | 0.05-0.2 (5) | 1.236 (0.018) | 0.122 (0.199) | 12.2, -20.7 (0.011) | -1.69 | 1.238 |
| binary:scrambled0 | 0 | [10, 11] | 0.05-0.5 (7) | 1.303 (0.055) | 0.296 (1.103) | 38.9, -40.9 (0.046) | -1.05 | 1.309 |
| binary:scrambled0 | 4 | [10, 11] | 0.05-0.3 (6) | 1.156 (0.026) | 0.164 (0.330) | 34.5, -32.8 (0.003) | -0.95 | 1.122 |
| binary:scrambled1 | 0 | [10, 11] | 0.05-0.5 (7) | 1.325 (0.058) | 0.296 (1.199) | 32.8, -36.0 (0.047) | -1.10 | 1.339 |
| binary:scrambled1 | 1 | [10, 11] | 0.05-0.5 (7) | 1.277 (0.026) | 0.301 (1.064) | 38.1, -36.4 (0.060) | -0.96 | 1.278 |
| binary:scrambled1 | 2 | [10, 11] | 0.05-0.3 (6) | 1.244 (0.018) | 0.195 (0.430) | 27.5, -35.5 (0.029) | -1.29 | 1.221 |
| binary:scrambled1 | 3 | [10, 11] | 0.05-0.5 (7) | 1.227 (0.017) | 0.284 (0.942) | 42.3, -35.1 (0.049) | -0.83 | 1.231 |
| binary:scrambled2 | 3 | [3, 4] | 0.05-0.5 (4) | 1.133 (0.012) | 0.263 (0.784) | 9.7, -5.0 (0.033) | -0.52 | 1.133 |

| tree | series | power p (mean ± SE) | range | log rms range (unw.) | max rms power / A/sigma+B | B < 0 |
|---|---|---|---|---|---|---|
| binary:quadtree | 4 | 1.178 ± 0.074 (n = 4) | 1.06-1.36 | 0.11-0.19 (0.15-0.42) | 0.022 / 0.040 | 4/4 |
| binary:scrambled0 | 2 | 1.229 ± 0.074 (n = 2) | 1.16-1.30 | 0.16-0.30 (0.33-1.10) | 0.055 / 0.046 | 2/2 |
| binary:scrambled1 | 4 | 1.268 ± 0.022 (n = 4) | 1.23-1.32 | 0.19-0.30 (0.43-1.20) | 0.058 / 0.060 | 4/4 |
| binary:scrambled2 | 1 | 1.133 (n = 1) | 1.13-1.13 | 0.26-0.26 (0.78-0.78) | 0.012 / 0.033 | 1/1 |
| all | 11 | 1.216 ± 0.031 (n = 11) | 1.06-1.36 | 0.11-0.30 (0.15-1.20) | 0.058 / 0.060 | 11/11 |

## All events_table series refitted (both laws with relative residuals)

| tree | features | seed | drop # | signature | k | sigma (n) | power p (rms) | log slope (rms; unw. rms) | preferred |
|---|---|---|---|---|---|---|---|---|---|
| binary:quadtree | raw | 0 | [1] | [[3, 2], [4, 2]] | 3 | 0.05-0.3 (6) | 1.06 (0.002) | 99.5 (0.159; 0.28) | power |
| binary:quadtree | raw | 0 | [2] | [[3, 3], [4, 3]] | 3 | 0.05-0.2 (5) | 0.33 (0.008) | 59.7 (0.007; 0.01) | log |
| binary:quadtree | raw | 1 | [1] | [[0, 2], [1, 2]] | 3 | 0.1-0.5 (5) | 1.36 (0.022) | 21.9 (0.186; 0.42) | power |
| binary:quadtree | raw | 2 | [1] | [[3, 2], [4, 2]] | 3 | 0.05-0.2 (5) | 1.06 (0.008) | 325 (0.109; 0.15) | power |
| binary:quadtree | raw | 2 | [2] | [[3, 3], [4, 3]] | 3 | 0.05-0.2 (5) | 1.11 (0.122) | 101 (0.019; 0.02) | log |
| binary:quadtree | raw | 3 | [1] | [[3, 2], [4, 2]] | 3 | 0.05-0.2 (5) | 1.24 (0.018) | 105 (0.122; 0.20) | power |
| binary:quadtree | raw | 3 | [2] | [[3, 3], [4, 3]] | 3 | 0.05-0.5 (6) | 0.56 (0.122) | 72 (0.037; 0.05) | log |
| binary:quadtree | raw | 4 | [2, 3] | [[3, 3], [4, 3]] | 3 | 0.05-0.2 (5) | -0.34 (0.513) | 38.9 (0.352; 0.70) | log |
| binary:quadtree | white | 0 | [2, 3] | [[2, 2], [5, 2]] | 3 | 0.05-0.2 (4) | 1.62 (0.088) | 1.21e+03 (0.191; 0.35) | power |
| binary:quadtree | white | 0 | [1, 2] | [[3, 3], [4, 3]] | 3 | 0.05-0.2 (4) | 0.47 (0.430) | 16.7 (0.385; 0.50) | log |
| binary:quadtree | white | 1 | [2, 3] | [[2, 2], [5, 2]] | 3 | 0.05-0.2 (4) | 2.03 (0.132) | 746 (0.154; 0.58) | power |
| binary:quadtree | white | 2 | [3] | [[2, 2], [5, 2]] | 3 | 0.05-0.2 (4) | 1.44 (0.012) | 698 (0.167; 0.32) | power |
| binary:quadtree | white | 2 | [4] | [[2, 3], [5, 3]] | 3 | 0.05-0.2 (4) | 1.27 (0.016) | 997 (0.116; 0.20) | power |
| binary:quadtree | white | 2 | [1] | [[3, 2], [4, 2]] | 3 | 0.05-0.2 (4) | 0.97 (0.009) | 45.2 (0.091; 0.12) | power |
| binary:quadtree | white | 2 | [2] | [[3, 3], [4, 3]] | 3 | 0.05-0.2 (4) | 0.93 (0.008) | 75 (0.075; 0.10) | power |
| binary:quadtree | white | 3 | [1, 2] | [[0, 2], [1, 2]] | 3 | 0.05-0.2 (4) | 0.38 (0.243) | 30.7 (0.229; 0.26) | log |
| binary:quadtree | white | 4 | [1, 2] | [[3, 3], [4, 3]] | 3 | 0.05-0.2 (4) | 0.83 (0.120) | 80.4 (0.173; 0.22) | power |
| binary:scrambled0 | raw | 0 | [2, 3] | [[7, 2]] | 2 | 0.1-0.5 (5) | -1.50 (0.753) | -68.4 (0.513; 1.90) | log |
| binary:scrambled0 | raw | 0 | [1] | [[10, 2], [11, 2]] | 3 | 0.05-0.5 (7) | 1.30 (0.055) | 158 (0.296; 1.10) | power |
| binary:scrambled0 | raw | 1 | [1, 2] | [[2, 2], [3, 2], [4, 2], [5, 2]] | 5 | 0.07-0.5 (4) | 0.79 (0.233) | 835 (0.150; 0.22) | log |
| binary:scrambled0 | raw | 1 | [2] | [[2, 3], [5, 3]] | 3 | 0.05-0.3 (5) | 0.74 (0.096) | 988 (0.170; 0.23) | power |
| binary:scrambled0 | raw | 2 | [2, 3] | [[2, 3], [5, 3]] | 3 | 0.07-0.3 (5) | 1.45 (0.289) | 1.91e+03 (0.053; 0.10) | log |
| binary:scrambled0 | raw | 2 | [1, 2] | [[3, 2]] | 2 | 0.07-0.3 (5) | 1.43 (2.930) | 818 (0.707; 5.41) | log |
| binary:scrambled0 | raw | 4 | [1, 2] | [[7, 2], [8, 2], [9, 2], [12, 2]] | 5 | 0.14-0.5 (4) | 2.22 (0.098) | 431 (0.361; 1.22) | power |
| binary:scrambled0 | raw | 4 | [1] | [[10, 2], [11, 2]] | 3 | 0.05-0.3 (6) | 1.16 (0.026) | 226 (0.164; 0.33) | power |
| binary:scrambled0 | white | 0 | [2, 3] | [[2, 2], [5, 2]] | 3 | 0.05-0.2 (4) | 1.18 (0.017) | 1.45e+03 (0.101; 0.16) | power |
| binary:scrambled0 | white | 0 | [1, 2] | [[7, 2], [8, 2]] | 3 | 0.05-0.2 (4) | 0.22 (0.052) | 37.4 (0.057; 0.06) | power |
| binary:scrambled0 | white | 1 | [1, 3] | [[2, 2], [5, 2]] | 3 | 0.05-0.2 (4) | 0.65 (0.084) | 173 (0.121; 0.14) | power |
| binary:scrambled1 | raw | 0 | [1] | [[10, 2], [11, 2]] | 3 | 0.05-0.5 (7) | 1.32 (0.058) | 133 (0.296; 1.20) | power |
| binary:scrambled1 | raw | 1 | [2, 3] | [[7, 2], [8, 2], [9, 2], [12, 2]] | 5 | 0.1-0.5 (4) | -0.82 (0.650) | -34.6 (0.409; 1.00) | log |
| binary:scrambled1 | raw | 1 | [1] | [[10, 2], [11, 2]] | 3 | 0.05-0.5 (7) | 1.28 (0.026) | 155 (0.301; 1.06) | power |
| binary:scrambled1 | raw | 2 | [1] | [[10, 2], [11, 2]] | 3 | 0.05-0.3 (6) | 1.24 (0.018) | 173 (0.195; 0.43) | power |
| binary:scrambled1 | raw | 3 | [1] | [[10, 2], [11, 2]] | 3 | 0.05-0.5 (7) | 1.23 (0.017) | 177 (0.284; 0.94) | power |
| binary:scrambled2 | raw | 3 | [1] | [[3, 2], [4, 2]] | 3 | 0.05-0.5 (4) | 1.13 (0.012) | 43.4 (0.263; 0.78) | power |
| mps:snake | raw | 0 | [3, 4, 5, 6] | [[6, 3], [7, 2], [8, 2], [9, 2]] | 5 | 0.05-0.2 (5) | 0.18 (0.263) | 305 (0.244; 0.28) | log |
| mps:snake | raw | 4 | [1] | [[4, 2]] | 2 | 0.05-0.2 (5) | 0.27 (0.004) | 12.1 (0.005; 0.01) | power |

## First route over all binary runs (window-checked attribution, full main sweep)

| tree | seeds fitted | power p (mean ± SE) | max rms power / A/sigma+B | log rms range | B < 0 |
|---|---|---|---|---|---|
| binary_quadtree/raw | 3 | 1.105 ± 0.066 (n = 3) | 0.029 / 0.033 | 0.12-0.24 | 3/3 |
| binary_scrambled0/raw | 2 | 1.229 ± 0.074 (n = 2) | 0.055 / 0.046 | 0.16-0.30 | 2/2 |
| binary_scrambled1/raw | 4 | 1.268 ± 0.022 (n = 4) | 0.058 / 0.060 | 0.19-0.30 | 4/4 |
| binary_scrambled2/raw | 1 | 1.133 (n = 1) | 0.012 / 0.033 | 0.26-0.26 | 1/1 |

All fitted seeds: power p 1.198 ± 0.032 (n = 10), range 1.02-1.32.

Excluded (seed, sigma, reason): binary_quadtree/raw: [[1, 0.05, 'between drops'], [1, 0.07, 'between drops'], [1, 0.5, 'merged'], [2, 0.5, 'other route'], [3, 0.3, 'between drops'], [3, 0.5, 'merged'], [4, 0.2, 'other route'], [4, 0.3, 'other route'], [4, 0.5, 'other route']]; binary_scrambled0/raw: [[1, 0.05, 'between drops'], [1, 0.07, 'between drops'], [1, 0.1, 'between drops'], [1, 0.14, 'between drops'], [1, 0.3, 'between drops'], [1, 0.5, 'between drops'], [2, 0.2, 'between drops'], [2, 0.3, 'between drops'], [2, 0.5, 'other route'], [3, 0.14, 'other route'], [3, 0.2, 'other route'], [3, 0.3, 'other route'], [3, 0.5, 'other route'], [4, 0.5, 'between drops']]; binary_scrambled1/raw: [[2, 0.5, 'other route'], [4, 0.14, 'between drops'], [4, 0.2, 'other route'], [4, 0.5, 'other route']]; binary_scrambled2/raw: [[0, 0.5, 'between drops'], [1, 0.2, 'between drops'], [1, 0.5, 'other route'], [2, 0.2, 'other route'], [2, 0.5, 'other route'], [4, 0.5, 'between drops']]

## Final train loss per seed, main sweep (round-0 orders and the MPS)

| sigma | scrambled runs ending below the quadtree of the same seed | MPS below quadtree |
|---|---|---|
| 0.05 | 1/15 | 5/5 |
| 0.07 | 0/10 | 5/5 |
| 0.1 | 0/15 | 5/5 |
| 0.14 | 0/10 | 5/5 |
| 0.2 | 3/15 | 5/5 |
| 0.3 | 4/9 | 5/5 |
| 0.5 | 14/15 | 3/5 |

## MPS per-edge laws over the full horizon (main sweep, relative residuals for both laws)

First rise at tau = 0: 10/35; resolved: single edge 25/25 (edges [(5,)]), before the first drop's peak 24.

| edge | seeds | power p (mean ± SE) | power rms | log slope | log rms | seeds preferring log |
|---|---|---|---|---|---|---|
| 0 | 5 | 0.94 ± 0.05 (n = 5) | 0.101 | 1.13e+03 ± 1.8e+02 | 0.125 | 2/5 |
| 1 | 5 | 1.27 ± 0.08 (n = 5) | 0.105 | 156 ± 15 | 0.276 | 0/5 |
| 2 | 5 | 1.00 ± 0.08 (n = 5) | 0.131 | 94.2 ± 7.6 | 0.141 | 1/5 |
| 3 | 5 | 0.83 ± 0.19 (n = 5) | 0.252 | 30.3 ± 0.77 | 0.029 | 5/5 |
| 4 | 5 | 0.90 ± 0.23 (n = 5) | 0.347 | 14.7 ± 1.5 | 0.028 | 4/5 |
| 6 | 4 | 0.90 ± 0.09 (n = 4) | 0.166 | 411 ± 24 | 0.067 | 4/4 |
| 7 | 5 | 1.21 ± 0.15 (n = 5) | 0.362 | 1.89e+03 ± 1e+02 | 0.108 | 5/5 |
| 8 | 5 | 1.14 ± 0.12 (n = 5) | 0.296 | 1.92e+03 ± 1.1e+02 | 0.102 | 5/5 |
| 9 | 5 | 1.12 ± 0.11 (n = 5) | 0.271 | 1.91e+03 ± 1.1e+02 | 0.104 | 5/5 |

## Route-amplitude rule, rank-1 states, critical-point units

States: 124, dropped (rank1_rel_err > 0.05): 28. Hits per kept state: quadtree 27/28, scrambled 64/68. Units: 26; outcomes {'hit': 22, 'mixed': 2, 'miss': 2} (chance per unit 0.25). The first pair is two edges crossing together in 96/96 kept states.

| ordering | seed | sigmas | plateau L | predicted pair | first pair | outcome |
|---|---|---|---|---|---|---|
| quadtree | 0 | 0.05, 0.07, 0.1, 0.14, 0.2 | 0.270 | [3, 4] | [3, 4] | hit |
| quadtree | 1 | 0.05, 0.07, 0.1, 0.14, 0.2, 0.3 | 0.349 | [0, 1] | [0, 1] | hit |
| quadtree | 2 | 0.05, 0.07, 0.1, 0.14, 0.2, 0.3 | 0.269 | [3, 4] | [3, 4] | hit |
| quadtree | 3 | 0.05, 0.07, 0.1, 0.14, 0.2 | 0.271 | [3, 4] | [3, 4] | hit |
| quadtree | 4 | 0.05, 0.07, 0.1 | 0.270 | [3, 4] | [3, 4] | hit |
| quadtree | 4 | 0.14, 0.2, 0.3 | 0.373 | [0, 1] | [3, 4] | mixed (2/3) |
| scrambled0 | 0 | 0.05, 0.07, 0.1, 0.14, 0.2, 0.3 | 0.323 | [10, 11] | [10, 11] | hit |
| scrambled0 | 1 | 0.05, 0.07 | 0.297 | [0, 1] | [0, 1] | hit |
| scrambled0 | 1 | 0.1, 0.14, 0.2 | 0.301 | [0, 1] | [0, 1] | hit |
| scrambled0 | 2 | 0.05, 0.07, 0.1, 0.14 | 0.297 | [0, 1] | [0, 1] | hit |
| scrambled0 | 2 | 0.2, 0.3 | 0.301 | [0, 1] | [0, 1] | hit |
| scrambled0 | 3 | 0.05, 0.07, 0.1 | 0.322 | [10, 11] | [10, 11] | hit |
| scrambled0 | 3 | 0.14, 0.2 | 0.322 | [3, 4] | [0, 1] | miss |
| scrambled0 | 4 | 0.05, 0.07, 0.1, 0.14, 0.2, 0.3 | 0.323 | [10, 11] | [10, 11] | hit |
| scrambled1 | 0 | 0.05, 0.07, 0.1, 0.14, 0.2, 0.3 | 0.295 | [10, 11] | [10, 11] | hit |
| scrambled1 | 1 | 0.05, 0.07, 0.1, 0.14, 0.2, 0.3 | 0.295 | [10, 11] | [10, 11] | hit |
| scrambled1 | 2 | 0.05, 0.07, 0.1, 0.14, 0.2 | 0.294 | [10, 11] | [10, 11] | hit |
| scrambled1 | 3 | 0.05, 0.07, 0.1, 0.14, 0.2 | 0.294 | [10, 11] | [10, 11] | hit |
| scrambled1 | 4 | 0.05, 0.07, 0.1 | 0.294 | [10, 11] | [10, 11] | hit |
| scrambled1 | 4 | 0.14, 0.2 | 0.323 | [7, 8] | [10, 11] | mixed (1/2) |
| scrambled2 | 0 | 0.05, 0.1, 0.2 | 0.249 | [10, 11] | [10, 11] | hit |
| scrambled2 | 1 | 0.05, 0.1 | 0.248 | [10, 11] | [10, 11] | hit |
| scrambled2 | 2 | 0.05 | 0.248 | [10, 11] | [10, 11] | hit |
| scrambled2 | 2 | 0.1 | 0.351 | [3, 4] | [10, 11] | miss |
| scrambled2 | 3 | 0.05, 0.1, 0.2 | 0.343 | [3, 4] | [3, 4] | hit |
| scrambled2 | 4 | 0.05, 0.1, 0.2 | 0.248 | [10, 11] | [10, 11] | hit |

## Share of sum lambda^2 on k <= 5 routes, one value per (order, seed), sigma = 0.05

Pair-count chance level: 0.43 (12 of 28 pairs have k <= 5).

| order | mean ± SE over seeds | values (seeds 0-4) |
|---|---|---|
| quadtree | 0.89 ± 0.01 (n = 5) | 0.911, 0.835, 0.907, 0.908, 0.908 |
| scrambled (3 orders) | 0.17 ± 0.07 (n = 15) | 0.027, 0.813, 0.815, 0.026, 0.027, 0.110, 0.107, 0.112, 0.107, 0.111, 0.012, 0.012, 0.012, 0.230, 0.011 |
| scrambled0 | 0.34 ± 0.19 (n = 5) | 0.027, 0.813, 0.815, 0.026, 0.027 |
| scrambled1 | 0.11 ± 0.00 (n = 5) | 0.110, 0.107, 0.112, 0.107, 0.111 |
| scrambled2 | 0.06 ± 0.04 (n = 5) | 0.012, 0.012, 0.012, 0.230, 0.011 |

## Locality per (order, seed), sigma = 0.1 (tau_end = 1e4)

| seed | order | source | k <= 5 share | L(tau_end) | test acc | tau(L<=0.25) | tau(L<=0.2) | tau(L<=0.15) | t_a |
|---|---|---|---|---|---|---|---|---|---|
| 0 | quadtree | fix1 | 0.92 | 0.0957 | 0.950 | 155.9 | 332.9 | 2156 | 1.1e+15 |
| 0 | scrambled8 | audit | 0.88 | 0.1192 | 0.940 | 90.64 | 3976 | 4031 | 1.19e+15 |
| 0 | rowmajor | audit | 0.93 | 0.1355 | 0.948 | 48.46 | 53.78 | 9788 | 2.74e+15 |
| 0 | scrambled4 | audit | 0.20 | 0.1429 | 0.918 | 109 | 4935 | 6014 | 6.31e+15 |
| 0 | scrambled7 | audit | 0.47 | 0.1483 | 0.928 | 9393 | 9521 | 9941 | 2.25e+15 |
| 0 | scrambled6 | audit | 0.02 | 0.1699 | 0.909 | 606 | 8443 | - | 6.86e+15 |
| 0 | scrambled1 | fix1 | 0.12 | 0.1744 | 0.897 | 7242 | 7316 | - | 1.25e+15 |
| 0 | scrambled3 | audit | 0.80 | 0.1866 | 0.896 | 63.67 | 100.3 | - | 1.37e+15 |
| 0 | scrambled0 | fix1 | 0.03 | 0.2025 | 0.889 | 8965 | - | - | 1.54e+15 |
| 0 | scrambled5 | audit | 0.17 | 0.2243 | 0.869 | 111.2 | - | - | 5.27e+14 |
| 0 | quadtreeT | audit | 0.11 | 0.2257 | 0.861 | 139.4 | - | - | 2.3e+15 |
| 0 | scrambled2 | fix1 | 0.01 | 0.2292 | 0.858 | 244.2 | - | - | 1.21e+15 |
| 1 | quadtree | fix1 | 0.78 | 0.0985 | 0.956 | 97.06 | 6027 | 6071 | 1.67e+15 |
| 1 | rowmajor | audit | 0.92 | 0.0991 | 0.953 | 86.62 | 91.6 | 4752 | 1.17e+15 |
| 1 | scrambled1 | fix1 | 0.12 | 0.1720 | 0.896 | 6559 | 6634 | - | 8.77e+14 |
| 1 | scrambled0 | fix1 | 0.07 | 0.1721 | 0.892 | 1527 | 1672 | - | 4.67e+14 |
| 1 | quadtreeT | audit | 0.13 | 0.1942 | 0.878 | 88.65 | 7126 | - | 1.99e+15 |
| 1 | scrambled2 | fix1 | 0.01 | 0.2292 | 0.859 | 284.9 | - | - | 2.04e+15 |
| 2 | quadtree | fix1 | 0.91 | 0.0968 | 0.953 | 363 | 468.9 | 3730 | 1.88e+14 |
| 2 | scrambled1 | fix1 | 0.13 | 0.1729 | 0.897 | 7341 | 7412 | - | 5.03e+14 |
| 2 | scrambled0 | fix1 | 0.81 | 0.1760 | 0.896 | 2856 | 2984 | - | 2.62e+14 |
| 2 | scrambled2 | fix1 | 0.04 | 0.2303 | 0.856 | 304.4 | - | - | 3.05e+14 |
| 3 | quadtree | fix1 | 0.91 | 0.0981 | 0.952 | 112.8 | 291.2 | 1841 | 2.06e+15 |
| 3 | scrambled0 | fix1 | 0.02 | 0.1988 | 0.887 | 4167 | 6729 | - | 1.65e+15 |
| 3 | scrambled2 | fix1 | 0.24 | 0.2097 | 0.870 | 147 | - | - | 9.76e+14 |
| 3 | scrambled1 | fix1 | 0.12 | 0.2588 | 0.814 | - | - | - | 1.02e+15 |
| 4 | quadtree | fix1 | 0.91 | 0.1031 | 0.958 | 121.5 | 236.3 | 6251 | 3.78e+14 |
| 4 | scrambled0 | fix1 | 0.03 | 0.1894 | 0.892 | 4045 | 7228 | - | 2.93e+14 |
| 4 | scrambled2 | fix1 | 0.01 | 0.2293 | 0.859 | 208.9 | - | - | 5e+14 |
| 4 | scrambled1 | fix1 | 0.13 | 0.2589 | 0.814 | - | - | - | 3.82e+14 |

spearman_all: share vs L(tau_end): rho = -0.72, p = 8.1e-06, n = 30

spearman_committed_orders: share vs L(tau_end): rho = -0.69, p = 0.00076, n = 20

spearman_seed0: share vs L(tau_end): rho = -0.77, p = 0.0034, n = 12

| order | n | L(tau_end) mean ± SE | values | median tau(L<=0.25) / (0.2) / (0.15) |
|---|---|---|---|---|
| quadtree | 5 | 0.0984 ± 0.0013 (n = 5) | 0.096, 0.099, 0.097, 0.098, 0.103 | 121.5 / 332.9 / 3730 |
| quadtreeT | 2 | 0.2099 ± 0.0158 (n = 2) | 0.226, 0.194 | 114 / 7126 / - |
| rowmajor | 2 | 0.1173 ± 0.0182 (n = 2) | 0.136, 0.099 | 67.54 / 72.69 / 7270 |
| scrambled0 | 5 | 0.1878 ± 0.0060 (n = 5) | 0.203, 0.172, 0.176, 0.199, 0.189 | 4045 / 4857 / - |
| scrambled1 | 5 | 0.2074 ± 0.0210 (n = 5) | 0.174, 0.172, 0.173, 0.259, 0.259 | 7242 / 7316 / - |
| scrambled2 | 5 | 0.2255 ± 0.0040 (n = 5) | 0.229, 0.229, 0.230, 0.210, 0.229 | 244.2 / - / - |
| scrambled3 | 1 | 0.1866 (n = 1) | 0.187 | 63.67 / 100.3 / - |
| scrambled4 | 1 | 0.1429 (n = 1) | 0.143 | 109 / 4935 / 6014 |
| scrambled5 | 1 | 0.2243 (n = 1) | 0.224 | 111.2 / - / - |
| scrambled6 | 1 | 0.1699 (n = 1) | 0.170 | 606 / 8443 / - |
| scrambled7 | 1 | 0.1483 (n = 1) | 0.148 | 9393 / 9521 / 9941 |
| scrambled8 | 1 | 0.1192 (n = 1) | 0.119 | 90.64 / 3976 / 4031 |

Other orders with a lower final loss than the quadtree / reaching L <= 0.25 first: seed 0: 0/11 lower, 6/11 earlier; seed 1: 0/5 lower, 2/5 earlier; seed 2: 0/3 lower, 1/3 earlier; seed 3: 0/3 lower, 0/3 earlier; seed 4: 0/3 lower, 0/3 earlier

## Hessian at the first plateau: per-seed exponent of |lambda_min| vs sigma

| tree | plateau band | seed | exponent | sigmas (decades) | n_neg (projected) |
|---|---|---|---|---|---|
| binary | 0.265-0.275 | 0 | 0.871 | 0.05, 0.07, 0.1, 0.14, 0.2, 0.3 (0.78) | [1, 1, 1, 1, 3, 5] |
| binary | 0.265-0.275 | 2 | 0.888 | 0.05, 0.07, 0.1, 0.14, 0.2, 0.3 (0.78) | [0, 1, 1, 2, 4, 5] |
| binary | 0.265-0.275 | 3 | 0.873 | 0.05, 0.07, 0.1, 0.14, 0.2 (0.60) | [1, 1, 1, 2, 4] |
| binary | 0.265-0.275 | 4 | 1.149 | 0.05, 0.07, 0.1 (0.30) | [1, 1, 2] |
| binary | | mean ± SE | 0.945 ± 0.068 (n = 4) | | |
| mps | 0.225-0.235 | 0 | 0.045 | 0.05, 0.07, 0.1, 0.14, 0.2, 0.3 (0.78) | [2, 2, 3, 3, 3, 6] |
| mps | 0.225-0.235 | 1 | 0.044 | 0.05, 0.07, 0.1, 0.14 (0.45) | [2, 2, 2, 3] |
| mps | 0.225-0.235 | 2 | -0.009 | 0.05, 0.07, 0.1, 0.14, 0.2, 0.3 (0.78) | [2, 2, 3, 3, 3, 4] |
| mps | 0.225-0.235 | 3 | 0.078 | 0.05, 0.07, 0.1 (0.30) | [2, 2, 3] |
| mps | 0.225-0.235 | 4 | 0.078 | 0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.5 (1.00) | [3, 3, 3, 3, 4, 6, 10] |
| mps | | mean ± SE | 0.047 ± 0.016 (n = 5) | | |

## MPS first rank increase vs the first drop

35 runs: 10 recorded at tau = 0 (s2/s1 >= 0.05 from the start of phase 2, gap < -5); 25 resolved, 24 of them before the first drop's rate peak (gap -1.97 to 0.01).
Local reruns: seed 0 sigma 0.1: [5] at_start=False gap -1.57; seed 1 sigma 0.1: [5] at_start=False gap -0.55; seed 2 sigma 0.1: [5] at_start=False gap -1.16; seed 3 sigma 0.1: [5] at_start=False gap -1.30; seed 4 sigma 0.1: [5] at_start=True gap -10.08

## Regime map checks

Lazy floor L_lin(inf) per seed: 0: 0.1276, 1: 0.1119, 2: 0.0962, 3: 0.0902, 4: 0.0990 (mean 0.105).

| seed | lazy floor | plain runs sigma 0.1-0.3: L(1e4) range | below floor | rich sigma 1.4: L(T) |
|---|---|---|---|---|
| 0 | 0.1276 | 0.0953-0.0957 | 4/4 | 0.0797 |
| 1 | 0.1119 | 0.0953-0.0985 | 4/4 | 0.0791 |
| 2 | 0.0962 | 0.0953-0.0968 | 3/4 | 0.0892 |
| 3 | 0.0902 | 0.0954-0.0981 | 0/4 | 0.0779 |
| 4 | 0.0990 | 0.0953-0.1031 | 3/4 | 0.0904 |

Lazy-labelled runs (14): L(T) = 0.130-0.192, L(T) - floor = 0.037-0.064: not converged at the horizon.
sigma = 1.4 runs: seed 0: 1 drops, delta_K 827, L(T) 0.0797 vs floor 0.1276, T 3000; seed 1: 1 drops, delta_K 562, L(T) 0.0791 vs floor 0.1119, T 3000; seed 2: 0 drops, delta_K 23.5, L(T) 0.0892 vs floor 0.0962, T 405.5; seed 3: 1 drops, delta_K 1.14e+03, L(T) 0.0779 vs floor 0.0902, T 3000; seed 4: 0 drops, delta_K 13, L(T) 0.0904 vs floor 0.0990, T 416.6

Label counts per sigma under each threshold pair (dK = delta_K threshold, rho = separation threshold):

- dK0.05_rho2.0: 0.05|S2S 5, 0.07|S2S 5, 0.1|S2S 4, 0.1|rich 1, 0.14|S2S 4, 0.14|rich 1, 0.2|S2S 5, 0.3|S2S 5, 0.5|S2S 4, 0.5|rich 1, 0.7|S2S 4, 0.7|rich 1, 1|S2S 4, 1|rich 1, 1.4|rich 5, 2|lazy 2, 2|rich 3, 2.8|lazy 5, 4|lazy 5
- dK0.05_rho3.0: 0.05|S2S 5, 0.07|S2S 5, 0.1|S2S 4, 0.1|rich 1, 0.14|S2S 4, 0.14|rich 1, 0.2|S2S 4, 0.2|rich 1, 0.3|S2S 5, 0.5|S2S 2, 0.5|rich 3, 0.7|S2S 3, 0.7|rich 2, 1|S2S 3, 1|rich 2, 1.4|rich 5, 2|lazy 2, 2|rich 3, 2.8|lazy 5, 4|lazy 5
- dK0.05_rho5.0: 0.05|S2S 5, 0.07|S2S 4, 0.07|rich 1, 0.1|S2S 3, 0.1|rich 2, 0.14|S2S 3, 0.14|rich 2, 0.2|S2S 1, 0.2|rich 4, 0.3|S2S 1, 0.3|rich 4, 0.5|rich 5, 0.7|S2S 2, 0.7|rich 3, 1|S2S 1, 1|rich 4, 1.4|rich 5, 2|lazy 2, 2|rich 3, 2.8|lazy 5, 4|lazy 5
- dK0.1_rho2.0: 0.05|S2S 5, 0.07|S2S 5, 0.1|S2S 4, 0.1|rich 1, 0.14|S2S 4, 0.14|rich 1, 0.2|S2S 5, 0.3|S2S 5, 0.5|S2S 4, 0.5|rich 1, 0.7|S2S 4, 0.7|rich 1, 1|S2S 4, 1|rich 1, 1.4|rich 5, 2|lazy 4, 2|rich 1, 2.8|lazy 5, 4|lazy 5
- dK0.1_rho3.0: 0.05|S2S 5, 0.07|S2S 5, 0.1|S2S 4, 0.1|rich 1, 0.14|S2S 4, 0.14|rich 1, 0.2|S2S 4, 0.2|rich 1, 0.3|S2S 5, 0.5|S2S 2, 0.5|rich 3, 0.7|S2S 3, 0.7|rich 2, 1|S2S 3, 1|rich 2, 1.4|rich 5, 2|lazy 4, 2|rich 1, 2.8|lazy 5, 4|lazy 5
- dK0.1_rho5.0: 0.05|S2S 5, 0.07|S2S 4, 0.07|rich 1, 0.1|S2S 3, 0.1|rich 2, 0.14|S2S 3, 0.14|rich 2, 0.2|S2S 1, 0.2|rich 4, 0.3|S2S 1, 0.3|rich 4, 0.5|rich 5, 0.7|S2S 2, 0.7|rich 3, 1|S2S 1, 1|rich 4, 1.4|rich 5, 2|lazy 4, 2|rich 1, 2.8|lazy 5, 4|lazy 5
- dK0.2_rho2.0: 0.05|S2S 5, 0.07|S2S 5, 0.1|S2S 4, 0.1|rich 1, 0.14|S2S 4, 0.14|rich 1, 0.2|S2S 5, 0.3|S2S 5, 0.5|S2S 4, 0.5|rich 1, 0.7|S2S 4, 0.7|rich 1, 1|S2S 4, 1|rich 1, 1.4|rich 5, 2|lazy 5, 2.8|lazy 5, 4|lazy 5
- dK0.2_rho3.0: 0.05|S2S 5, 0.07|S2S 5, 0.1|S2S 4, 0.1|rich 1, 0.14|S2S 4, 0.14|rich 1, 0.2|S2S 4, 0.2|rich 1, 0.3|S2S 5, 0.5|S2S 2, 0.5|rich 3, 0.7|S2S 3, 0.7|rich 2, 1|S2S 3, 1|rich 2, 1.4|rich 5, 2|lazy 5, 2.8|lazy 5, 4|lazy 5
- dK0.2_rho5.0: 0.05|S2S 5, 0.07|S2S 4, 0.07|rich 1, 0.1|S2S 3, 0.1|rich 2, 0.14|S2S 3, 0.14|rich 2, 0.2|S2S 1, 0.2|rich 4, 0.3|S2S 1, 0.3|rich 4, 0.5|rich 5, 0.7|S2S 2, 0.7|rich 3, 1|S2S 1, 1|rich 4, 1.4|rich 5, 2|lazy 5, 2.8|lazy 5, 4|lazy 5

## chi = 8

- binary_quadtree/raw: two-point slopes (0.05 -> 0.14, rank clock) seed 0: 1.26, seed 1: 1.59, seed 2: 1.23, seed 3: 1.13 (routes [[1, 2, 4, 5], [3, 4]]), seed 4: 1.14; mean 1.27 ± 0.08 (n = 5); seeds with one route 1.31 ± 0.10 (n = 4)
- binary_scrambled/raw: two-point slopes (0.05 -> 0.14, rank clock) seed 0: 1.12, seed 1: 1.07, seed 2: 1.44, seed 3: 1.16 (routes [[0, 1], [1, 2, 5]]), seed 4: 1.20; mean 1.20 ± 0.06 (n = 5); seeds with one route 1.21 ± 0.08 (n = 4)
- stop reasons (run logs): {'quadtree|completed': 1, 'quadtree|max_wall_time': 14, 'mps|max_wall_time': 9, 'mps|completed': 6, 'scrambled|completed': 15}

## Absolute clock: first-escape time t_a at sigma = 0.1

| tree | order | seed | t_a | source |
|---|---|---|---|---|
| binary | scrambled5 | 0 | 5.27e+14 | audit |
| binary | quadtree | 0 | 1.1e+15 | regime_analysis |
| binary | scrambled8 | 0 | 1.19e+15 | audit |
| binary | scrambled2 | 0 | 1.21e+15 | fix1 |
| binary | scrambled1 | 0 | 1.25e+15 | fix1 |
| binary | scrambled3 | 0 | 1.37e+15 | audit |
| binary | scrambled0 | 0 | 1.54e+15 | fix1 |
| binary | scrambled7 | 0 | 2.25e+15 | audit |
| binary | quadtreeT | 0 | 2.3e+15 | audit |
| binary | rowmajor | 0 | 2.74e+15 | audit |
| mps | rowmajor | 0 | 4.33e+15 | audit |
| binary | scrambled4 | 0 | 6.31e+15 | audit |
| binary | scrambled6 | 0 | 6.86e+15 | audit |
| mps | snake | 0 | 8.36e+15 | fix1 |
| mps | scrambled3 | 0 | 1.44e+16 | audit |
| mps | scrambled0 | 0 | 2.48e+16 | audit |
| binary | scrambled0 | 1 | 4.67e+14 | fix1 |
| binary | scrambled1 | 1 | 8.77e+14 | fix1 |
| binary | rowmajor | 1 | 1.17e+15 | audit |
| mps | snake | 1 | 1.36e+15 | fix1 |
| binary | quadtree | 1 | 1.67e+15 | regime_analysis |
| binary | quadtreeT | 1 | 1.99e+15 | audit |
| binary | scrambled2 | 1 | 2.04e+15 | fix1 |
| binary | quadtree | 2 | 1.88e+14 | regime_analysis |
| binary | scrambled0 | 2 | 2.62e+14 | fix1 |
| binary | scrambled2 | 2 | 3.05e+14 | fix1 |
| binary | scrambled1 | 2 | 5.03e+14 | fix1 |
| mps | snake | 2 | 2.46e+16 | fix1 |
| binary | scrambled2 | 3 | 9.76e+14 | fix1 |
| binary | scrambled1 | 3 | 1.02e+15 | fix1 |
| binary | scrambled0 | 3 | 1.65e+15 | fix1 |
| binary | quadtree | 3 | 2.06e+15 | regime_analysis |
| mps | snake | 3 | 4.16e+15 | fix1 |
| binary | scrambled0 | 4 | 2.93e+14 | fix1 |
| binary | quadtree | 4 | 3.78e+14 | regime_analysis |
| binary | scrambled1 | 4 | 3.82e+14 | fix1 |
| binary | scrambled2 | 4 | 5e+14 | fix1 |
| mps | snake | 4 | 2.44e+15 | fix1 |

t_a(MPS) / t_a(quadtree) per seed: 0: 7.60, 1: 0.82, 2: 130.78, 3: 2.02, 4: 6.44; binary runs with t_a below the MPS's, per seed: 0: 12, 1: 3, 2: 4, 3: 4, 4: 4

## Reproduction: fix1 reruns vs committed values (quadtree, sigma = 0.1)

- seed 0: |t_a / t_a(committed) - 1| = 1.8e-15, L(1e4) 0.09572 vs 0.09572
- seed 1: |t_a / t_a(committed) - 1| = 6.9e-15, L(1e4) 0.09853 vs 0.09853
- seed 2: |t_a / t_a(committed) - 1| = 4.3e-15, L(1e4) 0.09677 vs 0.09677
- seed 3: |t_a / t_a(committed) - 1| = 2.2e-16, L(1e4) 0.09812 vs 0.09812
- seed 4: |t_a / t_a(committed) - 1| = 2.8e-15, L(1e4) 0.10307 vs 0.10307

## Creep between drops (binary fix1 reruns, sigma = 0.1): 56 intervals, loss decrease between drops / next drop: median 0.76, range 0.05-9.55

## chi = 8: two-point slopes (sigma 0.05 -> 0.14) of the rank-clock gap; first rises at tau = 0

- binary_quadtree/raw: seed 0: 1.26 (routes [[3, 4]]), seed 1: 1.59 (routes [[0, 1]]), seed 2: 1.23 (routes [[3, 4]]), seed 3: 1.13 (routes [[1, 2, 4, 5], [3, 4]]), seed 4: 1.14 (routes [[0, 1]]); mean 1.27 ± 0.08 (n = 5); runs with the first rise at tau = 0: 1
- binary_scrambled/raw: seed 0: 1.12 (routes [[10, 11]]), seed 1: 1.07 (routes [[0, 1]]), seed 2: 1.44 (routes [[0, 1]]), seed 3: 1.16 (routes [[0, 1], [1, 2, 5]]), seed 4: 1.20 (routes [[10, 11]]); mean 1.20 ± 0.06 (n = 5); runs with the first rise at tau = 0: 0
- mps_snake/raw: seed 0: - (routes [[5]]), seed 1: - (routes [[5]]), seed 2: - (routes [[5]]), seed 3: - (routes [[5]]), seed 4: - (routes [[5]]); mean -; runs with the first rise at tau = 0: 4

## Fashion-MNIST: two-point slopes (sigma 0.05 -> 0.14) of the rank-clock gap; first rises at tau = 0

- binary_quadtree/raw: seed 0: 0.97 (routes [[7], [10]]), seed 1: 0.93 (routes [[3, 4], [7]]), seed 2: 0.93 (routes [[7], [10, 11]]), seed 3: 0.96 (routes [[4], [7]]), seed 4: 0.97 (routes [[3, 4], [7]]); mean 0.95 ± 0.01 (n = 5); runs with the first rise at tau = 0: 0
- binary_scrambled/raw: seed 0: 0.88 (routes [[0], [10]]), seed 1: 0.86 (routes [[0], [10]]), seed 2: 0.88 (routes [[0], [10]]), seed 3: 0.90 (routes [[7, 10], [10], [10, 11]]), seed 4: 0.88 (routes [[7, 8], [10]]); mean 0.88 ± 0.01 (n = 5); runs with the first rise at tau = 0: 0
- mps_snake/raw: seed 0: 1.03 (routes [[11], [12, 14], [14]]), seed 1: 0.10 (routes [[3], [11]]), seed 2: 0.34 (routes [[11], [11, 12, 13]]), seed 3: 0.10 (routes [[3], [11]]), seed 4: 0.21 (routes [[11], [12, 13]]); mean 0.36 ± 0.17 (n = 5); runs with the first rise at tau = 0: 0

## Whitened features: S2S fraction of binary runs at sigma <= 0.1: 17/30 = 0.57

## MPS reruns (configs/fix1_mps.yaml, tau <= 1000): first rise and per-edge laws

35 runs; first rise already at tau = 0 (threshold met at the start of phase 2): 10; resolved: 25, of which a single edge 25 (edges [(5,)]), before the first drop's rate peak 25, inside the first drop's widened window 24.

| edge | seeds | power p (mean ± SE) | power rms | log slope | log rms | seeds preferring log |
|---|---|---|---|---|---|---|
| 1 | 4 | 1.25 ± 0.10 (n = 4) | 0.113 | 142 ± 8 | 0.272 | 0/4 |
| 2 | 5 | 1.00 ± 0.08 (n = 5) | 0.131 | 94.2 ± 7.6 | 0.141 | 1/5 |
| 3 | 5 | 0.83 ± 0.19 (n = 5) | 0.254 | 30.3 ± 0.77 | 0.029 | 5/5 |
| 4 | 5 | 0.91 ± 0.23 (n = 5) | 0.355 | 14.7 ± 1.5 | 0.027 | 4/5 |

| seed | sigma | t_a | first drop peak | first rising edges | at tau = 0 | rise of edges 5 / 4 / 3 / 2 |
|---|---|---|---|---|---|---|
| 0 | 0.05 | 1.37e+20 | 12.12 | [5] | False | 10.85 / 53.85 / 97.62 / 342.7 |
| 0 | 0.07 | 1.232e+18 | 12.69 | [5] | False | 11.28 / 48.1 / 86.21 / 195 |
| 0 | 0.1 | 8.359e+15 | 11.37 | [5] | False | 9.79 / 40.18 / 72.31 / 152.1 |
| 0 | 0.14 | 7.522e+13 | 11.13 | [5] | False | 9.175 / 33.83 / 60.33 / 119.7 |
| 0 | 0.2 | 5.102e+11 | 10.33 | [5] | True | 0 / 26.7 / 47.31 / 88.08 |
| 0 | 0.3 | 1.748e+09 | 10.72 | [5] | True | 0 / 20.37 / 34.81 / 57.21 |
| 0 | 0.5 | 1.37e+06 | 9.84 | [5, 6] | True | 0 / 12.48 / 19.94 / 25.87 |
| 1 | 0.05 | 2.23e+19 | 15.33 | [5] | False | 15.28 / 58.09 / 94.77 / 320.5 |
| 1 | 0.07 | 2.006e+17 | 15.70 | [5] | False | 15.37 / 52.17 / 82.77 / 284.2 |
| 1 | 0.1 | 1.361e+15 | 14.55 | [5] | False | 14 / 44.41 / 68.46 / 167.3 |
| 1 | 0.14 | 1.225e+13 | 15.07 | [5] | False | 14.23 / 38.6 / 56.49 / 132.2 |
| 1 | 0.2 | 8.306e+10 | 15.25 | [5] | False | 14.24 / 32.24 / 44.47 / 99.73 |
| 1 | 0.3 | 2.845e+08 | 15.22 | [5] | False | 14 / 24.86 / 31.85 / 66.38 |
| 1 | 0.5 | 2.229e+05 | 13.49 | [5] | False | 12.08 / 13.91 / 14.78 / 32.52 |
| 2 | 0.05 | 4.029e+20 | 15.32 | [5] | False | 14.13 / 66.7 / 113.9 / 365.3 |
| 2 | 0.07 | 3.626e+18 | 14.74 | [5] | False | 13.57 / 59.75 / 101.5 / 319.9 |
| 2 | 0.1 | 2.459e+16 | 13.96 | [5] | False | 12.79 / 52.21 / 88.32 / 268.5 |
| 2 | 0.14 | 2.213e+14 | 13.24 | [5] | False | 12.18 / 45.3 / 76.24 / 194 |
| 2 | 0.2 | 1.501e+12 | 12.07 | [5] | False | 11.1 / 37.7 / 63.49 / 150.8 |
| 2 | 0.3 | 5.142e+09 | 12.20 | [5] | False | 11.34 / 30.9 / 51.56 / 114.4 |
| 2 | 0.5 | 4.029e+06 | 10.59 | [5] | False | 9.828 / 21.69 / 38.25 / 77.46 |
| 3 | 0.05 | 6.821e+19 | 13.70 | [5] | False | 12.45 / 48.76 / 95.72 / 338.4 |
| 3 | 0.07 | 6.139e+17 | 14.42 | [5] | False | 13.13 / 44.93 / 86.71 / 225.6 |
| 3 | 0.1 | 4.163e+15 | 13.98 | [5] | False | 12.69 / 39.78 / 76.06 / 171 |
| 3 | 0.14 | 3.747e+13 | 13.40 | [5] | False | 12.11 / 34.83 / 65.9 / 138 |
| 3 | 0.2 | 2.541e+11 | 12.70 | [5] | False | 11.43 / 29.62 / 55.08 / 106.9 |
| 3 | 0.3 | 8.704e+08 | 11.77 | [5] | False | 10.54 / 23.76 / 42.69 / 74.56 |
| 3 | 0.5 | 6.821e+05 | 11.89 | [5] | False | 10.76 / 18.34 / 29.45 / 41.35 |
| 4 | 0.05 | 3.99e+19 | 9.93 | [5] | True | 0 / 54.76 / 111.7 / 343.2 |
| 4 | 0.07 | 3.591e+17 | 10.34 | [5] | True | 0 / 50.69 / 102.1 / 304.1 |
| 4 | 0.1 | 2.436e+15 | 10.08 | [5] | True | 0 / 45.86 / 91.36 / 244 |
| 4 | 0.14 | 2.192e+13 | 9.62 | [5] | True | 0 / 41.35 / 81.2 / 180.6 |
| 4 | 0.2 | 1.487e+11 | 10.13 | [5] | True | 0 / 37.94 / 71.4 / 141.1 |
| 4 | 0.3 | 5.092e+08 | 10.30 | [5] | True | 0 / 34.28 / 59.01 / 96.89 |
| 4 | 0.5 | 3.99e+05 | 9.43 | [5] | True | 0 / 29.38 / 40.17 / 45.29 |

Reproduction: first rise - first drop peak, reruns vs committed full-horizon runs (35 runs): median |difference| 0.0101, max 0.0676 (time units).
