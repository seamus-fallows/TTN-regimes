# Rerun vs committed (two-component rows; per-seed fits, mean +- SE over seeds)

| row (protocol) | k | quantity (prediction) | committed | rerun |
|---|---|---|---|---|
| binary4 S=01 small | 1 | gap exponent; classes | -1.002 +- 0.001 (gap ~ sigma, 7.6e-4 at 1e-3); 28/28 | identical |
| mps4 S=12 small | 2 | log slope (1.000); lambda_min(1e-4) | 1.0085 +- 0.0024; -0.4987 | identical; -0.4987 |
| mps4 S=12 saddle | 2 | log slope (2.000); lambda_min(1e-4) | 2.0178 +- 0.0243; -0.444 | identical; -0.444 (own, well-conditioned gauge: -0.5000) |
| mps8 S=34 saddle | 2 | log slope (2.000); lambda_min(1e-4) | 2.0022 +- 0.0102; -0.460 | identical; -0.460 (own: -0.5000) |
| binary4 S=12 small | 3 | p (1); Hessian exp (1) | 1.0112 +- 0.0049; 1.0418 +- 0.0132 | 1.0112 +- 0.0049; 1.0416 +- 0.0131 |
| mps4 S=0123 small | 4 | p (2); Hessian exp | 1.9896 +- 0.0063; 1.9943 | 1.9896 +- 0.0063; 1.9940 |
| mps4 S=03 small | 4 | p; greedy/rot/stalled | 2.0154 +- 0.0339 (3 seeds); 17/1/30 | identical; stalled seeds 1,3,4,5,7 |
| mps4 S=03 balanced | 4 | p; stalled | 1.9904 +- 0.0039; 0/48 | identical; 0/48 |
| binary8 S=34 small | 5 | p (3); local; Hessian exp | 3.0123 +- 0.0818; 2.986; 3.046 | 3.0123 +- 0.0818; 2.986; 3.047 |
| ht8 S=34 saddle | 7 | p (5) | 5.74 +- 1.06 (median 5.01), 28 esc. + 2 stalled | 5.68 +- 1.00 (median 5.01), 28 esc. (seed 4 sigma <= 0.2 not finished) |
| TD mps8 saddle | k* = 2, 2, 3 | level exponents | 0.267, 0.649, 1.012 +- 0.010 | identical (agent's levels.py and own analysis) |
| growth gamma -> k_est | 2..7 | 1.993 / 2.024 / 2.991 / 3.985 / 5.120 / 7.733 | same to 3 decimals |
