# Part 5: rerun vs committed (own fits, my_fits.py; per-seed slopes, mean +- SE)

| row | k | quantity | committed | rerun |
|---|---|---|---|---|
| binary4 S=01 small | 1 | p of t2-t1 vs 1/sigma (gap shrinks ~ sigma) | -1.002 +- 0.001, 28/28 order split | -1.002 +- 0.001, 28 greedy / 28 reversed |
| mps4 S=12 small | 2 | log slope (pred 1.000) | 1.0085 +- 0.0024 | 1.008 +- 0.002 |
|  |  | lambda_min at sigma=1e-4 | -0.4987 | -0.499 |
| mps4 S=12 saddle | 2 | log slope (pred 2.000) | 2.0178 +- 0.0243 | 2.018 +- 0.024 |
|  |  | lambda_min at sigma=1e-4 / sigma-exponent | -0.444 / -0.003 +- 0.003 | -0.444 / -0.003 +- 0.003 |
| mps8 S=34 saddle | 2 | log slope (pred 2.000) | 2.0022 +- 0.0102 | 2.002 +- 0.010 |
|  |  | lambda_min at sigma=1e-4 | -0.460 | -0.460 |
| binary4 S=12 small | 3 | p (pred 1) / Hessian exp | 1.011 +- 0.005 / 1.042 +- 0.013 | 1.011 +- 0.005 / 1.042 +- 0.013 |
| mps4 S=0123 small | 4 | p (pred 2) / Hessian exp | 1.990 +- 0.006 / 1.994 +- 0.005 | 1.990 +- 0.006 / 1.994 +- 0.005 |
| mps4 S=03 small | 4 | p; classes | 2.015 +- 0.034 (3 seeds); 17/0/1/30 stalled | 2.015 +- 0.034; 17 greedy / 1 rot / 30 stalled (seeds 1,3,4,5,7) |
| mps4 S=03 balanced | 4 | p; stalls | 1.990 +- 0.004; 0 stalled | 1.990 +- 0.004 (46/48 done at this point); 0 stalled |
