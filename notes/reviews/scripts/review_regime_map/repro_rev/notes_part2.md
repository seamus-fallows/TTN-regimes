# regime_map reproduction review - working notes (part 2): independent check

Own numpy/scipy implementation (indep_b4.py): W = A R B^T, fan-in init, odeco [3,2,1], centred target,
DOP853 rtol 1e-10 atol 1e-14 with root-found events (a_r = 0.1/0.5/0.9), hand-derived probe kernel
(checked vs finite differences: 5e-11), own RNG streams for teacher / init / probes, 6 own seeds.
Boundaries by bisection in sigma (continuous rho = 3 and delta_K = 0.1 crossings).

| quantity | committed | independent |
|---|---|---|
| deep-S2S widths (s = 3,2,1) | 0.466, 0.798, 2.010 | 0.464-0.473, 0.798, 2.009-2.01 |
| rho slope per seed (S2S side) | -1.02..-1.03 | -1.011, -1.011, -1.015 (chi 9); -1.015, -1.022, -1.019 (chi 64) |
| sigma_lazy sqrt(chi), median | 6.29 (32), 6.18 (64), 6.10 (128) | 6.41 (32), 6.37 (64), 6.09 (128); 6.84 (16) |
| lazy slope chi 32-128 | -0.522 +- 0.020 | -0.536 (medians) |
| sigma_c sqrt(chi), median | 0.29 (32), 0.31 (64), 0.27 (128) | 0.32 (32), 0.29 (64), 0.32 (128); 0.36 (16) |
| S2S slope | -0.52 +- 0.05 | -0.55 (16-128), -0.50 (32-128) medians |
| rich window lazy/S2S | ~21 | 19.8, 22.2, 18.9 (chi 32, 64, 128) |
| floor: l_end / l_best (chi 3,4 at sigma 1.0, 2.5; 3 seeds) | 1 to 1e-6 | 1 to 1e-15 (12/12) |
| chi = 6, sigma = 1.0 | rich (floor < 1e-3) | rich, l_end = l_best = 1.6e-4..8.8e-4 |
| large chi dK slope SP / NTK / muP | -3.09 / -0.84 / +0.03 | -3.06 +- 0.07 / -0.78 +- 0.05 / +0.07 +- 0.04 |
| dth slope SP / NTK / muP | -1.63 / -0.52 / -0.06 | -1.66 / -0.55 / -0.06 |
| root slope SP / NTK / muP | -1.02 / -0.93 / -0.50 | -1.09 / -0.96 / -0.51 |
| leaf slope SP / NTK / muP | -2.96 / -0.53 / -0.06 | -2.98 / -0.56 / -0.06 |
| muP delta_K level | 4-5 | 4.1-5.1 |
| imbalance chi16 alpha 3.375 median dK, lambda .01..100 | 1e-4, 7e-4, .0069, .036, .167, .45, .53, .50, .50 | 6.1e-5, 5.5e-4, .0057, .036, .158, .41, .54, .50, .49 |
| local slopes (smallest lambda up) | 2.18, 1.86, 1.51, 1.27 | 2.00, 1.95, 1.67, 1.24 |
| labels | lazy lambda <= 0.3, rich >= 1 | same (6/6 each) |
