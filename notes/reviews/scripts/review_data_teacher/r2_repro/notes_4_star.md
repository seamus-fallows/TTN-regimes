# part 4: star teachers (R2.2)

Rerun (agent code): r2_star.py hessian mode reproduces r2_star_hessian.csv (all 20 rows, to ~1e-15).
Independent Hessian (indep_star.py, own einsum): same values; sign flips between w = 1.4138 and 1.4146.

Threshold scan (agent saddle_job, eps 1e-6): mps4 w = 1.4138 seeds 0,1 -> learn_r at t = 2.62e6 (committed: learn_r,
t_end 2621420); w = 1.4146 seeds 0,1 -> forget_q, t_end 4.09e4 (committed: forget_q, t_end 40940). [binary4: see results]

INTERPRETATION ISSUE in the random-init "forgetting depth" (r2_star_random.csv, committed data):
- forgotten runs (w = 1.6) have a_q peak 0.55-0.99 (mps4 median ~0.73), spend a median 0.018 time units at the
  "q learned" loss level (sub-critical runs: 16-227), and r crosses 1/2 a median 5.5 time units after q
  (t_r < t_q in 2%). So q is never learned: the first escape goes toward the BLENDED rank-1 maximum of the star
  (independent HOPM: a_q = 0.673, a_r = 0.295; strict saddle on mps4, lambda_min = -0.642), a_q transiently
  reaches ~0.6-0.75 and then decays while r, r' are learned. "Super-critical stars forget q down to
  min a_q ~ sigma^5.7" therefore measures the decay of a partially aligned (blended) direction, not the
  forgetting of a learned component through the theta_q saddle (the P8 scenario tested by the saddle-start runs).
- depth exponent: per-seed values 4.47-6.84 (8 seeds, one decade of sigma, 4-6 points each): 5.7 +- 0.3 is the
  SE of a wide spread; seed 2 at sigma = 0.01 has min a_q = -3.4e-11 (excluded by the 1e-10 cut).

## rerun results (agent code, star_subset.py)
- threshold scan (eps 1e-6): mps4 and binary4, w = 1.4138 seeds 0,1 -> learn_r (no rotation; t_end 2.62e6 / 1.31e6 =
  committed), w = 1.4146 seeds 0,1 -> forget_q; t_rotate equal to committed within 0.05%, aq_min equal to 7 digits.
- rotation (mps4 w = 1.6, seed 0, eps 1e-2, 1e-4, 1e-6): same outcomes; linear slope A = 2.679 (pred 2.691; committed
  5 seeds x 4 eps 2.686 +- 0.005). t_rotate is grid-limited (first row with a_q < 0.9; 0.8% diff at eps 1e-2).
- ht4 w = 1.6 eps 1e-3 seed 0: t(a_q < 1/2) 2806.44 (committed 2806.38); stall a_q = 0.6732, a_r = 0.2945 = the
  blended rank-1 maximum from my own HOPM (0.6732 / 0.2945).
- random init mps4 w = 1.6 seed 2, sigma 0.01..0.1 (4 runs) + w = 1.2 sigma 0.1: forgotten flags equal, min a_q equal
  to <= 3e-4 relative (e.g. 4.7983e-9 vs 4.7985e-9). a_q peak 0.58-0.65, time at the q-learned level 0.009-0.06 vs
  24-538 at the q-forgotten level, t_r - t_q = -0.6..+5.8: q is never learned in these runs (see above).
