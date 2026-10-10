# R2 audit note 4: star teachers

- Hessian at theta_q: |lambda_min - formula| <= 2.8e-15 (w = 1.45, 1.6; 4 trees) - verified. Same teacher,
  trees and values as theory_saddles.md C11 (already "matched to 6 digits"): a replication, plus w = 1.45 and
  sub-critical controls. On binary4 the star = DLN with top mode r + r' (known, Achour et al. 2024).
- Threshold: mps4/binary4, eps = 1e-6, 3 seeds per w: 0/3 leave q at w <= 1.4138, 3/3 at w >= 1.4146
  (rho/s_q 0.99972 / 1.00028) - verified bracket. ht4 at eps = 1e-6: the w = 1.42 runs are missing (5 stopped),
  bracket [1.40, 1.44]; on ht4 near threshold (w = 1.42, 1.44) the "a_q < 0.9" event follows eps^-1
  (rotation-law p = 0.995, 0.998), i.e. it is reached through the polynomial second stage, not the strict escape.
- Rotation law (r2_star_rotation_law.csv; 4 eps values 1e-2..1e-6, min_points = 3): measured vs 1/|lambda_min|
  with lambda = -a^(K-2)(sqrt2 w - 2): binary4 w = 1.42/1.44/1.48/1.60: -0.02/-0.06/+0.03/+0.15%; ht4 1.48/1.60:
  +0.17/-0.12%; mps4 1.48/1.60: -0.12/-0.21%; mps4 1.42/1.44: +50.5/+51.6% (agent: "needs a second step").
  The agent compared only w = 1.6.
- Forgetting depth (r2_star_random.csv, w = 1.6): mps4 peak a_q per seed: seeds 0, 2, 4, 7, 9 peak at
  0.55-0.68 (never near q: the flow goes to the blended best rank-1 point), seeds 3, 5, 8 at 0.78-0.91, seed 6
  0.99 (forgets only at sigma <= 0.0158), seed 1 learns q last. The depth fit pools both classes:
  blend-path exponents 4.47, 6.02, 4.49, 6.72, 5.05 (mean 5.35 +- 0.45); q-path 6.84, 5.86, 6.44 (6.38 +- 0.29).
  Per-seed local exponents drift within one decade (seed 0: 5.3 -> 3.7 from sigma 0.1 to 0.01; seed 3: 5.5-8.1).
  Fit window 1e-10 < min a_q < 0.5 selects on the outcome; seeds 2 and 7 have NEGATIVE min a_q at sigma = 0.01,
  0.0158 (-3.4e-11, -4.1e-9, -5.8e-9): a_q crosses zero, so "min a_q" is not a depth there ("deepest 3e-11" is
  one of these). One decade of sigma, 4-6 points per seed.
- Relearning time "~sigma^-1.4": not in any committed CSV; per seed from r2_star_random.csv (t_relearn - t_min):
  mps4 1.78 +- 0.15 (blend path 1.37-1.76, q path 2.03-2.57), binary4 1.71 +- 0.11.
- Sub-critical "min a_q >= 0.993": lowest 0.9918 (mps4, sigma = 0.1). Trivial.
- ht4 two-stage: t(a_q < 1/2) = 2.8e3-5.6e3, 2.8e4-5.5e4, 2.8e6-5.5e6 at eps = 1e-3, 1e-4, 1e-6 (5 seeds; ratio
  100.0 per 2 decades): 1/eps verified. "Higher-order saddle on HT" for the blended point is inferred from
  timing; no Hessian at that point is committed. Float-floor scan does not cover r2_star_* (a_q reaches 1e-26 in
  ht4 saddle runs after forgetting).
