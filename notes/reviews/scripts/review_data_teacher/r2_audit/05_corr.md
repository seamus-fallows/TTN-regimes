# R2 audit note 5: correlated inputs (r2_corr_runs.csv)

- Same protocol, sigma grid (0.05 * 2^-j, j = 0..5) and d = 3 as theory_covariance.md N14, which already
  reported these increments (c = 0.4: 2.12-2.19 -> 2.13; c = 0.2: 17.4 -> 14.7, "still decreasing toward 14.4").
  R2.3 is a replication with 4 seeds (chi = 4), not an independent test.
- Seeds: 5 run, seed 2 excluded (minor saddle) -> 4 per fit (< 5-seed rule).
- inc_mean +- inc_sem pools 20 increments (4 seeds x 5 sigma pairs) as independent; the sigma values are
  rounded (0.0016, 0.0031, 0.0063: ratios 1.9375, 2.032, 1.984), so "per halving" increments are biased
  by -4.6% / +2.3% / -1.2% on the first three pairs. Per-seed regression on log2(1/sigma):
  rho = 0.4: 2.144 +- 0.007 (all sigma), 2.133 +- 0.003 (3 smallest) vs 2.128;
  rho = 0.2: 14.97 +- 0.30 (all), 14.67 +- 0.06 (3 smallest) vs 14.39 (+1.9%, not yet asymptotic);
  rho = 0.3: 4.667 / 4.616; rho = 0.1: 117.4 / 113.9 (no prediction computed, though the formula of
  theory_covariance 7.4 3c applies at any c).
- 1.5 decades of sigma: log beats power per seed (rms 0.001-0.012 vs 0.036-0.081). OK.
- Binary-4 sigma^-1 (p 1.005-1.011): OK. Separation-ratio medians include the minor-saddle seed.
