# part 3: correlated inputs (R2.3) - rerun with the agent's r2_corr.job (copied code)

Reran mps rho 0.4 and 0.2, binary rho 0.2, seed 0, all 6 sigma (+ seeds 1,3,4 for mps queued):
per-run gap / t_event1 / separation ratio equal to the committed r2_corr_runs.csv to <= 2e-8 relative
(scipy LSODA with exact Hessian is near-bitwise reproducible across platforms).

Re-analysis of the committed runs (compare_corr.py):
- SIGMAS = 0.0016, 0.0031, 0.0063, 0.0125, 0.025, 0.05 are not factor-2 spaced (ratios 1.94, 2.03, 1.98, 2, 2);
  analyze_r2_corr.py takes raw consecutive differences as "increment per halving" without dividing by
  ln(ratio)/ln 2, pools all 5 pairs x 4 seeds and quotes the pooled SEM (n = 20), not per-seed.
- Normalized per ln 2: rho = 0.4: all-pair mean 2.145 +- 0.007 (per-seed SE), smallest-sigma pair 2.131,
  largest pair 2.171 (pred 2.128); rho = 0.2: 14.99 +- 0.33, smallest pair 14.62, largest 15.56 (pred 14.39).
  The increments still decrease toward small sigma (theory_covariance 7.3c says the same), so the best
  estimate is the smallest-sigma pair: +0.2% (rho 0.4) and +1.6% (rho 0.2). The conclusion (log law with the
  predicted two-stage increment) stands; the quoted "2.131 +- 0.013" / "14.89 +- 0.22" are raw pooled means whose
  0.1% agreement at rho = 0.4 is partly the grid spacing (mean step ln-ratio 0.688 < ln 2 lowers raw
  increments by 0.7%), and the SEs do not reflect the systematic sigma trend.
- rho = 0.2 seed 1 trends the other way (normalized increments 12.7 -> 14.5 toward small sigma) while seeds 0, 3, 4
  fall 16.3 -> 14.7.
