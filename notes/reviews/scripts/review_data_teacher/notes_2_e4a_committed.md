# Part 2a: step 4(a) plateau law — analysis issues visible in the committed CSVs

Source: committed `results/e4a_sigma_law.csv` (per run: t1, t2, dt2, plateau_rel) and `_fits.csv`.

1. **Heterogeneous runs are averaged into one slope.** With s2 = 0.7 and c = 0.2 there are two rank-1
   saddles (theory_log_slope: loss 0.2449 -> A = 34.1; loss 0.4999 -> A = 25.7). Of the 10 seeds,
   8 visit the s=1 ("big") saddle at every sigma, seed 3 visits the small one (plateau_rel 0.667), and
   seed 1 has plateau_rel = 0.011 at every sigma: `analyze_sigma_law.events()` found no rank-1 plateau
   (the two components are learned together) and took the speed minimum in the slow tail just above
   its 1e-2 L(0) cut, so its "dt2" is the time to reach ~0.5% of L(0), not a plateau duration.
   The fit uses per-sigma means over all 10 runs, so A = 31.9 is a mixture:
   per-seed slopes (A from dt2 vs log(1/sigma), 6 sigma): 49.7, 1.2 (seed 1), 40.4, 19.5 (seed 3),
   37.4, 40.0, 30.6, 37.0, 30.8, 32.2. Big-saddle seeds only: A = 37.3 +- 2.2 (8 seeds) vs the
   prediction 34.1 for that saddle (+9%, 1.4 s.e.). Small saddle (1 seed): 19.5 vs 25.7.
   Same issue at c = 0.45: seeds 0,2,4,5,7,9 (plateau_rel 0.300 = predicted saddle 0.2345/0.7737) give
   A = 5.2-6.7 (mean ~6.0, pred 5.68); seeds 1,3,6,8 have plateau_rel 0.26-0.28 (speed minimum below
   the saddle level: the trajectory does not come close to the saddle at these sigma) and A = 2.0-3.7.
   The reported 4.7 +- 0.3 (means) / 5.6 (medians) mixes the two groups.
   At c = 0 the same misdetection occurs (e.g. sigma 0.0474 seed 0: dt2 = 5.7, plateau_rel 0.011), but
   the reported exponent uses medians, which are robust to it.
2. **Error bars are too small.** `fit()` takes the s.e. from the residuals of a 6-point regression of
   per-sigma means, but the same 10 seeds (same normalized init direction, `spec.init(sigma, seed)`) are
   used at every sigma, so the per-sigma means are strongly correlated and seed-to-seed variation of
   the slope is invisible to the residuals. Per-seed slopes give A = 31.9 +- 4.2 (sd 13.4) instead of
   +- 1.6 for c = 0.2 (identical point estimate, since the mean of per-seed OLS slopes equals the OLS
   slope of the means when every seed is present at every sigma).
3. e2_sigma (theta = 20) does not have this problem: all 10 seeds visit the same saddle (plateau_rel
   0.073) and per-seed slopes give 14.96 +- 0.40, identical to the reported 15.0 +- 0.4.
4. Minor: crosscheck_library.csv contains a row "e1_order: failed, BadZipFile (Bad CRC-32 for
   spectra.npy)": one raw npz on the remote machine was unreadable when the cross-check ran.

## Part 2b: predicted slopes recomputed independently (indep_theory_slope.py)
HOPM (rank-1 power iteration on W*) started near each component, R6 formula coded separately:
c = 0.05: A = 569.9 (s=1 saddle, rel. level 0.3289) / 474.2 (s=0.7 saddle, 0.6711);
c = 0.2: 34.141 (0.3283) / 25.719 (0.6699); c = 0.45: 5.682 (single blended saddle, 0.3030; both
starts converge to it). Identical to the committed e4a_theory_slopes.csv (constrained.py + theory_log_slope.py).
