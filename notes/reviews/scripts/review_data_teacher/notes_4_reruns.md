# Part 4: rerun results for steps 4(a), 4(b), 2

## 4(a) plateau law, MPS-4, s2 = 0.7, library LSODA two-stage dense runner (run_dense.py, unchanged)
Reduced config: overlap in {0, 0.2} (dropped 0.05, 0.45), all 6 sigma (0.03..0.3), seeds 0-9 = 120 runs,
all `converged`, max conservation drift 3.9e-10 (committed: 2.6e-10). ~10 min on 6 processes.
- Per run vs committed e4a_sigma_law.csv (120/120 matched): t1 max rel diff 7e-4 (median 1.4e-5),
  t2 4e-3 (1e-4), dt2 6e-3 (5e-4). Pilot DOP853 runs reproduce bitwise; LSODA differs at 1e-5..1e-4 in
  the end-of-plateau time (different Fortran/BLAS build on Windows, amplified by the unstable escape).
- analyze_sigma_law.py fits, rerun vs committed: c = 0: medians exponent 1.9667 +- 0.0113 vs
  1.9673 +- 0.0113; mean-log exponent 1.7314 vs 1.7317. c = 0.2: A = 31.85 +- 1.60 vs 31.87 +- 1.60,
  log rel rms 0.0496 vs 0.0497; power exponents 1.028 / 0.727 vs 1.028 / 0.728. Reproduced.
- Reviewer re-analysis (compare_e4a.py, per seed; plateau classified by time spent within 2% of each
  predicted saddle level; alternative events: t_a = first L < (L0 + L_saddle)/2, t_b = first L < L_saddle/2):
  * c = 0, 6 seeds with a resolved s=1 plateau at every sigma: exponent 1.981 +- 0.007 (dt2) /
    1.995 +- 0.004 (midpoint events) vs 2. Seeds 0 (both drops merge at most sigma), 1 and 3 (small
    saddle) differ; medians are robust (1.967).
  * c = 0.2: same misdetections as in the committed data (seed 1 at all sigma, seeds 0 and 6 at
    sigma = 0.3: plateau_rel 0.01); seed 3 visits the s=0.7 saddle. Clean big-saddle runs only:
    A(means) = 36.2, per-seed A = 39.0 +- 3.1 (8 seeds; 58.7, 40.5, 37.4, 39.7, 35.9, 37.0, 30.8, 32.2);
    the 4 seeds with a well-formed plateau at every sigma (2, 5, 7, 9): 37.4 +- 1.9 (sd 3.7).
    Prediction 34.1 -> measured slope is 10-15% above it (1.8 s.e.), not "31.9 +- 1.6, within error".
  * Log vs power law: on per-sigma means of the clean runs the two fits are equally good (log rel
    rms 6.4%, power rms 5.3% in log units, p = 0.67) because seeds are heterogeneous; per seed the log
    law wins in 7/8 seeds (the 4 cleanest: log 0.6-2% vs power 6-10%). The law change sigma^-2 -> log is
    confirmed; the slope agreement is looser than stated.

## 4(b) random MPS-4 teacher 103 (the case where HSVD and the rank-1 critical point differ most),
sigma in {0.02, 0.0356, 0.0632}, seeds 0-5 = 18 dense runs; analyze_ttn.py (uses constrained.py)
- 17 plateaus in both rerun and committed tables, same ranks at every plateau, distances agree to
  ~2-5% (e.g. rank-1: dist_crit 0.0253 / 0.0324 / 0.0414 vs committed 0.0254 / 0.0322 / 0.0415;
  dist_hsvd 0.0973 vs 0.0973; crit-HSVD gap 0.0994; loss_crit 0.158769 vs loss_hsvd 0.163743).
- 3/3 rank-1 plateaus closer to the constrained critical point than to HSVD (by ~3x); 16/17 plateaus
  overall (the [2,2,1] plateau at sigma 0.02 seed 2 is closer to HSVD: 0.0025 vs 0.0033, in both).
- Distance to the critical point shrinks only slowly with sigma for this teacher (0.041 / 0.032 / 0.025
  at sigma = 0.063 / 0.036 / 0.02); the committed median trend 0.042 / 0.005 / 0.004 mixes teachers
  (only teacher 103 has a rank-1 plateau at sigma = 0.063).

## Step 2 misaligned, alpha 1, theta 20, MPS-4, sigma 0.1, seeds 0-5 (run_generic.py + predict_rank1.py
+ analyze_plateaus.py)
- Bitwise reproduction of the committed rows (t1, t2, plateau levels, cosines) and of the predicted
  rank-1 critical value 0.422870 (rel. 0.073294) and race value.
- 4/6 runs have a resolved rank-1 plateau: level = predicted critical value to 2e-4..9e-4, |cos| with
  the predicted critical tensor 0.99992-0.99999. Seeds 0, 1: plateau not resolved (< 0.2 time units;
  |cos| 0.95-0.98 at t1), as the note's caveat on short plateaus says.

## 4(b) independent check of constrained.py (indep_constrained.py, ttsvd_all_teachers.py)
- L-BFGS on the rank-constrained TTN started from the TT-SVD truncation of each plateau state (not
  random starts) converges to the same critical point as constrained.py at all 17 plateaus (same loss,
  same distance to 1e-6); HOPM gives the same rank-1 critical value 0.158769 and distances. The
  constrained.py critical points are right, and loss_crit matches the committed value for all 5
  teachers (0.062444, 0.073409, 0.158769, 0.029570, 0.072835).
- But HSVD is a weak foil. The agent's root-to-leaves HSVD coincides (in distance from the critical
  point) with one TT-SVD sweep direction; the other sweep direction lies 0.0014 / 0.0050 / 0.0032 /
  0.0015 from the rank-1 critical point for teachers 101 / 102 / 103 / 104 (HSVD: 0.012 / 0.019 / 0.099 /
  0.012), i.e. closer than the plateau states are to the critical point (median 0.005; 0.025-0.041 for
  teacher 103). For teacher 103 the plateau is 0.028 / 0.035 / 0.044 from the left-to-right TT-SVD vs
  0.025 / 0.032 / 0.041 from the critical point (sigma = 0.02 / 0.036 / 0.063), and the TT-SVD loss is
  0.158771 vs 0.158769. Only teacher 105 separates the critical point from every truncation
  (TT-SVD 0.028 / 0.050, HSVD 0.057; plateaus 0.0035-0.0052 from the critical point). So the runs show
  "not this HSVD", and "critical point rather than an SVD-type truncation" cleanly only for teacher 105.
- At higher ranks ([2,1,1], [2,3,1]) TT-SVD and the critical point coincide to 1e-4 or better (the note
  already says HSVD and critical point coincide there).

## 4(a) sensitivity of the c = 0.2 slope to the plateau bookkeeping (refit_cut.py, data/review_data_teacher/refit_cut.txt)
Same rerun data, different but reasonable choices: agent's events, all runs: 31.9; agent's events,
s=1-saddle runs only (plateau_rel 0.2-0.5): 36.2 (means); window cut 0.1 L0 instead of 1e-2: 32.1 (all) /
34.2 (s=1 only, still includes some no-plateau seed-1 runs); occupancy-validated s=1 plateaus (42 runs):
32.9 (means, unbalanced across sigma) / 40.9 +- 2.7 (per-seed); 4 seeds with a clean plateau at all six
sigma: 37.4 +- 1.9 (per-seed log fits with 0.6-2% residuals, power-law fits 6-10%).
=> logarithmic law confirmed per seed; slope 32-41 depending on bookkeeping vs predicted 34.1, i.e.
agreement at the 10-20% level; the quoted 31.9 +- 1.6 overstates the precision.
c = 0: exponent 1.967 (medians, either cut) / 1.98-2.00 (clean seeds), robust.
