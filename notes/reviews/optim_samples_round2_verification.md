# Verification of the optim-samples fix round (2026-10-10)

Recomputed every headline number of the fix round from the committed CSVs (pandas, ~5 CPU-min; no reruns).
**Verdict: ready for the report with caveats.** All 7 review items addressed; no conclusion changes.

Confirmed: 765/765 runs below 0.95 eta_max keep GF's persistent saddle sequence; chi = 9 centred model with 45
converged GF references, lambda_GF ~ sigma^(3.9648 +- 0.0136) (sigma = 1.41-16); 86/90 catapult runs end below
1e-6 L0; shift accuracy 65.8/89.5/97.1% of 526 comparisons within 0.3/1/5%; Richardson median 1.18e-5; P7 delays
+0.244 +- 0.096 / -0.007 +- 0.121 / +0.128 +- 0.070 vs predicted 0.167/0.121/0.085; P9 refit c_gamma = 32.62,
chi^2 378.6 (P9 law) vs 534.5 (pole) vs 929.2 (linear); P10 0/72, 1/36, 20/36.

Errata to apply when citing these results (the lead applies them in the report):
1. "0% forgetting at <= 0.24 eta*_ms" is contradicted by sgd_runs.csv: 1/50 at 0.235 eta*_ms (B = 2, sigma = 0.1).
2. "Persistent order differs in 4-8% at B = 1": all 10 flipped (and all 19 unresolved) runs are forget-and-relearn
   runs, not changes of escape order.
3. Random-init P9 match (+19.8 +- 5.1% vs +19%, sigma = 0.03, B = 1) uses 16/30 runs (the rest forgot); state it.
4. Tension with theory P3: chi = 9 catapult runs end at 0.58-1.00 x 2/eta per run (group means 0.75-0.97), not
   "slightly below 2/eta"; P3 is not simply confirmed.
5. The 2/lambda_GF edge holds only to grid resolution: say "between 0.95 and 1.2 x 2/lambda_GF". The Lewkowycz et al.
   comparison is 6-8/lambda vs their 4/lambda_0 (factor 1.5-2; mention). B = 16 vs 64 agree within 1.8 combined
   s.e. (B = 64 higher at all 3 escaped b0 = 0.003 points). The regime map's 3x line at sigma = 1 should be ~1.5x.
6. GF transient counts and the "t = 7.97" example are only in raw data (now in data/optim_samples/).
7. Nits: 15 (not 14) comparisons exceed 5%; first events at 0.9 eta_max median 3.0% (10% above 5%); aligned-init
   spread 0.77-1.21x; MPS8 P5 within 3.2%; "parameters move 30-70%" only at chi = 4 (chi = 16: 12-32%); bond-rank
   ordering fails at chi = 16 for n >= 256 (lazy limit exact, rank 2).
