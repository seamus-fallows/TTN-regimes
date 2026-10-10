# regime_map reproduction review - working notes (part 3): reruns with committed code

Driver: scratchpad repro_rev/driver.py = run_regime.safe_run under a spawn pool, configs from the committed
YAMLs via sweep.expand, outputs data/review_regime_map/<exp>/. Comparison via analysis.load/per_seed_table/
sensitivity_table with analysis.OUT monkeypatched.

- sigma_chi_b4 centred chi = 36 seed 0 (19 sigmas): s2s_2/3/5 = 0.073307/0.050616/0.031192 vs committed
  0.073302/0.050614/0.031192; lazy_0.05/0.1/0.2 = 1.061224/0.922362/0.794605 vs 1.061228/0.922369/0.794608.
  Agreement to ~1e-5 relative (platform/integrator noise).
- imbalance_b4 chi = 16, alpha = 3.375, 9 lambda x 6 seeds: medians 8.35e-5, 7.35e-4, 0.00689, 0.0363,
  0.167, 0.452, 0.533, 0.502, 0.497 = note's 1e-4, 7e-4, 0.0069, 0.036, 0.167, 0.45, 0.53, 0.50, 0.50.
  Labels: lazy 6/6 for lambda <= 0.3, lambda = 1: 5 rich / 1 lazy, rich 6/6 above. Local slopes
  1.98, 1.86, 1.51, 1.27 -> note's first slope 2.18 is the undivided ln ratio (ln 8.86); correct 1.98.
- lazy_predict.py (out dir data/review_regime_map/lazy, chi 3, 9, 36, seeds 0-1): residual, kappa,
  rel_dtheta1 identical to lazy_pred_b4.csv to all printed digits.
- large_chi_b4 centred sigma 1.5 (partial, 4-11 runs per lr): slopes consistent with committed.
- chi = 12, seed 4: overlap a_2 has a transient above 1/2 (max 0.556) while component 1 escapes; at
  sigma = 0.01, 0.0149, 0.0158 the logged event time of component 2 is the transient (t2 = 54.5 vs
  t1 = 55.4), rho_1 = 0.61 -> 'rich'; at 0.004-0.0107 the transient falls between log-grid points
  and the label is S2S (rho 13-41). With the last upward crossing (diagnostics.md rule) all are S2S
  (rho_1 = 13.8 at 0.01). Cause: run_regime.refine_events keeps the coarse first crossing when it lies
  before the final-rise window (docstring says so), contrary to notes/diagnostics.md; and it uses only
  the log grid (not the fine rows), so detection depends on grid sampling -> erratic labels.
  Effect: chi_b4 chi=12 seed 4 boundary 0.0126 (pred 0.075); sigma_chi_b4 chi=12 seed 4 NaN, silently
  dropped. Independent implementation: 0 of 60 (chi, seed) sweeps (960 runs) had a transient above 1/2.
