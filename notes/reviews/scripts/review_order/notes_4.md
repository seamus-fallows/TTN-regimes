# review_order notes (part 4)
- h1b_saddle_b4 rerun (24 runs, agent code, env vars): all rows identical to committed (t1, lam_min, a2_reup...).
  Asymptotic per-seed slope (sigma 1e-4, 1e-6): 2.525 +- 0.006 / 5.399 +- 0.013 vs 1/|lam| 2.520 / 5.386 (0.2%).
- h1b_natural_b4 rerun (16 runs = first 8 wrong seeds at sigma 1e-2, 1e-4, rho .8): identical rows; 16/16 rotate;
  |lam|(t1-t2) = 1.07 (1e-4) vs 1.08 (1e-2): flat. 5/8 rows at 1e-4 have NaN lam_min (unresolved midpoint) - not counted in fits.
- multi2 rerun MPS4 rho .8, 40 seeds: R=3 P(greedy order) .35 (model .344) vs committed .354 (.355); race exact
  order .85 vs .856; R=4 stalled 33/40 (82%) vs 149/200 (75%).
- chi scan indep MPS8 uncapped chi=4, 150 seeds: sd log tau .728 +- .030 vs .769 +- .017 (500 seeds): ok.
