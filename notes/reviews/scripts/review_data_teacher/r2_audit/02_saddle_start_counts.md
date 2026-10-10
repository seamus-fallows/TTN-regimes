# R2 audit note 2: saddle-start tallies, Hessian table, near-degenerate edges

- "Log law in 65/65 seeds" (13 teachers x 5) includes misaligned8_20 (eps 1e-3..5e-2 only, 1.7 decades;
  A = 47434 +- 4069 vs 1/|lambda_min| = 1975, i.e. 24x slower than the strict-saddle rate). A 2-parameter
  log fit "winning" over 1.7 decades with a slope 24x off is not evidence for the strict-saddle law.
  Clean tally: 60/60 seeds, 12 teachers, eps 1e-8..5e-2.
- "Binary/HT p = 0.999-1.013 in all 26 cases": 24 cases. misaligned8_20 binary8: p = 1.079 +- 0.006 from
  2 seeds, 12/29 runs never escaped by t = 1e7 (censored, excluded); ht8: 3/30 escaped, no fit.
- r2_saddle_hessian.csv 'type' column (library fallback rule lambda_min < -1e-3 lambda_max): MPS-8
  saddles of nonorth8_c0.45 (lambda_min -4.77e-3, lambda_max 8.0) and misaligned8_20 (-5.06e-4,
  lambda_max 177) are labelled "non-strict" with n_negative = 0; randbin_302 mps4 has n_negative = 2
  (projected spectrum has 5 negatives: -0.0764, -0.0360, -0.0025 x3); randbin8_303 13. The notes say
  "strict on every MPS student ... multiplicity 5 on mps4, 13-17 on mps8". Mathematically strict (|lambda|
  >> residual-gradient noise 1e-9), but the committed CSV contradicts the text.
- Near-degenerate edges (recomputed Hessians with the agent's saddle_for): randbin8_303 lambda_min =
  -0.18972 on edge (6,7) (multiplicity 1) vs -0.18267 (x3) on edge (4,5); the event fires on edge 4 in
  4/5 seeds at every eps (3/5 at eps <= 1e-7). 1/0.18267 = 5.474 vs measured 5.46 (global) / 5.43
  (local): the "+3.6%" vs 1/|lambda_min| is the wrong-edge comparison. randmps8_303: lambda_min -0.16424
  (edge 6, x1) vs -0.16178 (edge 1, x3); edge 1 fires first in 5/5 seeds for eps <= 1e-3: 1/0.16178 =
  6.181 vs local 6.20 (+0.3%). With the realized edge every non-degenerate MPS teacher agrees to <= ~0.4%.
- f-sensitivity of MPS slopes: up to 1.8% (misaligned8_45 57.69 / 57.41 / 58.47), not "< 1.5%".
- Escape times are the first grid time above threshold (no interpolation): + bias up to 0.5%
  (r2_saddle_resolution_check.csv covers only 3 teachers at eps >= 1e-3).
- Summary-table row 2 quotes the saddle-start misaligned_20 slope as "55.5 +- 1.1 vs 52.9": that is the
  base-range (eps >= 1e-3) value; the extended value in R2.1 is 53.90 +- 0.41, local 53.04.
