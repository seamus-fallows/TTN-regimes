# Claims audit: optim-samples round 1 (part 1: evidence tables)

All numbers below were recomputed from the committed CSVs in experiments/optim_samples/results/
(pandas), plus three small independent re-implementations (scripts in scratchpad
review_optim_samples/scripts, outputs in data/review_optim_samples/).

## A. GD stability boundary (claim 1)
- gd_sweep.csv, binary4 small init: 675/675 runs at <= 0.95 eta_max are class gf_like (195 with seeds
  0-4, 480 order seeds); 25/25 oscillating at 1.05. The "170/170" count in notes/REPORT matches no subset.
- "converge": 39/195 (seeds 0-4) end on the last plateau with L = 0.5 (L/L0 = 0.071 > 1e-3); seed 3
  never learns component 3 within T at any sigma (GF too). Class gf_like = monotone tail, not convergence.
- 'oscillating' at 1.05 can be assigned by sharpness*eta/2 > 1.001 alone (automatic once component 1
  is learned); the P5 loss match is the real evidence of a 2-cycle.
- Reviewer re-run (edge.jsonl; sigma 0.03, 0.1 x seeds 0-4; ratios 0.9/0.97/0.99/1.01/1.03):
  all 0.97 and 0.99 runs converge exactly like GF (final loss 1e-15 or the seed-3 plateau 0.5, stab =
  ratio); all 1.01/1.03 runs end on the 2-cycle with excess loss 0.0273 / 0.0801 = 9 x map loss (0.0273,
  0.0801) to 3-4 digits. Boundary at eta_max confirmed to +-1%.
- N=8 trees (sigma = 0.9) are not small init: GF sharpness 1.10-1.16x (binary8), 1.12-1.35x (MPS8,
  excl. seed 3) lambda_bal, not "8-12%"; GF references learn only 1-2 of 3 components by T (MPS8 seeds
  0,1,4: only comp 1; seed 3: only comp 2; binary8 seed 1: only comp 1).

## B. GD event shift (claim 2)
- First events: meas/pred = 0.9994-0.9999 (ratio 0.01), 0.998-0.9997 (0.03), 0.9945-0.9992 (0.1),
  0.95-1.01 at 0.9 (5% off, not "within 2%").
- Second events at 0.1 eta_max: sigma 0.003 seed 3: -0.845 vs -0.975 (13%); seed 1: -9.28 vs -9.56 (3%);
  sigma 0.01 seed 1: -3.35 vs -3.51 (5%), seed 3: 1.405 vs 1.332 (5.5%); sigma 0.03 seed 1: -0.050 vs
  -0.147 (factor 3; abs 0.1 eta). These are the near-tie seeds 1, 3.
- At 0.9 eta_max second events of seeds 1/3: 0.63, -0.60, 0.48, -5.5, 1.64 x pred (flips excluded);
  third events up to 8% off. fig_gd_vs_gf (b) plots medians with ylim (0.5, 1.5), hiding these.
- Linear extrapolation of meas(ratio 0.01, 0.03) to eta -> 0 recovers pred (e.g. -0.147): the O(eta)
  theory is right; the stated accuracy at finite eta is not.
- "MPS N=8 seed 2: -32.40 vs -32.48": CSV has -32.46/-32.43/-32.30 at 0.01/0.03/0.1.
- gd_shift_prediction.csv lacks the 'centred' column the final predict_shift.py writes: CSV predates the
  final script (contradicts "all committed results regenerated").
