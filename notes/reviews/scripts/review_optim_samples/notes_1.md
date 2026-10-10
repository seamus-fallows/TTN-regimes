# Reproduction review notes (optim_samples), part 1

## README run instructions
- `gd_sweep.py configs/gd_sweep.yaml` run from the repo root (as README says) fails:
  FileNotFoundError 'configs\gd_sweep.yaml' (config path is relative to the script folder).
  Needs `experiments/optim_samples/configs/...`.
- Default --raw-dir/--out-dir are /root/outputs/... (on Windows: C:\root\outputs\...), created at import
  time by common.py (RESULTS.mkdir). Must pass both flags off the remote machine. --workers default 30.
- Scripts write __pycache__ into experiments/optim_samples when imported; ran from a copy
  (data/review_optim_samples/repro/code) with PYTHONDONTWRITEBYTECODE=1.

## Committed-CSV consistency
- "170/170 runs at <= 0.95 eta_max": committed gd_sweep.csv has 195 such runs for seeds 0-4 (675 incl.
  order-flip seeds 5-44); all are gf_like. Count in note/report does not match, conclusion holds.
- P5 excess loss (0.1306 vs 0.1306 etc.) is not computed by any committed script (ad hoc); recomputed from
  committed gd_sweep.csv + map1d_attractor.csv: 0.1306/0.1306, 0.2472/0.2479, 0.4496/0.4496, 0.5527/0.5574. OK.
- gd_shift_prediction.csv lacks the `centred` column the current predict_shift.py writes -> produced by an
  older script version (minor; analyze_gd tolerates it).
- Event-shift "3 digits per seed at eta <= 0.1 eta_max, all events": committed data give median
  |meas/pred - 1| = 0.2% at 0.1, 0.08% at 0.03, 0.02% at 0.01 eta_max; 84% of 199 event comparisons within
  0.3%, but outliers: binary4 sigma=0.03 seed1 event2: -0.050 meas vs -0.147 pred (66%) at 0.1;
  -0.119 (19%) at 0.03; -0.137 (6%) at 0.01 -> converges to prediction as eta->0, but not 3 digits at 0.1.
  Others: sigma=0.003 seed3 ev2 13%, sigma=0.01 seed3 ev2 5.5%, sigma=0.01 seed1 ev2 4.7%, sigma=0.003 seed1 ev2 3%.
- eta_ms = 0.00107 B "for B <= 16": actual 0.001070, 0.001066, 0.001056, 0.001038, 0.001004 per unit B
  (6% below 0.00107 B at B = 16).

## Centred model re-run (sigma 1, 1.4, 2; seeds 0,1; 7 ratios)
- Below eta lambda_GF/2 < 1 every run gf_like (22/22). Above: oscillating / reentry / diverged, or "gf_like"
  after GD moved to a flatter point (stab_end < 1, tail loss well above GF: EoS selection, not GF-like).
- GF references at sigma >= 1 are NOT converged at T (loss_gf_end 0.39-1.66 of L0 = 7): lambda_GF is the
  sharpness of the GF state at T, not of a minimum.
- "on the lazy side, steps above the limit never give a bounded oscillation": my sigma=2 seed1 r=0.2 is
  classified oscillating (stab 1.015); committed CSV: sigma=1.4 8/21 and sigma=2.0 2/33 unstable runs oscillating.
- analyze_samples.py pools the lazy-limit rows of chi4_centred (sigma 1) and chi4_centred_large (sigma 1.5, 2):
  identical values (same init direction) counted 3x -> samples_summary.csv n_seeds = 15 and test_se too small by
  sqrt(3) for chi=4 lazy at n = 32, 64, 128, 256, 512 (means unaffected).
- Committed per-seed values (N=8 d=2): chi4 small sigma=0.5 n=256: 5/5 < 1e-2 (max 5.1e-3); n=192: 4/5;
  chi16 small n=256: 3/5 (max 0.43), n=384: 5/5; chi16 lazy n=256: ~1e-21 (exact), n=192: 0.64-2.98;
  chi4 lazy: 1.55-41.7 at n = 128-384, rank_J = 112 (chi4) / 256 (chi16).
- BUG (analysis): common.gradient_flow records t10 = FIRST upward crossing of 0.1 (solve_ivp event e[0]); analyze_sgd.gf_reference
  uses width = t90 - t10 for rho, contrary to diagnostics.md (final rise = last upward crossing of 0.1 before 0.9). Components
  with a transient rise above 0.1 get a width spanning the whole plateau. Re-run GF refs: binary4 sigma=0.1 seed2 comp2 first
  crossing 7.55 vs final-rise start 12.51 (t90 13.29) -> rho 0.94 instead of ~6.9. Committed sgd_runs.csv: rho_gf(0.1, seed2) =
  0.937 while its rival_peak = 0.118 (< 0.15, not a near-tie) -> wrongly excluded from the second-plateau (P9) statistics
  (sigma=0.1 uses seeds 0,4 only). Same pattern in re-run GF refs for (0.03,1), (0.1,1), (0.3,1), (0.3,2), (0.03,50), (0.1,50).
- analyze_samples.py crashes on Windows (cp1252 stdout) at the summary print (labels contain chi/sigma Greek letters) after
  writing samples_runs/summary.csv but before samples_curves.csv and fig_samples.png; works with PYTHONIOENCODING=utf-8.
