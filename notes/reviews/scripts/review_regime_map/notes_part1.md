# regime_map reproduction review - working notes (part 1)

## README run instructions
- Step 1 `sweep.py ... --n-proc 340` fails on Windows: run_regime.run_sweep uses get_context("fork")
  (ValueError: cannot find context for 'fork'). Default raw dir /root/outputs/regime_map (Linux path).
  Reviewer driver (scratchpad driver.py) calls run_regime.run_point/safe_run with spawn; configs expanded
  with the committed sweep.expand.
- Step 2 `lazy_predict.py b4 ...` as written refuses (results/lazy_pred_b4.csv exists); out dir only via
  positional args 8-10. pilots/lazy_pred_b8.py has no out option and silently overwrites
  experiments/regime_map/results/lazy_pred_b8.csv; uses torch.set_num_threads(4) x Pool(12).
- Step 3 make_figures.py has --out for figures but the raw-data root is hard-coded
  (analysis.OUT = /root/outputs/regime_map); must monkeypatch to analyse another raw dir.

## Committed-CSV consistency
- sigma_c_vs_K.csv n_unresolved counts only status == "unresolved"; "unresolved+stall_below" is missed
  and n_resolved = count of non-NaN sigma_c. mps8 (K=8): 5/6 seeds unresolved+stall_below, CSV says
  n_resolved 6, n_unresolved 0; mps12: 6/6 unresolved, CSV 0; binary16: 5/6, CSV 0; mps10: 5/6, CSV 2.
  summary.png right panel labels "6/6" (resolved/seeds) on open (unresolved) markers.
  REPORT table marks only K = 12, 15 as unresolved; K = 8 (1.01) and K = 10 (0.90) are 5/6 unresolved too.
- u_0c = sigma_c^K / sqrt(D) from sigma_c_vs_K.csv medians (centred chi=4): K=3 0.0012, 4 0.015,
  5 0.016, 6 0.034, 7 0.0085, 8 0.069, 10 0.010, 12 0.0060, 15 0.0012; spread x11 over K=4..12
  (alpha_c = sigma_c^K spread x19). Bracket at K=15 [0.63,1.36] -> u_0c in [4e-6, 0.4].
- chi_b4 per-seed CSV: S2S slope chi>=16 refit -0.528 +- 0.053 (note: -0.524 +- 0.045); chi>=9
  -0.528 +- 0.061 (note -0.535 +- 0.060); lazy slopes reproduce exactly (-0.543 +- 0.015, -0.522 +- 0.020).
- sigma_chi_b4 chi=12: seed 4 has NaN rho-boundary (deepest run not above rho=3) and is silently
  dropped from the 0.088 geometric mean (5 seeds). chi_b4 chi=12 seed 4: label boundary 0.0126 vs
  aligned prediction 0.075 (ratio 5.9) - the "0.97-1.08x" statement holds for medians, not every seed.
