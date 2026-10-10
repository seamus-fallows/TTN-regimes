# review_order notes (part 3)
- MPS4 sigma scan indep (400 seeds): wrong counts 74/158/196 of 400 at sigma .1/.03/.01/.003 = committed m4_sigma
  table exactly; 100% of seeds same winner at all four sigma. Agent-code rerun vs indep: 612/612 winners.
- binary4 (m=2) indep, all 1000 seeds, chi=4: wrong counts 41/52/72/91/119 (rho .8, sigma 1e-4..1e-2) and
  182/293 (rho .9, 1e-4/1e-2) = committed sigma_table exactly. Own binomial-ML fit: gamma = 0.232 +- 0.026
  (committed 0.2321 +- 0.028; theory 0.25); rho .9 two-point 0.103 +- 0.010 (committed 0.1046; theory 0.111).
  Wrong seeds at 1e-4 are a strict subset of those at 1e-2 (0 exceptions). Agent rerun (200 seeds x 2 sigma): 400/400.
- chi scan indep MPS4 uncapped chi=32 (500 seeds): sd log tau 0.14498 (committed 0.145); race P at 6 rho identical.
- BUG: h1b.py with --raw/--res fails on Windows (spawn): workers re-import common with argv stripped -> RAW
  = /root/outputs/order -> FileNotFoundError at np.savez (all 3 h1b configs lost ~2.5 min each). Env vars work.
- README: analyze_multi.py multi -> FileNotFoundError (raw name is multi2). Confirmed.
