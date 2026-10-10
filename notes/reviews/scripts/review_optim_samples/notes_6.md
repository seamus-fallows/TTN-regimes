# Further observations
- fig_stability.png, MPS N=8 panel: the 1/5 "converges (GF-like)" band at 1.05-1.75 eta_max is seed 3, whose GF learns only
  component 2 within T = 60 (loss_end 5.0 of 7.06, n_events 1): never reaches a minimum, so "GF-like above eta_max" is a
  short-T artefact, not convergence.
- P10 evidence (committed p9_sgd.csv): min a1 < 1/2 at eta/B = 2.5e-4 in 2/8 b0 = 0.01 runs (0.463 B=64 @0.30 eta*_ms,
  0.438 B=16 @0.25 eta*_ms); "erased at 0.15 eta*_ms (min a1 = 0.04, -0.07)": 0.04 is 1/4 seeds at 0.15 eta*_ms
  (b0 = 0.001, eta = 0.008), -0.07 is at 0.30 eta*_ms; all runs end with a1 ~ 0.95-1.06 (transient forget-and-relearn).
  Still a violation of P10's "none at <= 0.25 eta*_ms" (B = 16 run at 0.25 reproduced: 0.438).
- P9 new noise seed 10 (b0 = 0.003, B = 64, eta/B = 1.25e-4): T_SGD/T_GF = 2.29 (committed 4 seeds 2.65-3.83, mean 3.06).
- analyze_sgd.aligned_theory: for event 2 the residual of component 1 is taken as s_1^2 although component 1 is learned
  (minor; only the aligned block's event-2 predictions).
- P9 with NEW noise seeds 10-13 (agent p9_sgd.run): b0=0.01, eta/B=2.5e-4: B=64 1.391/1.346/1.317/1.345 (mean 1.350; committed
  1.386), B=16 1.382/1.322 (mean 1.352; committed 1.380) -> B-independence at fixed eta/B holds (0.2%); b0=0.003, eta/B=1.25e-4,
  B=64: 2.29/2.36 (committed 2.65/2.85/3.83/2.94, mean 3.06 +- 0.26) -> pooled 6 seeds 2.82 +- 0.23; c_gamma from -ln(1-x)/x:
  ~38 (b0 0.01) and ~42 (b0 0.003, new seeds) vs 40.1 / 45.1 committed. No run escaped faster than GF; min a1 0.69-0.83.
