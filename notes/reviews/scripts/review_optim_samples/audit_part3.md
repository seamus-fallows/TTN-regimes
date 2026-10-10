# Claims audit part 3: SGD

## E. First plateau (claim 5a, sgd_runs.csv / sgd_summary.csv)
- The "first escape" statistic is t1 = component 1's crossing. In GF, sigma = 0.3 seeds 1, 3 learn
  component 2 first (t2 = 2.42/2.39 < t1 = 3.02/2.88): for 2/5 inits at sigma 0.3 the "first plateau"
  numbers are second-event numbers. Under SGD, component 1 is first in only 43% of runs (sigma 0.1
  seed 1), 81% (seed 3), 88%/94% (sigma 0.03 seeds 1/3), 6%/8% (sigma 0.3 seeds 1/3).
- Reported "+0.29 +- 0.41 / +0.19 +- 0.24 / +0.28 +- 0.21 (mean +- s.e. over 5 inits)": the +- is the
  mean single-init SE. SE of the 5-init mean: <= 0.18 / 0.11 / 0.09 (noise-propagated) or 0.04 / 0.05 /
  0.05 (empirical over inits, under-dispersed). Delays are positive (1.6-7 SE) and 1.7 / 1.6 / 3.4x
  P7's +0.17 / +0.12 / +0.08 (clean inits 0, 2, 4: 1.5 / 1.8 / 2.8x); P7's mean delay is not resolved
  (consistent at 0.03, 0.1 within the propagated SE, ~2 SE off at 0.3), not "confirmed".
- With min-over-components first crossings, near-tie inits show earlier "first events" (sigma 0.1
  seed 1: -0.087 +- 0.016 at B = 256) - the same transient a_2 crossings as in G (part 4), not real
  shortening. With robust (last-crossing) events, "no shortening" stands.
- Spread exponent with the true first event: 0.526 +- 0.018 / 0.479 +- 0.020 / 0.463 +- 0.023
  (B sweeps, sigma 0.03/0.1/0.3), 0.50 +- 0.06 / 0.47 +- 0.05 (eta sweeps): still ~1/2. Per-init spread
  vs 1D formula: 0.6-4.7x (near-tie inits 2-5x), not uniformly "1.2-1.4x".
- Noise streams: generator seeded by (init seed, noise seed) only, shared across B and eta.

## F. Later plateaus (claim 5b)
- Random init (3 inits at sigma 0.03, 2 at 0.1): per-init T/T_GF at sigma 0.03, B = 1: 1.13, 1.27,
  1.44 -> 1.28 +- 0.09 (SE over inits), not +-0.05 (pooled-run SE). Per-init ordering follows T_GF
  (10.8, 18.5, 23.1) as P9 predicts (c = 35: 1.11, 1.21, 1.28).
- "within errors of 0 for eta/B <= 1.25e-4" is wrong: sigma 0.03, B = 4 (eta/B = 1.25e-4): +4.2 +- 0.4%
  (= P9 c35: +4.0%); sigma 0.1 B = 16-128: +0.8-2.6% per init (B=128 seed 0: +1.5 +- 0.3%) where P9
  predicts <= 0.3%.
- P9 aligned test (p9_sgd.csv): B = 16 and B = 64 runs use the same generator seed (30000 + seed) and
  consume identical samples per unit time -> pathwise coupled (per-seed t2 agree to 0.01-1.5 while the
  across-seed SD is 0.45-2.6). The "0.5% agreement" is a coupled-path comparison (tests first-order
  equivalence of 4 small steps vs 1 large step), not 8 independent samples; trapping/eta/B-only tested
  only at B = 64 for b0 < 0.01.
- c_gamma is obtained by inverting -ln(1-x)/x per run: 28-47 across seeds at fixed x (b0 = 0.01), mean
  34 -> 45 with x; the law's shape is not independently tested ("qualitative" is the right tag).
- P10 (claim 8): b0 = 0.001 (1600 time units): min a1 < 1/2 in 1/4 seeds at 0.076 eta*_ms (0.495),
  2/4 at 0.15 (0.46, 0.04), 2/4 at 0.30 (-0.07, -0.02); b0 = 0.003 (665 units): 3/4 at 0.30;
  b0 = 0.01: one seed (coupled B=16/64) at 0.25-0.3.
  "erased even at 0.15 (min a_1 = 0.04, -0.07)" mixes 0.15 (0.04) and 0.30 (-0.07). Fractions are of the
  full-minimum eta*_ms; the learned-block threshold P10 asks for is larger (principal submatrix of G_B),
  so the failure is if anything stronger. Conclusion OK for this teacher; 4 seeds per point.
