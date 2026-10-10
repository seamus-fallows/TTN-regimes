# Claims audit part 4: order flips, finite samples, misc

## G. Order flips / "a flip can remove a whole plateau" (finding 5, notes 3.2, SGD 'joint' runs)
- Events use the FIRST upward crossing of a_r = 1/2 (common.crossing_times); diagnostics.md requires the
  last upward crossing when a transient exceeds 1/2. All flipped GD runs except sigma 0.3 seed 37 have
  max_cross = 2 in gd_sweep.csv.
- Reviewer re-run (flip_check.json, per-step overlaps): sigma 0.1 seed 1: GF-like GD (0.01 eta_max)
  a_2 peaks at 0.487 during the first event, learned at 23.24; at 0.3 / 0.9 eta_max a_2 peaks at
  0.525 / 0.591 at t = 7.8 / 8.0, falls back, learned at 23.34 / 23.57. sigma 0.1 seed 3 at 0.9:
  transient at 8.04 (peak 0.564), learned at 17.06 (GF 16.61). sigma 0.03 seed 12 at 0.3 / 0.9:
  transient at 24.27 / 24.54, learned at 55.88 / 56.16 (GF 55.77). No plateau is removed in any case;
  the "flips" are transient threshold crossings. sigma 0.3 seed 37 (max_cross 1): a_3 crosses at 11.69
  but L_end = 0.4999 = GF's, i.e. also transient.
- SGD re-run (sgd_flip_check.jsonl, sigma 0.1 seed 1, eta 5e-4, new noise seeds 100-105, T = 30):
  B = 256: 2/6 transient a_2 crossings at 7.6 (peak 0.51); B = 16: 2/6 (peak up to 0.675, a_1 dips
  below 1/2 once); all 12 runs learn component 2 at 22.6-24.5 (GF 23.24).
- SGD sweep stops 25 checks (0.05 time units) after the second first-crossing (stop_after: 2), so the
  225 'joint' runs cannot be distinguished from transient crossings; the second-plateau statistics
  exclude them on this basis (selection on a misclassified variable).
- Event-2 shift outliers at >= 0.3 eta_max (e.g. -110 eta at 0.9) are the same artifact.

## H. Finite samples (claim 7)
- Recovery 5/5 from n = 256 (chi 4) / 384 (chi 16) confirmed; mean eff. rank at n = 256 is 2.1 for
  sigma 0.3 (exactly 2 from n = 384), 2.0 for sigma 0.5.
- No GD run met the stopping tolerance (all max_steps). Centred sigma 1.5 / 2 at n = 512: train loss
  2.6e-3-5.4e-2 / 3.7e-2-0.16 of initial (note says ~1e-3-1e-2): the 0.53 / 1.51 test errors are
  unfinished fits. "Monotone degradation with init scale" mixes plain small-init and centred models
  and fit quality at one n; within small init it is not monotone (sigma 0.3 vs 0.5: 4.5e-5 vs 2.5e-5 at
  n = 512, 1.9e-3 vs 1.5e-3 at 256).
- Lazy limit chi = 4 at n in {32, 64, 128, 256, 512}: three identical copies per (seed, n) (blocks
  chi4_centred and chi4_centred_large both compute it; init direction identical) -> n_seeds = 15,
  SE too small by sqrt(3); figure says "5 seeds".
- Lazy-limit failures are dimension counting (tangent dim 112 < 256 at chi 4; min-norm interpolation in
  R^256 needs n >= 256 for a generic target at chi 16): expected, should be framed as such.
- Small-init "GD" caps the relative step at 1% near the origin (normalized GD on the first plateau).
- One teacher (s = (3, 2), seed 101), 5 seeds with data and init seeds coupled.

## I. Other checks
- Prop 1 proof checked (correct; J_v J_v^T factorization, exponent bookkeeping n_c, K - n_v; Q_e = 0
  <=> orthogonal gauge when chi = R since M D M = D with M SPD forces M = I).
- 1D map, backward-error variational equation, event-shift formula, SGD Ito mean/variance
  formulas re-derived: correct. Typos: small-a shift drops R^2/s^2 from kappa; k = 2 collapse condition
  should read eta q 3^n s_new ~ 2B.
- k = 2 saddle: exact balanced saddle has lambda_min = -0.519615 (doubly degenerate; reviewer
  k2_exact_saddle.py), = prediction. Plateau-midpoint values -0.494..-0.511 at sigma 0.01-0.02 are
  2-5% less negative, seed-dependent, not explained by a2_mid; classification (strict) unaffected.
- SGD MS stability: tensor-space operator code checked; equals the theory note's independent
  pattern-basis values (P8). "0.00107 B for B <= 16" is 0.00107-0.00100 per unit B (-6% at 16).
- EoS selection with imbalanced init: selected sharpness = min(lambda_GF, 2/eta) within 1-2% for
  converged runs; failures for root x6 start at 0.5 eta_max (not > 0.75); conditioned on convergence.
