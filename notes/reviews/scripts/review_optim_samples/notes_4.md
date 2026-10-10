# GD sweep re-run (agent code, gd_review.yaml: binary4, sigma 0.01-0.3 seeds 0-2 + new seeds 50-52, sigma 0.003 seeds 0-1)
- 116 GD runs + 20 GF refs. Matched 95 runs to committed rows: event times, final sharpness identical to <= 1e-12 rel for
  ratio <= 1.2; class agreement 93/95: the 2 flips are sigma=0.3 ratio 1.5 seeds 1/2 (diverged <-> reentry swapped):
  chaotic regime, roundoff-amplified (same 2:1 proportions).
- Classes: <= 0.95 eta_max 50/50 gf_like (incl. new seeds at 0.95: 6/6); 1.05: 18/18 oscillating; 1.1, 1.2: 12/12 osc;
  1.5: 11 diverged + 1 reentry; 2.0: 12/12 reentry.
- Excess loss vs 9 L_map: 1.05: 0.1307 +- 0.0008 (n=18) vs 0.1306; 1.1: 0.2483 vs 0.2479; 1.2: 0.4495 vs 0.4496.
# Finite samples re-run (agent code, NEW seeds 5-7 = new data draws 505-507 and new inits), N=8 d=2
- chi4 small sigma=0.5: n=128 0/3 (test 1.0-3.1, rank 3.6); n=256 3/3 (1.0e-4..4.4e-4, rank 2.0); n=384 3/3 (2-5e-5, rank 2.0)
- chi4 lazy: n=128 17.5-40.4; 256 1.5-5.2; 384 1.4-2.7; rank_J 112 -> never recovers
- chi16 lazy: n=192 0.87, 3.1, 11.0; n=256 ~1e-21 (exact). Note: "worse than zero below d^N" not per-seed (0.87 here; committed has 0.64, 0.79)
- chi16 small: n=192 2/3; n=256 3/3 (max 7.6e-3)
# P9/P10 subset (agent p9_sgd.run, same seeds): b0=0.01 B=64 eta/B=2.5e-4 seeds 0,1: t2 = 71.7957 / 71.2845, min a1 0.863 / 0.463
  = committed to all digits. min a1 < 1/2 only in seed 1 (committed: 2/8 runs at this eta/B: 0.463, 0.438) -> "dips below 1/2
  within T ~ 70 (min a1 = 0.44-0.46)" quotes the worst seeds.
- P9 subset (cont.): B=16 eta/B=2.5e-4 seeds 0,1: ratio 1.4229/1.4283, min a1 0.861/0.438; b0=0.003 B=64 eta/B=1.25e-4 seeds 0,1:
  ratio 2.648/2.845, min a1 0.791/0.816 -> all equal to committed p9_sgd.csv rows to all digits.
# predict_shift re-run (13 GF refs): c_e equal to committed to 9e-11 (21 events); new seeds 50-52: measured shift/eta at 0.1
  vs c_e: events 1-2 within 0.03-0.4%; third events 0.6%, 0.8%, 2.1%, 5.5% (sigma=0.1 seed 52 ev3: 0.405 vs 0.384).
# SGD m.s. limit, independent (own Theta via exact multilinear Jacobian, own F, symmetric-matrix basis, 40 bisections):
  rank Theta = 45, lam_max 12.98024613276667; B=1: eta_ms = 0.00107017 vs committed 0.00107014.
- SGD m.s. limit, agent code subset (sgd_stability.operator_pieces/spectral_radius): B=1 0.0010701448272656521, B=16
  0.01605970320743281 = committed to 1e-15. Independent (own code) all B: 0.00107017, 0.00213113, 0.00422571, 0.00830772,
  0.0160597, 0.0527863 (B=64), 0.111970 (B=256), 0.154080 (B=inf) = committed to ~3e-5 (bisection resolution).
  F check: per-site Isserlis formula by MC (2e6 samples) 0.35% error; tensor-product composition exact (2e-16); a direct MC of
  the 81x81 F is too heavy-tailed to be a useful check (19-60% error at 2-4e5 samples).
# Independent finite samples (own binary-8 TTN, own teacher [3,2], own data; scratch indep_samples.py)
- rank J = 112 (chi 4), 256 (chi 16). Lazy chi4: n=128 7.7-46; 192 3.9-6.2; 256 2.2-3.0; 384 1.2-2.3 (3 seeds each).
  Lazy chi16: n=128 4.8-9.7; 192 3.0-10.5; n=256, 384 < 1e-20 (exact).
- Small-init GD (sigma 0.5, 60k steps, eta = 1/lambda_max, 1% cap): n=256 test 0.0022, 0.0065, bond ranks all 2; n=128 0.72 (ranks 3-4).
