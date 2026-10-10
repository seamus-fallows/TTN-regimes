# Claims audit part 2: above eta_max, centred model

## C. Above eta_max (claim 3)
- binary4 excess loss (L_tail - L_GF(T)) vs 9 x map loss: 1.05: mean 0.13061 (per run 0.1297-0.1316)
  vs 0.13064; 1.1: 0.2472 (0.2408-0.2550) vs 0.2479; 1.2: 0.4496 (0.4424-0.4567) vs 0.4496; 1.3 (period
  4): 0.5527 (0.5007-0.6152) vs 0.5574. Means: 3-4 digits; per run 0.7-2% (1.05-1.2), 10% (1.3).
  No committed script produces this comparison (reproduced here from gd_sweep.csv + map1d_attractor.csv).
- MPS8 means are 6% (1.05), 17% (1.1), 20% (1.2) below the map: seed 3 never learns component 1 within
  T (excess 2e-5 at every ratio). "agree to a few % up to 1.1-1.2" holds only after excluding it.
- binary4 1.5: 23/25 diverged, 2/25 (sigma 0.3) reentry; 1.75/2.5/3.0 only 15 runs (sigma >= 0.03).
- Map regimes from a single x0 = 0.3 are interleaved: above the first divergence ratio, 30 (K=3), 45
  (K=7) grid ratios are still 're-entry'; "diverges from 2.16/2.41/2.72/2.86" = first occurrence only.

## D. Centred model across the lazy transition (claim 4)
- GF references are not converged at T: L_GF(T)/L0 = 0.06-0.10 (sigma 1, T=5), 0.06-0.17 (1.4, T=3),
  0.10-0.55 (2.0, T=2). lambda_GF is the sharpness at an arbitrary T.
- Ratio grid 0.5/0.7/0.9/1.05 gives 10-40% resolution; per-seed 2/lambda_GF spans 0.45-0.72 (1.4) and
  0.14-0.30 (2.0) x eta_max; only 2 lazy-side sigmas -> sigma^{-2(K-1)} not tested.
- In the agent's own data: sigma = 0.5, seeds 1, 4 at 1.05 are 'gf_like' (monotone tail) with loss
  25-100x GF's and stab 0.76: gf_like does not check loss. At sigma = 2, 9/33 runs above the per-seed
  2/lambda_GF end at or below GF's loss; 6/11 'reentry' runs do, with 1.5-3.5x lower sharpness.
- Reviewer re-run (centred.jsonl, T = 20 = 10x agent's): GF at sigma 2 still stalls at L/L0 = 6-17%
  (chi = 3 < cut dim 9: the lazy model cannot fit W*). GD at sigma = 2: ratio 0.2-0.3 (1.0-2.2x the
  per-seed 2/lambda_GF) converges to exactly GF's loss in 7/10 runs with sharpness*eta/2 = 0.71-0.98
  (flatter point selected), 1/10 chaotic (seed 1 at 0.3), 1 ends 14% above GF (seed 3 at 0.5), seed 4
  converges at 0.5 (1.7x edge). Divergence only from 0.5 (seeds 0-2: 2.5-3.6x edge). sigma = 1.4:
  ratio 0.7 (1.0-1.5x edge) 5/5 converge to GF loss; 0.9 (1.3-2x edge) 3/5 converge, 2 oscillate;
  1.05: 1 diverges, 4 oscillate. => above 2/lambda_GF the lazy side shows catapult/EoS selection
  (Lewkowycz et al. 2020) over a window ~1.5-2.5x wide before divergence; "steps above the limit never
  give a bounded oscillation (runs re-enter or diverge)" and "GD limit = 2/lambda_GF" are not supported.
