# part 2: analysis-script reproduction and independent MPS

## route_amplitudes.py on my reruns (vs routes_raw.json, same run names)
share(k<=5): quadtree s0.05 0.911/0.911, s0.1 0.916/0.916, s0.2 0.928/0.928, s0.3 0.937/0.938, s0.5 0.938/0.938;
scrambled0 s0.1 0.029/0.029, s0.5 0.097/0.098. best-k3 lambda and max lambda equal to 3-4 digits (identical when tau_mid coincides).
## plateau_hessians.py (proj lambda_min, mine/ref)
quadtree s0.05 -0.01438/-0.01438, s0.1 -0.0267/-0.0260, s0.2 -0.0502/-0.0502, s0.3 -0.0664/-0.0639 (seed-0 exponent ~0.86);
MPS s0.05 -0.0567/-0.0563, s0.1 -0.0573/-0.0567, s0.2 -0.0577/-0.0571 (sigma-independent: strict). n_neg 1 at s<=0.1 quadtree.
## Independent MPS s0.1 seed0
t_a 8358842052475747 vs 8.358842079e15 (3e-9); drops 12.23, 50.58 (agent 11.35, 49.70; constant +0.88 offset of the tau
origin: agent's phase-1 ends at the first DP5 step below the 1% floor, mine at the exact crossing) -> gap 38.35 = 38.35;
first rank-2 edge [5] at 10.66 (agent 9.79), then edge 4 at 41.07 (agent 40.18).
## Measurement caveat found (MPS first route)
MPS edge 5 data-weighted s2/s1 never falls below the 0.05 threshold at sigma >= 0.2 (min 0.06 at s0.2, 0.146 at s0.5;
0.03 at s0.1, 0.015 at s0.05): rank_events() returns t[0] (tau = 0) and the run counts as "single edge first".
Agent's own scaling.json grank_mixed shows seed 4 at all 7 sigma and seed 0 at sigma >= 0.2 with rank time ~ -10
(= tau 0): ~10 of the 34/35 "single edge" MPS runs are never-below-threshold events, not resolved rank increases.
## Fit weighting (analysis standard 5)
scaling.py fit_seed / events_table.fit: power law fitted by OLS in log space, log law by unweighted OLS in linear
space, both scored by relative rms -> log law handicapped. Refit with relative weights for both: log-law rms drops
2-4x (e.g. quadtree seed0 k=3: 0.284 -> 0.159; scrambled0 seed0: 1.10 -> 0.30) but power law still wins (0.002, 0.055).
Notes text says both fitted "in linear-gap space with relative residuals": not what the code does.
