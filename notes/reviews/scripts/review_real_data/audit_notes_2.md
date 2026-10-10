# real-data claims audit: working notes, part 2 (committed data, continued)

## MPS "first route" (claim 2, MPS part)
- scaling.json grank_mixed: edge 5's s2/s1 crosses 0.05 BEFORE the first loss-drop peak (gaps -0.5 to
  -10.7 at every sigma; chi = 8 the same). bond_spectra_raw_s0.1: edge 5 starts phase 2 at s2/s1 ~ 0.03.
  So the MPS first route is merged with the first escape; no gap is measured for it.
- The log law is shown for edges 4 and 3 only (edge 3 is not adjacent to edge 5; edge 6, adjacent, has
  slope 458 and log rms 0.149 vs power 0.166 = no discrimination). Edges 0, 1, 2, 7, 8, 9 prefer power laws
  (p 0.94-1.27). R2/R5 predict k = 2 (log) for every MPS edge: 2/10 support it.
- growth_order_raw_s0.1: MPS ranks spread outward from edge 5 as a wave (4: ~45, 3: ~80, 2: ~200,
  1: ~350, 0: ~2000; 6: ~800; 7-9: ~3500; 10-14 never within 1e4).
- Hessian at L = 0.230: after edge 5 has risen; 2-3 eigenvalues below -1e-3 lambda_max (several routes).

## Fashion / chi = 8 / 8x8
- fashion_tables.md: "first rise = sibling bottom pair" 0.20 (3/15) quadtree and scrambled, single edge
  0.80; the notes quote only "first two rising edges = siblings" (15/15, 14/15). On MNIST the "together"
  criterion (35/35) is the one quoted as evidence. MPS Kendall tau(level, time) = -0.24 on Fashion.
- Fashion: the MPS is slowest to reach L <= 0.25/0.2/0.15 at sigma 0.05 (117/124/374 vs quadtree 35/37/44,
  scrambled 21/22/26); "MPS fastest on every task" (REPORT caveats) is false.
- chi = 8: all 23 runs that hit the 2 h wall limit are quadtree (14) or MPS (9); every scrambled run ran
  to tau = 1e4. "train L(1e4)" for quadtree/MPS is L at tau_end < 1e4. At sigma 0.5 all 5 quadtree runs
  stopped early, so "scrambled fits better (0.055 vs 0.079)" is confounded by the horizon.
- chi = 8 first-route exponents 1.17 +- 0.06 / 0.97 +- 0.11: two-point slopes; chi8_scaling.json
  first_route is empty (n_seeds 0); the committed rank-gap two-point slopes are quadtree 1.26, 1.59,
  1.23, 1.13, 1.14 (1.27 +- 0.08) and scrambled 1.12, 1.07, 1.44, 1.16, 1.20 (1.20 +- 0.06).
- 8x8: 1 scrambled permutation, 2 seeds x 2 sigma; t1 "61.0000" from two sigma values.

## Regimes (regime_analysis.json)
- sigma = 1.4: 5/5 "rich" but n_drops = 1, 1, 0, 1, 0 (two runs have no detected event; label by
  elimination); horizon 405-417 for seeds 2, 4 vs 3000 for the others.
- L_lin floor per seed 0.128 / 0.112 / 0.096 / 0.090 / 0.099; S2S plain runs end at ~0.095 (sigma >= 0.1):
  the floor is NOT "well above what feature learning reaches" for seeds 2-4.
- lazy runs end at L(T) = L_lin(T) = 0.130-0.192, far above their floors: horizon-limited, so the
  0.931 accuracy and "they fit only the tangent space" are statements at a fixed kernel time.
- L = L_lin to 5-6 digits at sigma 2.8 and 4; at sigma 2 seeds 0, 1 differ by 0.0016, 0.0024 (1-1.5%).

## Smaller items
- notes 4.4 scrambled2 "0.91 +- 0.23 (2 seeds)" is stale: scaling.json/scaling.txt give 0.74 +- 0.21 (3).
- notes cite results/scaling_raw.json (does not exist; scaling.json).
- main.yaml lists scrambled1 at 4 sigma; data have 7 sigma for seeds 0-3 and 6 for seed 4 (34 runs).
- 4.9 "at sigma <= 0.1 binary runs are S2S (80%)": tables give 11/15 quadtree, 6/15 scrambled (57%).
- binary-tree routes between two bottom cores have odd k (3, 5, 7): "k = 4-5" for exponents 2.1-2.4 and
  "3.3 close to k = 5" are not route identifications.
