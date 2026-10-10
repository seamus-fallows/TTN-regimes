# real-data claims audit: working notes, part 6 (MPS with other pixel orders; attribution, creep)

## MPS, sigma = 0.1, seed 0, raw, chi = 4 (audit/runs, audit/runs3)
| chain order | t_a | first rank-2 edges (tau) | drops' rho | tau L<=0.25 / 0.2 / 0.15 / 0.12 | L(1e4) | acc |
|---|---|---|---|---|---|---|
| snake | 8.36e15 | [5] (9.8, inside drop 0) | 3.0, 1.7, 3.8, 2.5 | 12 / 52 / 296 / 518 | 0.0891 | 0.957 |
| scrambled0 | 2.48e16 | [3,4] together (226.5) = a 3-core route | 16.6, 8.5, 12.4, 30.0 | 299 / 575 / 1560 / 4366 | 0.0840 | 0.962 |
| scrambled3 | 1.44e16 | [13] (14.1) | 3.1, 9.3 | 20 / 50 / 377 / - | 0.1283 | 0.929 |
| (binary quadtree, same seed) | 1.10e15 | [3,4] (150) | 15.7, 19.8, 290, 12.2 | 156 / 333 / 2156 / 2245 | 0.0957 | 0.950 |
- The MPS advantage is order-dependent: scrambled0 chain is slower than the binary quadtree to L <= 0.25
  and 0.2 (299 / 575 vs 156 / 333) and is S2S-like (rho 8.5-30) with a 3-core first route; scrambled3 chain
  ends worse than the quadtree (0.128 vs 0.096). "Route degree (shape) beats locality" is not established:
  with a scrambled chain the k = 2 routes can be too weak and a 3-core route goes first.

## attribution (audit/attrib_summary.json)
- scrambled2 seed 0 sigma 0.5: route [10,11] crosses at tau 151.7 (141.5 after drop 0) during creep,
  inside no drop window; repo assigns gap 560 (drop 1 = different edges 2, 6, 12, 13). Together with
  quadtree seed 3 (sigma 0.3: 128.9 assigned vs 12.4 true; sigma 0.5: 34.4 vs merged) this explains low
  per-seed exponents (scrambled2 0.74, quadtree mean 1.01).

## creep (audit/runs): loss decrease between consecutive drops' widened windows, relative to the next drop
- quadtree 71%, 47%, 22%; rowmajor 43%, 23%; scrambled3 36%; scrambled4 15%, 88%, 53%; scrambled5 115%;
  scrambled6 73%, 274%, 8%; scrambled7 246%, 4%, 13%; scrambled8 20%, 61%, 471%; quadtreeT 226%.
  Plateaus are far from flat (claim 6 understated, not overstated); much of the loss decrease happens
  between detected drops, which also lets routes rise without their own drop (scrambled2 s0.5 above).
