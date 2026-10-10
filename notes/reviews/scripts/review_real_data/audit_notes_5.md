# real-data claims audit: working notes, part 5 (new runs: amplitudes, growth exponents, attribution)

Outputs: data/review_real_data/audit/{audit_summary.json, audit_routes.json, audit_table.json,
attrib_summary.json}; runs in audit/runs (batch 1), audit/runs2 (attribution), audit/runs3 (MPS orders).

## Binary orders, sigma = 0.1, seed 0 (route amplitudes at the rank-1 plateau, rank1_rel_err <= 0.016)
| order | first route (tau after drop 0) | best k=3 pair = first? | best3 | share k<=5 | gamma (P4; k=3 -> 1.33) | L(1e4) |
|---|---|---|---|---|---|---|
| quadtree | [3,4] 134.5 | yes | 0.106 | 0.92 | 1.35 / 1.36 | 0.0957 |
| scrambled8 | [3,4] 70.6 | yes | 0.121 | 0.88 | 1.29 / 1.31 | 0.1192 |
| rowmajor | [3,4] 31.3 | yes | 0.333 | 0.93 | 0.93 / 0.90 | 0.1355 |
| scrambled4 | [7,8] 81.3 | yes | 0.113 | 0.20 | 0.99 / 1.03 | 0.1429 |
| scrambled7 | [10,11] 1870 | yes | 0.017 | 0.47 | 1.40 / 1.41 | 0.1483 |
| scrambled6 | [3,4] 545 | yes | 0.034 | 0.02 | 1.39 / 1.38 | 0.1699 |
| scrambled3 | [10,11] 50.1 | yes | 0.280 | 0.80 | 1.27 / 1.24 | 0.1866 |
| scrambled5 | [10,11] 276 | yes | 0.124 | 0.17 | 1.46 / 1.45 | 0.2243 |
| quadtreeT | [0,1] 698 | yes | 0.024 | 0.11 | 1.27 / 1.29 | 0.2257 |
- Amplitude rule out of sample: 9/9. First-route gap vs best3: log-log r = -0.95, slope -1.25,
  gap x best3 = 8.6-34.
- Final loss vs k<=5 share: Spearman -0.75 (p = 0.02, n = 9): the share predicts L(1e4) across orders;
  vs k = 3 share -0.25, vs best3 -0.15. Random orders' share: 0.80, 0.20, 0.17, 0.02, 0.47, 0.88
  (mean 0.42 = pair-count chance 12/28 = 0.43); the 3 committed orders (0.013-0.12 at seed 0) are the
  low tail, so "scrambled 21%" is not representative of random orders.
- The equally local transposed quadtree has share 0.11 and the worst L(1e4): the effect is about where
  label-relevant interactions fall on the tree (orientation), not locality as such.
- Early speed: tau(L <= 0.25) quadtree 156; 6 of 8 other binary orders are faster (48-139).

## first_route attribution bug, confirmed (attrib_summary.json)
- quadtree seed 3 sigma 0.3: route [3,4] crosses at tau 28.0 (12.4 after drop 0), inside no drop window;
  drop 0 already goes 0.495 -> 0.220 (escape + route merged); repo rule assigns gap = drop 1 - drop 0 =
  128.9 (drop 1 = a different route, edges 1, 2, 5).
- sigma 0.5: crossing inside drop 0's window (merged); repo assigns 34.4 (next event).
- seed 3 exponent: 1.235 clean (0.05-0.2) vs 0.652 with the two misassigned points; with seed 3 corrected
  the quadtree mean would be ~1.16 instead of 1.01.
