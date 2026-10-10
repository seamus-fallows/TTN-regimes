# real-data claims audit: working notes, part 1 (from committed JSON only)

## Fair refit of gap laws (events_table.json, per seed series)
Repo fits the log law by unweighted OLS in linear gap space but scores it by relative residuals
(power law is fitted in log space = ~relative). Refitting the log law and A/sigma + B with relative
weights:
- first sibling route (k = 3) series: log-law rms 0.15-1.20 (repo) -> 0.11-0.30 (fair); power 0.002-0.06;
  A/sigma + B (fair) 0.003-0.06. Conclusion "power beats log" survives; quoted margins (log rms 0.56-0.91)
  are inflated ~2-4x. scrambled1 (free p = 1.23-1.32 per seed) is fit as well by A/sigma + B (rms 0.03-0.06)
  as by the free power law: the p = 1.27 +- 0.02 outlier is an offset effect, not a contradiction of k = 3,
  but then the free-p mean "1.01 +- 0.14" is not the right summary statistic either.
- third singular value on the same sibling pair ((3,3),(4,3)): quadtree seeds 0/2/3/4 p = 0.33 / 1.11 /
  0.56 / -0.34; seeds 0 and 3 prefer the log law (rms 0.007, 0.037). The table labels these "k = 3,
  predicted p = 1" from a naive edge count; not discussed in the notes.
- MPS seed 4 edge 4 (k = 2): power (p = 0.27) rms 0.004 vs log 0.005: indistinguishable over 0.05-0.2.

## Hessians (hessians_raw.json)
- binary L = 0.27 plateau: per-seed exponents 0.871 / 0.888 / 0.873 (seeds 0, 2, 3; 0.05-0.3, <1 decade),
  seed 4 (3 pts) 1.149; with seed 4: 0.945 +- 0.068. lambda/lambda_max = -0.0010 ... -0.0075.
  "exactly one eigenvalue below -1e-3 lambda_max at sigma <= 0.14" holds in 11/15 runs (seed 2 s0.05 has 0).
- MPS L = 0.230 plateau: per-seed exponents 0.045, 0.044, -0.009, 0.078, 0.078 (mean 0.047 +- 0.016);
  2-3 eigenvalues below threshold at small sigma (several active routes), state is AFTER edge 5 rose.

## Route amplitudes (routes_raw.json, 124 states)
- 11 states have rank1_rel_err > 0.1 (up to 0.995): not rank-1, formula inapplicable; quadtree seed 3
  sigma 0.3/0.5 (err 0.81/0.94) counted as hits although the plateau midpoint is after the route rose.
- Same-seed runs across sigma sit at the same critical point (identical lambda pattern), so 35 / 89 runs
  are ~3 / ~9 distinct critical points; per (seed, critical point) the hit rate is ~80%, chance 25%.
- Misses at genuine rank-1 states with large ratios: quadtree seed 4 s0.14 (best/second 19.4),
  scrambled2 seed 2 s0.1 (6.3).
- k <= 5 share is not the speed predictor: scrambled0 seeds 1, 2 have 81% of sum lambda^2 on k <= 5 routes
  (quadtree-like) but the slowest first route (gap 3440 at sigma 0.05 vs ~300 quadtree). The first-route
  gap tracks 1/(best k = 3 amplitude): gap x lambda3 = 24-74 across 11 (order, seed) pairs.
- Using one run per (order, seed) at sigma = 0.05: k <= 5 share quadtree 0.89, scrambled 0.17 (per order
  0.34 / 0.11 / 0.055); the "88% vs 21%" holds descriptively but its SEs are over pseudo-replicates.
