# real-data claims audit: working notes, part 7 (batch 2b: audit/runs3, audit/b2/*.json)

sigma = 0.1, raw, chi = 4, tau = 1e4.
| run | first route (tau after drop 0) | tau L<=0.25 / 0.2 / 0.15 | L(1e4) | acc | share k<=5 | gamma |
|---|---|---|---|---|---|---|
| binary quadtreeT seed 1 | [0,1] 1378 (= best k=3 pair) | 89 / 7126 / - | 0.1942 | 0.878 | 0.13 | 1.33 / 1.33 |
| binary rowmajor seed 1 | [3,4] 69 (= best pair) | 87 / 92 / 4752 | 0.0991 | 0.953 | 0.92 | 1.22 / 1.24 |
| (binary quadtree seed 1, committed) | | | 0.0985 | 0.956 | | |
| MPS rowmajor seed 0 | [5] 1.7 | 16 / 41 / 481 | 0.0830 | 0.956 | | |
| MPS scrambled0 seed 0 | [3,4] 213 (3-core route) | 299 / 575 / 1560 | 0.0840 | 0.962 | | |
| MPS scrambled3 seed 0 | [13] -1.4 | 20 / 50 / 377 | 0.1283 | 0.929 | | |

- Transposed quadtree is bad at both seeds (0.226, 0.194 vs quadtree 0.096, 0.099): an equally local
  order loses most of the advantage. Row-major (horizontal pairs) is close to the quadtree at seed 1.
- Route-amplitude rule out of sample: 11/11 new binary runs (largest sibling amplitude = first route).
- MPS results depend on the chain order (0.083-0.128 final; 16-299 to reach L <= 0.25); a scrambled chain
  can start with a 3-core route and show S2S-like plateaus (rho 8.5-30).
