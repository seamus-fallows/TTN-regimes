# real-data claims audit: working notes, part 4 (new runs, data/review_real_data/audit/runs)

Setting: MNIST 3v5 4x4, raw, chi = 4, sigma = 0.1, init seed 0, tau = 1e4, repo code (copy with data dir
patched; ordering quadtreeT = Morton order of the transposed image: vertical pairs, same 2x2 blocks,
left/right halves). Wall 450-720 s per run here.

Reproduction: quadtree seed 0 L(1e4) = 0.0957, t_a = 1.100e15, drops at tau 15.5 / 158.6 / 339.6 / 2157.8 /
4743.8, first route edges [3, 4] at tau 150.1 -- matches committed values (regime_analysis 0.095720,
notes 7: 2157.79 / 4743.3).

| binary order | t_a | first route (edges, tau) | tau L<=0.25 | tau L<=0.2 | L(1e4) | test acc |
|---|---|---|---|---|---|---|
| quadtree (Morton) | 1.10e15 | [3,4] 150 | 156 | 333 | 0.0957 | 0.950 |
| rowmajor (1D local) | 2.74e15 | [3,4] ~50 | 48 | 54 | 0.1355 | 0.948 |
| scrambled4 | 6.31e15 | [7,8] 95 | 109 | 4935 | 0.1429 | 0.918 |
| scrambled6 | 6.86e15 | [3,4] ~570 | 606 | 8443 | 0.1699 | 0.909 |
| scrambled3 | 1.37e15 | [10,11] 65 | 64 | 100 | 0.1866 | 0.896 |
| scrambled5 | 5.27e14 | [10,11] ~290 | 111 | - | 0.2243 | 0.869 |
| quadtreeT (2D local) | 2.30e15 | [0,1] ~750 | 139 | - | 0.2257 | 0.861 |
| MPS snake | 8.36e15 | [5] 9.8 (before drop 0 at 11.3) | 12 | 52 | 0.0891 | 0.957 |

- Quadtree beats all new random orders on L(1e4) (with the 3 committed orders: 9/9), but the equally local
  transposed quadtree is as bad as the worst random order, and several random orders and row-major reach
  L <= 0.25 / 0.2 earlier than the quadtree. The advantage is specific to the Morton order's horizontal
  pairing / top-bottom split for this task, not a generic locality effect.
- MPS t_a / quadtree t_a = 7.6 at sigma 0.1 (prediction from sigma 0.5 prefactors: 0.76/sigma = 7.6):
  on the absolute clock the quadtree reaches every loss level ~7.6x earlier.
- MPS: edge 5 crosses s2/s1 = 0.05 at tau 9.8, before the first drop's peak (11.3); drops have rho
  3.0, 1.7, 3.8, 2.5 (not S2S).
