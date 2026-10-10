# part 3: full-horizon, lazy, independent scrambled

## Full main.yaml settings, seed 0, sigma 0.1 (t2 = 1e4), agent code
quadtree: n_drops 5, rho [15.67, 19.83, 290.31, 12.2] (ref [15.7, 19.99, 290.55, 12.2]); drops at 15.5, 158.6, 339.6,
2157.8, 4743.8 (notes tol-check 2157.79, 4743.3); L(1e4) 0.09572 (ref 0.09572); test acc 0.9500 (ref 0.9500).
scrambled0: L(1e4) 0.2025, test acc 0.889 (ref mean over 15 runs/3 orders 0.207 +- 0.008, 0.872); tau(L<=0.25) 8965
vs quadtree 156; L<=0.2 never (quadtree 333). 683 s / 766 s wall.
## Lazy (centred, sigma 4, seed 0, regime.yaml settings), agent code + independent L_lin
L(T) 0.191935010 (ref 0.191935006); my L_lin(T) 0.19193499342 (ref 0.19193499342); rank J 304 of 528 (ref 304);
delta_K 3.008e-6 (ref 3.008e-6); floor 0.127614 (ref 0.127614). Note L_lin(T) is far above the floor: lazy
statement is at finite time (1e4 kernel units).
## Independent scrambled0 sigma 0.5 seed 0
t_a 1261428.28 vs 1261428.71 (3.4e-7); drops 10.18, 48.73 -> gap 38.55 (ref 38.59); first rank-2 edges [10, 11];
route share k<=5 0.107 (ref 0.098), best k3 pair (10,11); my rank-1 error 0.015 (agent's core-SVD version 0.092).
## Committed-config mismatch
main.yaml as committed lists scrambled1 only at sigma {0.05, 0.1, 0.2, 0.5} (220 runs total) but the analysed sweep has
234 runs: 14 scrambled1 runs at 0.07/0.14/0.3 finished before the grid was trimmed and were kept. The scrambled1 first-route
exponent 1.27 +- 0.02 (6-7 sigma per seed) uses them; re-running main.yaml does not regenerate them.
