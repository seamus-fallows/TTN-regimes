
### features = raw

| tree | sigma | runs | S2S (rho > 3) | rho > 2 | rho > 5 | drops (mean) | min rho (median) |
|---|---|---|---|---|---|---|---|
| MPS snake | 0.1 | 2 | 1.00 | 1.00 | 0.50 | 4.5 | 6.5 |
| MPS snake | 0.3 | 2 | 1.00 | 1.00 | 0.00 | 4.0 | 3.5 |
| binary quadtree | 0.1 | 2 | 1.00 | 1.00 | 1.00 | 3.5 | 8 |
| binary quadtree | 0.3 | 2 | 1.00 | 1.00 | 0.00 | 4.0 | 3.9 |
| binary scrambled | 0.1 | 2 | 0.50 | 0.50 | 0.50 | 4.0 | 6 |
| binary scrambled | 0.3 | 2 | 1.00 | 1.00 | 1.00 | 3.0 | 12 |

| tree | runs | first rise = sibling bottom pair | first two rising edges = siblings | = single edge | first rise at tau = 0 (unresolved) | single edge, resolved runs | Kendall tau(level, time) (MPS: level = position along the chain) |
|---|---|---|---|---|---|---|---|
| MPS snake | 4 | 0.00 | nan | 1.00 | 0/4 | 4/4 | 0.122 ± 0.026 |
| binary quadtree | 4 | 1.00 | 1.00 | 0.00 | 0/4 | 0/4 | 0.0713 ± 0.051 |
| binary scrambled | 4 | 0.75 | 1.00 | 0.25 | 0/4 | 1/4 | 0.379 ± 0.056 |

| tree | sigma | tau(L<=0.25) | tau(L<=0.2) | tau(L<=0.15) | tau(L<=0.12) | train L(tau_end) | test L(tau_end) | test acc | tau_end: median (runs stopped early) |
|---|---|---|---|---|---|---|---|---|---|
| MPS snake | 0.1 | 327 (2/2) | 495 (2/2) | 2.07e+03 (2/2) | - (0/2) | 0.12 ± 5.6e-05 | 0.124 ± 0.00015 | 0.95 ± 0.00094 | 3000 (0) |
| MPS snake | 0.3 | 207 (2/2) | 306 (2/2) | 683 (2/2) | 1.6e+03 (2/2) | 0.091 ± 0.012 | 0.1 ± 0.014 | 0.971 ± 0.0087 | 3000 (0) |
| binary quadtree | 0.1 | 1.96e+03 (2/2) | 1.29e+03 (1/2) | - (0/2) | - (0/2) | 0.214 ± 0.036 | 0.219 ± 0.029 | 0.873 ± 0.03 | 3000 (0) |
| binary quadtree | 0.3 | 662 (2/2) | 826 (1/2) | - (0/2) | - (0/2) | 0.193 ± 0.029 | 0.2 ± 0.024 | 0.897 ± 0.012 | 3000 (0) |
| binary scrambled | 0.1 | - (0/2) | - (0/2) | - (0/2) | - (0/2) | 0.273 ± 0.00025 | 0.275 ± 0.00034 | 0.828 ± 0.00031 | 3000 (0) |
| binary scrambled | 0.3 | - (0/2) | - (0/2) | - (0/2) | - (0/2) | 0.252 ± 0.0018 | 0.258 ± 0.001 | 0.848 ± 0.0084 | 3000 (0) |
