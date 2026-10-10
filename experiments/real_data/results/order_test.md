# Pre-registered order test (notes/real_data.md section 9): results

| order | group | S | best rank-1 loss | O1 = mean L(1e4) | O2 = mean log10 tau(L<=0.25) | censored | per-seed L(1e4) |
|---|---|---|---|---|---|---|---|
| scrambled2 | old_random | 0.008 | 0.2471 | 0.2296 | 2.442 | 0/3 | 0.2292, 0.2292, 0.2303 |
| scrambled118 | new_random | 0.014 | 0.2477 | 0.2280 | 2.192 | 0/3 | 0.2370, 0.2235, 0.2235 |
| scrambled116 | new_random | 0.015 | 0.2912 | 0.2376 | 2.449 | 0/3 | 0.2287, 0.2424, 0.2418 |
| scrambled105 | new_random | 0.017 | 0.2579 | 0.1559 | 2.511 | 0/3 | 0.1183, 0.2312, 0.1183 |
| scrambled109 | new_random | 0.023 | 0.3035 | 0.1254 | 2.454 | 0/3 | 0.1243, 0.1271, 0.1246 |
| hilbert | structured | 0.033 | 0.3122 | 0.1579 | 2.156 | 0/3 | 0.1433, 0.1651, 0.1653 |
| scrambled102 | new_random | 0.035 | 0.2961 | 0.2007 | 3.575 | 1/3 | 0.2861, 0.1600, 0.1559 |
| scrambled108 | new_random | 0.041 | 0.2767 | 0.1220 | 3.368 | 0/3 | 0.1232, 0.1248, 0.1181 |
| scrambled119 | new_random | 0.085 | 0.2395 | 0.2152 | 2.297 | 0/3 | 0.2011, 0.2436, 0.2011 |
| scrambled104 | new_random | 0.102 | 0.3085 | 0.2439 | 3.958 | 1/3 | 0.2575, 0.2372, 0.2369 |
| scrambled115 | new_random | 0.158 | 0.2411 | 0.1747 | 1.792 | 0/3 | 0.2049, 0.1657, 0.1535 |
| scrambled100 | new_random | 0.167 | 0.3002 | 0.2400 | 2.950 | 0/3 | 0.2401, 0.2398, 0.2399 |
| scrambled107 | new_random | 0.196 | 0.2235 | 0.1814 | 1.044 | 0/3 | 0.1814, 0.1813, 0.1815 |
| scrambled106 | new_random | 0.252 | 0.2886 | 0.1769 | 2.845 | 0/3 | 0.1981, 0.1660, 0.1665 |
| scrambled1 | old_random | 0.254 | 0.2924 | 0.1731 | 3.847 | 0/3 | 0.1744, 0.1720, 0.1729 |
| colmajor | structured | 0.338 | 0.2423 | 0.2041 | 2.458 | 0/3 | 0.2281, 0.2422, 0.1421 |
| quadtreeT | structured | 0.338 | 0.2423 | 0.1891 | 2.082 | 0/3 | 0.2257, 0.1942, 0.1473 |
| scrambled113 | new_random | 0.369 | 0.2970 | 0.1346 | 3.207 | 0/3 | 0.1344, 0.1339, 0.1355 |
| scrambled103 | new_random | 0.380 | 0.2871 | 0.1940 | 1.730 | 0/3 | 0.1943, 0.1937, 0.1940 |
| mortonrev | structured | 0.475 | 0.2879 | 0.2193 | 3.734 | 1/3 | 0.1910, 0.2764, 0.1904 |
| scrambled117 | new_random | 0.672 | 0.2894 | 0.1563 | 1.586 | 0/3 | 0.1682, 0.1682, 0.1325 |
| scrambled0 | old_random | 0.757 | 0.2957 | 0.1835 | 3.531 | 0/3 | 0.2025, 0.1721, 0.1760 |
| scrambled101 | new_random | 0.888 | 0.2979 | 0.1170 | 3.015 | 0/3 | 0.1186, 0.1165, 0.1161 |
| scrambled111 | new_random | 0.943 | 0.2993 | 0.1735 | 3.566 | 1/3 | 0.1350, 0.2532, 0.1322 |
| scrambled114 | new_random | 0.951 | 0.2167 | 0.1237 | 1.434 | 0/3 | 0.1225, 0.1239, 0.1246 |
| quadtree | structured | 0.989 | 0.3100 | 0.0970 | 2.247 | 0/3 | 0.0957, 0.0985, 0.0968 |
| rowmajor | structured | 0.989 | 0.3100 | 0.1119 | 1.883 | 0/3 | 0.1355, 0.0991, 0.1011 |
| snake | structured | 0.989 | 0.3100 | 0.1001 | 1.627 | 0/3 | 0.1028, 0.0990, 0.0984 |
| scrambled112 | new_random | 0.990 | 0.2840 | 0.1190 | 2.384 | 0/3 | 0.1189, 0.1190, 0.1191 |
| scrambled110 | new_random | 0.994 | 0.2862 | 0.1018 | 1.516 | 0/3 | 0.1033, 0.1028, 0.0992 |

## Primary analysis (Spearman over orders, 95% bootstrap CI, permutation p)

- all orders, S vs O1: rho = -0.60 [-0.82, -0.22], p_perm = 0.0013, n = 27 -> supported
- all orders, S vs O2: rho = -0.30 [-0.59, +0.06], p_perm = 0.122, n = 27 -> not supported
- random orders, S vs O1: rho = -0.55 [-0.85, -0.09], p_perm = 0.0147, n = 20 -> supported
- random orders, S vs O2: rho = -0.26 [-0.63, +0.23], p_perm = 0.256, n = 20 -> not supported

## Secondary analyses (registered)

- (a) without the orders seen before registration, S vs O1: rho = -0.51 [-0.80, -0.08], p_perm = 0.0103, n = 24 -> supported
- (a) without the orders seen before registration, S vs O2: rho = -0.22 [-0.57, +0.22], p_perm = 0.294, n = 24 -> not supported
- (b) with scrambled0-2 added, all orders, S vs O1: rho = -0.60 [-0.82, -0.24], p_perm = 0.0009, n = 30 -> supported
- (b) with scrambled0-2 added, all orders, S vs O2: rho = -0.24 [-0.53, +0.13], p_perm = 0.207, n = 30 -> not supported
- (b) with scrambled0-2 added, random orders, S vs O1: rho = -0.57 [-0.84, -0.17], p_perm = 0.0055, n = 23 -> supported
- (b) with scrambled0-2 added, random orders, S vs O2: rho = -0.13 [-0.51, +0.32], p_perm = 0.54, n = 23 -> not supported

(c) predictor validity: S and best rank-1 loss vs the plateau state reached by the runs (sigma = 0.1):

- quadtree: S = 0.989 at best rank-1 loss 0.3100; plateau states: seed 0: L 0.270 share 0.92, seed 1: L 0.348 share 0.78, seed 2: L 0.269 share 0.91, seed 3: L 0.271 share 0.91, seed 4: L 0.271 share 0.91
- scrambled0: S = 0.757 at best rank-1 loss 0.2957; plateau states: seed 0: L 0.322 share 0.03, seed 1: L 0.301 share 0.07, seed 2: L 0.297 share 0.81, seed 3: L 0.322 share 0.02, seed 4: L 0.322 share 0.03
- scrambled1: S = 0.254 at best rank-1 loss 0.2924; plateau states: seed 0: L 0.294 share 0.12, seed 1: L 0.294 share 0.12, seed 2: L 0.294 share 0.13, seed 3: L 0.294 share 0.12, seed 4: L 0.294 share 0.13
- scrambled2: S = 0.008 at best rank-1 loss 0.2471; plateau states: seed 0: L 0.249 share 0.01, seed 1: L 0.249 share 0.01, seed 2: L 0.351 share 0.04, seed 3: L 0.341 share 0.24, seed 4: L 0.248 share 0.01

## Exploratory (not registered): plateau-state share of the test runs, averaged over seeds

- all orders, S_plateau vs O1: rho = -0.83 [-0.91, -0.66], p_perm = 0.0001, n = 27 -> supported
- all orders, S_plateau vs O2: rho = -0.15 [-0.51, +0.24], p_perm = 0.441, n = 27 -> not supported
- random orders, S_plateau vs O1: rho = -0.85 [-0.95, -0.60], p_perm = 0.0002, n = 20 -> supported
- random orders, S_plateau vs O2: rho = -0.08 [-0.54, +0.42], p_perm = 0.745, n = 20 -> not supported
