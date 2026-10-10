# part 6: round-1 corrections (4a, 4b) and random-init item-1 reruns

4a (run_dense.py, copied config e4a_rev = e4a_sigma with c = 0.45 only; seeds 2, 4, 9; sigma 3e-4, 1e-3, 3e-3, 0.03;
analyze_law_v2.py e4a_rev): t1 (first escape, up to 2.1e7) equal to <= 7e-6 relative; gap_0.05 within 0.75% (one dense-grid
step); plateau lambda_min equal to 1e-4 (-0.1509 at sigma 3e-4 = exact saddle value). Fit on rerun (3 seeds, 2 decades):
A = 5.722 +- 0.026 (f 0.05), 5.719 (0.03), 5.724 (0.1); pred 5.682; committed 5.764 +- 0.035 (9 seeds, sigma to 0.19).
c = 0.2 (36.5 +- 0.8) not rerun.
4b (e4b_rev = teachers 104, 102 at sigma 1e-3, seeds 0, 1; analyze_ttn.py): plateau times equal (1e-7); distance plateau ->
rank-1 critical point 0.00021-0.00024 (104), 0.00164 (102) [committed 0.00021-0.00023, 0.00166]; nearest SVD-type truncation
0.0015 (104), 0.0047-0.0060 (102) [same]; critical point -> truncation gap 0.001482 / 0.004955 [same]. Supports
"separated for 104, 102 (and 105)"; 101, 103 not resolved (committed CSV).
Random init (run_dense.py r2_random_rev = nonorth_c0.45, seed 2, mps sigma 3e-4..0.1186, binary 1e-4..0.0631):
gaps within 0.8%; plateau lambda_min mps4 -0.151 at every sigma (strict), binary4 -1.7e-4, -1.7e-3, -1.9e-2, -0.16 (prop. to
sigma: higher order); seed-2 A = 5.77 (pred 5.68), binary p = 1.033 (local p 1.001 between 1e-4 and 1e-3).

Spawn test (spawn_test.py): on Windows the Pool worker sees T_CAP = 1e7 although the parent set 1e11 -> confirms README issue 1.
