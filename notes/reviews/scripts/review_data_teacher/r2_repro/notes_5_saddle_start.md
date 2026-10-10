# part 5: saddle-start reruns (R2.1), agent's r2_saddle_runs.py on reduced configs (r2_repro/configs/repro_*.yaml)

Exact-saddle Hessian rows (rerun) = committed r2_saddle_hessian.csv: l_saddle identical, lambda_min equal to 1e-15
(nonorth, misaligned) or 3e-10 (randbin_302: L-BFGS tolerance); binary/HT rows: no negative eigenvalue (raw ~ -1e-9).
Library MPS student caps the outer bonds at d = 2 (bond dims 2,4,2), so 5 negative eigenvalues (all = lambda_min)
on mps4; my own chi = 4 everywhere student has 9 (same value). Not a bug.

Per-run t_esc (f = 0.05 onset rule, recomputed by the agent's analyze_r2_saddle.py on my raw files) vs committed
r2_saddle_runs_all.csv, 54 runs: nonorth_c0.45 (mps 14, binary 6, ht 2), nonorth_c0.2 mps 8, misaligned_20 mps 6,
randbin_302 binary 4: max relative difference 2e-3 (median <= 1.4e-5; grid quantization of t[h[0]]).
DISCREPANCY randbin_302 mps4 (12 runs) and ht4 (2): per-run t_esc differ by a constant per-seed offset on mps4
(seed 0: +2.4..+3.7, seed 1: -29.3, seed 2: -16 time units) and by factors 1.14 / 2.52 on ht4; binary4 agrees to 1e-5.
Cause (verified): constrained_minima() sorts the 12 L-BFGS results by loss and keeps the first of each distinct
tensor; gauge-equivalent copies (same W, different core sign patterns) have losses equal to 1 ulp
(0.048540957081886119 vs ...126), so the representative, hence how a given noise seed projects on the unstable
directions, is decided by roundoff (my best start: 7, loss ...119; committed l_saddle ...126). Only intercepts
change: slopes agree (rerun A = 13.061 +- 0.005 over 2 seeds, eps 1e-8..5e-2; committed same subset 13.10 +- 0.09;
committed all 13.04 +- 0.11; pred 13.094).

Law fits from the reruns (agent's lawfit per seed): nonorth_c0.45 mps4 A = 6.636 +- 0.005 (2 seeds, 6 eps 1e-8..5e-2;
committed 6.648 +- 0.010, pred 6.627); log wins 2/2 for both teachers.
Own implementation (no agent code): A = 6.627 (c = 0.45) and 35.14 (c = 0.2) = 1/|lambda_min| to 1e-4.
