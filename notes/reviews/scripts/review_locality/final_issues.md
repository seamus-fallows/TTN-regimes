# Discrepancies and bugs (final list)

Discrepancies (claim -> evidence -> likely cause)
1. (2) k = 2 saddle-start lambda_min: REPORT "-0.49..-0.50"; committed/rerun -0.444 +- 0.024 (mps4 S=12), -0.460 +- 0.013
   (mps8 S=34). Own Hessian with only well-conditioned gauge generators, or unprojected: -0.5000 (+-5e-5), 16/16 states.
   Cause: O(sigma) gauge generators on the unused bond index pass the 1e-10 rank cut and are projected out; normalized
   they overlap the escape direction (route rescaling). Physics claim right, reported number biased.
2. (7) "gauge-balanced init removes almost all (2/1008)": committed twocomp_balanced_table.csv has 16/1008 stalled
   (fan-in on the same cases: 169/1042) -> ~10x reduction; balanced still stalls on mps8 S=25 (k = 4, 2/48).
3. (7) chi = 4 "75 of 1,002": fully stalled midroot6 S=05 (30/30) is missing from twocomp_chi4_table (make_figures
   drops cases with no escaping run); the notes' own per-case list sums to 103.
4. (4) 98.7%: bond_events.py compares ranks at the plateau midpoint with ranks at the END of the run, where W ~ W*
   has rank 2 exactly on the Steiner edges by construction; the 65 mismatches are k = 2 small-init runs at sigma
   0.03-0.1 whose "midpoint" is already half-escaped. The stronger statement (only Steiner edges rise, together, at
   t2) holds in my own test: 131/131 N = 8 runs + 111/111 mps4 S=12 runs.
5. (5) single-run k: committed growth_summary gives k_est 2.36 / 2.89 / 3.33 for small-init k = 2 on mps6 S=23,
   mps8 S=01, S=34 -- fails there (only the saddle start recovers k = 2); 5.06-5.32 for k = 5, 7.66-7.89 for k = 8,
   7.7 for ht8 k = 7: biased high.
6. (1) "k - 2 to +-0.05 for k <= 4": 5/16 k = 4 rows outside (saddle start 1.49, 1.93, 2.39, 2.19; small mps6 S=14 2.06).
7. (3) ht8 S=34, k = 7: p = 5.7 +- 1.1 rests on sigma 0.15-0.6 where at sigma >= 0.45 the 'saddle start' is not near
   the saddle (learned overlap starts < 1/2: first event at t = 0.1-3.9); one pinned seed (seed 4, gaps 3.4e5 ->
   1e9, per-seed slope ~9) carries the mean; median 5.01. Consistent with 5, not a measurement of it.
8. (6) TD: level 2 exponent 1.012 +- 0.010 reproduced (k* = 3 confirmed vs 2 / 4); notes say "1.02 +- 0.01 (7)",
   CSV 1.012 (8). Level 1 labelled "log" but the committed fair comparison prefers a power law (RMS 0.389 log vs
   0.162 power, p = 0.65; 12/55 durations unresolved).
9. Provenance: ht4 S=0123, ht8 S=0123, mps8 S=07 rows include runs on sigma grids/seeds not in configs/twocomp.yaml.
