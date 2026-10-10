# Verification of the locality fix round (2026-10-10)

Independent check that the fix round addressed `notes/reviews/locality_round1.md` and the lead's items 1-7.
Method: recomputed claims from the committed CSVs; reran from scratch all 256 k >= 6 extension runs, all 168
lambda2(K) runs and 45/48 mps8 {1,5} k = 5 extension runs (eager mode, Windows). Scripts/outputs in the lead's
scratchpad (`verify_locality_fix/`).

**Verdict: ready for the report with caveats.** Solid: two-component H2 for k = 1-4 incl. the k = 2 log law with
head start and lambda2's K-dependence; k = 5 on binary8/ht4; k = 6-7 with balanced init (reproduced exactly:
4.0067 +- 0.0211, 4.0606 +- 0.0483, 4.9572 +- 0.0609); H3 (k = 2 strict with lambda_min = -lambda2; lambda_min -> 0
for k >= 3); route rule for TC/TD and rank profiles at multi-component saddles; dead routes under fan-in for k >= 4.

Remaining fixes requested from the agent (final round):
(a) MPS k = 5 exception: restate from the committed numbers (3.67 +- 0.20 saddle start; 3.13 +- 0.25 balanced,
    median 2.95; mps6 {1,3,5} 3.05/3.15 at sigma <= 0.03; mps8 {1,5} small init 2.97 +- 0.05 on never-stalling
    seeds): only mps8 {1,5} from the saddle start stays steep (4 seeds, 0.4 decades, ~40% stalled).
(b) Drop "spare bond channels remove most dead routes": chi = 4 gives 106/1005 vs 169/1008 at chi = 2 on the same
    24 cases, and stalls RISE on routes with both ends capped.
(c) Present the bond-spectra result as a leakage bound (non-Steiner edges <= 4.3 sigma^2 s2, slope ~2) plus
    single-route consistency, not as "99.75% prediction success" (the rank test cannot register non-Steiner
    rises; the spread tests "one route", not timing; spread grows with sigma).
(d) Refresh stall counts (fan-in 398/5682, chi = 4 106/1005) and the dropout list (9 rows, up to 1.38); fix the
    pinned-run amplitude; flag the 7 dead runs with a live route end (theory P14(v)).
(e) Label k = 6-7 as balanced-init only (fan-in k = 6 rows give 3.68 / 4.23); give full Hessian exponent ranges
    (k = 3 0.81-1.06, k = 4 1.64-2.16, k = 5 2.82-3.76, mps8 {1,6} 4.27, k = 7 up to 6.5); "raw and projected agree
    to 0.01" is false (up to 0.11; projection still biased at sigma >= 1e-3 - quote raw).
(f) TA on mps4 first level is power by the fair comparison (0.22 vs 0.53); "every k* >= 3 level within 0.2" omits
    B4 ht4 fan-in (4.73 +- 0.90); ht4 {0,1,2} has 7 fitted seeds, not 8.
(g) Provenance: finalize.sh must generate run_classes_by_sweep.csv, stall_dropout_*.csv, r12_partial.csv and the
    multi-component events CSV; remove the empty extensions-table row.
