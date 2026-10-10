# Part 2: analysis issues found by reading the code

1. bond_events.py (claim 4, "98.7% of 5132 events"): for a two-component run the comparison is
   rank at the plateau midpoint vs rank at the END of the run (mids = [t_mid] + [t[-1]]; the run ends when
   min a_r > 0.97). At the end W ~ W*, whose bond ranks are exactly 2 on the Steiner edges and 1 elsewhere
   (proven, notes/locality.md section 1). So the statistic mostly checks convergence + that the dormant s_2
   at the midpoint is below 0.1 x s_2*. It does not test that the Steiner edges rise *together at the event*
   or that other edges stay rank 1 *during* the escape. (Reviewer's own test: my_bonds.py.)
2. analysis.summarize pools runs by final-crossing order ("0>1"), so 'rotation' runs (a component learned,
   lost, relearned) are fitted together with greedy runs (analysis standard 3). Effect measured: small
   (binary4 S=12: 1.011 -> 1.008 without the rotation run).
3. Hessian gauge projection (ttn.diagnostics.project_out_gauge, rtol 1e-10) near the aligned saddle: gauge
   generators acting on the unused bond index give O(sigma) columns that pass the 1e-10 cut; normalized, they
   point along the route's new-index entries, so projecting them out removes part of the escape direction.
   Hypothesis for the saddle-start k = 2 lambda_min = -0.44..-0.48 (sigma-independent, seed-dependent) instead
   of -0.5 (committed table: mps4 S=12 -0.444, mps8 S=34 -0.460, mps4 S=01 -0.479, mps8 S=01 -0.484).
   Tested in indep.py (lam_min_cut3, lam_min_unprojected).
4. run.py: the post-processing calls fun() (speed at stored steps), which still checks the wall-clock limit:
   a run finishing close to wall_limit raises TimeoutError outside the try block -> recorded as 'error' and
   retried. Minor.
5. Stall detector threshold: |v|max < 1e-12 for 5000 consecutive evaluations after a component is learned.
   1e-12 is ~4 orders above gradient roundoff; a slow but live route with route-entry gradient < 1e-12 would be
   labelled 'stalled'. Tested by rerunning stalled runs with the detector off (run_nostall.py).
6. make_figures.main_rows keeps only (order 0>1, j = 0) plateau rows, so a case in which NO run escapes vanishes
   from the table, together with its stall count: midroot6 S=05 at chi = 4 (30 runs, all stalled per notes 7.10)
   is missing from results/twocomp_chi4_table.* (23 rows for 24 configured cases; 972 of 1008 runs). The text's
   per-case list sums to 6+12+18+10+15+30+11+1 = 103 stalls at chi = 4, but REPORT/notes say "75 of 1,002".
7. Provenance: rows ht4 S=0123 (small, saddle), ht8 S=0123 saddle, mps8 S=07 small contain runs on sigma grids /
   seeds that are not in configs/twocomp.yaml (table: 11 sigmas 0.08-0.6, 5-7 seeds; config: 6 sigmas 0.15-0.6,
   5 seeds; 51-58 runs vs 30). The k = 7, 8 numbers (e.g. ht8 S=0123 saddle 6.86 +- 0.64) cannot be regenerated
   from the committed configs.
8. (refines 3) theory_saddles.md 6.2 already says that at a plateau point the gauge contains the
   product-preserving rescaling of the route amplitudes, so the flat-metric projected k = 2 eigenvalue is
   -2 lambda a b / (a^2 + b^2) for route-end amplitudes a, b: = -lambda only for a balanced route. Small-init
   routes are balanced by their long growth (-0.499); saddle-start routes at the midpoint are not -> the
   projected lambda_min is gauge dependent. Test: indep.py lam_min_balanced (same state moved to the balanced
   representative of its gauge orbit with run.balance_gauge, W unchanged to 3e-14).
