# R2 audit note 6: the "degenerate" misaligned8_20 teacher is an event-definition failure

Rerun with the agent's own run() (r2_saddle_runs.py), mps8, eps = 1e-2, seed 0, T_CAP 2e6:
t_esc (absolute 0.05 rule) = 2.07e5, matching the agent's runs (f-rule mean 2.0e5 at eps = 1e-2).
Log: misaligned8_20_eps0.01.log. Teacher function-space s_2 per edge: 0.061-0.068, f = 0.05 threshold
0.0031-0.0034 (all 7 edges kept).

Trajectory: max_e s_2 grows from ~2e-5 (t ~ 100) to 3.7e-4 on EVERY edge by t ~ 1.1e4 (growth over
t = 342 -> 2496 at ~1e-3 per unit time on edge 3 = 2|lambda_min| = 1.01e-3), then sits at s_2 = 3.72e-4..3.85e-4
(an intermediate critical point, rank 2 on every bond, 0.55% of the teacher's s_2) with the loss unchanged
(0.0023259) until t ~ 1e5; only then does s_2 jump to the teacher values and L -> 6e-8 (t ~ 2e5).
So the strict escape from the rank-1 saddle happens at the predicted rate and ends at an intermediate saddle
whose s_2 lies 9x below the f = 0.05 onset threshold; the f-rule (f = 0.03-0.1) times the NEXT, much slower
escape. "Degenerate (no escape law)" is a misdiagnosis; the 24x-slow "log law" counted in 65/65 is the second
stage. Fix: time the first escape with a threshold relative to the intermediate level (or loss/speed events),
report the intermediate saddle, and drop this teacher from the 1/|lambda_min| tally or re-time it.

Second rerun at eps = 1e-3 (T_CAP 4e4; log misaligned8_20_eps0.001.log): same intermediate level 3.7e-4 reached by
t ~ 1.3e4-1.9e4. Time for max_e s_2 to reach 5e-5 / 1e-4 / 1.5e-4 / 2e-4: eps = 1e-2: 716 / 1386 / 1932 / 2443;
eps = 1e-3: 5147 / 5998 / 6598 / 7135 -> increment per decade 4431-4692 vs ln(10)/|lambda_min| = 4547
(ratio 0.97-1.03; one seed, two eps). The first escape obeys the strict-saddle law; the agent's t_esc does not
time it. scripts/misaligned8_20_stage1.py.
