# route-nonstar: training on pattern targets that break the star condition

Status: working - Branch: agent/route-nonstar - Updated: 2026-10-13 19:30 UTC

## Summary
(in progress)

## Results
(in progress)

## What was done
(in progress)

## Decisions
(in progress)

## Files left on the machine
(in progress)

## Caveats and next steps
(in progress)

## Log
- 19:30 Started. Brief read; files present (notes/briefs/route_theory.md, ttn/routes.py, tests/test_routes.py, paper PDF); `uv run pytest tests/test_routes.py`: 13 passed. Machine: 128 cores, cgroup quota 122.9 CPUs, 251 GB RAM.
- 19:25 Read brief sections 1-3, 4.1 (runs/record), 4.3; paper abstract, Secs 1-3, 6.3, 7, 8, 10, App. B, C; notes/locality.md 5, 7.4-7.5, 8-11; experiments/locality/README.md; notes/saddle_dyn.md 8, 9.6.
- 19:35 Driver (experiments/route_nonstar/nsdyn.py: dense contraction, step-driven LSODA as in sdyn.py) and per-run analysis (run.py). Pilots: 2-25 s per run; non-star targets learn several patterns in one drop and plateau at rank-constrained optima carrying recombinant strings.
- 19:43 Pre-registration notes/route_nonstar.md Part I committed and pushed (504dd37). Sweep started: 9,520 runs, 120 workers (PID 4211, log /root/outputs/route-nonstar/logs/sweep.log).
- 19:55 Deviation 1 (transient segments merged into escapes), pushed.
- 20:10 Deviations 2 (secondary fits at the float64 floor) and 3 (existence check for "no plateau" predictions), pushed.
- 20:51 Sweep finished: 9,520 runs in 68 min; 9,230 converged, 211 trapped (dead routes on mps4 chains: R4, R4d, B4_mps4), 79 wall-capped (CTL1_mps6, TD_binary8 at the float64 floor); no errors.
- 20:58 First results tables pushed (7e1fb6b).

