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

