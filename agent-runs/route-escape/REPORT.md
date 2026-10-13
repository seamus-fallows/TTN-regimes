# route-escape: leaving a known multi-pattern plateau (route theory, brief section 4.2)

Status: working - Branch: agent/route-escape - Updated: 2026-10-13 19:32 UTC

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
- 19:20 Read BRIEF.md, notes/briefs/route_theory.md (1-3, 4.2), the paper (all sections), notes/saddle_dyn.md, sdyn.py, saddle_order.py. Branch check OK: the four route files exist, `uv run pytest tests/test_routes.py` passes (13 tests). Machine: 128 cores, cgroup quota 122.9 CPUs.
- 19:25 Chose 11 dynamics cells and 4 static-only Theorem 9.2 instances with ttn.routes (all star).
- 19:33 Wrote common.py (cancellation-free model), driver.py (step-driven LSODA after sdyn.py), escape.py, static.py. Pilot: full net = reduced flow to 2e-9 (S3, A); 10 pilot runs 8-73 s, floor index <= 1.5e-7.
- 19:36 Pre-registration notes/route_escape.md committed and pushed (d487b50) before any sweep.
- 19:37 Sweep started (118 workers): 30 static tasks + 8085 escape runs, PID 2396, log /root/outputs/route-escape/logs/sweep1.log.
- 19:49 Static orders (27/30): inside M* the order = c*_J to 3 digits in every cell; full space = c*_J except PS2, T92b, T92d (order 4.000 < c*_J = 5, the Thm 9.2 shortcut). First results table pushed (08d22ad).
- 19:50 Escape sweep at 537/8085 runs (deep cells first, ~80 s per run under full load). First failure seen: U5B8 kind C seed 9 escapes with exponent 4 (prediction 3); its route entry at leaf node 0 decays to 0 on the eps^-2 time scale. U5B8 C runs at eps <= 1e-4 partly floor-flagged (Phi up to 35).
- 20:12 Static orders complete (30/30): inside M* = c*_J everywhere; full space = c*_J except PS2, T92b, T92d (order 4, the shortcut the Thm 9.2 formula predicts); the formula holds on all 5 family instances. Pushed (de9cc03).
- 20:13 Solver stall found (3 wall-capped runs: restarted LSODA stuck in nonstiff mode); fix: restarted solvers are BDF (deviation 1).
- 20:19 Added kind Cp (C direction projected onto M*) as a causal test of the shift in U5B8 C seed 9 (deviation 2); 4 spare workers run it.
- 20:40 Main sweep stopped at 3990/8085 runs (13 more stalls, mostly S4 A/B) and restarted with the fixed driver (114 workers), wall-capped runs rerun.

