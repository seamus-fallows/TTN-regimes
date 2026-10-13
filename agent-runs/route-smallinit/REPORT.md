# route-smallinit: route theory vs gradient-flow training from small random init (star targets)

Status: working - Branch: agent/route-smallinit - Updated: 2026-10-13 19:37 UTC

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
- 19:37 Started. Checked: brief file, ttn/routes.py, tests/test_routes.py and the paper PDF exist; `uv run pytest tests/test_routes.py` passes (13 tests). Machine: 320 cores, cgroup quota 307 CPUs.
- 19:37 Read brief sections 1-3 and 4.1, the paper (abstract, Secs. 1-3, 6-8, 10, App. C), notes/saddle_dyn.md section 8, sdyn.py and locality/run.py.
- 19:40 Chose 10 star targets with ttn.routes (brief's ht4/binary8/mps5-decoupled examples replaced: their later route depths exceed 4); wrote model (dense residual), step-driven LSODA driver, run reduction.
- 19:45 Pilots (seed 0, sigma 1e-4, all targets): 5-25 s per run. Roundoff bound flags depth-4 plateaus but duplicate runs with different arithmetic agree to 1e-9..1e-6 -> empirical floor check adopted.
- 19:49 Pre-registration pushed (d44bc78). First sweep launch stopped after 1 min: trap rule fired at the origin for K=7 at sigma 1e-5 (all amplitudes ~1e-38); fixed (rule only after a pattern is learned), outputs set aside in /root/outputs/route-smallinit/runs_try1_trapbug/.
- 19:50 Sweep restarted (7350 runs, 300 processes).
