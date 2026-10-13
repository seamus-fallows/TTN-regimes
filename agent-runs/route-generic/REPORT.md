# route-generic: does "route depth = cores that change" hold outside pattern targets?

Status: working - Branch: agent/route-generic - Updated: 2026-10-13 19:31 UTC

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
- 19:31 Started. Checked files: notes/briefs/route_theory.md, ttn/routes.py, tests/test_routes.py and the paper PDF exist; `uv run pytest tests/test_routes.py` passes (13 tests). Machine: 128 logical CPUs, cgroup quota ~122.9 CPUs, 251 GB RAM.
- 19:34 Read brief (sections 1-3, 4.4), paper (abstract, 1, 2, 7, 8, 10, App. C), data_teacher / real_data notes and reviews, saddle_dyn section 8. MNIST downloaded to /root/outputs/route-generic/data (torchvision mirror, OK).
- 19:38 Pilots: 16 synthetic runs (8-150 s each, all converged), 4 MNIST runs (tau = 2000: 150-250 s).
- 19:40 Pre-registration committed (f31079c, notes/route_generic.md Part I). Sweeps launched: synthetic 2640 runs (64 workers), MNIST 180 runs (56 workers).
