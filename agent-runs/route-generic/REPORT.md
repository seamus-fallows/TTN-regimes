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
- 19:44 Jacobian switched to the library's contraction-based exact Hessian (Problem.hessian, 0.12 s vs 2.5 s for the dense double-backward; identical to 1e-21); synthetic sweep restarted, the 30 finished runs kept (numerics-only, Part III of the note).
- 20:25 Interim table pushed (experiments/route_generic/results/partial/, 1680 synthetic runs). First reading: pattern control and random TTN on binary8 follow the picture; on generic targets the events often involve more cores than the growth-law depth and the duration exponent indicate. MNIST seeds 10-19 queued to start when the synthetic sweep ends.
