# data-teacher: do input covariance and teacher structure change the saddle-to-saddle picture? (Q6, Q7)

Status: working - Branch: agent/data-teacher - Updated: 2026-10-09 18:20 UTC

## Summary
In progress. Plan: starting from the whitened / odeco baseline, add one complication at a time
(aligned anisotropic covariance, misaligned covariance, cross-site correlations, non-orthogonal CP
teachers, random TTN teachers, teacher on another tree) and compare with the baseline theory.

## Results
(none yet)

## What was done
(none yet)

## Decisions
(none yet)

## Files left on the machine
(none yet)

## Caveats and next steps
(none yet)

## Log
- 18:20 Read brief, project map, model, diagnostics, plan, library code. Machine: CPU quota 30.7
  vCPUs (cgroup cpu.max 3071999/100000), 503 GB RAM. Merged origin/main (report skeleton).
- 18:22 Theory for aligned covariance written (notes/data_teacher.md s.1): exact per-component
  clock Lambda_r t for aligned init; first escape = whitened problem with weights Lambda s; swap
  (piggyback) criterion = Lambda s^2 (function-space norm). binary-4 = depth-3 DLN (Fact A).
- 18:25 Pilot (8 seeds, binary-4 + MPS-4) confirms: first escape matches (I, Lambda s) paired runs;
  binary-4 shows swaps exactly when Lambda s and Lambda s^2 orderings disagree; MPS-4 never swaps.
- 18:27 E1.1 exact aligned-init check: TTN GF matches scalar ODE to 1e-9 in half-times; s -> Lambda s
  substitution gets half-times to 1-2% but widths wrong by up to 45%.
- 18:30 Launched e1_order sweep (2 trees x 3 sigma x 2 teachers x 4 variants x 40 seeds).
- 18:44 Lead messages: literature v1 merged; added variant "white_sqrtLs_mapped" (absorbed white
  problem from the mapped init: same landscape and start, different GF metric) to separate landscape
  from flow metric. Library fix 57d5262 merged: my teachers (standard basis, deterministic) and
  spec.init are unaffected (checked: saved init identical to spec.init after merge).
- 18:50 Theory drafts merged (theory_teachers.md covers step 4 predictions P1-P11); step 2 and 4a
  configs written, rank-1 prediction tool (constrained.py) tested.
