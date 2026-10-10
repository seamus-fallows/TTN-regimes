# saddle-dyn: test the dynamical consequence of the saddle-order theorem

Status: working - Branch: agent/saddle-dyn - Updated: 2026-10-10 15:00 UTC

## Summary
Testing whether the escape time from a learned rank-one saddle scales as eps^-(order-2) (theorem's order)
instead of the route law eps^-(k*-2), from saddle starts (H-A) and from small init (H-B).
Pre-registration: notes/saddle_dyn.md (Part I). Results: pending.

## Results
Pending.

## What was done
1. Pre-registration notes/saddle_dyn.md (hypotheses, 11 H-A cells, 3 H-B cells, windows, decision rules).

## Decisions
- Added three cells (HT4L, HT4S, B8W05) from the lead's saddle_hopper message before committing the
  registration (no runs yet), labelled as such in the note.

## Files left on the machine
- /root/outputs/saddle_dyn/ (raw per-run JSON; pending)

## Caveats and next steps
Pending.

## Log
- 15:00 read brief, notes, saddle_order code; wrote pre-registration; merged origin/main (lead's hopper results)
- 15:22 code committed (sdyn/runs/sweep); validated: H-B seed 7 sigma=0.01 dt=211.88 (verifier 211.9); launched H-A (1836 runs, 125 procs) and H-B (842 runs, 60 procs); trap rule + event fix as deviations
- 15:38 first sweep stopped: scipy LSODA dense output unreliable (events, states); outputs set aside in /root/outputs/saddle_dyn_try1_badevents
- 15:46 new step-wise driver validated; H-B re-gauge after 50-unit relaxation; sweeps relaunched (H-A 125 procs, H-B 62 procs)
