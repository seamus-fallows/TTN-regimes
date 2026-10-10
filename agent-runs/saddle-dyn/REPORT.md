# saddle-dyn: test the dynamical consequence of the saddle-order theorem

Status: working - Branch: agent/saddle-dyn - Updated: 2026-10-10 18:51 UTC

## Summary
The brief asked whether the escape time from a learned rank-one saddle scales as eps^-(p-2), with p the
saddle order of notes/theory_saddle_order.md, rather than as the route law eps^-(k*-2). I pre-registered
hypotheses, cells, windows and decision rules (notes/saddle_dyn.md Part I, commit 3377a9f) before any run,
then ran 1836 saddle-start runs (11 cells x 12 directions, eps = 1e-1 ... 1e-5), 842 small-init runs and
165 accuracy checks. **Where p < k* and the window lies below the crossover, the theorem's exponent holds to
three digits:** PS08 (2.001 ± 0.001), W6 (order 4 vs k* = 6: 2.001 ± 0.001), HT4W (3.996 ± 0.003), HT4L
(3.997 ± 0.003), binary8 rho = 0.5 (3.998 ± 0.007), and the pair-star from small init (2.002 ± 0.001, 16
visiting seeds). In all of these every seed is within 0.05 and the route law is worse by an RSS factor of 1e3-2e5.
Two test cells are not below the crossover for all seeds: binary8 at rho = 0.8 (INCONCLUSIVE; seeds at 4
or at the route law 5) and HT4S (REFUTED by the registered rule: 8 of 11 seeds at 5). Controls (p = k*) are
INCONCLUSIVE by the 10-seed rule because of dead routes. Their escaping seeds follow the route law exactly
or exactly one power higher, never lower.

## Results
Evidence: `experiments/saddle_dyn/results/*.csv` (from `analyze.py`), tables by `tables.py`, figures by
`figures.py`; raw JSON in /root/outputs/saddle_dyn/. Commands in experiments/saddle_dyn/README.md.

| cell | p (k*) | H-A gamma (seeds) | registered decision | H-B (small init) |
|---|---|---|---|---|
| PS08 MPS6 pair-star (0.8) | 4 (5) | 2.001 ± 0.001 (12) | SUPPORTED | 2.002 ± 0.001 (16 of 40 seeds visit): SUPPORTED |
| W6 MPS6 110001 (1.5) | 4 (6) | 2.001 ± 0.001 (12) | SUPPORTED | - |
| HT4W HT4 1111 (1.5) | 6 (7) | 3.996 ± 0.003 (12) | SUPPORTED | - |
| HT4L locality HT4 (s_q 0.5) | 6 (7) | 3.997 ± 0.003 (12) | SUPPORTED | - |
| B8W05 binary8 rho 0.5 | 6 (7) | 3.998 ± 0.007 (12) | SUPPORTED | - |
| B8W binary8 rho 0.8 | 6 (7) | 4.48 ± 0.15 (7 seeds at 4, 5 at 5) | INCONCLUSIVE; route law at sigma >= 0.1: SUPPORTED (5.10) | 4.31 ± 0.12 (13: 10 at 4.0-4.2, 3 at 5): INCONCLUSIVE; route law above 0.1: SUPPORTED (4.87) |
| HT4S s2s_core close teacher | 6 (7) | 4.72 ± 0.14 (3 at 4, 8 at 5) | REFUTED (window above crossover) | - |
| PS06 control (0.6) | 5 | 3.000 (4 complete seeds) | INCONCLUSIVE (traps) | 5 of 12 seeds visit; INCONCLUSIVE |
| C1 MPS6 110001 (0.7) | 6 | 3.97 ± 0.04 (3) | INCONCLUSIVE | - |
| C2 MPS5 10101 (1.5) | 5 | 3.40 ± 0.24 (5: three at 3, two at 4) | INCONCLUSIVE | - |
| C3 HT4 1111 (0.5) | 7 | 6.0 (1) | INCONCLUSIVE | - |

- Escape direction: every window escape of the supported cells is split-type (generalized-piggyback
  direction); afterwards q is forgotten in all PS08/W6 runs and 71-79 % of HT4W/HT4L runs.
- Controls: seeds follow the route law exactly, or k* - 1 (or k*), or are trapped; the shift belongs to the
  direction and the shifted seeds tend to become trapped at smaller eps. A diagnostic shows that the
  dormant state reached from small init is exactly theta_S + sigma xi', so the shift arises in the
  dynamics, not in the initial amplitudes (D7 in the note is a conjecture for the mechanism).
- The verifier's PS06 "12/12 q-first" and "exponent ~4 in half the seeds" partly come from misclassification:
  7 of 12 seeds learn the residual pair first, which its test cannot detect for the 0.6 teacher.
- Accuracy: tolerance reruns change t_esc by <= 1.6e-7 (77 pairs); plain-loss reruns by <= 8.3e-8 (48
  pairs) where both escape. In 4 cases the plain loss produces roundoff-seeded escapes from routes that the
  cancellation-free run finds dead (residual 1e-30). Conservation drift <= 2e-9 in all escaped test-cell runs.

Figures: experiments/saddle_dyn/results/ha_escape.png, ha_local.png, hb_dt.png, hb_local.png.

## What was done
1. Pre-registration notes/saddle_dyn.md Part I (3377a9f); three cells added from the lead's hopper message
   before the commit.
2. Cancellation-free gradient-flow code on the verifier's vttn contraction, H-A/H-B runs, sweeps (38debac).
3. First attempt discarded: scipy LSODA dense output unreliable; new step-wise driver with events refined
   by re-integration (ae9fee9, a0e1727).
4. Second attempt discarded: floor dominated by the theta_S reference; re-reference after relaxation,
   floor-index quadrature fixed (6a42da6).
5. Final sweeps 16:21; runs without outcome (overload + solver stalls) rerun with an in-solver reference
   switch and a tighter clock-restart rule (fed62f3, ae3012e).
6. Analysis, tables, figures, PS06 diagnostic (dab8a77 and the final commit); results in notes/saddle_dyn.md Part II.

## Decisions
- **Three extra cells (HT4L, HT4S, B8W05)** from the lead's message, added before the registration was
  committed (labelled in the note), because they are the order-6 saddles of past runs and cheap.
- **Per-entry Gaussian perturbations** for H-A (as the note's own check), but the order agent's fan-in
  perturbation for binary8 so that sigma and the crossover estimate refer to the same quantity.
- **Cancellation-free loss** (telescoping W - W_S), with the reference re-taken at the saddle point actually
  reached, instead of the library's integrate_gf: the plain loss reaches the float64 floor a factor eps
  sooner (borne out: 4 roundoff-seeded plain escapes).
- **Trap stop rule** (deviation 1): otherwise each dead-route run would burn the 1-hour cap.
- **Discarded and rerun** two whole attempts when numerical defects were found (dense output, reference
  choice), rather than patching their outputs; old outputs were moved aside, not deleted (reversible).
- **Reran only the runs without an outcome** after fixing solver stalls (deviation 7), keeping finished runs.
  A stall costs time, not accuracy, and this does not select on outcome.
- **Applied the registered verdicts as written** (HT4S "REFUTED", controls "INCONCLUSIVE"), then gave the
  per-seed reading separately.
- Did not edit CLAUDE.md or the theory notes (lead's files); suggested changes are below.

## Files left on the machine
- /root/outputs/saddle_dyn/ (HA, HB: one JSON per run, ~SIZE_OUT): needed to rerun analyze.py.
- /root/outputs/saddle_dyn_try1_badevents/, saddle_dyn_HA_try2_thetaref/, saddle_dyn_HB_try2_quadrature/,
  saddle_dyn_unfinished_try3/: discarded attempts and the unfinished runs that were rerun (not needed).

## Caveats and next steps
- The prediction is asymptotic: HT4S and binary8 (rho 0.8) need smaller eps (HT4S: fitted crossover ~1e-9,
  below the float64 floor; binary8 reachable down to sigma ~1e-4).
- The controls need more seeds or another perturbation family: only 3-4 of 12 keep their route alive at all eps.
- The integer shifts of the control exponents (and of 3/13 binary8 small-init seeds) are unexplained (conjecture D7).
- Suggested edits for the lead: theory_saddle_order.md section 6 / S9 and theory_saddles.md section 11 can cite
  D1-D4 for the order-4 and order-6 cells; the "exponent about 4 of half the control seeds" is a
  classification artefact plus the integer shift (D6, D8); the crossover estimate b should be called qualitative.
- Numerics for others: solve_ivp events with LSODA rely on an unreliable dense output; a fresh LSODA started on
  a plateau may never turn stiff.

## Log
- 15:00 read brief, notes, saddle_order code; wrote pre-registration; merged origin/main (lead's hopper results)
- 15:22 code committed (sdyn/runs/sweep); validated: H-B seed 7 sigma=0.01 dt=211.88 (verifier 211.9); launched H-A (1836 runs, 125 procs) and H-B (842 runs, 60 procs); trap rule + event fix as deviations
- 15:38 first sweep stopped: scipy LSODA dense output unreliable (events, states); outputs set aside in /root/outputs/saddle_dyn_try1_badevents
- 15:46 new step-wise driver validated; H-B re-gauge after 50-unit relaxation; sweeps relaunched (H-A 125 procs, H-B 62 procs)
- 15:55 LSODA time-resolution failures at t ~ 1e13+ fixed by local-clock restarts; sweeps resumed
- 16:21 H-A re-reference after relaxation, small first step on restarts, floor-index quadrature fixed; earlier attempts set aside; final H-A and H-B sweeps started
- 17:06 H-B done except 10 long runs; PS08 from small init: 14 seeds fitted, gamma = 2.0017 +- 0.0008 (SUPPORTED); PS06 control: 7/12 seeds learn the residual pair first (the verifier's q-first test misclassifies these); H-A at 831/1836; checks running
- 18:04 stall found (fresh LSODA on a plateau stays nonstiff); in-solver reference switch; 88 runs without outcome rerun (deviation 7)
- 18:51 final analysis; note Part II, README, report written; waiting for the last 40 control runs
