# Brief: `s2s-core` — saddle-to-saddle phenomenology and escape-time scaling (Q1, Q2)

Read `notes/briefs/README.md` first (instructions for all agents).

## Task
Establish whether, and how, gradient flow on a TTN from small initialization shows
saddle-to-saddle dynamics, and measure how plateau durations scale with init scale, tree size and
bond dimension, against theory.

## Goal
Clean, quantitative evidence for the report's core chapter: (1) the known Saxe-type result
reproduced in our setup; (2) the saddle sequence for random init; (3) escape-time scaling laws with
fitted exponents and prefactors compared to `notes/theory_escape.md`.

## Context
- Questions Q1 and Q2 in `PLAN.md`; theory in `notes/theory_escape.md` (escape from the origin,
  effective initial amplitude, dependence on bond dimension, learning order) and
  `notes/theory_saddles.md` (saddle types; read the parts on odeco teachers).
- Library: `ttn.network.aligned_init`, `ttn.teachers.odeco_teacher`, `ttn.dynamics.integrate_gf`,
  `ttn.theory` (exact balanced-mode solutions, escape times), `ttn.diagnostics` (events, plateaus,
  Hessian with gauge projection), `ttn.runner` (configs, sweeps).
- Other agents cover: dependence on where components live in the tree (`locality`), learning order
  statistics (`order`), the regime map (`regime-map`), input covariance and teacher structure
  (`data-teacher`), learning rate / SGD / finite samples (`optim-samples`).

## What to do
1. **Known result in our setup.** Odeco teacher, aligned balanced init: the dynamics is exactly a
   set of independent Saxe-type modes. Show TTN gradient flow vs `ttn.theory` for binary N = 4, 8,
   MPS N = 4, 8, HT N = 4 (overlay; max deviation). This is a known result (Saxe et al. 2014 for
   DLNs); the figure demonstrates it, it is not a new claim.
2. **Random init phenomenology.** Fan-in init, odeco teachers with 3 well-separated weights
   (e.g. [3, 2, 1]) and 3 close weights (e.g. [1, 0.85, 0.7]); trees binary N = 4, 8, MPS N = 4, 8,
   HT N = 4; chi = number of components (exact) and 2x that (overparameterized); sigma over >= 1.5
   decades chosen so runs are tractable; 10 seeds. Record loss, overlaps, bond spectra, effective
   ranks. Apply the S2S criterion of `notes/diagnostics.md`. Are plateau levels the partial-fit
   levels L_k? Do bond spectra grow one value at a time on every edge simultaneously? In which order
   are components learned (just record it; `order` studies it in depth)?
3. **Escape-time scaling.** t_1 (first event) and the later gaps vs sigma: fitted exponents vs
   K - 2; prefactor distribution over seeds vs the theory note's prediction; t_1 vs chi at fixed
   sigma; t_1 vs K across trees and N. Separate clearly what is predicted with no free parameters.
4. **Saddle type.** At plateau midpoints, gauge-projected Hessian spectra: number of negative
   eigenvalues and lambda_min, for the odeco runs (theory: higher-order saddles, no negative
   curvature in the escape direction, when K >= 3).

## Deliverables and success criteria
- `experiments/s2s_core/` (README, configs, code, results with CSV + figures) and
  `notes/s2s_core.md` with status-tagged findings.
- Figures: (a) exact-solution overlay; (b) representative run: loss (log and linear time),
  overlaps, bond spectra per edge, effective ranks; (c) t_1 vs sigma per tree with fits and theory;
  (d) gaps vs sigma; (e) t_1 vs chi; (f) Hessian spectra at plateaus.
- A table of fitted exponents (with standard errors) vs predictions.

## Constraints
- First interim report (pushed) after step 1 and a first pass of step 2 (~3 h of work).
- Raw outputs to `/root/outputs/s2s_core/`.
- Time estimate: about one working day of agent time.
