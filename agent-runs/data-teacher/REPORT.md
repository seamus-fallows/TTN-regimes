# data-teacher: do input covariance and teacher structure change the saddle-to-saddle picture? (Q6, Q7)

Status: working (steps 1-3 and 4a done; 4b/4c running) - Branch: agent/data-teacher - Updated: 2026-10-09 19:45 UTC

## Summary
Asked: starting from whitened inputs and an odeco teacher, add one complication at a time and decide
whether it changes the saddle-to-saddle (S2S) picture, and whether the baseline theory still
predicts it after a substitution. The deliverable table is at the top of `notes/data_teacher.md`.
- **Aligned covariance (step 1): quantitative change only**, predicted with *two* substitutions.
  The race (order, escape times) follows the whitened teacher with weights Lambda_r s_r (paired runs
  agree to < 1%). The landscape (plateau levels, stable saddles, swaps) follows the absorbed white
  teacher sqrt(Lambda_r) s_r. When the two disagree, binary-4 unlearns a component and learns the
  other one: a strict-saddle swap whose Hessian eigenvalue matches the derived formula to 1e-15.
- **Misaligned covariance (2), non-orthogonal CP teachers (4a), and partly cross-site correlations (3)
  change the later plateaus *qualitatively* on MPS students**: the second escape becomes logarithmic in
  sigma (a 2-core strict-saddle route opens) instead of sigma^-(K-2). Measured log slopes match the
  theory_teachers.md prediction (A = 31.9 +- 1.6 vs 34.1; 4.7-5.6 vs 5.68). Plateaus are always
  critical points of the bond-rank-constrained loss (levels to 1e-5 - 5e-3), but they are blends,
  not teacher components. Binary-4 (a depth-3 linear network) keeps its sigma^-1 law.
- **Cross-site correlations (3)**: S2S survives, and online SGD reproduces the exact population flow.
  No whitening substitution exists; on binary-4 the first saddle picks up non-teacher pairing features.
- 4(b) random TTN teachers and 4(c) cross-tree teachers are running (dense library runs).

## Results

Gradient flow throughout: scipy DOP853, rtol 1e-9, atol 1e-14, float64 (the pilot integrator, until
`ttn/dynamics.py` lands). N = 4, d = 4, chi = 4. Site covariance `power_law_cov(4, alpha)`; teachers
T1 s = (1, 0.8) and T2 s = (1, 0.6), with the smaller component on the high-variance eigenvector
(Lambda ratio 2). 40 seeds per setting. Conservation-law drift < 1e-9 in every run.

| claim | number (evidence) | status |
|---|---|---|
| Aligned init: Pi_r(t; Sigma) = Pi_r^white(Lambda_r t) exactly | half-times match the scalar ODE to 1e-9 relative; the substitution s -> Lambda s gets half-times to 0.1-2% but widths wrong by up to 45% (`experiments/data_teacher/results/e1_exact_aligned.csv`, `.png`) | derived + empirical |
| First escape = whitened problem with weights Lambda s | same first component in 100% (MPS-4) / 95-100% (binary-4) of 960 pairs; times differ 0.02-0.7% (max 1.3%), shrinking like 1/t_1 (`results/e1_paired_agg.csv`, `e1_paired_times.png`) | derived + empirical |
| Covariance reverses the order (T1) | the 0.8 component is learned first in 90-98% (binary-4) / 78% (MPS-4) of runs; P(first) under Sigma = P(first) under (I, Lambda s) (`results/e1_order_table.csv`, `e1_order_probs.png`) | empirical |
| Swap criterion = Lambda s^2 (landscape) | binary-4, T2: P(swap given the Lambda s winner was learned first) = 0.33 / 0.68 / 0.94 at sigma = 0.1 / 0.05 / 0.02; P(swap given the other) = 0/30; MPS-4: 0 swaps in 960 runs (no 2-core route) (`e1_order_table.csv`, `e1_examples.png`) | derived (Hessian, notes 1.3) + empirical |
| Race is set by the flow metric, not the landscape | same landscape and same start (mapped init, absorbed white problem): different first component in 10-15% (MPS) / 12-53% (binary) of seeds, escape 11-23% later (`e1_paired_agg.csv`) | empirical |
| Plateau levels 1/2 sum_unlearned Lambda_r s_r^2 | relative error 1e-7 to 1e-3 (`results/e1_plateaus.csv`) | empirical |
| Exponent of t_1 in sigma unchanged | binary-4 −1.010 +- 0.002 (Sigma) vs −1.012 +- 0.002 (white); MPS-4 −1.999 +- 0.001 vs −2.001 +- 0.001; 6 sigma over 1 decade, 10 seeds (`results/e1_scaling_fits.csv`, `e1_scaling.png`) | empirical |
| t_1 ~ (Lambda s)^-1 | slope −0.960 +- 0.007 (binary-4), −1.009 +- 0.001 (MPS-4) over 3.6 decades of Lambda s (`e1_scaling_fits.csv`) | derived (leading order) + empirical |

| Step 2: plateau = rank-1 critical point of the Sigma-loss | median rel. error 1e-6 - 1e-3 at every theta, both variants (`results/e2_misaligned_plateaus.csv`, `e2_misaligned_summary.png`) | empirical |
| Step 2: early alignment with Sigma W*, then realignment | |cos| with race tensor 0.90-0.997 early, with saddle 0.998-1 on the plateau | empirical |
| Step 2: later plateau law on MPS-4 changes | theta = 0: sigma^-1.98+-0.005; theta = 20 deg: Dt_2 = 15.0(+-0.4) log(1/sigma) + B (`results/e2_sigma_law_fits.csv`, `e2_sigma_law.png`; library LSODA) | empirical |
| Step 3: S2S survives correlated inputs; plateaus = rank-1 critical points of the non-product loss | levels to <= 5e-3; binary-4 first plateau 0.33 -> 0.10 L(0) at rho = 0.5, blended saddle with non-teacher pairing features; MPS-4 later plateau 137 -> 0.14 (`results/e3_correlated_plateaus.csv`, `e3_correlated_summary.png`) | empirical |
| Step 3: sampled data | online SGD (eta 0.01/0.002, B 256/4096) matches exact population GF event times within 0.2-1.4 time units (`results/e3_sgd_check.csv`, `.png`) | empirical |
| Step 4a: rank-1 saddle count | MPS-4: 2 for c < 1/2, 1 above (s2 = 1); binary-4 always 1 (`results/rank1_predictions_*.csv`) | derived + numerical |
| Step 4a: later plateau law (MPS-4) | c = 0: sigma^-1.97+-0.01; c = 0.2: A = 31.9 +- 1.6 (pred. 34.1); c = 0.45: A = 4.7 +- 0.3 (pred. 5.68) (`results/e4a_sigma_law_fits.csv`, `e4a_sigma_law.png`) | derived (theory_teachers R6) + empirical |
| Swap Hessian (step 1) | library gauge-projected lambda_min = derived formula to 1e-15; strict/non-strict correct in 24/24 cases (`results/e1_hessian_check.csv`) | derived + numerical |
| Integrator cross-check | pilot (scipy DOP853) vs library integrate_gf (LSODA): max rel. difference 7e-7 in event times and losses (`results/crosscheck_library.csv`) | check |

Figures: `experiments/data_teacher/results/e1_examples.png` (linear-time trajectories with a swap),
`e1_order_probs.png`, `e1_paired_times.png`, `e1_scaling.png`, `e1_exact_aligned.png`. Derivations and
the full list are in `notes/data_teacher.md` (sections 0, 1).

## What was done
1. Theory for aligned covariance, Facts A (binary-4 = depth-3 DLN) and B (whitened frame), swap
   Hessian (notes/data_teacher.md) — a613ec8.
2. Pilot (8 seeds) showing swaps on binary-4 and none on MPS-4 — a613ec8.
3. Experiment code: config-driven runners, exact aligned check, constrained-minima tool, setups for
   steps 2-4, exact moment matrix for correlated inputs — 0620017..6c60e71.
4. Step-1 sweeps (1920 + 360 + 520 runs) and analysis — 60a648d, this commit.

## Decisions
- **Pilot integrator (scipy DOP853 on the flat vector) for all runs so far**, rather than waiting for
  `ttn/dynamics.py`. The briefs allow it, the runs are cheap, and when the library lands I will rerun
  key configs with it as a cross-check. Loss evaluated through the dense tensor: identical to
  `population_loss` to 1e-17 and 4.5x faster at N <= 8.
- **Four variants per run, all from the same init**: (Sigma, s), (I, Lambda s), (I, sqrt(Lambda) s), and
  the last from the mapped init. Paired comparisons separate race from landscape far more sharply than
  comparing distributions; the mapped variant answers the lead's request to separate the landscape
  from the gradient-flow metric.
- **Teachers with a controlled conflict between orderings** (T1: all orderings except s agree; T2: race
  and landscape disagree), on binary-4 (2-core correction route) and MPS-4 (none). Covariance
  exponent alpha = 0.25 gives a Lambda ratio of 2: enough to reverse the order without making the
  problem stiff.
- **Correlated inputs (step 3) via the exact moment matrix** (Isserlis, N = 4: 256 x 256) rather than
  only SGD on samples: exact population gradient flow with a non-product metric, plus an online-SGD
  check (`e3_sgd_check.py`) as the brief asks.
- **Stopping rule**: runs stop at L < 1e-6 L(0) (1e-8 for runs started before 18:45). The terminal
  state is recorded (a recorder bug that dropped it was found and fixed; the 99 affected step-1 runs
  were rerun).
- **Library fix 57d5262** (shared teacher/init RNG streams): it does not affect my runs. CP teachers
  use the deterministic standard basis, and `spec.init` is unchanged (checked: the saved init is
  bit-identical after the merge). TTN teachers (step 4b/c) use `teacher_generator` with teacher seeds
  101+/201+.

## Files left on the machine
- `/root/outputs/data_teacher/e1_order/` (2.7 GB, 1920 npz: trajectories, spectra, parameters),
  `e1_scaling_sigma/` (0.4 GB), `e1_scaling_lambda/` (0.6 GB), `e1_exact/`, `pilots/`. They are
  needed only to redraw figures; all are regenerated by the commands in
  `experiments/data_teacher/README.md`.

## Caveats and next steps
- Time resolution: 2000 log-spaced output points per run. Event times are interpolated, about 0.5%
  resolution.
- The step-1 swap probability depends on sigma (strict saddle in log time vs polynomial learning).
  The sigma -> 0 limit is extrapolated, not reached.
- Next (running now): step 2 (misaligned, 800 runs), 4(a) non-orthogonal CP (560), step 3 correlated
  inputs (480 + SGD check), 4(b) random TTN teachers (150, sigma sweep), 4(c) cross-tree teachers
  (N = 8), 4(a) sigma sweep for the polynomial-to-log crossover (theory_teachers.md P8).

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
- 18:55 Recorder bug found (terminal state not stored when the stop event fell between output
  points; 99 step-1 runs missed their last event); fixed, affected runs rerun.
- 19:00 Step-1 sweeps done (order 1920 runs, sigma scaling 360, Lambda scaling 520); analysis and
  interim report. Steps 2-4 sweeps queued.
- 19:10 Steps 2 (800 runs), 4(a) (560), 3 (240) done and analyzed: plateaus = rank-1 critical points
  of the (Sigma- or M-) loss in every case; misalignment / overlap / correlation collapse the MPS-4
  rank-1 plateau (2-core strict-saddle route), not binary-4's.
- 19:20 Library v0 merged. Cross-check pilot vs ttn.dynamics.integrate_gf (LSODA): max relative
  difference 7e-7 in event times and losses (results/crosscheck_library.csv). Float64-floor scan
  (lead's note): no dormant signal below 1e-11 in any CP run (results/float_floor_check.csv).
- 19:25 Found that the 2000-point log grid cannot resolve the short (log-time) later plateaus of
  generic teachers; new two-stage dense runner on the library integrator (run_dense.py); 4(b), 4(c)
  and the sigma sweeps (e4a_sigma, e2_sigma) rerun with it (pilot versions discarded).
- 19:40 Plateau laws: e4a_sigma and e2_sigma (dense, library LSODA) show polynomial -> logarithmic
  change with overlap / misalignment, slopes as predicted. Exact Hessian check of the swap criterion.
