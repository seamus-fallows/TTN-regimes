# s2s-core: saddle-to-saddle phenomenology and escape-time scaling in TTNs (Q1, Q2)

Status: done (fix round 1 complete) - Branch: agent/s2s-core - Updated: 2026-10-10 10:20 UTC

## Summary
Asked: does gradient flow (GF) on a TTN from small init show saddle-to-saddle (S2S) dynamics, which
saddles it visits, and how plateau durations scale with sigma, the tree (K cores) and the bond
dimension chi, against theory. The lead added a question: are dormant components suppressed during
earlier escapes, and are the runs stuck or on ever-longer plateaus? A fix round after review round 1
(`notes/reviews/s2s_core_round1.md`) re-analysed all raw data; the claims that changed are listed
below and in `notes/s2s_core.md` section 4.

Answer (odeco teachers, d = 4):
- **First escape.** Predicted per seed, with no free parameter, by the correlation-flow blow-up time
  (a replication of theory_escape E1), to <= 2e-3. The Kumar-Haupt law is a lower bound, 3-2700x too
  small.
- **Where S2S was measured.** Complete S2S sequences were measured on binary4 and mps4, and on binary8
  and ht4 at chi = 6 (0.6 decade of sigma). On mps8 no run learns the third component. Plateaus sit at
  the bond-rank-constrained partial fits, and every edge gains one rank at each event.
- **Gaps.** Fitted on runs classified by order and S2S status, the gaps scale as sigma^-(K-2) per seed:
  - K = 3, 4: within 0.1;
  - binary8: per-seed exponents 4.9-5.1;
  - ht4: 3.9-5.8;
  - mps8: first gap only.
- **Saddle type.** Plateaus are higher-order saddles except on binary4 wrong-order plateaus. There a
  strict saddle appears, as predicted, measured at 0.98 of the predicted eigenvalue.
- **Main limit at large K: the conservation-law trap of fan-in init.**
  - Trapped fraction: 0 at K = 3; 40% (separated teacher) and 18% (close) at K = 4, chi = 3, fixed per
    init direction; at least 30-80% at K = 7, 8, chi = 3; 0-14% at chi = 6.
  - Gauge-balanced init: 0/120 trapped where fan-in traps, but most of its K = 7, 8 runs stall.
  - Stalls are predicted to be ever-longer plateaus; this was checked on only a few runs.

## Fix round (review round 1): claims that changed
| claim before | now | evidence |
|---|---|---|
| no strict saddles with odeco teachers; plateaus higher-order | strict saddle on the binary4 wrong-order plateau (close teacher, seed 8, 7/8 sigmas): lambda_min -0.118 .. -0.146, flat in sigma (slope 0.03), 0.98 of the predicted -0.133; higher-order only on greedy binary4 and tensor-like trees | `main_strict_saddles.csv` |
| every gap ~ sigma^-(K-2) (pooled fits) | per-seed fits on runs classified by order and S2S status (rho > 3); binary4 close gap 2 = 1.016 +- 0.011 (was 0.92 +- 0.11 pooled); ht4 per-seed 3.9-5.8; mps8 gap 1 = 5.95 +- 0.05 (new) | `main_gap_fits.csv`, `main_exponents.csv` |
| S2S on every tree | complete S2S on binary4, mps4 (chi 3, 6), binary8 / ht4 at chi = 6 only; mps8 0/284 learn a 3rd component; K = 7 at chi = R completes rarely (binary8 14/180, ht4 12/150) | `main_runs.csv` |
| trap decay ~ sigma^2 within 1.8x, two seeds switch levels by 20-100x | constant per seed to 1-3% (4 separated seeds, sigma 0.003-0.1), 1.25-1.33x (close) once fitted above the float64 floor (1e-9); the switch was an artifact; K = 7, 8 rates not sigma^2-clean | `dormant_decay.csv` |
| measured rates 3-15x below the single-end estimate | 1.1-5.6x below (ratio 0.18-0.87) | notes 2.7c |
| first component = correlation flow's except binary4 close (80-90%, wrong order at finite sigma) | 100% everywhere: the 80-90% came from stopping the flow at amplitude 1e6; now stopped on direction convergence; T moved by <= 1.7e-3 | `first_runs.csv`, `recompute_T.py` |
| stalls are ever-longer plateaus (10 of 11 escape) | derived prediction; 8/8 shortest-estimate k78 runs (3 seeds) escape at 0.50-0.87 of the estimate; mps8 chi = 6 spread: 2 escape at 1.4-1.8x, 1 not at 2x, 5 unresolved; random sample of 8 of the 74 k78 runs with estimate <= 100x elapsed: 7 escape at 0.24-2.19x, 1 not by 3x | `continued_*.csv` |
| trap fraction 40% at K = 4; 30-80% at K = 7, 8 | 40% separated teacher only (close: 14/80 trapped + 8/80 stalled); K = 7, 8 fractions are lower bounds (stalled runs partly still decaying) | `main_classes.csv`, `slice_trends.csv` |
| balanced init removes the trap (0/210); chi = 2R removes it | 0/120 in the informative cells (fan-in 12/30, 47/90); balanced K = 7, 8 still stall 69/90; chi = 6: ht4 5/50 trapped (seed 9 at every sigma), mps8 7/50; "2R" literal only on binary trees (MPS end / HT leaf bonds capped at d = 4) | `main_classes.csv` |
| edge rank steps 517/551; >100x drops 0% at chi = 2R; later escapes "same picture"; R10 magnitude 0.8-1.7; plateau error <= 2e-2, mps4 2e-3; KH 3-500x | 497/531 (332 untested); 0.6-2%; up to 8-11% >100x at K = 7, 8, chi = 3; 0.87-1.15 (K = 3, 4) and 1.4-3.2 (K = 7, 8); binary4 2.0e-2 (up to 0.50 at unresolved close plateaus), mps4 2.8e-3 (8.1e-3 balanced); 3-2700x | notes 2.3-2.7 |
| (tags) | first escape, KH bound, d-flatness, chi slopes and race order are replications of theory_escape E1/E2/E5/E7/E8; binary4 "always largest first" rests on 20 init directions; sd ln T (d = 2) 0.218 vs 0.250 +- 0.02 | notes 2.2b, 2.2c |

## Results
Gradient flow throughout: LSODA with exact Hessian Jacobian, rtol 1e-8 / atol 1e-11 sigma (main,
first, balanced), rtol 1e-6 / atol 1e-9 sigma (large, k78; event times agree with rtol 1e-8 to 7e-7,
and 9 K = 7, 8 trapped / stalled runs re-run at rtol 1e-8 keep class and order, events to 3.6e-6);
DP5 rtol 1e-10 for the aligned runs. Whitened inputs, d = 4, odeco teachers (seed 1000) separated
[3, 2, 1] and close [1, 0.85, 0.7], fan-in init unless stated, 10 seeds per cell. Event times and
widths are resolved by re-integrating each event window on linear grids (>= 40 samples per rise).
All tables: `experiments/s2s_core/results/`; derivations, status tags and full numbers:
`notes/s2s_core.md`.

**1. Known result** (`aligned.csv`, `fig_a_aligned.png`): aligned balanced init follows the exact
Saxe modes; max |a_GF - a_theory| = 2.0e-10 (binary4), 3.2e-10 (mps4), 2.3e-9 (binary8), 2.6e-9 (ht4),
2.7e-9 (mps8). Derived; known for DLNs (Saxe 2014 eq. 15), explicit for TTNs here.

**2. Phenomenology** (`main_runs.csv`, `main_classes.csv`, `fig_b_run_*.png`):
- complete S2S sequences: binary4 320/320; mps4 160/160 at chi = 6, 48/80 + 58/80 at chi = 3 (rest
  trapped / stalled); binary8 50/50 and ht4 45/50 at chi = 6 (sigma 0.07-0.3); at chi = 3, K = 7 rarely
  (binary8 14/180, ht4 12/150); mps8 never (0/284 learn a third component, 132 learn two);
- plateau loss = partial-fit level L_S at resolved plateaus to 2.0e-2 relative (binary4, sigma = 0.1),
  2.8e-3 (mps4; 8.1e-3 balanced), <= 2.1e-5 (K = 7), -> 6e-7 at sigma = 0.003; up to 0.50 at unresolved
  binary4 close-teacher plateaus. Every edge's top |S| singular values = the learned weights (truncated
  teacher spectrum), next value <= 0.07 s_min: each plateau is the bond-rank-constrained critical point W_S;
- each event raises the rank by one on every edge at the same time in 497/531 runs with event-window
  spectra (332 early complete runs untested; exceptions are overlapping events);
- S2S (every rho > 3) among complete runs: mps4 266/266, binary8 83/83, ht4 74/74; binary4 separated
  158/160, close teacher 5/20 at sigma = 0.1 rising to 20/20 at sigma = 0.003;
- order: binary4 largest first in all runs (20 init directions); on mps, binary8, ht4 the first
  component is random (P(largest) 0.4-0.8) but the same at every sigma for each seed (replication of
  theory_escape E8).

**3. Escape-time exponents** (per-seed fits on runs classified by order and S2S status, mean +- SE
[range of per-seed values]; `main_exponents.csv`, `main_gap_fits.csv`, `first_summary.csv`, figures c, d;
predicted K - 2):

| tree | K | chi | teacher | sigma range | t_1 (first sweep, offset removed) | t_2 - t_1 | t_3 - t_2 | seeds (gap 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| binary4 | 3 | 3 | sep. | 0.1-0.003 | 0.9996 +- 0.0001 | 1.003 +- 0.001 [1.00, 1.01] | 1.004 +- 0.003 [0.99, 1.02] | 10 / 10 |
| binary4 | 3 | 6 | sep. | 0.1-0.003 | 0.9996 +- 0.0001 | 1.007 +- 0.001 [1.00, 1.01] | 1.011 +- 0.002 [1.00, 1.02] | 10 / 10 |
| binary4 | 3 | 3 | close | 0.1-0.003 | 1.000 +- 0.003 raw (0.993 offset removed) | 1.004 +- 0.010 [0.95, 1.06] | 1.016 +- 0.011 [1.00, 1.09] | 9 / 8 |
| mps4 | 4 | 3 | sep. | 0.1-0.003 | 2.0000 +- 0.0000 | 2.000 +- 0.000 | 1.990 +- 0.005 [1.97, 2.00] | 10 / 6 |
| mps4 | 4 | 6 | sep. | 0.1-0.003 | 2.0000 +- 0.0000 | 2.001 +- 0.001 | 1.998 +- 0.001 | 10 / 10 |
| binary8 | 7 | 6 | sep. | 0.3-0.07 | 4.9990 +- 0.0002 | 5.020 +- 0.009 [4.98, 5.05] | 4.969 +- 0.022 [4.88, 5.09] | 10 / 10 |
| ht4 | 7 | 6 | sep. | 0.3-0.07 | 4.9987 +- 0.0007 | 5.10 +- 0.08 [4.83, 5.78] | 4.75 +- 0.14 [3.94, 5.33] | 10 / 9 |
| mps8 | 8 | 6 | sep. | 0.3-0.07 | 6.000 (raw, sweep large) | 5.95 +- 0.05 [5.79, 6.20] | not reached | 7 / 0 |
| mps8 | 8 | 3 | sep. | 0.4-0.03 (t_1: 0.4-0.01) | 5.9995 +- 0.0002 | 6.0 +- 0.3 (3 seeds) | not reached | 3 / 0 |

(Other close-teacher cells within 0.02 of K - 2. K = 7, 8 gaps span only 0.6 decade, below the
project's 1-decade rule; ht4 is consistent with 5 in the mean of gap 1 only.)

**4. First-escape prefactor** (`first_*.csv`, `lo_*.csv`, figures c, e, e2; replications of theory_escape
E1, E2, E5, E7 unless marked):
- per seed, |t_1 / (sigma^-(K-2) T(theta_hat) + c_K) - 1| at the smallest sigma: <= 8.5e-6 (binary4
  separated), <= 2.1e-3 (binary4 close), <= 6e-5 (mps4), <= 4e-8 (binary8, ht4, mps8 at sigma = 0.01);
  first component = the correlation flow's in 100% of runs. No free parameter.
- Kumar-Haupt (rho_0) is a lower bound: median T / t_KH = 3.1-4.8 (binary4), 7.9-13.7 (mps4), 95-210
  (ht4), 184-2650 (binary8), 1460 (mps8); up to 3e7 at K = 15 (leading order).
- d at full bond (200 inits): binary4 median 1.544, 1.569, 1.563 for d = 2, 3, 4 = pi/2, flat, while t_KH
  falls 1.9x; sd ln T 0.218 / 0.157 / 0.135 vs predicted 0.250 / 0.167 / 0.125 (+- 0.02).
- chi (100 inits): slope of the median over chi = 4-16: mps4 -0.26 +- 0.04, binary4 -0.34 +- 0.03 (t_KH
  -0.47); K = 7, 8 trees -0.69 .. -0.76 over chi = 2-16 vs -1.5 .. -2.5 for t_KH.
- K (50 inits, K = 3-15; new, heuristic): the RMS-overlap heuristic P3 = d^{N(K-2)/(2K)}/((K-2) s_1) (no
  fit) matches the median within 0.87-2.1x while the prefactor spans 0.4-814.
- Gauge-balancing the same init changes T per seed by 0.88-1.92x (median 0.99-1.18; usually slower).

**5. Saddle type** (`hess_*.csv`, `main_strict_saddles.csv`, figure f; gauge-projected and raw):
- higher-order where predicted: per-seed slope of log(-lambda_min) vs log sigma 1.06 / 1.02 (binary4
  separated), 1.17 / 1.12 (close), 2.00 / 1.99 (mps4), 5.10 / 5.07 (binary8), 5.13 / 4.88 (ht4), 5.99
  (mps8, first plateau); magnitude vs the route formula -(K-1) s_r (s_r |a_r|)^((K-2)/K): 0.87-1.15 at
  K = 3, 4, 1.4-3.2 at K = 7, 8 (routes not balanced at these sigma);
- strict where predicted: the binary4 wrong-order plateau (learned {1, 0.7}, 0.85 unlearned) has
  lambda_min -0.118 .. -0.146 over sigma 0.005-0.1, flat (slope 0.03), median 0.98 of the piggyback
  value s_q^((K-2)/K)(s_q - s_r) = -0.133; its gap does not grow (exponent -0.08).

**6. Dormant-component suppression** (`dormant_*.csv`, `main_classes.csv`, `slice_trends.csv`,
`continued_*.csv`, `tolcheck_k78.csv`, figure g):
- Escapes: median factor x2-3 (K = 3, growth), x0.64-0.74 (K = 4), x0.12-0.75 (K = 7, 8); > 100x drops
  in <= 1.3% (K = 4), 0.6-2% (K = 7, 8, chi = 6), 3-8% (K = 7, 8, chi = 3; 11% for mps8 later escapes,
  separated teacher).
- Plateaus kill them through the conservation-law trap (theory_saddles R12): route end slices decay
  exponentially at a rate ~ sigma^2 per seed (mps4: constant to 1-3% over sigma 0.003-0.1 for the 4
  trapped separated seeds), 1.1-5.6x below the single-end estimate; both ends on the light side of the
  conserved Q (2 runs). Run classes (separated teacher; trapped = dead route end, slice < 1e-8 sigma;
  at K = 7, 8 trapped is a lower bound):

| tree | K | chi | init | sweep, sigma | complete | trapped | stalled | unresolved |
|---|---|---|---|---|---|---|---|---|
| binary4 | 3 | 3, 6 | fan-in, balanced | main, balanced | 220/220 | 0 | 0 | 0 |
| mps4 | 4 | 3 | fan-in | main, 0.003-0.1 | 48/80 | 32 | 0 | 0 |
| mps4 (close) | 4 | 3 | fan-in | main, 0.003-0.1 | 58/80 | 14 | 8 | 0 |
| mps4 | 4 | 3 | balanced | balanced, 0.003-0.03 | 30/30 | 0 | 0 | 0 |
| binary8 | 7 | 3 | fan-in / balanced | k78, 0.15-0.3 | 2 / 3 of 30 | 9 / 0 | 18 / 27 | 1 / 0 |
| ht4 | 7 | 3 | fan-in / balanced | k78, 0.15-0.3 | 3 / 17 of 30 | 24 / 0 | 3 / 12 | 0 / 1 |
| mps8 | 8 | 3 | fan-in / balanced | k78, 0.15-0.3 | 0 / 0 of 30 | 14 / 0 | 16 / 30 | 0 / 0 |
| binary8 | 7 | 6 | fan-in | large, 0.07-0.3 | 50/50 | 0 | 0 | 0 |
| ht4 | 7 | 6 | fan-in | large, 0.07-0.3 | 45/50 | 5 (seed 9) | 0 | 0 |
| mps8 | 8 | 6 | fan-in | large, 0.07-0.3 | 0/50 | 7 | 34 | 9 |

- At K = 4 the trap is fixed by the init direction: the same 4/10 seeds trap at all 8 sigmas
  (trap time ~ sigma^-2, escape ~ sigma^-2). Gauge-balanced init: 0/120 trapped in the cells where
  fan-in traps (12/30, 47/90), but 69/90 balanced K = 7, 8 runs stall.
- Stalled runs with live routes (derived prediction: remaining plateau <= the balanced-route estimate
  b^-(K-2)/((K-2) s_r), b = (s_r |a_r|)^(1/K)): continued from their final state, the 8 k78 runs with
  the shortest estimate (3 seeds) all escape, at 0.50-0.87 of it (zoom-resolved; per seed constant
  across sigma); of 8 mps8 chi = 6 runs spread over the distribution, 2 escape at 1.4-1.8x (the
  reduction does not hold there), 1 not by 2x, 4 wall-limited, 1 integrator failure;
  a seeded random 8 of the 74 k78 stalled runs with estimate <= 100x the elapsed time:
  7 escape at 0.24-2.19x (median 0.55; 2 above the bound), 1 not by 3x. For most stalled runs the estimate is 10-400x the time integrated, so
  the prediction is checked on few runs only. Slice trends: k78 binary8 and mps8 balanced runs are
  pinned at 0.01-0.04 sigma; many fan-in mps8 runs are near the trap threshold and still decaying.

## What was done
1. Read brief, notes, library; theory derivation of the first-escape prediction (correlation-flow
   blow-up time; later equal to theory_escape E1), bounds, decoupling, saddle types; scipy pilots
   (stiffness of later plateaus, K sweep) (0c50e36, c53866f, 43b1620).
2. Experiment code on the library (`experiments/s2s_core/`: s2s.py, run.py, aligned.py,
   first_escape.py, lo_sweep.py, analyze_*.py, plot_*.py, recompute_hessians.py,
   continue_stalled.py) (137cc70, 55c4afb, be74002, ab25042).
3. Step 1, aligned init vs exact solution (137cc70).
4. Sweep `main` (1600 runs, K = 3, 4, 7, 8; fan-in), interim reports 1-2 (be74002, 9789170).
5. First-escape sweep (`first`, `first_k78`), leading-order sweeps (`lo_d`, `lo_chi`, `lo_K`)
   (94c4163, 788e235, 117a741, 5df02a7).
6. Dormant suppression: trap decay rate and sign rule (601a1c9, 7b7e7dc), dead-end classification
   (5ade509, 4b5392b), gauge-balanced inits (`balanced`, `k78`), long K = 7, 8 runs (`large`),
   continuation of stalled runs (ce7b78c, e878dc1).
7. Hessians recomputed after the library's projection fix (25fb840); final tables and figures
   (61afd74, 8fa2130); notes `notes/s2s_core.md` (sections 1-3), README.
8. Fix round after review round 1 (merged main 55519f3; WIP 7761da5, final: this commit): correlation
   flow stopped on direction convergence and T recomputed for all first-escape jobs (`recompute_T.py`,
   old values kept as `T_v1`); LO sweeps rerun; aligned runs rerun with one file per integrator;
   `analyze_main.py` refactored (per-plateau table, run-set classification `fit_sets`, strict-saddle
   table, class table); trap decay refitted above 1e-9; escape-drop and slice-trend tables; continuation
   escapes zoom-resolved, random continuation sample, tolerance check (`tolcheck.py`); figures c-g
   regenerated; notes 2.2-2.7, 3, 4 (claims changed), README, this report.

## Decisions
- **d = 4** at every site (odeco needs d >= 3; d = 4 leaves one physical direction outside the
  teacher). **Teacher seed 1000**, init seeds 0-9 (independent streams after the library fix).
- **Integrator: LSODA with the exact Hessian** for random-init runs (later plateaus are stiff: DP5 would
  need ~sigma^-(K-2) steps); DP5 as reference for aligned runs. Radau was tried on a K = 7 run and did
  not finish in 25 min (rejected). Tolerances: rtol 1e-8, atol 1e-11 sigma (main, first, balanced);
  rtol 1e-6, atol 1e-9 sigma for the long K = 7, 8 runs (large, k78) after checking on binary8 chi = 6
  sigma = 0.07 that event times agree with rtol 1e-8 to 7e-7 relative (975 s vs 1499 s).
- **Events = final rise of a_r** (last upward crossing of 1/2), widths 0.1 -> 0.9 of the final rise,
  both refined by re-integrating the window on linear grids until the rise holds >= 40 samples. No
  duration is read off the log grid.
- **Run budgets**: t_max = 1e3 x the RMS-overlap escape time of the weakest component; wall limits
  300-2400 s; a stop when a never-learned component has a dead route end (slice < 1e-8 sigma for a
  whole output chunk: conservation-law trap, nothing after it is physics). Stalled / unresolved runs
  are reported as classes, never dropped.
- **Corrected mid-course**: (1) an amplitude-based roundoff criterion (|<W,U_r>| < 1e-15 -> stop;
  < 1e-14 -> "floor") turned out too strict at K >= 7 (the amplitude is a product of K per-core
  amplitudes): replaced by the per-core dead-end criterion; K = 7, 8 runs affected are re-run (k78).
  (2) The gauge projection of the Hessian was fixed in the library: all lambda_min values recomputed
  from the saved midpoint parameters (raw and projected reported).
- **Cuts for compute** (each documented in the configs): K = 7, 8 full runs at sigma >= 0.07 only (at
  0.05, 0.03 none of ~50 runs completed); chi = 6 at K = 7, 8 only with the separated teacher (large);
  first-escape sweep at K = 7, 8 only with the separated teacher and fan-in init; k78 at sigma =
  0.3-0.15. Finished runs of the larger grids are kept and reported.
- **sigma grids**: K = 3, 4: 0.1 .. 0.003 (1.5 decades) for full runs, 0.1 .. 0.002 for the first
  escape; K = 7, 8: first escape 0.4 .. 0.01 (1.6 decades), full runs 0.4 .. 0.07 (later gaps at
  smaller sigma are not reachable in float64 within budget; reported as such).
- **Own sweep driver** (`run.py`) and analysis code in the experiment folder (the library runner landed
  after the sweeps had started); the library's integrator, Recorder, diagnostics and theory are used
  throughout.

- **Trapped-class threshold 1e-8 sigma** (7 decades below the init scale, above both atol settings).
  The minimum dormant slice has a large mass below 1e-12 sigma (dead ends) and a continuum 1e-10 ..
  1e-4 sigma (ends still decaying when a run stopped), so the trapped fraction moves by a few runs per
  cell between 1e-10 and 1e-8 (sweep `main`, chi = 3: binary8 / ht4 / mps8 trapped 24 / 77 / 24% at
  1e-10 sigma vs 27 / 82 / 30% at 1e-8 sigma).
- **Pooling**: exponents are fitted per seed only over the sigmas at which the seed follows its modal
  learning order, and only on complete runs; K = 7, 8 cells are taken from the sweep with the longest
  budget (large / k78), never pooled across tolerance settings.
- **Continuation test** of stalled runs on the 8 with the shortest predicted remaining plateau (to fit
  the time left); a spread sample from `large` was started as a check of the bias.

- **Fix round: run sets for the fits.** Options: (i) pool complete runs (before; mixes wrong-order and
  overlapping-event runs into the fits); (ii) require every rho > 3 in a run; (iii) classify per step.
  Chose (iii): t_1 on runs with the seed's modal first component, gap 1 on runs whose first two
  components are the modal pair with rho_1 > 3 (so mps8 runs that learn two components count), gap 2
  on complete runs in the modal order with rho_1, rho_2 > 3. The same sets drive the Hessian slopes and
  figure d (`analyze_main.fit_sets`).
- **Fix round: correlation-flow stop.** Options: a larger fixed amplitude stop (the reviewer used 1e10,
  1e14) or a direction criterion. Chose the direction criterion: integrate in segments (target x1e3)
  until the second-largest component amplitude is <= 1e-6 of the largest and still falling, cap 1e30;
  T is extrapolated from the stop as before. The old T, r1, T_single are kept in each JSON (reversible).
- **Fix round: trap-fit floor 1e-9 absolute** (dead slices bottom out at 1e-12 .. 1e-15 absolute; three
  decades of margin; a relative-to-sigma floor would cut long fits at small sigma).
- **Fix round: optional runs done, small**: zoom re-resolution of the 8 continuation escapes (values
  moved by <= 1.8%), a 9-run tolerance check (one per tree / init / class), a seeded random continuation
  sample of 8 (`--seed 1`, among the 74 stalled k78 runs with estimate <= 100x the elapsed time; the
  33 with longer estimates would not fit the wall limit).
- **Stale files kept**: `/root/outputs/s2s_core/aligned/{tree}.npz` (pre-fix, one file for both
  integrators) are left in place, not deleted; nothing reads them.

## Files left on the machine
All raw outputs are in `/root/outputs/s2s_core/` (not in git; ~220 MB in total), one `.npz` (log-grid
trajectory, refined event windows with spectra, parameters at plateau midpoints, final state) and one
`.json` summary per run: `main/` (1600 runs, 132 MB), `large/` (150, 36 MB), `k78/` (180, 33 MB),
`balanced/` (120, 12 MB), `first/` (238 first-escape jobs, 1 MB; JSONs carry the recomputed T and the
old `T_v1`), `aligned/` (2 MB; `{tree}_{dopri5,lsoda}.npz` current, `{tree}.npz` stale pre-fix copies),
`k78_tight/` (9 tolerance-check reruns, 2 MB). Needed to recompute
the analysis tables (`analyze_*.py`, `recompute_hessians.py`, `continue_stalled.py`); everything is
regenerated by the commands in `experiments/s2s_core/README.md` (~6 h on 60 CPUs).
`/root/outputs/s2s_core/first_test/` is an empty test directory.

## Caveats and next steps
- **K = 7, 8 later gaps** span only sigma = 0.07-0.3 (0.6 decade, below the project's 1-decade rule)
  and come from chi = 6 (binary8, ht4); ht4 per-seed exponents scatter (3.9-5.8); at chi = 3 too few runs
  complete; mps8 never learns its third component, so only its first gap is measured (7 seeds). Smaller
  sigma is out of reach: the gaps grow like sigma^-5..-6 and LSODA advances slowly on these plateaus.
- **First-escape sweep at K = 7, 8** was cut for compute (separated teacher, fan-in only); mps8 at
  chi = 6 is missing (jobs stopped at the end: first escapes at sigma = 0.01 are at t ~ 1e13).
- **Trap statistics** are for one teacher basis (seed 1000) and the fan-in init of the model; the trap
  fraction depends on the init's Q_e, i.e. on the init scheme.
- **Stalled vs stuck** is a derived prediction checked on few runs: 8 shortest-estimate k78 runs
  (3 seeds, all escape at 0.50-0.87x), a random 8 of the 74 k78 runs with estimate <= 100x the elapsed
  time (7 escape at 0.24-2.19x, 1 not by 3x) and 8 mps8 chi = 6 runs (2 escape at 1.4-1.8x, 1 not by
  2x, 5 not integrated far enough). Most stalled runs have estimates 10-400x the integrated time and
  are untested; at K = 7, 8 the trapped fractions are lower bounds (stalled fan-in mps8 runs are partly
  still decaying toward the threshold).
- **Trap decay rate ~ sigma^2** is checked at K = 4 only (mps4, 6 seed / teacher pairs); at K = 7, 8 the
  fits are short and the rate / sigma^2 varies with sigma.
- R10 magnitudes at K = 7, 8 deviate 1.4-3.2x (unbalanced routes at moderate sigma); the slopes are clean.
- The strict saddle rests on one seed (7 sigmas) plus a second seed at 2 sigmas, close teacher, binary4.
- The "close" teacher [1, 0.85, 0.7] gives overlapping events on binary4 for sigma >= 0.03.
- Next: K = 7, 8 gaps over a wider sigma range would need a cheaper stiff integrator or a reduced
  route ODE (theory_saddles R5) seeded from measured amplitudes; the trap probability vs the init's
  per-bond Q statistics (theory_saddles "all ends unfavourable" rule) could be tested on many seeds at
  the leading order without full runs.

## Log
- 18:18 Started; read BRIEF.md, CLAUDE.md, notes/model.md, notes/diagnostics.md, PLAN.md, library.
  Machine: cgroup CPU quota 61.4 CPUs (6144000/100000), 251 GB RAM.
- 18:25 Pilot (scipy DOP853, throwaway): binary N=4 GF at sigma 0.3..0.003: t_1 sigma^{K-2} -> 0.7696.
- 18:29 Derived first-escape prediction (P1): t_1 = sigma^{-(K-2)} T(theta_hat) + c_K(s) with T the
  blow-up time of the correlation flow theta' = grad <W, W*> from the sigma=1 init (exact scaling).
  Pilot: matches to 4-5 digits for binary 4/8, MPS 8 (incl. the identity of the first component).
- 18:31 Found stiffness problem: after the first escape, explicit RK steps are capped at
  O(1/s^{2-2/K}), so integrating gaps costs ~ sigma^{-(K-2)}. scipy LSODA/BDF with exact Hessian
  Jacobian (torch.func.hessian, 0.1 s for 850 params): cost independent of sigma (binary 4:
  19 s at sigma 3e-3 and 3e-4 vs 55 s / ~550 s with DOP853), same event times to 1e-6.
- 18:34 Pilot, 40 seeds x 5 trees x chi {3,6} x 2 teachers, correlation flow only: prefactor of t_1 is
  ~ (sigma d^{-N/2K})^{-(K-2)}/((K-2)s) (RMS-overlap heuristic) within factor 1-1.75; the rho_0-based
  hypothesis in PLAN.md (aligned at start) underestimates by 4x-5600x; chi-dependence weak.
- 18:45 Lead messages: merged main (literature review; core fixes 57d5262 incl. teacher/student
  stream bug -> pilots before it marked [old lib], to be redone; draft theory notes). Read
  literature.md 1.3-1.4, 2.1, 2.4, 3, 5 and theory_saddles.md.
- 18:50 K sweep (fixed lib, 17 trees, K=3..15, 30 seeds): P3 within 0.94-2.1x of T; Kumar-Haupt
  formula with delta=||theta_0|| off by 3x..3e7x. Gaps pilot (BDF, 5 trees x 6 sigmas x 5 seeds)
  running to test gap exponents and the conservation-law trap (theory_saddles R12) at K=7,8.
- 19:20 Library v0 landed and merged; experiment code on the library: `experiments/s2s_core/s2s.py`
  (setup, fan-in / gauge-balanced / aligned inits, run with event zoom + plateau Hessians + dormant
  diagnostics, correlation-flow prediction), `run.py` (YAML sweep driver until ttn.runner lands).
- 19:45 Step 1 done: aligned balanced init vs ttn.theory on binary4/8, mps4/8, ht4: max |a_GF - a_th|
  2e-10..3e-9 (dopri5, rtol 1e-10), event times equal to 1e-11 relative (same grid).
- 20:10 Lead: dormant-component suppression is now an explicit question. First observations
  (binary8 / mps8, K = 7, 8, sigma = 0.1-0.2, seed 0, chi = 3): after one or two escapes a dormant
  component's amplitude |<W,U_r>| collapses to 1e-17 .. 1e-41 (below the float64 floor), the loss
  stalls at the partial-fit level, and LSODA crawls. It happens DURING another component's escape
  (e.g. a_2: 2e-12 -> 1e-17 while component 3 escapes), and also slowly on plateaus; the
  gauge-balanced init of the same seed is also suppressed (binary8 chi = 3). chi = 6 keeps the
  dormant amplitude at ~1e-12 in the one test so far. Merged theory_saddles v2.
- 20:45 Interim report 1 (step 1 done, first pass of step 2). Sweep A running.
- 20:58 Sweep A restarted twice: (1) added a roundoff stop (all dormant amplitudes < 1e-15 for a whole
  chunk) and tighter step/wall limits; (2) dropped sigma = 0.05, 0.03 for K = 7, 8 full runs (none of
  ~50 completed; LSODA crawls to the step limit after 1-2 escapes). Per-seed exponents so far: t_1 and
  both gaps 1.00 +- 0.01 (binary4) and 2.000 (mps4) vs K - 2. Trap decay rate ~ sigma^2 per seed.
  Event windows now record bond spectra: every event raises the rank by one on all edges at the same
  sample (mps4 test).
- 21:35 K = 7, 8 later gaps are long (binary8 chi 6 sigma 0.07: events 7.5e5, 5.8e6, 4.0e7) and LSODA
  needs ~15-25 min per run; rtol 1e-6 / atol 1e-9 sigma reproduces rtol 1e-8 event times to 7e-7
  relative, so `large.yaml` (K = 7, 8, chi = 6) uses those. Queue: main -> (large || first) ->
  balanced -> LO sweeps (lo_d, lo_K, lo_chi). Trap sign rule checked (both route ends unfavourable).
- 21:50 Interim report 2. Sweep A done and reclassified (dead-route-end criterion instead of the
  amplitude threshold, which was too strict at K >= 7). Running: first (B), large; queued: k78,
  balanced (K = 3, 4), LO sweeps. Library final merged; projection-bias bug noted.
- 22:44 Mistake, fixed at once: a `mv <file> /dev/null` replaced /dev/null with a regular file for
  under a minute; restored with `mknod -m 666 /dev/null c 1 3` (char device 1,3). No job depended on it.
- 22:44 Continuation test: 8 k78 runs classed "stalled" (route ends alive, budget exhausted), continued
  from their final state, ALL escape, at 0.49-0.85 x the balanced-route estimate of the remaining
  plateau b^-(K-2)/((K-2) s_r), b = (s_r |a_r|)^(1/K) (constant per seed across sigma). Stalled =
  ever-longer plateau, not stuck. A spread sample from `large` is running.
- 23:50 All sweeps done except first_k78 (38/52 jobs) and a continuation batch from `large`. Final
  analysis: Hessians recomputed with the fixed projection; edge check with collision-free thresholds;
  plateau spectra = truncated teacher spectra; K = 7 gap exponents 5.02 / 4.97 (binary8, chi = 6).
- 00:10 Final report written; waiting for the last first-escape jobs (mps8).
- 00:28 Continuation of 8 mps8 chi = 6 stalled runs (`large`): 2 escape at 1.4-1.8x the route estimate,
  1 not by 2x, 4 wall-limited, 1 integrator failure; corrected the claim that the estimate is an upper
  bound (it is only for a balanced-reducible route).
- 00:47 Stopped the last 10 first-escape jobs (mps8 chi = 6; t_1 ~ 1e13 at sigma = 0.01). Final
  first-escape exponents 4.998-4.999 (K = 7), 5.9995 (K = 8, 10 seeds). Figures regenerated. Done.
- 09:46 Machine restarted (replied ok). 09:54 Lead: fix round after review round 1; merged main, read
  `notes/reviews/s2s_core_round1.md` in full.
- 09:56 Correlation flow now stops on direction convergence; T recomputed for all 238 first-escape jobs
  (r1 = GF's first component in 100%; binary4 close T moved <= 1.7e-3); lo_d / lo_K / lo_chi rerun
  (unchanged to 3 digits); aligned runs rerun (one file per integrator). Started: zoom-resolved
  continuation of the 8 k78 runs, random continuation sample, 9-run tolerance check.
- 10:05 analyze_main refactored: fits on S2S-classified run sets; strict saddle on binary4 wrong-order
  plateaus confirmed (0.98 of the prediction); class table; mps8 gap 1 5.95 +- 0.05. Trap fits above
  1e-9: rates per seed constant to 1-3% (the 20-100x switch was a floor artifact). Tolerance check:
  9/9 keep class and order.
- 10:10 WIP commit 7761da5.
- 10:17 Figures c-g regenerated (d and f on the classified sets); slice trends; random continuation
  done (7/8 escape at 0.24-2.19x, 1 not by 3x); notes 2.2-2.7, 3, 4 (claims changed), README, report.
