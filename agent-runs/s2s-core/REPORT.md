# s2s-core: saddle-to-saddle phenomenology and escape-time scaling in TTNs (Q1, Q2)

Status: done - Branch: agent/s2s-core - Updated: 2026-10-10 00:50 UTC

## Summary
Asked: does gradient flow (GF) on a TTN from small init show saddle-to-saddle (S2S) dynamics, which
saddles it visits, and how plateau durations scale with sigma, the tree (K cores) and the bond
dimension chi, against theory; added by the lead: are dormant components suppressed during earlier
escapes (stuck vs ever-longer plateaus)? Answer (odeco teachers, d = 4, binary4/8, mps4/8, ht4):
yes, S2S on every tree, with plateaus at the bond-rank-constrained partial fits (truncated teacher
spectra on every bond), one rank step per event on all edges at once, higher-order saddles, and every
gap ~ sigma^-(K-2) per seed (measured 1.00, 2.00, 5.0 for K = 3, 4, 7). The first escape is predicted
per seed with no free parameter by the correlation-flow blow-up time (theory_escape E1) to <= 2e-3; the
Kumar-Haupt / rho_0 law is a lower bound off by 3-2700x (K = 3-8) and fails qualitatively in d, chi and
K. The main limitation of S2S at large K is the conservation-law trap of fan-in init: at chi = R it
kills dormant components for a sigma-independent fraction of seeds (0 at K = 3, 40% at K = 4, 30-80%
at K = 7, 8); gauge-balanced init removes it (0/210), chi = 2R removes it at K = 3, 4 and on binary8,
and the remaining stalls are ever-longer plateaus (continued runs escape at 0.5-1.8x the route estimate in 10 of 11 cases).

## Results
Gradient flow throughout: LSODA with exact Hessian Jacobian, rtol 1e-8 / atol 1e-11 sigma (main,
first, balanced), rtol 1e-6 / atol 1e-9 sigma (large, k78; event times agree with rtol 1e-8 to 7e-7);
DP5 rtol 1e-10 for the aligned runs. Whitened inputs, d = 4, odeco teachers (seed 1000) separated
[3, 2, 1] and close [1, 0.85, 0.7], fan-in init unless stated, 10 seeds per cell. Event times and
widths are resolved by re-integrating each event window on linear grids (>= 40 samples per rise).
All tables: `experiments/s2s_core/results/`; derivations and status tags: `notes/s2s_core.md`.

**1. Known result** (`aligned.csv`, `fig_a_aligned.png`): aligned balanced init follows the exact
Saxe modes; max |a_GF - a_theory| = 2e-10 (binary4), 3e-10 (mps4), 2e-9 (binary8), 3e-9 (ht4, mps8).
Derived; known for DLNs (Saxe 2014 eq. 15), explicit for TTNs here.

**2. Phenomenology** (`main_runs.csv`, `fig_b_run_*.png`): among complete runs
- plateau loss = partial-fit level L_S to <= 2e-2 relative (binary4 at sigma = 0.1; <= 2e-3 for mps4,
  <= 2e-5 at K = 7; -> 6e-7 at sigma = 0.003), and every
  edge's top |S| singular values = the learned weights (truncated teacher spectrum), next value <= 0.07
  s_min: each plateau is the bond-rank-constrained critical point W_S;
- each event raises the rank by one on every edge at the same time (517/551 runs; spread <= 0.05 event
  widths except HT4; exceptions are overlapping events);
- S2S (every rho > 3): 100% at K = 4, 7 (mps4 266/266, binary8 83/83, ht4 74/74); binary4 separated
  158/160, close teacher 5/20 at sigma = 0.1 rising to 20/20 at sigma = 0.003;
- order: binary4 always largest first (matrix-like); on mps, binary8, ht4 the first component is random
  (P(largest) 0.4-0.8) but the same at every sigma for each seed (all 10 seeds in every cell).

**3. Escape-time exponents** (per-seed fits, mean +- SE over seeds; `main_exponents.csv`,
`first_summary.csv`, figures c, d; predicted K - 2 for t_1 and every gap):

| tree | K | chi | sigma range | t_1 (first sweep) | t_2 - t_1 | t_3 - t_2 | seeds |
|---|---|---|---|---|---|---|---|
| binary4 | 3 | 3 | 0.1-0.003 | 0.9996 +- 0.0001 | 1.003 +- 0.001 | 1.004 +- 0.003 | 10 |
| binary4 | 3 | 6 | 0.1-0.003 | 0.9996 +- 0.0001 | 1.008 +- 0.002 | 1.012 +- 0.002 | 10 |
| mps4 | 4 | 3 | 0.1-0.003 | 2.0000 +- 0.0000 | 1.999 +- 0.000 | 1.990 +- 0.005 | 6 |
| mps4 | 4 | 6 | 0.1-0.003 | 2.0000 +- 0.0000 | 2.001 +- 0.001 | 1.998 +- 0.001 | 10 |
| binary8 | 7 | 6 | 0.3-0.07 | 4.9990 +- 0.0002 | 5.020 +- 0.009 | 4.969 +- 0.022 | 10 |
| ht4 | 7 | 6 | 0.3-0.07 | 4.9987 +- 0.0007 | 5.11 +- 0.09 | 4.75 +- 0.14 | 9 |
| mps8 | 8 | 3 | - | 5.9995 +- 0.0002 | not reached | not reached | 10 |

(separated teacher; t_1 column from the first-escape sweep, sigma down to 0.002 (K = 3, 4) and 0.01
(K = 7, 8), with the predicted offset c_K removed. Close teacher: within 0.04 of K - 2 except binary4 gap
2 (0.92 +- 0.11, 1.12 +- 0.05). mps8 completes no run within budget, see 5.)

**4. First-escape prefactor** (`first_*.csv`, `lo_*.csv`, figures c, e, e2):
- per seed, t_1 / (sigma^-(K-2) T(theta_hat) + c_K) - 1 at the smallest sigma: <= 2e-3 (binary4),
  6e-5 (mps4), <= 2e-7 (binary8, ht4, mps8 at sigma = 0.01); first component = the correlation flow's in
  100% of runs except binary4 close teacher (80-90%, matrix-like wrong order). No free parameter.
- Kumar-Haupt (rho_0) is a lower bound: median T / t_KH = 3.1-4.8 (binary4), 7.9-13.7 (mps4), 95-210
  (ht4), 184-2650 (binary8), 1460 (mps8); up to 3e7 at K = 15 (leading order); all inside the ranges
  theory_escape 9.2 predicts.
- d at full bond (200 inits): binary4 median 1.544, 1.569, 1.563 for d = 2, 3, 4 = pi/2, flat, as
  theory_escape predicts, while t_KH falls 1.9x; mps4 1.39, 1.41, 1.50 (pred. 1.30, 1.35, 1.46).
- chi (100 inits): slope of the median over chi = 4-16: mps4 -0.26 +- 0.04 (pred. -0.20 +- 0.05), binary4
  -0.34 +- 0.03 (pred. -0.36 .. -0.38); t_KH -0.47; K = 7, 8 trees -0.69 .. -0.76 vs -1.5 .. -2.5 for t_KH;
  mps8 chi 2 -> 8: 4.0x faster (pred. 3.7x; t_KH 35x).
- K (50 inits, K = 3-15): the RMS-overlap heuristic P3 = d^{N(K-2)/(2K)}/((K-2) s_1) (no fit) matches the
  median within 0.87-2.1x while the prefactor spans 0.4-814.
- Gauge-balancing the same init changes T per seed by 0.88-1.92x (median 0.99-1.18; usually slower).

**5. Saddle type** (`hess_*.csv`, figure f; Hessians recomputed with the fixed gauge projection, raw
and projected): lambda_min -> 0 like sigma^(K-2) per seed: slopes 1.06 / 1.02 (binary4), 2.00 / 1.99
(mps4), 5.10 / 5.07 (binary8), 5.14 / 4.88 (ht4) at the first / second plateau (raw: same to 0.01).
Higher-order saddles, as predicted for odeco teachers. Magnitude vs the route formula
-(K-1) s_r (s_r |a_r|)^((K-2)/K) (theory_saddles R10, measured dormant overlap): 0.8-1.15 at K = 3, 4,
1.4-2.9 at K = 7 (route not balanced at these sigma).

**6. Dormant-component suppression** (`dormant_*.csv`, `continued_*.csv`, figure g):
- Escapes barely shrink dormant components: median factor x1.5-2 (K = 3, growth), x0.7 (K = 4),
  x0.4-0.8 (K = 7, 8); > 100x drops in 0% (chi = 2R) to 6% (K >= 7, chi = R) of cases.
- Plateaus kill them through the conservation-law trap (theory_saddles R12): the route end cores'
  slices decay exponentially at a rate ~ sigma^2 per seed (constant within 1.8x over sigma 0.003-0.1
  for 5/7 trapped mps4 seeds), both ends on the light side of the conserved Q (checked on 2 runs).
  Run classes (separated teacher; trapped = dead route end, slice < 1e-8 sigma):

| tree | K | chi | init | complete | trapped | stalled + unresolved | n |
|---|---|---|---|---|---|---|---|
| binary4 | 3 | 3, 6 | fan-in / balanced | 1.00 / 1.00 | 0 / 0 | 0 / 0 | 160 / 60 |
| mps4 | 4 | 3 | fan-in / balanced | 0.60 / 1.00 | 0.40 / 0 | 0 / 0 | 80 / 30 |
| mps4 | 4 | 6 | fan-in / balanced | 1.00 / 1.00 | 0 / 0 | 0 / 0 | 80 / 30 |
| binary8 | 7 | 3 | fan-in / balanced | 0.07 / 0.10 | 0.30 / 0 | 0.63 / 0.90 | 30 / 30 |
| ht4 | 7 | 3 | fan-in / balanced | 0.10 / 0.57 | 0.80 / 0 | 0.10 / 0.43 | 30 / 30 |
| mps8 | 8 | 3 | fan-in / balanced | 0 / 0 | 0.47 / 0 | 0.53 / 1.00 | 30 / 30 |
| binary8 | 7 | 6 | fan-in | 1.00 | 0 | 0 | 50 |
| ht4 | 7 | 6 | fan-in | 0.90 | 0.10 | 0 | 50 |
| mps8 | 8 | 6 | fan-in | 0 | 0.14 | 0.86 | 50 |

  (K = 7, 8 at chi = 3: sigma 0.15-0.3, sweep k78; chi = 6: sigma 0.07-0.3, sweep large.) At K = 4 the
  same seeds trap at every sigma (trap time ~ sigma^-2, escape ~ sigma^-2). Gauge-balanced init
  removes the trap (0/210).
- "Stalled" runs (all route ends alive) are mostly not stuck: continued from their final state, 10 of
  the 11 runs integrated beyond 2x the balanced-route estimate b^-(K-2)/((K-2) s_r), b = (s_r |a_r|)^(1/K)
  escape, at 0.49-1.81x of it (8/8 k78 runs at 0.49-0.85x, same ratio per seed across sigma; 2 of 3 mps8
  chi = 6 runs at 1.4-1.8x, 1 not by 2x; 4 more mps8 runs could not be integrated that far). For all
  stalled runs the estimate is a median 10-60x (k78) and ~400x (mps8, chi = 6) the elapsed time. So with live routes, S2S at large K is ever-longer plateaus (exponent K - 2, prefactor inflated
  by the dormant suppression S as S^-(K-2)/K), and stuck only where fan-in imbalance kills a route.

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

## Files left on the machine
All raw outputs are in `/root/outputs/s2s_core/` (not in git; ~220 MB in total), one `.npz` (log-grid
trajectory, refined event windows with spectra, parameters at plateau midpoints, final state) and one
`.json` summary per run: `main/` (1600 runs, 132 MB), `large/` (150, 36 MB), `k78/` (180, 33 MB),
`balanced/` (120, 12 MB), `first/` (238 first-escape jobs, 1 MB), `aligned/` (1 MB). Needed to recompute
the analysis tables (`analyze_*.py`, `recompute_hessians.py`, `continue_stalled.py`); everything is
regenerated by the commands in `experiments/s2s_core/README.md` (~6 h on 60 CPUs).
`/root/outputs/s2s_core/first_test/` is an empty test directory.

## Caveats and next steps
- **K = 7, 8 later gaps** span only sigma = 0.07-0.3 (0.6 decade) and come from chi = 6 (binary8, ht4);
  at chi = 3 too few runs complete; mps8 completes no run within budget (all stall or trap), so its
  later-gap exponent is not measured. Smaller sigma is out of reach: the gaps grow like sigma^-5..-6 and
  LSODA advances slowly on these plateaus.
- **First-escape sweep at K = 7, 8** was cut for compute (separated teacher, fan-in only); mps8 at
  chi = 6 is missing (jobs stopped at the end: first escapes at sigma = 0.01 are at t ~ 1e13).
- **Trap statistics** are for one teacher basis (seed 1000) and the fan-in init of the model; the trap
  fraction depends on the init's Q_e, i.e. on the init scheme.
- **Stalled vs stuck**: the continuation test covers 8 k78 runs (shortest predicted escapes) and 8
  mps8 chi = 6 runs spread over the distribution, of which only 3 could be integrated past 2x their
  route estimate (2 escaped at 1.4-1.8x, 1 had not at 2x). Whether every live-route stall eventually
  escapes is therefore shown for k78-type runs, and only partly for mps8.
- R10 magnitudes at K = 7 deviate 1.4-2.9x (unbalanced routes at moderate sigma); the slopes are clean.
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
