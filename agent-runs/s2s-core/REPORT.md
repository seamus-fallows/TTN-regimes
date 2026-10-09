# s2s-core: saddle-to-saddle phenomenology and escape-time scaling in TTNs (Q1, Q2)

Status: working (interim report 2) - Branch: agent/s2s-core - Updated: 2026-10-09 21:50 UTC

## Summary
Asked: does gradient flow (GF) on a TTN from small init show saddle-to-saddle (S2S) dynamics, which
saddles it visits, and how plateau durations scale with sigma, tree (K cores) and bond dimension chi;
added by the lead: are dormant components suppressed during earlier escapes (stuck vs ever-longer
plateaus)? So far, odeco teachers, d = 4: (1) the Saxe-type exact solution is reproduced on all five
trees. (2) The first escape is predicted per seed with no free parameter by the correlation-flow
blow-up time T(theta_hat) (= theory_escape E1): binary4 and mps4 to <5e-5 relative at small sigma;
Kumar-Haupt is a lower bound off by 3-10x already at K = 3, 4. (3) At K = 3, 4 the dynamics is clean
S2S: plateaus at the partial-fit levels, ranks rise by one on every edge at once, per-seed exponents of
t_1 and both later gaps equal K - 2 (1.00 / 2.00), plateaus are higher-order saddles (lambda_min ~
sigma^{K-2} per seed, magnitude within 0.8-1.15 of the route formula). (4) Dormant components get
trapped (conservation-law trap, theory_saddles R12): never at K = 3; for the same 4/10 seeds at every
sigma at K = 4, chi = 3, and never at chi = 6; for 77% of HT4 (K = 7, chi = 3) runs. At K = 7, 8 most
other runs did not finish within budget; two dedicated sweeps (k78, large) are running to settle them.

## Results
Gradient flow, LSODA with exact Hessian Jacobian, rtol 1e-8, atol 1e-11 sigma (unless stated),
whitened inputs, d = 4, odeco teachers (seed 1000) "separated" [3, 2, 1] and "close" [1, 0.85, 0.7],
fan-in init, 10 seeds per cell. Tables: `experiments/s2s_core/results/`. Status tags in
`notes/s2s_core.md`.

**Step 1, aligned init vs exact solution** (`results/aligned.csv`, `fig_a_aligned.png`; DP5 rtol
1e-10): max |a_GF - a_theory| = 2e-10 (binary4), 3e-10 (mps4), 2e-9 (binary8), 3e-9 (ht4, mps8);
event times equal to 1e-11 relative on the same grid. Known for DLNs (Saxe 2014 eq. 15); explicit for
TTNs here.

**Step 3a, first escape vs the per-seed prediction** (`first_escape.py`, partial: binary4, mps4;
`analyze_first.py`): per-seed exponent of t_1 vs sigma (sigma = 0.1 .. 0.002, 8 values):
binary4 1.0045 +- 0.0002 raw, 0.9996 +- 0.0001 after subtracting the predicted O(1) offset c_K; mps4
2.0002 / 2.0000. max |t_1 / (sigma^-(K-2) T + c_K) - 1| at sigma = 0.002: < 5e-5 (separated teacher),
1.7e-3 (close). Median T / t_KH = 3.1-4.8 (binary4), 9.1 (mps4) (theory_escape predicts 2.5-5.2,
5.2-18). Fan-in and gauge-balanced inits give the same exponents.

**Steps 2-4, full runs** (`configs/main.yaml`, 1600 runs; `results/main_*.csv`, figures b, d, f, g):

| tree | K | chi | per-seed exponent t_1 | gap 1 | gap 2 | lambda_min slope (pl. 1 / 2) | lambda_min / route formula |
|---|---|---|---|---|---|---|---|
| binary4 | 3 | 3 | 1.005 +- 0.000 | 1.003 +- 0.001 | 1.004 +- 0.003 | 1.06 / 1.02 | 1.15 / 1.07 |
| binary4 | 3 | 6 | 1.006 +- 0.001 | 1.008 +- 0.002 | 1.012 +- 0.002 | 1.08 / 1.04 | 1.01 / 1.01 |
| mps4 | 4 | 3 | 2.000 +- 0.000 | 1.999 +- 0.000 | 1.990 +- 0.005 | 2.01 / 1.99 | 1.09 / 0.88 |
| mps4 | 4 | 6 | 2.000 +- 0.000 | 2.001 +- 0.001 | 1.998 +- 0.001 | 2.01 / 2.00 | 1.07 / 1.05 |

(separated teacher, sigma = 0.1 .. 0.003, mean +- SE over the 6-10 seeds that complete at >= 4 sigmas in
their modal order; predicted K - 2 for all; route formula -(K-1) s_r (s_r |a_r|)^((K-2)/K) from
theory_saddles R10 with the measured dormant overlap.) Close teacher: same within 0.04 except gap 2
on binary4 (0.92 +- 0.11, 1.12 +- 0.05; overlapping events at sigma >= 0.03). Single complete seeds
at K = 7: binary8 4.95 / 4.86, ht4 5.10 / 4.27 (predicted 5).
- Plateau losses = partial-fit levels L_S to 3e-6 (binary8, ht4), 2e-3 (mps4) and 2e-2 (binary4 at
  sigma = 0.1) relative.
- S2S (every rho > 3) among complete runs: 158/160 binary4 separated; close teacher 5/20 at
  sigma = 0.1 rising to 20/20 at sigma = 0.003; mps4 266/266.
- Edges: in 213/236 runs with event-window spectra every event raises the rank by exactly one on every
  edge, and all edges step at the same sample (spread < width/40); the exceptions are overlapping
  events (two ranks in one window, binary4 close teacher, sigma >= 0.05).
- Order: binary4 always learns the largest first (matrix-like); on MPS and binary8 the first
  component is random but the same for every sigma of a seed (10/10 seeds in every cell), as the
  race model predicts.
- Gauge-projected Hessian values may be biased by up to ~20% (library bug notified by the lead; fix
  pending); the slopes are unaffected; values will be recomputed (`recompute_hessians.py`).

**Dormant suppression** (`results/dormant_classes.csv`, `fig_g_dormant.png`; class rules in
`analyze_main.py`): fraction of runs complete / trapped (a never-learned component has a dead route
end, slice < 1e-10 sigma) / stalled / unresolved, separated teacher, pooled over sigma:

| tree | K | chi | n | complete | trapped | stalled | unresolved |
|---|---|---|---|---|---|---|---|
| binary4 | 3 | 3, 6 | 160 | 1.00 | 0 | 0 | 0 |
| mps4 | 4 | 3 | 80 | 0.60 | 0.40 | 0 | 0 |
| mps4 | 4 | 6 | 80 | 1.00 | 0 | 0 | 0 |
| ht4 | 7 | 3 | 60 | 0.10 | 0.77 | 0.07 | 0.07 |
| binary8 | 7 | 3 | 79 | 0.05 | 0.24 | 0.19 | 0.52 |
| mps8 | 8 | 3 | 70 | 0 | 0.24 | 0.19 | 0.57 |

mps4 chi = 3: the same 4 seeds (separated) trap at every sigma from 0.003 to 0.1; their dead route ends
decay at a rate proportional to sigma^2 per seed (rate / sigma^2 constant within a factor ~2 for 5 of
7 trapped seed/teacher pairs over sigma = 0.003-0.1; two pairs jump by 30-100x between sigma = 0.01 and
0.02); the sign of the conserved Q_nu nu makes every route end the light side (checked on 2 runs). The
"unresolved" K = 7, 8 runs are being re-run without the amplitude stop and with longer budgets (k78:
chi = 3, fan-in vs gauge-balanced; large: chi = 6).

## Decisions
- **d = 4** at every site (odeco needs d >= 3; d = 4 leaves one physical direction outside the teacher).
- **Teacher seed 1000**, init seeds 0-9 (independent streams after the library fix).
- **Integrator: LSODA with the exact Hessian** for every random-init run (later plateaus are stiff;
  DP5 would need ~sigma^-(K-2) steps); DP5 as the reference for the aligned runs. Tolerances rtol 1e-8,
  atol 1e-11 sigma: a K = 7 run at rtol 1e-9 / atol 1e-12 sigma gave the same event times to 1e-8.
- **Event = final rise of a_r** (last upward crossing of 1/2; first crossings and relearning are
  recorded too), refined by re-integrating windows on linear grids until the 0.1-0.9 rise holds >= 40
  samples (`s2s.zoom_rise`). No duration is read off the log grid.
- **Run budget**: t_max = 1e3 x the RMS-overlap escape time of the weakest component, wall limit
  400-600 s, and a stop when every dormant amplitude has stayed below 1e-15 for a whole chunk (the
  rest of such a run is roundoff-driven). These end stalled runs; they are reported as a class, not
  dropped.
- **sigma grids**: K = 3, 4: 0.1 .. 0.003 (1.5 decades); K = 7, 8: 0.4 .. 0.03 for full runs (smaller
  sigma is floor-limited after the first escape), 0.4 .. 0.01 for the first escape.
- **Own sweep driver** (`run.py`) until `ttn.runner` lands; resumable, one process per run.

## Files left on the machine
- `/root/outputs/s2s_core/main/` (per-run .npz + .json; ~100 kB per run), `/root/outputs/s2s_core/aligned/`.
  Regenerate with the commands in `experiments/s2s_core/README.md`.

## Caveats and next steps
- Pilot numbers marked [old lib] in `notes/s2s_core.md` predate the RNG fix; being redone.
- Next: finish sweep A; sweep B (first escape vs sigma with per-seed prediction, fan-in vs balanced);
  LO sweeps for the chi- and d-dependence predictions of theory_escape.md section 9; dormant
  suppression statistics (vs sigma, K, chi, init); Hessian spectra at plateau midpoints vs sigma
  (classify by the sigma-scaling of lambda_min, as theory_saddles v2 recommends).

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
