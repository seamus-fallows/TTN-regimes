# s2s-core: saddle-to-saddle phenomenology and escape-time scaling in TTNs (Q1, Q2)

Status: working (interim report 1) - Branch: agent/s2s-core - Updated: 2026-10-09 20:45 UTC

## Summary
Asked: does gradient flow (GF) on a TTN from small init show saddle-to-saddle (S2S) dynamics, which
saddles it visits, and how plateau durations scale with sigma, tree (K cores) and bond dimension chi.
So far: (1) the Saxe-type exact solution is reproduced on all five trees (overlaps to 3e-9).
(2) The first escape is predicted per seed with no free parameter by the blow-up time of the correlation
flow theta' = grad <W, W*> (derived here, same as theory_escape.md E1/2.5); in pilots the prediction
holds to 4-5 digits and the Kumar-Haupt rho_0 law is a lower bound off by 3x to 3e7x.
(3) Random init on odeco teachers gives clean S2S at K = 3, 4: plateaus at the partial-fit levels, every
bond spectrum gains one value per event on every edge at once, gap exponents exactly K-2 per seed.
(4) Main caveat, now an explicit question: at chi = R = 3 (and at K = 7, 8 for small sigma), dormant
components are often driven below the float64 floor (amplitude 1e-15 .. 1e-41) after one or two escapes,
so GF effectively stalls at a partial fit (24/67 mps4 chi = 3 runs so far). chi = 2R removes this at
K = 4 (60/60 runs complete).

## Results
Interim, partial. Every number below is gradient flow (LSODA with exact Hessian Jacobian, rtol 1e-8,
atol 1e-11 sigma unless stated), whitened inputs, d = 4, odeco teacher (teacher seed 1000), fan-in init.

**Step 1: aligned init vs exact solution** (`experiments/s2s_core/results/aligned.csv`,
`fig_a_aligned.png`; `aligned.py`, config `configs/aligned.yaml`; DP5 rtol 1e-10, atol 1e-14):

| tree | K | a0 | max abs dev. a_r(t) | event times (same grid) | Q drift |
|---|---|---|---|---|---|
| binary4 | 3 | 0.01 | 2.0e-10 | 2e-12 rel. | 4e-15 |
| mps4 | 4 | 0.07 | 3.1e-10 | 1e-12 | 5e-15 |
| binary8 | 7 | 0.29 | 2.2e-9 | 1e-11 | 1e-14 |
| ht4 | 7 | 0.29 | 2.5e-9 | 1e-11 | 2e-14 |
| mps8 | 8 | 0.34 | 2.8e-9 | 9e-12 | 2e-14 |

(LSODA at the same tolerances: deviations 4e-9 .. 7e-7.) Known result (Saxe et al. 2014 eq. 15 for
DLNs); explicit for TTNs here: "derived; known for DLNs".

**First escape (pilots, scipy, on the fixed library for the K sweep):** t_1 sigma^{K-2} -> T(theta_hat) per
seed (binary4 seed 0: 0.76962 at sigma 3e-3 vs T = 0.77021; offset -0.198 vs c_3(3) = -0.196).
K sweep, chi = 3, 30 seeds, 17 trees with K = 3..15: geometric mean T / (RMS-overlap heuristic P3) =
0.94-2.1 while T spans a factor 1500; T / (Kumar-Haupt) = 3 .. 3e7. Details: `notes/s2s_core.md` 2.1.
The systematic GF test across sigma (`first_escape.py`, `configs/first.yaml`) is queued.

**Step 2, first pass (sweep `configs/main.yaml`, ~310 of 1600 runs done; `results/main_summary.csv`
to come):**
- binary4 (K = 3), chi 3 and 6, sigma 0.003-0.02, 10 seeds per cell: 160/160 runs learn all 3
  components; S2S (every rho > 3) in 80/80 separated-teacher runs; close teacher [1, 0.85, 0.7]:
  20/20 at sigma = 0.003, 15/20 at sigma = 0.02 (overlapping transitions). Order greedy (largest
  first) in 157/160.
- mps4 (K = 4), sigma 0.003-0.02 (127 runs so far): chi = 6: 60/60 complete and S2S, order random
  (greedy 9/30 close teacher, 21/30 separated). chi = 3: 43/67 complete; the other 24 stall at a
  partial fit with a dormant amplitude below 1e-14 ("floor" class).
- Plateau losses equal the partial-fit levels L_S = 1/2 sum_{r not learned} s_r^2 to 2e-5 relative
  (binary4) and 2e-7 (mps4) on complete separated-teacher runs at sigma <= 0.01.
- Representative run (`fig_b_run_mps4_chi3_w3-2-1_fanin_s0.01_seed0.png`): three plateaus at the
  partial-fit levels; on every edge the k-th singular value rises at the k-th event, on all edges at
  the same time; effective bond rank 0 -> 1 -> 2 -> 3; min separation ratio 6e4.
- Gap exponents (pilot, scipy BDF, 5 seeds, sigma 0.1-0.005): per-seed slopes -1.00 (binary4) and
  -2.00 (mps4) for both gaps, i.e. K - 2, as derived for odeco teachers (every route has k = K).

**Dormant-component suppression (new question from the lead):** two mechanisms seen so far, each in
single runs (to be quantified over the sweeps):
1. Plateau decay (MPS4, chi 3, sigma 0.003, seed 6): after the second escape, the dormant component's
   slices on the two MPS end cores decay exponentially at rate ~3e-6 ~ sigma^2/3 (3e-4 -> 1e-12 between
   t = 3e5 and 7e6) while its middle-core slices stay ~1e-3; its amplitude falls to 1e-33. This is the
   conservation-law trap R12 of `notes/theory_saddles.md`.
2. Collapse during another component's escape (binary8, chi 3, sigma 0.1, seed 0): a_2 drops from
   ~2e-12 to ~1e-17 while component 3 escapes, with fan-in AND with gauge-balanced init.

## What was done
1. Read brief, notes, library; report skeleton (commit 7e4dbd4 and following).
2. Theory: correlation-flow first-escape prediction, bounds, decoupling, race model, odeco later
   escapes, saddle types (`notes/s2s_core.md` section 1).
3. Pilots (throwaway scipy): first-escape prediction, stiffness of later plateaus, chi and K sweeps.
4. Experiment code on library v0 (`experiments/s2s_core/`: s2s.py, run.py, aligned.py,
   first_escape.py, analyze_main.py, plot_aligned.py, plot_run.py) (137cc70, 55c4afb).
5. Step 1 done (137cc70). Step 2 sweep running.

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
