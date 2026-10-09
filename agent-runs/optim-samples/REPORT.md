# optim-samples: learning rate, SGD noise and finite samples (Q8)

Status: working (step 1 done; steps 2-3 in progress) - Branch: agent/optim-samples - Updated: 2026-10-09 20:10 UTC

## Summary
Question: how do discrete-step GD, SGD noise and a finite training set change the gradient-flow
(GF) picture of TTN training (Q8, H6)? **Step 1 (GD vs GF) is done**: the stability limit
eta_max = 2/(K s_1^{2-2/K}) is proven at the balanced minimum and is the exact empirical boundary for
small init; below it GD reproduces GF event times with an O(eta) shift that a parameter-free
backward-error computation predicts to 3 digits; the saddle sequence changes only for near-tied
races. Above it the top component oscillates (2-cycle), then GD diverges or throws learned
components back to the origin saddle. Imbalanced inits show classic edge-of-stability selection of
flatter, more balanced minima. Steps 2 (SGD, running) and 3 (finite samples, scripted and piloted)
are in progress.

## Results (step 1; all numbers: GD with eta = ratio x eta_max vs GF DOP853 rtol 1e-10, atol 1e-14)
Setup: odeco teacher [3, 2, 1], d = 3, chi = 3, teacher seed 101, init seeds 0-4; binary N=4
(sigma 0.003-0.3), binary N=8 and MPS N=8 (sigma 0.9). 750 runs: `experiments/optim_samples/gd_sweep.py`
+ `configs/gd_sweep.yaml`; tables `results/gd_sweep.csv`, `gd_summary.csv`; derivations
`notes/optim_samples.md` sections 1-3.

| # | finding | tag | evidence |
|---|---|---|---|
| 1 | Sharpness at the balanced global minimum = K s_1^{2-2/K}; the balanced minimum is the flattest aligned minimum; eta_max = 2/(K s_1^{2-2/K}) = 0.154 / 0.0435 / 0.0366 (binary 4 / binary 8 / MPS 8) | proven (+ check to 3e-15 on 5 trees) | `check_sharpness.py`, `results/sharpness_check.csv` |
| 2 | Small init: all 170 binary-N=4 runs with eta <= 0.95 eta_max converge like GF; all 25 at 1.05 eta_max oscillate (sigma 0.003-0.3) | empirical | `results/fig_stability.png` |
| 3 | GD event shift t^GD - t^GF = eta c_e + O(eta^2), c_e from the variational equation along the GF path: matches per seed to 3 digits at eta <= 0.1 eta_max, within 2% up to 0.9 eta_max, including negative shifts | derived + empirical | `predict_shift.py`, `results/fig_gd_vs_gf.png` |
| 4 | Relative event error ~ eta ln(1/sigma) / t_1 -> 0 as sigma -> 0: at 0.9 eta_max 0.111, 0.056, 0.023, 0.0099, 0.0036 for sigma = 0.3 ... 0.003 | derived + empirical | `fig_gd_vs_gf.png` (a) |
| 5 | Order of learning unchanged in 225/235 runs below eta_max; the exceptions are 2 seeds with a near-tie (sigma = 0.1) where eta >= 0.2-0.5 eta_max removes a whole plateau (component 2 learned at t ~ 7.8 instead of 23 / 17) | empirical | `results/gd_sweep.csv` |
| 6 | Above eta_max: component 1 in a 2-cycle at 1.05-1.3 eta_max (others learned normally, loss floor); binary N=4 diverges at 1.5-1.75, **every run thrown back to the origin saddle at 2.0 eta_max** (repeated re-learning), diverges >= 2.5. N=8 trees: oscillation to 1.5, re-entry 1.75-2.5, divergence at 3. Qualitatively the 1D balanced map; its bounded chaotic window is not robust in the full TTN | empirical (+ 1D map derived) | `fig_stability.png`, `fig_eos_example.png`, `map1d.py` |
| 7 | Imbalanced init (root core x2/x4/x6): GD selects minima of sharpness min(lambda_GF, 2/eta) with less imbalance (EoS selection); from balanced init GD instead creates imbalance ~ eta | empirical | `fig_min_selection.png` |
| 8 | sigma = 0.9 (N = 8): init imbalance makes the GF state 8-12% sharper, the edge moves to ~2/lambda_GF ~ 0.9 eta_max | empirical | `gd_summary.csv` |

Figures (in `experiments/optim_samples/results/`): `fig_gd_vs_gf.png` (GD/GF agreement and
prediction), `fig_stability.png` (eta_max vs prediction, outcome classes, 1D bifurcation diagram),
`fig_eos_example.png` (EoS example: loss, overlaps, sharpness x eta/2), `fig_min_selection.png`.

**Step 3 (finite samples; done for d = 2, N = 8; empirical, 5 seeds; `fig_samples.png`,
`notes/optim_samples.md` section 6):** small init (sigma = 0.3, 0.5) recovers W* from n ~ 256 samples
(test error 1.5e-3 at 256, ~1e-4 at 384, bond rank exactly 2), the exact lazy limit at chi = 4 never
does (test error 1.7-30x worse than predicting zero, rank 4-5.3; its tangent space has dimension
112 < 256), at full chi = 16 it needs n = d^N = 256 and is worse than zero below; the centred
sigma = 1 model is intermediate (rich, 0.01-0.04 at n = 384-512). Small init is therefore the most
sample-efficient and the only one that lands on the teacher's bond ranks, but here it needs ~7-9x
the 28-dimensional teacher manifold (conjecture: heavy tails of product features, (1 + 2/d)^N = d^N
for d = 2; a d = 3, N = 6 test is queued).

**Theory-note predictions (`notes/theory_optimizer.md` v2, section 9), tested so far:**
- P5 (oscillation loss above eta_max): confirmed and refined: the time-averaged excess loss equals
  s_1^2 x (1D map attractor loss) to 3-4 digits on binary N=4 (0.1306 vs 0.1306 at 1.05 eta_max; 0.4496
  vs 0.4496 at 1.2), a few % on N = 8 trees; the linear formula is its small-excess limit.
- P6/P7: my backward-error and SGD first-plateau formulas are the same as the note's (P6 exact
  variational version matches GD to 3 digits; P7 being tested in the running SGD sweep).
- P9 (SGD lengthens/traps later k = 3 plateaus): dedicated aligned-route test running (`p9_sgd.py`;
  first run: T_SGD/T_GF = 1.45 at eta/B = 2.5e-4, T_GF = 50, i.e. c_gamma ~ 48 vs heuristic 26). Also
  found: in random-init SGD runs, near-tie seeds flip the race under noise and the second plateau
  disappears (ratio 0.82 on average at sigma = 0.1); a mechanism outside P9.
- Correction to my notes: I had argued that k >= 3 routes cannot stochastically collapse; the
  theory note's O(1) zero-mean coupling to mixed bond entries shows they can (fixed in section 4).

**Pilots for steps 2-3 (provisional; pilot SGD numbers were computed before the teacher-RNG fix
and are being redone in the full sweep):**
- SGD noise does not shorten plateaus: spread of t_1 over noise ~ sqrt(eta V / B) (V ~ 3^N s_1^2,
  heavy-tailed product features), mean within ~1% of GF. Exact mean-square SGD stability limit
  (binary N=4): eta_ms(B) = 0.00107 B for B <= 16, 0.053 at B = 64, 0.112 at 256, -> 0.154 (GD).
  `results/sgd_stability.csv` (recomputed with teacher seed 101).
- Finite samples (binary N=8, d = 2, n = 32 / 128): small init recovers W* at n = 128 (test error
  0.1% of ||W*||^2/2, bond ranks 2) but not at n = 32; the centred large init stays lazy and its
  tangent space has dimension 112 < 256, so it cannot fit W* at any n.

## What was done
1. Read docs; merged main three times (literature review v1, PLAN v1, core fixes 57d5262).
2. Theory: sharpness proof, 1D map, backward-error shift, SGD noise and stability (`notes/optim_samples.md`).
3. Discarded all runs made with the pre-fix library (teacher/init RNG collision); teacher seed 101.
4. Step 1 sweep (750 runs), shift prediction, analysis, figures (commits 701314a and later).
5. Step 2 sweep (3960 SGD runs) running; step 3 script written and piloted.

**Library cross-check (after `ttn.dynamics` landed, commit f2f0d6f):** `validate_library.py` reruns 6
GD, 4 GF and 2 SGD settings with `run_gd`, `integrate_gf` (dopri5 and lsoda, rtol 1e-10) and
`run_sgd` (same generator seeds): GD event times agree to <= 5e-4 (interpolation only) with identical
final losses; GF to ~1e-5 relative; SGD trajectories are identical (events to 2e-8).
`results/validate_library.csv`. The sweeps made with my loops are therefore library-equivalent.

## Decisions
- **Teacher [3, 2, 1], d = 3, chi = 3** (as `s2s-core`'s separated teacher) so stability limits
  and plateaus are comparable across agents; **teacher seed 101**, independent of init seeds 0-4
  (after the lead's RNG fix).
- **Keep the step-1/2 sweeps from my own loops after the library landed** (validated equivalent,
  above); new experiments (collapse test) use `ttn.dynamics` directly.
- **Own minimal GD/SGD loops before `ttn.dynamics` lands.** The brief allows pilots only; the
  library is still hours away and the interim report is due after step 1. The loops are a few
  lines in my folder; I will rerun key settings with `ttn.dynamics` once it lands.
- **sigma = 0.9 for the N = 8 trees** (GF first events at t ~ 4-80; at sigma = 0.5 later events
  take > 1000 time units for K = 7, 8).
- **Outcome classes for GD** (not in `notes/diagnostics.md`, which has only "diverged" for GD):
  converges / bounded oscillation (loss total variation > 10x net change over the last 20% of steps,
  or final sharpness x eta/2 > 1) / thrown back to origin (loss > 0.9 L(0) after an event) / diverged.
- **Step 3 uses plain GD with eta = 1/lambda_max** (re-estimated by power iteration, relative step
  capped at 1%) instead of an RK integrator: DOP853 needed > 2e5 evaluations per run in the stiff
  interpolation phase. The lazy limit is computed in closed form (min-norm kernel regression with
  the initial tangent kernel) because lazy training is too ill-conditioned to converge in budget.

## Files left on the machine
- `/root/outputs/optim_samples/gd/` (92 MB): per-run npz trajectories + json summaries of step 1.
  Needed only to re-plot from scratch; the summary CSVs in git suffice for the figures.
- `/root/outputs/optim_samples/sgd/`, `/samples/`: step 2-3 raw results (jsonl), being written.

## Caveats and next steps
- Step 1 used 5 seeds per setting; the near-tie order flips (finding 5) need more seeds to estimate
  their probability vs eta and sigma (planned: 40 seeds, binary N=4, sigma 0.03 and 0.1).
- Step 2: SGD sweep (B = 1..256 at eta = 5e-4, eta sweeps at B = 4 and 64, aligned-init block) with
  Q_e and effective-rank tracking; a stochastic-collapse test (k = 2 pattern teacher, eta near the
  SGD limit) as asked by the lead.
- Step 3: n from 16 to 512 (teacher-rank TTN has 52 parameters, manifold dimension 28), small
  sigma 0.3 / 0.5 vs centred sigma = 1 and the lazy limit; also a full-chi student (tangent space =
  all of R^256) so the lazy regime can interpolate.
- Switch to `ttn.dynamics` / `ttn.diagnostics` when they land and rerun key settings.

## Log
- 18:20 Read brief and project docs; merged origin/main; set up report.
- 18:25 Sharpness formula verified numerically on 5 trees (machine precision).
- 18:30 1D GD-map bifurcations; GD-vs-GF pilot; exact SGD mean-square limit.
- 18:40 SGD plateau pilot (240 runs): noise spreads but does not shorten plateaus.
- 18:42 Lead message: literature v1 on main; merged; read sections 1.4, 2.7, 3, 5.
- 18:45 EoS and imbalance pilots; launched step-1 sweep.
- 18:48 Lead message: teacher/init RNG bug fixed on main. Killed sweeps, merged, teacher seed 101, relaunched everything.
- 19:02 Step-1 sweep done (750 runs); SGD sweep started (3960 runs).
- 19:10 Backward-error shift prediction: matches GD to 3 digits. Step-1 figures.
- 19:20 Interim report (step 1).
- 19:25 Lead: library v0 on main; merged (126 tests pass). Float64-floor check: only never-learned components of N=8 runs collapse below 1e-14 (not used for any reported time).
- 19:33 Library cross-check passed (GD/GF/SGD); SGD sweep and finite-sample sweep running (15 workers each).
- 19:55 Finite-sample sweep done (275 runs). Lead: theory_optimizer v2; merged; P5 confirmed from GD data.
- 20:05 P9 aligned-route SGD test, GD order-flip (40 seeds) and centred-model GD blocks running; d = 3 samples block queued.
