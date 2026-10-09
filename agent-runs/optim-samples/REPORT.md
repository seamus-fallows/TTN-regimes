# optim-samples: learning rate, SGD noise and finite samples (Q8)

Status: done - Branch: agent/optim-samples - Updated: 2026-10-09 21:24 UTC

## Summary
Asked: how discrete-step GD, SGD noise and a finite training set change the gradient-flow (GF)
picture of TTN training regimes (Q8, H6), where the learning rate sits on the regime map, and what
saddle-to-saddle (S2S) implies for generalization. Found: (1) the GD stability limit at small init is
exactly eta_max = 2/(K s_1^{2-2/K}) (proven sharpness, sharp empirical boundary); below it GD reproduces
the GF saddles and event times up to an O(eta) shift that a parameter-free backward-error computation
predicts to 3 digits, and only near-tied races can flip; above it the top component oscillates as in a
1D map (loss floor exact to 3-4 digits), then runs diverge or are thrown back to the origin saddle;
across the lazy transition the limit falls as 2/lambda_GF(sigma) ~ sigma^{-2(K-1)}. (2) SGD noise never
shortens plateaus: the first plateau keeps its GF law with a spread ~ (eta/B)^{1/2}; later degenerate
plateaus are lengthened and trapped as the theory note's P9 predicts (only eta/B matters); SGD on
product features is ~150x less stable per sample than GD. (3) With finite data, small init (S2S)
recovers the teacher with its bond ranks from n ~ 250 samples, while the lazy limit never does at
finite bond dimension and needs n = d^N at full bond dimension; generalization degrades monotonically
with the init scale. Details, derivations and all numbers: `notes/optim_samples.md`.

## Results
Setups (unless stated): whitened Gaussian inputs, odeco teacher [3, 2, 1], d = chi = 3, teacher seed 101,
fan-in init seeds 0-4; trees binary N=4 (K=3), binary N=8 (K=7), MPS N=8 (K=8). Status tags as in the
project. Every figure is produced by a script in `experiments/optim_samples/` from CSVs in
`experiments/optim_samples/results/` (raw data in `/root/outputs/optim_samples/`).

| # | finding | tag | evidence |
|---|---|---|---|
| 1 | Sharpness at the balanced global minimum = K s_1^{2-2/K}; balanced = flattest aligned minimum; eta_max = 0.154 / 0.0435 / 0.0366 (binary 4 / binary 8 / MPS 8) | proven; checked to 3e-15 on 5 trees x 2 chi | `check_sharpness.py`, `sharpness_check.csv` |
| 2 | Small init, binary N=4, sigma 0.003-0.3, 5 seeds: all 170 runs at <= 0.95 eta_max converge like GF, all 25 at 1.05 eta_max oscillate | empirical | `fig_stability.png`, `gd_summary.csv` |
| 3 | GD event shift = eta c_e + O(eta^2), c_e from the variational equation along the GF path: matches per seed to 3 digits at eta <= 0.1 eta_max (3 trees, all events, incl. negative shifts), within 2% up to 0.9 eta_max | derived + empirical | `predict_shift.py`, `fig_gd_vs_gf.png` |
| 4 | Relative event-time error at 0.9 eta_max: 0.111, 0.056, 0.023, 0.0099, 0.0036 for sigma = 0.3 ... 0.003 (binary N=4): vanishes as sigma -> 0 | derived + empirical | `fig_gd_vs_gf.png` a |
| 5 | Learning order below eta_max: unchanged at 0.1 eta_max (0/45 seeds per sigma = 0.03, 0.1, 0.3); flipped in 1-4/45 at 0.3-0.9 eta_max, 6 of 7 flipped seeds are near-ties (GF's eventual second component peaks at 0.23-0.49 during the first event); a flip can remove a whole plateau | empirical | `analyze_order.py`, `fig_order_flips.png` |
| 6 | Above eta_max: 2-cycle of the top component at 1.05-1.3 eta_max (others learned at GF times); excess loss = s_1^2 x (1D balanced-map attractor loss) to 3-4 digits (0.1306 vs 0.1306 at 1.05; 0.4496 vs 0.4496 at 1.2, binary N=4, 25 runs); binary N=4 diverges at 1.5-1.75, is thrown back to the origin saddle in 25/25 runs at 2.0, diverges >= 2.5; N=8 trees: oscillation to 1.5, re-entry 1.75-2.5, divergence at 3 | empirical (+ 1D map derived) | `fig_stability.png`, `fig_eos_example.png`, `map1d.py` |
| 7 | Imbalanced init (root core x20-60): GD selects minima with sharpness min(lambda_GF, 2/eta) and less imbalance (EoS selection); from balanced init GD creates imbalance ~ eta | empirical | `fig_min_selection.png` |
| 8 | Centred model, binary N=4, sigma 0.3-2: GD limit = 2/lambda_GF(sigma): 1.0, 0.99, 0.97, 0.88, 0.57, 0.19 x eta_max predicted vs boundaries 0.9-1.05, 0.9-1.05, 0.9-1.05, ~0.9, 0.5-0.7, ~0.2 observed; no bounded-oscillation regime on the lazy side | empirical | `fig_lr_regime_map.png` |
| 9 | SGD first plateau (binary N=4, 1930 runs): mean never shorter than GF (shift +0.2-0.3 +- 0.2-0.4 at B = 1); spread ~ (eta/B)^p, per-seed p = 0.50-0.55 (pred. 1/2); gauge balance drifts ~ (eta/B)^p, p = 0.90-1.03 (pred. 1) | derived + empirical | `fig_sgd.png`, `sgd_fits.csv` |
| 10 | SGD later plateau (theory note P9), aligned-route test, 51 runs: T_SGD/T_GF = 1.06-3.06, trapped for (eta/B) T_GF >~ 0.03; B = 16 and 64 at equal eta/B agree to 0.5%; c_gamma = 34-45 (heuristic 26); random-init runs with resolved plateaus: +28 +- 5% at sigma = 0.03, B = 1 | empirical | `fig_p9.png`, `p9_fit.csv`, `sgd_second_plateau.csv` |
| 11 | SGD stability: exact mean-square limit eta*_ms(B) = 0.00107 B (B <= 16), 0.053 (64), 0.112 (256), GD 0.154; empirically SGD converges up to 1.5 eta*_ms and diverges from 2.5 eta*_ms (B = 1, 16), as the theory note's almost-sure limit (~1.9 eta*_ms) predicts | derived + empirical | `sgd_stability.py`, `sgd_limit_test.csv`, `fig_sgd_stability.png` |
| 12 | Stochastic collapse: no collapse of a k = 2 route without label noise (MPS N=4, 5/5 learned, plateau +8 +- 5%); the route is a strict saddle by the sweep-level rule (gauge-projected lambda_min -> -0.50 +- 0.005 as sigma -> 0, predicted -0.52; 6 sigma over a decade x 5 seeds, 2/30 points unresolved); with label noise zeta >= 10 at eta = 5e-4 the learned component itself is erased (15/15), so collapse is not testable there | empirical | `collapse_test.csv`, `k2_saddle_sweep.csv` |
| 13 | Finite samples (binary N=8, d = 2, teacher [3, 2], 5 seeds): small init recovers W* (test < 1e-2) in 5/5 seeds from n = 256 (chi = 4) / 384 (chi = 16) with bond rank exactly 2; exact lazy limit: never at chi = 4 (test error 1.7-30x the zero predictor; tangent space 112 < 256), at n >= 256 = d^N at chi = 16; at n = 512 test error 2.5e-5 / 0.010 / 0.53 / 1.51 / 1.74 and rank 2.0 / 2.4 / 5.0 / 5.1 / 5.3 from small init to centred sigma = 1, 1.5, 2 to the lazy limit | empirical | `fig_samples.png`, `samples_summary.csv` |
| 14 | Second setting d = 3, N = 6 (W in R^729): small init recovers in 5/5 at n = 384 (2/5 at 192), the lazy limit is worse than zero at every n <= 384 | empirical | `fig_samples.png` |

**Theory-note predictions tested (`notes/theory_optimizer.md` v2, section 9):** P1 confirmed (finding 1);
P3 confirmed (finding 7); P4 confirmed (binary N=4 diverges at 1.5-1.75 eta_c); P5 confirmed and refined
(finding 6: the exact 1D-map loss replaces the linear formula, which is 5% high at 1.05 and 23% at 1.2);
P6 confirmed in its exact (variational) form, the tangential formula only for first events; P7
confirmed (finding 9); P8 confirmed (finding 11); P9 confirmed qualitatively with a larger c_gamma
(finding 10); **P10 fails for this teacher**: at 0.25-0.3 eta*_ms the learned component dips below 1/2
within ~70 time units and is erased at 0.15 eta*_ms over 1600 time units.

**Corrections made during the work:** my first SGD analysis claimed k >= 3 routes cannot collapse; the
theory note's O(1) zero-mean coupling shows they can, and the P9 test confirms it (notes section 4).
My conjecture that the small-init sample threshold scales like (1 + 2/d)^N was refuted by the d = 3 run.

## What was done
1. Read docs; merged main several times (literature review, PLAN v1, core fixes 57d5262, library v0
   f2f0d6f, theory notes, analysis standards, diagnostics changes). Commits 6ee7442-9d70187.
2. Theory: sharpness proof, 1D map, backward-error shift, SGD noise, exact SGD mean-square limit
   (`notes/optim_samples.md` sections 1-5). Commits 9d70187, 4dec138.
3. Discarded all pre-fix runs after the teacher/init RNG bug; teacher seed 101. Commit 4da8f4d.
4. Step 1: 750-run GD sweep, backward-error prediction, analysis and figures; interim report.
   Commits 701314a, 6ae7e49, d263db7.
5. Library cross-check (my loops = `ttn.dynamics` to roundoff). Commit 23e558d.
6. Step 3: 275 + 90 finite-sample runs, exact lazy limit. Commits 9e949e5, 9239653.
7. Theory-note tests: P5 from GD data, P9/P10 (51 runs), order flips (600 runs), centred GD (360 runs),
   SGD limit test (80 runs), collapse test, k = 2 Hessian sweep. Commits c633ad3, a88284b, f4746bd,
   880e2bb, e1085a4.
8. Step 2: 1930-run SGD sweep, per-seed fits, run classification. Commit 6b28ae4.
9. Analysis standards (lead, round 1): output-dir options, per-seed fits, classification before
   pooling, Hessian-backed saddle claims. Commit 880e2bb.

## Decisions
- **Teacher [3, 2, 1], d = 3, chi = 3** (as `s2s-core`) so limits and plateaus are comparable across
  agents; **teacher seed 101**, independent of init seeds 0-4 after the RNG fix.
- **Own minimal GD/SGD loops before `ttn.dynamics` landed, kept afterwards**: the library was hours
  away and the interim report was due; after it landed, `validate_library.py` showed identical
  trajectories (GD events to 5e-4, GF to 1e-5 relative, SGD events to 2e-8), so the 2700 runs were not
  redone. New experiments (P9, collapse, SGD limit, Hessians) use `ttn.dynamics` directly.
- **sigma = 0.9 for the N = 8 trees** in step 1 (later events at sigma = 0.5 take > 1000 time units).
- **GD outcome classes** (not defined for GD in `notes/diagnostics.md`): converges / bounded oscillation
  (loss total variation > 10x its net change over the last 20% of steps, or final sharpness x eta/2 > 1)
  / thrown back to origin (loss > 0.9 L(0) after an event) / diverged.
- **Step 3: plain GD with eta = 1/lambda_max** (power iteration) rather than an RK integrator (> 2e5
  evaluations per run in the stiff interpolation phase); **lazy regime via the exact closed form** (GD
  on the lazy side is too ill-conditioned to converge within budget; centred sigma = 1-2 GD runs are
  shown alongside and labelled by convergence status).
- **SGD sweep reduced to 10 noise seeds per init** (20 aligned) when the full grid would have taken ~5 h
  on shared CPUs; earlier runs with more seeds are kept.
- **Collapse test on MPS N=4**: the first version used a pattern that sits inside one core of binary
  N=4 (k = 1); those runs are kept separately and not used for the collapse claim.
- **Second-plateau statistics only for inits with a resolved GF plateau (rho > 3) and no near-tie**,
  following the analysis standards; this leaves 3 inits at sigma = 0.03 and 2 at sigma = 0.1.
- **Scripts write to `/root/outputs/optim_samples/results` by default**; the committed results were
  regenerated with `--out-dir experiments/optim_samples/results`.

## Files left on the machine
All under `/root/outputs/optim_samples/` (~0.3 GB), none needed to regenerate the figures (the summary
CSVs in git suffice); needed only to recompute summaries from scratch:
- `gd/` (~100 MB): per-run npz trajectories + json of the 1710 GD/GF runs (steps 1, order flips, centred).
- `sgd/*.jsonl` (~1 MB): 1930 SGD run summaries.
- `samples/*.jsonl`: 365 finite-sample runs (with test-error curves).
- `p9/`, `collapse/`, `collapse_binary4_k1/`: P9 and collapse runs (npz + json).
- `results/`: default output directory of the scripts (scratch).

## Caveats and next steps
- Step 1 statistics use 5 seeds per setting (45 for order flips); N = 8 trees only at sigma = 0.9, where
  the fan-in init is already mildly imbalanced (edge at ~2/lambda_GF ~ 0.9 eta_max).
- The SGD study is on binary N=4 only (product features make N = 8 SGD ~100x less stable per sample:
  eta*_ms ~ 5e-6 at B = 1). Second-plateau statistics rest on 5 inits.
- P9's c_gamma is larger than the theory note's heuristic and grows with x; the -ln(1-x)/x shape is
  only approximate near the trap. P10's safety fraction depends on the teacher.
- The small-init sample threshold (~250 samples in both settings, 5-9x the teacher manifold
  dimension) is unexplained; worth a theory look (first-escape direction from the empirical
  correlation tensor).
- Collapse with label noise could not be tested at eta = 5e-4; a much smaller eta with larger zeta
  would isolate it, but only for components far below the noise level.
- The lead may want to fold section 7 of `notes/optim_samples.md` (learning-rate axis, practical rule)
  into the report's chapter 5 and the regime map.

## Log
- 18:20 Read brief and project docs; merged origin/main; set up report.
- 18:25 Sharpness formula verified numerically on 5 trees (machine precision).
- 18:30 1D GD-map bifurcations; GD-vs-GF pilot; exact SGD mean-square limit.
- 18:40 SGD plateau pilot: noise spreads but does not shorten plateaus.
- 18:42 Lead message: literature v1 on main; merged; read sections 1.4, 2.7, 3, 5.
- 18:48 Lead message: teacher/init RNG bug fixed. Killed sweeps, merged, teacher seed 101, relaunched.
- 19:02 Step-1 sweep done (750 runs); SGD sweep started.
- 19:10 Backward-error shift prediction matches GD to 3 digits. Step-1 figures.
- 19:20 Interim report (step 1).
- 19:25 Lead: library v0 on main; merged (126 tests pass). Float64-floor check: only never-learned components of N=8 runs collapse below 1e-14.
- 19:33 Library cross-check passed.
- 19:55 Finite-sample sweep done. Lead: theory_optimizer v2; P5 confirmed from GD data.
- 20:15 P9 confirmed (c_gamma 34-45, only eta/B matters, trapping); centred GD limit; order flips (45 seeds).
- 20:20 Collapse test v1 used a k = 1 pattern by mistake; redone on MPS N=4 (k = 2).
- 20:25 Lead: analysis standards. Output-dir options, per-seed fits, run classification, Hessian check.
- 20:35 Finite-sample follow-ups (d = 3, rich-to-lazy path): small-init threshold conjecture refuted.
- 20:45 Empirical SGD stability limit ~2 eta*_ms (matches eta_as).
- 21:05 SGD sweep done (1930 runs); lead: diagnostics changes (saddle type by sigma scaling); k = 2 Hessian sigma sweep.
- 21:20 k = 2 saddle confirmed strict by sigma sweep; SGD stability table regenerated (binary4, mps4; matches the theory note).
- 21:30 All committed results regenerated from raw data with the final scripts; final report.
