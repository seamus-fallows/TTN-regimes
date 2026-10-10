# optim-samples: learning rate, SGD noise and finite samples (Q8) - fix round 1

Status: working - Branch: agent/optim-samples - Updated: 2026-10-10 00:00 UTC

## Summary
Asked: how discrete-step GD, SGD noise and a finite training set change the gradient-flow (GF)
picture of TTN training regimes (Q8, H6). Fix round 1 (review `notes/reviews/optim_samples_round1.md`):
every review item was addressed by rerunning with persistent events, more data or a new setup. Six
round-1 claims were withdrawn or narrowed (list below).
- **GD at small init.** The stability limit is exactly eta_max = 2/(K s_1^{2-2/K}). Below it GD keeps
  the GF saddle sequence in every checked case (765/765 runs) and shifts event times by eta c_e
  + O(eta^2), with c_e the exact eta -> 0 limit.
- **Across init scales** (chi = 9, converged GF). 2/lambda_GF(sigma) ~ sigma^4 marks the end of
  monotone GF-like GD. Above it, small init oscillates and then diverges; larger init catapults to
  GF's loss up to ~1.5-3 x 2/lambda_GF.
- **SGD** never shortens plateaus. The first escape keeps its GF law. Later plateaus are lengthened
  and trapped roughly as P9 predicts. Learned components are erased at >= ~0.25 eta*_ms on long
  plateaus.
- **Finite samples** (one teacher). Small init recovers the teacher with its bond ranks from n ~ 256,
  and the lazy limit fails as dimension counting predicts.

## Claims changed in fix round 1
| review item | round-1 claim | now | evidence |
|---|---|---|---|
| 1 (major) | "An O(eta) step flips near-tied races (1-4/45 inits at 0.3-0.9 eta_max) and can remove a whole plateau"; conjecture: flips vanish as sigma -> 0 | **Withdrawn** (artifact of first-crossing events and runs stopped right after the second crossing). Now: **GD below eta_max kept the GF saddle sequence in every checked case**: persistent order = GF in 765/765 runs (binary N=4 at 5 sigma, 5-45 seeds; binary N=8, MPS N=8). What grows with eta is the fraction of inits with a *transient* crossing (sigma = 0.1: 0.09 +- 0.04 at 0.1 eta_max to 0.18 +- 0.06 at 0.9 eta_max), after which the component is learned at its GF time. The event-2 shift outliers (e.g. -110 eta) were the same artifact and are gone. | notes 3.2; `analyze_order.py`, `order_flips.csv`, `fig_order_flips.png` |
| 2 (major) | "Centred model, sigma 0.3-2: the GD limit is 2/lambda_GF(sigma) ~ sigma^{-2(K-1)}; no bounded oscillation on the lazy side" | **Withdrawn**: with chi = 3 the GF references stalled (L/L_0 = 0.06-0.55) because W* + W_0 has bond rank up to 6. Redone with **chi = 9** (cut dimension), centred model, all 45 GF references converged to loss 1e-8. lambda_GF ~ sigma^{3.965 +- 0.014} (per-seed fits over sigma = 1.41-16, 1.05 decades; predicted 2(K-1) = 4). **2/lambda_GF is the end of monotone GF-like GD** at every sigma (all runs at <= 0.95 x 2/lambda_GF GF-like). Above it: small init (sigma = 0.1, 0.3) oscillates (1.05-1.2) then diverges, no catapult. sigma = 1: catapult at 1.05-1.5, divergence from 2. sigma = 2-8: catapult window 1.05 to 2.5-3 x 2/lambda_GF, ending at GF's loss at a flatter minimum (final sharpness x eta/2 = 0.75-0.97); divergence general between 3 and 4. GF-like now requires GF's final loss. | notes 3.5; `analyze_centred.py`, `centred_chi9_*.csv`, `fig_centred_chi9.png`, `fig_lr_regime_map.png` |
| 3 (major) | "Event shift matches c_e per seed to 3 digits for every event at <= 0.1 eta_max, within 2% up to 0.9 eta_max" | **Narrowed**, per event and seed: over 526 comparisons at eta <= 0.1 eta_max, within 0.3% / 1% / 5% of c_e in 66% / 90% / 97% (98% within 0.3% at 0.01 eta_max, 89% at 0.03, 58% at 0.1; the review's subset gave 84% of 199). First events: all within 1% (max 0.69%) at <= 0.1 eta_max, 1-3% (max 9%) at 0.9 eta_max. Later events where c_e ~ 0 deviate by up to ~3x (O(eta^2) term). c_e is the exact limit: Richardson extrapolation agrees to median 1.2e-5 (57 events). The unsupported MPS8 value "-32.40" is removed; the figure shows every run. | notes 3.1; `analyze_shift.py`, `shift_accuracy*.csv`, `shift_richardson.csv`, `fig_shift_accuracy.png` |
| 4 (moderate) | "P7 confirmed: B = 1 delays +0.2-0.3 +- 0.2-0.4; spread 1.2-1.4x the 1D formula" | **Changed.** The s.e. quoted was for a single init, and "t1" sometimes was the co-learned second event or a re-learning after forgetting. Now: escape time = first crossing of any component vs GF's, forgetting classified per run, s.e. across 5 inits. B = 1 delays +0.24 +- 0.10, -0.01 +- 0.12, +0.13 +- 0.07 (sigma = 0.03, 0.1, 0.3) vs P7 +0.17, +0.12, +0.08: consistent within 1.3 s.e., resolved as positive only at sigma = 0.03. P7's magnitude is therefore only weakly tested. Spread: exponents 0.43-0.52 (predicted 1/2); per init 0.4-2.2x the 1D formula (typically 1.0-1.3x). | notes 4.0; `analyze_sgd.py`, `sgd_fits.csv`, `sgd_summary.csv`, `fig_sgd.png` |
| 5 (moderate) | "P9: B = 16 and B = 64 agree to 0.5%; c_gamma = 34-45; random init +28 +- 5%"; "sigma = 0.03, B = 4 within errors of 0" | **Rerun with independent noise seeds** (144 runs, 6 seeds per point, b0 = 0.002-0.01). B = 16 and 64 agree within s.e. at all 8 matched points (largest gap 2.69 +- 0.15 vs 2.45 +- 0.17; the review's independent 2.82 +- 0.23 at b0 = 0.003 is consistent). **Law shape tested across x** with one c_gamma: P9 -ln(1-x)/x fits best (chi^2 379 vs 535 pole vs 929 linear; c_gamma = 32.6), and x = 1 classifies 23/24 conditions correctly. It is not exact: chi^2/dof ~ 21, and the effective c rises from ~30 to ~45 near the trap. Random init (3 inits, s.e. across inits): +20 +- 5% at sigma = 0.03, B = 1 (P9: +19%); +3.6 +- 1.1% at B = 4 (P9: +4.0%; consistent with P9, not with 0). | notes 4.0, 4.1; `p9_sgd.py`, `plot_p9.py`, `p9_conditions.csv`, `p9_fit.csv`, `fig_p9.png`; round-1 data kept as `p9_sgd_round1.csv` |
| 6 (moderate) | "Generalization degrades monotonically with init scale" | **Narrowed**: one teacher per setting, capped GD steps; plain and centred models are named separately. Small init recovers W* with bond rank 2.0-2.1 from n = 256. Within small init the trend is not monotone. The centred sigma = 1.5, 2 runs are not converged even after 300 000 steps (train loss 1.4% and 7.4% of initial), so they are not ranked. The lazy failures follow from dimension counting (chi = 4 tangent space has dimension 112 < 256; full chi needs n = d^N). The chi = 4 lazy rows were deduplicated (they had been counted 3x). | notes 6; `analyze_samples.py`, `samples_summary.csv`, `fig_samples.png`, config block `chi4_centred_long` |
| 7 (minor) | "170/170"; "eta*_ms = 0.00107 B"; P10 pooled; ... | 195/195 runs at <= 0.95 eta_max (675/675 with the order seeds). eta*_ms/B = 0.00107 (B = 1), 0.00100 (B = 16), 0.00082 (B = 64). MPS8 seed 3 (never learns component 1 by T) is annotated in `fig_stability.png` and excluded from P5. N = 8 GF references learning only 1-2 components by T is stated. P10 is split by step fraction: 0/72 runs at <= 0.076 eta*_ms, 1/36 at 0.12-0.15, 20/36 at 0.25-0.30. The P5 script `analyze_p5.py` is committed (excess loss within 0.9% of the 1D map on binary N=4). README run instructions rewritten (full config paths, `--raw-dir`/`--out-dir`, environment variables). | notes 3.3, 4.1, 5; README |
| gauge fix (lead) | k = 2 saddle: gauge-projected lambda_min -> -0.50 +- 0.005 | Recomputed with the fixed library. **Raw** lambda_min is primary: -0.4871 (sigma = 0.1) to -0.5190 +- 0.0001 (sigma = 0.01), tending to the exact -0.5196, so a **strict saddle**. With the default rtol the projected value stays biased at these saddle-start states (-0.503 at sigma = 0.01); both are in the CSV. No other lambda_min tables of mine. | notes 4.2; `check_k2_saddle.py`, `k2_saddle_sweep.csv` |

## Results
Setups (unless stated): whitened Gaussian inputs, odeco teacher [3, 2, 1], d = chi = 3, teacher seed 101,
fan-in init seeds 0-4; trees binary N=4 (K=3), binary N=8 (K=7), MPS N=8 (K=8); events persistent
(last upward crossing of 1/2). Every figure is produced by a script in `experiments/optim_samples/`
from CSVs in `experiments/optim_samples/results/` (raw data in `/root/outputs/optim_samples/`).

| # | finding | tag | evidence |
|---|---|---|---|
| 1 | Sharpness at the balanced global minimum = K s_1^{2-2/K}; balanced = flattest aligned minimum; eta_max = 0.154 / 0.0435 / 0.0366 (binary 4 / binary 8 / MPS 8) | proven; checked to 3e-15 on 5 trees x 2 chi | `check_sharpness.py`, `sharpness_check.csv` |
| 2 | Small init, binary N=4, sigma 0.003-0.3: all 195 runs (seeds 0-4) at <= 0.95 eta_max are GF-like, all 25 at 1.05 eta_max oscillate | empirical | `fig_stability.png`, `gd_summary.csv` |
| 3 | Below eta_max the persistent saddle sequence equals GF's in 765/765 runs; only transient excursions grow with eta | empirical | `order_flips.csv`, `fig_order_flips.png` |
| 4 | Event shift = eta c_e + O(eta^2), c_e from the variational equation (Richardson: median 1.2e-5); accuracy per event in claims table, item 3; relative event-time error at 0.9 eta_max 0.111 ... 0.0036 for sigma = 0.3 ... 0.003 (vanishes as sigma -> 0) | derived + empirical | `shift_accuracy*.csv`, `fig_shift_accuracy.png`, `fig_gd_vs_gf.png` |
| 5 | Above eta_max: 2-cycle of the top component at 1.05-1.3 eta_max; excess loss = s_1^2 x (1D-map attractor loss) within 0.9% (binary N=4: 0.1306 vs 0.1306 at 1.05, 0.4496 vs 0.4496 at 1.2); binary N=8 within 1% at 1.05-1.1, 14-32% low at 1.2-1.3; diverges at 1.5-1.75, thrown back to the origin in 25/25 at 2.0 (binary N=4) | empirical (+ 1D map derived) | `p5_excess_loss.csv`, `fig_stability.png`, `fig_eos_example.png` |
| 6 | Imbalanced init (root core x2-6): GD selects minima with sharpness min(lambda_GF, 2/eta) and less imbalance (catapult/EoS) | empirical | `fig_min_selection.png` |
| 7 | chi = 9 across init scales: lambda_GF ~ sigma^{3.965 +- 0.014}; 2/lambda_GF = end of monotone GF-like GD; catapult window up to ~1.5x (sigma = 1) / ~3x (sigma = 2-8) | derived scaling + empirical | `fig_centred_chi9.png`, `fig_lr_regime_map.png` |
| 8 | SGD first escape: no shortening; P7 delays consistent (claims table, item 4); spread ~ (eta/B)^{0.43-0.52}; gauge drift ~ (eta/B)^{0.90-1.02} | derived + empirical | `sgd_fits.csv`, `fig_sgd.png` |
| 9 | Forgetting (random init): learned component erased and re-learned in 20-66% of runs at 0.47 eta*_ms, 0% at <= 0.24 eta*_ms (runs to 1.3 x the later GF event + 2) | empirical | `sgd_runs.csv`, notes 4.0 |
| 10 | P9 (aligned route, 144 runs): lengthening -ln(1-x)/x with c_gamma = 32.6 (best of three laws, chi^2/dof 21); trapping threshold correct in 23/24 conditions; only eta/B matters (independent noise) | empirical | `fig_p9.png`, `p9_fit.csv`, `p9_conditions.csv` |
| 11 | SGD stability: exact eta*_ms(B) (0.00107 at B = 1, 0.0161 at 16, 0.112 at 256, GD 0.154); empirically converges to 1.5 eta*_ms (5/5), 2/5 at 2.0, diverges from 2.5 (B = 1, 16) | derived + empirical | `sgd_stability.csv`, `sgd_limit_test.csv` |
| 12 | k = 2 route (MPS N=4): strict saddle (raw lambda_min -> -0.519); no collapse without label noise (5/5 learned, plateau +8 +- 5%); with label noise the learned block itself is erased (inconclusive) | empirical | `k2_saddle_sweep.csv`, `collapse_test.csv` |
| 13 | Finite samples (binary N=8, d = 2, teacher [3, 2]): small init recovers W* (test < 1e-2) in 5/5 seeds from n = 256 (chi = 4) / 384 (chi = 16), bond rank 2.0-2.1; exact lazy limit never at chi = 4 (tangent dim 112 < 256), from n = d^N = 256 at chi = 16; converged centred sigma = 1: error 0.01-0.04 at n = 384-512, rank 2.4-2.9 | empirical | `samples_summary.csv`, `fig_samples.png` |
| 14 | d = 3, N = 6 (W in R^729): small init recovers in 5/5 at n = 384 (2/5 at 192); lazy limit worse than zero at every n <= 384 | empirical | `fig_samples.png` |

**Theory-note predictions (`notes/theory_optimizer.md` v2, section 9).**
- Confirmed:
  - P1 (#1);
  - P3 (#6);
  - P4 (#5);
  - P5, in its exact 1D-map form on binary N=4;
  - P6, in the exact variational form (the tangential formula only for first events);
  - P8 (#11).
- Consistent but weakly tested: P7 (mean delays resolved only at sigma = 0.03).
- P9: qualitatively right; the best of three laws but not exact; c_gamma ~ 33.
- P10 fails for this teacher on long plateaus (20/36 dips at 0.25-0.30 eta*_ms; ~0.1 eta*_ms safe
  over <= 1000 time units).

## What was done
Round 1 (commits 6ee7442-91edd1d): theory (sections 1-5 of the notes), step-1 GD sweep, step-2 SGD
sweep, step-3 finite samples, theory-note tests (P5, P9/P10, centred GD, SGD limit, collapse, k = 2
Hessian); see the git history.

Fix round 1:
1. Merged main (review, gauge fix 6c42d85); events switched to last-crossing / persistent; GF records
   all crossings; chi and GF-relative step options. Commits f562cbd, 39658f3.
2. Reran step 1 (750 runs) and the 40-seed order sweep with persistent events; new GD classes
   (catapult, loss criterion); shift accuracy per event + Richardson; P5 script; SGD analysis
   rewritten (escape time, forgetting, s.e. across inits). Commits 2f3a280, a0c243c.
3. P9 rerun with independent noise seeds and a law-shape test; P10 split by step fraction; README
   run instructions. Commit 151c9c8.
4. SGD sweep rerun past the second GF event (1780 runs, independent generators); k = 2 Hessian sweep
   recomputed with the fixed projection (raw + projected); samples: 300 000-step centred runs, lazy
   dedup. Committed in 525637d.
5. chi = 9 centred sweep (45 GF references to loss 1e-8, 300 GD runs); `analyze_centred.py`; regime
   map redrawn from chi = 9 data; notes 3.3, 3.5, 7 rewritten; README summary. Commits 525637d,
   0bce4f0 (merge main), and the final commit of this round.

## Decisions
Round 1 (unchanged):
- **Setup.** Teacher [3, 2, 1], d = chi = 3, teacher seed 101.
- **Own GD/SGD loops, validated** against `ttn.dynamics` to roundoff.
- **sigma = 0.9** for the N = 8 trees.
- **Step 3.** GD at eta = 1/lambda_max, and the exact closed form for the lazy limit.
- **Collapse test** on MPS N=4.

Fix round 1:
- **Persistent events** (diagnostics.md):
  - Options: first crossing, last crossing, a persistence window.
  - Chose the last upward crossing of 1/2 with a_r(T) >= 1/2, keeping first crossings and the number
    of crossings as transient diagnostics.
  - Why: it is the definition the review used and the one that cannot count a transient as an event.
    Runs now continue past the GF events (SGD: 1.3 x the later of GF's first two events + 2).
- **SGD first-escape time:**
  - Options: the persistent t1 of the first-learned component, or the first time any component
    reaches 1/2.
  - Chose the latter, compared with GF's first crossing, with forgetting classified per run.
  - Why: at B = 1 the persistent t1 mixes escape with forget-and-relearn. Classifying before pooling
    follows the analysis standards.
- **GD classes:**
  - "GF's level" = final loss <= 1.05 L_GF(T) + 1e-3 L(0); a separate catapult class; oscillation
    requires the tail maximum above that level.
  - Why: the round-1 class ignored the loss (review item 2), and a factor-2 tolerance counted
    plateau-level 2-cycles as converged.
- **Lazy side with chi = 9 (cut dimension), centred model:**
  - Setup: GF via LSODA to loss 1e-8; GD steps relative to each init's own 2/lambda_GF; GD run as
    long as GF took.
  - Alternative: chi = 6 (just enough for W* + W_0).
  - Why: chi = 9 represents every target and was the review's suggestion. The sigma range
    1-16 gives > 1 decade for the fit.
- **P9 shape test:** one c_gamma for all conditions, fitted in R-space with the same weights for
  three laws (P9, pole, linear), rather than inverting the law per run (which cannot test the shape).
- **k = 2 Hessian:** raw lambda_min quoted as primary, as the lead instructed for saddle-start states;
  projected values kept alongside.
- **Round-1 raw data moved, not deleted** (`gd_round1/`, `sgd_round1/`, `p9/`; reversible), so the
  resumable sweeps reran everything.
- **Centred sigma = 1.5, 2 in step 3:** after 300 000 steps they were still at 1.4% / 7.4% train loss,
  so they are excluded from the ranking rather than run longer (5x more steps lowered the train loss
  only ~3x).

## Files left on the machine
All under `/root/outputs/optim_samples/` (~420 MB). None is needed to regenerate the figures (the
summary CSVs in git suffice); they are needed only to recompute summaries from scratch:
- `gd/` (236 MB): npz + json of the 1703 GF/GD runs of the fix round (step 1, order seeds,
  chi = 9 sweep).
- `gd_round1/` (173 MB): the round-1 GD runs (first-crossing events; incl. the withdrawn chi = 3
  centred runs).
- `sgd/` (1.4 MB): 1780 SGD run summaries (fix round). `sgd_round1/` (1.2 MB): round-1 runs.
- `p9_r2/` (6.3 MB): 144 P9 runs with independent noise seeds. `p9/` (2.5 MB): round-1 P9 runs
  (coupled seeds).
- `samples/` (1.6 MB): 385 finite-sample runs.
- `collapse/`, `collapse_binary4_k1/`: collapse-test runs.
- `results/`: the scripts' default output directory (scratch).

## Caveats and next steps
- **chi = 9 map.** One tree (binary N=4), 5 seeds, and a coarse ratio grid (0.95, 1.05, 1.2, 1.5, 2,
  2.5, 3, 4). The catapult-window edges are known only to that resolution. The window's dependence
  on K was not measured.
- **Event shifts.** The O(eta^2) term was not computed. Where c_e ~ 0 the relative accuracy of the
  first-order formula is poor by construction.
- **SGD** was run on binary N=4 only, and the P7 mean delay is resolved only at sigma = 0.03.
  - The P9 law is approximate near the trap.
  - Forgetting depends on run length, so a "safe fraction of eta*_ms" needs a plateau length
    attached.
- **Finite samples:** one teacher per setting, capped steps; larger centred inits are not converged.
  The small-init threshold (~250 samples, 5-9x the teacher manifold dimension) is unexplained.
- **Collapse with label noise** remains untested (the learned block is erased first at eta = 5e-4).
- **For the lead:** the project `CLAUDE.md` still states round-1 claims: "GD event shift predicted to
  3 digits" and "generalization degrades monotonically with init scale". I left it alone because
  that file is yours. A suggested replacement for the optim-samples bullet:
  > eta_max = 2/(K s_1^(2-2/K)) exact at small init; below it GD keeps the GF saddle sequence (765/765)
  > with event shift eta c_e + O(eta^2) (exact limit; first events within 1% at <= 0.1 eta_max);
  > 2/lambda_GF(sigma) ~ sigma^4 ends monotone GD, with a catapult window up to ~3x on the lazy side
  > (chi = 9); SGD never shortens plateaus and lengthens/traps later ones (P9 shape, c_gamma ~ 33;
  > eta/B only); finite samples (one teacher): small init recovers W* with its bond ranks from
  > n ~ 256, and the lazy limit fails as dimension counting predicts.

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
- 22:00 Lead: FIX ROUND 1 (review on main) and gauge-projection fix; merged main; persistent events in the code; round-1 raw data moved aside.
- 22:07 Step-1 and order sweeps rerun with persistent events: no order change below eta_max (765/765).
- 22:30 Shift prediction rerun; per-event accuracy + Richardson; P5 script.
- 22:34 P9 rerun with independent noise seeds (144 runs); law-shape test.
- 23:17 SGD sweep rerun past the second GF event (1780 runs); P7 with escape time and s.e. across inits; forgetting classified.
- 23:25 k = 2 Hessian sweep recomputed with the fixed projection (raw lambda_min primary); 300k-step centred samples runs done.
- 23:58 chi = 9 centred sweep (373/375 runs): lambda_GF ~ sigma^3.97, catapult window; notes 3.3/3.5/7, README; merged main.
