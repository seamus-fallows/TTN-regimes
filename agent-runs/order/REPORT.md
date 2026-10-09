# order: which teacher component a TTN learns first, and how random the order is (Q4, H1, H1b)

Status: working - Branch: agent/order - Updated: 2026-10-09 20:56 UTC

## Summary
Interim (main sweep still running). Theory (notes/order.md, consistent with the draft
notes/theory_saddles.md): what decides the order is m, the number of cores carrying sites where the
two components differ. **m >= 3** (odeco on MPS N >= 4, binary N = 8): every component is a local
maximum of g = <W, W*> on the sphere, so the order is random; P(greedy) is independent of sigma;
on an MPS it is an exact race between independent per-component flows, P = P(tau_2/tau_1 > rho).
**m = 2** (binary N = 4, a depth-3 deep linear network, and "matrix-like" pattern pairs): only the
largest component is a local maximum; at finite sigma the smaller one still comes first with
probability ~ sigma^((1-rho)/rho), and the order is then corrected through a strict saddle (H1b).
Pilots confirm all three predictions quantitatively (details under Results). The full sweep (step 1),
sigma scans, m-controls, the chi scan, H1b runs and the R >= 3 runs are queued.

## Results

### Step 1 (interim, complete): order statistics, two odeco components, weights (1, rho)
Full gradient flow (scipy RK45 = Dormand-Prince 5(4), rtol 1e-9, atol 1e-14, event location at the
first crossing a_r = 1/2), sigma = 0.01, 400 init seeds (the same seeds for every rho), 28 distinct
(tree, d, chi) settings (binary4/8, MPS4/6/8; d 2, 4; chi 2, 4, 8 with standard bond caps), 67200 runs.
Command: `sweep.py configs/main.yaml`, `analyze_main.py main`; table `experiments/order/results/main_summary.csv`,
figure `results/main_pg_vs_rho.png`. 1105 runs first stopped with a step-size failure at t ~ 1e13-1e14
(float64 time resolution; MPS8 only); all were re-integrated with a clock restart (`repair.py`), 0 left,
and no winner changed.

P(larger first), selected (binomial SE <= 0.025):

| tree (K, m) | d, chi -> P at rho = 0.5 / 0.8 / 0.9 |
|---|---|
| binary4 (3, 2) | 2,2: 0.99/0.86/0.69; 2,4: 1.00/0.87/0.70; 4,4: 1.00/0.91/0.76; 4,8: 1.00/0.92/0.77 |
| mps4 (4, 4) | 2,2: 0.79/0.60/0.54; 2,4: 0.82/0.61/0.55; 4,4: 0.87/0.61/0.52; 4,8: 0.88/0.65/0.55 |
| mps6 (6, 6) | 2,2: 0.69/0.55/0.52; 2,8: 0.76/0.60/0.54; 4,4: 0.81/0.61/0.56; 4,8: 0.82/0.62/0.53 |
| binary8 (7, 4) | 2,2: 0.70/0.58/0.53; 2,4: 0.76/0.59/0.54; 4,4: 0.78/0.59/0.51; 4,8: 0.86/0.64/0.57 |
| mps8 (8, 8) | 2,2: 0.63/0.51/0.50; 2,4: 0.71/0.58/0.54; 4,4: 0.73/0.55/0.52; 4,8: 0.76/0.58/0.53 |

Findings (all at sigma = 0.01, 400 seeds per setting):
- **H1 holds for m >= 3 (empirical + derived): the order is random.** Even at rho = 0.5 the smaller
  component is learned first in 12-40% of inits on MPS4-8 and binary8; at rho = 0.9 P is 0.50-0.57.
- **Exact race model (derived, verified per seed).** Per init, the leading-order race
  (winner = argmin tau_r / s_r, tau_r the blow-up time of the LO flow with only component r present)
  predicts the GF winner in **100.0% of the 40800 MPS runs** and 96-99% of binary8 runs (shared internal
  cores make it approximate). The race curve P(tau_2/tau_1 > rho) from the pooled blow-up times overlays
  the GF points (figure).
- **Dependence on K, chi, d (empirical).** sd(log tau) grows with K (MPS d=2, chi=2: 0.59, 0.93, 1.20 for
  K = 4, 6, 8) and falls with chi (MPS8 d=4: 1.19, 0.76, 0.66 for chi = 2, 4, 8), so P(greedy) falls toward
  1/2 with K and rises with chi, slowly (`results/main_settings.csv`, `results/main_spread.png`).
  d matters only through the bond caps on MPS: settings with identical bonds and d = 2 vs 4 differ by
  chi^2 = 9.2 / 0.6 / 1.3 (6 dof each) for MPS4 / 6 / 8 (`results/main_d_pairs.csv`).
- **binary4 (m = 2, a depth-3 deep linear network) is close to greedy and sigma-dependent**: P = 0.99-1.00
  at rho = 0.5; the independent race (0.90-0.97) underestimates it, as expected for a unique local max.
- **Selection rule "s_r X_r^q" (lead's request; `analyze_rule.py`, `results/main_rules.csv`).** With
  X_r = projection of the init onto the nearest balanced bond-rank-one configuration of component r (its
  KKT orbit), the fitted exponent of log tau vs -log X is q = 2.1-2.4 (MPS4), 4.1-4.9 (MPS6), 4.9-6.2
  (binary8), 6.4-7.3 (MPS8), i.e. ~K-2 as in the Kumar-Haupt escape time, and "argmax s_r X_r^q" predicts
  76-92% of GF winners on the m >= 3 trees (exact race: 97-100%). Weaker predictors on the same trees:
  |grad T_r(theta_0)|^2 82-91%, product of per-core overlaps 75-90%, |T_r(theta_0)| 68-77%, slice norm
  66-83%. The library heuristic `ttn.theory.winner_probability` (decoupled balanced modes) misses
  P(greedy) by -0.13 to +0.07 (c_r = |T_r(0)|^(1/K)) and -0.27 to +0.05 (product-Gaussian c_r) on the
  m >= 3 trees (`results/main_heuristic.csv`).

### Pilots for steps 2-3 (final runs queued)
- binary4 sigma scan (pilot, 1000 seeds, chi 4): P(wrong order) = C sigma^gamma, gamma = 0.42 +- 0.05,
  0.248 +- 0.018, 0.102 +- 0.008, 0.040 +- 0.004 (seed bootstrap) for rho = 0.7, 0.8, 0.9, 0.95; predicted
  (1-rho)/rho = 0.43, 0.25, 0.11, 0.053 (rho = 0.95 never reaches the small-P regime: P >= 0.34 for all sigma).
- MPS4 sigma scan (400 seeds): P(greedy) identical for every sigma in 0.003-0.1.
- H1b: binary4 started at the wrong saddle + noise sigma rotates to the larger component after
  t_corr = 57.06 (sigma 1e-4) and 81.87 (1e-6), slope 5.39 vs 1/|lambda_min| = 5.386 (rho 0.8, one seed);
  gauge-projected Hessian at the exact wrong saddle: lambda_min = -0.18566 = -(1-rho) rho^(1/3).

## What was done
1. Read brief, repo, model, diagnostics, plan; merged origin/main repeatedly (literature, core fixes,
   theory drafts, library v0).
2. Theory in notes/order.md (54174e2 and later).
3. Pilots (scratch scripts), then experiment code in experiments/order/ (22094ee, 7b440f4).

## Decisions
- **Teacher basis = standard basis** (basis='standard'): fan-in Gaussian init is invariant in law
  under orthogonal rotations of physical legs, so all order statistics are identical to Haar-random
  bases; the standard basis makes the component slices explicit and the per-seed LO comparison exact.
  It also means no teacher randomness at all (so the pre-57d5262 stream bug could not touch these runs).
- **First-event runs use scipy solve_ivp RK45 (Dormand-Prince 5(4)) with event location**, not
  ttn.dynamics.integrate_gf: the library integrator has no event/stop-on-condition, and stopping at
  the first event makes 36k runs cheap (1-10 s each). Same method and tolerances (rtol 1e-9,
  atol 1e-14); will be cross-checked against integrate_gf on a subset.
- **Full trajectories (H1b) use LSODA with the exact Hessian** (same as integrate_gf(method="lsoda"));
  R >= 3 runs use the library integrator directly.
- **sigma = 0.01 for the main sweep**: for m >= 3 the outcome is sigma-independent (checked), for
  binary4 sigma matters and is scanned separately (b4_sigma).

## Files left on the machine
- `/root/outputs/order/`: raw outputs (JSON lines per sweep, H1b trajectories); see README.

## Caveats and next steps
- R >= 3 at small sigma hits the float64 floor (dormant overlaps < 1e-14 on MPS6 at sigma = 0.03) and
  a conservation-law stall was seen (MPS4, d = chi = 4, R = 4: one component's overlap decays to 1e-254);
  step 4 will use larger sigma, flag floor runs and report stalls separately.

## Log
- 18:18 Started; read brief and repo; merged origin/main.
- 18:23 Derived theory (m-count rule, MPS exact race, m=2 wrong-order exponent, H1b Hessian) in notes/order.md; pilots: GF to first event costs 1-9 s/run; leading-order race on MPS gives P(greedy)~0.73-0.87 at rho=0.5
- 18:48 Per-seed pilot: full GF first event agrees 100% with the leading-order flow (MPS4, binary8; binary4 when the LO flow is stopped at radius 1/sigma). Independent-race approx: 100% MPS4, 98% binary8.
- 18:48 binary4 sigma-scan (1000 seeds x 5 rho x 7 sigma): P(wrong order) ~ sigma^gamma with gamma = 0.41, 0.24, 0.10, 0.04 for rho = 0.7, 0.8, 0.9, 0.95 vs predicted (1-rho)/rho = 0.43, 0.25, 0.11, 0.05.
- 18:48 Merged origin/main (literature v1, core fixes 57d5262). My pilots used basis='standard' (no random teacher draw), so the teacher/init stream bug did not affect them; final runs will use the fixed library anyway.
- 18:52 mps4 sigma-scan (400 seeds, sigma 0.003-0.3): P(greedy) identical for every sigma <= 0.1 (0.815/0.605/0.510 at rho 0.5/0.8/0.95): sigma-independent as derived for m >= 3.
- 18:52 Merged draft theory notes (theory_saddles.md agrees with my m-rule and H1b derivation). Launched main sweep (configs/main.yaml: 28 settings x 400 seeds x 6 rho, sigma=0.01, PID 7480, ETA ~75 min).
- 19:15 H1b pilot: LSODA(+exact Hessian Jacobian) needed for later plateaus (explicit RK45 is stability-limited near learned saddles); LSODA and RK45 agree to 1e-9 on event times. Saddle-start binary4 rho=0.8: t_corr = 57.06 (sigma 1e-4), 81.87 (1e-6): slope 5.39 per ln(1/sigma) vs predicted 1/|lambda| = 5.386; relearning time of the smaller component 1.3e4 vs 1.3e6 (sigma^-1, k = 3).
- 19:15 Balance check (lead item 3): at the end of the leading-order flow (|theta| x 1e4), per-core norm fractions equal 1/K to < 5e-5 and per-bond imbalance |Q|/|theta|^2 < 3e-5 on all five trees (results/balance_check.csv).
- 19:50 Merged library v0; switched gauge projection to ttn.diagnostics (identical eigenvalues); wrote multi.py (step 4, library integrate_gf lsoda). Pilot: MPS4 R=4 chi=4 one component stalls (a -> 1e-254); MPS6 at sigma 0.03 hits the float floor.
- 20:11 Partial main data: race model per-seed agreement 100% (MPS4), 97-99% (binary8); KKT-projection rule predicts 85-92% of winners with fitted exponent ~K-2. Natural binary4 wrong-order runs (sigma 1e-4): t1 - t2 = 2.6-4.5 (O(1/|lambda|)), a2 then -> 0 and is relearned after ~6000 (sigma^-1). MPS4 saddle start (m=4): no rotation (a2 = 0.9987), second rank added, lambda_min = -7.8e-4 (small, ~sigma^2).
- 20:11 R>=3 pilots: binary8 at sigma 0.1 has dormant overlaps ~1e-11 after the first escape; MPS5 at sigma 0.05 loses its third component (a -> 1e-16, never learned by t = 1e9). multi.yaml uses sigma 0.05-0.25 per tree.
- 20:55 Main sweep finished (67200 runs, 111 min); 1105 MPS8 runs re-integrated with clock restart (repair.py), winners unchanged. Step-1 interim results written above.
- 21:04 Symmetry check (fresh seeds 100000+, MPS4 d=4 chi=2, 1500 inits): mean log(tau1/tau2) = -0.028 +- 0.022, so the small positive bias across main settings was sampling (settings share draws for equal seeds); the main MPS4 d=4 chi=2 value 0.73 at rho=0.5 is a seed-sample fluctuation (fresh seeds: 0.81, race curve 0.79).
- 21:25 b4_sigma (1000 seeds, chi 2 and 4) done: gamma (sigma <= 0.01, seed bootstrap) = 0.35 +- 0.05 / 0.45 +- 0.09 (rho 0.7), 0.232 +- 0.027 / 0.232 +- 0.028 (0.8), 0.110 +- 0.010 / 0.105 +- 0.010 (0.9), 0.045 +- 0.005 / 0.041 +- 0.005 (0.95) vs (1-rho)/rho = 0.43, 0.25, 0.11, 0.053. m4_sigma/b8_sigma: P identical for every sigma <= 0.1 (gamma = 0.000 +- 0.004).
- 21:25 H1b done: saddle-start binary4 (8 seeds x 6 sigma 1e-7..1e-2 x 4 rho): per-seed slopes 2.555/3.809/5.466/10.51 (+-1.5%) vs 1/|lambda| = 2.520/3.754/5.386/10.357; log law rms 0.13-0.56 vs power law 1.5-6.0; lambda_min sigma-exponent 0.000 (strict), all 192 runs rotate. MPS4 saddle start: lambda_min ~ sigma^2.00 (higher order), t1 ~ sigma^-1.97, no rotation. Natural binary4 wrong-order runs: rotation in 92-100% at sigma <= 0.01, |lambda|(t1-t2) = 1.07-1.38 flat in sigma, separation ratio median 0.3-0.6 (no resolved wrong plateau).
