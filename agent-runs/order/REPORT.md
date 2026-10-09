# order: which teacher component a TTN learns first, and how random the order is (Q4, H1, H1b)

Status: working - Branch: agent/order - Updated: 2026-10-09 19:50 UTC

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
Pilot numbers so far (final numbers will come from the committed configs):
- Leading-order (LO) flow vs full GF, per seed (180 seeds each): agreement 100% (MPS4 chi 4, binary8
  chi 4, binary4 with the LO flow stopped at radius 1/sigma). The independent-race approximation is
  exact for MPS4 (100%) and 98% for binary8 (shared internal cores).
- binary4 (m = 2), 1000 seeds per point, sigma 1e-4 .. 0.1: P(wrong order) = C sigma^gamma with
  gamma = 0.42 +- 0.03, 0.248 +- 0.016, 0.102 +- 0.009, 0.040 +- 0.006 for rho = 0.7, 0.8, 0.9, 0.95;
  predicted (1-rho)/rho = 0.43, 0.25, 0.11, 0.053. Wrong-order seeds at small sigma are a subset of those
  at larger sigma (nested, as the threshold picture predicts).
- MPS4 (m = 4), 400 seeds: P(greedy) = 0.815 / 0.605 / 0.510 at rho = 0.5 / 0.8 / 0.95, identical for all
  sigma from 0.003 to 0.1 (one seed of 400 differs at sigma = 0.3).
- H1b, binary4 started at the wrong saddle + noise sigma: rotation to the larger component after
  t_corr = 57.06 (sigma 1e-4) and 81.87 (1e-6): slope 5.39 per unit ln(1/sigma) vs predicted
  1/|lambda_min| = 1/((1-rho) rho^(1/3)) = 5.386 (rho = 0.8). Gauge-projected Hessian at the exact
  wrong saddle: lambda_min = -0.18566 (predicted -0.18566). Natural wrong-order runs at sigma = 0.01:
  a_2 falls to 3e-4 while a_1 rises (rotation, bond rank stays 1), plateau at L = 0.32 = s_2^2/2 follows.

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
