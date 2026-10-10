# order: which teacher component a TTN learns first, and how random the order is (Q4, H1, H1b)

Status: working - Branch: agent/order - Updated: 2026-10-10 02:56 UTC

## Summary
Asked: is incremental learning in TTNs greedy (largest component first) or decided by the init, on
what does it depend, and is a wrong order corrected through a strict saddle (H1b)? Answer: it depends on
**m, the number of cores carrying the sites where two components differ**. For **m >= 3** (odeco teachers
on MPS N >= 4 and binary N = 8) every component is an attractor of the first escape and the order is
**random and independent of sigma**: P(larger first) = 0.60-0.88 at weight ratio rho = 0.5 and 0.50-0.65 at
rho = 0.8. On an MPS it is an exact race, winner = argmin tau_r/s_r of independent leading-order blow-up
times (100% of 40800 GF runs). The order gets more random with K and more greedy with chi, slowly:
sd(log tau) ~ (K-2)^0.6 chi^-0.53 with over-parameterized bonds. For **m = 2** (binary N = 4, a depth-3
deep linear network, and "matrix-like" pattern pairs) the order is greedy as sigma -> 0, with
P(smaller first) ~ sigma^((1-rho)/rho) measured in GF. **H1b holds only for m = 2**: from the wrong saddle,
the gauge-projected Hessian has lambda_min = -(s_1-s_2) s_2^((K-2)/K) (5 digits) and the learned bond index
rotates to the larger component after ln(1/sigma)/|lambda_min|. In natural runs, though, the correction
takes only O(1/|lambda_min|) with no resolved plateau. For m >= 3 there is no strict saddle, and a wrong
order is kept (rarely the larger component's route even dies). With three components, the full order on m >= 3 trees is a random permutation that favours large components and is close to an iid race (Thurstone) model. With four components and bond dimension equal to the teacher rank, gradient flow mostly stalls before learning the last component.

## Results
All numbers: full gradient flow on the exact population loss, float64, unless marked "LO" (leading-order
flow of g = <W, W*>). Odeco/pattern teachers in the standard basis, weights (1, rho), fan-in init, d = 2
unless stated. Notes with derivations and full tables: `notes/order.md`; code/configs/commands:
`experiments/order/README.md`; tables `experiments/order/results/*.csv`.

**1. Order statistics (step 1; `configs/main.yaml`, `results/main_summary.csv`, figure
`results/main_pg_vs_rho.png`).** 28 (tree, d, chi) settings x 6 rho x 400 seeds at sigma = 0.01, RK45
rtol 1e-9 / atol 1e-14 with event location (67200 runs; 1105 MPS8 runs re-integrated after a float64
time-resolution failure, winners unchanged).

| tree (K, m) | P(larger first) at rho = 0.5 / 0.8 / 0.9, for (d, chi) = (2,2) ... (4,8) |
|---|---|
| binary4 (3, 2) | 0.99-1.00 / 0.86-0.92 / 0.69-0.77 (sigma-dependent, see 2) |
| mps4 (4, 4) | (2,2) 0.79/0.60/0.54; (4,8) 0.88/0.65/0.55 |
| mps6 (6, 6) | (2,2) 0.69/0.55/0.52; (4,8) 0.82/0.62/0.53 |
| binary8 (7, 4) | (2,2) 0.70/0.58/0.53; (4,8) 0.86/0.64/0.57 |
| mps8 (8, 8) | (2,2) 0.63/0.51/0.50; (4,8) 0.76/0.58/0.53 |

- Per-seed agreement of GF with the LO race (argmin tau_r/s_r): 100% on all 40800 MPS runs, 96-99% on
  binary8 (shared internal cores), 79-99.5% on binary4 (where the race is not the right model).
- d matters only through bond caps on MPS (pairs with identical bonds, d = 2 vs 4: chi^2 = 9.2 / 0.6 / 1.3
  with 6 dof for MPS4/6/8; fresh-seed check of the one 2.9-SE outlier: consistent).
- Selection rule (lead's request): with X_r = projection of the init on component r's KKT orbit,
  log tau_r ~ -q log X_r with q = 2.1-2.4 / 4.1-4.9 / 4.9-6.2 / 6.4-7.3 for MPS4 / MPS6 / binary8 / MPS8
  (~K-2), and argmax s_r X_r^q predicts 76-92% of winners (`results/main_rules.csv`). The library's
  decoupled-mode heuristic misses P by -0.27 to +0.07.

**2. sigma dependence (`b4_sigma`, `m4_sigma`, `b8_sigma`; `results/sigma_fits.csv`,
`results/sigma_scan.png`).** m >= 3: P identical for every sigma in 0.003-0.1 (MPS4) and 0.01-0.1 (binary8);
>= 99% of seeds have the same winner at every sigma <= 0.1. m = 2 (binary4, 1000 seeds, sigma 1e-4..1e-2,
seed-bootstrap SE): P(smaller first) ~ sigma^gamma, gamma = 0.35 +- 0.05 / 0.45 +- 0.09 (rho 0.7; chi 2 / 4),
0.232 +- 0.027 / 0.232 +- 0.028 (0.8), 0.110 +- 0.010 / 0.105 +- 0.010 (0.9), 0.045 / 0.041 +- 0.005 (0.95),
against (1-rho)/rho = 0.43, 0.25, 0.11, 0.053 (rho = 0.95 never reaches small P).

**3. Controls on one tree (MPS6, `pat_mps6_*`).** P(smaller first) at sigma = 0.01, rho = 0.7 / 0.8 / 0.9:
m = 2 adjacent 0.00 / 0.045 / 0.20; m = 2 ends 0.065 / 0.17 / 0.33 (gamma = 0.46 / 0.25 / 0.10, as
predicted); m = 3 0.22 / 0.32 / 0.45; m = 4 0.33 / 0.41 / 0.47; m = 6 (odeco) 0.40 / 0.46 / 0.49.
P(smaller first) rises monotonically with m; m >= 3 are sigma-independent. 

Multiplicity (MPS4, GF to sigma = 1e-5, 1000 seeds): the separated m = 2 pair has gamma = 0.240 +- 0.017 / 0.095 +- 0.006 (rho 0.8 / 0.9; predicted 0.25 / 0.111); the adjacent pair (shared bond, chi_e = 4) has 3.2-3.4x larger exponents, 0.82 +- 0.06 / 0.31 +- 0.02, still steepening toward n (1-rho)/rho = 1.0 / 0.44 (theory_escape 5.3).

**4. H1b (`h1b_*`; `results/h1b_fits.csv`, figures `results/h1b_example.png`,
`results/h1b_correction_time.png`, `results/h1b_natural.png`).** LSODA + exact Hessian, rtol 1e-10, events
by root finding; Hessians gauge-projected with the fixed library cut (6c42d85), raw values also stored.
- binary4 started at the wrong saddle + noise sigma (8 seeds x 6 sigma 1e-7..1e-2 x 4 rho): all 192
  runs rotate (a_2 < 6e-4 afterwards), plateau at s_2^2/2, relearning after ~sigma^-1. lambda_min =
  -0.39685 / -0.26637 / -0.18566 / -0.09655 = -(1-rho) rho^(1/3), sigma-exponent 0.000 (strict). Per-seed
  slopes of t_corr vs ln(1/sigma): 2.555 +- 0.036, 3.809 +- 0.056, 5.466 +- 0.082, 10.51 +- 0.16 vs
  1/|lambda_min| = 2.520, 3.754, 5.386, 10.357; log law rms 0.13-0.56 vs power law 1.5-6.0.
- MPS4 from the wrong saddle: no rotation, |lambda_min| ~ sigma^(2.00 +- 0.01) (higher-order saddle),
  t_1 ~ sigma^(-1.97 +- 0.03).
- Natural wrong-order runs, binary4 (421 runs): at sigma <= 0.01, 92.5-100% rotate. |lambda| (t_1 - t_2) =
  1.07-1.38 (+-0.14-0.30), flat in sigma from 1e-4 to 1e-2, so there is no ln(1/sigma) growth. The
  separation ratio has median 0.3-0.6, so no wrong plateau is resolved. MPS4 (40 runs): no rotation; the
  larger component is added after ~2.5e4; 1 run stalls (route dies).

**5. Over-parameterized bonds (`chi_scan`, `chi_gf`; `results/chi_scan*.csv`, `results/chi_scan.png`).**
LO race, uncapped chi = 1..32, 500 seeds: sd(log tau) = 0.57 (K-2)^(0.61+-0.03) chi^-(0.53+-0.01) (rms 5%).
P(larger first) at chi = 32, rho = 0.8: 0.86 (MPS4), 0.78 (MPS6), 0.73 (MPS8); GF at chi = 32 agrees (MPS4
0.84, binary8 0.745; per-seed race agreement 100% / 99.5%). Greedy order is restored only slowly
(P >= 0.9 at rho = 0.9 would need chi ~ 160 for K = 4, ~570 for K = 8, by extrapolation).

**6. Three and four components (step 4; `configs/multi.yaml` -> raw `multi2`, `results/multi2_summary.csv`,
`results/multi2_perms.csv`, figure `results/multi2_orders.png`).** d = chi = 4, weights rho^(r-1), 200 seeds,
sigma 0.05-0.25 by tree, LSODA + exact Hessian with exact event location and a stall stop.

| R = 3 | rho | completed & used | P(greedy order) GF vs iid race | Kendall tau (greedy) GF vs model | own race = GF order |
|---|---|---|---|---|---|
| binary4 | 0.8 / 0.9 | 200 / 200 | 0.70 vs 0.52 / 0.49 vs 0.32 | 0.80 vs 0.60 / 0.58 vs 0.33 | 0.64 / 0.53 |
| mps4 | 0.8 / 0.9 | 195 / 195 | 0.35 vs 0.35 / 0.23 vs 0.25 | 0.36 vs 0.39 / 0.14 vs 0.19 | 0.86 / 0.89 |
| mps6 | 0.8 / 0.9 | 85 / 88 | 0.25 vs 0.29 / 0.20 vs 0.22 | 0.25 vs 0.27 / 0.16 vs 0.13 | 0.82 / 0.77 |
| binary8 | 0.8 / 0.9 | 180 / 177 | 0.23 vs 0.28 / 0.19 vs 0.22 | 0.19 vs 0.25 / 0.05 vs 0.12 | 0.72 / 0.70 |

- For m >= 3 the full order is a random permutation favouring large components, within ~1-2 SE of the
  iid race (Thurstone) model built from the measured blow-up times. Each seed's own LO race predicts
  the complete order in 70-89% of runs. binary4 (m = 2) is far more greedy than the race model.
- Excluded runs are counted: in "stalled" runs some component's overlap decays below 1e-25 (its route
  dies). With R = 3 this happens in 2.5% (MPS4), 1% (binary8) and 44% (MPS6, sigma 0.2; another 9% had
  not finished by t = 1e9) of runs.
  With **R = 4 = chi** most runs do not learn all four components (stalled, or unfinished at t = 1e9):
  MPS4 289/400, MPS6 396/399, binary8 389/393; binary4 0/400.
  The abandoned library-integrator run (integrated to t = 1e9 with no stop) gives the same stall
  classification on all 1679 overlapping runs, so this is GF behaviour, not a stopping artefact. It
  looks like the conservation-law trap of theory_saddles section 8; it is outside Q4 and is passed on
  to s2s-core/locality.
- 10 slow runs (7 binary8 R = 4, 1 MPS6 R = 4, 1 MPS6 R = 3, 1 binary8 R = 3) were stopped unfinished
  after ~3 h to free the machine (`multi.py` resumes them if re-run). They are not in the tables and cannot
  change the conclusions (2 of 1600 R = 3 runs).

## What was done
1. Read brief, repo, notes; merged origin/main repeatedly (literature, core fixes, theory notes v1/v2,
   library v0/final, gauge-projection fix).
2. Theory (notes/order.md): leading-order scale invariance, m-count classification, exact MPS race,
   m = 2 wrong-order law, H1b Hessian (54174e2 and later).
3. Pilots (scratch) -> experiment code `experiments/order/` (22094ee, 7b440f4, 608e564).
4. Step 1 main sweep + repair (6d9c81b); sigma scans, H1b (recomputed after the projection fix),
   pattern controls, chi scan, R >= 3 (later commits; see git log).

## Decisions
- **Teacher basis = standard basis**: the fan-in init is rotation-invariant in law on every physical leg,
  so order statistics equal those of Haar bases; slices become explicit, the per-seed LO comparison exact,
  and the pre-57d5262 teacher/init stream bug could not touch these runs.
- **First-event runs use scipy solve_ivp RK45 (Dormand-Prince 5(4)) with event location** instead of
  ttn.dynamics.integrate_gf: the library integrator had no stop-at-event (and v0 only log-grid output);
  stopping at the first event makes 100k+ runs cheap and gives exact event times. Same method/tolerances.
  Clock restart when the step size hits the float64 resolution of t (repair.py).
- **Full trajectories (H1b, R >= 3) use LSODA with the exact Hessian (as integrate_gf(method="lsoda")) with
  exact event location and a stall stop** (some overlap < 1e-25), because the library integrator cannot
  stop on events and stalled runs otherwise run to t_max through denormals.
- **sigma = 0.01 for step 1**, after checking sigma-independence for m >= 3; binary4's sigma dependence
  is measured separately (b4_sigma).
- **Wrong-order probability fits** use binomial ML with a seed bootstrap SE (the same seeds at every sigma)
  and also a fit restricted to sigma <= 0.01 (further from saturation).
- **R >= 3 uses larger sigma (0.05-0.25 by tree)**: at smaller sigma, dormant components of K >= 5 trees
  fall below the float64 floor; for m >= 3 the first escape does not depend on sigma.
- **Uncapped bonds for the chi scan**: with standard caps chi saturates at the ends (theory_escape 5.1);
  uncapped bonds test over-parameterization directly.

## Files left on the machine
`/root/outputs/order/` (173 MB): raw JSON lines per sweep (`main`, `b4_sigma`, `m4_sigma`, `b8_sigma`,
`pat_*`, `chi_scan`, `chi_gf`, `multi2`, `symmetry_check`; `main/runs_orig.jsonl` = before repair),
H1b trajectories (`h1b_*/*.npz`), pilots (`pilots/`, `multi/` = abandoned log-grid R >= 3 run). All are
regenerated by the configs (README, run order given there). The figures are drawn by the `analyze_*`
scripts from these raw files; the committed CSVs in `experiments/order/results/` hold all reported
numbers. Needed to redraw figures: everything except `pilots/`, `multi/` and `main/runs_orig.jsonl`.

## Caveats and next steps
- binary4 rho = 0.95: measured exponent 1.6-2.4 SE below (1-rho)/rho (pre-asymptotic, P >= 0.33).
- Adjacent m = 2 pairs: exponent between n = 1 and n = chi_e at accessible sigma (theory: slow approach).
- The KKT-projection rule misses 8-24% of winners; a closed form for the blow-up-time distribution
  of a chain is open.
- R >= 3: the order statistics are conditional on completed runs; stall rates depend on sigma and chi = R
  and were not explored further (gauge-balanced inits were not tested).
- Throughput bug found late: workers did not cap OpenBLAS threads (fixed with threadpoolctl); only speed
  was affected.
- Suggested report figures: `main_pg_vs_rho.png`, `sigma_scan.png`, `h1b_example.png`,
  `h1b_correction_time.png`, `chi_scan.png`, `multi2_orders.png`.

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
- 22:04 Merged final library + gauge-projection fix (6c42d85). h1b.py now records raw and projected lambda_min; all four H1b configs are being recomputed (queue3). Found a stall: MPS4 natural wrong-order run (seed 25, rho 0.8, sigma 0.01): after the smaller component is learned, the larger one's overlap decays to 1e-50 by t = 1e7 (route dies, conservation-law trap), GF never learns it; gf_trajectory now stops when an overlap falls below 1e-25 and flags the run as stalled. multi.py switched to merged fine rows for event times.
- 22:54 H1b recomputed (fixed projection; raw + projected): unchanged (binary4 saddle lambda_min = -0.39685/-0.26637/-0.18566/-0.09655, sigma-exponent 0.000; MPS4 |lambda_min| ~ sigma^2.00). Natural MPS4 wrong-order runs: no rotation, 1/40 stalled. Pattern controls on MPS6: m=2 non-adjacent gamma = 0.46+-0.06 / 0.249+-0.029 / 0.101+-0.014 vs (1-rho)/rho; m=2 adjacent (chi_e = 4) gamma = 0.67+-0.09 / 0.22+-0.03 at rho 0.8/0.9 (between n=1 and the asymptotic n=4, as in theory_escape's LO emulation at this sigma range); added MPS4 adjacent/separated configs down to sigma 1e-5 to test n asymptotically.
- 00:12 Performance bug found and fixed: worker processes did not cap OpenBLAS threads (64 per process); with LSODA (LAPACK) and the chi=16/32 init statistics the load reached ~3600 and the queue crawled. All workers now call threadpoolctl.threadpool_limits(1) (and flush denormals). Results unaffected (only speed).
- 00:12 Step 4 restarted as multi2: own LSODA + exact Hessian with exact event location for every crossing and a stall stop (some overlap < 1e-25), instead of the library integrator, because stalled runs (frequent at chi = R = 4) otherwise integrate to t_max through denormals for hours. The partial log-grid run (multi, 1678 jobs) is abandoned (kept in /root/outputs/order/multi).
- 01:13 Integrator check (tol_check.py, 4 trees x 100 seeds, rho 0.8, sigma 0.01): rtol 1e-9 vs 1e-11 give the same winner in 400/400, |dt1|/t1 < 8e-10, conservation drift at the event < 4e-11 (default) / 4e-13 (tight). Multiplicity test done (adjacent m=2 pair exponent 3.2-3.4x the separated one, steepening toward n = chi_e = 4).
- 02:56 Final: all sweeps done except 17 slow multi2 runs (still integrating; not in the tables). Report, notes (notes/order.md) and README complete.
- 10:00 FIX ROUND (review notes/reviews/order_round1.md) started. Code: slice-norm stall criterion (gf_trajectory, gf_events), --raw/--res propagated to spawned workers, default worker count = cgroup CPUs, run_queue.sh passes --raw/--res. Launched: H1b on MPS4 adjacent/separated pairs (saddle + natural), m >= 3 saddle starts on MPS6 and binary8, multi3 (R = 3 rerun with slice criterion), multi_nostop (R = 4, MPS6/binary8, no stall stop).
