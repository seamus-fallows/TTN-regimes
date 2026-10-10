# order: which teacher component a TTN learns first, and how random the order is (Q4, H1, H1b)

Status: done - Branch: agent/order - Updated: 2026-10-10 11:38 UTC

## Summary
Asked: is incremental learning in TTNs greedy (largest component first) or decided by the init, on
what does it depend, and is a wrong order corrected through a strict saddle (H1b)? Round 1 and a fix
round after the review `notes/reviews/order_round1.md` (section "Fix round" below lists every changed
claim). The errata of the verification `notes/reviews/order_round2_verification.md` (E1-E10) were
applied on 2026-10-10 in the lead's notes-cleanup pass (list: `notes/order.md` section 12; the
verification's "Safe to cite" table C1-C17 is the final list of citable claims). The answer depends on
**m, the number of cores carrying the sites where two components differ**:
- **m >= 3** (odeco on MPS N >= 4 and binary N = 8): every component is an attractor of the first escape;
  the order is **random and sigma-independent** (sigma <= 0.1). P(larger first) = 0.60-0.88 at rho = 0.5,
  0.51-0.65 at 0.8 (sigma = 0.01, 400 seeds). On an MPS it is an exact race, winner = argmin tau_r/s_r of
  independent leading-order blow-up times (40800/40800 GF runs). It gets more random with K and more
  greedy with chi, slowly (sd(log tau) ~ sqrt(K-2)/sqrt(chi) with over-parameterized bonds).
- **m = 2** (binary N = 4 = a depth-3 deep linear network; matrix-like pattern pairs): greedy as
  sigma -> 0; P(smaller first) is consistent with sigma^((1-rho)/rho) for non-adjacent pairs, adjacent
  pairs are 2-3.4x steeper (n = chi_e is a conjecture, supported at the small-sigma end on MPS4 at
  rho = 0.9: 0.40 +- 0.05 vs 0.444).
- **After a wrong order (H1b)**: non-adjacent m = 2 pairs (k >= 3) pass a strict saddle,
  lambda_min = -(s_1-s_2) s_2^((K-2)/K), rotate to the larger component in ln(1/sigma)/|lambda_min| from
  the saddle (only O(1/|lambda_min|) in natural runs) and forget the smaller one; **adjacent pairs
  (k = 2) add the larger component on a spare bond index instead** (lambda = -s_1 s_2^((K-2)/K), nothing
  forgotten; corrected in the fix round); m >= 3: no strict saddle, the wrong order is kept (MPS4, MPS6,
  binary8).
- **Three components** (d = chi = 4, one sigma per tree): a random permutation favouring large components,
  consistent with an independent-race model in every m >= 3 case (|z| <= 1.6) and, paired against each
  run's own leading-order race, no less greedy (41 vs 43 discordant runs, McNemar p >= 0.45). [The fix
  round's "less greedy in 6/6 m >= 3 cases (combined z = -1.9)" is withdrawn after verification, E1: the
  corrected combined z is -1.4.] **Four components with chi = 4**: most runs do not learn all four by
  t = 1e9 (MPS6 often only two).

## Results
All numbers: full gradient flow on the exact population loss, float64, unless marked "LO" (leading-order
flow of g = <W, W*>). Odeco/pattern teachers in the standard basis, weights (1, rho), fan-in init, d = 2
unless stated. Notes with derivations, full tables and scope: `notes/order.md` (sections 7.1-7.6);
code/configs/commands: `experiments/order/README.md`; tables `experiments/order/results/*.csv`.

**1. Order statistics (`configs/main.yaml`, `results/main_summary.csv`, figure `main_pg_vs_rho.png`).**
28 (tree, d, chi) settings x 6 rho x 400 seeds at sigma = 0.01, RK45 rtol 1e-9 / atol 1e-14 with event
location (67200 runs; tolerance check 500/500 winners equal at rtol 1e-11, `results/tol_check.csv`).

| tree (K, m) | P(larger first) at rho = 0.5 / 0.8 / 0.9, for (d, chi) = (2,2) ... (4,8) |
|---|---|
| binary4 (3, 2) | 0.99-1.00 / 0.86-0.92 / 0.69-0.77 (sigma-dependent, see 2) |
| mps4 (4, 4) | (2,2) 0.79/0.60/0.54; (4,8) 0.88/0.65/0.55 |
| mps6 (6, 6) | (2,2) 0.69/0.55/0.52; (4,8) 0.82/0.62/0.53 |
| binary8 (7, 4) | (2,2) 0.70/0.58/0.53; (4,8) 0.86/0.64/0.57 |
| mps8 (8, 8) | (2,2) 0.63/0.51/0.50; (4,8) 0.76/0.58/0.53 |

- Per-seed agreement with the LO race: 100% on all 40800 MPS runs, 96-99% on binary8. The GF P sits
  0.018 below the pooled race curve on average (shared seeds; race curves 0.66-0.89 at rho = 0.5).
- d: derived to enter only through bond caps on MPS; independent-seed test (1000 + 1000 seeds): no effect
  detected at equal bonds on binary8 and MPS4 (|diff| <= 0.021, |z| <= 1.2; differences larger than about
  0.06 excluded at 95%, E3; `results/d_test.csv`).
- Selection rule: X_r = projection of the init on component r's KKT orbit; log tau_r ~ -q log X_r with
  q = (0.99-1.25)(K-2), SE 1-4% (E7); argmax s_r X_r^(K-2) predicts 76-92% of winners on m >= 3 trees, the same
  out of sample (`results/main_rules.csv`).

**2. sigma dependence (`b4_sigma`, `m4_sigma`, `b8_sigma`, `pat_*`; `results/sigma_fits.csv`,
`sigma_scan.png`).** m >= 3: P identical for every sigma <= 0.1 (MPS4, binary8, MPS6 patterns). m = 2,
non-adjacent pairs, gamma over the full measured range (seed bootstrap SE) vs (1-rho)/rho:

| pair | rho = 0.7 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|
| binary4, chi 2 / 4 (sigma 1e-4..0.1, 1000 seeds) | 0.37 / 0.42 (+-0.03-0.05) | 0.237 / 0.248 (+-0.018) | 0.103 / 0.102 (+-0.008) | 0.042 / 0.040 (+-0.004) |
| MPS6 ends (0.003..0.1, 400 seeds) | 0.46 +- 0.06 | 0.249 +- 0.029 | 0.101 +- 0.014 | - |
| MPS4 cores 0,2 (1e-5..0.1, 1000 seeds) | - | 0.240 +- 0.017 | 0.095 +- 0.006 | - |
| prediction (1-rho)/rho | 0.429 | 0.25 | 0.111 | 0.053 |

Fits over the three smallest sigma agree with (1-rho)/rho within 1.1 SE in all 13 non-adjacent sets;
the full-range fits above are low in 12/13 (pre-asymptotic; E5). Adjacent pairs (shared bond,
chi_e = 4): MPS6 0.67 +- 0.09 / 0.22 +- 0.03, MPS4 0.82 +- 0.06 / 0.31 +- 0.02 at rho = 0.8 / 0.9 over
the full range, i.e. 2-3.4x steeper; on MPS4 at rho = 0.9 the fit over sigma = 1e-3..1e-5 gives
0.40 +- 0.05, consistent with n (1-rho)/rho = 0.444, while MPS6 (to sigma = 0.003) and rho = 0.8 stay
below n (1-rho)/rho (n = chi_e: conjecture; E2). Wrong-order seeds are exactly nested across sigma
(binary4, all 8 sets).

**3. H1b (`h1b_*`; `results/h1b_fits.csv`, `h1b_local_slopes.csv`; figures `h1b_example.png`,
`h1b_correction_time.png`, `h1b_natural.png`).** LSODA + exact Hessian; gauge-projected (fixed cut) and
raw Hessians.

| from the wrong saddle (8 seeds x sigma 1e-7..1e-2) | slope of t1 vs ln(1/sigma) | 1/|lambda_pred| | lambda_min (raw) | after t1 |
|---|---|---|---|---|
| binary4 (m = 2, k = 3), rho 0.5/0.7/0.8/0.9 | 2.555/3.809/5.466/10.51 | 2.520/3.754/5.386/10.357 | -0.3967/-0.2663/-0.1856/-0.0965 | rotated, a_2 < 6e-4 (192/192) |
| MPS4 cores 0,2 (m = 2, k = 3), rho 0.5/0.8/0.9 | 2.8282/5.5903/10.541 | 2.8284/5.5902/10.541 | -0.3536/-0.1789/-0.0949 | rotated (144/144) |
| MPS4 cores 1,2 (m = 2, k = 2), rho 0.5/0.8/0.9 | 1.4130/1.1171/1.0533 | 1.4142/1.1180/1.0541 | -0.7070/-0.8943/-0.9485 (x3) | added, a_2 stays 1 (144/144) |

- Local slope x |lambda| -> 1.0000 at sigma = 1e-7 in all three (binary4's 1.4% excess comes from sigma =
  1e-2); a power law fits 11-88x worse (rotation sets; 104-110x for the adjacent set; E9).
- m >= 3: no rotation; |lambda_min| ~ sigma^2.00 (MPS4), sigma^4.2 (MPS6), sigma^5.2 (binary8, narrow
  range): higher-order saddles.
- Natural wrong-order runs: k = 3 (binary4 301 runs, MPS4 cores 0,2 240 runs, sigma 1e-4..1e-2): rotate
  in 98%, |lambda| (t1 - t2) = 1.0-1.4 flat in sigma, no resolved wrong plateau; k = 2 (150 runs): the
  larger component follows within 0.1-1 time units, nothing forgotten; m >= 3 (MPS4, 40): resolved
  plateau, larger component added after ~sigma^-2, 1 stall.

**4. Over-parameterized bonds (`chi_scan`, `chi_gf`).** sd(log tau) ~ chi^-(0.48-0.54) per tree; the K
dependence approaches sqrt(K-2) at large chi (K-exponent 0.66 -> 0.52 from chi = 1 to 32). P(larger
first) at chi = 32, rho = 0.8: 0.86 (MPS4), 0.78 (MPS6), 0.73 (MPS8); GF agrees at chi = 16, 32.

**5. Three and four components (`multi3` for R = 3, `multi`/raw `multi2` and `multi_nostop` for R = 4;
d = chi = 4; sigma 0.05 / 0.1 / 0.2 / 0.25 for binary4 / MPS4 / MPS6 / binary8).**
- R = 3, GF vs iid race model on the same runs (P(greedy order)): binary4 0.71 vs 0.52, 0.49 vs 0.32 (m = 2,
  far more greedy); MPS4 0.35 vs 0.36, 0.23 vs 0.25; binary8 0.22 vs 0.28, 0.19 vs 0.21; MPS6 0.25 vs 0.30,
  0.22 vs 0.23 (rho = 0.8, 0.9). Used runs: 200, 195, 196-197, 91-97 of 200 (MPS6: 87 stalled + 15-21
  unfinished; there the excluded runs differ in race order, only marginally per the verification, and
  the smallest component is unlearned most often). Own race = GF first event 97-100%, full order 70-89%.
  For m >= 3 every case is consistent with the iid race (|z| <= 1.6), and paired against each run's own
  race GF is no less greedy (41 runs greedy for GF only, 43 for the race only; McNemar p >= 0.45 per
  row; E1). multi3 reproduces multi2 run for run; only the roundoff flag changed (E10).
- R = 4 = chi: most runs do not learn all four components by t = 1e9 (MPS4 289/400, MPS6 396/399, binary8
  389/393; binary4 0/400; observed directly for the t_max and no-stop runs, inferred for the other
  multi2 runs, whose "learned" counts are at the stop time; E10). No-stop check (22 runs, MPS6 and
  binary8, t = 1e9): none learns all four; MPS6 learns 2 of 4 in 7/11, binary8 3 of 4 in 11/11 (the unlearned one is the largest in 6). MPS4 (K = 4)
  stalls too, which `notes/theory_regimes.md` R10 states for K >= 5.

## Fix round (review notes/reviews/order_round1.md): changed claims
1. **H1b** narrowed: holds for m = 2 with k >= 3 only; the k = 2 (adjacent) branch adds instead of
   rotating (new runs on MPS4 cores 1,2 and 0,2).
2. **m = 2 law**: n carried everywhere, n = chi_e tagged conjecture, "still steepening" withdrawn, one fit
   range with local slopes and a goodness-of-fit note, "consistent with" instead of "confirmed".
3. **R = 3**: rerun with a slice-norm stall criterion; same-subset comparison; excluded runs reported;
   "close to an independent race" qualified (6/6 below the model, combined z = -1.9). [Superseded after
   verification, E1: the "less greedy" reading is withdrawn; GF is consistent with the race.]
4. **R = 4**: "stalls before the last component" -> "most runs do not learn all four by t = 1e9";
   MPS6 often 2 of 4; no-stop check added; scope stated; R10 cross-reference.
5. **m >= 3 / no strict saddle**: now checked on MPS4, MPS6 and binary8; scope stated.
6. **d test**: invalid chi^2 withdrawn; independent-seed test added.
7. **Unresolved Hessians**: all natural runs now resolved (finer record window); full-precision event
   times (round-1 CSVs rounded t ~ 1e8 to 6 digits).
8. Smaller: H1b precision (local slope -> 1/|lambda|), lambda_min statement restricted to sigma <= 1e-2
   (refined after verification, E4: at or above the saddle value to within 1.3%), race-curve ranges and
   offset, sqrt(K-2) wording, q with SE (1-4%, E7) and out-of-sample accuracy.
9. Code: --raw/--res reach spawned workers, default worker count = container CPUs, run_queue.sh passes
   options, README corrected, tol_check includes MPS8, symmetry check completed.
Unchanged (reproduced by the review): scale symmetry, m-count classification, exact MPS race, P tables
and trends, separated m = 2 power law, H1b for k >= 3, natural |lambda| (t1 - t2) flat in sigma, MPS4
contrast, R = 3 on MPS4.

## What was done
1. Round 1 (2026-10-09/10): theory (notes/order.md), main sweep, sigma scans, pattern controls, H1b,
   chi scan, R >= 3 (commits up to 3f8f741).
2. Fix round (2026-10-10): merged the review; code fixes (slice-norm stall criterion, output dirs in
   workers, default procs, run_queue options); new runs h1b_saddle_mps4_adj/sep, h1b_natural_mps4_adj/sep,
   h1b_saddle_mps6, h1b_saddle_b8, multi3, multi_nostop, d_test_*, finished symmetry_check, tol_check with
   MPS8; reran the natural H1b sets with a finer record window and full-precision CSVs; revised analyses
   (same-subset R >= 3, local slopes, q SE, K exponent per chi, race offset); notes sections 0, 5-8, 10,
   11 and this report rewritten.

## Decisions
- **Teacher basis = standard basis** (rotation invariance of the init); no teacher randomness.
- **First-event runs use scipy RK45 with event location**, full trajectories LSODA + exact Hessian with
  exact event location (the library integrator cannot stop on events); clock restart at the float64
  resolution of t.
- **Stall criterion (fix round)**: a component's slice of some core at 1e-14 x the largest core norm
  (route decaying exponentially; justified empirically by the verification: no learned component's slice
  fell below 6.7e-6 before its event in the 22 no-stop runs; it is not a roundoff floor, since unlearned
  slices keep falling and underflow). The round-1 overlap criterion |a_r| < 1e-25 also fired during a
  normal forget phase (review); runs that neither converge nor reach the slice floor are reported as "unfinished at
  t_max", not stalled.
- **One fit range for all m = 2 power laws** (the full measured sigma range), with the sigma <= 0.01 fit
  and local slopes as sensitivity checks.
- **R = 3 at one sigma per tree (0.05-0.25)**: smaller sigma puts dormant components of K >= 5 trees below
  the float64 floor; the resulting scope (outside the checked sigma-independence for MPS6 and binary8) is
  stated rather than removed.
- **binary8 saddle starts: one slow run stopped** after 80 min and the CSV rebuilt from the 29 stored
  trajectories (`rebuild_h1b_csv.py`), rather than waiting for t = 1e10.
- **Slow tails stopped**: 3 multi3 runs and 2 multi_nostop runs still integrating at the end were
  stopped (resumable); they are reported as missing.

## Files left on the machine
`/root/outputs/order/` (346M): raw JSON lines per sweep (`main` (+ `runs_orig.jsonl` before repair),
`b4_sigma`, `m4_sigma`, `b8_sigma`, `pat_*`, `chi_scan`, `chi_gf`, `multi2` (round 1), `multi3`,
`multi_nostop`, `d_test_*`, `symmetry_check`), H1b trajectories (`h1b_*/*.npz`), pilots and the abandoned
`multi` run. All are regenerated by the configs (README gives the run order). The figures are drawn by
the `analyze_*` scripts from these files; all reported numbers are in `experiments/order/results/*.csv`.

## Caveats and next steps
- m = 2: the law is a heuristic consistent with GF for non-adjacent pairs; its prefactor and the
  adjacent-pair multiplicity are open; rho = 0.95 never reaches small P.
- R >= 3 statistics are at one sigma per tree with d = chi = 4 and fan-in init; MPS6 R = 3 is conditional
  on completion (half the runs). Next: spare channels (chi >= 2R) and gauge-balanced init for the stalls.
- Selection rule: q is above K-2 by 0-25% and the rule misses 8-24% of winners; the LO race is exact.
- binary8 m >= 3 "wrong order kept" is shown at sigma 0.15-0.35 only, at or above the crossover of the
  saddle-order theorem (`notes/theory_saddle_order.md` v2: order 6 < k = 7, forgetting predicted below
  it; added in the notes-cleanup pass).
- Suggested report figures: `main_pg_vs_rho.png`, `sigma_scan.png`, `h1b_example.png`,
  `h1b_correction_time.png`, `chi_scan.png`, `multi3_orders.png`.

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
- 21:25 [SUPERSEDED in part: the "+-1.5%" is round-1 wording; the excess comes from the sigma = 1e-2 point and the local slope x |lambda| -> 1.0000 at small sigma (fix round; verification E8)] H1b done: saddle-start binary4 (8 seeds x 6 sigma 1e-7..1e-2 x 4 rho): per-seed slopes 2.555/3.809/5.466/10.51 (+-1.5%) vs 1/|lambda| = 2.520/3.754/5.386/10.357; log law rms 0.13-0.56 vs power law 1.5-6.0; lambda_min sigma-exponent 0.000 (strict), all 192 runs rotate. MPS4 saddle start: lambda_min ~ sigma^2.00 (higher order), t1 ~ sigma^-1.97, no rotation. Natural binary4 wrong-order runs: rotation in 92-100% at sigma <= 0.01, |lambda|(t1-t2) = 1.07-1.38 flat in sigma, separation ratio median 0.3-0.6 (no resolved wrong plateau).
- 22:04 Merged final library + gauge-projection fix (6c42d85). h1b.py now records raw and projected lambda_min; all four H1b configs are being recomputed (queue3). Found a stall: MPS4 natural wrong-order run (seed 25, rho 0.8, sigma 0.01): after the smaller component is learned, the larger one's overlap decays to 1e-50 by t = 1e7 (route dies, conservation-law trap), GF never learns it; gf_trajectory now stops when an overlap falls below 1e-25 and flags the run as stalled. multi.py switched to merged fine rows for event times.
- 22:54 H1b recomputed (fixed projection; raw + projected): unchanged (binary4 saddle lambda_min = -0.39685/-0.26637/-0.18566/-0.09655, sigma-exponent 0.000; MPS4 |lambda_min| ~ sigma^2.00). Natural MPS4 wrong-order runs: no rotation, 1/40 stalled. Pattern controls on MPS6: m=2 non-adjacent gamma = 0.46+-0.06 / 0.249+-0.029 / 0.101+-0.014 vs (1-rho)/rho; m=2 adjacent (chi_e = 4) gamma = 0.67+-0.09 / 0.22+-0.03 at rho 0.8/0.9 (between n=1 and the asymptotic n=4, as in theory_escape's LO emulation at this sigma range); added MPS4 adjacent/separated configs down to sigma 1e-5 to test n asymptotically.
- 00:12 Performance bug found and fixed: worker processes did not cap OpenBLAS threads (64 per process); with LSODA (LAPACK) and the chi=16/32 init statistics the load reached ~3600 and the queue crawled. All workers now call threadpoolctl.threadpool_limits(1) (and flush denormals). Results unaffected (only speed).
- 00:12 Step 4 restarted as multi2: own LSODA + exact Hessian with exact event location for every crossing and a stall stop (some overlap < 1e-25), instead of the library integrator, because stalled runs (frequent at chi = R = 4) otherwise integrate to t_max through denormals for hours. The partial log-grid run (multi, 1678 jobs) is abandoned (kept in /root/outputs/order/multi).
- 01:13 [SUPERSEDED in part: "steepening toward n = chi_e = 4" was withdrawn in the fix round; after verification (E2) n = chi_e is a conjecture supported only at the small-sigma end on MPS4, rho = 0.9] Integrator check (tol_check.py, 4 trees x 100 seeds, rho 0.8, sigma 0.01): rtol 1e-9 vs 1e-11 give the same winner in 400/400, |dt1|/t1 < 8e-10, conservation drift at the event < 4e-11 (default) / 4e-13 (tight). Multiplicity test done (adjacent m=2 pair exponent 3.2-3.4x the separated one, steepening toward n = chi_e = 4).
- 02:56 Final: all sweeps done except 17 slow multi2 runs (still integrating; not in the tables). Report, notes (notes/order.md) and README complete.
- 10:00 FIX ROUND (review notes/reviews/order_round1.md) started. Code: slice-norm stall criterion (gf_trajectory, gf_events), --raw/--res propagated to spawned workers, default worker count = cgroup CPUs, run_queue.sh passes --raw/--res. Launched: H1b on MPS4 adjacent/separated pairs (saddle + natural), m >= 3 saddle starts on MPS6 and binary8, multi3 (R = 3 rerun with slice criterion), multi_nostop (R = 4, MPS6/binary8, no stall stop).
- 11:38 FIX ROUND done: H1b k=2 branch (MPS4 adjacent pairs add, not rotate), m=2 law with n (n = chi_e conjecture), R=3 same-subset comparison with slice-norm criterion (multi3), R=4 no-stop check, m>=3 contrast on MPS6 and binary8, independent-seed d test, unresolved Hessians resolved, code/README fixes. Changed claims listed in notes/order.md section 11 and above. [Section 11's R = 3 "less greedy in 6/6" claim was later withdrawn, E1.]
- 2026-10-10 (lead's notes-cleanup pass, not the agent) Errata of notes/reviews/order_round2_verification.md applied to notes/order.md (section 12), this report and the experiment README: E1 R = 3 "less greedy than independent clocks" withdrawn (consistent with the race; paired 41 vs 43, McNemar p >= 0.45, corrected combined z = -1.4); E2 n = chi_e supported at the small-sigma end on MPS4 (0.40 +- 0.05 vs 0.444), still a conjecture; E3 d test excludes differences > 0.06; E4-E10 precision and scope fixes; the 21:25 and 01:13 entries marked superseded.
