# locality: tree locality of saddle-to-saddle escapes (Q3, H2/H3)

Status: working (interim report after step 1 on MPS/binary N = 4; **numbers below are from the pre-fix library and are being rerun**) - Branch: agent/locality - Updated: 2026-10-09 19:33 UTC

## Summary
Task: test H2 (after a component is learned, the next escape needs the k cores of the Steiner subtree of
the sites where the new component differs; plateau ~ sigma^-(k-2), log for k = 2, none for k = 1) and H3
(strict saddle iff k = 2) across trees. **Interim answer (N = 4, MPS and binary, 8 seeds): H2's exponent
rule holds exactly for two-component teachers (k = 1, 2, 3, 4 all as predicted), with one quantitative
correction: the k = 2 log-law slope is (1 - s2/s1)/lambda, not 1/lambda, because the second component's
dormant entries grow during the first plateau. H3's Hessian prediction holds: lambda_min -> -lambda_2 for
k = 2, and lambda_min ∝ sigma^(k-2) -> 0 for k >= 3.** For teachers with 3+ components the literal H2
("sites where it differs from what is learned") is ambiguous and, on our derivation, wrong; a bond-rank rule
replaces it (being tested now).

## Results

### Interim exponent table, N = 4, two-component pattern teachers (weights 1.0 / 0.5, d = 2, chi = 2)
Gradient flow, scipy RK45 (Dormand-Prince 5(4)), rtol 1e-9, atol 1e-14, float64; conservation-law drift
< 2e-10 in every run. sigma: 6 values, 1e-3..0.1 (k <= 3) or 0.03..0.3 (k = 4); 8 seeds (6 for the two k = 4
MPS rows, two seeds still running when the pilot was stopped). Exponent p: per-seed least-squares slope of
log(t2 - t1) vs log(1/sigma), mean ± SE over seeds, greedy-order runs only. Source:
`experiments/locality/results/pilot_n4_table.csv` (from `/root/outputs/locality/pilot2_n4.jsonl`, made by
`experiments/locality/pilot/pilot2.py` + `fit.py`).

| tree | S | k (Steiner) | predicted | measured exponent p | log-law slope (pred.) | λ_min at plateau, σ = 1e-3 |
|---|---|---|---|---|---|---|
| binary4 | {0,1} | 1 | no plateau | t2 - t1 ≈ 3e-4 (∝ σ) | – | – |
| binary4 | {2,3} | 1 | no plateau | t2 - t1 ≈ 4e-4 (∝ σ) | – | – |
| mps4 | {0,1} | 2 | log | 0.21 ± 0.03 (log law) | **1.005 ± 0.004** (1.0; naive 2.0) | -0.4969 (pred. -0.5) |
| mps4 | {1,2} | 2 | log | 0.19 ± 0.02 (log law) | **1.008 ± 0.003** (1.0; naive 2.0) | -0.4977 (pred. -0.5) |
| binary4 | {1,2} {0,3} {0,2} {0,1,2} {0,1,2,3} | 3 | 1 | 1.010, 1.006, 1.007, 1.008, 1.003 (SE ≤ 0.004) | – | -3e-4..-7e-4, ∝ σ^1.01-1.05 |
| mps4 | {0,2} {0,1,2} {1,2,3} | 3 | 1 | 1.000, 0.996, 1.000 (SE ≤ 0.006) | – | -5e-4..-1e-3, ∝ σ^1.00-1.01 |
| mps4 | {0,3} {0,1,2,3} | 4 | 2 | 1.998 ± 0.006, 1.964 ± 0.009 | – | ∝ σ^2.1-2.3 |

Figures: `experiments/locality/results/pilot_n4_escape.png` (t2 - t1 vs 1/σ by k, with predicted slopes) and
`pilot_n4_hessian.png` (−λ_min vs σ: flat at λ2 = 0.5 for k = 2, slopes 1 and 2 for k = 3, 4).

Other observations (empirical):
- Order: k = 2 runs are always greedy (larger component first), as derived (the smaller component is not a
  local max of the first-escape problem for k = 2). For k >= 3 some seeds learn the smaller component first
  (mps4 S = {0,1,2,3}: 3 of 6 seeds); the exponent is the same in both orders.
- For m = 2 S-cores (e.g. mps4 S = {0,3}) a wrong-order run shows the H1b "piggyback" rotation: the smaller
  component is learned, the larger one replaces it within O(1) time (strict saddle), then the smaller is
  relearned after the k = 4 plateau. The runner now records every crossing so this is tracked.
- Separation ratio rho at the smallest σ: 2e-4 (k = 1, no plateau), ~2 (k = 2: log plateaus are short),
  ~1e3 (k >= 3).

## What was done
1. Read brief, model, diagnostics, plan, literature (v1). (b0f9ad7)
2. Theory: `notes/locality.md` (Steiner rule derived from Eckart–Young + lowest-order expansion; carry-over of
   dormant entries from the first plateau; Hessian at the saddle; rank rule for several components; literature
   framing). (a656ce4, later commits)
3. Pilot runner with scipy RK45 + compiled torch gradient (library integrator not yet on main); N = 4 sweep
   (14 (tree, S) cases x 6 σ x 8 seeds). (a656ce4)
4. Running: all trees (MPS 4/6/8, binary 4/8, HT 4/8, star3, midroot6), small-init and saddle-start
   protocols; nested multi-component teachers (step 3).

## Decisions
- **Discarded all results computed with the pre-57d5262 library** (lead's instruction: pattern-teacher bases and
  student inits shared a random stream, so seed-0 runs started correlated with the teacher). Old outputs moved
  to `/root/outputs/locality/obsolete_prefix/`; the interim N = 4 table below is kept only as a record and is
  superseded by the rerun. Teacher seed is now 1000 (init seeds 0-7), stored in every record.
- **torch.compile shim.** The fixed library computes einsums through cached opt_einsum expressions, which
  `torch.compile` cannot trace; the runner swaps in `torch.einsum` only while compiling (same contraction,
  agreement 3e-20). No library change.
- **Pilot integrator = scipy `solve_ivp` RK45 on the flat vector with a `torch.compile`d autograd gradient**
  (the brief's suggested pilot path). torch.compile gives ~10x (2 ms -> 0.2-0.4 ms per gradient), which makes
  the stiff later plateaus affordable (~1 step per unit time). Will port to `ttn.dynamics`/`ttn.runner` when
  they land and rerun the key rows.
- **Per-seed exponent fits.** sigma rescales a fixed random init direction per seed, so the law holds seed by
  seed; fitting per seed and averaging removes the large seed-to-seed prefactor scatter.
- **Saddle-start protocol (in addition to the brief's small init).** For K = 15 (HT N = 8) the first plateau is
  ~1e9 already at σ = 0.3 and never ends below σ ≈ 0.14, so the brief's protocol cannot reach small σ there.
  Starting at the exact balanced saddle s1 U1 plus σ-scale fan-in noise isolates the escape law for every tree,
  and for k = 2 it should give slope 1/λ (no carry-over), a direct check of the carry-over explanation.
- **Teachers for step 3** chosen by an exhaustive search over nested 3-component pattern teachers on N = 4 for
  maximally different predicted sequences on MPS/binary/HT with all predicted k <= 4 (feasible):
  TA = {0000, 1100, 1110} (weights 1, .6, .35) and TB = TA + {1111} (.2).

## Files left on the machine
- `/root/outputs/locality/pilot2_n4.jsonl` (N = 4 pilot results, 648 runs), `pilot2/` (subsampled trajectories).
- `/root/outputs/locality/pilot3_*.jsonl`, `pilot3/` (current sweeps).

## Caveats and next steps
- The N = 4 table above comes from the first pilot (first crossings only; a few k = 4 seeds missing); it is being
  rerun with every crossing recorded and with wall-clock limits.
- Next: full table over all trees; saddle-start log slopes; nested teachers on three trees; bond spectra at
  events; robustness (chi = 4, 8; d = 4; weight ratio).

## Log
- 18:20 read brief, model, diagnostics, plan; library dynamics/diagnostics/runner not on main yet, so
  piloting with solve_ivp as instructed
- 18:27 first pilot (solve_ivp, MPS/binary/HT N = 4): k = 1, 2, 3, 4 behave as H2 predicts
- 18:31 theory note: Steiner rule derived; carry-over predicts log slope (1 - s2/s1)/lambda
- 18:36 N = 4 sweep (8 seeds x 6 sigma): exponents 1.00 (k = 3), 2.0 (k = 4), log slope 1.005 (pred 1.0)
- 18:40 lead: literature v1 merged; results framed against Furman's N, Rawal-DeWeese, Kumar-Haupt
- 18:42 HT8 small init infeasible (t1 > 1e14 below sigma 0.14); added saddle-start protocol
- 18:45 interim report pushed
- 18:46 lead: core fix (teacher/init random streams) -> killed sweeps, merged, teacher seed 1000, reran
- 18:51 lead: draft theory_saddles.md merged; nested teachers TA/TB have non-partial-sum saddles
  (k* = 1 at the partial sum); implemented the route rule and the route-sequence predictor
- 19:17 lead: library v0 merged; new runner = library Problem + LSODA (exact-Hessian Jacobian) with event
  root-finding; 35-60x faster than RK45, event times agree to 3e-8 (LSODA/Radau/RK45)
- 19:25 multi-component results: B4, E8 exponents as predicted; TD supports the route rule over the exact
  rank rule; B4 on HT4 reproduces the R12 trap with fan-in init, balanced init restores p = 3
- 19:33 two-component sweep restarted cheap-first (k >= 6: 5 seeds, narrower sigma)
