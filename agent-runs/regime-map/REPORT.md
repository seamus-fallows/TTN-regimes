# regime-map: phase diagrams of TTN training regimes (Q5)

Status: working - Branch: agent/regime-map - Updated: 2026-10-09 18:56 UTC

## Summary
In progress. Task: phase diagrams of saddle-to-saddle / rich non-incremental / lazy dynamics over
init scale sigma, bond dimension chi, number of cores K and core imbalance, with predicted
boundaries and threshold sensitivity (brief `notes/briefs/regime-map.md`).

## Results
(none yet)

## What was done
1. Read brief, model, diagnostics, plan; merged `origin/main`.

## Decisions
1. **d = 3, odeco teacher [3, 2, 1]** (minimal d for 3 odeco components; keeps the cut dimension
   small: 9 for N = 4, 81 for the top cut of N = 8, so full bonds are reachable). Alternative d = 4/5
   would make full bonds unaffordable for N = 8.
2. **N = 4 bond dimensions uncapped** (`cap=False`), chi in {3,4,6,9,12,18,24,36}: chi = 9 is the
   cut dimension (full, square cores, ill-conditioned), chi > 9 are overparameterized bonds, which is
   what the brief's "up to 8x the teacher rank" requires. N = 8 uses capped bonds (bottom bonds <= 9),
   since uncapped middle cores chi^3 are unaffordable at chi = 81.
3. **Interim integrator.** The library's dynamics/runner had not landed when the first diagram was
   due, so the interim sweeps use scipy RK45 (rtol 1e-9, atol 1e-14) on the flat parameter vector,
   as the brief allows for pilots, with an explicit time budget (README). They will be rerun or
   spot-checked with `ttn.dynamics` when it lands.
4. **All pilot results computed before the core-fix merge (teacher stream bug) were discarded**; the
   pilots had used teacher seed 1000 + s vs init seed s, but every number in this report is from
   runs after merging 57d5262.
5. **Stalled runs are split** (in figures only; the official label stays "stalled") into
   delta_K < 0.1 (kernel-like stall at the lazy residual, a representability effect) and
   delta_K >= 0.1 (feature-learning stall), because below the cut dimension the centred target
   W* + W_0 is unrealizable (theory R6) and "stalled" otherwise mixes two different mechanisms.
6. **Teacher seed = 1000 + init seed**, so seeds vary both; by the O(d) invariance of the init the
   teacher basis is statistically irrelevant, so this costs nothing and avoids any shared stream.

## Files left on the machine
- `/root/outputs/regime_map/` raw outputs (none yet).

## Caveats and next steps
(none yet)

## Log
- 18:18 started; read brief and notes; merged origin/main; machine has ~368 CPUs (cgroup quota), 125 GB RAM
- 18:25 theory draft in notes/regime_map.md: centred-model lazy limit only fits W* at maximal bond dimension (lazy residual), delta_K ~ kappa/sigma^K, rho ~ sigma^-(K-2), imbalance rule
- 18:30 pilot (scipy RK45) binary N=4 sigma x chi, 2 seeds: S2S at small sigma, lazy at large sigma for chi >= 9; final loss = best rank-chi approx of W0+W* (exact); delta_K matches kappa/sigma^3 to <1% at sigma >= 3 when converged
- 18:45 merged main (core fixes: teacher stream, cancellation-free centred loss); discarded old pilots; read literature.md (H4 corrections) and draft theory_regimes.md (agrees with my derivations; adds muP-TTN, exact aligned S2S boundary)
- 18:52 launched interim sweep b4_sigma_chi (N=4, centred/plain/two-copy, 8 chi x 19 sigma x 6 seeds = 2736 runs)
