# regime-map: phase diagrams of TTN training regimes (Q5)

Status: working - Branch: agent/regime-map - Updated: 2026-10-09 18:18 UTC

## Summary
In progress. Task: phase diagrams of saddle-to-saddle / rich non-incremental / lazy dynamics over
init scale sigma, bond dimension chi, number of cores K and core imbalance, with predicted
boundaries and threshold sensitivity (brief `notes/briefs/regime-map.md`).

## Results
(none yet)

## What was done
1. Read brief, model, diagnostics, plan; merged `origin/main`.

## Decisions
(none yet)

## Files left on the machine
- `/root/outputs/regime_map/` raw outputs (none yet).

## Caveats and next steps
(none yet)

## Log
- 18:18 started; read brief and notes; merged origin/main; machine has ~368 CPUs (cgroup quota), 125 GB RAM
- 18:25 theory draft in notes/regime_map.md: centred-model lazy limit only fits W* at maximal bond dimension (lazy residual), delta_K ~ kappa/sigma^K, rho ~ sigma^-(K-2), imbalance rule
- 18:30 pilot (scipy RK45) binary N=4 sigma x chi, 2 seeds: S2S at small sigma, lazy at large sigma for chi >= 9; final loss = best rank-chi approx of W0+W* (exact); delta_K matches kappa/sigma^3 to <1% at sigma >= 3 when converged
