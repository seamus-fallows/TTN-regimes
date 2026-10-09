# real-data: do the teacher-student TTN phenomena survive on real images? (B5)

Status: working - Branch: agent/real-data - Updated: 2026-10-09 21:19 UTC

## Summary
In progress. Task: train TTNs (quadtree binary, snake MPS, scrambled binary) by gradient flow on
downsampled MNIST with cos/sin site features, and test plateaus, stepwise bond growth, the locality
rule and the regime map.

## Results
(pending)

## What was done
1. Reverted the launch commit b622158 (in-progress library edits), `uv sync`, pushed (98042d5).
2. Data download + downsampling script `experiments/real_data/download.py` (MNIST, Fashion-MNIST; 4x4 and 8x8).
3. Problem builder `data.py`, two-phase GF runner `run.py`, analysis module `analysis.py`; merged main (39ca8d7).

## Decisions
- **Task: MNIST 3 vs 5 (+-1), n_train = 4000, n_test = 1600, 4x4** (central 24x24 crop, 6x6 block means).
  Options: MNIST/Fashion, one-vs-rest or a pair. Chose a classic confusable, balanced pair, fixed a priori
  (not tuned to make locality look good); Fashion-MNIST is kept as a possible robustness check.
- **Features "white" = per-site whitening with the uncentred second moment** (floor 1e-3 of the top
  eigenvalue). Centring changes the function class (span{cos, sin} does not contain constants), so the
  two variants are raw and whitened; this keeps the network's function class identical.
- **Bond spectra on real data are data-weighted** (`data.function_spectra`: Schmidt values w.r.t. the
  empirical marginals of the two sides of each edge, from per-sample messages). Reason: after the escape
  the Frobenius norm of W grows to ~1e6-1e7 along directions the data barely sees (probe, sigma = 0.5),
  so Frobenius spectra say nothing about the learned function. Checked: equal to the Frobenius spectra
  to 2e-12 on an exactly enumerated product measure with identity second moments.
- **Event references = the run's own final spectra** (no teacher), learned if s_k(T) >= 0.02 s_1(T).
- **Integrator: two phases.** DP5 through the first plateau (cheap, O(log) steps), then LSODA on its own
  clock tau = t - t_a (t_a ~ sigma^-(K-2) up to 1e15: absolute times cannot resolve later gaps in
  float64). After the escape the flow is stiff (lambda_max ~ 57): DP5 would need ~1e9 steps; GD likewise.
  LSODA's Newton matrix: Gauss-Newton part J^T J / n (per-sample Jacobians via vmap, 0.1-1 s) instead of
  autograd's exact Hessian (20 s). The error control is unchanged; GN was faster in wall time than the
  exact Hessian (also implemented, dense residual tensor, 1.2 s) on the probe.
- **Merged main (34c76ad) at the lead's request**; event times use `merged_rows` (log + fine rows; fine rows
  also carry the data-weighted spectra).

## Files left on the machine
- `/root/outputs/real_data/data/` (~25 MB): raw IDX files and downsampled npz (regenerate with download.py).

## Caveats and next steps
(pending)

## Log
- 21:19 reverted launch commit, read brief/notes/library, downloaded data, launched cost pilot
- 21:37 cost probes: post-escape stiffness (lambda_max ~ 57); LSODA + Gauss-Newton Jacobian chosen; function-space spectra added; merged main; pilot (18 runs) launched
