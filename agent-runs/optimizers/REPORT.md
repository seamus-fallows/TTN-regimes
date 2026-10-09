# optimizers: does the saddle-to-saddle picture survive Adam, preconditioned/Riemannian GD and ALS?

Status: working (interim report, steps 1-2 done on binary N=4; deep trees running) - Branch: agent/optimizers - Updated: see Log

## Summary
Asked: which features of the gradient-flow (GF) saddle-to-saddle (S2S) picture (plateaus, one-at-a-time
rank growth, learning order, plateau ~ sigma^-(K-2)) survive the optimizers used in practice.
Answer so far (binary N=4 complete, 6 seeds x 7 sigma over 1.8 decades; deep trees in progress):
GD and heavy ball reproduce GF exactly. **Adam/RMSProp keep the sequential, weight-ordered learning
but break the plateau scaling**: the first escape takes a sigma-independent number of steps (exponent
0.01 +- 0.01 vs GF 1.01 +- 0.03), later plateaus are set by Adam's second-moment memory and the
adaptive edge of stability rather than by sigma. **Gauge-aware / second-order methods remove the
plateaus altogether**: GD with re-gauging to canonical form, Riemannian GD, per-core Gram
preconditioning, ALS and DMRG learn all components at once, in a sigma-independent time (exactly
sigma-free for ALS/DMRG, proven), in an order unrelated to the teacher weights.

## Results

Dynamics: GF = `ttn.dynamics.integrate_gf` (LSODA, rtol 1e-9, atol 1e-14); GD lr 0.02; heavy ball lr
0.002, beta 0.9; Adam lr 1e-3 (default betas, eps 1e-8) unless stated; all on the exact population loss.
Problem: binary N=4 (K=3), d=3, chi=4, odeco teacher (3, 2, 1), fan-in init; seeds 0-5;
sigma = 0.002 ... 0.128. Source: `experiments/optimizers/results/scaling_b4_summary.csv`,
`scaling_b4_exponents.csv`, `scaling_b4_order.csv` (`make_figures.py b4`).

### Plateau scaling, binary N=4 (fit T ~ sigma^-p over all seeds, p +- s.e.)

| method | first escape | gap 1->2 | gap 2->3 | weight order | S2S (rho>3) | balance index at end |
|---|---|---|---|---|---|---|
| GF (theory: p = K-2 = 1) | 1.007 +- 0.026 | 1.008 +- 0.046 | 1.011 +- 0.038 | 100% | 95% | 0.00 |
| GD lr 0.02 | 1.005 +- 0.026 | 1.008 +- 0.046 | 1.011 +- 0.038 | 100% | 95% | 0.00 |
| heavy ball | 0.970 +- 0.025 | 1.006 +- 0.047 | 1.010 +- 0.038 | 100% | 95% | 0.00 |
| Adam lr 1e-3 | 0.012 +- 0.006 | 0.50 +- 0.13 | 0.03 +- 0.11 | 98% | 19% | 0.38 |
| Adam lr 1e-3, beta2 0.99 | 0.011 +- 0.007 | 0.36 +- 0.10 | 0.03 +- 0.05 | 98% | 7% | 0.34 |
| Adam lr 1e-3, beta2 0.9 | 0.013 +- 0.007 | 0.21 +- 0.07 | 0.04 +- 0.02 | 93% | 0% | 0.33 |
| Adam lr 1e-2 | 0.009 +- 0.008 | 0.23 +- 0.18 | 0.12 +- 0.07 | 86% | 12% | 0.39 |
| RMSProp lr 1e-3 | -0.003 +- 0.012 | 0.14 +- 0.07 | 0.04 +- 0.03 | 98% | 0% | 0.33 |
| GD + re-gauge (every step / 50 steps) | 0.000 +- 0.03 | 0.00 +- 0.09 | 0.00 +- 0.09 | 67% | 0% | gauge-fixed |
| Riemannian GD (lr 0.1, 1) | 0.000 +- 0.02 | 0.00 +- 0.09 | 0.00 +- 0.15 | 0% | 0% | gauge-fixed |
| ALS / DMRG | 0 (exact) | 0 | 0 | 0% | 0% | gauge-fixed |

- Event = component overlap a_r reaching 1/2; S2S = all separation ratios > 3 (`notes/diagnostics.md`).
- Adam's first escape takes ~185 steps at lr 1e-3 and ~30 at lr 1e-2 for every sigma (derived: Adam forgets the init
  scale, section 1.2 of `notes/optimizers.md`). Its later gaps shrink with shorter beta2 memory.
- Adam ends at balance index 0.38 on the root edges, as predicted by the entry-count heuristic (36 vs 16 entries:
  (36-16)/(36+16) = 0.385); GF stays at O(sigma^2).
- Figures: `experiments/optimizers/results/figures/scaling_b4_scaling.png` (scaling, all methods),
  `scaling_b4_side_sigma0.008_seed0.png` (loss, overlaps and bond spectrum for one init, per method).

### Adam's later plateaus: memory, then the adaptive edge of stability

`experiments/optimizers/results/figures/adam_probe.png` (`adam_probe.py`, config `adam_probe.yaml`): after a
component is learned, the median preconditioner sqrt(vhat) decays at the memory rate beta2^(n/2) (binary4,
beta2 0.999: on the guide), so dormant components see GD with a growing step lr/P; on long plateaus (binary8,
mps8) the decay stops when lambda_max(P^-1 H) reaches the adaptive-edge-of-stability threshold 38/lr (ratio 1.0 in
the last panel row), after which Adam is GD with an lr-independent step ~38/lambda_max. Theory in
`notes/optimizers.md` 1.2 (e').

### Gram-preconditioned GD (all cores at once)

`precond_b4` (210 runs): with scale-free damping the first step overshoots the loss by up to 1e34 L0 (derived:
each core jumps towards its own least-squares solution, ~sigma^-(K-1) too large), then returns; with an absolute
damping floor (Levenberg-Marquardt-like) it converges in 14-36 steps for every sigma with no overshoot (fastest
gradient method here). My prediction that an absolute floor brings the GF plateau back (as Adam's eps does) is
right in form but the floor acts once per bond leg, so near the origin it is GD with lr / floor^(#legs)
(10-1000 GF units per step here): the plateau would only reappear at far smaller sigma.

### Deep trees (in progress)
- With fan-in init, GF on mps8 (chi=4 > d=3) and binary8 stalls: dormant components are annihilated during earlier
  escapes (overlaps ~1e-40; conservation-law trap of `notes/theory_saddles.md` R12). A gauge-balanced version of
  the same W_0 (Q_e = 0) converges in 12/12 pilot runs with clean S2S; Adam converges from either init.
  Deep-tree comparisons therefore use the balanced init as the GF baseline and fan-in as the practitioners' default.

## What was done
1. Read brief and notes; merged main (library v0, RNG fix); pilot code in `experiments/optimizers/`.
2. Optimizers: Adam/RMSProp/heavy ball/GD via torch.optim; GD + re-gauging (QR sweeps to root-centred canonical
   form); per-core Gram preconditioning (Kronecker-factored exact Hessian blocks); Riemannian GD (tangent projection
   through the dense Jacobian + leaves-to-root HSVD retraction); one-site ALS; two-site DMRG with SVD truncation;
   gauge balancing. Tests: `experiments/optimizers/test_optimizers.py`.
3. Sweeps: `scaling_b4` (672 runs), `precond_b4` (210), `pilot_trap`, `adam_probe`; running: `deep` (864),
   `eps_mps8` (192), `implicit_bias_b4` (240).

## Decisions
- **Balanced init on deep trees.** Options: fan-in init only (practitioners' default), chi = d, or a gauge-balanced
  version of the same W_0. Chosen: balanced for the GF baseline plus fan-in for the gauge-dependent methods,
  because fan-in GF stalls (dead dormant components) on mps8/binary8, chi = 3 stalls too, and balancing keeps W_0.
- **Clock.** GF time for GF/GD/heavy ball (t = lr n / (1 - beta)); steps for Adam/RMSProp/preconditioned/Riemannian
  (their natural cost unit); sweeps for ALS/DMRG. Exponents are clock-independent.
- **GD/heavy ball on deep trees only at sigma = 0.8**: at sigma = 0.3 GF plateaus last 1e4-1e6 GF time units,
  beyond any affordable GD budget, and binary N=4 already shows GD = GF.
- **Riemannian GD implemented densely** (Jacobian + least squares, HSVD retraction on the dense tensor): exact and
  simple at d^N <= 6561; not a scalable implementation.
- **Gram preconditioning without a divergence guard** (the overshoot is the method's behaviour, not a failure).

## Files left on the machine
- `/root/outputs/optimizers/<sweep>/<run_id>.npz`: per-run trajectories (`ttn.dynamics.Trajectory` format; cfg in
  meta), `index.jsonl` per sweep. Regenerate with `uv run python -m experiments.optimizers.run
  experiments/optimizers/configs/<sweep>.yaml` (resumable). Sizes: see final report.

## Caveats and next steps
- Deep-tree comparison, eps crossover and implicit-bias sweeps still running.
- The Adam later-plateau model (memory + AEoS) is heuristic; the AEoS threshold is quoted from memory (Cohen et
  al. 2022) and measured, not derived.

## Log
- 19:15 Read brief, notes; merged origin/main (59bac30, includes the RNG-stream fix 57d5262). Nothing computed before the fix.
- 19:20 Created report and experiment folder; CPU quota 92 cores, 503 GB RAM.
- 19:22 Pilot code for all optimizers; pilot on binary4.
- 19:24 Merged library v0 (f2f0d6f), ported to ttn.dynamics; calibration on deep trees showed the fan-in trap; added balanced init.
- 19:26 Launched scaling_b4, pilot_trap, implicit_bias_b4; then precond_b4, deep, eps_mps8, adam_probe.
- 19:33 binary4 results, figures, Adam probe figure; interim report.
