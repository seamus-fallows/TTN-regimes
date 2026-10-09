# Brief: `optimizers` — does the saddle-to-saddle picture survive the optimizers people actually use? (backlog B1)

Read `notes/briefs/README.md` first (instructions for all agents).

## Task
Compare the training dynamics of TTNs under plain gradient descent / gradient flow with the optimizers
used in practice, on the same teachers and initializations, and find out which features of the
saddle-to-saddle picture (plateaus, incremental bond-rank growth, learning order, plateau scaling with init)
survive.

## Goal
A clear statement, for practitioners and for the report, of whether our gradient-flow results describe
TTN training with Adam (the default in TN-ML code), with preconditioned/Riemannian variants, and how a
non-gradient sweep method (ALS/DMRG-style) differs.

## Context
- Backlog B1 in `PLAN.md`. Literature (`notes/literature.md` sections 2.7, 3 (H6), 5 item 12): Adam lacks
  the gauge equivariance behind GD's low-rank bias (Singh 2026); GD conserves the per-bond quantities Q_e
  (gauge symmetry), Adam does not; Riemannian / dynamical low-rank (BUG) methods act on the fixed-rank
  manifold; Stoudenmire & Schwab 2016 train MPS by DMRG-like sweeps. Theory notes:
  `notes/theory_escape.md`, `notes/theory_saddles.md`, `notes/theory_optimizer.md` (in progress).
- Other agents: `s2s-core` (GF phenomenology and scaling), `locality`, `order`, `regime-map`,
  `data-teacher`, `optim-samples` (GD step size, SGD noise, finite samples: do not duplicate; you study the
  optimizer *algorithm*, they study step size and noise for plain GD/SGD).

## What to do
1. **Baselines**: gradient flow (library integrator) on binary N = 4, 8 and MPS N = 8 with odeco teachers
   (3 components, weights e.g. [3, 2, 1]) and one locality-type pattern teacher, small init sigma (deep in the
   saddle-to-saddle regime) and one moderate sigma; 5+ seeds.
2. **Adam** (and RMSProp; plain momentum as a control): learning rates spanning stable values; same
   inits. Measure: plateaus (count, durations vs sigma), bond-spectrum steps, learning order, the
   conservation-law drift (Q_e is not conserved: how far does balance move?), final bond ranks / implicit
   bias in an underdetermined variant (finite samples). Is the plateau duration still sigma^-(K-2)?
   Derive what you can (e.g. sign-GD / normalized updates near the origin change the escape law).
3. **Gauge-aware / preconditioned GD**: (a) GD with periodic re-gauging to a canonical form (QR sweeps);
   (b) Riemannian gradient on the fixed-rank TTN manifold or a simple natural-gradient-like preconditioner
   (per-core Gram-matrix preconditioning, as in ALS normal equations). Do they remove plateaus?
4. **One non-gradient contrast**: ALS / DMRG-style sweeps (one or two cores at a time, exact least squares
   on the population loss) from the same small init. Does incremental rank growth occur, and in which order?
5. Summarize per optimizer: plateaus yes/no, scaling law, order, rank growth pattern, wall-clock cost.

## Deliverables and success criteria
- `experiments/optimizers/` and `notes/optimizers.md`. Figures: side-by-side loss curves and bond spectra per
  optimizer for the same init; plateau duration vs sigma per optimizer with fitted exponents; conservation
  drift under Adam; a summary table.

## Constraints
- Interim report after steps 1-2 (~3 h of work). Raw outputs to `/root/outputs/optimizers/`.
- The shared library's `ttn/dynamics.py` has gradient flow, GD and SGD; implement other optimizers in your
  folder (torch.optim for Adam/RMSProp on the flat parameter vector is fine).
