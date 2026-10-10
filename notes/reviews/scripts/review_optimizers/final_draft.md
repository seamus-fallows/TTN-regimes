# Reproduction review: optimizers (B1) round 1 — reviewer A (reproduction + independent check)

## What I ran
- README: `uv run pytest experiments/optimizers` passes (4 tests). `python -m experiments.optimizers.run ...` fails on
  Windows (run.py:124 `mp.get_context("fork")`). I used a scratch driver that imports `run.jobs_from_config/run_id/work`
  unchanged and runs them in a spawn pool; reduced YAMLs are strict subsets of the committed ones, so all 375 runs have
  the committed run ids and were compared row-by-row with the committed `*_summary.csv`.
- Agent code (5 workers): scaling_b4 273 runs (GF, GD, HB, Adam 1e-4/1e-3/1e-2, RMSProp 1e-3, re-gauge, Riemannian
  0.1/1, precond 0.1, ALS, DMRG; 7 sigma x seeds 0-2); later_b8 30 (GF + Adam 1e-3, 5 sigma x 3 seeds); eps_mps8 12
  (sigma 0.03/0.05, lr 1e-3/1e-2, eps 1e-8/1e-30, seeds 0-1); deep 30 (MPS8 GF balanced+fan-in, binary8 fan-in GF,
  Riemannian lr 1/ALS/DMRG at sigma 0.3/0.8); implicit_bias_b4 12 (n = 45); precond_b4 18.
- Independent (no experiments/ code; library only for TTNSpec, fan-in init, to_dense, teacher factors): own dense
  loss/overlaps, own Adam/RMSProp, GF first escape located exactly with scipy LSODA + event function; own Hessian,
  balance, bond ranks; own finite-sample data.

## Numbers (committed vs reproduced with agent code vs independent)
| claim | quantity | committed | reproduced (agent code) | independent | verdict |
|---|---|---|---|---|---|
| 1 | binary4 GF p_first (7 sigma 0.002-0.128) | 1.007 +- 0.001 (6 seeds) | TBD (3 seeds) | 1.006 +- 0.001 (3 seeds) | confirmed |
| 1 | binary4 Adam lr 1e-3 / 1e-4 p_first | 0.009 +- 0.009 / 0.012 +- 0.003 | TBD | -0.004 +- 0.010 / 0.012 +- 0.005 | confirmed |
| 1 | binary4 RMSProp 1e-3 p_first | 0.001 +- 0.019 | TBD | -0.005 +- 0.013 | confirmed |
| 1 | lr n_1 (lr 1e-3 / 1e-4) | 0.180 / 0.168 | | 0.177 / 0.169 | confirmed |
| 1 | binary8 GF p_first (0.25-0.8) | 4.76 +- 0.06 | 4.96 on 0.25-0.5 | 4.74 +- 0.05 (same inits, 6 seeds) | confirmed (5 at sigma <= 0.5) |
| 1 | binary8 Adam 1e-3 / 1e-2 p_first | 0.049 +- 0.022 / 0.136 +- 0.055 | TBD | 0.037 +- 0.019 / 0.145 +- 0.033 | confirmed; lr 1e-2 small but nonzero |
| 2 | eps law ratio, sigma <= 0.03 | 0.989-1.053 | TBD | exact T_GF: 0.999-1.000 (lr 1e-3), 1.001-1.006 (lr 1e-2, s <= 0.02), 1.010-1.035 (lr 1e-2, s = 0.03); fan-in 0.999/1.000 | confirmed, tighter |
| 2 | eps law on binary4 (K=3) | sigma_eps = 2e-4 stated | - | law holds (0.95-1.10; 0.998 at 1e-9) but plateau visible only below ~1e-8 | crossover formula not general |
| 3 | ALS/DMRG/Riemannian lr 1 sigma-independence | identical | TBD | - | confirmed |
| 3 | Riemannian lr 0.1, deep, sigma 0.3 vs 0.8 | (data in deep_summary) | - | - | 10-55% different event times |
| 3 | Gram precond sigma-free | 'sigma-free' | lr 0.01: 1673/1461 steps (0.002) vs 502/526 (0.128); 2 budget at 0.016 | - | not sigma-free at lr 0.01 (p = 0.69) |
| 4 | GD / HB first escape vs GF | GD 1.000-1.010, HB 1.007-1.185 (median by sigma) | TBD | - | GD yes; HB within 1-19% |
| 5 | lambda_max / pattern rule (binary4) | GF 1.000; Adam 1.09-1.94 | - | own Hessian TBD | confirmed |
| 5 | rank-4 endpoint at n = 45 | GF 1/12, Adam 1e-3 8/12 | identical per run | own data: GF 1/6, Adam 1/6; agent data + own code TBD | not robust |
| 6 | fan-in GF stall MPS8 | 0/12 converged | seeds 0,1 stall; balanced converge | tight-tol (atol 1e-30) same collapse | confirmed |

## Discrepancies and likely causes
1. Claim 5 (finite samples): rank-4 endpoints of Adam are dataset-specific. My code on the agent's data draws reproduces
   the committed endpoints exactly (ranks and population loss to 4 digits), but on 3 fresh data draws (6 inits) GF and Adam
   both end rank-4 in 1/6 and Adam's population loss is lower than GF's in 4/6. Committed: GF 1/12 vs Adam 8/12 on 6
   datasets. And at n <= 45 the committed population losses favour Adam lr 1e-2 / RMSProp over GF (medians 2.7 / 3.9 vs
   8.0), so "worse generalization at n = 45-60" holds only at n = 60.
2. GF event times: committed values are interpolations on a fixed output grid (+-1.7% on deep trees); exact event location
   moves them by up to 1.7%. This sets the 0.97-1.05 spread of the eps-law test; with exact times the law holds to 0.1%.
3. Heavy ball: first escape 1-19% later than GF (ratio grows with sigma), p = 0.970 vs 1.007 - the agent's "fixed delay"
   explanation (~0.18 time units) does not fit the 1-2.4 time-unit delays; likely the momentum discretization error near the
   escape (lr beta/(1-beta)^2 = 0.18 per unit rate vs GD 0.01).
4. Adam lr 1e-2 on binary8: 10-step fine trace mis-times escapes at 15-25 steps by -7%..+18%; with step resolution p = 0.145 +- 0.033
   (all 42 runs) vs committed 0.136 +- 0.055 (27 runs) - same value, now resolved, small but nonzero.
5. Riemannian GD lr 0.1 on deep trees is not sigma-independent at sigma = 0.3 vs 0.8 (only lr 1 is exact).
6. Gram preconditioning (scale-free damping) is sigma-dependent at lr 0.01 (p = 0.69 +- 0.11; theory itself predicts
   ln(1/sigma)/lr); "sigma-free" holds only for lr 0.1/0.5 and with an absolute damping floor.
7. sigma_eps formula: the GF plateau under Adam only becomes visible where (eps/lr) T_GF(sigma) ~ n_sign (lr-dependent);
   on binary4 that is sigma ~ 1e-8, not sigma_eps = 2e-4.

## Bugs / issues in the code used
1. run.py:124 mp.get_context("fork") -> README run command fails on Windows; default --out /root/... is mkdir-ed before --dry.
2. configs/scaling_b4.yaml precond entries lack `blowup: 1.0e300`: all lr 0.1 (42/42) and 33/42 lr 0.01 runs stop
   'diverged' after 1 step (reproduced); their interpolated 'orders' feed order_agreement_with_gf.csv and the REPORT's
   "33%" GF-order agreement for Gram-preconditioned GD on binary4.
3. plots.per_seed_fits: log-law fitted in linear T, scored in ln T with 1e-300 clipping -> "rms 370" artifact (notes 2.1).
4. GF event times from a fixed output grid (+-1.7% on deep trees) - limits the eps-law precision claim; (the current,
   uncommitted library Recorder 'fine rows' fix this).
5. Adam lr 1e-2 binary8: 10-step fine trace too coarse for 15-25-step escapes (flagged, but resolved ones still +-5%).
6. fig_implicit_bias docstring says non-converged runs are excluded from the rank panel; code counts them (RMSProp).
7. adam_escape.png right panel lacks the eps 1e-30 curves and the dashed prediction its title promises (regenerated: TBD).
8. README run count for scaling_b4 (672 vs 798).
