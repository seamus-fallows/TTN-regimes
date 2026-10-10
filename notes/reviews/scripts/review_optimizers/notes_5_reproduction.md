# Part 5: reproduction with the agent's code (per-run comparison with committed rows, same run ids)

Columns compared per run: t_first, t_0..t_2, gap_0/1, n_steps, loss_end, pop_loss_end, balance, t_loss_0.001,
order, converged, eff_ranks_end, stop_reason, resolved flags, s2s. ("0.0" = |rel| < 5e-5.)

- implicit_bias_b4 (n = 45, sigma 0.01/0.1, seeds 0-2; GF t_max 1e6 + Adam lr 1e-3; 9 of 12 checked so far):
  events, orders, convergence, eff ranks, stop reasons identical; n_steps at the tolerance crossing within
  0.3-4%, final loss (~1e-8 L0 level) within 1-7% (platform floating point).
- precond_b4 (18 runs: lr 0.01/0.1 scale-free, lr 0.1 abs 1e-2; sigma 0.002/0.016/0.128; seeds 0-1):
  overshoot maxima identical to all printed digits (1e21-1e29 L0), orders/stop reasons identical; steps to
  convergence differ by up to 15 steps (161 vs 176) after 1e17-1e29 overshoots (roundoff amplification);
  lr 0.01 sigma 0.016 both seeds hit the 5e4 budget (as committed).
- later_b8 GF (12 runs so far, sigma 0.25-0.5): identical events/orders; per-seed p(t_first) on sigma 0.25-0.5
  = 4.96 +- 0.001 (theory 5); committed 4.76 +- 0.06 over 0.25-0.8 (sigma >= 0.6 points are pre-asymptotic).
- deep GF (agent code): MPS8 odeco sigma 0.3 balanced seeds 0, 1 converged; fan-in seeds 0, 1 'completed' (stalled
  at t_max = 2e6, loss 0.5); binary8 fan-in seed 1 stalled, seed 3 converged - same outcomes as committed.
  Wall here 33-77 s per run (committed 120-330 s on the remote machine).
- scaling_b4 precond lr 0.1 (no blowup key): reproduced 'diverged' after 1 step at every sigma (bug 2).
- eps_mps8 (12 runs, agent code): n_1 identical to committed for all 8 eps 1e-8 runs (9972.4493, 44672.2934, 568.3374,
  2187.3888, 1049.2445, 4536.0897, 86.9601, 259.9123). With the reproduced GF reference (new-library fine rows ->
  exact T_GF(0.3) = 984.79 / 4453.56 instead of committed 968.04 / 4503.75) the ratios become
  lr 1e-3: sigma 0.03 0.9988 / 1.0000, sigma 0.05 0.953 / 0.988; lr 1e-2: sigma 0.03 1.035 / 1.012, sigma 0.05 1.163 / 1.100
  (committed: 1.016 / 0.989, 0.967 / 0.978, 1.053 / 1.001, 1.178 / 1.090).
- FINAL per-run comparison (all 375 runs): scaling_b4 273 - every discrete method identical to committed (events
  < 5e-5 rel., orders/convergence/eff ranks/stop reasons 100%); GF events within 0.27% (t_first), 0.36% (gaps);
  RMSProp events within 0.26% but stop step differs (hovering runs). later_b8 30 - Adam t_first identical, one Adam
  t_2 off by 3% (long AEoS plateau, roundoff amplification); GF t_first within 0.35%. deep 30 - ALS/DMRG/Riemannian
  identical; GF outcomes identical, t_first within 1.7% (fine rows). eps 12, implicit_bias 12 identical; precond 18
  identical up to <= 15 steps in convergence.
- Exponents on the reproduced subsets equal the committed values restricted to the same runs (e.g. binary4 GF 1.006,
  GD 1.004, HB 0.970, Adam 1e-3 -0.004, 1e-2 0.015, 1e-4 0.011, RMSProp -0.005; binary8 GF 4.68, Adam 0.023).
- GD/HB vs GF (GF clock, same init): GD 1.0002-1.0053 (median by sigma, max 1.010); HB 1.0036-1.1045 (max 1.196).
- Independent Hessian at reproduced endpoints (21 runs each): Adam lambda_max/pattern 1.09-1.94 (means 1.28-1.31),
  RMSProp 1.11-1.56, GF/GD/HB 1.000; balance Adam/RMSProp 0.23-0.50 vs GF/GD/HB <= 0.04.
