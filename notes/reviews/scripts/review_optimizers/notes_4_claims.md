# Part 4: claim-by-claim observations from the committed CSVs (checked against reproduction where rerun)

Claim 4 (GD / heavy ball = GF), committed scaling_b4, first escape in GF clock vs GF, median over 6 seeds:
- GD lr 0.02: ratio 1.000 (sigma 0.002) ... 1.010 (sigma 0.128).
- heavy ball lr 0.002 beta 0.9: ratio 1.007, 1.011, 1.022, 1.041, 1.069, 1.115, 1.185 (sigma 0.002 ... 0.128);
  delay 2.4 -> 1.0 GF time units, i.e. NOT the "fixed delay" of ~1/(1-beta) steps (= 0.18 time units) given
  in notes 2.1 item 1. Likely mechanism: HB's per-step growth error O(lr beta/(1-beta)^2 * lambda) = 0.18 lambda
  near the escape (18x GD's lr/2 error), so the escape phase runs slow; this is why p_first = 0.970 +- 0.002
  (significantly != GF 1.007). Gaps: HB = GF exponents (1.006 / 1.010). Verdict: GD reproduces GF to 1%;
  HB to 1-19% on the first escape (exact orders and gap exponents agree).

Claim 5 (implicit bias), committed implicit_bias_b4: at n = 45 population loss median/mean: GF 7.96/9.50,
Adam 1e-3 8.00/12.48, Adam 1e-2 2.70/5.85, RMSProp 3.87/6.19; at n = 30: GF 18.7/57.5, Adam 1e-2 13.7/15.2,
RMSProp 12.8/14.9. So at n <= 45 Adam lr 1e-2 and RMSProp generalize BETTER than GF in population loss;
"worse generalization than GF at n = 45-60" (notes section 3) is supported only at n = 60 (recovery 10/12 GF vs
9, 6, 8/12). The rank part (rank-4 edge at n = 45: GF 1/12 vs Adam 8-9/12, RMSProp 11/12) is as stated;
RMSProp never reached the training tolerance (all 'budget') so its ranks are not at an interpolant.

Claim 3 (gauge-aware methods): orders per seed are sigma-independent (Riemannian, ALS, DMRG identical across
7 sigma) and never weight-ordered in 6 seeds (P = (5/6)^6 = 0.33 under a uniform random order, so "unrelated
to weights" is consistent but weakly tested). For Riemannian lr 1 / ALS / DMRG the events fall inside the
first step/micro-steps; the "order" is the ranking of a_r after that step (interpolated crossing), not a
temporal sequence. Gram-preconditioned GD with scale-free damping is NOT sigma-free at lr 0.01
(t_loss_0.001 p = 0.69 +- 0.11, 5 seeds, 9/42 non-converged; precond_b4_exponents.csv); sigma-free only with
lr 0.1/0.5 (converged runs) and with the absolute damping floor.

Riemannian GD lr 0.1 (scaling_b4_side figure, sigma 0.008 seed 0): all three a_r follow the same curve
~1 - 0.9^n; the "order" (o201 etc.) is decided by small differences between simultaneous rises. "All at once"
is the finding; "order unrelated to weights" is not meaningful for these methods (rho << 1 by construction).

Committed adam_escape.png right panel: title promises "dashed: eps prediction" and the code also draws eps =
1e-30 curves, but neither the dashed prediction nor any eps 1e-30 curve is visible (legend lists only eps 1e-8
entries). The summary CSV has 96 eps 1e-30 rows, so the committed PNG is stale or was made from a cache without
them (to be checked by regenerating with the agent's fig_adam_escape on my subset).

Committed deep_summary, gauge-invariant methods, seeds 0-1, sigma 0.3 vs 0.8: ALS / DMRG event times identical
to 6 digits (n_micro 11/10 binary8, 7/6 mps8); Riemannian lr 1 identical (5-8 steps); Riemannian lr 0.1 NOT
identical: e.g. binary8 seed 0 t_0 = 15.17 vs 17.32 steps, seed 1 t_2 = 11.86 vs 18.32, mps8 seed 1 t_2 =
16.07 vs 23.03 (10-55% differences; n_steps 68-75). Expected: sigma-independence of Riemannian GD is exact
at lr = 1 (W_1 = HSVD(P_T W*)) and only asymptotic (sigma^K -> 0) for lr < 1; at sigma = 0.8, K = 7-8 the
(1 - lr) W_0 term is not negligible. Notes 2.5 item 6 correctly says "lr 1 ... identical"; the summary
statements ("sigma-independent time") should carry the lr = 1 / small-sigma qualifier.

Gram-preconditioned GD, scale-free damping (committed precond_b4, seeds 0-1): t(L < 1e-3 L0) at lr 0.01 =
1676/1458 steps (sigma 0.002), budget/budget (0.016), 168/200 (0.128) -> sigma-dependent (committed p = 0.69
+- 0.11, 5 seeds; 9/42 not converged). Notes 1.7 itself derives a return time (K-1) ln(1/sigma)/lr, i.e.
log-dependent on sigma. So "Gram-preconditioned GD ... in sigma-independent time" (REPORT summary, claim 3)
holds only for lr 0.1 / 0.5 among converged runs and with the absolute damping floor.
