# Claims audit: order (Q4) - part 3: chi-scan refit and stall reproduction

## chi-scan refit (results/chi_scan.csv)
- joint all chi>=2: a = 0.608 +- 0.029, b = 0.530 +- 0.012 (agent's numbers reproduced)
- MPS only: a = 0.577 +- 0.023; binary8 mean residual +0.060 (log), MPS6 -0.027, MPS8 -0.033
- MPS K-exponent per chi: 0.657 (chi 1), 0.655 (2), 0.594 (4), 0.583 (8), 0.523 (16), 0.530 (32)
  -> not separable; ~sqrt(K-2) only at large chi.

## Stall rule reproduction (data/review_order/stall_mps6_R3*.{log,jsonl}; MPS6 R=3 rho 0.8 sigma 0.2,
## d = chi = 4, agent's LSODA + exact Hessian, NO stall stop, t_max 1e9)
- seed 0, 4, 5, 6: converged, no |a|<1e-25 event; third event late (6.9e5 - 3.5e6).
- seed 3: comps 2, 3 learned (2612, 32175); comp 1 (LARGEST) |a_1| < 1e-25 at t = 3.68e4, never learned
  by t = 1e9 -> agent's stall label confirmed for this seed (a stall of the LARGEST component; such runs
  are excluded from the R = 3 order statistics -> selection against non-greedy orders).
- (more seeds pending)
- seed 2: comps 3 (2034), 1 (3388) learned; comp 2 never learned by 1e9 and |a_2| never < 1e-25
  (agent would label t_max/'stalled'): slow/pinned route, not a dead one.
- seed 1: comps 1 (2992), 2 (3330) learned; comp 3 never by 1e9; |a_3| < 1e-25 only at t = 2.2e8.
=> seeds 0-6: 4 converge, 3 never learn the third component by t = 1e9 (43%; agent 54%); the no-stop
   integration agrees with the agent's stalled/t_max label on 3/3. The stuck component is the largest
   (seed 3), middle (seed 2) or smallest (seed 1): exclusions remove both greedy and non-greedy runs.
- seed 10: converged; LARGEST component learned last at 7.4e6 (others 1458, 2162).
- seed 11: comps 3 (1950), 1 (8137) learned; comp 2 |a| < 1e-25 at 1.18e4, never learned by 1e9.
FINAL (seeds 0-6, 10, 11; 7-9 unfinished when stopped): 5 converge, 4 stuck (44%; agent 54%).
dead25 fired in 3 runs, none recovered by t = 1e9; 1 stuck run never crossed 1e-25 (pinned/slow).
Files moved to data/review_order/claims_audit/.
