# Claims audit: order (Q4) - working notes part 2 (reproductions)

## H1b scope check (data/review_order/claims_audit_h1b_scope.jsonl; agent's saddle_init + projected_hessian)
rho = 0.8, d = 2, chi = 4, exact wrong saddle (sigma = 0) and GF from saddle + 1e-4 (3 seeds):
| case | lam_min | rotation pred -(s1-s2)q^(K-2) | add pred -s1 q^(K-2) | GF |
|---|---|---|---|---|
| MPS4 adjacent cores 1,2 (k=2) | -0.89443 (x3), then -0.17889 | -0.17889 | -0.89443 | a1 up at t~10-11, a2 stays 1.0000: ADDED, no rotation |
| MPS4 separated 0,2 (k=3) | -0.17889 | -0.17889 | - | rotation (a2 -> 1e-12), relearned |
| MPS6 ends 0,5 (k=6) | -0.17235 | -0.17235 | - | rotation; a2 -> 1e-25 -> gf_trajectory dead-floor STOP at t~200 |
| MPS6 adjacent 2,3 (k=2) | -0.86177 (x3) | -0.17235 | -0.86177 | added, a2 stays 1 |
| binary4 odeco (k=3) | -0.18566 | -0.18566 | - | rotation, relearned |
=> H1b-with-rotation only for m = 2 AND k >= 3 (non-adjacent), as theory_saddles 4.4 case 2 says;
   adjacent m = 2 pairs: strict saddle with lam = -s1 s2^((K-2)/K) (multiplicity chi_e - 1), larger
   component added in a new bond index, smaller kept. Agent's "H1b holds for m = 2" too broad.
=> Separated-pair formula verified beyond K = 3 (K = 4, 6), exact to 1e-15.
=> Dead-floor rule (|a_r| < 1e-25) fires during the normal forget phase on MPS6 ends (k = 6).
