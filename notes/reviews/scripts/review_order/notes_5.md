# review_order notes (part 5): adjacent vs separated m = 2 wrong-saddle starts (agent h1b.py, new configs)
MPS4 d=2 chi=4, rho=0.8, sigma 1e-3/1e-5, seeds 0-1 (configs/h1b_saddle_mps4adj.yaml, ..sep.yaml here).
- adjacent (cores 1,2; chi_e=4): lam_min at plateau midpoint -0.89442 = -s1 q^(K-2) (pred -0.894427), n_neg=4;
  dt1/dln(1/sigma) = 1.1175 / 1.1174 vs 1/(s1 q^(K-2)) = 1.1180 (rotation law would be 5.590);
  a2 stays 1.00001 after t1 (no forgetting), converges directly, no relearning plateau.
- separated (cores 0,2): lam_min -0.178885 = -(s1-s2) q^(K-2); slope 5.591 / 5.586 vs 5.590;
  a2 falls to 1e-10..1e-16 (forgotten), relearned at t ~ 1.4e3 (1e-3) / 1.4e5 (1e-5).
- h1b_saddle_mps4 rerun (12 rows) identical to committed; |lam_min| ~ sigma^2.00 (1.99-2.03), t1 ~ sigma^-2.0, a2 >= .998.
- committed natural b4: 43/421 wrong-order runs (all sigma = 1e-4) have NaN midpoint lam_min (unresolved; not counted).
- data/review_order/claims_audit/ belongs to another reviewer (not mine); a foreign `tail -f` watches data/review_order.
- Rerun wall time 09:59:54 - 10:33:38 (34 min), <= 4 busy processes.
