# Brief: saddle-dyn (remote, launched 2026-10-10 ~15:00 UTC)

Machine: AMD Threadripper PRO 7995WX, 192 vCPUs, 257 GB RAM (Norway), $1.02/h, chosen by the launch guide.
Expected cost $8-12 (about 4 h of compute).

Task: test the dynamical prediction of the saddle-order theorem (notes/theory_saddle_order.md v2;
verifications in notes/reviews/saddle_order_verification_{proof,numerics}.md; summary notes/theory_saddles.md v3).
- Pre-register in notes/saddle_dyn.md (hypotheses, cases, fit windows, decision rules) before running anything.
- H-A (saddle start): from theta_S + eps * random perturbation, escape time ~ eps^-(order-2) with the theorem's
  order; the route law eps^-(k*-2) on control teachers (lambda_max < s_q).
- H-B (small init): runs that visit theta_S show the reduced exponent below the crossover b, the route law above.
- Cases: MPS6 pair-star (k* = 5, order 4) and its control (weights 0.6, order 5); MPS6 110001 wrong order
  (order 4 vs k* = 6); HT4 1111 wrong order (order 6 vs 7); binary8 wrong-order saddle (order 6 vs route law 5,
  b ~ 0.155; sigma down to <= 0.05); two or three controls with order = k*.
- >= 10 perturbation directions / seeds per cell, >= 1.5 decades in the asymptotic range, per-seed fits in the
  registered windows, every run reported (incl. those that never visit theta_S), float64-floor flags.
- float64, one thread per process, LSODA for stiff phases; raw outputs in /root/outputs/saddle_dyn/;
  experiment folder experiments/saddle_dyn/; deliverables notes/saddle_dyn.md, README, agent-runs/saddle-dyn/REPORT.md.
- Budget: <= ~4 h wall-clock compute.
