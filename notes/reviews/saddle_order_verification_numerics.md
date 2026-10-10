# Numerical verification of `notes/theory_saddle_order.md` (v1, 2026-10-10)

Verifier: independent numerics agent, 2026-10-10. Code (written from scratch, imports only the tree
topologies from `ttn.tree`; contraction checked against `ttn.network.to_dense` to 9e-16):
`experiments/theory_checks/saddle_order_verify/` (`vttn.py`, `mps_fast.py`, `order_curves.py`,
`fixed_curve.py`, `gf_origin.py`, `analyze_gf.py`; JSON in `results/`). The teammate's scripts in
`../saddle_order/` were not run or imported; their JSON was read only for the comparison below.

## Verdict

- **Saddle orders and constants (S8, S10): confirmed.** 12 instances (k* = 2..6 on mps4/5/6, binary4/6,
  ht4; trees and patterns chosen independently of theirs), incl. the counterexample: the measured order
  equals the theory's prediction in every case (slope vs distance within 5e-4 of the integer), and every
  measured leading constant equals the analytic value to 0.1-0.4 %. Counterexample (MPS6 pair-star 0.8):
  order **4** (slope 3.9997), constant **-0.0350** (pred. (1 - 2 0.64)/8 = -0.0350); control (0.6):
  order **5**, constant -0.01077 (single route -0.01073).
- **New check, supports the mechanism:** the order-4 descent needs a *curved* path. Following the optimal
  point to smaller scales with new entries ~tau and old entries ~tau^2 gives slope 4.000 (4 digits);
  the straight line through the same point, and the new-entries-only line, turn positive (L > L_S) at small
  tau. For the control all three paths give slope 5 (pure route).
- **Gradient flow from small fan-in init (S9): confirmed with caveats.** In the seeds that learn q first,
  the q-plateau length goes as sigma^-2 (local exponents 2.08, 2.03, 2.01, 2.005 for sigma = 0.01 -> 0.0025),
  not sigma^-3, and the escape is along 110000 with old rotations (amplitudes 0.085 / 0.065 / 0.065,
  same numbers as the note). Caveats: (i) the sigma^-2 law is asymptotic: at sigma >= 0.035 there is no
  q plateau at all, and the exponent is still 2.9 -> 2.15 for sigma = 0.025 -> 0.01; (ii) from origin init,
  3 of the 5 seeds that finished learn the residual pair first and never visit theta_S; (iii) only 2
  q-first seeds finished within the budget (seeds 2, 7; runs for 7 more seeds were still going, see end).
- **Where the flow goes next (open in the note):** after the generalized-piggyback escape the network
  **forgets q**: it goes to W ~ 0.8 (e_110001 + e_110010), L = 0.500 = s_q^2/2, sits there, then relearns
  q and converges to L = 0. The relearning plateau is long (sigma^-3 in seed 7, ~sigma^-4 in seed 2).
- **Control teacher (0.6): consistent with the note, but the "sigma^-3" law holds only in some seeds.**
  12/12 seeds learn q first; 2/12 are trapped (r1, r2 amplitudes ~1e-38 at t = 3e11: dead routes);
  10/12 escape, always through r2 (k* = 5 route, r1 = 0 at escape), then converge to L = 0 within ~50 time
  units. Local exponents tend to 3 in about 4 seeds and to ~4 in about 6 (e.g. 3.99, 4.00, 3.96). No
  control seed shows exponent 2. The note reports only "3.15 -> 3.54" for one seed; an exponent of ~4 in
  half of the seeds is a stronger effect than the note suggests and is not explained by the saddle order
  (the r1 route has k = 6, i.e. sigma^-4, but r1 stays at 0, so it is not that route).

## 1. Saddle order at theta_S (task 1)

**Method (`order_curves.py`).** For each t, minimize L - L_S over degree-2 curves
theta_S + t a1 + t^2 a2 with ||a1|| = 1, ||a2|| < 3, a1 and a2 orthogonal to the gauge-orbit tangent at
theta_S (numerical basis from the GL(chi) generators of every bond; rank 15 of 20 at the MPS6 theta_S) and
a2 orthogonal to a1. L-BFGS (scipy), 12 random starts per t (full / new-entries-only / random 2-3-core
cluster / new + old) plus continuation from the previous t; t = 0.2 -> 0.0125 (9 values); float64;
L - L_S = 1/2 ||Delta||^2 - <R, Delta>, Delta = W - W_S. Order = local slope of log|m| against log of the
actual distance r = ||theta - theta_S|| (against log t the degree-2 family is stretched by
r = t (1 + O(t^2)), which biases slopes upward by ~0.01 at the smallest t). The counterexample was also run
with degree-1 curves (= plain search on the sphere). Hessian at theta_S (MPS6 pair-star, both weights):
PSD, lambda_min = -1.2e-15 (roundoff), 33 zero modes (15 gauge), lambda_max = 6. k* from my own Steiner
computation: k(r1 = 110001) = 6, k(r2 = 110010) = 5, so k* = 5 (as claimed).

| case (s_q = 1, d = chi = 2) | k* | theory order | slope vs r (smallest t) | m/t^p measured | predicted constant | theirs (slope, constant) |
|---|---|---|---|---|---|---|
| mps4 1100 (0.8) | 2 | 2 | 1.9999 | -0.4006 | -0.4000 | |
| mps4 1010 (0.8) | 3 | 3 | 3.0000 | -0.1543 | -0.1540 | |
| binary4 1010 (0.8) | 3 | 3 | 3.0000 | -0.1543 | -0.1540 | |
| ht4 1100 (0.8) | 3 | 3 | 3.0000 | -0.1543 | -0.1540 | |
| binary6 100001 (0.8) | 4 | 4 | 4.0000 | -0.05014 | -0.05000 | |
| mps5 10001 (0.8) | 5 | 5 | 5.0000 | -0.01436 | -0.01431 | |
| ht4 1010 (0.8) | 5 | 5 | 5.0000 | -0.01436 | -0.01431 | |
| binary6 100100 (0.8) | 5 | 5 | 5.0000 | -0.01436 | -0.01431 | |
| mps6 pair-star (0.6, 0.6), rho = 0.85 | 5 | 5 | 5.0000 | -0.01077 | -0.01073 (route of r2) | 5.000, -0.01073 |
| **mps6 pair-star (0.8, 0.8), rho = 1.13** | 5 | **4** | **3.9997** (deg 1: 3.9997) | **-0.03509** (deg 1: -0.03500) | -0.0350 | 4.000, -0.0350 |
| mps6 110001 (1.5), wrong order | 6 | 4 | 3.9995 | -0.1567 | -0.15625 | 3.999, -0.1562 |
| mps5 10001 (1.5), wrong order | 5 | 2 | 1.9999 | -0.2503 | -0.2500 | 2.000, -0.2500 |

Predicted constants: single route s_r k^(-k/2) (a = 1); pair-star (1 - s1^2 - s2^2)/8; single wrong-order
residual with a degree-2 split (1 - s^2)/8; classical piggyback -(s - 1)/2. The 0.1-0.4 % excess of the
measured constants is the O(t^2) stretch of the degree-2 family plus O(t) corrections at t = 0.0125 (the
degree-1 run of the counterexample gives -0.034997). At the smallest t, 6 to 13 of the 13 starts land within
1 % of the best value; no start ever found a faster descent than the predicted law (a minimum found by
local search is only an upper bound on m(t), so this is evidence, not proof, of the lower bound).
Their blank cells: they ran the same kinds of cases with other names/weights (their groups A-C); I compare
those to the same analytic constants instead.

**One curve followed to smaller scales (`fixed_curve.py`, t0 = 0.02 -> 0.005).** Split the optimal
displacement at t0 into new entries n and old entries o (pair-star 0.8: ||o|| = 2.27e-4 = sqrt(2) 0.8 t0^2/2,
i.e. the predicted old rotations z = s y on cores 4 and 5):

| path | pair-star 0.8: L - L_S at tau = 0.02, 0.014, 0.01, 0.007, 0.005 | slopes |
|---|---|---|
| curved: (tau/t0) n + (tau/t0)^2 o | -5.64e-9, -1.35e-9, -3.52e-10, -8.46e-11, -2.20e-11 | 4.000 x 4 |
| straight: (tau/t0)(n + o) | -5.64e-9, -2.18e-10, +1.26e-9, +1.25e-9, +8.84e-10 | sign change |
| new entries only: (tau/t0) n | +2.01e-8, +4.84e-9, +1.26e-9, +3.02e-10, +7.87e-11 | positive (cost y^2/2) |

So the order-4 descent is exactly the note's curve (pair cluster on cores 0, 1 at amplitude tau, old
rotations of cores 4, 5 at tau^2); no straight line through theta_S descends at order 4. Control (0.6): all
three paths give slope 5.000 (the old part is negligible: pure r2 route).

## 2. Gradient flow on the MPS6 pair-star from small fan-in init (task 2)

**Setup (`gf_origin.py`).** Unlike the note (start at theta_S + sigma xi), every run starts at the
origin-scale fan-in init G_v = sigma Z_v / sqrt(F_v) (same Z at every sigma for a given seed; init seeds
12345 + seed; the teacher is deterministic). LSODA, rtol 1e-10, atol 1e-14, hand-written MPS gradient
(checked against autograd to 4e-17). t_q = time at which W[q] crosses 1/2; escape t_esc = first time
L < L_S - 1e-3 (the note's threshold); plateau length dt = t_esc - t_q, only for runs where q is learned
first. sigma = 0.1 -> 0.01 (8 values; down to 0.0025 for seed 7), seeds 0-11 per teacher.

**Counterexample (0.8, 0.8).** Finished: seeds 0, 1, 2, 7, 8. Seeds 0, 1, 8 learn the residual pair first
(L = 0.5 state) and never visit theta_S; seeds 2 and 7 learn q first.

| sigma | 0.035 | 0.025 | 0.0175 | 0.0125 | 0.01 | 0.007 | 0.005 | 0.0035 | 0.0025 |
|---|---|---|---|---|---|---|---|---|---|
| dt, seed 7 | 2.62 | 21.2 | 59.9 | 131.0 | 211.9 | 444.3 | 880.2 | 1804.8 | 3543.8 |
| local exponent, seed 7 | | 6.2 | 2.91 | 2.33 | 2.15 | 2.08 | 2.03 | 2.01 | **2.005** |
| dt, seed 2 | 12.5 | 40.5 | 98.1 | | | | | | |
| local exponent, seed 2 | | 3.50 | 2.48 | | | | | | |

For sigma >= 0.05 the escape happens during the rise of q (dt ~ 1, no plateau). Seed 7 converges to the
note's sigma^-2 (route law would be sigma^-3); seed 2 follows the same pre-asymptotic curve as seed 7 at the
same sigma (3.5, 2.5 vs 2.9, 2.3) but its small-sigma runs did not finish in the budget. The note's runs
from theta_S + sigma xi gave 2.15 -> 2.03, 2.15 -> 2.04, 1.98 -> 2.00 already at sigma = 0.04 -> 0.005: from
the origin the asymptotic regime starts at a smaller sigma (the dormant amplitudes at theta_S are set by
the first escape, not directly by sigma).

*Escape direction* (W at t_esc, all 11 q-first runs with dt > 5): W[110000] = +-0.085 (0.0849-0.0850),
W[000001] = W[000010] = +-0.065, W[110001] = W[110010] = 0.0056, W[000011] = 0.004, all other non-q entries
< 1e-4 (seed 7, sigma = 0.0025): the non-residual
pattern 110000 plus old rotations of cores 4, 5, as the note says (same 0.085 amplitude).

*Where it goes next (new; open in the note).* Within ~1.5 dt after the escape the flow reaches
W = 0.8 (e_110001 + e_110010) with W[q] = 0.000 and L = 0.5000: **q is forgotten** and the residual pair
is learned instead. It stays there, then relearns q and converges to L = 0 (all finished runs). Length of
the L = 0.5 plateau: seed 7: 4.7e6, 3.8e7, 3.0e8 at sigma = 0.01, 0.005, 0.0025 (exponent 3.0); seed 2:
2.8e8, 1.1e9, 5.0e9 at sigma = 0.035, 0.025, 0.0175 (exponent ~4.2). So after this escape the plateau
sequence is: q learned (L_S = 0.64), q forgotten / pair learned (0.5), everything learned (0); the second
plateau is much longer than the first one.

**Control (0.6, 0.6).** All 12 seeds learn q first. dt local exponents at the three smallest sigma
(0.0175 -> 0.0125 -> 0.01):

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exponents | 2.99, 3.00 | trapped | 3.98, 3.99 | 1.53, 3.02 | 3.32, 3.40 | 3.98, 4.00 | trapped | 3.94, 3.96 | 2.87, 2.90 | 3.63, 3.91 | 3.74, 3.78 | 3.10, 3.08 |

At escape W[110010] = 0.0017 and |W[110001]| < 1e-5 in all 80 escaping runs: the escape is always the
k* = 5 route of r2, never the pair cluster (|W[110000]| <= 0.003) and never r1; after it the flow converges
to L = 0 within <= 70 time units. Trapped seeds 1 and 6 (every sigma): L = L_S = 0.36 to t = 3e11 with
W[r1], W[r2] ~ 1e-38: dead routes (amplitudes decay far below the float64 floor; no roundoff seeding
observed). The note's control row (3.15 -> 3.54, 2.86 -> 2.97, one trapped seed) agrees in kind, but the
"anomalous" exponent is common: it tends to ~4 in seeds 2, 5, 7 and is still rising past 3.4 in 4, 9, 10.
Neither the saddle order (5) nor the r1 route (k = 6, unused) explains it; it needs its own explanation
(R12 pinned routes / conserved quantities are the note's guess, untested here). No control seed has
exponent near 2, so the dichotomy rho > s_q vs rho < s_q is sharp in the dynamics.

## 3. Summary of agreement

| claim in the note | note's numbers | mine | status |
|---|---|---|---|
| pair-star 0.8: k* = 5, Hessian PSD, order 4, constant -0.0350 | slope 4.000, -0.0350 | 3.9997, -0.0350; PSD (lambda_min = -1e-15) | confirmed |
| order = prediction on covered cases (k* = 2..7) | 32 instances | 12 instances, k* = 2..6, independent code | confirmed (k* = 7 not re-run) |
| t_esc ~ sigma^-2 for the pair-star | 2.03-2.04 (3 seeds, from theta_S + sigma xi) | 2.005 (seed 7, from origin; seed 2 pre-asymptotic) | confirmed asymptotically; from origin needs sigma <= 0.005 |
| escape along 110000 + old rotations, amplitude 0.085 | 0.085 | 0.085 / 0.065 | confirmed |
| next plateau "not r learned (not studied)" | - | q forgotten, pair learned (L = 0.5), then q relearned | new result |
| control: route escape (sigma^-3) or trapped | 3 seeds | 10/12 escape via r2, exponents 3 or ~4 (seed-dependent), 2/12 trapped | consistent; the sigma^-3 law is not generic for this teacher |

**Verdict.** The numerical evidence of the note is reproduced by independent code: the saddle orders
(including the order-4 counterexample with k* = 5 and a PSD Hessian) and the leading constants are right,
and gradient flow on the counterexample escapes the q plateau with t_esc ~ sigma^-2 instead of the route law.
Suggested edits to the note: (1) state that from small init only part of the seeds visit theta_S, and the
sigma^-2 law sets in only for sigma <~ 0.005; (2) add the post-escape behaviour (q forgotten, then
relearned); (3) say that the control's exponent is ~4 in about half the seeds, so "route law sigma^-3" holds
only seed-wise for that teacher.

Run status at writing: 0.8-teacher runs for seeds 3-6, 9-11 and seed 2 at sigma <= 0.0125 were still
running (processes `gf_origin.py 0.8 ...`; they write further `results/gf_s0.8_*.json`; rerun
`analyze_gf.py` to update). They were not used above.

## How to reproduce (from the repo root; each order case 15-120 s, each GF run 1-10 s except long post-escape runs)

    uv run python experiments/theory_checks/saddle_order_verify/test_vttn.py
    uv run python experiments/theory_checks/saddle_order_verify/order_curves.py mps6_pairstar_08:2 mps6_pairstar_08:1 mps6_pairstar_06:2 ...
    uv run python experiments/theory_checks/saddle_order_verify/fixed_curve.py mps6_pairstar_08 mps6_pairstar_06
    uv run python experiments/theory_checks/saddle_order_verify/gf_origin.py 0.8 0-11 0.1,0.07,0.05,0.035,0.025,0.0175,0.0125,0.01
    uv run python experiments/theory_checks/saddle_order_verify/gf_origin.py 0.6 0-11 0.1,0.07,0.05,0.035,0.025,0.0175,0.0125,0.01
    uv run python experiments/theory_checks/saddle_order_verify/analyze_gf.py

Case names are in `order_curves.CASES`. Known limitation of `gf_origin.py`: runs integrate to convergence
(L < 1e-9) or t = 3e3 sigma^-4, so 0.8-teacher runs that forget and relearn q spend most of their cost on
the long L = 0.5 plateau; a terminal event at L = 0.5 would make them cheap.
