# Saddle order at a learned rank-one saddle (theory check)

**Question.** At the rank-constrained critical point theta_S (one learned component q), is the saddle
order (lowest order at which L can decrease along a curve leaving theta_S) equal to
k* = |Steiner(terminal cores of a residual)| (R2 of `notes/theory_saddles.md`)? Theory and full
answer: `notes/theory_saddle_order.md`.

**Short answer.** Yes when the generalized piggyback operator S (sector couplings between core sets of
degree < k*/2) has lambda_max(S) < s_q (Theorem 1, proven). When S exceeds s_q at a level l < k*/2,
the order drops to 2l < k* (Theorem 2, proven for pattern teachers with a spare bond index; exact order
2l* by Theorem 1'). Borderline cases (lambda_max = s_q) are not covered. First counterexamples at k* = 5
(MPS6 "pair-star"). See notes/theory_saddle_order.md v2 and its two verifications in notes/reviews/.

## Method

`saddle_order.py`: m(t) = min over ||delta|| = t of L(theta_S + delta) - L_S, by L-BFGS on the
sphere (12 random starts per t, full / new-entries-only / mostly-old directions, plus continuation
from the previous t), t = 0.3 -> 0.025; float64; L - L_S = 1/2 ||Delta||^2 - <R, Delta> with
Delta = W - W_S (no cancellation). Order = local slope of log|m| vs log t. Pattern teachers in the
standard site basis, q = 0...0, s_q = 1, d = 2, chi = 2 unless stated.

`gp_operator.py`: builds S on the low sets for every level l < k*/2 from a dense residual and returns
the predicted order (Theorems 1-2). `check_predictions.py` prints it for every case.

`random_R.py`: general (non-pattern) residuals, random on all sectors with |Steiner(B)| >= 5, rescaled
to a target lambda_max(S).

`gf_escape.py`: gradient flow (LSODA, rtol 1e-10) from theta_S + sigma xi; escape time
(L - L_S < -1e-3) versus sigma, and the amplitude of the non-residual pattern 1100..0 at escape.

## How to run (light: each case takes 10-60 s on one core)

    uv run python experiments/theory_checks/saddle_order/saddle_order.py all
    uv run python experiments/theory_checks/saddle_order/check_predictions.py
    uv run python experiments/theory_checks/saddle_order/random_R.py rR_mps5_d3_125 mps 5 3 5 1.25 1
    uv run python experiments/theory_checks/saddle_order/gf_escape.py mps6_pairstar_08 mps6_pairstar_06
    uv run python experiments/theory_checks/saddle_order/summarize.py

Results (small JSON) go to `results/`.

## Results

32 instances (trees mps4-6, binary4/6, ht4; k* = 2..7; one or several residuals; general random R).
The measured order equals the predicted one in every case where the theory makes a prediction (slopes
within 0.003 of the integer at the smallest t; one near-critical case converges slowly, see the note),
and the measured constants equal the analytic leading-order optima. Table: section 5 of the note.
Gradient flow on the MPS6 pair-star escapes with t_esc ~ sigma^-2 (order 4) instead of the route law
sigma^-3 (k* = 5): local exponents 2.15 -> 2.03, 2.15 -> 2.04, 1.98 -> 2.00 (3 seeds). The control
(weights 0.6) escapes through the k* = 5 route (exponents 2.86 -> 2.97 and 3.15 -> 3.54) or not at all
(seed 2: routes die, conservation-law trap of theory_saddles R12).
