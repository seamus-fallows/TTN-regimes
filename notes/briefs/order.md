# Brief: `order` — which component is learned first, and how random is the order? (Q4)

Read `notes/briefs/README.md` first (instructions for all agents).

## Task
Measure and explain the order in which a TTN learns teacher components from small random
initialization, and the probability that it learns the largest component first.

## Goal
A quantitative answer to: is incremental learning in TTNs "greedy" (largest component first, as in
deep linear networks) or decided by the initialization, and on what does it depend? Plus a test of
the "wrong order is corrected through a strict saddle" hypothesis (H1b).

## Context
- Q4, hypotheses H1 and H1b in `PLAN.md`; theory in `notes/theory_escape.md` (local maxima of
  g = <W, T> on the sphere, race model, basin picture) and `notes/theory_saddles.md` (piggyback
  directions, H1b).
- Known: for odeco tensors every component is an attracting fixed point of the tensor power method
  (Anandkumar et al. 2014); small-init gradient flow of homogeneous models converges in direction
  to a KKT point before growing (Kumar & Haupt). See `notes/literature.md`.
- Other agents: `s2s-core` (scaling of plateau times), `locality` (where components live in the
  tree), `regime-map`, `data-teacher`, `optim-samples`.

## What to do
1. **Order statistics.** Odeco teachers with two components, weights (1, rho), rho in
   {0.5, 0.7, 0.8, 0.9, 0.95, 0.99}; trees binary N = 4, 8, MPS N = 4, 6, 8 (K from 3 to 8); chi in
   {2, 4, 8}; d in {2, 4}; small sigma (deep in the S2S regime); >= 200 seeds per setting (use the
   leading-order dynamics or short runs where the first escape suffices; justify any shortcut by
   checking agreement with full gradient flow on a subset). Report P(larger first) with binomial
   error bars vs rho, K, chi, d.
2. **Theory comparison.** Compare with the theory note's race model / basin predictions. Control:
   a matrix-like teacher (unique local max), where the order should be greedy.
3. **H1b.** In runs where the smaller component is learned first: does the TTN then pass through a
   strict saddle (gauge-projected Hessian negative eigenvalue ~ q(s_learned - s_new)) and rotate to
   the larger component in time ~ log(1/sigma), before adding the second rank? Show the plateau
   sequence (loss levels and overlaps) and the Hessian spectrum at that plateau; scaling of that
   correction time with sigma.
4. Three or more components: is the full order a random permutation with weights favouring large
   components? Quantify (e.g. Kendall tau with the greedy order).

## Deliverables and success criteria
- `experiments/order/` and `notes/order.md`. Figures: P(greedy) vs rho for each K (and chi, d);
  theory overlay; H1b example run with loss/overlaps/Hessian; correction-time scaling.
- Clear statement of whether H1 and H1b hold, with status tags.

## Constraints
- Interim report after step 1 (~3 h of work). Raw outputs to `/root/outputs/order/`.
