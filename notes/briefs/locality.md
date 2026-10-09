# Brief: `locality` — does the escape time depend on where a component lives in the tree? (Q3)

Read `notes/briefs/README.md` first (instructions for all agents).

## Task
Test hypotheses H2 and H3 of `PLAN.md` and map out how the saddle-to-saddle sequence of a TTN depends
on the tree topology relative to the teacher's structure.

## Goal
A table of predicted vs measured escape-time exponents across trees and teachers, Hessian evidence
for the saddle types, and a clean demonstration that the same teacher is learned through different
saddle sequences on different trees. This is potentially the main new result of the project.

## Context
- Q3, H2, H3, H1b in `PLAN.md`. Hypothesis H2: after some components are learned, the next escape
  needs k cores to acquire new directions, where k is the number of cores in the Steiner subtree of
  the cores carrying the sites at which the new component differs from what is learned (k = 1 if one
  core). Predicted escape time: no plateau for k = 1; ~ log(1/sigma)/lambda for k = 2 (strict
  saddle); ~ sigma^{-(k-2)} for k >= 3 (higher-order saddle).
- Theory: `notes/theory_saddles.md` (being finished; you will be told when it is on main).
- A small local pilot is running (`experiments/pilot_h2_locality`, two-component pattern teachers
  on MPS and binary trees with N = 4); its results will be sent to you. Do not wait for it.
- Library: `ttn.teachers.pattern_teacher` (component r uses basis vector patterns[r][i] at site i),
  `ttn.tree` (custom trees from parent arrays; `Tree.steiner_nodes`), diagnostics for events,
  bond spectra and gauge-projected Hessians.
- Other agents: `s2s-core` (first-escape scaling, odeco phenomenology), `order` (learning order and
  H1b), `regime-map`, `data-teacher`, `optim-samples`.

## What to do
1. **Exponent table.** Two-component pattern teachers (weights e.g. 1.0 and 0.5) where the second
   component differs from the first on a chosen site set S: adjacent pair, distant pair, a triple,
   all sites. Trees: MPS N = 4, 6, 8; binary N = 4, 8; HT N = 4, 8; one custom tree (e.g. a root with
   three children, or a tree whose root sits mid-chain). For each (tree, S): predicted k from the
   Steiner rule; measured exponent of t_2 - t_1 vs sigma (5+ sigma values over >= 1 decade, 5+
   seeds); for k = 2 test the log law ((t_2 - t_1) linear in log(1/sigma)) and its slope 1/lambda.
2. **Saddle type.** Gauge-projected Hessian at the plateau midpoint between events: negative
   eigenvalue (value vs predicted lambda) for k = 2; none for k >= 3; no plateau for k = 1.
3. **Same teacher, different trees.** Pick 2-3 teachers with nested structure (3-4 components that
   differ from each other on nested site sets) and show, side by side for MPS, binary and HT
   students, the loss curves, bond-rank steps and plateau scalings, with the predicted sequences.
4. **Bond spectra at events.** Which edges' singular values turn on at each event; compare with the
   set of edges whose rank must increase.
5. Robustness: chi = exact teacher rank vs overparameterized (2x, 4x); d = 2 vs 4; weight ratio.

## Deliverables and success criteria
- `experiments/locality/` and `notes/locality.md`. The exponent table (predicted k, predicted and
  fitted exponent with error, log-law slope for k = 2), Hessian figure, the side-by-side trees
  figure, bond-spectra-at-events figure.
- A clear verdict on H2 and H3 with status tags, and the precise rule that the data support if it
  differs from H2.

## Constraints
- Interim report after step 1 on MPS and binary N = 4 (~3 h). Raw outputs to `/root/outputs/locality/`.
