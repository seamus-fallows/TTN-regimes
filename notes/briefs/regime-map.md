# Brief: `regime-map` — the map of training regimes (Q5)

Read `notes/briefs/README.md` first (instructions for all agents).

## Task
Build phase diagrams of TTN training regimes over the key parameters, using the regime rules of
`notes/diagnostics.md`, and compare the boundaries with theory.

## Goal
The report's "map of regimes": where saddle-to-saddle, rich non-incremental and lazy dynamics occur
as functions of init scale, bond dimension, tree size and core imbalance, with the boundaries
predicted where possible and their sensitivity to the diagnostic thresholds reported.

## Context
- Q5 and H4 in `PLAN.md`; theory in `notes/theory_regimes.md` (tangent kernel at init, lazy
  criterion, S2S separation criterion, large-chi parameterizations, imbalance).
- Known in other models: Chizat, Oyallon & Bach 2019 (lazy training by scaling); Woodworth et al.
  2020 (kernel to rich in diagonal nets, two-copy construction); Geiger et al. 2020 (phase diagram);
  Kunin et al. 2024 and Domine et al. (imbalance decides lazy vs rich in linear nets); Yang & Hu 2021
  (muP). See `notes/literature.md`. Show these in our setup where they carry over.
- Constructions: centred model = train on `ttn.teachers.ShiftedTeacher` (target W* + W_0), the
  main one-knob family; two-copy model = `ttn.network.antisymmetric_double` + gradient mask, as a
  cross-check. Large sigma needs time rescaled by the tangent-kernel scale (~ sigma^{2(K-1)}).
- Other agents: `s2s-core`, `locality`, `order`, `data-teacher`, `optim-samples`.

## What to do
1. **sigma x chi** phase diagram (binary N = 4 and 8; odeco teacher with 3 components, weights
   e.g. [3, 2, 1]); sigma over >= 3 decades from deep S2S to lazy; chi from the teacher rank to
   >= 8x it. Per point >= 5 seeds: delta_K (tangent kernel on 128 probe inputs), delta_theta,
   event separation ratios, regime label, final loss. Threshold sensitivity per
   `notes/diagnostics.md`.
2. **sigma x K**: trees with K from 3 to ~15 (binary N = 4, 8, 16; MPS N = 4..12). Test
   sigma_c(K) -> 1 predicted by theory; plot in both sigma and function-space scale sigma^K.
3. **Imbalance**: per-core scales (root vs rest) at fixed function-space scale; does imbalance move
   the TTN between regimes as in Kunin et al.?
4. **Large bond dimension**: sequences chi = 4..64 (as far as affordable) under the NTK-type and
   muP-type parameterizations from the theory note (per-core init scales and learning-rate
   multipliers): delta_K and delta_theta vs chi, and whether feature learning survives.
5. Cross-check (1) on a few points with the two-copy construction.

## Deliverables and success criteria
- `experiments/regime_map/` and `notes/regime_map.md`. Figures: phase diagrams (categorical) with
  the predicted boundaries overlaid; delta_K and separation ratio as continuous heatmaps;
  threshold-sensitivity panel; large-chi scaling plots; representative runs from each regime.
- A clear statement of which parameters move each boundary and how (fitted vs predicted).

## Constraints
- Interim report after the first sigma x chi diagram (~4 h of work). Raw outputs to
  `/root/outputs/regime_map/`. This is the most compute-heavy task: estimate cost before each sweep
  and prefer coarse grids refined near boundaries.
