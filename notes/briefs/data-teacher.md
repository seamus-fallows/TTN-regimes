# Brief: `data-teacher` — do input covariance and teacher structure change the picture? (Q6, Q7)

Read `notes/briefs/README.md` first (instructions for all agents).

## Task
Starting from the whitened-input, odeco-teacher baseline, add one complication at a time and find
out whether it changes the saddle-to-saddle picture qualitatively, quantitatively, or not at all.

## Goal
For each added complexity: why it was worth checking, what changed, and whether the theory for the
baseline still predicts it (possibly after a simple substitution such as effective weights).

## Context
- Q6, Q7 and H5 in `PLAN.md`. Library: `ttn.losses.power_law_cov` and site metrics (product
  covariances handled exactly in the population loss), `ttn.teachers` (random CP, random TTN teachers
  on any tree, `TTNTeacher` across trees), `ttn.losses.sample_inputs` + SGD/empirical loss for
  non-product inputs.
- Theory: `notes/theory_escape.md`, `notes/theory_saddles.md`.
- Other agents: `s2s-core` (baseline scaling), `locality`, `order`, `regime-map`, `optim-samples`.

## What to do
1. **Anisotropic product covariance, aligned with the teacher.** Site covariances with power-law
   spectra; odeco teacher whose factors are covariance eigenvectors. Prediction (H5): component r
   behaves as if its weight were Lambda_r s_r with Lambda_r = prod_i lambda_{r,i}; order and escape
   times follow Lambda_r s_r; plateau levels are Lambda_r s_r^2 / 2-type values. Test quantitatively,
   including a case where covariance reverses the order of two components.
2. **Misaligned covariance** (teacher factors not eigenvectors): what changes?
3. **Cross-site correlated inputs** (one simple model, e.g. a shared latent factor across sites),
   with sampled data (large batches / online SGD with small steps to approximate gradient flow):
   does S2S survive, and do the saddles change?
4. **Teacher structure.** (a) Non-orthogonal CP components (controlled overlap between
   components): saddle sequence and plateau levels vs overlap. (b) Random TTN teachers with given
   bond ranks on the student's tree: are the visited saddles the hierarchical-SVD truncations
   (compare bond spectra at plateaus with the teacher's), and in which order do bond ranks grow?
   (c) Teacher on a different tree than the student (e.g. MPS teacher, binary student): which
   bonds grow first?

## Deliverables and success criteria
- `experiments/data_teacher/` and `notes/data_teacher.md` with a table: complication -> why checked
  -> what changed -> does baseline theory (with substitution) predict it.
- Figures for each complication against the whitened/odeco baseline.

## Constraints
- Interim report after step 1 (~3 h). Raw outputs to `/root/outputs/data_teacher/`.
