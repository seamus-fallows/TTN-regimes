# Brief: `optim-samples` — learning rate, SGD noise and finite samples (Q8)

Read `notes/briefs/README.md` first (instructions for all agents).

## Task
Determine how discrete-step gradient descent, SGD noise and a finite training set modify the
gradient-flow picture of TTN training regimes.

## Goal
Place the learning-rate axis on the regime map and state when gradient-flow results apply to
practical training; show the implicit-bias consequence of saddle-to-saddle dynamics for
generalization.

## Context
- Q8 and H6 in `PLAN.md`. Library: `ttn.dynamics.run_gd`, `run_sgd`, `ttn.losses.empirical_loss`,
  `sample_inputs`. Known in other models: edge of stability (Cohen et al. 2021), catapult phase
  (Lewkowycz et al. 2020), large steps in diagonal nets (Even et al. 2023), implicit regularization in
  (hierarchical) tensor factorization / completion (Razin et al. 2021, 2022). See
  `notes/literature.md`.
- Other agents: `s2s-core`, `locality`, `order`, `regime-map`, `data-teacher`.

## What to do
1. **GD vs GF.** Binary N = 4, 8 and MPS N = 8 with odeco teachers. Predicted stability limit:
   sharpness at the balanced global minimum ~ K s_1^{2-2/K} (verify this formula first; derive the
   correct one if it is wrong), so eta_max ~ 2 / sharpness. Measure: below which eta GD tracks the
   gradient-flow event times (in units t = eta * steps); edge-of-stability behaviour near eta_max
   (sharpness along the trajectory, oscillations, which minimum is selected, balancedness); and
   divergence. Does a large step change the saddle sequence or plateau times at small sigma?
2. **SGD noise.** Online SGD with batch sizes B from 1 to large, at fixed small eta: do plateaus at
   small sigma get shorter (noise-assisted escape from higher-order saddles)? Scaling of plateau time
   with B and eta. Compare with gradient flow.
3. **Finite samples.** Fixed training sets of size n (from below the number of parameters of the
   teacher-rank TTN to well above), small vs large init (centred model for large init): test error
   vs n; which init generalizes from fewer samples; effective bond ranks of the learned W.

## Deliverables and success criteria
- `experiments/optim_samples/` and `notes/optim_samples.md`. Figures: GD/GF agreement and eta_max
  vs prediction; EoS example; plateau time vs B, eta; test error vs n for small vs large init.

## Constraints
- Interim report after step 1 (~3 h). Raw outputs to `/root/outputs/optim_samples/`.
