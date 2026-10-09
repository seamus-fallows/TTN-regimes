# Brief: `real-data` — does the teacher-student picture show up on real data? (backlog B5)

Read `notes/briefs/README.md` first (instructions for all agents, including the analysis standards).

## Task
Train TTNs on a small real dataset with the standard tensor-network feature maps and test whether the
phenomena established in the teacher-student setting appear: plateaus and stepwise bond-rank growth at
small init, the regimes as init scale varies, and, most importantly, the **locality rule**: correlations
between sites that are far apart in the tree are learned later and more slowly at small init.

## Goal
External validity for the report: one clear section saying which teacher-student predictions survive on
real data, which do not, and why, with figures.

## Context
- Main findings so far: `CLAUDE.md` "Key findings"; theory in `notes/theory_saddles.md` (route rule:
  the escape needed to learn a correlation between a set of sites needs k cores = the Steiner subtree of
  those sites; plateau ~ sigma^-(k-2), log for k = 2, none for k = 1), `notes/theory_escape.md`,
  `notes/theory_regimes.md`; `notes/diagnostics.md` (events, regimes; spectral events at f = 0.05 of the
  target).
- Literature: Stoudenmire & Schwab 2016 (MPS on MNIST with phi(x) = (cos(pi x/2), sin(pi x/2))); Cheng et
  al. 2019 and Liu et al. 2019 (TTNs on images); Levine et al. 2018 (matching tree structure to data
  correlations); MPS miss long-range correlations (Tang 2025). See `notes/literature.md` section 2.5.
- Library: `ttn.network`, `ttn.losses.empirical_loss`, `ttn.dynamics` (GD, SGD; `Problem` with a fixed
  dataset), `ttn.diagnostics`. The model is the project's TTN regression f(x) = <W, phi_1 (x) ... (x) phi_N>;
  local features are not Gaussian here, so the population loss is replaced by the empirical loss on a
  training set (report train and test loss).

## What to do
1. **Data**: MNIST or Fashion-MNIST, downsampled to 4x4 (N = 16 sites) and, if affordable, 8x8 (N = 64);
   local feature map phi(x) = (cos(pi x / 2), sin(pi x / 2)) (d = 2), optionally centred/whitened per site
   (report both; whitening makes the setting closer to the theory). Target: a scalar regression target
   (e.g. +-1 for one class vs another, or a one-vs-rest label), square loss; n_train 2k-20k.
2. **Trees**: (a) a binary tree whose leaves follow a 2D quadtree ordering of the pixels (spatially local
   subtrees), (b) an MPS along a snake ordering, (c) the same binary tree with a scrambled pixel order
   (locality destroyed). Same bond dimension chi (e.g. 4, 8) for all.
3. **Small-init dynamics** (GD with small step or the GF integrator on the empirical loss; several init
   scales over >= 1 decade, >= 5 seeds): loss curves (train/test), bond spectra on every edge, effective
   bond ranks, events and separation ratios. Do plateaus and stepwise bond growth appear? In which order do
   edges acquire rank (bottom-up? local before global)? Do plateau durations scale with init scale as the
   theory predicts for the route degree of the correlations being learned?
4. **Locality test**: compare (a), (b), (c): time to reach a given train loss and the order of bond growth.
   Prediction: the spatially local binary tree learns faster and with shorter later plateaus than the
   scrambled tree; the MPS sits in between, with slow learning of correlations between pixels far apart
   along the snake. Quantify with the diagnostics, not by eye.
5. **Regimes**: one sweep of init scale from small to large (centred model) on tree (a): S2S vs rich vs
   lazy per `notes/diagnostics.md` (tangent kernel on 128 probe images), with train/test error, to see the
   regime map on real data.

## Deliverables and success criteria
- `experiments/real_data/` and `notes/real_data.md`. Figures: loss curves and bond spectra per tree; bond
  growth order; plateau duration vs init scale; locality comparison (a)/(b)/(c); regime sweep.
- A table: teacher-student prediction -> observed on real data (yes / partly / no) -> evidence.

## Constraints
- Interim report after steps 1-3 on 4x4 data (~3 h). Raw outputs to `/root/outputs/real_data/`.
- Keep runs affordable (empirical loss over n_train samples per gradient: use minibatches only if needed,
  and say so).
