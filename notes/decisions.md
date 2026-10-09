# Decision log

Newest last. Each entry: date, decision, why, alternatives considered. Dropped directions are logged
here too.

## 2026-10-09 — Project setup

- **GitHub repo is private.** The brief left visibility unspecified; private is the reversible choice.
- **Vast.ai budget: $300 total** (set by the user). Per-instance guidance from the brief: about
  $2/h normally, up to $10/h for a single instance when a large experiment genuinely benefits.
  Running spend is tracked in `notes/spend.md`.
- **Stack: Python 3.13, uv, PyTorch, float64.** Same conventions as the user's DLN codebase
  (`Training-Dynamics-of-Deep-Learning`). float64 because plateau dynamics at small init involve
  quantities around 1e-12 and smaller.
- **Model** (details and justification in `notes/model.md`): TTN regression
  f(x) = <W, (x)_i phi_i(x)> with whitened Gaussian site features, exact population loss
  ||W - W*||^2 / 2 by tree contraction, CP/odeco or TTN teachers, fan-in initialization with a
  per-core scale sigma. Alternatives considered:
  - *TTN as a sensing/completion problem* (Razin et al. style partial observations): adds an
    observation operator and conflates dynamics with implicit bias. Kept for finite-sample
    experiments only.
  - *Nonlinear local feature maps* (cos/sin as in Stoudenmire & Schwab): identical population
    dynamics when orthonormal, so not a separate model.
  - *Leaf cores on every site (HT format)* as the default: adds N cores that are redundant when
    the leaf bond dimension >= d and changes K. Supported as the `ht` topology, not the default.
- **Lazy regime construction: antisymmetric doubling** (W_0 = 0 exactly at any init scale) rather
  than Chizat's centred model f - f_0, because it stays a genuine TTN.
