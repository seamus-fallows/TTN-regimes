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
- **Lazy regime construction: centred model** (train on target W* + W_0, i.e. f_theta - f_theta0 as
  in Chizat et al.; `ttn.teachers.ShiftedTeacher`) as the main one-knob family, with Woodworth's
  two-copy model as a cross-check. *Corrected the same day*: I first chose antisymmetric doubling
  (a doubled-bond block-diagonal TTN with W_0 = 0) believing gradient flow preserves its blocks; a
  test showed it does not (the root's mixed blocks get gradient), so the two-copy model needs a
  gradient mask (`antisymmetric_double` returns one). The centred model keeps the plain TTN
  parameterization, so it is the primary construction.

## 2026-10-09 — Launch remote agents before the library is finished

- **Decision**: launch the six remote agents (`s2s-core`, `locality`, `order`, `regime-map`,
  `data-teacher`, `optim-samples`) now, while `ttn/dynamics.py`, `diagnostics.py`, `runner.py` etc.
  are still being built and reviewed locally. Agents start by reading, deriving, designing
  experiments and piloting with the core modules + scipy; they merge `main` when told the library
  has landed.
- **Why**: maximize parallelism (user's instruction: keep 4-8 agents doing useful work); the core
  library is tested and stable; machine setup and the reading/derivation phase take an hour or more
  anyway. Risk: some throwaway pilot code; agents are told not to build their own frameworks.
- **Mode**: autonomous (agents record decisions instead of blocking on questions); the lead checks in
  every ~2 h and redirects with messages.
