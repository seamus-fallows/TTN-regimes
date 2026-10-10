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

## 2026-10-09 (evening) — Review gate, follow-ups, deep-tree baselines

- **Review gate before results enter the report.** Every agent's round of results gets an independent
  local review: (A) partial reproduction from the committed code plus one independent check, (B) an
  adversarial audit of claims vs data. Reviews are committed to `notes/reviews/`. The first review
  (data-teacher round 1) found the runs correct but several headline numbers unsupported by the analysis
  (grid-limited durations, unfair log-vs-power comparison, residual-based error bars); the resulting
  "Analysis standards" were added to `notes/briefs/README.md` and sent to all agents.
- **data-teacher continues (round 2)** instead of being shut down: the tree-shape contrast for generic
  teachers (MPS log plateaus vs binary/HT polynomial) is central to the report, and the agent has the tools.
  It also fixes the review's issues first.
- **Deep-tree baselines use gauge-balanced inits** (decision by `optimizers`, endorsed): fan-in inits on
  MPS8/binary8 with chi = teacher rank stall (dormant routes die; theory_saddles R12, theory_regimes P14).
  Stalls are a finding in their own right (regime-map studies them); experiments that need clean S2S on deep
  trees use balanced inits or spare bond channels, and must say so.
- **Theory programme complete**: all six theory notes are at v2 after derive -> adversarial verify -> revise.

## 2026-10-09 (late) — Report bookkeeping

- Open check before the final report: theorem/lemma numbers cited from preprints with several arXiv versions
  (e.g. "Thm 8" of Kumar & Haupt "Neuron Pursuit", whose v2 is twice the length of v1; Furman et al. 2026
  lemma numbers) must be re-checked against the version listed in references.bib.

## 2026-10-10 — Monitoring failure and wrap-up plan

- **Missed completions.** The background watcher that flags remote-agent state changes hung (no timeout on
  `crab-student list`), so six agents that finished between 01:30 and 08:00 UTC sat idle until the user asked.
  Nothing was lost (every agent autosaved and pushed). Fix: polling commands get hard timeouts, and the lead
  checks in manually at least every ~2 h instead of relying on a single watcher.
- **Wrap-up plan.** All research agents are done. Remaining: (1) reviews of s2s-core, order, regime-map,
  data-teacher (round 2) and real-data (running in parallel locally); (2) small fix rounds where needed —
  analysis/wording fixes by local agents on the committed code and CSVs, restarting a remote machine only if new
  runs are required; (3) copy + verify raw outputs, then delete machines (user's standing OK); (4) write the
  remaining report sections from reviewed results, final consistency pass, compile PDF.

## 2026-10-10 (morning) — Fix rounds after the five reviews

- **Triage.** Each review was condensed into a MUST/SHOULD fix list (s2s-core 8 MUST, order 7, regime-map 8,
  data-teacher 7, real-data 8). Fix lists were sent to the three agents whose machines restarted (s2s-core,
  order, data-teacher); order also runs a few small new runs (H1b on adjacent / separated MPS4 pairs).
- **regime-map and real-data machines cannot restart** (Vast: "required resources are currently unavailable,
  state change queued"; the locality machine likewise). Their fix rounds run **locally** instead, in git worktrees
  (`../wt/<name>`, branches `wip/<name>`), on the committed code and CSVs plus the reviewers' local re-runs in
  `data/review_<experiment>/`; light local reruns (<= ~1 h on 12 cores) are allowed for the few raw-dependent
  items; anything heavier is scripted and left for when the machines come back. A watcher retries `start`
  every ~30 min; when a machine starts, its raw outputs are copied and verified first, then it is deleted.
- **Report writing in parallel worktrees.** Separate local agents for Sec. 5.5 (optim-samples), Sec. 2 polish,
  Appendix A trim, and a citation / prior-work audit, each editing disjoint files on its own branch; the lead
  merges. Sections 3, 4.1, 4.3, 5.1-5.4 and 6 wait for the corresponding fix rounds.

## 2026-10-10 (midday) — Event rules, prior work, new theory tasks

- **Spectral-event rules are documented as implemented** (Sec. 2, notes/diagnostics.md): (a) onset f = 0.05 of the
  teacher value (recommended; locality bond events, data-teacher v2), (b) half the teacher value (optimizers,
  data-teacher correlated inputs), (c) last half-crossing of the final value (library default `spectral_events`).
  Appendix B states the rule per experiment. The library is not changed (it would silently change old summaries).
- **Prior-work audit** (notes/reviews/citation_check.md): five results tagged new have prior or concurrent work (Adam
  eps law, EoS two-cycle amplitude, KKT balance, per-init blow-up identity, k >= 3 head start). They are re-tagged as
  known (or "TTN form / closed form new"); Appendix A tags are aligned with the verified theory notes (heuristic ->
  conjecture or empirical-with-heuristic-mechanism). The report must not overclaim novelty.
- **Real data:** the review's optional extra-order test is run locally (light: <= 8 processes, ~1.5 h) and
  PRE-REGISTERED in notes/real_data.md: does the k <= 5 route-amplitude share predict the final loss across >= 20 random
  and 6 structured pixel orders (seeds averaged per order; Spearman with bootstrap CIs over orders)?
- **New theory tasks (local):** (1) prove the lower bound saddle order = k* (open in theory_saddles section 11) or find
  a counterexample, numerics first; (2) a "saddle hopper" that predicts the whole saddle sequence from the theory and is
  scored against the committed runs of locality, s2s-core, data-teacher and order without tuning.
- **Machines:** s2s-core and real-data deleted after byte-verified copies (real-data's 45 remote symlinks are listed in
  data/real_data/SYMLINKS_on_remote.txt; all targets were copied). regime-map restarted on its own (queued start):
  copy its raw outputs for the local fix round, then delete.

## 2026-10-10 (afternoon) — Results merged; saddle-order theorem; regime-map round 2

- **Merged:** s2s-core errata (E1-E21), data-teacher errata (E1-E16), regime-map fix round (local, on the copied raw
  outputs), real-data fix round + round 2 (pre-registered order test), citation fixes, report Sec. 4.1-4.3 + 4.6
  (s2s-core), Sec. 6 (data-teacher). Report builds at 89 pages.
- **Saddle order (theory, UNVERIFIED until the two verifications land):** the claim "saddle order = k*" (old R2) is
  false in general. New Theorem 1: order = k* if the generalized piggyback operator has lambda_max < s_q (proven per the
  note); corollaries cover k* <= 4 and two-component teachers in the correct order; Theorem 2: above the threshold the
  order drops to 2l < k*; counterexample MPS6 (k* = 5, PSD Hessian, order 4; GF escape ~ sigma^-2). Two independent
  verifications (proof; numerics) run before notes/theory_saddles.md and the report are changed.
- **Saddle hopper (UNVERIFIED):** an adversarial audit checks for post-hoc "rare branches" and uninformative matches
  before any number is cited.
- **Regime map decisions:** regime labels and delta_K use the tensor kernel wherever feasible (N <= 8); overlap events
  are dated at the last upward 1/2-crossing before the component first reaches 0.9 (to be made consistent in
  notes/diagnostics.md and Sec. 2 in the final consistency pass); the library's `tensor_tangent_kernel` needs a
  memory-light (reverse-mode) version (34 GB at chi = 128, two-copy) — library task for the cleanup pass.
- **regime-map round 2 (remote, restarted):** realizable K sweep (spare channels or full bonds), d = 2 and 3,
  pre-registered comparison of sigma^K vs sigma^K/sqrt(D), to settle sigma_c at large K (now: brackets, conjecture).

## 2026-10-10 (evening) — Second monitoring miss; Vast 2FA expiry; handoff

- **Missed completion again.** saddle-dyn finished at 19:13 UTC; the lead noticed at ~21:57. Cause: Vast's login
  expired ("Vast wants a fresh 2FA login"), crab-student reported the agent as "unknown", and the watcher had been
  changed to ignore "unknown" (to avoid exiting on transient API errors). Lesson: a watcher must also watch the
  agent's branch (the "Autosave (done)" commit and REPORT status), not only the crab-student state, and must treat
  a persistent "unknown" as an alarm.
- **Vast 2FA** must be refreshed by the user (`vastai tfa login --method-type totp -c <code>`); until then the last
  machine cannot be started, copied from or deleted (it is stopped; disk cost only).
- **Handoff** written at the top of CLAUDE.md: what remains is the 2FA login, the copy + delete of saddle-dyn, its
  verification (workflow saved in notes/handoff/), filling three placeholders, and final bookkeeping.
