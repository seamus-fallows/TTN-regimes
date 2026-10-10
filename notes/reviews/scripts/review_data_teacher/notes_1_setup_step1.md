# Reproduction review of data-teacher (round 1) — part 1: setup and step 1

## How the rerun was done
- Repo HEAD 897b000 (merge of agent/data-teacher). `ttn/` is identical between the agent branch tip
  (0cd09c1) and HEAD (`git diff 897b000^2 HEAD -- ttn/` empty), so the library is the one the agent used.
- `experiments/data_teacher/` copied verbatim (diff -r: identical) to
  `data/review_data_teacher/repo_copy/experiments/data_teacher/` (gitignored) because every runner and
  analysis script writes into `HERE/results/`, which would overwrite the committed CSVs. Reduced configs
  were written over the copy's configs (same `name`, fewer sigma/seeds/cases); queue scripts copied with
  `--procs 28` -> `--procs 6` (`run_queue_review6.sh`, `run_dense_queue_review6.sh`).
- `DATA_TEACHER_OUT=C:/Users/seamu/Iliad-ML/TTN_regimes/data/review_data_teacher/raw`.
- Chain: e1_exact_aligned.py (48 s) -> run_queue e1_order (160 runs, 1m36s, 6 procs) -> run_dense_queue
  e4a_sigma (120 runs) + e4b_ttn (18 runs).

## Step 1 (pilot integrator, DOP853)
### Exact aligned-init check (e1_exact_aligned.py, unchanged)
Rerun output identical to the committed `e1_exact_aligned.csv` to all printed digits (half-times,
widths, deviations; only mps comp 1 max-dev differs in the 7th digit: 3.10655e-09 vs 3.10655e-09).

### Paired runs (Sigma, s) vs (I, Lambda s), sigma = 0.1 (binary-4), 0.2 (MPS-4), T1 and T2, seeds 0-9,
all four variants (analyze_e1_order.py needs all of them)
- Per-seed event times reproduce the committed `e1_paired.csv` to 1.7e-14 (t1) and 2e-11 (t2) relative;
  same_first and same_swaps identical in 120/120 rows. Bitwise-level reproduction.
- At these (largest) sigma: same first component 10/10 in every (tree, teacher); rel. difference of
  the first-escape time mean 0.57% / 0.46% (binary T1 / T2), max 1.02% / 1.13%; MPS 0.08% / 0.11%,
  max 0.35% / 0.39%. Committed (40 seeds): binary mean 0.68% / 0.65%, max 1.13% / 1.29%; MPS 0.10% / 0.10%.
- Absorbed white problem (I, sqrt(Lambda) s): same first only 50-80%, times 18-33% off; mapped init:
  60-100% / 13-19%. Matches the committed rows for the same seeds exactly.

### Independent check of the exact clock (indep_exact_clock.py; library LSODA rtol 1e-11, no agent code)
Same teacher (s = 1, 0.7, 0.5 on e3, e2, e1 of power_law_cov(4, 0.5)), aligned init a0 = 0.05.
A separate whitened run (I, s) from the same init, sampled exactly at Lambda_r t:
max_t |a_r^Sigma(t) - a_r^white(Lambda_r t)| = 1.7e-12 / 1.7e-12 / 4.7e-12 (binary-4) and
8.0e-10 / 4.1e-9 / 1.5e-11 (MPS-4). Half-times vs the closed form T(s, a0)/Lambda (quadrature):
-2e-8 .. -4e-7 (binary), -4e-7 .. -1.2e-5 (MPS) relative, i.e. grid-interpolation level.
Side note: the committed e1_exact_aligned.csv half-times carry a log-grid linear-interpolation error
(e.g. MPS r=0: 421.960 vs closed form 422.098, 3e-4 relative); it is identical in the TTN and ODE
columns, so their 1e-9 agreement is a same-grid comparison. Immaterial to the claim.
