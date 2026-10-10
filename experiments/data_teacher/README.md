# data_teacher — does input covariance or teacher structure change the saddle-to-saddle picture? (Q6, Q7)

Agent `data-teacher`. Findings and derivations: `notes/data_teacher.md`. Running report:
`agent-runs/data-teacher/REPORT.md`.

## Question

Starting from the baseline (whitened inputs, odeco teacher, small fan-in init, gradient flow), add one
complication at a time and decide whether it changes the saddle-to-saddle (S2S) picture
qualitatively, quantitatively or not at all, and whether the baseline theory still predicts it after
a simple substitution:

1. aligned anisotropic product covariance (H5): `e1_*`
2. misaligned product covariance: `e2_misaligned`
3. cross-site correlated inputs (shared latent factor): `e3_correlated`, `e3_sgd_check.py`
4. teacher structure: (a) non-orthogonal CP components `e4a_*`; (b) random TTN teacher on the
   student's tree `e4b_ttn`; (c) teacher on a different tree `e4c_cross`

## Setup

- Model, init and loss as in `notes/model.md`: N = 4 sites (N = 8 in `e4c`), d = 4 (3 in `e4b`, 2 in
  `e4c`), chi = 4, fan-in init with scale sigma, exact population loss.
- **Dynamics: gradient flow**, integrated with scipy `solve_ivp` (DOP853, rtol 1e-9, atol 1e-14,
  float64) on the flat parameter vector (`pilots/pilot_gf.py`). This is the pilot integrator the briefs
  allow until `ttn/dynamics.py` lands. The loss is evaluated through the dense tensor (identical to
  `ttn.losses.population_loss` to 1e-17, 4.5x faster for these sizes). Runs stop once
  L < `stop_rel_loss` x L(0) (the terminal state is recorded) or at `t_max`. Conservation-law drift
  max |Q_e(t) - Q_e(0)| is stored for every run (all step-1 runs: < 1e-9).
- Covariances: `ttn.losses.power_law_cov(d, alpha)` (eigenvalues k^-alpha, trace d), rotated by
  theta in the (e1, e2) plane for step 2. Correlated inputs: `correlated.py` builds the exact
  moment matrix E[Phi Phi^T] by Isserlis' theorem (tested against Monte Carlo and the N = 2 closed form).
- Teachers: `setups.py`. CP teachers are deterministic (standard basis); random TTN teachers use
  `ttn.teachers.teacher_generator(teacher_seed)` with teacher seeds (101+, 201+) independent of the
  init seeds (0..39).
- Predicted saddles: `constrained.py` finds the local minima of the loss restricted to fixed bond ranks
  (L-BFGS from many starts); `predict_rank1.py` applies it to every case of steps 2, 3, 4(a).

## How to run

Environment: `DATA_TEACHER_OUT` = raw-output folder (default `/root/outputs/data_teacher` on the remote
machines, else `<repo>/data/data_teacher`); `DATA_TEACHER_RESULTS` = folder for summary CSVs and
figures (default `results/`; **set it when re-running**, or the committed results are overwritten);
`PROCS` = process count for the queue scripts (default 28); `env1.sh` caps BLAS/OpenMP threads at 1
(LSODA's dense LAPACK calls otherwise oversubscribe the machine).

Run order (each step reads what the previous ones wrote):

```
# round 1, step 1 (pilot integrator: scipy DOP853 on the flat vector)
uv run python experiments/data_teacher/e1_exact_aligned.py
experiments/data_teacher/run_queue.sh e1_order e1_scaling_sigma e1_scaling_lambda  # -> summary_rows.pkl
uv run python experiments/data_teacher/analyze_e1_order.py        # needs all 4 variants of e1_order
uv run python experiments/data_teacher/analyze_e1_scaling.py
uv run python experiments/data_teacher/analyze_e1_kexp.py
uv run python experiments/data_teacher/e1_hessian_check.py
# round 1, steps 2, 3, 4a at one sigma (pilot integrator) and their rank-one predictions
experiments/data_teacher/run_queue.sh e2_misaligned e4a_nonorth e3_correlated
uv run python experiments/data_teacher/predict_rank1.py e2_misaligned e4a_nonorth   # -> .pt + csv
uv run python experiments/data_teacher/predict_rank1.py e3_correlated
uv run python experiments/data_teacher/analyze_plateaus.py e4a_nonorth e2_misaligned_e4a_nonorth  # needs the .pt
# sigma sweeps on the library integrator (LSODA, two-stage dense grid, fine rows)
experiments/data_teacher/run_dense_queue.sh e4a_sigma e2_sigma e4b_ttn e4c_cross r2_random r2_cross
uv run python experiments/data_teacher/theory_log_slope.py        # -> e4a_theory_slopes.csv
uv run python experiments/data_teacher/analyze_law_v2.py e4a_sigma   # (also e2_sigma, e4b_ttn, r2_random)
uv run python experiments/data_teacher/analyze_ttn.py e4b_ttn
uv run python experiments/data_teacher/analyze_e4c_theory.py; uv run python experiments/data_teacher/analyze_cross_v2.py e4c_cross
# round 2
uv run python experiments/data_teacher/r2_saddle_runs.py experiments/data_teacher/configs/r2_saddle.yaml
uv run python experiments/data_teacher/r2_saddle_runs.py experiments/data_teacher/configs/r2_saddle_b.yaml
uv run python experiments/data_teacher/r2_saddle_runs.py experiments/data_teacher/configs/r2_saddle_ext.yaml
uv run python experiments/data_teacher/recompute_saddle_hessians.py
uv run python experiments/data_teacher/analyze_r2_saddle.py       # tree contrast (r2_saddle_*.csv/png)
uv run python experiments/data_teacher/r2_corr.py; uv run python experiments/data_teacher/analyze_r2_corr.py
uv run python experiments/data_teacher/r2_star.py hessian; ... r2_star.py saddle; ... r2_star.py random
uv run python experiments/data_teacher/analyze_r2_star.py
uv run python experiments/data_teacher/check_float_floor.py
```

The queues actually used on the remote machine are `run_queue.sh`, `run_dense_queue.sh`, `r2_chain.sh`,
`r2_chain2.sh`, `r2_chain3.sh`. All runners are resumable (existing outputs are skipped).

**Variants**: copy or edit a YAML in `configs/`. Sweep keys are cartesian products; `cases:` gives an
explicit list. `variant: white_mapped` (step 2) runs the absorbed white problem from the mapped init;
`variant: white_sqrtLs_mapped` (step 1) the same for the aligned case.

## Methods added after the review of round 1

- `lawfit.py`: the log law and the power law, both fitted and scored in log space, per seed; errors
  are standard errors over seeds (the same seeds are reused at every sigma).
- `analyze_law_v2.py`: events by the onset rule of notes/diagnostics.md (s_2^e >= f sigma_2^e(T),
  f = 0.05, sensitivity 0.03 / 0.1), runs classified by the rank-one saddle they visit, runs without a
  resolved plateau flagged and excluded, gauge-projected and raw Hessian lambda_min at the plateau point.
- Saddle-start protocol (`r2_common.py`, `r2_saddle_runs.py`): the exact balanced rank-one critical
  point + noise eps; it isolates the escape law from the first saddle.
- Event times from merged log + fine rows (`ttn.dynamics.merged_rows`) for every run made with the
  final library.

## Analysis and figures (each figure from saved results by the script named)

| figure / table in `results/` | script | content |
|---|---|---|
| `e1_exact_aligned.{png,csv}` | `e1_exact_aligned.py` | aligned init: TTN GF vs scalar ODE vs substitution s -> Lambda s |
| `e1_order_probs.png`, `e1_examples.png`, `e1_paired_times.png`, `e1_order_table.csv`, `e1_paired_agg.csv`, `e1_plateaus.csv` | `analyze_e1_order.py` | step 1: order, swaps, paired comparisons, plateau levels |
| `e1_scaling.png`, `e1_scaling_fits.csv` | `analyze_e1_scaling.py` | t_1 vs sigma and vs Lambda s |
| `e1_kexp.csv` | `analyze_e1_kexp.py` | escape degree per sigma (window above the init transient) |
| `e1_hessian_check.csv` | `e1_hessian_check.py` | swap Hessian at the exact saddle vs formula (raw and projected) |
| `e2_misaligned_summary.png`, `e2_misaligned_plateaus.csv`, `e3_correlated_*`, `e4a_nonorth_*` | `analyze_plateaus.py`, `plot_cp_steps.py` | round-1 single-sigma plateau levels and alignment (durations there are WITHDRAWN, see notes R2.0) |
| `e4a_sigma_v2_{runs,perseed,summary}.csv`, `e4a_sigma_v2.png` (also `e2_sigma_v2_*`, `e4b_ttn_v2_*`, `r2_random_v2_*`) | `analyze_law_v2.py <config>` | per-seed log vs power law, saddle classification, plateau lambda_min (raw and projected) vs sigma |
| `e4a_sigma_law*.{csv,png}`, `e2_sigma_law*` | `analyze_sigma_law.py` | round-1 analysis, superseded by the v2 files (kept for the record) |
| `e4a_theory_slopes.csv` | `theory_log_slope.py` | predicted R6 slopes for the 4a teachers |
| `e4b_ttn_plateaus.csv`, `e4b_ttn_growth.csv` | `analyze_ttn.py e4b_ttn` | plateau states vs the rank-constrained critical point and every SVD-type truncation |
| `e4b_law.png`, `e4b_order{,_fits}.csv`, `e4b_theory.csv` | `analyze_e4b_order.py` | round-1 order analysis (fixed threshold; superseded by `e4b_ttn_v2_*`; `e4b_theory.csv` still used) |
| `e4c_cross_v2_*`, `r2_cross_v2_*`, `e4c_theory.csv` | `analyze_cross_v2.py [e4c_cross / r2_cross]`, `analyze_e4c_theory.py` | cross-tree teachers: P11, sibling-first, naive rule (f-rule events) |
| `r2_saddle_hessian.csv`, `r2_saddle_b_hessian.csv` | `recompute_saddle_hessians.py`, `r2_saddle_runs.py` | exact rank-1 saddle Hessians (13 teachers x 3 trees) |
| `r2_saddle_law.png`, `r2_saddle_collapse.png`, `r2_saddle_fits.csv`, `r2_saddle_perseed.csv`, `r2_saddle_runs_all.csv` | `analyze_r2_saddle.py` | round 2 item 1: tree contrast (saddle start) |
| `r2_saddle_resolution_check.csv` | `check_resolution.py` | log-row vs fine-row escape times (<= 0.5%) |
| `r2_corr_{runs,fits}.csv`, `r2_corr_law.png` | `r2_corr.py`, `analyze_r2_corr.py` | round 2 item 3: correlated inputs, sigma law |
| `r2_star_hessian.csv`, `r2_star_saddle.csv`, `r2_star_threshold{,_scan}.csv`, `r2_star_rotation_law.csv`, `r2_star_depth{,_exponents}.csv`, `r2_star.png` | `r2_star.py {hessian,saddle,threshold,random}`, `analyze_r2_star.py` | round 2 item 2: stars |
| `e3_sgd_check.{png,csv}` | `e3_sgd_check.py` | online SGD vs exact population GF (binary-4, rho = 0.3) |
| `crosscheck_library.csv` | `crosscheck_library.py` | pilot vs library integrator |
| `float_floor_check.csv` | `check_float_floor.py` | dormant signals after the first escape vs the float64 floor |

`pilots/` holds the throwaway pilot (`pilot_reversal.py`) and the pilot integrator used by the
single-sigma runners and by `r2_corr.py`.

## Results

Summary table, the one-line answer, all numbers, and the list of round-1 claims changed after the
review (section R2.0): `notes/data_teacher.md`. Round 2 figures: `r2_saddle_law.png`,
`r2_saddle_collapse.png` (tree contrast), `r2_random_v2.png`, `r2_corr_law.png`, `r2_star.png`. In short: an aligned
covariance only reweights (race by Lambda s, landscape by sqrt(Lambda) s; extra swap events on
binary-4). Misalignment, overlapping CP components, cross-site correlations and generic TTN
teachers turn the later plateaus of MPS students from sigma^-(K-2) into log(1/sigma), with slopes as
predicted by theory_teachers.md. Plateaus are always critical points of the bond-rank-constrained
loss (not HSVD truncations). Binary students learn bottom-up.
