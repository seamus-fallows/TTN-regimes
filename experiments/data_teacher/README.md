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

```
# step 1 (pilot integrator; exact aligned check runs and plots in one go)
uv run python experiments/data_teacher/e1_exact_aligned.py
experiments/data_teacher/run_queue.sh e1_order e1_scaling_sigma e1_scaling_lambda   # run_aligned.py
# steps 2, 3, 4a at one sigma (pilot integrator)
experiments/data_teacher/run_queue.sh e2_misaligned e4a_nonorth e3_correlated      # run_generic.py
# sigma sweeps and TTN teachers: library integrator (LSODA), two-stage dense grid
experiments/data_teacher/run_dense_queue.sh e4a_sigma e2_sigma e4b_ttn e4c_cross    # run_dense.py
# predictions (rank-constrained critical points)
uv run python experiments/data_teacher/predict_rank1.py e2_misaligned e4a_nonorth
uv run python experiments/data_teacher/predict_rank1.py e3_correlated
```

All runners are resumable (existing outputs are skipped) and use 28 processes (CPU quota 30.7).
`run_aligned.py` runs `configs/e1_*.yaml`, `run_generic.py` the single-sigma configs of steps 2-4, and
`run_dense.py` the configs that need to resolve short later plateaus. The library integrator is used
for those (`ttn.dynamics.integrate_gf`, LSODA, rtol 1e-9, atol 1e-14): stage A runs on a log grid
until L < 0.98 L(0), stage B on a grid geometric in t - t_A.

**Variants**: copy or edit a YAML in `configs/`. Sweep keys are cartesian products. `variant:
white_mapped` (step 2) runs the absorbed white problem from the mapped init; `variant:
white_sqrtLs_mapped` (step 1) does the same for the aligned case. Raw outputs go to
`/root/outputs/data_teacher/<config name>[_dense]/` (override with `DATA_TEACHER_OUT`): one npz per
run with t, loss, overlaps, bond spectra and the parameter trajectory (dense runs: library `Trajectory`
format). They are not in git; the commands above regenerate them (total about 1.5 h on 28 cores).

## Analysis and figures (each figure from saved results by the script named)

| figure / table in `results/` | script | content |
|---|---|---|
| `e1_exact_aligned.{png,csv}` | `e1_exact_aligned.py` | aligned init: TTN GF vs scalar ODE vs substitution s -> Lambda s |
| `e1_order_probs.png`, `e1_examples.png`, `e1_paired_times.png`, `e1_order_table.csv`, `e1_paired_agg.csv`, `e1_plateaus.csv` | `analyze_e1_order.py` | order, swaps, paired comparisons, plateau levels |
| `e1_scaling.png`, `e1_scaling_fits.csv` | `analyze_e1_scaling.py` | t_1 vs sigma and vs Lambda s |
| `e1_kexp.csv` | `analyze_e1_kexp.py` | local growth exponent (escape degree) |
| `e1_hessian_check.csv` | `e1_hessian_check.py` | gauge-projected Hessian at the exact swap saddle vs formula |
| `e2_misaligned_summary.png`, `e2_misaligned_plateaus.csv` | `analyze_plateaus.py e2_misaligned e2_misaligned_e4a_nonorth`, `plot_cp_steps.py e2_misaligned theta_deg rank1_predictions_e2_misaligned_e4a_nonorth.csv` | step 2 plateaus vs predicted critical values, alignment |
| `e3_correlated_summary.png`, `e3_correlated_plateaus.csv` | `analyze_plateaus.py e3_correlated e3_correlated`, `plot_cp_steps.py e3_correlated rho rank1_predictions_e3_correlated.csv` | step 3 |
| `e3_sgd_check.{png,csv}` | `e3_sgd_check.py` | online SGD vs exact population GF |
| `e4a_nonorth_summary.png`, `e4a_nonorth_plateaus.csv` | `analyze_plateaus.py e4a_nonorth e2_misaligned_e4a_nonorth`, `plot_cp_steps.py e4a_nonorth overlap rank1_predictions_e2_misaligned_e4a_nonorth.csv` | step 4a |
| `e4a_sigma_law.{png,csv}`, `e4a_sigma_law_fits.csv`, `e4a_theory_slopes.csv` | `analyze_sigma_law.py e4a_sigma overlap`, `theory_log_slope.py` | plateau law vs overlap, predicted slopes |
| `e2_sigma_law.{png,csv}`, `e2_sigma_law_fits.csv` | `analyze_sigma_law.py e2_sigma theta_deg` | plateau law vs misalignment |
| `e4b_ttn_plateaus.csv`, `e4b_ttn_growth.csv` | `analyze_ttn.py e4b_ttn` | plateau states vs HSVD vs critical points |
| `e4b_law.png`, `e4b_order{,_fits}.csv`, `e4b_theory.csv` | `analyze_e4b_order.py` | order of bond growth, log law vs predicted slopes |
| `e4c_cross.png`, `e4c_cross_{events,summary}.csv`, `e4c_theory.csv` | `analyze_cross.py`, `analyze_e4c_theory.py` | cross-tree teachers |
| `summary_plateau_law.png` | `plot_summary_law.py` | later-plateau law, baseline vs complications |
| `crosscheck_library.csv` | `crosscheck_library.py` | pilot vs library integrator |
| `float_floor_check.csv` | `check_float_floor.py` | dormant signals vs float64 floor |

`pilots/` holds the throwaway pilot (`pilot_reversal.py`) and the pilot integrator used by the
single-sigma runners.

## Results

Summary table, the one-line answer and all numbers: `notes/data_teacher.md`. In short: an aligned
covariance only reweights (race by Lambda s, landscape by sqrt(Lambda) s; extra swap events on
binary-4). Misalignment, overlapping CP components, cross-site correlations and generic TTN
teachers turn the later plateaus of MPS students from sigma^-(K-2) into log(1/sigma), with slopes as
predicted by theory_teachers.md. Plateaus are always critical points of the bond-rank-constrained
loss (not HSVD truncations). Binary students learn bottom-up.
