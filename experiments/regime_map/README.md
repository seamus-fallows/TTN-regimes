# Experiment `regime_map`: map of training regimes (Q5, H4)

## Question
Where do saddle-to-saddle (S2S), rich non-incremental and lazy dynamics occur as functions of the
per-core init scale sigma, the bond dimension chi, the number of cores K and the root/rest
imbalance, and do the boundaries follow the predictions of `notes/theory_regimes.md`?
Findings: `notes/regime_map.md`. Agent report: `agent-runs/regime-map/REPORT.md`.

## Setup (defaults)
- Whitened Gaussian inputs, exact population loss, odeco teacher [3, 2, 1] at d = 3 or [3, 2] at
  d = 2 (teacher seed 1000 + init seed, independent teacher stream), fan-in init with per-core scale
  sigma, probe set of 128 Gaussian inputs (generator seed 12345) for the tangent kernel.
- Models: **centred** (target W* + W_0; the main one-knob family), **plain** (target W*), **two-copy**
  (`antisymmetric_double` + gradient mask, W_0 = 0).
- Labels: rules of `notes/diagnostics.md` (final-rise widths), with convergence judged against
  1/2 ||W*||^2 (= L(0) except for the plain model at large sigma; see the report, decision 11), at all
  9 threshold pairs delta_K in {0.05, 0.1, 0.2} x rho in {2, 3, 5}.

## Dynamics
Gradient flow with the shared library (`ttn.dynamics.integrate_gf`): DP5 (rtol 1e-8, atol 1e-12), or
LSODA with the exact Hessian for long stiff plateaus (`method: auto` = LSODA below sigma = 0.7).
Every event (final rise of a component overlap) is re-integrated from the saved parameters on a
200-point linear grid (`run_regime.refine_events`), so widths are resolved in linear time. Stop
when L < 1e-6 x 1/2 ||W*||^2, at the time budget T = 50 t_esc + 1e4 / max(lambda_bar, 1)
(t_esc = aligned escape-time estimate, lambda_bar = tr Theta(0) / D), or at max_steps. Accuracy
check: conservation-law drift (recorded per run, <= 1e-9 on the N = 4 grid) and an independent
scipy RK45 integrator on 1937 matched runs (`pilots/interim.py`, labels agree in 98.1%).

## Files
| file | role |
|---|---|
| `run_regime.py` | one run (`run_point`: build, integrate, refine events, kernel change, labels) and the parallel resumable sweep loop |
| `sweep.py` | expands YAML configs (`configs/*.yaml`) and runs them; `--n-proc`, `--out`; skips runs already in `<out>/<exp>/results.jsonl` |
| `analysis.py` | labels at all thresholds, per-seed boundaries (`seed_rho_boundary`, `seed_kernel_boundary`), aligned-mode S2S extrapolation (`aligned_sigma_c`), exact aligned boundary (`aligned_boundary_a0`), stall classification from bond spectra |
| `lazy_theory.py`, `lazy_predict.py`, `pilots/lazy_pred_b8.py` | lazy-limit predictions at init (residual, kappa) by linear response |
| `balance.py` | gauge-balanced init (same W_0, every Q_e = 0) |
| `plots.py`, `make_figures.py` | figure functions and the script that regenerates all figures and tables |
| `pilots/` | throwaway pilots from before the library landed (kept for the record; `interim.py` is the scipy cross-check driver) |
| `results/` | figures, per-seed tables (CSV), lazy predictions |

## How to run (in this order)
```
# 1. sweeps (raw outputs: /root/outputs/regime_map/<exp>/{results.jsonl, traj/*.npz})
C=experiments/regime_map/configs
uv run python experiments/regime_map/sweep.py $C/sigma_chi_b4.yaml $C/chi_b4.yaml $C/sigma_chi_b8.yaml \
    $C/sigma_chi_b8_d2.yaml $C/sigma_K.yaml $C/sigma_K_d2.yaml $C/sigma_K_full_d2.yaml $C/imbalance_b4.yaml \
    $C/large_chi_b4.yaml $C/large_chi_b8.yaml $C/stall_test.yaml $C/stall_test_plain.yaml --n-proc 340
# 2. lazy-limit predictions at init (writes results/lazy_pred_*.csv; refuses to overwrite)
uv run python experiments/regime_map/lazy_predict.py b4 binary 4 3 3,4,6,9,12,18,24,36 6 nocap
uv run python experiments/regime_map/pilots/lazy_pred_b8.py
# 3. figures and tables (results/; pass --out DIR or --force to regenerate)
uv run python experiments/regime_map/make_figures.py
```
Raw outputs are not in git; rerunning step 1 regenerates them (seeded; ~2-4 h on 350 cores).

## Experiments (configs)
| config | what |
|---|---|
| `sigma_chi_b4` | binary N = 4 (K = 3), d = 3: 8 chi (3..36, uncapped) x 19 sigma (0.01-40) x 6 seeds, centred / plain / two-copy |
| `chi_b4` | binary N = 4: chi = 9..128, sigma 0.004-4 (centred) and 0.25-4 (two-copy): boundaries vs chi over > 1 decade |
| `sigma_chi_b8` | binary N = 8 (K = 7), d = 3, deficient bonds chi = 3..24, centred and plain |
| `sigma_chi_b8_d2` | binary N = 8, d = 2, chi = 2..16 and 8/32 (full bonds), centred: the K = 7 lazy region |
| `sigma_K`, `sigma_K_d2`, `sigma_K_full_d2` | sigma x K: binary N = 4, 8, 16, MPS N = 4..12; plain model; chi = teacher rank (stalls), 2x rank, and full bonds |
| `imbalance_b4` | root scale lambda at fixed alpha = lambda sigma^K, binary N = 4, chi = 3, 16, 24 |
| `large_chi_b4`, `large_chi_b8` | chi up to 128 (N = 4) / 32 (N = 8, d = 2) under SP, NTK and muP-TTN parameterizations |
| `stall_test`, `stall_test_plain` | fan-in vs gauge-balanced init at chi = teacher rank, K = 7, 8 |

Variants: any key of `run_regime.run_point` (`root_scale`, `lr` in {sp, ntk, mup}, `bonds` by
subtree size, `balance`, `method`, `max_steps`, `tau_max`, `weights`) can be set in `base`, `grid` or
a `variants` entry.

## Results
Full account: `notes/regime_map.md` (section 0: which parameter moves which boundary); report
`agent-runs/regime-map/REPORT.md`. In short (6 seeds per point, per-seed statistics):
- binary N = 4: S2S / rich / lazy with quantitatively predicted boundaries (S2S boundary per seed from
  the deepest run; rho ~ sigma^-1.03; lazy boundary = theory P5 6.50 / sqrt(chi) to 4-8%); lazy regime only
  at full bonds, centred runs stall at the Eckart-Young floor below the cut dimension;
- both boundaries ~ chi^-1/2: the map collapses in sigma sqrt(chi) (per-bond Gram scale);
- sigma x K (d = 2, centred): the S2S boundary is set by the per-component amplitude sigma^K / sqrt(D)
  ~ 0.01, so sigma_c -> sqrt(d); stalls (dead / long routes, floors) take over at large K;
- large chi: delta_K ~ chi^-3.09 (SP), chi^-0.84 (NTK), chi^+0.03 (muP-TTN);
- imbalance: light root -> lazy (delta_K ~ lambda^2), heavy root -> not lazy; S2S widest near balance.

Figures in `results/` (all regenerated by `make_figures.py --force`): `summary.png`,
`phase_sigma_chi_b4_*.png`, `sensitivity_sigma_chi_b4_*.png`, `boundaries_vs_chi_chi_b4.png`,
`phase_sigma_chi_b8*.png`, `phase_sigma_K_*.png`, `sigma_c_vs_K.png`, `stalls_vs_K.png`,
`phase_imbalance_b4.png`, `large_chi_*.png`, `examples_b4_centred.png`; per-seed CSV tables next to them.
`results/interim/` holds the first (scipy, log-grid) N = 4 map of the interim report (superseded).
In the sigma x K phase plots the "x" (aligned extrapolation from the deepest run) is not meaningful
for trees whose deepest runs stalled.
