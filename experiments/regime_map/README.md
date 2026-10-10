# Experiment `regime_map`: map of training regimes (Q5, H4)

## Question
Where do saddle-to-saddle (S2S), rich non-incremental and lazy dynamics occur as functions of the
per-core init scale sigma, the bond dimension chi, the number of cores K and the root/rest
imbalance, and do the boundaries follow the predictions of `notes/theory_regimes.md`?
Findings: `notes/regime_map.md`. Agent report: `agent-runs/regime-map/REPORT.md`.

## Setup (defaults)
- Whitened Gaussian inputs, exact population loss, odeco teacher [3, 2, 1] at d = 3 or [3, 2] at
  d = 2 (teacher seed 1000 + init seed, independent teacher stream), fan-in init with per-core scale
  sigma, probe set of 128 Gaussian inputs (generator seed 12345) for the tangent kernel. The label
  rule uses this probe kernel (`dK`); comparisons with `notes/theory_regimes.md` (P5-P7, P16, P20) use
  the tensor-space kernel J D J^T that its constants refer to (`dK_tensor`, fix round 1; the probe
  value is 0.54-1.11 of it, median 0.78, near the lazy boundary at chi >= 32).
- Models: **centred** (target W* + W_0; the main one-knob family), **plain** (target W*), **two-copy**
  (`antisymmetric_double` + gradient mask, W_0 = 0).
- Labels: rules of `notes/diagnostics.md` (final-rise widths; event time = the last upward crossing of
  1/2 before the component first reaches 0.9, fix round 1), with convergence judged against
  1/2 ||W*||^2 (= L(0) except for the plain model at large sigma; see the report, decision 11), at all
  9 threshold pairs delta_K in {0.05, 0.1, 0.2} x rho in {2, 3, 5}. A run that ends on the floor of
  an unrealizable centred target counts as converged when that floor is below 1e-3 (chi = 6,
  sigma = 1: floors 1.6e-4 to 8.8e-4, labelled rich); "stalled" mixes floors, long or dead routes and
  runs cut off at max_steps (e.g. MPS 8, sigma = 0.1: stopped at 2e4 steps, t = 1.4e9 < T = 5.7e9).

## Dynamics
Gradient flow with the shared library (`ttn.dynamics.integrate_gf`): DP5 (rtol 1e-8, atol 1e-12), or
LSODA with the exact Hessian for long stiff plateaus (`method: auto` = LSODA below sigma = 0.7).
Every event (final rise of a component overlap) is re-integrated from the saved parameters on a
200-point linear grid (`run_regime.refine_events`), so widths are resolved in linear time; the event
time is the last upward crossing of 1/2 there (before fix round 1 a transient above 1/2 that fell back
was kept as the event, `n_up` > 1 flags such runs). Stop
when L < 1e-6 x 1/2 ||W*||^2, at the time budget T = 50 t_esc + 1e4 / max(lambda_bar, 1)
(t_esc = aligned escape-time estimate, lambda_bar = tr Theta(0) / D), or at max_steps. Accuracy
check: conservation-law drift (recorded per run, <= 1e-9 on the N = 4 grid) and an independent
scipy RK45 integrator on 1937 matched runs (`pilots/interim.py`, labels agree in 98.1%).

## Files
| file | role |
|---|---|
| `run_regime.py` | one run (`run_point`: build, integrate, refine events, kernel change, labels) and the parallel resumable sweep loop |
| `sweep.py` | expands YAML configs (`configs/*.yaml`) and runs them; `--n-proc`, `--out`; skips runs already in `<out>/<exp>/results.jsonl` |
| `analysis.py` | labels at all thresholds, per-seed boundaries (`seed_rho_boundary`, `seed_rho_boundary_stalls`, `seed_kernel_boundary`), relabelling of pre-fix rows (`relabel_last_crossing`), tensor-space kernel for pre-fix rows (`add_tensor_kernel`), aligned-mode S2S extrapolation (`aligned_sigma_c`), exact aligned boundary (`aligned_boundary_a0`), stall classification from bond spectra |
| `fix_round1.py` | fix round 1: reruns of the runs the review's fixes need (`run`, local), tensor-space kernel of saved runs from theta_end (`tensor`, cached), fix-round tables and figures in `results/fix1/` (`analyse`) |
| `lazy_theory.py`, `lazy_predict.py`, `pilots/lazy_pred_b8.py` | lazy-limit predictions at init (residual, kappa) by linear response |
| `balance.py` | gauge-balanced init (same W_0, every Q_e = 0) |
| `plots.py`, `make_figures.py` | figure functions and the script that regenerates all figures and tables |
| `report_tables.py` | report tables in `results/report/` from the raw runs (per-run tables of the N = 4 map, the K sweeps and large chi; floors, imbalance map, stall test, sweep inventory) |
| `plot_report.py` | the report figures `results/report_map.pdf` (Sec. 3) and `results/report_other.pdf` (Sec. 5.1-5.4), from committed tables only |
| `report_numbers.py` | the derived numbers quoted in Sec. 3 and 5.1-5.4, from committed tables only (`results/report/numbers.csv`) |
| `pilots/` | throwaway pilots from before the library landed (kept for the record; `interim.py` is the scipy cross-check driver) |
| `results/` | figures, per-seed tables (CSV), lazy predictions |

## How to run (in this order)
The raw-output root is `--out` / `--raw`, default `$REGIME_MAP_OUT`, else `/root/outputs/regime_map`
(the remote machine). Workers are spawned, so the scripts run on Windows too.
```
# 1. sweeps (raw outputs: <raw>/<exp>/{results.jsonl, traj/*.npz}; resumable)
C=experiments/regime_map/configs
uv run python experiments/regime_map/sweep.py $C/sigma_chi_b4.yaml $C/chi_b4.yaml $C/sigma_chi_b8.yaml \
    $C/sigma_chi_b8_d2.yaml $C/sigma_K.yaml $C/sigma_K_large.yaml $C/sigma_K_d2.yaml $C/sigma_K_d2_large.yaml \
    $C/sigma_K_full_d2.yaml $C/sigma_K_centred_d2.yaml $C/imbalance_b4.yaml $C/large_chi_b4.yaml \
    $C/large_chi_b8.yaml $C/stall_test.yaml $C/stall_test_plain.yaml --n-proc 340 [--out RAW]
# 2. lazy-limit predictions at init. Both refuse to overwrite: the committed CSVs exist, so pass an
#    output directory (9th positional argument of lazy_predict.py, 1st of lazy_pred_b8.py)
uv run python experiments/regime_map/lazy_predict.py b4 binary 4 3 3,4,6,9,12,18,24,36 6 nocap sp OUTDIR
uv run python experiments/regime_map/pilots/lazy_pred_b8.py OUTDIR
# 3. figures and tables (results/; pass --out DIR, or --force to overwrite)
uv run python experiments/regime_map/make_figures.py --raw RAW
# fix round 1 on the raw outputs RAW (e.g. the copy data/regime_map/); CACHE: dK_tensor cache dir
uv run python experiments/regime_map/fix_round1.py tensor  --raw RAW --tensor-cache CACHE --n-proc 12
uv run python experiments/regime_map/fix_round1.py analyse --raw RAW --tensor-cache CACHE --local LOCAL --force
REGIME_MAP_TENSOR_CACHE=CACHE uv run python experiments/regime_map/make_figures.py --raw RAW --force
# (fix_round1.py run --raw LOCAL --n-proc 12: the local reruns made before the copy arrived)
# 4. report tables (results/report/, from the raw runs) and report figures (from committed tables only)
uv run python experiments/regime_map/report_tables.py --raw RAW --tensor-cache CACHE [--force]
uv run python experiments/regime_map/report_numbers.py
uv run python experiments/regime_map/plot_report.py
```
Raw outputs are not in git; rerunning step 1 regenerates them (seeded; ~2-4 h on 350 cores). A copy of
the remote outputs is in `data/regime_map/` of the main checkout (2026-10-10, read-only); fix-round
reruns and the tensor-kernel cache in `data/regime_map_local/` (both gitignored).

## Round 2: sigma_c(K) in realizable problems (pre-registered: `notes/regime_map.md` section 4)
Question: does the S2S/rich boundary sigma_c(K) tend to a constant (1 or sqrt(d)) as K grows? Laws
compared: sigma_c^K = const (H_A) vs sigma_c^K / sqrt(D) = const (H_B), through
y = ln(sigma_c^K / u_0c(K)) = -gamma + beta ln sqrt(D) (beta = 0: H_A, beta = 1: H_B).
- Arm P (`r2_plain`): plain model (target W*), chi = 4 = 2 x teacher rank, teacher weights (3, 2),
  `binary_tree(N)` with N = 4..16 (K = 3..15), d = 2 and 3, 6 seeds.
- Arm C (`r2_centred_full`): centred model, full bonds (chi_e = cut dimension), same teacher;
  d = 2 for K = 3..8 and d = 3 for K = 3..4 (larger P was too slow).
- Per seed, `round2.py` walks sigma adaptively until the first S2S-like -> rich-like transition
  (rho* = 3) is bracketed within 2^(1/(K-2)) in sigma. Plain runs whose initial overlap
  max_r |a_r(0)| >= 0.1 are outside the event rule's domain ('D').
```
uv run python experiments/regime_map/round2.py run --arms P --n-proc 300   # raw: /root/outputs/regime_map_r2/
uv run python experiments/regime_map/round2.py run --arms C --n-proc 48
uv run python experiments/regime_map/round2_analysis.py --force           # results/round2/
```
Tensor-kernel labels for the N = 4 maps (notes section 5):
```
R=/root/outputs/regime_map; C=$R/tensor_kernel
uv run python experiments/regime_map/relabel_tensor.py compute --raw $R --cache $C   # dK_tensor of every run
uv run python experiments/regime_map/relabel_tensor.py compare --raw $R --cache $C   # results/tensor_relabel/
export REGIME_MAP_LABEL_KERNEL=tensor REGIME_MAP_TENSOR_CACHE=$C
uv run python experiments/regime_map/make_figures.py --raw $R --force --only sigma_chi_b4 chi_b4 imbalance_b4 summary
uv run python experiments/regime_map/report_tables.py --raw $R --tensor-cache $C --force --only sigma_chi_b4 imbalance
uv run python experiments/regime_map/report_numbers.py && uv run python experiments/regime_map/plot_report.py
```

## Experiments (configs)
| config | what |
|---|---|
| `sigma_chi_b4` | binary N = 4 (K = 3), d = 3: 8 chi (3..36, uncapped) x 19 sigma (0.01-40) x 6 seeds, centred / plain / two-copy |
| `chi_b4` | binary N = 4: chi = 9..128, sigma 0.004-4 (centred) and 0.25-4 (two-copy): boundaries vs chi over > 1 decade |
| `sigma_chi_b8` | binary N = 8 (K = 7), d = 3, deficient bonds chi = 3..24, centred and plain |
| `sigma_chi_b8_d2` | binary N = 8, d = 2, chi = 2..16 and 8/32 (full bonds), centred: the K = 7 lazy region |
| `sigma_K`, `sigma_K_d2`, `sigma_K_full_d2` | sigma x K: binary N = 4, 8, 16, MPS N = 4..12; plain model; chi = teacher rank (stalls), 2x rank, and full bonds |
| `sigma_K_centred_d2` | sigma x K in the centred model (the H4 test): d = 2, teacher [3, 2], chi = 4 on every tree, plus full bonds for MPS 6 / binary 8 / MPS 8 |
| `sigma_K_large`, `sigma_K_d2_large` | the large trees (MPS 10, 12, binary 16) of `sigma_K` / `sigma_K_d2` (same exp), lower-priority pool, step cap 1e4 |
| `imbalance_b4` | root scale lambda at fixed alpha = lambda sigma^K, binary N = 4, chi = 3, 16, 24 |
| `large_chi_b4`, `large_chi_b8` | chi up to 128 (N = 4) / 32 (N = 8, d = 2) under SP, NTK and muP-TTN parameterizations |
| `stall_test`, `stall_test_plain` | fan-in vs gauge-balanced init at chi = teacher rank, K = 7, 8 |

Variants: any key of `run_regime.run_point` (`root_scale`, `lr` in {sp, ntk, mup}, `bonds` by
subtree size, `balance`, `method`, `max_steps`, `tau_max`, `weights`) can be set in `base`, `grid` or
a `variants` entry.

## Results
Full account: `notes/regime_map.md` (section 0: which parameter moves which boundary; section 3: fix
round 1, every changed claim with old -> new and status). Report `agent-runs/regime-map/REPORT.md`. In short (6 seeds per point, per-seed statistics; status tags in
the notes):
- binary N = 4, centred, fan-in: S2S / rich / lazy. Per seed rho ~ sigma^-(1.02-1.05) on the S2S side
  (four seeds with transients above 1/2: -1.09 to -1.27), so each seed's boundary follows from its
  deepest run (a consistency check, not an a-priori prediction). The lazy boundary matches theory P5
  with the tensor-space kernel (inside its bracket at chi = 24-96, at its lower edge at chi = 128; the
  128-probe kernel of the label rule gives ~5% lower boundaries). No lazy regime below the cut dimension (Eckart-Young floor);
- both boundaries ~ chi^-1/2 at N = 4 (chi 16-128, < 1 decade): the map collapses in sigma sqrt(chi),
  the TTN analogue of known width scales (empirical);
- sigma x K (d = 2, centred, chi = 4): resolved up to K = 7 (0.17-0.75); at K >= 8 the label rule
  gives only brackets. With floor-stalled runs classified by rho, K = 8 gives 1.01 (5/6 seeds,
  deficient bonds). The K = 10 and 12 brackets mostly end at runs that learned nothing, so they are not
  boundaries (verification erratum E1). sigma^K and sigma^K / sqrt(D) are indistinguishable on these
  data. sigma_c -> sqrt(d) is a conjecture, and stalls take over at large K;
- round 2 (realizable, d = 2 and 3, `results/round2/`, notes 4.7-4.8):
  - Plain model with spare channels: at K >= 8 most seeds stay S2S up to the event rule's domain
    limit (the sigma where W_0's overlap with a teacher component reaches 0.1, which tends to sqrt(d)),
    with lower bounds 1.2-2.0. So sigma_c does not tend to 1.
  - Mechanism: the plain model shrinks its cores (unlearns W_0) on the plateau once
    sigma^K sqrt(D) / (s d) > ~1.
  - Centred model with full bonds (K <= 8 at d = 2, K <= 4 at d = 3): sigma^K and sigma^K / sqrt(D)
    are not resolved (beta = -0.07 +- 0.51).
  - Run time: 0.1-0.4 h per run at K = 13-15 (plain, chi = 4); about 1 h at full bonds with P = 2349;
- large chi (tensor kernel): delta_K ~ chi^-3.02 (SP), NTK chi^-1 at chi >= 64, muP-TTN delta_K -> 6.2
  (theory P16);
- imbalance: light root -> lazy (delta_K ~ lambda^2, saturating at 0.6, the lower edge of P20's
  0.6-0.7); with tensor-kernel labels a heavy root is lazy from alpha = 100, as at balance, a light root
  from alpha = 1;
- the N = 4 maps (sigma_chi_b4, chi_b4, imbalance_b4) and the report tables / figures built from them
  are labelled with the tensor-space kernel since round 2 (`relabel_tensor.py`,
  `REGIME_MAP_LABEL_KERNEL=tensor`; 65 of 5,298 labels change, lazy boundaries +4-9%; notes section 5).

Figures and tables in `results/` were regenerated in fix round 1 by
`make_figures.py --raw data/regime_map --force` (the copied remote outputs, relabelled with the
last-crossing rule; `REGIME_MAP_TENSOR_CACHE=data/regime_map_local/tensor_kernel` adds the tensor-space
kernel): `summary.png`, `phase_sigma_chi_b4_*.png`, `sensitivity_sigma_chi_b4_*.png`,
`boundaries_vs_chi_chi_b4.png`, `phase_sigma_chi_b8*.png`, `phase_sigma_K_*.png`, `sigma_c_vs_K.png`,
`stalls_vs_K.png`, `phase_imbalance_b4.png`, `large_chi_*.png`, `examples_b4_centred.png`, with the
per-seed CSV tables next to them. Fix-round tables and figures: `results/fix1/` (written by
`fix_round1.py analyse`; `results/fix1/prefix/` keeps the pre-fix per-seed CSVs that the reproduction
check compares against). `results/interim/` holds the first (scipy, log-grid) N = 4 map of the interim
report (superseded). In the sigma x K phase plots the "x" (aligned extrapolation from the deepest run)
is not meaningful for trees whose deepest runs stalled.
