# s2s_core: saddle-to-saddle phenomenology and escape-time scaling (Q1, Q2)

Agent `s2s-core`, brief `notes/briefs/s2s-core.md`. Findings, derivations and status tags:
`notes/s2s_core.md`. Running report: `agent-runs/s2s-core/REPORT.md` (on the agent branch).

## Question

Does gradient flow on a TTN from small initialization show saddle-to-saddle (S2S) dynamics, which
saddles does it visit, and how do plateau durations scale with the init scale sigma, the tree (K
cores) and the bond dimension chi, against theory (`notes/theory_escape.md`,
`notes/theory_saddles.md`)? Added by the lead: do dormant components get suppressed during earlier
escapes and plateaus (which would turn long plateaus into stalls)?

## Setup (common to all parts)

- Model as in `notes/model.md`: whitened Gaussian inputs, exact population loss, d = 4 at every site,
  odeco teachers (`ttn.teachers.odeco_teacher`, teacher seed 1000), weights "separated" [3, 2, 1] and
  "close" [1, 0.85, 0.7].
- Trees: binary N = 4 (K = 3), MPS N = 4 (K = 4), binary N = 8 (K = 7), HT N = 4 (K = 7), MPS N = 8
  (K = 8). Bond dimension chi = 3 (= number of components) and chi = 6, capped at the largest rank of
  each bipartition (`TTNSpec.uniform`).
- Inits: fan-in (`spec.init(sigma, seed)`, seeds 0-9); gauge-balanced (same W_0, every Q_e = 0,
  `s2s.gauge_balance`); aligned balanced (Saxe-type, `ttn.network.aligned_init`).
- Dynamics: gradient flow. Random-init runs: `ttn.dynamics.integrate_gf(method="lsoda")` (exact
  Hessian Jacobian; later plateaus are stiff), rtol 1e-8, atol 1e-11 sigma. Aligned runs: DP5
  (`method="dopri5"`), rtol 1e-10, atol 1e-14, and LSODA as a cross-check.
- Events: `a_r(t) = component_overlaps`, event = final rise through 1/2 (last upward crossing),
  width = 0.1 -> 0.9 of the final rise (`notes/diagnostics.md`), both refined by re-integrating the
  event window on linear grids until the rise holds >= 40 samples (`s2s.zoom_rise`). No duration is
  read off the log output grid. Plateau midpoints = minimum parameter speed between events; the
  gauge-projected Hessian is evaluated there (`ttn.diagnostics.project_out_gauge`).
- First-escape prediction (P1 / theory_escape E1): blow-up time T of the correlation flow
  theta' = grad <W(theta), W*> from the sigma = 1 init (`s2s.correlation_flow_blowup`), so that
  t_1 = sigma^-(K-2) T + O(1) per seed with no free parameter.

## Parts, scripts and configs

| part | script | config | raw output (`/root/outputs/s2s_core/...`) | results |
|---|---|---|---|---|
| 1. aligned init vs exact solution | `aligned.py`, `plot_aligned.py` | `configs/aligned.yaml` | `aligned/<tree>_<integrator>.npz` | `aligned.csv`, `fig_a_aligned.*` |
| 2-4. full S2S runs, K = 3, 4 (and first K = 7, 8 pass) | `run.py` (+ `s2s.py`) | `configs/main.yaml` | `main/` | via `analyze_main.py` |
| K = 7, 8 at chi = 6, long budget | `run.py` | `configs/large.yaml` | `large/` | via `analyze_main.py` |
| K = 7, 8 at chi = 3, fan-in vs gauge-balanced | `run.py` | `configs/k78.yaml` | `k78/` | via `analyze_main.py` |
| gauge-balanced init, K = 3, 4 | `run.py` | `configs/balanced.yaml` | `balanced/` | via `analyze_main.py` |
| analysis of the four sweeps above | `analyze_main.py`, `analyze_dormant.py`, `slice_trends.py`, `recompute_hessians.py` | - | - | `main_*.csv`, `hess_*.csv`, `dormant_*.csv`, `slice_trends.csv` |
| 3a. first escape vs sigma, per-seed prediction | `first_escape.py`, `recompute_T.py`, `analyze_first.py` | `configs/first.yaml`, `first_k78.yaml` | `first/` | `first_*.csv` |
| 3b. chi, d, K dependence at leading order | `lo_sweep.py`, `plot_lo.py` | `configs/lo_chi.yaml`, `lo_d.yaml`, `lo_K.yaml` | - | `lo_*.csv`, `fig_e_chi.*`, `fig_e2_K.*` |
| stalled runs continued from their final state | `continue_stalled.py` | - | - | `continued_*.csv` |
| tolerance check of run classes | `tolcheck.py` | - | `k78_tight/` | `tolcheck_k78.csv` |
| figures b, c, d, f, g | `plot_run.py`, `plot_scaling.py`, `plot_dormant.py` | - | - | `fig_*` |
| report figures and derived numbers (Sec. 4 of the report) | `plot_report.py`, `report_numbers.py` | - | - | `report_runs.*`, `report_prefactor.*`, `report_numbers.csv` |

## How to run

Every script that writes results takes a required `--out` (so a rerun never overwrites the committed
`results/` by accident; pass `--out experiments/s2s_core/results` to regenerate them) and, where it runs
a pool, `--workers` (one thread per process). Raw outputs go to `/root/outputs/s2s_core/` (option
`--raw` of `run.py`, `first_escape.py`, `aligned.py`). With `R=/root/outputs/s2s_core` and
`O=experiments/s2s_core/results`, in this order:

```
D=experiments/s2s_core; C=$D/configs
# sweeps (CPU hours on 60 CPUs in brackets)
uv run python $D/aligned.py $C/aligned.yaml --out $O                      # (0.01)
uv run python $D/run.py $C/main.yaml --workers 60                         # (~3)
uv run python $D/run.py $C/large.yaml --workers 30                        # (~2)
uv run python $D/run.py $C/k78.yaml --workers 30                          # (~1)
uv run python $D/run.py $C/balanced.yaml --workers 30                     # (~0.2)
uv run python $D/first_escape.py $C/first.yaml --workers 60               # (~1.5)
uv run python $D/first_escape.py $C/first_k78.yaml --workers 30           # (~1.5; add --reverse for a 2nd instance)
uv run python $D/recompute_T.py --raw $R/first                            # only for jobs run before the T fix
for c in lo_d lo_K lo_chi; do uv run python $D/lo_sweep.py $C/$c.yaml --workers 60 --out $O; done
uv run python $D/recompute_hessians.py --raw $R/main --tag hess_main --out $O
uv run python $D/recompute_hessians.py --raw $R/large $R/k78 $R/balanced --tag hess_other --out $O
uv run python $D/continue_stalled.py --raw $R/k78 --names <8 runs, see continued_k78_resolved.csv> --factor 4 --tag _resolved --out $O
uv run python $D/continue_stalled.py --raw $R/k78 --random --seed 1 --n 8 --max_ratio 100 --factor 3 --tag _random --out $O
uv run python $D/continue_stalled.py --raw $R/large --factor 3 --tag _spread --out $O --names \
  mps8_chi6_w3-2-1_fanin_s0.15_seed1 mps8_chi6_w3-2-1_fanin_s0.2_seed9 mps8_chi6_w3-2-1_fanin_s0.1_seed9 \
  mps8_chi6_w3-2-1_fanin_s0.2_seed0 mps8_chi6_w3-2-1_fanin_s0.15_seed3 mps8_chi6_w3-2-1_fanin_s0.15_seed5 \
  mps8_chi6_w3-2-1_fanin_s0.15_seed7 mps8_chi6_w3-2-1_fanin_s0.1_seed5     # see the note below the block
uv run python $D/tolcheck.py --raw $R/k78 --per-class 1 --workers 12 --out $O
# analysis (minutes)
uv run python $D/analyze_main.py --raw $R/main $R/large $R/k78 $R/balanced --hess $O/hess_main.csv $O/hess_other.csv --out $O
uv run python $D/analyze_dormant.py --raw $R/main $R/large $R/k78 $R/balanced --out $O
uv run python $D/slice_trends.py --runs $O/main_runs.csv --out $O
uv run python $D/analyze_first.py --raw $R/first --out $O
# figures
uv run python $D/plot_aligned.py --out $O
uv run python $D/plot_run.py mps4_chi3_w3-2-1_fanin_s0.01_seed0 --out $O
uv run python $D/plot_run.py binary8_chi6_w3-2-1_fanin_s0.1_seed0 --raw $R/large --out $O
uv run python $D/plot_scaling.py --results $O --out $O
uv run python $D/plot_dormant.py --results $O --out $O
uv run python $D/plot_lo.py --results $O --out $O
# report (Sec. 4): compact figures and the derived numbers quoted in the text (from the tables above)
uv run python $D/plot_report.py --raw-main $R/main --raw-large $R/large --results $O --out $O
uv run python $D/report_numbers.py --results $O --out $O
```

**`continued_large_spread.csv` (round-1 artefact; errata E13 of
`notes/reviews/s2s_core_round2_verification.md`).** The earlier command here, `--spread --n 8
--max_ratio 1000 --factor 3`, does not regenerate it: on the finished `large/` folder (28 candidates
with estimate <= 1000x the elapsed time) `--spread` picks a different set (the last 3 of the 8 runs
differ). The same rule reproduces the committed 8 runs exactly on the 19 candidates whose raw files
existed at 22:43 UTC (from the raw file times), i.e. the sample was drawn while `large` was still
running (REPORT log, 22:44). The command above therefore names the 8 runs (`--names`, same rows and
order); no script change is needed. The continuation factor of the committed file was not recorded,
but the file fixes it: its two escape times are exact geometric midpoints of log-grid cells (to 2e-14)
for `--factor 3` and not for 2 or 4, just as the log-grid values in `continued_k78_resolved.csv` and
`continued_k78_random.csv` are midpoints for their documented factors 4 and 3 (all 15 rows). So the
run that "has not escaped at 2x" (s0.2 seed 9, stop `completed`) in fact ran to 3x the estimate
without escaping, as the round-2 verification's rerun with `--factor 3` also found. The current
script also zoom-resolves each escape (`measured_remaining`) and writes the log-grid value to a new
column `measured_remaining_grid`, while the committed file holds the round-1 log-grid values (+-1.8%)
in `measured_remaining`. A rerun therefore reproduces the runs, components, estimates and
`predicted_over_elapsed` exactly; its `measured_remaining_grid` is the counterpart of the committed
`measured_remaining`, and its zoom-resolved `measured_remaining` agrees with that to the grid
resolution. Which runs hit the 3000 s wall limit (checked between chunks) depends on CPU speed.

`run.py` and `first_escape.py` are resumable (finished runs are skipped). `analyze_main.py --hess`
must name the Hessian tables recomputed for the same raw folders (no default: a stale table would
silently replace the new values). Variants: copy a config and change trees, chis, teachers, inits,
sigmas, seeds, tolerances or budgets. The sweep driver is experiment code written before `ttn.runner`
landed.

## Run budget and stopping (affects only runs that do not complete)

t_max = 1e3 x the RMS-overlap escape time of the weakest component. A run also stops when the loss
reaches 1e-9 L(0), after a wall-clock limit (300-2400 s, so the number of "unresolved" runs depends on
CPU speed) or a per-chunk step limit, or when a never-learned component has a dead route end (slice
< 1e-8 sigma for a whole output chunk: the conservation-law trap; added for `k78` and `balanced`).
`main.yaml` also used an amplitude-based roundoff stop (every dormant |<W, U_r>| < 1e-15 for a chunk);
it is too strict at K >= 7 (the amplitude is a product of K per-core amplitudes), so those K = 7, 8
runs are classed "unresolved" unless a slice is dead, and were re-run in `k78` without it. The first
~310 runs of `main.yaml` had no dead-end stop and ran to t_max (their dead slices sit at the ~1e-12
floor for the rest of the run, which is why decay rates are fitted above 1e-9 only).

## Raw outputs

`/root/outputs/s2s_core/<config name>/<run>.npz` (log-grid trajectory: loss, overlaps, spectra, core
norms, slice norms, per-edge imbalance; refined event windows; parameters at plateau midpoints) and
`<run>.json` (summary: events, widths, order, separation ratios, plateaus with Hessian spectra,
dormant amplitudes). Not in git; regenerate with the commands above.

## Results

Full statements with status tags and numbers: `notes/s2s_core.md` section 2 (section 4 lists the claims
changed after review round 1); summary in `agent-runs/s2s-core/REPORT.md`. In short (odeco teachers,
d = 4, gradient flow):
- Aligned init reproduces the exact mode solutions on all five trees (max |a_GF - a_theory| <= 3e-9).
- First escape (replication of theory_escape E1, E2): t_1 = sigma^-(K-2) T(theta_hat) + c_K per seed with
  no free parameter (exponent 0.9996 (binary4 separated; 0.990-0.996 close), 2.0000, 4.999, 5.9995 for
  K = 3, 4, 7, 8; prefactor to <= 2.2e-3 at the smallest sigma; first component = the correlation flow's
  in 1847/1848 runs); the Kumar-Haupt rho_0 law is a lower bound off by 3-2700x (K = 3-8) and up to 3e7
  at K = 15; at full bond the binary4 prefactor is pi/2 independent of d; the RMS-overlap heuristic P3
  tracks the K dependence within 2x.
- Random init, as measured: complete S2S sequences on binary4, mps4 (chi = 3, 6) and on binary8, ht4 at
  chi = 6 (0.6 decade of sigma); mps8 never learns a third component (0/284). Plateaus at the partial-fit
  levels, one rank step per event on every edge at once (497/531 tested runs). Gaps on runs classified
  by order and S2S status: K - 2 per seed on binary4 (+-0.1) and mps4 (+-0.03); binary8 4.88-5.09, ht4
  3.9-5.8 per seed (chi = 6, 0.6 decade); binary8 gap 1 at chi = 3 4.93 +- 0.05 (up to 1.1 decades);
  mps8 first gap 5.95 +- 0.05. Plateaus are higher-order saddles (lambda_min ~
  sigma^(K-2)) on greedy binary4 and tensor-like trees; the wrong-order binary4 plateau (close teacher) is
  a strict saddle: -0.106 .. -0.146 at the 9 wrong-order plateaus vs the exact -0.133186 (median ratio
  0.98 for seed 8, 0.93 for seed 5), flat over sigma = 0.005-0.1. binary4 is a depth-3 DLN: its S2S
  sequence, sigma^-1 plateaus and strict wrong-order saddle are known DLN results ([Achour2024];
  [Jacot2021] Lemma 15), so these binary4 results are a replication; new for TTNs are mps4, binary8,
  ht4 and mps8.
- Dormant suppression: escapes shrink dormant components by a median 0.1-0.75 at K >= 4 (x2-3 growth at
  K = 3), with a >100x tail of 2-8% at K = 7, 8, chi = 3 (11% for the mps8 separated later escapes
  only; 0-2% at chi = 6); plateaus kill them
  through the conservation-law trap of fan-in init (dead route ends; decay rate ~ sigma^2 per seed,
  checked at K = 4; the K = 7, 8 rows of `dormant_decay.csv` have no sweep column and duplicate
  main/k78 keys, see notes 2.7c): never at K = 3; at K = 4, chi = 3 fixed per init direction for the
  separated teacher (4/10 seeds at every sigma) but not for the close teacher (14/80 trapped + 8/80
  stalled; seed 7 traps at 6 of 8 sigmas); at K = 7, 8 lower bounds of 30-80% at
  chi = 3; at chi = 6 binary8 0/50, ht4 5/50 (one seed), mps8 7/50 ("chi = 2R" is literal only on binary
  trees: end / leaf bonds are capped at d = 4). Gauge-balanced init: 0/120 trapped where fan-in traps
  (12/30, 47/90), but 69/90 balanced K = 7, 8 runs stall. Stalled runs with live routes are predicted to
  be on ever-longer plateaus: of 23 continued stalled runs 16 escape, at 0.24-2.19x the balanced-route
  estimate, which is exceeded in 4 of the 16 (so it is an estimate, not a bound); 91 of the 106 k78
  stalled runs were never tested, so for most stalled runs this is a conjecture (notes 2.7).

## Figures (results/)

| figure | file |
|---|---|
| (a) aligned init vs exact solution | `fig_a_aligned.*` |
| (b) representative runs | `fig_b_run_mps4_chi3_..._s0.01_seed0.*` (K = 4), `fig_b_run_binary8_chi6_..._s0.1_seed0.*` (K = 7) |
| (c) t_1 vs sigma with per-seed prediction | `fig_c_first_escape_chi{3,6}_{fanin,balanced}.*` |
| (d) gaps vs sigma (S2S-classified run sets of the fits) | `fig_d_gaps_chi{3,6}_{separated,close}.*` |
| (e) prefactor vs chi and vs K (leading order) | `fig_e_chi.*`, `fig_e2_K.*` |
| (f) Hessian lambda_min vs sigma at plateau midpoints | `fig_f_hessian.*` |
| (g) dormant suppression: run classes (trapped = lower bound at K = 7, 8), trap decay rate / sigma^2 (mps4, fitted above 1e-9) | `fig_g_dormant.*` |
