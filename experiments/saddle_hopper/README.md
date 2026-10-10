# saddle_hopper: the saddle-to-saddle sequence predicted from the theory, tested on the committed runs

Agent `saddle-hopper`. Method, results, failures and status tags: `notes/saddle_hopper.md`.

## Question

Can the team's theory (`notes/theory_saddles.md`, `theory_escape.md`, `theory_teachers.md`,
`theory_covariance.md`) be turned into an algorithm that predicts, for a given tree, bond dimension,
teacher, input covariance and init, the whole saddle-to-saddle (S2S) sequence of gradient flow: which
component or route is learned next (or that the order is random / rare), the plateau levels, the route
degrees k, the time laws (sigma^-(K-2), log(1/sigma) with slope, sigma^-(k-2)), the saddle types, and
the failure modes (dead routes, stars and forgetting, wrong-order strict saddles)? And do the runs the
team already made follow the predicted sequences?

## What the hopper does (rules in `notes/saddle_hopper.md` section 1, fixed before the comparisons)

1. First escape: fixed points of the grouped rank-one problem of `Sigma W*` (alternating power
   iteration), classified by the Riemannian Hessian of `g = <W, Sigma W*>` in the full student
   parameterization: attractors (deterministic if one, random race if several) and rare branches
   (P ~ sigma^gamma).
2. At each saddle (a balanced critical point of the loss at fixed bond ranks, found by descent from
   where the previous escape left): the fixed-rank block (Hessian at fixed ranks) and the local
   residuals R_C of every connected core set C with room; k* = min |C| with R_C != 0.
3. Next event: rotation (fixed-rank instability), else the accumulated-utility race of the k = 2 routes
   (head start from the first escape), else the k*-route(s) (several attractors of the route problem:
   random; saddles of the route problem: rare); the new saddle is the fixed-rank critical point reached
   from the route direction.
4. Per event: law and exponent / slope, saddle type, learned and forgotten components, flags (trap risk
   for k >= 4 under unbalanced init, forgetting, blocked routes, near ties).

Everything is numerical on dense tensors (N <= 8), so pattern, odeco, CP and TTN teachers and product
covariances go through one code path. Library `ttn/` only (not modified).

## Files

| file | role |
|---|---|
| `hopper.py` | the predictor (`Setting`, `SaddleHopper(setting).run()`, `summarize(sequence)`) |
| `cases.py` | teachers of the experiments rebuilt for the hopper (pattern, odeco, CP with overlaps, misaligned covariance, random TTN teachers with the experiments' streams) |
| `predict.py` | cached parallel runner (`run_many`), predicted sequence -> allowed observable orders (`orders`, `allowed`) |
| `examples.py` | the worked examples of the theory notes (B4, E8, N4, stars, wrong order, mixed and symmetric CP teachers): hopper vs the numbers stated in the notes |
| `validate_locality.py` | per-run sequences of `experiments/locality` (two- and multi-component pattern teachers, 9 trees), exponent / slope / Hessian tables, plateau levels |
| `validate_s2s_core.py` | per-run orders and classes of `experiments/s2s_core` (odeco, R = 3, d = 4), exponents, strict wrong-order saddles |
| `validate_data_teacher.py` | `experiments/data_teacher`: generic teachers from the saddle (A) and from random init (B), stars (C), aligned covariance (D), MPS teachers on a binary-8 student (E) |
| `validate_order.py` | `experiments/order` (round 1): wrong-order exponents gamma, H1b rotation, R = 3, 4 orders |
| `summarize_results.py` | summary tables, mismatch list with causes, figures |

## How to run

From the repository root (each script uses `--workers` processes, one torch thread each; predictions
are cached in `results/predictions_*.json`, delete them to recompute):

```bash
X=experiments/saddle_hopper
uv run python $X/examples.py --workers 8
uv run python $X/validate_locality.py --workers 8        # needs the raw locality records (below)
uv run python $X/validate_s2s_core.py --workers 8
uv run python $X/validate_data_teacher.py --workers 8    # --parts ABCDE
uv run python $X/validate_order.py --workers 8
uv run python $X/summarize_results.py                    # tables + figures (argument "causes": exponent causes only)
```

Inputs: the committed tables in `experiments/{locality,s2s_core,data_teacher,order}/results/`. The
per-run sequences of `locality` exist only in its raw run records (JSON lines written by
`experiments/locality/run.py`), read from `data/locality/` (override with `LOCALITY_OUT`) and
restricted to the committed configs exactly as locality's own tables are. No new gradient-flow runs
are made. Cost: about 0.1-45 s per case; the whole set (about 420 cases, from empty caches) took 8
minutes on 8 processes; the outputs are deterministic (re-run identical).

Using the hopper on a new problem:

```python
import sys; sys.path.insert(0, "experiments/saddle_hopper")
import cases as C
from hopper import SaddleHopper, summarize
from ttn.tree import make_tree
W, comps = C.pattern([[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]], [1.0, 0.3, 0.6], d=2)
out = SaddleHopper(C.setting(make_tree("mps", 4), 2, 3, W, comps, init="fanin")).run()
for s in out["sequences"]:
    print(s["pclass"], summarize(s))
# main [sigma^-2] L=0.225 +0 -> [log 2.33] L=0.18 +1 -> [sigma^-2] L=3.3e-28 +2  (converged)
# rare [sigma^-2] L=0.545 +2 -> [rot 0.31] L=0.225 +0-2 -> ...   (q_B first: P ~ sigma^0.667)
```

Each event carries `kind` (first / route / rotation), `k`, `C` (route cores), `rate`, `law`,
`exponent`, `slope_small` (with head start) and `slope_saddle` (from an exact saddle),
`aligned_prefactor`, `type` (saddle type of the plateau it leaves), `lam_old` (fixed-rank block),
`from_level` / `to_level`, `learned` / `newly_learned` / `forgotten`, `branch` and `flags`.

## Results (`results/`; numbers and discussion in `notes/saddle_hopper.md`)

| file | content |
|---|---|
| `theory_examples.csv`, `theory_sequences.txt` | hopper vs the worked examples of the theory notes (all reproduced) |
| `locality_runs_scored.csv` | 11,446 locality runs: observed order, class, result (main / rare / mismatch / unscored), cause |
| `locality_twocomp_laws.csv`, `locality_levels.csv`, `locality_rare_trend.csv` | exponents, log slopes, Hessians, plateau levels; rare-branch frequency vs sigma |
| `s2s_core_runs_scored.csv`, `s2s_core_match_summary.csv`, `s2s_core_exponents.csv`, `s2s_core_strict.csv` | s2s_core |
| `data_teacher_saddle_start.csv`, `data_teacher_random_init.csv`, `data_teacher_star_hessian.csv`, `data_teacher_star_classes.csv`, `data_teacher_aligned_cov.csv`, `data_teacher_cross_tree.csv` | data_teacher parts A-E |
| `order_sigma_exponents.csv`, `order_h1b.csv`, `order_multi.csv` | order |
| `summary_sequences.csv`, `summary_twocomp_per_tree.csv`, `summary_laws.csv`, `mismatches.csv`, `exponent_mismatches.csv` | summaries |
| `fig_laws.png` | predicted vs measured log slopes, exponents and Hessian eigenvalues |
| `fig_match.png` | fraction of scored runs on a main branch, a rare branch, or mismatched, per teacher family |

In short: 99.2% of the 11,432 scored locality runs and all 1,151 scored s2s_core runs follow a
predicted sequence (96.9% and 98.9% a main branch); every sequence mismatch has an identified cause
(unresolved near-simultaneous events, transient dips, slow stalls under balanced init); data_teacher's
generic teachers pass 174/174 law checks (log slopes from the saddle within 0.4% for 10 of 12 teachers,
from random init within 1.3% for the r2 teachers); the locality exponents that miss k - 2 are the known
few-seed, stall-dropout, pre-asymptotic and steep-MPS rows. The order experiment's fix round had not
landed on main when this was run: re-run `validate_order.py` and `summarize_results.py` when it does.
