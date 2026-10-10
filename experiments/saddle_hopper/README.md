# saddle_hopper: the saddle-to-saddle sequence predicted from the theory, tested on the committed runs

Agent `saddle-hopper`; v2 after the audit `notes/reviews/saddle_hopper_verification.md`. Method,
results, failures and status tags: `notes/saddle_hopper.md`. **Status: bookkeeping of known theory**
(the predictions re-derive the experiments' own), except the TC multi-saddle slopes and the
saddle-order test (`gp_*.py`, note section 9).

## Question

Can the team's theory (`notes/theory_saddles.md`, `theory_escape.md`, `theory_teachers.md`,
`theory_covariance.md`) be turned into an algorithm that predicts, for a given tree, bond dimension,
teacher, input covariance and init, the whole saddle-to-saddle (S2S) sequence of gradient flow: which
component or route is learned next (or that the order is random / rare), the plateau levels, the route
degrees k, the time laws (sigma^-(K-2), log(1/sigma) with slope, sigma^-(k-2)), the saddle types, and
the failure modes (dead routes, stars and forgetting, wrong-order strict saddles)? And do the runs the
team already made follow the predicted sequences?

## What the hopper does (rules in `notes/saddle_hopper.md` section 1)

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

Known gap: the hopper sets saddle order = k* (plus the classical piggyback). `notes/theory_saddle_order.md`
shows the order is 2 l < k* when the generalized piggyback operator exceeds s_q at a level l < k*/2;
`gp_check.py` computes it at every rank-one saddle (note section 9). Do not cite hopper exponents on
wrong-order or multi-residual saddles with k* >= 5.

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
| `validate_order.py` | `experiments/order` (fix round): wrong-order exponents gamma, H1b rotation, R = 3, 4 orders |
| `summarize_results.py` | summary tables, baselines (hopper vs trivial order rules), mismatch list with causes, figures |
| `gp_check.py` | generalized piggyback operator (theory_saddle_order) at every rank-one saddle of the scored teachers: k*, lambda_max(S\|L_l)/s_q, predicted order |
| `gp_constants.py` | leading constants m(t)/t^p at the order-6 and tie saddles (sphere minima of the theory check) |
| `gp_escape_direction.py` | escape direction at those saddles from stored trajectories (loss-overlap ratio, mixed patterns, forgetting) |

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
uv run python $X/gp_check.py --workers 8                 # saddle-order test (a few minutes)
uv run python $X/gp_constants.py                         # a few minutes, one process
uv run python $X/gp_escape_direction.py                  # needs data/order, data/s2s_core, data/locality (DATA_ROOT)
```

Inputs: the committed tables in `experiments/{locality,s2s_core,data_teacher,order}/results/`. The
per-run sequences of `locality` exist only in its raw run records (JSON lines written by
`experiments/locality/run.py`), read from `data/locality/` (override with `LOCALITY_OUT`) and
restricted to the committed configs exactly as locality's own tables are. No new gradient-flow runs
are made. Cost: about 0.1-45 s per case; the whole set (about 420 cases, from empty caches) took 5.4
minutes on 8 processes in the audit's re-run (machine-dependent); the outputs are deterministic
(byte-identical re-run). From a worktree, point `LOCALITY_OUT` / `DATA_ROOT` at the main checkout's
`data/`.

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
| `baselines.csv` | hopper (main + rare, strict main, trap-flag stalls) vs B1 / B1r / B2 per subset (note section 4) |
| `gp_saddle_order.csv`, `gp_later_saddles.csv` | k*, lambda_max(S\|L_l)/s_q and predicted order at the 587 rank-one saddles; hopper events with k >= 5 from saddles with several learned components |
| `gp_constants.csv` | m(t)/t^p and the single-route value at the order-6 and tie saddles |
| `gp_escape_direction.csv` | per escape: loss-overlap ratio x, mixed-pattern ratio, forgetting (order h1b, s2s_core, locality) |
| `fig_laws.png` | predicted vs measured log slopes, exponents and Hessian eigenvalues |
| `fig_match.png` | fraction of scored runs on a main branch, a rare branch, or mismatched, per teacher family |

In short (strict rates, next to the trivial baseline "weight order, no rotation" = B1;
`results/baselines.csv`):
- **locality** (11,432 scored runs): 10,437 main branch without stalling (91.3%), 262 rare (2.3%), 645
  stalls admitted only by the k >= 4 trap flag (5.6%; admissible, not predicted), 88 mismatches (0.8%).
  Runs that did not stall: hopper main 97.1% vs B1 91.3%; runs with one main order: 97.0% vs 95.8%;
  fan-in / balanced two-component runs: 96.1% each (the predictions coincide). **The hopper beats the
  baseline only on 107 multi-component runs (B4 mps4, E8 mps8) whose main order is not the weight order
  (88.8% vs 0%), through the route rule.** Citable stall content: no stall at k <= 3 (0 of 4,531).
- **s2s_core**: binary4 is the only informative cell: 96.6% main, equal to B1. The 13 rare runs (1-3-2)
  depend on rule R4c, added after reading the s2s_core notes; 288 trapped runs match only through the
  trap flag; 425 runs are unscored; the rotation flag is not scored.
- **data_teacher** "174/174" = 112 static recomputations at data_teacher's own saddles + 58
  reproductions of data_teacher's own predictions + 4 new (weak) rows: implementation agreement.
- **order** (fix round, re-run after fixing the H1b set filter): values unchanged; the gamma and H1b
  rows reproduce the order agent's own predictions.
- **Mismatch causes are post hoc.** "No mismatch needs theory wrong" is withdrawn: the 41 balanced-init
  stalls are time-law failures, and 19 mismatches (12 of the 13 "transient dips") are
  generalized-piggyback escapes with forgetting at order-6 and tie saddles (note section 9).
- **Saddle-order test** (`gp_check.py`): 30 of 587 rank-one saddles of the scored teachers have order 6
  < k* = 7 (wrong-first saddles on ht4, ht8, binary8; none at a correct-order saddle, none on an MPS);
  the old runs show the polymer escape there (loss-overlap ratio 0.2998 vs the predicted 0.300 in the
  s2s_core ht4 seed whose gap exponent is 4.03 instead of 5), and it does not explain the 5 steep MPS
  exponents or any stall.
