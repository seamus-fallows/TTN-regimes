# Experiment `route_nonstar`: training on pattern targets that break the star condition

**Question.** The saddle-order / aligned-dynamics paper
(`report/ttn_saddle_order_and_aligned_training_dynamics.pdf`) proves its later-saddle results only
for pattern targets that satisfy the *edgewise star condition*. What does gradient-flow training of
a tree tensor network (TTN) from small random init do on pattern targets that **violate** it, and
which of the paper's predictions still describe training? Brief: `notes/briefs/route_theory.md`
section 4.3. Pre-registration, results and deviations: **`notes/route_nonstar.md`**.

Plain-words glossary (details in the note, section 1):

- *pattern target*: W* = sum_a s_a U_a, each U_a a one-hot tensor given by a string of site values
  (e.g. 0110), s_a its weight;
- *star condition*: cut the tree at an edge; join each target string's left part to its right part;
  every connected piece of this graph must be a star. It fails when one pattern shares its left part
  with a second pattern and its right part with a third. Then a network that uses one bond channel
  per piece also produces the *recombinant* string (left part of one pattern + right part of another),
  which is not in the target;
- *partial-target point* (canonical plateau): W = sum over the learned set J of s_a U_a;
- *route depth* c_J(r) (paper): the number of cores that pattern r cannot borrow from learned
  patterns; the plateau before r lasts ~ sigma^-(c - 2);
- *numerical route rule* (`experiments/locality/rules.route_rule`): at the actual plateau point, the
  smallest connected set of cores C such that changing the cores in C alone can lower the loss
  (non-zero local residual R_C); k* = |C| plays the role of the depth;
- *pairwise rule*: the depth of r is the Steiner-subtree size of the sites where r differs from the
  closest learned pattern (Prim's rule, paper eq. C.8);
- *order* p of a plateau: the power of the distance at which the loss first decreases; p = 2 means
  a negative Hessian eigenvalue (strict saddle, plateau ~ log(1/sigma)), p >= 3 a flat
  (degenerate) saddle, plateau ~ sigma^-(p - 2);
- *effective depth*: the d of the scalar route equation dy/dt = d gamma^(2/d) y^(2-2/d) (s - y),
  read off the growth law of an escape (`ttn.routes.effective_depth`).

## Setup

- 20 targets (`configs/targets.yaml`, written by `make_targets.py`; table with star violations and
  predicted sequences in the note, section 2): the recombinant family on chains of 3 and 4 cores
  (R3, R3p, R3d, R4, R4p, R4d), the nested chain 0000, 1100, 1110, 1111 on mps4, binary4 and ht4
  (NC_*; = `experiments/locality`'s TB), two separate local edits on mps6 and binary8 (TD_*), three
  random non-star targets each on mps6 and binary8 (RND*), and star-satisfying controls through the
  same code (B4_mps4, CTL1_mps6, CTL1_binary8). d = 2 everywhere; K = number of cores <= 7.
- Model: whitened inputs, exact population loss 1/2 ||W - W*||^2, standard-basis patterns, fan-in
  Gaussian init `spec.init(sigma, seed)` on all parameters, bond dimension = rank of W* across the
  edge + 2 spare channels. float64; dense contraction (at most 256 entries).
- Gradient flow (`nsdyn.integrate`): scipy `LSODA` class stepped directly with the exact Hessian as
  Jacobian, rtol 1e-10, atol 1e-14; events and plateaus read off solver steps (no dense output); the
  solver clock is re-zeroed at late times (as in `experiments/saddle_dyn/sdyn.py`), and times are
  analysed in long double relative to the start of the first escape. Stop: converged
  (L < 1e-10 ||W*||^2 / 2), trapped (gradient < 1e-14 for two decades of time after a pattern was
  learned), wall (1800 s) or t_max (1e22).
- Grid: log10 sigma = -1 ... -4 (K <= 4) or -1 ... -3.5 (K = 6, 7) in quarter decades; seeds 0-39;
  9,520 runs. Fit window log10 sigma <= -1.5.
- Per run (`run.py`): amplitudes, learning events, loss drops and plateaus
  (`ttn.diagnostics.find_plateaus`, min_drop 0.02), and at every plateau's minimum-gradient point:
  nearest partial target and distance, off-target strings with recombinant tags, Hessian spectrum
  ends, float64-floor flag, route rule; per drop: what enters the learned set, effective depths.

## How to run (from the repo root)

```
uv run python experiments/route_nonstar/make_targets.py          # configs/targets.yaml (1 min)
uv run python experiments/route_nonstar/run.py sweep --workers 120   # 9,520 runs, resumable
uv run python experiments/route_nonstar/analyze.py               # results/*.csv
uv run python experiments/route_nonstar/headline.py              # results/headline.md
uv run python experiments/route_nonstar/figures.py               # results/fig*.png/.pdf
uv run python experiments/route_nonstar/run.py one R3 -2 0       # one run, record to stdout
```

- `run.py sweep` options: `--targets A,B`, `--sigmas -1,-2`, `--seeds 0-9`, `--workers N`,
  `--out DIR` (default `/root/outputs/route-nonstar`, or `$ROUTE_NONSTAR_OUT`). Finished runs
  (existing JSON) are skipped; errors go to `<out>/errors/`.
- Raw output per run (not in git): `<out>/raw/<target>/s<log10 sigma>_seed<k>.json` (the record)
  and `.npz` (every solver step: t relative to the first escape, L, |grad|, amplitudes, off-target
  mass; W at <= 400 steps; theta at every plateau point).
- `analyze.py --raw DIR --out DIR` and `figures.py --raw DIR --res DIR` read / write elsewhere.

## Files

| file | contents |
|---|---|
| `make_targets.py`, `configs/targets.yaml` | target choice, star violations, predicted sequences, grids |
| `nsdyn.py` | model (dense contraction, gradient, Hessian, roundoff bound) and the step-driven LSODA driver |
| `run.py` | one run + its analysis; the resumable sweep |
| `analyze.py` | CSVs, per-seed fits, decisions, escape types, sequence scores |
| `figures.py` | draft figures |
| `headline.py` | cross-target numbers -> `results/headline.md` |
| `load.py` | `read(name)`: load a result CSV with pattern columns as text |
| `results/` | CSVs (below) and figures |

## CSV columns (`results/`)

**Read them with `load.read(name)`** (or `dtype=str` for the pattern columns listed in
`load.PATTERN_COLUMNS`): pattern strings such as `0110` otherwise become the number 110.
All CSVs are long format, one row per observation. Identifying columns (every CSV that has them):
`target` (name in `configs/targets.yaml`), `tree` (kind + number of sites, e.g. `mps6`), `d`
(physical dimension, 2), `chi` (largest bond dimension of the network), `sigma` (init scale),
`log10_sigma`, `seed` (init seed 0-39). Times are gradient-flow time (dimensionless; the loss is
1/2 ||W - W*||^2 with ||W*||^2 = sum of squared weights). Losses in the same units. Pattern strings
list the site values in site order. A *learned set* J is written `{p1,p2,...}`.

**`runs.csv`** (one row per run): `family` (recombinant / nested / two_edits / random / control),
`K` (cores), `n_params`, `status` (converged / trapped / wall = wall-clock cap / t_max / failed),
`order_learned` (patterns in the order of their learning events), `n_learned`, `n_drops` (loss
drops found by `find_plateaus`), `n_plateaus` (resolved plateaus after the first escape),
`n_transients` (unresolved segments merged into escapes, deviation 1), `forgotten` (pattern@time of
every forgetting event: P falls below s/4 after crossing s/2), `n_forgotten`, `floor_flagged`
(some plateau point has |grad| < 100 x its roundoff bound), `L_end`, `off_end` (off-target mass at
the end), `t_first_escape` (absolute time of the first loss drop), `t_end_rel` (end time relative to
the start of the first escape), `n_steps` (solver steps), `n_restart` (solver clock re-zeroings),
`n_fails` (failed steps retried), `secs` (wall-clock seconds).

**`transitions.csv.gz`** (gzipped CSV, read with `pandas.read_csv`; one row per plateau of a run, with what the escape from it does; the
initial plateau from the origin is `segment` 0, `key` "origin"): `segment` (index among the
run's loss segments), `t_start`, `t_end` (times of the half-way points of the loss drops that enter
and leave the plateau, relative to the start of the first escape), `duration` (= t_end - t_start,
the plateau length used in the fits), `censored` (run ended on this plateau), `resolved`,
`t_mid` (minimum-gradient point), `L` (loss there), `key` (plateau identity: learned set, `*` if not
a partial-target point, `@RS<j>` = matches level j of the route-rule sequence within 5 %, else
`@L<loss>`), `J` (learned set {a : P_a > s_a/2}), `n_J`, `canonical` (d_near < 0.03), `level_tag`,
`d_near` (distance ||W - sum_J' s_a U_a|| / ||W*|| to the nearest partial target J'), `J_near`,
`d_amp` (same distance to the partial target of J), `off_mass` (fraction of ||W||^2 outside the
span of the target patterns), `off_strings` (off-target strings with |W[x]| >= 0.01 and their
signed amplitudes), `off_recombinants` (those that are recombinants: `x=e<edge>:a|b` = x equals
target a on the lower side of edge e and target b on the upper side), `n_off_strings`, `lam_min`,
`lam_2` (two smallest Hessian eigenvalues), `n_neg` (eigenvalues < -1e-9), `gnorm` (|grad|),
`grad_bound` (its roundoff bound), `floor_flag`, `route_k` (numerical route rule k*: size of the
smallest connected core set with non-zero local residual), `route_nodes` (that set, node indices of
`ttn.tree`), `route_string` (largest entry of its local residual; the route rule's predicted next
string), `route_string_is_target`, `t_escape` (time from the plateau start until L < L - 1e-3),
`pred_paper` (paper's route depth c*_J = min over missing r of |cap_a St(V_ra)|; empty = nothing
missing; K for the origin), `pred_paper_next` (greedy rule's next pattern), `paper_depths`
(c_J(r) of every missing pattern), `pred_route` (= route_k), `pred_route_next`, `pred_pair`
(pairwise rule: min over missing r of min over a in J of |St(V_ar)|), `pred_pair_next`,
`pred_weight_next` (largest missing weight), `pred_hand` (registered by-hand order: K for the
rank-1 optimum of R3/R4/R3d/R4d, Theorem 3.2's min(k*, 2 l*) for a canonical one-pattern plateau),
`next_learned` / `next_lost` (patterns entering / leaving J during the escape), `next_n_drops`
(sub-drops merged into the escape), `next_top_change` (string whose W entry changes most),
`next_top_change_is_target`, `next_L_after`, `next_off_peak` (largest off-target mass during the
escape), `next_eff_depth_loss` (effective depth from the growth of L_plateau - L(t)),
`next_eff_depth_amp` (pattern:effective depth of each pattern learned in the escape, from its
amplitude), `next_kind` (learn = some pattern enters J; cleanup = none does).

**`events.csv`** (brief's events table, one row per learning event = last upward crossing of
P_a = s_a/2): `pattern`, `weight`, `t_event` (relative to the start of the first escape),
`J_before`, `n_J_before`, `duration` (time since the previous learning event; the first event's is
from t = 0), `drop` (index of the loss drop that contains it), `depth_paper` (c_J(r) for J =
J_before; K if J is empty; 0 = r would be produced for free as a recombinant), `pair_cost`,
`pred_paper_next`, `pred_pair_next`, `pred_weight_next` (each rule's next pattern from J_before),
`route_k_before`, `pred_route_next` (route rule at the last plateau point before the event),
`eff_depth` (amplitude estimator over [previous event, event]; empty if the amplitude passes the
window [1e-4 s, 1e-2 s] too fast), `off_peak` (peak off-target mass in the drop),
`plateau_before_L`, `plateau_before_L_J` (partial-target loss L_J = 1/2 sum of missing s^2),
`plateau_before_d_canon` (distance of the plateau point before to the partial target of J_before),
`same_drop` (see below).

**`plateaus.csv`** (brief's plateau table, one row per (plateau, missing pattern); `missing` =
"none" when J is complete): `segment`, `key`, `J`, `J_near`, `d_near`, `canonical`, `off_mass`,
`off_strings`, `off_recombinants`, `L`, `L_J`, `route_k`, `missing`, `weight`, `depth_paper`,
`pair_cost`, `P_r` (signed amplitude of the missing pattern at the plateau point), and for the
star controls only `entry_min`, `entry_max`, `entry_geo` (minimum, maximum, geometric mean over
the nodes of R_J(r) of the route-entry norms from `align` + `route_entry_norms`, divided by sigma)
and `align_quality` (smallest singular value of the normalised message matrices; 1 = aligned).

**`fits.csv`** (one row per (target, plateau key, seed)): `n_points` (window sigmas at which the
seed visits the key, converged, not floor-flagged), `span` (decades), `exponent` (minus the
least-squares slope of log10 duration on log10 sigma), `rms` (residual, decades), `fitted`
(>= 5 points over >= 1.5 decades), `pred_*` (median predicted order of each rule).
**`fits_secondary.csv`**: the same for the labelled secondary variants (deviation 2), column
`variant` = `unflagged_span1` (>= 1 decade) or `with_flagged` (floor-flagged points included).

**`fit_summary.csv`** (one row per (target, plateau key)): `n_seeds`, `n_fitted`, `median`, `q25`,
`q75` (of the per-seed exponents); for each rule X in paper, route, pair, hand: `order_X`
(predicted order), `exp_X` (predicted exponent max(order - 2, 0); 0 when nothing is missing),
`frac_within_X` (fraction of fitted seeds within 0.3), `decision_X` (holds / partial / fails /
"fails (no plateau predicted)" / n/a, note section 5 and deviation 3), `predicts_plateau_X`; what
the plateau is: `n_points`, `frac_canonical`, `d_near_median`, `off_mass_median`, `L_median`,
`off_recombinants` (most common), `next_kind`, `next_learned` (most common), `eff_depth_loss_median`,
`route_k_median`, `floor_flagged` (count); secondary fits: `n_fitted_*`, `median_*`, `q25_*`,
`q75_*`, `frac_within_{route,paper,hand}_*` for the two variants.

**`event_fits.csv`, `event_fit_summary.csv`**: the same fits on the brief's event-to-event
durations, grouped by `ident` = J_before -> pattern (for comparison with `route-smallinit`).
`events.csv` has `same_drop` (the event lies in the same loss drop as the previous learning event,
so no plateau separates them) and `event_fit_summary.csv` has `frac_same_drop`; where a rule
predicts no plateau (depth <= 1) its decision is "holds (same drop)" if >= 90 % of the events share
the previous event's drop and "fails (plateau before it)" if <= 10 % do. Caveat: when several
patterns cross s/2 within one drop, J_before and the event-to-event "durations" (fractions of a
time unit) depend on crossing order; exponents of such identities are meaningless. The plateau-based
`fits.csv` / `fit_summary.csv` are the primary duration analysis.

**`escape_type.csv`** (one row per (target, key)): `n_points`, `n_sigmas`, `lam_min_at_smallest` /
`lam_min_at_largest` (median lambda_min over seeds at the smallest / largest window sigma),
`all_neg`, `slope_x` (slope of log10 |median lambda_min| on log10 sigma; estimates order - 2),
`type` (strict: lambda_min < -1e-4 at every window sigma and x < 0.25; degenerate: x >= 0.5;
else unclear), `n_neg_median`. Floor-flagged points excluded.

**`sequence.csv`** (one row per (target, sigma)): `hit_paper`, `hit_route`, `hit_pair`, `hit_weight`
(fraction of escapes whose outcome the rule predicts: its next pattern enters J; for clean-up
escapes only the route rule can be right, when its string is the off-target string that changes
most), `n_transitions`.

**`run_classes.csv`** (one row per (target, sigma)): number of runs per status, `floor_flagged`,
`forgetting` (runs with a forgetting event).

**`summary_by_target.md`**: generated per-target tables (plateaus, predictor scores, escape type,
sequence agreement, run counts).

## Results

(filled in after the sweep; see `notes/route_nonstar.md` Part II)
