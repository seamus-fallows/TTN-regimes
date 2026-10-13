# route_escape: leaving a known multi-pattern plateau

**Question.** Start a tree tensor network exactly at a plateau theta_J, where it represents a set J of
learned patterns of a pattern target, add a perturbation of size eps, and measure (a) the order of
the saddle (how steeply the loss can first go down), (b) the escape time against eps, and (c) which
missing pattern is learned first. Do these match the route theory of
`report/ttn_saddle_order_and_aligned_training_dynamics.pdf` (Thm 8.4: order c*_J inside the aligned
manifold M*, at most c*_J outside; eq. 6.16 / C.6: escape time eps^-(c-2) / ((c-2) s_r A_r(J)), log
law for c = 2; the greedy rule; Cor. 8.6; Thm 9.2), and where do they fail?
Brief: `notes/briefs/route_theory.md` section 4.2. Pre-registration and results: `notes/route_escape.md`.
Report: `agent-runs/route-escape/REPORT.md`.

**Short answer.** See `notes/route_escape.md` Part II (filled in when the sweep is analysed).

## Setup

- Targets: sums of one-hot product tensors ("patterns") with weights, standard site basis, all
  satisfying the edgewise star condition. 15 cells = (target, learned set J), listed in
  `configs/cells.yaml` and `results/cells.csv`: 11 dynamics cells (mps6, mps7, ht4, binary8; route
  depths 2-5; ties; two missing patterns; Corollary 8.6 pair; a predicted spectral shortcut) and 4
  static-only Theorem 9.2 instances.
- Network: TTN on the cell's tree, every bond = max number of star blocks + 2 (spare channels),
  float64, exact population loss L = 1/2 ||W - W*||^2 (whitened inputs), evaluated cancellation-free
  about a reference point on M* (`common.py`).
- Start: the balanced partial-target point theta_J = `AlignedManifold.partial_target(J, relax_from=1e-2)`.
- Static order (`static.py`): m(t) = min of L - L_J over the sphere of radius t around theta_J, inside
  M* and in the full space; order = slope of log|m| against log t.
- Dynamics (`escape.py`, `driver.py`): gradient flow (LSODA, rtol 1e-10, atol 1e-15, exact Hessian
  Jacobian, events on solver steps) from theta_J + eps xi, kinds A (positive dormant entries of M*),
  A_bal (all dormant entries = eps), B (dormant entries of M* with random signs), C (Gaussian on every
  parameter); 16 directions per kind (1 for A_bal); eps = 10^-1 ... 10^-4.5 in quarter decades. For
  A, A_bal, B also the reduced flow on M* and the isolated-route prediction.

## How to run

    uv run python experiments/route_escape/sweep.py --workers 118          # all static + escape runs (resumable)
    uv run python experiments/route_escape/escape.py S3 A 0 -2             # one run, prints a summary
    uv run python experiments/route_escape/static.py PS2 full              # one static order
    uv run python experiments/route_escape/analyze.py                      # CSVs in results/
    uv run python experiments/route_escape/figures.py                      # figures in results/

Raw outputs (one JSON per run, ~8100 files) go to `/root/outputs/route-escape/{escape,static}/`
(`ROUTE_ESCAPE_OUT` overrides); they are not in git. The full sweep took about 1 h on 118 cores.

## Results files (`results/`)

All times are gradient-flow time (dimensionless, the loss is 1/2 ||W - W*||^2). Identifying columns
in every file: `cell` (cell name), `target` (patterns with weights, `pattern:weight`), `tree`, `d`
(physical dimension), `chi` (bond dimension of every edge), `J` (learned patterns at the start);
escape files add `kind`, `seed` (direction index) and `eps`.

### cells.csv (one row per cell and missing pattern)
| column | meaning |
|---|---|
| role | `dyn` (static + dynamics) or `thm92` (static only) |
| n_params, n_entries | number of TTN parameters; number of coordinates of M* |
| L_J | loss at theta_J = 1/2 sum of s_r^2 over missing r |
| c_star | c*_J = smallest route depth over the missing patterns |
| greedy | the pattern the greedy rule predicts (smallest depth, ties: larger s_r A_r(J)) |
| missing, weight | the missing pattern and its weight s_r |
| depth | route depth c_J(r) = number of cores r cannot borrow from J (= |R_J(r)|) |
| region | routing region R_J(r) (node indices of the tree) |
| pairwise | min over a in J of |St(V_ar)| (cost of budding r from one learned pattern) |
| A, sA | A_r(J) = product of r's active entries at the balanced theta_J; s_r A_r(J) |
| shared_dormant | number of r's dormant entries shared with another missing pattern |
| parent, parent_k_star, parent_l_star, parent_order | Thm 9.2 bookkeeping: the parent of r (config, or smallest pair cost), k*, l* and min(k*, 2 l*) of that parent's one-component problem |
| thm92_formula, thm92_family | min over parents of the parent orders; whether the cell belongs to the Thm 9.2 family (only then is the formula a prediction) |

### static_curves.csv (one row per cell, space and radius t)
| column | meaning |
|---|---|
| space | `Mstar` (perturbation restricted to M*) or `full` |
| t | radius of the sphere around theta_J (Euclidean norm in parameter space) |
| m | min of L - L_J on the sphere (negative = descent) |
| m_over_tc | m / t^(c*_J) |
| local_slope | slope of log|m| between this t and the previous (larger) t |
| w_active, w_dormant, w_offM | fraction of the minimiser's squared norm on active entries, dormant entries of M*, entries outside M* |
| D_offtarget_frac | fraction of ||W(min) - W(theta_J)||^2 on non-target strings |

### static_orders.csv (one row per cell and space)
| column | meaning |
|---|---|
| order | measured order: local slope at the two smallest t (0.0175 -> 0.0125) |
| order_prev | the local slope one step earlier (0.025 -> 0.0175) |
| resolved | |order - order_prev| <= 0.1 |
| decision | Mstar: "P1 holds" if |order - c*_J| <= 0.2; full: "= c*_J", "shortcut" (order < c*_J - 0.5) or "unresolved" |
| *_smallest | minimiser description at the smallest t (as in static_curves); D_top_offtarget / D_target: largest non-target and the target entries of W(min) - W(theta_J) |
| hess_lam_min, hess_n_neg, hess_n_zero, hess_lam_min_pos | Hessian of L at theta_J (full space): smallest eigenvalue, number < -1e-10, number within 1e-10 of 0 (gauge and flat directions), smallest positive |
| hess_lam_min_Mstar, hess_n_neg_Mstar | the same restricted to M* |

### thm92.csv (one row per Thm 9.2 instance and parent)
| column | meaning |
|---|---|
| parent, parent_weight, residuals | learned parent a, s_a, and the missing patterns assigned to it |
| k_star, l_star, mu | one-component quantities of `RouteTarget.one_component_order(a, residuals)`: smallest Steiner size, first spectral level with mu_l > s_a (empty = none), mu_l per level |
| parent_order, mechanism | min(k*_a, 2 l*_a); direct or spectral |
| formula | min over parents (Thm 9.2 prediction) |
| order_full, order_full_resolved, order_Mstar | measured static orders |
| holds | |order_full - formula| <= 0.2 |

### escape_runs.csv (one row per run)
| column | meaning |
|---|---|
| kind | A, A_bal, B or C (see Setup) |
| seed | direction index (the same direction at every eps) |
| log_eps, eps | perturbation size: dormant entries ~ eps (A, B: eps x N(0,1) magnitudes; C: eps x N(0,1) on every parameter) |
| c_star, greedy, greedy_depth | as in cells.csv |
| outcome | escaped; floor_flagged (escaped, but roundoff could have moved the dormant entries by > 1 %: excluded from fits); trapped (trap rule: L - L_J in (0, 1e-30) or all missing amplitudes < 1e-45 for a decade after t > 1e6); unfinished (wall clock 1800 s); failed (LSODA failure) |
| status | driver status: learned, escaped_no_learn, trapped, wall, failed, t_max |
| t_esc | escape time: first t with L < L_J - 1e-3 (refined to 1e-10 relative) |
| t_esc_1e4, t_esc_1e2 | the same for thresholds 1e-4 and 1e-2 (linear interpolation between solver steps) |
| floor_phi | floor index (accumulated roundoff bound on the non-active entries / their norm); flag > 0.01 |
| first_learned, first_depth, t_learn | first missing pattern whose amplitude P_r = <U_r, W> crosses s_r/2 upward; its route depth; the time |
| first_is_greedy | first_learned == greedy |
| esc_pattern | missing pattern with the largest s_r P_r at t_esc |
| offmass_esc | off-target mass at t_esc: ||W on non-target strings||^2 / ||W||^2 |
| eff_depth | effective depth of the first learned pattern (`ttn.routes.effective_depth`, window P/s in [1e-4, 1e-2], before its event); empty if fewer than 6 points |
| forgotten | learned patterns whose amplitude fell below s_a/4 before the run stopped |
| J_end | patterns with P > s/2 when the run stopped (200 time units after the first learning event) |
| P_greedy_0 | amplitude of the greedy pattern at t = 0 |
| t_red, first_red, status_red | reduced flow on M* (A, A_bal, B): escape time, first learned, status |
| rel_full_red | t_esc / t_red - 1 |
| t_iso, t_616 | isolated-route prediction for the greedy pattern from the start's own dormant factors (eqs. 6.14-6.16 with Prop. 6.2; same threshold); leading term of eq. 6.16 with b(0) = eps |
| iso_winner, t_iso_winner | the missing pattern with the smallest t_iso and its t_iso |
| t_iso_first, ratio_iso | t_iso of the pattern actually learned first; t_esc / t_iso_first |
| ratio_616 | t_esc / t_616 (meaningful for A_bal) |
| P_greedy_0_sign | sign of the greedy pattern's amplitude at t = 0 (A, A_bal, B) |
| route_min/max/gm_{t0,sw,late}, P_{t0,sw,late}, t_*, align_* | diagnostics of the greedy route from `AlignedManifold.align` + `route_entry_norms`: smallest / largest / geometric-mean route-entry norm divided by eps, the amplitude P_r, the time, and the alignment quality (smallest singular value of the normalised learned messages) at t = 0, at the first checkpoint after t = 200 (`sw`) and at the last checkpoint before t_esc / 3 (`late`) |
| n_steps, n_restart, n_fails, secs | solver steps, clock restarts, failed steps, wall seconds |

### escape_fits.csv (one row per direction: cell, kind, seed)
| column | meaning |
|---|---|
| order_full_static, p_pred, gamma_pred | measured static full-space order; predicted order (c*_J for A, A_bal, B; the static full-space order rounded for C); predicted exponent p - 2 |
| n_window, n_escaped, n_trapped, n_flagged, n_unfinished, n_failed | runs in the fit window log10 eps in [-4.5, -3] (7 points) by outcome |
| complete | all 7 window runs escaped without floor flag |
| gamma | OLS slope of log t_esc on log(1/eps) over the window (the escape exponent) |
| local_first, local_last | local exponents at the largest and smallest window pair |
| log_slope, log_slope_pred | slope of t_esc against ln(1/eps) over the window; for depth-2 greedy routes the prediction 1/(s_r A_r(J)) |
| dev, shift_int | gamma - gamma_pred and its nearest integer |
| class | at_prediction (|dev| <= 0.3), shifted, trapped (a window run trapped), incomplete |
| first_at_smallest | first learned pattern at the smallest eps |
| greedy_frac_window | fraction of window runs whose first learned pattern is the greedy one |
| ratio_iso_smallest, ratio_616_smallest, max_abs_ratio_iso_minus1, max_rel_full_red | prefactor tests (A, A_bal, B): t_esc / t_iso and t_esc / t_616 at the smallest eps, the largest |t_esc / t_iso - 1| and |t_esc / t_red - 1| over the window |
| offmass_esc_smallest, eff_depth_smallest | off-target mass at escape and effective depth at the smallest eps |
| *_smallest | route diagnostics of escape_runs.csv at the smallest window eps |

### decisions.csv (one row per cell and kind)
Counts of directions per class, the distribution of gamma over complete directions (median, quartiles,
min, max, fraction within 0.3 of the prediction), the P5 decision (SUPPORTED / REFUTED /
INCONCLUSIVE, rules in the note), the greedy fraction over escaped window runs and the P6 decision,
for A/A_bal/B the P3 test (full vs reduced flow: number of runs, largest |t_esc / t_red - 1|, fraction
with the same first pattern) and the isolated-route test (fraction of window runs with
|t_esc / t_iso - 1| <= 0.05, median ratio, fraction whose first pattern is the isolated-route winner),
and run counts by outcome.

## Figures (`results/`)
Listed with their message in `notes/route_escape.md` Part II.
