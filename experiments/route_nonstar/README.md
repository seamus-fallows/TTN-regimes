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
| `results/` | CSVs (below) and figures |

## Results

(filled in after the sweep; see `notes/route_nonstar.md` Part II)
