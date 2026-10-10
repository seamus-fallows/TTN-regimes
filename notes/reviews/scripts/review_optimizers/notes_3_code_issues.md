# Part 3: bugs / analysis issues found in the committed code and tables

1. run.py:124 `mp.get_context("fork")` - README command fails on Windows (ValueError); default --out
   `/root/outputs/optimizers` is created (C:\root\...) even with --dry. [portability]
2. configs/scaling_b4.yaml: the two `precond` entries lack `blowup: 1.0e300` (precond_b4.yaml and deep.yaml
   have it). `Stop` default blowup = 1e6 L0, the first preconditioned step overshoots to 1e14-1e26 L0, so
   42/42 (lr 0.1) and 33/42 (lr 0.01) scaling_b4 precond runs stop as "diverged" after ONE step
   (meta_n_steps = 1). Their `order` (o102, o201, ...) is an interpolation between step 0 and the overshoot;
   make_figures.order_agreement() still uses them -> the "33%" GF-order agreement for Gram-preconditioned GD
   on binary4 (order_agreement_with_gf.csv rows 11-12; REPORT summary table "33% / 19%") is an artifact.
3. plots.per_seed_fits: the "log law" alternative is least-squares in linear T, then scored as rms(ln T)
   with predictions clipped at 1e-300 -> rms 370 for GF/GD/heavy ball (= ln(1e-300) artifacts), quoted in
   notes 2.1 ("0.005 against 370 for a log law"). Not the same space/weighting (analysis standard 5).
   Conclusion (power law) unaffected, the number is meaningless.
4. plots.fig_implicit_bias docstring: "Only runs that reached the training tolerance are counted for the
   rank panel" - the code does not filter (RMSProp never reaches tol and is counted, title "12 runs per point").
5. GF event times come from interpolation (linear in log t) on a fixed 600-point log grid + 1201-point linear
   grid over [0, t_max]; with t_max = 2e6 the log spacing is 3.6%, and my exact event location (scipy event)
   differs by up to 1.3-1.6% on deep trees (MPS8 fan-in seed 1 sigma 0.3: 4574.3 exact vs 4503.8 committed;
   binary8 fan-in seed 1: 326.75 vs 331.15). The same run at different t_max gives different t_first
   (binary8 balanced seed 0 sigma 0.3: 698.55 in deep vs 696.59 in later_b8). Percent-level; matters only
   for the "0.97-1.05" precision of the eps-law test, whose prediction is linear in this T_GF.
6. README table: scaling_b4 "672 runs"; the YAML gives 798.
7. ROOT CAUSE of the defective adam_escape.png and of a crash: analysis.collect() re-reads and re-writes the cached
   <sweep>_summary.csv on every call, and pandas' default (non round-trip) float parser turns "1e-30" into
   9.999999999999999e-31 and "9.999999999999999e-31" back into 1e-30, so `eps` flips by 1 ulp on every collect().
   make_figures.eps() and plots.fig_adam_escape() both filter with exact equality `df["eps"] == 1e-30`.
   Committed CSV stores 9.999999999999999e-31 (96 rows). In the agent's run, eps() matched (per-seed CSV fine) but the
   second collect() inside fig_adam_escape() did not -> no eps->0 curves, n_sign = NaN -> dashed prediction NaN
   (invisible) in the committed adam_escape.png. On my subset the first collect() got the flipped value -> sign dict
   empty -> `KeyError: (0.001, 1)` in make_figures.eps (README step 3 fails for `eps`). Fix: read_csv(float_precision=
   'round_trip') in collect(), or np.isclose / log10 comparisons.
