# Instructions for every research agent

You are one of several research agents on a project about the **training dynamics of tree tensor
networks (TTNs)**. The lead researcher assigned you one question (your brief, in this folder). The
lead reviews your work, may send you messages with redirections, and merges your branch.

## Status of shared resources when you were launched (2026-10-09)

- **Finished and on `main`**: the core library `ttn/tree.py`, `ttn/network.py`, `ttn/teachers.py`,
  `ttn/losses.py`, `ttn/general.py` (tested in `tests/test_core.py`), `notes/model.md`,
  `notes/diagnostics.md`, `notes/prior_knowledge.md`, `PLAN.md`, these briefs.
- **Being finished right now** (ETA a few hours; you will get a message when they are on `main`,
  then run `git fetch origin && git merge origin/main`): `ttn/dynamics.py` (adaptive gradient-flow
  integrator, GD, SGD, trajectory recorder), `ttn/diagnostics.py` (events, plateaus, tangent
  kernels, Hessians with gauge projection), `ttn/theory.py`, `ttn/plotting.py`, `ttn/config.py` +
  `ttn/runner.py` (YAML configs, parallel resumable sweeps), `experiments/00_smoke/`,
  `notes/literature.md`, `notes/theory_*.md`.
- **Until they land**: read, derive the theory for your question, design your experiments
  (parameter ranges, configs, cost estimates), and run small pilots with the core modules plus
  `scipy.integrate.solve_ivp` on the flat parameter vector (`spec.flatten`/`spec.unflatten`, torch
  autograd of `ttn.losses.population_loss`, float64, rtol ~1e-9, atol ~1e-14). Keep pilot code
  minimal and throwaway: do not build your own general integrator, diagnostics or sweep framework.

## Before you start

Read, in this order: `CLAUDE.md` (project map, library table), `notes/model.md` (the model),
`notes/diagnostics.md` (how regimes and events are measured: apply these definitions exactly),
`PLAN.md` (all research questions; yours is named in your brief), `notes/literature.md` (what is
already known), the theory note(s) named in your brief, and `experiments/00_smoke/` (a complete
example of an experiment folder using the library).

## How to work

- **Use the shared library `ttn/`** (model, teachers, exact population loss, gradient-flow
  integrator, GD/SGD, diagnostics, theory helpers, plotting, config-driven runner and sweeps). Do
  not reimplement it. If you need a library change, make it minimal, add a test, and describe it in
  your report; put experiment-specific code in your experiment folder.
- **Your folder** `experiments/<your-name>/`: `README.md` (question, setup, how to run, how to run
  variants, summary of results), `configs/` (YAML), code (scripts specific to the experiment),
  `results/` (small summary data such as CSV, and figures). Write findings to `notes/<your-name>.md`.
- **Raw outputs** (per-run trajectories, parameter snapshots) go to `/root/outputs/<your-name>/`,
  never into git. Say in your README how to regenerate them.
- **Reproducibility**: every figure is produced by a script in your folder from saved results;
  configs and seeds are committed; numbers state the dynamics used (gradient flow + tolerances, or
  GD/SGD + step size and batch).
- **Statistics**: at least 5 seeds per setting, mean +- standard error; scaling claims need at least
  5 values of the swept parameter over at least one decade and a fitted exponent with its error.
- **Theory with experiment**: compare every measurement with the predictions in your theory note(s),
  quantitatively. If a prediction fails, say so plainly and propose why. You may derive things
  yourself; write derivations into your notes file.
- **Status tags** on every claim: proven / derived (state assumptions) / empirical / conjecture.
- **Compute**: use all CPU cores (the runner's sweeps use multiprocessing with one thread per
  process). Run long jobs in the background with `nohup ... > log 2>&1 &` and check on them: the
  machine stops itself when nothing is running and nobody is talking to you. Integrating long
  plateaus is expensive (escape times grow like sigma^{-(K-2)}): choose parameter ranges so that
  single runs stay under ~10 minutes, and estimate sweep cost before launching it.
- **Reporting**: keep `agent-runs/<your-name>/REPORT.md` current (findings so far with figures,
  open problems, next steps, decisions you made and why). Commit and push to your branch at every
  milestone and at least every ~2 hours of work, so the lead can review. Small, frequent pushes are
  better than one big push at the end.
- **Stay in scope**: other agents work on the other questions in `PLAN.md`; do not duplicate them.
  If you find something important outside your scope, note it in your report.
- **Be skeptical of your own results**: check integrator accuracy (conservation-law drift, a
  tighter-tolerance rerun), check seeds, look for bugs before interpreting surprises, and
  reproduce surprising results before reporting them.

## Analysis standards (added after the first independent review)

The first review of an agent's results (`notes/reviews/data_teacher_round1.md`) found the runs correct
but several headline numbers unsupported by the analysis. Avoid these pitfalls:

1. **Resolve every duration you report.** Use dense output around events (the library integrator with
   a dense grid, or explicit extra points); a duration equal to one output interval is not a
   measurement. Report unresolved cases as unresolved, never silently drop them.
2. **Event rules**: use the definitions in `notes/diagnostics.md` and, for generic teachers,
   thresholds relative to the target (f x teacher sigma_2, f ~ 0.03-0.1, `notes/theory_teachers.md` §3.3),
   not fixed absolute thresholds like s_2 > 1e-2. Check the event rule on the baseline case.
3. **Classify runs before pooling**: runs that visit different saddles (or no resolved plateau) must
   be separated, not averaged together.
4. **Per-seed statistics**: when sigma rescales the same init direction per seed, fit each seed
   separately and take mean +- SE over seeds; errors from residuals of per-sigma means are too small.
5. **Compare competing laws fairly**: fit log and power laws in the same space with the same weighting,
   report both scores, and span enough range (>= 1.5-2 decades in sigma, or use a saddle-start protocol
   with a perturbation eps over several decades) to discriminate them.
6. **Claims no broader than the evidence**: state N, tree, sigma range and seed counts with every claim;
   "strict saddle" needs a Hessian; "matches theory" needs the comparison number.
7. **Reproducibility**: scripts must take an output directory (do not overwrite committed results when
   re-run) and a process count; document the order in which scripts must run.
