# Reproduction review of `locality` round 1 -- part 1: setup and portability

- Code used: experiments/locality/{run.py, rules.py, analysis.py, make_figures.py, growth.py, bond_events.py,
  levels.py} copied to data/review_locality/code/ (repo untouched).
- run.py as committed does NOT run on this machine (Windows, no MSVC): `torch.compile` (inductor) raises
  InvalidCxxCompiler ("cl is not found"). Patch: grad/hess = torch.func.grad / torch.func.hessian (eager).
  Numerics identical to the committed tables to all printed digits (see part 2), so compile only buys speed.
- run.py hard-codes the trajectory directory OUT = /root/outputs/locality/traj (on Windows: C:/root/...),
  not an argument -> violates analysis standard 7. Patched to an env var LOCALITY_TRAJ.
- README "How to run" omits configs/twocomp_chi4.yaml (finalize.sh uses $O/twocomp_chi4); make_configs.py does
  not write twocomp_chi4.yaml or k5_extension.yaml although the README says configs are written by it.
- finalize.sh hard-codes O=/root/outputs/locality (input dir not configurable).
- Speed (eager, 1 thread): N = 4 runs 3-8 s, binary8 k = 5 sigma = 0.03 12-16 s, ht8 k = 7 saddle sigma = 0.15 18-27 s,
  TD mps8 ~13 s per run.
- Failure mode of the README command on a machine without a C++ compiler: torch.compile is lazy, so the
  InductorError is raised at the first call inside run_one and caught by run_task's catch-all -> every run is
  written as status "error" and the sweep "finishes" without crashing (retried on the next invocation). Not run
  as written here because run.py would also create C:/root/outputs/locality/traj outside the repo.
- Independent Hessian smoke test (timing run mps4 S=12 small sigma=1e-3 seed 0): reviewer's own contraction +
  own gauge basis gives lambda_min = -0.47609706496143783 vs runner -0.4760970649614375 (1e-15 rel.).
