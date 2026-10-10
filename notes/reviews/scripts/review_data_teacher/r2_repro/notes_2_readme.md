# part 2: README run instructions (experiments/data_teacher/README.md, "How to run")

Works as written on Windows / Git Bash for the commands exercised (r2_saddle_runs.py <config> --procs N,
r2_star.py modes via their functions, r2_corr.py job(), run_dense.py) once DATA_TEACHER_OUT and
DATA_TEACHER_RESULTS are set: every script exercised wrote only to those folders (check: no results/
folder appeared in the code copy). Rerun numbers equal committed ones (see part 3).

Problems found:
1. Spawn platforms (Windows, macOS default): module globals set in `__main__` before creating a Pool are
   NOT seen by spawned workers. r2_saddle_runs.py sets T_CAP = cfg["t_cap"] via globals() "inherited by
   the forked workers" -> on Windows the r2_saddle_ext.yaml cap 1e11 silently becomes 1e7, so the slow
   binary/HT eps <= 1e-4 runs (committed t_esc up to 6e8 for ht8) would stop unescaped. r2_star.py
   threshold / complete_ht set STOP_ROTATE the same way (affects run length only, not the a_q < 0.9
   classification). README does not say Linux-only.
2. README run order lists `r2_star.py hessian; saddle; random` but not `r2_star.py threshold` (writes
   r2_star_threshold_scan.csv, the "threshold to 3e-4" claim) nor `complete_ht` (the 5 stopped ht4 runs).
   r2_chain4.sh is the only place the threshold mode is invoked.
3. r2_star.py hessian mode calls torch.set_num_threads(8) (CLAUDE.md: 1 thread per process).
4. run_dense_queue.sh logs to agent-runs/data-teacher/logs unless DATA_TEACHER_LOGS is set (undocumented);
   r2_chain*.sh hard-code --procs 28, the logs path, and r2_chain.sh polls
   experiments/data_teacher/results/r2_random_dense_summary.csv (ignores DATA_TEACHER_RESULTS).
5. recompute_saddle_hessians.py hard-codes Pool(4) (no --procs) and only covers r2_saddle.yaml
   (r2_saddle_b_hessian.csv comes from running r2_saddle_runs.py on r2_saddle_b.yaml).
6. analyze_r2_saddle.py needs r2_saddle_hessian.csv and r2_saddle_b_hessian.csv in DATA_TEACHER_RESULTS
   (written by step 68-71 of the README): fine if run in order.
