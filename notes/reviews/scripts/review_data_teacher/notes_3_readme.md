# Part 3: do the README run instructions work as written?

Tested (in a verbatim copy, with DATA_TEACHER_OUT set): `uv run python .../e1_exact_aligned.py`,
`run_queue.sh e1_order` (as `run_queue_review6.sh`, only `--procs 28 -> 6` changed),
`run_dense_queue.sh e4a_sigma e4b_ttn` (same change), `analyze_e1_order.py`,
`analyze_sigma_law.py e4a_sigma overlap`, `analyze_ttn.py e4b_ttn`, `theory_log_slope.py`,
`predict_rank1.py e2_misaligned` + `analyze_plateaus.py e2_misaligned e2_misaligned`. All ran without
error on Windows/Git Bash (multiprocessing spawn is fine: all job functions are top-level and every
script has a __main__ guard). Results reproduce. Practical problems:

1. Every runner and analysis script writes into `experiments/data_teacher/results/` (fixed
   `HERE / "results"`), so following the README to reproduce *overwrites the committed CSVs/figures*;
   a reduced or test run (e.g. a smaller sweep) silently replaces the committed summary with a partial
   one. No `--out`/results-dir option. (Reviewer workaround: run a copy of the folder.)
2. Default raw-output root `/root/outputs/data_teacher` is the remote machine's path; on Windows it
   resolves to `C:\root\outputs\...`. The override `DATA_TEACHER_OUT` is documented in the README;
   CLAUDE.md says raw outputs belong in `data/<experiment>/` locally, which is not the default.
3. `run_queue.sh` / `run_dense_queue.sh` hard-code `--procs 28` (no override): on this 24-core machine
   (and under the project's ~6-process etiquette for shared machines) they must be edited.
4. Analysis scripts need inputs in a specific order not stated in the README:
   `analyze_e1_order.py` needs `summary_rows.pkl` written by `run_aligned.py` and all four variants
   (KeyError otherwise); `analyze_sigma_law.py e4a_sigma` needs `results/e4a_theory_slopes.csv`
   (from `theory_log_slope.py`); `analyze_plateaus.py` needs `predict_rank1.py`'s `.pt` file in OUT_ROOT.
   (They work with the committed files present.)
5. README says "total about 1.5 h on 28 cores"; here the reduced chain (e1 exact + 160 pilot runs +
   138 dense runs) took ~17 min on 6 processes, consistent.
