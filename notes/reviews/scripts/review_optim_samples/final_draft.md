## Reproduction review: optim_samples (Q8), round 1

Verdict: every claim I checked reproduces. The agent's code reruns to roundoff on this machine, and independent code confirms the key numbers. Several statements are worded more strongly than the data support, and I found four analysis or portability bugs. None of them changes a headline conclusion.

Outputs are in `c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_optim_samples/repro/`: the code copy and reduced configs are in `code/`, raw runs in `raw*/`, CSVs in `results*/` and logs in `logs/`. Notes are in `C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_optim_samples/notes_1..6.md`, and the independent scripts (`indep_binary4.py`, `indep_sgd_ms.py`, `indep_samples.py`) are in the same folder. The reruns took about 25 minutes of wall time with at most 6 processes.

### What I reran (agent code)
I ran a copy of the code from `data/` with reduced configs and `--raw-dir`/`--out-dir` set.
- **GD sweep:** `gd_sweep.py` and `analyze_gd.py` on binary N=4.
  - σ = 0.01–0.3, seeds 0–2, η/η_max ∈ {0.1, 0.9, 0.95, 1.05, 1.1, 1.2, 1.5, 2}.
  - New seeds 50–52 (σ = 0.03, 0.1) at {0.1, 0.95, 1.05}.
  - σ = 0.003, seeds 0–1, at 0.1.
  - Total: 116 GD runs and 20 GF runs.
- **Event shift:** `predict_shift.py` on 13 GF references.
- **Centred model:** σ ∈ {1, 1.4, 2}, 2 seeds, 7 step sizes each.
- **SGD:** `sgd_sweep.py` and `analyze_sgd.py` at σ = 0.1, inits 0–2, B ∈ {1, 4, 16, 64, 256}, η = 5e-4, with **new noise seeds**: 120 runs.
- **Finite samples:** `samples_sweep.py` and `analyze_samples.py` on **new data/init seeds 5–7**: χ=4 at n ∈ {128, 256, 384}, χ=16 at n ∈ {192, 256}, the lazy limits, and the centred model at σ=1, n=384.
- **P9/P10:** `p9_sgd.run` on 8 of the committed tasks and on 10 tasks with new noise seeds.
- **SGD stability:** the `sgd_stability` functions at B = 1 and 16.

### Independent checks (my own numpy/torch TTN code, none of `experiments/`)
- **A.** Binary4 with my own random teacher basis and init RNG: sharpness, a GD scan from 0.9 to 2.5 η_max, and excess loss against my own 1D map.
- **B.** My own GD and GF, started from the agent's init and teacher arrays: shift/η at 0.1, 0.05 and 0.025 η_max, extrapolated to η→0 (Richardson).
- **C.** My own SGD second-moment operator: Θ from an exact multilinear Jacobian, my own F, a symmetric basis.
- **D.** My own binary-8 TTN with its own teacher and data: lazy limit and small-init GD.

TABLE_HERE

DISC_HERE

BUGS_HERE
