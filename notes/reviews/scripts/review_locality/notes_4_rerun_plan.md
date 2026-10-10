# Part 4: what was rerun (data/review_locality/)

Config data/review_locality/configs/repro_main.yaml = rows copied verbatim from configs/twocomp.yaml,
twocomp_balanced.yaml, multicomp_wide.yaml (same sigmas, seeds, protocol), 558 runs, 6 workers, ~10 min wall:
- k = 1: binary4 S=01 small (56)
- k = 2: mps4 S=12 small (56), mps4 S=12 saddle (56), mps8 S=34 saddle (56)
- k = 3: binary4 S=12 small (56)
- k = 4: mps4 S=0123 small (48), mps4 S=03 small (48), mps4 S=03 small_balanced (48)
- k = 5: binary8 S=34 small (48)
- k = 7: ht8 S=34 saddle (30)
- TD on mps8, saddle start (56) (multicomp_wide.yaml)
Runner: patched copy (eager instead of torch.compile; trajectory dir via env var). Outputs: out/main.part*,
traj/*.npz. Analyses: agent's make_figures.py / growth.py / levels.py on the rerun (figs/), reviewer's my_fits.py,
my_bonds.py, my_levels.py, indep.py (own model + own integrator + own Hessian), stall_check.py, steiner_check.py.
