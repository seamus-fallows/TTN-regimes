"""P11 tally over all step-4(c) sweeps (fix round 2): binary-8 student, MPS-8 teachers, f = 0.05 events
(analyze_cross_v2.py). Distinct runs only: a run is (teacher seed, sigma, init seed), so teacher 203 at
sigma = 0.4, run in both e4c_cross and r2_cross, counts once. P11 holds in a run if the root bipartition
gets its k = 2 event only after a bottom bond on each half (bottom_up). Counts by class:
- all: every teacher;
- discriminating: root s_2 above both bottom bonds of at least one half (a weight-order rule disagrees);
- root above all: root s_2 above every bottom bond (a weight-order rule predicts the root bond first).
Usage: uv run python experiments/data_teacher/count_p11.py
Writes results/p11_counts.csv (per teacher and sigma) and results/p11_totals.csv.
"""

import pandas as pd

from common import RESULTS

frames = []
for name in ("e4c_cross", "r2_cross", "r2_cross_b"):
    p = RESULTS / f"{name}_v2_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        frames.append(d[(d.pair == "mps_on_binary") & (d.f == 0.05)].assign(sweep=name))
df = pd.concat(frames)
df = df.drop_duplicates(["teacher_seed", "sigma", "seed"])
df = df[df.bottom_up.notna()]
df["bottom_up"] = df.bottom_up.astype(bool)
per = df.groupby(["sweep", "teacher_seed", "sigma"]).agg(
    n=("seed", "size"), p11=("bottom_up", "sum"), discriminating=("p11_discriminating", "first"),
    root_above_all=("root_above_all_bottoms", "first"), root_margin=("root_margin", "first")).reset_index()
per.to_csv(RESULTS / "p11_counts.csv", index=False)
rows = []
for label, sel in (("all", df), ("discriminating", df[df.p11_discriminating.astype(bool)]),
                   ("root above all bottoms", df[df.root_above_all_bottoms.astype(bool)])):
    rows.append(dict(cls=label, n_teachers=sel.teacher_seed.nunique(), n_runs=len(sel),
                     n_p11=int(sel.bottom_up.sum())))
tot = pd.DataFrame(rows)
tot.to_csv(RESULTS / "p11_totals.csv", index=False)
print(per.to_string())
print(tot.to_string())
