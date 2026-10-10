"""Resolution check of the base saddle-start runs (log rows only) against reruns with the final
library's fine rows (configs/r2_saddle_fine.yaml): same teacher, tree, eps and seed; escape times by
the same onset rule (analyze_r2_saddle.load_runs). Writes results/r2_saddle_resolution_check.csv."""
import numpy as np
import pandas as pd

from analyze_r2_saddle import load_runs
from common import OUT_ROOT, RESULTS

base = load_runs()
fine_dir = OUT_ROOT / "r2_saddle_fine"
import analyze_r2_saddle as a
orig = a.OUT_ROOT
a.OUT_ROOT = OUT_ROOT  # load_runs reads OUT_ROOT / "r2_saddle"; point it at the fine runs
import json
rows = []
for f in sorted(fine_dir.glob("*.npz")):
    d = np.load(f)
    r = json.loads(str(d["summary"]))
    rows.append(r)
fine = pd.DataFrame(rows)
# recompute fine escape times with the same onset rule
tmp = []
for f in sorted(fine_dir.glob("*.npz")):
    d = np.load(f)
    r = json.loads(str(d["summary"]))
    s2t = a.teacher_s2(r["teacher"], r["n"], r["tree"])
    t, sp = d["t"], d["spectra"][:, :, 1]
    hits = []
    for j in np.nonzero(s2t > 1e-2)[0]:
        h = np.nonzero(sp[:, j] >= max(0.05 * s2t[j], 10 * sp[0, j]))[0]
        if len(h):
            hits.append(t[h[0]])
    r["t_fine"] = min(hits) if hits else np.nan
    tmp.append(r)
fine = pd.DataFrame(tmp)
m = fine.merge(base[["teacher", "tree", "sigma", "seed", "t_esc_0.05"]], on=["teacher", "tree", "sigma", "seed"])
m["rel_diff"] = (m["t_esc_0.05"] - m["t_fine"]) / m["t_fine"]
m.to_csv(RESULTS / "r2_saddle_resolution_check.csv", index=False)
print(m.groupby("tree").rel_diff.agg(lambda x: np.nanmax(np.abs(x))).to_string())
print("max |rel diff| overall:", np.nanmax(np.abs(m.rel_diff)), "n =", len(m))
