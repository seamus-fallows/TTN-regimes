"""Independent re-fit of the random-init asymptotic slopes from the committed *_v2_runs.csv files."""
import sys, json
import numpy as np, pandas as pd
R = "experiments/data_teacher/results/"
name = sys.argv[1]
f = sys.argv[2] if len(sys.argv) > 2 else "0.05"
d = pd.read_csv(R + f"{name}_v2_runs.csv")
grid = {tr: np.sort(g.sigma.unique()) for tr, g in d.groupby("tree")}
print("grids:", {k: list(v) for k, v in grid.items()})
d = d[d.resolved == True]
gcol, ecol = f"gap_{f}", f"edge_{f}"
out = []
for (c, tr, sd), g in d.groupby(["case", "tree", "saddle"]):
    for seed, gs in g.groupby("seed"):
        gs = gs[np.isfinite(gs[gcol]) & (gs[gcol] > 0)].sort_values("sigma")
        if len(gs) < 3:
            continue
        sig = gs.sigma.values
        ranks = [int(np.searchsorted(grid[tr], s)) for s in sig]
        row = dict(case=c, tree=tr, saddle=sd, seed=seed, nres=len(gs), ranks=ranks[:3])
        for w, (i0, i1) in {"w012": (0, 3), "w123": (1, 4), "w234": (2, 5), "w345": (3, 6)}.items():
            gg = gs.iloc[i0:i1]
            if len(gg) < 3:
                continue
            x, t = np.log(1 / gg.sigma.values), gg[gcol].values
            row["A_" + w] = np.polyfit(x, t, 1)[0]
            row["p_" + w] = np.polyfit(x, np.log(t), 1)[0]
        e3 = gs.iloc[:3][ecol].values
        row["edges3"] = list(e3.astype(int))
        ape = json.loads(gs.iloc[0]["A_pred_edges"]) if isinstance(gs.iloc[0]["A_pred_edges"], str) else [np.nan]
        row["A_pred_e"] = ape[int(e3[0])] if len(set(e3)) == 1 and len(ape) > int(e3[0]) else np.nan
        row["A_pred"] = gs.iloc[0]["A_pred"]
        out.append(row)
o = pd.DataFrame(out)
pd.set_option("display.width", 250)
for (c, tr, sd), g in o.groupby(["case", "tree", "saddle"]):
    col = "A" if tr == "mps" else "p"
    s = f"{c:16s} {tr:6s} sd={sd:.0f} nseeds={len(g)} ranks={[r for r in g.ranks]} edges={[e for e in g.edges3]}"
    print(s)
    for w in ("w012", "w123", "w234", "w345"):
        if col + "_" + w in g and g[col + "_" + w].notna().sum() >= 1:
            v = g[col + "_" + w].dropna()
            print(f"    {col}_{w}: {v.mean():.4f} +- {v.sem() if len(v)>1 else np.nan:.4f} (n={len(v)}) per-seed: {np.round(v.values,4)}")
    if tr == "mps":
        print(f"    pred(first edge) {g.A_pred.iloc[0]:.4f}; pred by realized edge (seeds): {np.round(g.A_pred_e.values,4)}")
