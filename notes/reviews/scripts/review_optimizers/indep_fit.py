"""Per-seed power-law fits of the independent first-escape times, and per-run comparison with the committed
summaries (same library fan-in init).  uv run python indep_fit.py"""
import numpy as np, pandas as pd
from pathlib import Path
R = Path("c:/Users/seamu/Iliad-ML/TTN_regimes")
pd.set_option("display.width", 250)

def fits(df):
    out = []
    for (tree, m, lr), g in df.groupby(["tree", "method", "lr"], dropna=False):
        ps = []
        for seed, gs in g.groupby("seed"):
            if gs["comp"].nunique() > 1:  # different saddle (first component) across sigma: keep modal
                gs = gs[gs["comp"] == gs["comp"].mode()[0]]
            x, y = np.log(1 / gs.sigma.values), np.log(gs.t_first.values)
            ps.append(np.polyfit(x, y, 1)[0])
        ps = np.array(ps)
        out.append(dict(tree=tree, method=m, lr=lr, p=ps.mean(), se=ps.std(ddof=1) / np.sqrt(len(ps)),
                        per_seed=" ".join(f"{v:.3f}" for v in ps), sigma_range=f"{g.sigma.min()}-{g.sigma.max()}"))
    return pd.DataFrame(out)

b4 = pd.read_csv(R / "data/review_optimizers/indep/b4.csv"); b4["lr"] = b4["lr"].fillna(0)
b8 = pd.read_csv(R / "data/review_optimizers/indep/b8.csv"); b8["lr"] = b8["lr"].fillna(0)
f = pd.concat([fits(b4), fits(b8)])
print(f.round(4).to_string()); f.to_csv(R / "data/review_optimizers/results/indep_exponents.csv", index=False)
a = b4[(b4.method == "adam")].copy(); a["lr_n1"] = a["lr"] * a["t_first"]
print("sign-flow time lr*n_1 (median over sigma, seeds):", a.groupby("lr")["lr_n1"].median().round(4).to_dict())

# per-run vs committed (binary4 fan-in = scaling_b4; binary8 fan-in = deep sigma 0.3/0.8)
c4 = pd.read_csv(R / "experiments/optimizers/results/scaling_b4_summary.csv", dtype={"order": str})
c4["lr"] = c4["lr"].fillna(0)
c4 = c4[(c4.optimizer.isin(["gf", "adam", "rmsprop"])) & (c4.get("beta2").isna())]
m4 = b4.merge(c4[["optimizer", "lr", "sigma", "seed", "t_first", "order"]], left_on=["method", "lr", "sigma", "seed"],
              right_on=["optimizer", "lr", "sigma", "seed"], suffixes=("_mine", "_comm"))
m4["rel"] = m4.t_first_mine / m4.t_first_comm - 1
m4["first_same"] = [str(o)[1:2] == str(c) for o, c in zip(m4.order, m4.comp)]
print(m4.groupby(["method", "lr"])[["rel"]].agg(["count", "min", "max", lambda x: np.abs(x).median()]).round(4))
print("first component same as committed:", m4.groupby(["method", "lr"])["first_same"].mean().to_dict())
cd = pd.read_csv(R / "experiments/optimizers/results/deep_summary.csv", dtype={"order": str}, low_memory=False)
cd = cd[(cd.tree == "binary") & (cd.teacher == "odeco") & (cd.init == "fanin") & cd.optimizer.isin(["gf", "adam"])]
cd["lr"] = cd["lr"].fillna(0)
m8 = b8.merge(cd[["optimizer", "lr", "sigma", "seed", "t_first", "order"]], left_on=["method", "lr", "sigma", "seed"],
              right_on=["optimizer", "lr", "sigma", "seed"], suffixes=("_mine", "_comm"))
m8["rel"] = m8.t_first_mine / m8.t_first_comm - 1
print(m8[["method", "lr", "sigma", "seed", "t_first_mine", "t_first_comm", "rel", "comp", "order"]].round(4).to_string())
m4.to_csv(R / "data/review_optimizers/results/indep_vs_committed_b4.csv", index=False)
m8.to_csv(R / "data/review_optimizers/results/indep_vs_committed_b8.csv", index=False)
