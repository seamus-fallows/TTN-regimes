import numpy as np, pandas as pd
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/"
pd.set_option("display.width", 250)
d = pd.read_csv(R + "data/review_optimizers/indep/b8_balanced.csv")
c = pd.read_csv(R + "experiments/optimizers/results/later_b8_summary.csv", dtype={"order": str})
c["lr"] = c["lr"].fillna(0)
m = d.merge(c[c.optimizer.isin(["gf", "adam"])][["optimizer", "lr", "sigma", "seed", "t_first", "order", "t_first_resolved"]],
            left_on=["method", "lr", "sigma", "seed"], right_on=["optimizer", "lr", "sigma", "seed"], suffixes=("_mine", "_comm"))
m["rel"] = m.t_first_mine / m.t_first_comm - 1
print(m.groupby(["method", "lr"])["rel"].agg(["count", "min", "median", "max"]).round(4))
print("first comp same:", m.assign(s=[str(o)[1:2] == str(k) for o, k in zip(m.order, m.comp)]).groupby(["method", "lr"])["s"].mean().to_dict())
out = []
for (meth, lr), g in d.groupby(["method", "lr"]):
    for mode in ("all", "modal_first"):
        ps = []
        for seed, gs in g.groupby("seed"):
            if mode == "modal_first":
                gs = gs[gs.comp == gs.comp.mode()[0]]
            ps.append(np.polyfit(np.log(1 / gs.sigma), np.log(gs.t_first), 1)[0])
        ps = np.array(ps)
        out.append(dict(method=meth, lr=lr, mode=mode, p=ps.mean(), se=ps.std(ddof=1) / np.sqrt(len(ps)),
                        per_seed=" ".join(f"{v:.3f}" for v in ps)))
o = pd.DataFrame(out); print(o.round(3).to_string())
o.to_csv(R + "data/review_optimizers/results/indep_b8_balanced_exponents.csv", index=False)
m.to_csv(R + "data/review_optimizers/results/indep_b8_balanced_vs_committed.csv", index=False)
print(m[(m.method == "adam") & (m.lr == 0.01)][["sigma", "seed", "t_first_mine", "t_first_comm", "t_first_resolved"]].round(2).to_string())
