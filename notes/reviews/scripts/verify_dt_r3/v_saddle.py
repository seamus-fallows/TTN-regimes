"""Independent refit of saddle-start escape laws from r2_saddle_runs_all.csv (value-based eps windows)."""
import json
import numpy as np, pandas as pd
R = "experiments/data_teacher/results/"
d = pd.read_csv(R + "r2_saddle_runs_all.csv")
hes = pd.concat([pd.read_csv(R + f) for f in ("r2_saddle_hessian.csv", "r2_saddle_b_hessian.csv")]).drop_duplicates(["teacher", "tree"])
gam = {(r.teacher, r.tree): np.array(json.loads(r.edge_gammas)) for r in hes.itertuples()}
lmin = {(r.teacher, r.tree): r.lambda_min for r in hes.itertuples()}
fits = pd.read_csv(R + "r2_saddle_fits.csv")
pd.set_option("display.width", 250)
rows = []
for (t, tr), g in d.groupby(["teacher", "tree"]):
    for f in ("0.03", "0.05", "0.1", "int"):
        tc, ec = (f"t_esc_{f}", f"edge_{f}") if f != "int" else ("t_int", "edge_int")
        if (f == "int") != (t == "misaligned8_20"):
            continue
        for seed, gs in g.groupby("seed"):
            gs = gs[np.isfinite(gs[tc]) & (gs[tc] > 0)].sort_values("sigma")
            if tr == "mps":
                win = gs[gs.sigma <= (1e-4 if t != "misaligned8_20" else 2e-3)]
            else:
                win = gs[gs.sigma <= (1e-3 if t != "misaligned8_20" else 2e-2)]
            if len(win) < 3:
                continue
            x, tt = np.log(1 / win.sigma.values), win[tc].values
            A = np.polyfit(x, tt, 1)[0]
            p = np.polyfit(x, np.log(tt), 1)[0]
            edges = win[ec].astype(int).values
            single = len(set(edges)) == 1
            pe = 1 / gam[(t, tr)][edges[0]] if tr == "mps" and single else np.nan
            # per-run realized-edge prediction averaged over the window (for mixed seeds)
            pe_mix = np.mean([1 / gam[(t, tr)][e] for e in edges]) if tr == "mps" else np.nan
            rows.append(dict(teacher=t, tree=tr, f=f, seed=seed, n=len(win), eps=list(win.sigma.values), A=A, p=p,
                             edges=list(edges), single=single, pred_edge=pe, pred_mix=pe_mix,
                             pred_fast=1 / abs(lmin[(t, tr)]) if tr == "mps" else np.nan))
o = pd.DataFrame(rows)
o.to_pickle("C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/verify_dt_r3/saddle_local.pkl")
print("MPS (f=0.05 or int): per teacher")
for t, g in o[(o.tree == "mps") & (o.f.isin(["0.05", "int"]))].groupby("teacher"):
    dev = g.A / g.pred_mix - 1
    print(f"{t:16s} n_seeds={len(g)} n_pts={sorted(set(g.n))} A={g.A.mean():.4f}+-{g.A.sem():.4f} "
          f"pred_mix_mean={g.pred_mix.mean():.4f} dev(mean A/mean pred)={g.A.mean()/g.pred_mix.mean()-1:+.4%} "
          f"per-seed dev range [{dev.min():+.3%},{dev.max():+.3%}] single={int(g.single.sum())} fast={g.pred_fast.iloc[0]:.3f} edges={[e for e in g.edges]}")
print()
print("binary/HT p (f=0.05 or int)")
for (t, tr), g in o[(o.tree != "mps") & (o.f.isin(["0.05", "int"]))].groupby(["teacher", "tree"]):
    print(f"{t:16s} {tr:6s} n_seeds={len(g)} npts={sorted(set(g.n))} p={g.p.mean():.4f}+-{g.p.sem():.4f} eps_used={sorted(set(sum(g.eps, [])))}")
print()
print("f-sensitivity of asymptotic MPS slopes (mean A per f, rel. to f=0.05)")
m = o[o.tree == "mps"].groupby(["teacher", "f"]).A.mean().unstack()
if "0.05" in m:
    for c in ("0.03", "0.1"):
        m[c + "_rel"] = m[c] / m["0.05"] - 1
print(m.round(4).to_string())
