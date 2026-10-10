import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
gs = pd.read_csv(R+"gd_sweep.csv")
df = gs[(~gs.centred)&gs.root_scale.isna()&(gs.chi==3)&(gs.ratio<1)&(gs.ratio_ref=='teacher')].copy()
rows=[]
for _, r in df.iterrows():
    tg = np.array([r.t1_gf, r.t2_gf, r.t3_gf], float)
    order = [i for i in np.argsort(np.where(np.isnan(tg), np.inf, tg)) if np.isfinite(tg[i])]
    for rank, comp in enumerate(order):
        m, p = r[f"shift_over_eta{comp+1}"], r[f"pred_shift_over_eta{comp+1}"]
        if pd.isna(m) or pd.isna(p): continue
        rows.append(dict(tree=f"{r.tree}{r.n}", sigma=r.sigma, seed=r.seed, ratio=r.ratio, comp=comp+1, rank=rank+1, measured=m, predicted=p, rel=abs(m/p-1)))
x = pd.DataFrame(rows)
low = x[x.ratio<=0.1+1e-12]
print("n comparisons <=0.1:", len(low))
for key in ['comp','rank']:
    g1 = low[low[key]==1]
    print(f"{key}==1: n={len(g1)}, within 1%: {(g1.rel<=0.01).mean():.3f}, max {g1.rel.max():.4f}")
mis = low[(low.comp==1)&(low['rank']!=1)]
print("component-1 events that are not first in time (<=0.1):", len(mis), mis.groupby(['tree','sigma']).size().to_dict())
fr = low[(low['rank']==1)&(low.comp!=1)]
print("first-in-time events of other components (<=0.1):", len(fr))
print(fr.sort_values('rel', ascending=False).head(10).to_string())
# 0.9 eta_max first events (by component and by rank)
h = x[np.isclose(x.ratio,0.9)]
for key in ['comp','rank']:
    g1 = h[h[key]==1]
    print(f"0.9 eta_max {key}==1: n={len(g1)}, median {g1.rel.median():.4f}, quantiles {g1.rel.quantile([0.1,0.5,0.9]).round(4).to_dict()}, max {g1.rel.max():.4f}")
    print(g1.groupby('tree').rel.agg(['median','max','size']).round(4).to_string())
# median per ratio for comp 1
for r_ in [0.01,0.03,0.1]:
    g1 = x[np.isclose(x.ratio,r_)&(x.comp==1)]
    print("ratio",r_,"comp1 median", g1.rel.median())
# >5% count
print(">5%:", (low.rel>0.05).sum(), "  >=5%:", (low.rel>=0.05).sum(), sorted(low.rel[low.rel>0.04].round(4).tolist()))
