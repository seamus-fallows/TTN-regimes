import pandas as pd, numpy as np
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
gs = pd.read_csv(R+"gd_sweep.csv")
b4 = gs[(gs.tree=='binary')&(gs.n==4)&(gs.chi==3)&(~gs.centred)&(gs.root_scale.isna())]
s04 = b4[b4.seed<=4]
below = s04[s04.ratio<1]
lr = below.loss_end/below.loss0
print("seeds0-4 below:", len(below), "gf_like:", (below.cls=='gf_like').sum(), " ending at L/L0~0.071:", (np.abs(lr-0.0714)<0.002).sum())
r105 = s04[np.isclose(s04.ratio,1.05)]
print("1.05:", len(r105), r105.cls.value_counts().to_dict())
for r_ in [1.5, 2.0]:
    q = s04[np.isclose(s04.ratio, r_)]; print(r_, len(q), q.cls.value_counts().to_dict())
q = s04[np.isclose(s04.ratio,0.9)].groupby('sigma').rel_err1.mean()
print("rel_err1 at 0.9 by sigma:", q.round(4).to_dict())
# N=8 sharper than balanced
for (t,n), g in gs[gs.sigma==0.9].groupby(['tree','n']):
    print(t, n, "sharp_gf_rel range:", g.groupby('seed').sharp_gf_rel.first().round(3).tolist())
