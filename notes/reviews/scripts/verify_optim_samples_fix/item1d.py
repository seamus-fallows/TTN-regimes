import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
gs = pd.read_csv(R+"gd_sweep.csv")
c9 = gs[(gs.chi==9)&(gs.centred==False)]
print("chi9 plain ratios:", sorted(c9.ratio.unique()), "sigmas", sorted(c9.sigma.unique()))
m = c9[np.isclose(c9.ratio,0.5)][['sigma','seed','ratio','t1_gf','t2_gf','t1_gd','t2_gd','same_order','cls']]
print(m.to_string())
# Now main-sweep N=4 chi=3 no root scale, ratio < 1
b4 = gs[(gs.tree=='binary')&(gs.n==4)&(gs.chi==3)&(gs.centred==False)&(gs.root_scale.isna())]
below = b4[b4.ratio<1]
print("binary4 chi3 plain below eta_max:", len(below), "same_order:", below.same_order.value_counts(dropna=False).to_dict(), "cls:", below.cls.value_counts().to_dict())
print("seeds 0-4 below:", len(below[below.seed<=4]), below[below.seed<=4].cls.value_counts().to_dict())
# N=8 trees
for (tree,n), g in gs[(gs.sigma==0.9)].groupby(['tree','n']):
    gb = g[g.ratio<1]
    print(tree, n, "below eta_max:", len(gb), "same_order", gb.same_order.value_counts(dropna=False).to_dict(), "cls", gb.cls.value_counts().to_dict())
    print(gb.groupby('ratio').agg(n=('seed','size'), so=('same_order','sum'), ne_gd=('n_events_gd','mean'), ne_gf=('n_events_gf','mean')).T.to_string())
