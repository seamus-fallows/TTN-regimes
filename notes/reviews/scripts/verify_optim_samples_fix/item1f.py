import pandas as pd, numpy as np
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
gs = pd.read_csv(R+"gd_sweep.csv")
of = pd.read_csv(R+"order_flips.csv")
c9 = gs[(gs.chi==9)&(~gs.centred)]
is9 = np.zeros(len(of), bool)
for i, r in of.iterrows():
    m = c9[(c9.sigma==r.sigma)&(c9.seed==r.seed)&np.isclose(c9.ratio, r.ratio)]
    if len(m) and np.isclose(m.t1_gf.values[0], r.t1_gf, rtol=1e-9): is9[i] = True
of3 = of[~is9]
print("chi3 rows:", len(of3), "flips:", int(of3.flip.sum()), "chi9 rows:", is9.sum())
summ = of3.groupby(['sigma','ratio']).agg(n=('flip','size'), k=('transient','sum'))
summ['p'] = summ.k/summ.n; summ['se'] = np.sqrt(summ.p*(1-summ.p)/summ.n)
print(summ.round(3).to_string())
