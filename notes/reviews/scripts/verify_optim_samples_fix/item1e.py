import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
gs = pd.read_csv(R+"gd_sweep.csv")
# what analyze_order's filter would select now
f = gs[(gs.tree == "binary") & (gs.n == 4) & ~gs.centred & gs.root_scale.isna() & (gs.ratio < 1)]
print("analyze_order filter on committed gd_sweep:", len(f), "of which chi9:", (f.chi==9).sum(), sorted(f[f.chi==9].ratio.unique()))
of = pd.read_csv(R+"order_flips.csv")
# drop the chi9 rows by matching to gd_sweep chi=3
b4 = gs[(gs.tree=='binary')&(gs.n==4)&(gs.chi==3)&(~gs.centred)&(gs.root_scale.isna())&(gs.ratio<1)]
key = ['sigma','seed','ratio','t1_gf']
of3 = of.merge(b4[key], on=key, how='inner')
print("order rows chi3:", len(of3), "flips:", of3.flip.sum())
summ = of3.groupby(['sigma','ratio']).agg(n=('flip','size'), p_tr=('transient','mean'))
summ['se'] = np.sqrt(summ.p_tr*(1-summ.p_tr)/summ.n)
print(summ[summ.n>=45].round(3).to_string())
r = of3[(of3.sigma==0.1)&(of3.seed==1)&np.isclose(of3.ratio,0.9)]
print(r.to_string())
# GF transient: n_cross is GD. For GF transient counts, check at smallest ratio
print("transient at ratio 0.1 per sigma (n):", of3[np.isclose(of3.ratio,0.1)].groupby('sigma').transient.agg(['sum','size']).to_dict())
print("transient at ratio 0.01 per sigma:", of3[np.isclose(of3.ratio,0.01)].groupby('sigma').transient.agg(['sum','size']).to_dict())
# rival_peak >= 0.5 indicates GF transient crossing of eventual second component during first event
gfc = of3.drop_duplicates(['sigma','seed']).groupby('sigma').apply(lambda g: f"{(g.rival_peak>=0.5).sum()}/{len(g)}")
print("GF rival_peak >= 0.5 per sigma:", gfc.to_dict())
