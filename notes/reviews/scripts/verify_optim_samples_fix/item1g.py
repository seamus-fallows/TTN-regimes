import pandas as pd, numpy as np
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
gs = pd.read_csv(R+"gd_sweep.csv")
sel = gs[(gs.chi==3)&(~gs.centred)&(gs.root_scale.isna())&(gs.ratio<1)]
print(len(sel))
print(sel.groupby(['tree','n']).apply(lambda g: pd.Series(dict(n=len(g), ev_equal=(g.n_events_gd==g.n_events_gf).sum(), ev_dist=g.n_events_gf.value_counts().to_dict()))).to_string())
neq = sel[sel.n_events_gd!=sel.n_events_gf]
print(neq[['tree','n','sigma','seed','ratio','n_events_gd','n_events_gf','cls']].to_string())
