import pandas as pd, numpy as np
pd.set_option('display.width', 250)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
sr = pd.read_csv(R+"sgd_runs.csv")
sr['Tr'] = (sr.t2-sr.t1)/(sr.t2_gf-sr.t1_gf)
ok = sr[(sr.run_class=='same_order')&(~sr.forgot)&(~sr.diverged)&(sr.init=='fanin')]
g = ok.groupby(['block','sigma','seed','B','eta']).Tr.agg(['mean','std','count'])
g['se'] = g['std']/np.sqrt(g['count'])
g['z'] = (g['mean']-1)/g['se']
print("per-init conditions with T ratio < 1 by > 2 s.e.:")
print(g[g.z < -2].round(3).to_string())
print("\nsigma 0.1 seed 3, B=1 runs:", ok[(ok.sigma==0.1)&(ok.seed==3)&(ok.B==1)&(ok.block=='B_sweep')].Tr.round(3).tolist())
print("t1, t2 GF for sigma=0.1 seed 3:", sr[(sr.sigma==0.1)&(sr.seed==3)].iloc[0][['t1_gf','t2_gf','rho_gf']].to_dict())
