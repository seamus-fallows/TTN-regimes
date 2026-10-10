import pandas as pd, numpy as np
pd.set_option('display.width', 250)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
sr = pd.read_csv(R+"sgd_runs.csv")
ok = sr[(sr.run_class=='same_order')&(~sr.forgot)&(~sr.diverged)&(sr.init=='fanin')].copy()
ok['d1'] = ok.t1-ok.t1_gf; ok['d2'] = ok.t2-ok.t2_gf
for key in [(0.1,3,1,'B_sweep',0.0005),(0.1,3,4,'eta_sweep_B4',0.0002),(0.1,3,64,'eta_sweep_B64',0.001),(0.3,1,256,'B_sweep',0.0005)]:
    s,sd,B,blk,eta = key
    q = ok[(ok.sigma==s)&(ok.seed==sd)&(ok.B==B)&(ok.block==blk)&np.isclose(ok.eta,eta)]
    print(key, "n", len(q), "mean d1 (t1 delay) %.3f +- %.3f" % (q.d1.mean(), q.d1.std()/np.sqrt(len(q))), " mean d2 (t2 shift) %.3f +- %.3f" % (q.d2.mean(), q.d2.std()/np.sqrt(len(q))), " t2_gf %.2f" % q.t2_gf.iloc[0])
# Over all fanin conditions: is mean t2 shift ever significantly negative (per init)?
g = ok.groupby(['block','sigma','seed','B','eta']).d2.agg(['mean','std','count'])
g['z'] = g['mean']/(g['std']/np.sqrt(g['count']))
print("per-init conditions with t2 earlier than GF by >2 s.e.:", (g.z<-2).sum(), "of", len(g))
print(g[g.z<-2].round(3).to_string())
