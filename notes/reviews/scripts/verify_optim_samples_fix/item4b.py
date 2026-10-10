import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
sr = pd.read_csv(R+"sgd_runs.csv")
print(len(sr), sr.block.value_counts().to_dict())
print("run_class:", sr.run_class.value_counts().to_dict())
st = pd.read_csv(R+"sgd_stability.csv")
st = st[(st.tree=='binary')&(st.N==4)]
ems = dict(zip(st.B, st.eta_ms))
print("eta_ms:", ems)
sr['frac_ms'] = sr.eta / sr.B.map(ems)
g = sr.groupby(['block','init','sigma','B','eta']).agg(n=('seed','size'), forgot=('forgot','sum'), frac_ms=('frac_ms','first'), div=('diverged','sum')).reset_index()
print(g[g.forgot>0].to_string())
print("max frac_ms among runs with forgetting:", g[g.forgot>0].frac_ms.min())
print("rows with frac_ms <= 0.25 and forgot>0:"); print(g[(g.frac_ms<=0.25)&(g.forgot>0)].to_string())
# check t_end vs 1.3*later GF event + 2
sr['t_req'] = 1.3*np.fmax(sr.t1_gf, sr.t2_gf) + 2
print("runs with t_end < t_req - 1e-6:", (sr.t_end < sr.t_req - 1e-6).sum(), " (fanin only:", ((sr.t_end < sr.t_req-1e-6)&(sr.init=='fanin')).sum(), ")")
print(sr[sr.t_end < sr.t_req - 1e-6][['block','init','sigma','seed','B','eta','t_end','t1_gf','t2_gf','t_req','run_class']].head(10).to_string())
# flips vs forgot
b1 = sr[(sr.B==1)&(sr.init=='fanin')]
print(pd.crosstab(b1.run_class, b1.forgot))
