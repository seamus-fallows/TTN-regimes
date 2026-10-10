import pandas as pd, numpy as np
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
sr = pd.read_csv(R+"sgd_runs.csv")
sp = pd.read_csv(R+"sgd_second_plateau.csv")
for sig in [0.03, 0.1]:
    for B in [1, 4]:
        g = sr[(sr.block=='B_sweep')&(sr.sigma==sig)&(sr.B==B)].copy()
        g['Tr'] = (g.t2-g.t1)/(g.t2_gf-g.t1_gf)
        ok = g[(g.run_class=='same_order')&(~g.forgot)&(~g.diverged)]
        fg = g[g.forgot]
        print(f"sigma {sig} B {B}: usable per init {ok.groupby('seed').size().to_dict()} | rho_gf {g.groupby('seed').rho_gf.first().round(1).to_dict()}")
        print("    T/T_gf usable:", ok.groupby('seed').Tr.mean().round(3).to_dict(), " forgot runs (persistent t2-t1):", fg.groupby('seed').Tr.mean().round(3).to_dict())
