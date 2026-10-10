import pandas as pd, numpy as np
pd.set_option('display.width',300); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',500)
R='experiments/s2s_core/results/'
d=pd.read_csv(R+'main_runs.csv')
for tree in ['ht4','binary8','mps8']:
    x=d[(d.sweep=='large')&(d.tree==tree)]
    print(tree); print(pd.crosstab(x.seed,x.sigma,values=x['class'].str[0]+':'+x.order.astype(str),aggfunc='first'))
gf=pd.read_csv(R+'main_gap_fits.csv')
print(gf[gf.sweep=='large'][['tree','seed','order','n_sigma','sigma_min','sigma_max','p_t1','p_gap1','p_gap2','lam1_slope','lam2_slope','lam1_over_pred_median','lam2_over_pred_median']].round(3).to_string())
