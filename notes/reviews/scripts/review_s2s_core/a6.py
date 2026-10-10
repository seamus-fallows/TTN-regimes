import pandas as pd, numpy as np
pd.set_option('display.width',300); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',500)
R='experiments/s2s_core/results/'
d=pd.read_csv(R+'main_runs.csv')
b=d[(d.tree=='binary4')&(d.sweep=='main')&(d.teacher=='close')]
nb=b[b.order!='1-2-3']
print(nb[['chi','sigma','seed','order','s2s','t1','t2','t3','rho_min','plateau1_lam_min','plateau2_lam_min','plateau2_lam_pred','plateau2_n_neg','plateau1_lam_max']].sort_values(['chi','seed','sigma']).to_string())
print(pd.crosstab([b.chi,b.seed],b.order))
gf=pd.read_csv(R+'main_gap_fits.csv')
print(gf[(gf.tree=='binary4')&(gf.teacher=='close')][['chi','seed','order','n_sigma','sigma_min','sigma_max','p_t1','p_gap1','p_gap2','lam1_slope','lam2_slope','lam2_n_nonneg']].to_string())
