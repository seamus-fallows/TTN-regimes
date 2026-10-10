import pandas as pd, numpy as np
pd.set_option('display.width',300); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',500)
R='experiments/s2s_core/results/'
d=pd.read_csv(R+'main_runs.csv')
c=d[d['class']=='complete']
print('complete runs',len(c))
g=c.groupby(['sweep','tree','chi','teacher','init'])
print(g.agg(n=('seed','size'),s2s=('s2s','sum'),lvl_max=('max_rel_plateau_level_err','max'),spec_max=('max_rel_plateau_spectrum_err','max'),
            next_max=('max_plateau_next_sv_over_smin','max'),next_med=('max_plateau_next_sv_over_smin','median'),
            edge_ok=('edge_rank_change_all_one', lambda x: (x==True).sum()), edge_nan=('edge_rank_change_all_one', lambda x: x.isna().sum()),
            spread_max=('edge_spread_over_width_max2','max'),spread_med=('edge_spread_over_width_max2','median'),drift=('drift','max')).to_string())
print('edge ok total', (c.edge_rank_change_all_one==True).sum(), 'false', (c.edge_rank_change_all_one==False).sum(), 'nan', c.edge_rank_change_all_one.isna().sum())
bad=c[c.edge_rank_change_all_one==False]
print(bad.groupby(['sweep','tree','chi','teacher']).sigma.agg(['size','min','max']))
# K=7 plateau level errors
k7=c[c.K>=7]
print(k7.groupby(['sweep','tree','chi']).max_rel_plateau_level_err.agg(['max','median']))
