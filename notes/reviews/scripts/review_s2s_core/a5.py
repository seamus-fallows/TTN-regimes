import pandas as pd, numpy as np
pd.set_option('display.width',300); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',500)
R='experiments/s2s_core/results/'
d=pd.read_csv(R+'main_runs.csv')
b=d[(d.tree=='binary4')&(d.sweep=='main')]
print(b.groupby(['teacher','sigma']).agg(n=('seed','size'),s2s=('s2s','sum'),greedy=('greedy','sum'),lvl_max=('max_rel_plateau_level_err','max'),rho_med=('rho_min','median')).to_string())
print(b.groupby(['teacher','chi','order']).size())
# first components across all sweeps
d['first']=d.order.astype(str).str.split('-').str[0]
o=pd.read_csv(R+'main_order.csv'); print(o.to_string())
