import pandas as pd, numpy as np, json
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
R='C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/'
cols=['tree','case','start','chi','w','k_steiner','sigma_range','hess_col','hess_min_small','hess_min_small_se','hess_raw_min_small','hess_neg_small','hess_exp','hess_exp_se','hess_raw_exp']
allt=[]
for f in ['twocomp_table.csv','twocomp_balanced_table.csv','twocomp_chi4_table.csv','robust_table.csv','extensions_table.csv']:
    d=pd.read_csv(R+f); d['src']=f.split('_table')[0]; allt.append(d)
A=pd.concat(allt,ignore_index=True)
print(A[A.k_steiner==2][['src']+cols].round(4).to_string())
k3=A[A.k_steiner>=3].copy()
k3['dexp']=(k3.hess_exp-k3.hess_raw_exp).abs()
print('k>=3 rows', len(k3), 'max |proj-raw exp|', k3.dexp.max()); print(k3.sort_values('dexp',ascending=False)[['src','tree','case','start','k_steiner','hess_exp','hess_raw_exp','dexp']].head(8).round(3).to_string())
for k in range(3,9):
    s=k3[(k3.k_steiner==k)]
    print('k',k, 'hess_exp range', s.hess_exp.min().round(3), s.hess_exp.max().round(3), ' n', s.hess_exp.notna().sum())
    print('   ', s[['src','tree','case','start','hess_exp']].round(2).values.tolist())
print(A.hess_col.value_counts())
