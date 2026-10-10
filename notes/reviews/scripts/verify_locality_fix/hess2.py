import pandas as pd, numpy as np, json
pd.set_option('display.width',250)
R='C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/'
h=pd.read_csv(R+'hessians.csv')
k=h._key.apply(json.loads)
h['name']=k.str[0]; h['S']=k.str[1].apply(lambda s: ''.join(map(str,s)) if s else None); h['kind']=k.str[2]; h['n']=k.str[3]; h['start']=k.str[4]; h['chi']=k.str[6]; h['w']=k.str[7].apply(tuple); h['seed']=k.str[9]; h['sigma']=k.str[10]
k2=h[(h.name.isna())&(h.start=='saddle')&(((h.kind=='mps')&(h.S.isin(['01','12','23','34'])))) & (h.j==0)]
print('k=2 saddle-start midpoints:',len(k2))
print(k2.groupby(['kind','n','S','w'])[['hess_raw_min','hess_lib_min','hess_wc_min','stored_hess_min']].agg(['min','max']).round(4).to_string())
# number of negative eigenvalues
print(k2.groupby(['kind','n','S','w'])[['hess_raw_neg','hess_wc_neg']].agg(['min','max']))
print('rows total', len(h), ' NaN lib', h.hess_lib_min.isna().sum())
# difference lib vs wc
print('max |lib-wc|/|raw| overall', ((h.hess_lib_min-h.hess_wc_min).abs()/h.hess_raw_min.abs()).max())
print('--- at smallest sigma 1e-4')
s=k2[k2.sigma==1e-4]
print(s.groupby(['kind','n','S','w'])[['hess_raw_min','hess_lib_min','stored_hess_min']].agg(['mean','min','max']).round(4).to_string())
print('--- per sigma, mps4 S12 (1,0.5)')
m=k2[(k2.n==4)&(k2.S=='12')&(k2.w==(1.0,0.5))]
print(m.groupby('sigma')[['hess_raw_min','hess_lib_min','stored_hess_min']].mean().round(4))
