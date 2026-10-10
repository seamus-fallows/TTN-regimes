import pandas as pd, numpy as np, json
R='C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/'
s=json.load(open(R+'route_sequences.json'))
p=pd.read_csv(R+'bond_profiles_multicomp.csv')
p['k_star']=[s[f'{c}_{t}'][j]['k_star'] for c,t,j in zip(p.case,p.tree,p.level_j)]
r=p[p.resolved==True].copy()
r['match']=r['match'].astype(bool)
print('resolved',len(r),'match',r.match.mean())
r['init']=np.where(r.start=='small','fan-in','balanced/saddle')
print(r.groupby(['k_star','init']).match.agg(['mean','size']).round(4))
print(r.groupby(['k_star','start']).match.agg(['mean','size']).round(4))
print(r[r.k_star==2].groupby(['tree','case','start','level_j']).match.agg(['mean','size']).round(3))
print(r[r.k_star==4].groupby(['tree','case','start','level_j']).match.agg(['mean','size']).round(3))
print('unresolved by case/level:'); print(p[p.resolved!=True].groupby(['tree','case','start','level_j']).size())
