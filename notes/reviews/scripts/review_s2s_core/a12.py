import pandas as pd, numpy as np, warnings
warnings.simplefilter('ignore')
pd.set_option('display.width',300); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',500)
R='experiments/s2s_core/results/'
f=pd.read_csv(R+'first_runs.csv')
g=f.groupby(['tree','chi','teacher','init','seed']).agg(T=('T','first'),r1c=('r1_corrflow','first')).reset_index()
tb=g[g.init=='balanced'].set_index(['tree','chi','teacher','seed'])
tf=g[g.init=='fanin'].set_index(['tree','chi','teacher','seed'])
j=tb.join(tf,lsuffix='_b',rsuffix='_f',how='inner')
j['ratio']=j.T_b/j.T_f; j['same']=j.r1c_b==j.r1c_f
print(j.groupby(level=[0,1,2]).agg(rmin=('ratio','min'),rmax=('ratio','max'),rmed=('ratio','median'),same=('same','mean'),n=('ratio','size')).round(3).to_string())
t=f.groupby(['tree','chi','teacher','init','sigma']).ratio_with_offset.apply(lambda x: np.max(np.abs(x-1))).unstack()
print(t.apply(lambda col: col.map(lambda v: '%.1e'%v if pd.notna(v) else '')).to_string())
