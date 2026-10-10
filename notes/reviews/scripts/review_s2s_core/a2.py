import pandas as pd, numpy as np
pd.set_option('display.width',300); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',500)
R='experiments/s2s_core/results/'
f=pd.read_csv(R+'first_runs.csv')
print(f.groupby(['tree','chi','teacher','init']).agg(n=('seed','size'),res=('resolved','sum'),relearn=('relearned','sum')).to_string())
b=f[(f.tree=='binary4')&(f.teacher=='close')]
print(pd.crosstab([b.chi,b.init,b.sigma],[b.r1,b.r1_corrflow]))
print(b[b.r1!=b.r1_corrflow][['chi','init','seed','sigma','r1','r1_corrflow','r1_race','T','relearned']].to_string())
s=pd.read_csv(R+'first_summary.csv')
print(s.round(5).to_string())
