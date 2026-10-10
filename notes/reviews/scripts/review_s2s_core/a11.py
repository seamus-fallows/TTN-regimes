import pandas as pd, numpy as np
pd.set_option('display.width',300); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',500)
R='experiments/s2s_core/results/'
dr=pd.read_csv(R+'dormant_escape_drops.csv')
print(dr.groupby(['tree','chi','init']).size())
x=dr[(dr.event==0)]
print(x.groupby(['tree','chi','teacher','init']).agg(n=('ratio','size'),med=('ratio','median'),drop100=('ratio',lambda r:(r<0.01).mean()),minr=('ratio','min')).round(4).to_string())
y=dr[dr.event>0]
print(y.groupby(['tree','chi','teacher','init']).agg(n=('ratio','size'),med=('ratio','median'),drop100=('ratio',lambda r:(r<0.01).mean()),minr=('ratio','min')).round(4).to_string())
# sigma trend of median ratio, separated, event 0
z=x[x.teacher=='separated']
print(z.groupby(['tree','chi','sigma']).ratio.median().unstack().round(2).to_string())
